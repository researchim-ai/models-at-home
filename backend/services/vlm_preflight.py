"""Inspect a VLM experiment without instantiating model weights."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path


def inspect(config: dict) -> dict:
    report = {'ready': False, 'checks': [], 'sample': None, 'datasets': []}
    def check(name, status, detail):
        report['checks'].append({'name': name, 'status': status, 'detail': str(detail)[:2000]})
    try:
        import torch
        from PIL import Image
        from transformers import AutoConfig, AutoModelForImageTextToText
        from homellm.training.vlm_common import load_vlm_processor
        from homellm.training.vlm_data import _extract_image_refs, _build_messages_from_record, load_image
        from backend.services.vlm_runtime import adapter_source, runtime_types
    except Exception as exc:
        check('Окружение', 'error', exc)
        return report
    check('ML runtime', 'pass', f'Torch {torch.__version__}')
    if config.get('stage') == 'vlm_grpo':
        try:
            from homellm.training.rl.loss import GRPOLoss, SDPOLoss, compute_advantages
            from homellm.training.rl.experience import ReplayBuffer
            check('Native GRPO', 'pass', 'homellm.training.rl: собственные GRPOLoss/SDPOLoss, advantages и ReplayBuffer. Предпросмотр показывает эталон reward.')
        except Exception as exc:
            check('Native GRPO', 'error', exc)
    if not torch.cuda.is_available():
        check('GPU', 'error', 'CUDA недоступна в выбранном Python. Обучение VLM требует NVIDIA GPU.')
    else:
        try:
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            check('GPU', 'pass', f'{props.name}: свободно {free / 2**30:.2f} из {total / 2**30:.2f} GiB. Это не гарантия, что выбранный batch поместится.')
            if config['mixed_precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
                check('Precision', 'error', 'Выбранная GPU не поддерживает bf16; выберите fp16 или no.')
        except Exception as exc:
            check('GPU', 'error', exc)
    for module in (['peft'] if config['tuning_method'] in ('lora', 'qlora') else []) + (['bitsandbytes'] if config['tuning_method'] == 'qlora' else []):
        try:
            value = __import__(module)
            check(module, 'pass', getattr(value, '__version__', 'installed'))
        except Exception as exc:
            check(module, 'error', exc)
    source = config['base_model_path']
    try:
        base = adapter_source(source)
        if base != source:
            check('Базовая модель', 'error', 'Выбран каталог LoRA-адаптера. Для нового обучения выберите базовую модель; для продолжения используйте checkpoint resume. Адаптер можно загрузить в inference.')
        model_config = AutoConfig.from_pretrained(base, trust_remote_code=True)
        if model_config.model_type in ('qwen3_5', 'qwen3_5_moe') and config['tuning_method'] == 'qlora':
            check('Qwen3.5 quantization', 'warning', 'Для Qwen3.5 рекомендуется bf16 LoRA: Unsloth предупреждает о повышенных погрешностях 4-bit квантования.')
        local_base = Path(base)
        if local_base.is_dir():
            files = [local_base / name for name in ('model.safetensors', 'pytorch_model.bin',
                     'model.safetensors.index.json', 'pytorch_model.bin.index.json') if (local_base / name).is_file()]
            try:
                if not files:
                    raise ValueError('В локальной модели нет PyTorch/Safetensors весов или shard index')
                for filename in files:
                    if not filename.stat().st_size:
                        raise ValueError(f'Пустой файл весов: {filename.name}')
                    if filename.name.endswith('.index.json'):
                        shards = set(json.loads(filename.read_text())['weight_map'].values())
                        if not shards or any(not (local_base / shard).is_file() or not (local_base / shard).stat().st_size for shard in shards):
                            raise ValueError('В shard index отсутствуют или недоступны файлы весов')
                check('Файлы весов', 'pass', 'Файлы локальных весов присутствуют. Их содержимое будет прочитано только при обучении.')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                check('Файлы весов', 'error', exc)
        else:
            check('Файлы весов', 'warning', 'Веса удалённой модели не скачивались. Их доступность и содержимое этой проверкой не подтверждены.')
        if type(model_config) not in AutoModelForImageTextToText._model_mapping and 'AutoModelForImageTextToText' not in (getattr(model_config, 'auto_map', None) or {}):
            raise ValueError(f'{model_config.model_type} не поддерживается установленным AutoModelForImageTextToText')
        check('Архитектура', 'pass', model_config.model_type)
        from backend.services.vlm_recipes import evidence
        status, detail = evidence(model_config.model_type, config)
        check('Проверенный рецепт', status, detail)
        processor_source = source
        if base != source and not any((Path(source) / name).is_file() for name in ('processor_config.json', 'preprocessor_config.json')):
            processor_source = base
        processor = load_vlm_processor(processor_source, config)
        check('Processor', 'pass', type(processor).__name__)
    except Exception as exc:
        check('Модель / processor', 'error', exc)
        return report

    _, Collator = runtime_types(config)
    collator = Collator(processor, max_length=config['seq_len'], assistant_only_loss=config['assistant_only_loss'])
    if config.get('warmup_steps', 0) >= config.get('max_steps', 1) > 0:
        check('Расписание LR', 'warning', 'Warmup занимает весь выбранный план обучения; проверьте warmup_steps и max_steps.')
    # Every sampled row uses the very same collator/masks as the training worker.
    # Bound CPU work and response size even for huge JSONL files.
    for kind, filename in [('train', config['data_path']), ('validation', config.get('val_data_path'))]:
        if not filename:
            continue
        path = Path(filename)
        root = Path(config['data_base_dir']) if kind == 'train' else path.parent
        stats = {'kind': kind, 'scanned_rows': 0, 'valid_rows': 0, 'images_checked': 0, 'complete': True}
        errors = []
        remote_rows = 0
        try:
            with path.open(encoding='utf-8') as handle:
                for line_number in range(1, 202):
                    line = handle.readline(2_000_001)
                    if not line:
                        break
                    if line_number > 200:
                        stats['complete'] = False
                        break
                    if not line.strip():
                        continue
                    stats['scanned_rows'] += 1
                    try:
                        if len(line) > 2_000_000:
                            raise ValueError('JSONL-строка превышает лимит проверки 2 MB')
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError('Ожидается JSON object')
                        refs = _extract_image_refs(record, config.get('vlm_columns'))
                        messages = _build_messages_from_record(record, 'Describe this image.', config.get('vlm_columns'))
                        if not refs or not messages:
                            raise ValueError('Не найдены изображения или сообщения; проверьте формат/названия колонок')
                        refs = [str((root / ref).resolve()) if not ref.startswith(('http://', 'https://', 'file://')) and not Path(ref).is_absolute() else ref for ref in refs]
                        # Remote data remains supported, but a 200-row network probe
                        # would be unbounded. Check only the first three URL rows.
                        if any(ref.startswith(('http://', 'https://')) for ref in refs):
                            if remote_rows >= 3:
                                stats['complete'] = False
                                break
                            remote_rows += 1
                        example = {'messages': messages, 'image_refs': refs}
                        batch = collator([example])
                        if config.get('stage') == 'vlm_grpo':
                            from backend.services.vlm_grpo import prompt_and_answer
                            prompt, _ = prompt_and_answer(example)
                            images_for_prompt = [load_image(ref) for ref in refs]
                            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                            prompt_inputs = processor(text=[text], images=[images_for_prompt], return_tensors='pt')
                            if prompt_inputs['input_ids'].shape[1] + config['grpo_max_new_tokens'] > config['seq_len']:
                                raise ValueError('Expanded image prompt + completion budget exceeds seq_len')
                        stats['valid_rows'] += 1
                        stats['images_checked'] += len(refs)
                        if report['sample'] is None and kind == 'train':
                            ids, labels = batch['input_ids'][0], batch['labels'][0]
                            length = int(batch['attention_mask'][0].sum())
                            indices = list(range(length)) if length <= 240 else list(range(120)) + list(range(length - 120, length))
                            images = []
                            for ref in refs[:3]:
                                image = load_image(ref)
                                width, height = image.size
                                image.thumbnail((256, 256))
                                output = BytesIO()
                                image.save(output, format='PNG')
                                images.append({'reference': ref, 'width': width, 'height': height,
                                    'src': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()})
                            report['sample'] = {'line': line_number, 'images': images, 'expanded_tokens': length,
                                'supervised_tokens': int((labels[1:length] != -100).sum()),
                                'omitted_tokens': max(0, length - len(indices)),
                                'tokens': [{'index': i, 'id': int(ids[i]), 'text': str(processor.tokenizer.convert_ids_to_tokens(int(ids[i]))),
                                            'trainable': i > 0 and int(labels[i]) != -100} for i in indices]}
                    except Exception as exc:
                        if len(errors) < 10:
                            errors.append(f'Строка {line_number}: {exc}')
                        if len(line) > 2_000_000:
                            stats['complete'] = False
                            break
        except Exception as exc:
            errors.append(str(exc))
        report['datasets'].append(stats)
        if errors or not stats['valid_rows']:
            check(f'Данные {kind}', 'error', '\n'.join(errors) or 'Не найдено ни одного обучаемого примера')
        else:
            check(f'Данные {kind}', 'pass', f"Проверено примеров: {stats['valid_rows']}; изображений: {stats['images_checked']}")
        if not stats['complete']:
            check(f'Область проверки {kind}', 'warning', 'Проверка ограничена первыми 200 строками или тремя URL-примерами; остальная часть файла не проверена.')
    report['ready'] = not any(item['status'] == 'error' for item in report['checks'])
    return report
