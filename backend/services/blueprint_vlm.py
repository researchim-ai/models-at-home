"""Native SFT/pretraining for the Research Lab vision-prefix adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import torch
from backend.services.blueprint_multimodal import BlueprintVisionLM, byte_ids, save_vlm


def text_content(content):
    if isinstance(content, str):
        return content
    return ''.join(str(part.get('text', '')) for part in content if isinstance(part, dict) and part.get('type') == 'text')


def encode_messages(messages, assistant_only=True):
    ids, labels = [1], [-100]
    for message in messages:
        role = message.get('role', 'user')
        header = byte_ids(role.title() + ': ')
        content = byte_ids(text_content(message.get('content', '')))
        ending = [2] if role == 'assistant' else byte_ids('\n')
        ids.extend(header + content + ending)
        labels.extend(([-100] * len(header) if assistant_only else header) + (content + ending if role == 'assistant' or not assistant_only else [-100] * (len(content) + len(ending))))
    if not any(value != -100 for value in labels[1:]):
        raise ValueError('No supervised next-token targets in the Research VLM example')
    return ids, labels


def load_rows(config, filename):
    from backend.services.vlm_runtime import runtime_types
    Dataset, _ = runtime_types(config)
    data = Dataset(filename)
    rows = []
    for index in range(len(data)):
        row = data[index]
        if len(row['image_refs']) != 1:
            raise ValueError('Research VLM currently uses exactly one image per example')
        image = Path(row['image_refs'][0])
        if not image.is_file():
            raise ValueError(f'Research VLM requires a local image: {image}')
        ids, labels = encode_messages(row['messages'], config.get('assistant_only_loss', True))
        if len(ids) + config['vision_tokens'] > config['seq_len']:
            raise ValueError(f'Row {index + 1}: image patches + byte text exceed context; no silent truncation')
        rows.append({'image': str(image), 'ids': ids, 'labels': labels, 'messages': row['messages']})
    if not rows:
        raise ValueError('Empty Research VLM dataset')
    return rows


def collate(rows, config):
    from backend.services.vla_data import image_tensor
    length = max(len(row['ids']) for row in rows)
    ids = torch.zeros(len(rows), length, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        size = len(row['ids'])
        ids[index, :size] = torch.tensor(row['ids'])
        labels[index, :size] = torch.tensor(row['labels'])
        mask[index, :size] = 1
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels,
            'images': torch.stack([image_tensor(row['image'], config['vision_image_size']) for row in rows])}


def preflight(config):
    from PIL import Image
    import base64
    from io import BytesIO
    report = {'ready': False, 'checks': [], 'sample': None, 'datasets': []}
    def check(name, status, detail):
        report['checks'].append({'name': name, 'status': status, 'detail': str(detail)[:2000]})
    try:
        cuda = config.get('device') != 'cpu' and torch.cuda.is_available()
        if config.get('device') == 'cuda' and not cuda:
            raise ValueError('CUDA is unavailable')
        if cuda and config['mixed_precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('GPU does not support bf16')
        check('Research adapter', 'pass', f"Snapshot {config['effective_blueprint_hash'][:12]}; trainable image patches + byte tokenizer (259) + blueprint core")
        check('Device', 'pass', torch.cuda.get_device_name(0) if cuda else 'CPU')
        check('Training', 'pass', 'Native full-weight SFT/pretrain from scratch; no pretrained vision encoder or Unsloth')
        for kind, path in [('train', config['data_path']), ('validation', config.get('val_data_path'))]:
            if not path:
                continue
            rows = load_rows(config, path)
            for row in rows[:200]:
                collate([row], config)
            report['datasets'].append({'kind': kind, 'scanned_rows': len(rows), 'valid_rows': len(rows),
                'images_checked': min(200, len(rows)), 'complete': len(rows) <= 200})
            if len(rows) > 200:
                check('Image scope', 'warning', 'Text/context checked for every row; image decoding limited to 200 rows per split')
            if kind == 'train':
                first = rows[0]
                batch = collate([first], config)
                with Image.open(first['image']) as image:
                    image = image.convert('RGB'); width, height = image.size
                    image.thumbnail((256, 256)); output = BytesIO(); image.save(output, format='PNG')
                labels = batch['labels'][0]
                report['sample'] = {'line': 1, 'images': [{'reference': first['image'], 'width': width, 'height': height,
                    'src': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()}],
                    'expanded_tokens': len(first['ids']) + config['vision_tokens'],
                    'supervised_tokens': int((labels != -100).sum()), 'omitted_tokens': max(0, len(first['ids']) - 256),
                    'tokens': [{'index': i + config['vision_tokens'], 'id': token,
                        'text': {0:'<PAD>',1:'<BOS>',2:'<EOS>'}.get(token, bytes([token - 3]).decode('utf-8', errors='replace') if token >= 3 else ''),
                        'trainable': first['labels'][i] != -100} for i, token in enumerate(first['ids'][:256])]}
                check('Training data', 'pass', f'{len(rows)} examples; explicit byte-level supervision masks')
        report['ready'] = True
    except Exception as exc:
        check('Research VLM preflight', 'error', exc)
    return report


def signature(config):
    keys = ('model_blueprint_spec', 'vision_image_size', 'vision_patch_size', 'byte_tokenizer_version',
            'batch_size', 'gradient_accumulation', 'learning_rate', 'warmup_steps', 'mixed_precision', 'assistant_only_loss')
    payload = {key: config.get(key) for key in keys}
    for key in ('data_path', 'val_data_path'):
        if config.get(key):
            payload[key] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def complete_checkpoint(path):
    path = Path(path)
    try:
        marker = json.loads((path / 'complete.json').read_text())
        return marker['format'] == 'mah_blueprint_vlm_training_v1' and marker['step'] > 0 and (path / 'training_state.pt').stat().st_size > 0 and (path / 'model.safetensors').stat().st_size > 0
    except (OSError, ValueError, KeyError, TypeError):
        return False


def train(config, metrics):
    from safetensors.torch import load_file
    from contextlib import nullcontext
    import math
    torch.set_num_threads(4)
    random.seed(42); torch.manual_seed(42)
    cuda = config.get('device') != 'cpu' and torch.cuda.is_available()
    if config.get('device') == 'cuda' and not cuda:
        raise ValueError('CUDA is unavailable')
    if cuda and config['mixed_precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bf16')
    device = 'cuda' if cuda else 'cpu'
    metrics.update(status='loading_dataset', architecture='blueprint_vlm', stage=config['stage'], device=device,
                   source_blueprint_hash=config['source_blueprint_hash'], effective_blueprint_hash=config['effective_blueprint_hash'])
    rows = load_rows(config, config['data_path'])
    validation = load_rows(config, config['val_data_path']) if config.get('val_data_path') else []
    model = BlueprintVisionLM(config).to(device)
    from backend.services.research_signals import RuntimeSignals
    observer=RuntimeSignals(model.core.graph,config['model_blueprint_spec']) if config.get('research_diagnostics') else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    def schedule(step):
        if step < config['warmup_steps']:
            return step / max(1, config['warmup_steps'])
        ratio = (step - config['warmup_steps']) / max(1, config['max_steps'] - config['warmup_steps'])
        return .5 * (1 + math.cos(math.pi * min(1, ratio)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda and config['mixed_precision'] == 'fp16')
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(len(rows), generator=generator).tolist()
    step = cursor = 0
    fingerprint = signature(config)
    if config.get('resume_from_checkpoint'):
        path = Path(config['resume_from_checkpoint'])
        if not complete_checkpoint(path):
            raise ValueError('Resume requires a complete Research VLM optimizer checkpoint')
        state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=False)
        if state['signature'] != fingerprint:
            raise ValueError('Research VLM architecture/data/optimizer signature changed')
        model.load_state_dict(load_file(str(path / 'model.safetensors')), strict=True)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        step, cursor, order = state['step'], state['cursor'], state['order']
        generator.set_state(state['generator']); torch.set_rng_state(state['torch_rng']); random.setstate(state['python_rng'])
        if cuda:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if step >= config['max_steps']:
            raise ValueError('Checkpoint reached max_steps; extend the training plan')
        for group, base in zip(optimizer.param_groups, scheduler.base_lrs):
            group['lr'] = base * schedule(step)
    def amp():
        return torch.autocast('cuda', dtype=torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float16) if cuda and config['mixed_precision'] != 'no' else nullcontext()
    def prepare(items):
        return {k: v.to(device) for k, v in collate(items, config).items()}
    def save(path, final=False):
        save_vlm(path, model, config)
        if not final:
            torch.save({'format':'mah_blueprint_vlm_training_v1', 'signature':fingerprint, 'step':step, 'cursor':cursor,
                'order':order, 'generator':generator.get_state(), 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
                'scaler':scaler.state_dict(), 'torch_rng':torch.get_rng_state(), 'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if cuda else []}, path / 'training_state.pt')
            from backend.paths import atomic_json
            atomic_json(path / 'complete.json', {'format':'mah_blueprint_vlm_training_v1', 'step':step})
        metrics.log_checkpoint(str(path), step=step)
    metrics.update(status='training', trainable_params=sum(p.numel() for p in model.parameters()), total_steps=config['max_steps'], step=step)
    while step < config['max_steps']:
        batches = []
        for _ in range(config['gradient_accumulation']):
            items = []
            for _ in range(config['batch_size']):
                if cursor >= len(order):
                    order = torch.randperm(len(rows), generator=generator).tolist(); cursor = 0
                items.append(rows[order[cursor]]); cursor += 1
            batches.append(prepare(items))
        model.train()
        if observer: observer.begin(step+1)
        for attempt in range(16):
            optimizer.zero_grad(set_to_none=True); loss_value = 0.
            for batch in batches:
                with amp():
                    loss = model(**batch)['loss']
                if not torch.isfinite(loss):
                    raise ValueError('Research VLM loss is not finite')
                scaler.scale(loss / len(batches)).backward(); loss_value += float(loss.detach()) / len(batches)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm) and not scaler.is_enabled():
                raise ValueError('Research VLM gradients are not finite')
            previous = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= previous:
                break
        else:
            raise ValueError('FP16 overflow prevented an optimizer update after 16 attempts')
        if observer:
            signals=observer.collect()
            if signals: metrics.update(research_signals=signals)
        scheduler.step(); step += 1
        metrics.log_step(step, loss_value, optimizer.param_groups[0]['lr'])
        metrics.metrics.setdefault('grad_norm_history', []).append(float(norm))
        metrics.update(optimizer_step=step)
        if validation and config.get('eval_steps', 0) and step % config['eval_steps'] == 0:
            model.eval(); total = count = 0.
            with torch.no_grad(), amp():
                for offset in range(0, len(validation), config['batch_size']):
                    batch = prepare(validation[offset:offset + config['batch_size']])
                    tokens = int((batch['labels'] != -100).sum())
                    total += float(model(**batch)['loss']) * tokens; count += tokens
            metrics.metrics.setdefault('val_loss_history', []).append(total / count)
            metrics.metrics.setdefault('val_steps_history', []).append(step)
            metrics.update(val_loss=total / count)
        if step % config['save_every'] == 0:
            save(Path(config['output_dir']) / f'checkpoint-{step}')
    save(Path(config['output_dir']) / 'final_model', final=True)
    if observer: observer.close()
    metrics.update(status='completed', step=step, final_model_dir=str(Path(config['output_dir']) / 'final_model'))
