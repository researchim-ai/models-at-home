"""Compatibility layer for VLM training and inference in the installed runtime."""
from pathlib import Path
import json


def adapter_source(source):
    path = Path(source) / 'adapter_config.json'
    if path.is_file():
        return json.loads(path.read_text())['base_model_name_or_path']
    return source


def load_model(source, config, device):
    import torch
    from homellm.training import vlm_common as common
    from transformers import AutoModelForImageTextToText
    dtype = torch.float32 if device != 'cuda' or config.get('mixed_precision') == 'no' else (
        torch.float16 if config.get('mixed_precision') == 'fp16' else torch.bfloat16)
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; select fp16 or no mixed precision')
    attention = common.resolve_attn_implementation(config, device)
    if dtype == torch.float32 and attention == 'flash_attention_2':
        attention = 'sdpa'
    kwargs = dict(trust_remote_code=True, torch_dtype=dtype,
                  attn_implementation=attention)
    if config.get('tuning_method') == 'qlora':
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    if device == 'cuda':
        kwargs['device_map'] = {'': 0}
    base = adapter_source(source)
    if config.get('training_backend') == 'unsloth':
        from unsloth import FastVisionModel
        model, _ = FastVisionModel.from_pretrained(
            model_name=base, max_seq_length=config['seq_len'], dtype=dtype,
            load_in_4bit=config.get('tuning_method') == 'qlora',
            full_finetuning=config.get('tuning_method') == 'full',
            use_gradient_checkpointing=False,
            fast_inference=False, attn_implementation=attention,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(base, **kwargs)
    if base != source:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, source, is_trainable=False)
    model.config.use_cache = False
    return model


def runtime_types(config=None):
    config = config or {}
    import torch
    from homellm.training.vlm_data import VLMJsonlDataset, VLMDataCollator

    class Dataset(VLMJsonlDataset):
        def _load(self):
            from homellm.training.vlm_data import (_extract_image_refs, _build_messages_from_record,
                                                  _extract_target_text)
            with self.file_path.open(encoding='utf-8') as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        refs = _extract_image_refs(record, self.field_map)
                        messages = _build_messages_from_record(record, self.prompt_for_caption, self.field_map)
                        if not refs or not messages:
                            raise ValueError('Missing image references or conversation')
                        self.examples.append({'messages': messages, 'image_refs': refs,
                            'target_text': _extract_target_text(record, messages), 'metadata': {}})
                    except Exception as exc:
                        raise ValueError(f'{self.file_path}, line {line_number}: {exc}') from exc

        def __getitem__(self, index):
            value = dict(super().__getitem__(index))
            base = Path(config['data_base_dir']) if config.get('data_base_dir') and str(self.file_path) == config.get('data_path') else self.base_dir
            value['image_refs'] = [str((base / ref).resolve()) if not ref.startswith(('http://', 'https://', 'file://'))
                                   and not Path(ref).is_absolute() else ref for ref in value['image_refs']]
            return value

    class Collator(VLMDataCollator):
        def _load_example_images(self, example):
            images = super()._load_example_images(example)
            if len(images) != len(example['image_refs']):
                raise ValueError('An image failed to load; refusing to change the multimodal example silently')
            return images

        def _collect_special_token_ids(self):
            tokenizer = self.processor.tokenizer
            tokens = super()._collect_special_token_ids()
            return [value for value in tokens if value != tokenizer.unk_token_id]

        def __call__(self, examples):
            prepared = [(example, self._load_example_images(example), self._build_text(example['messages'])) for example in examples]
            batch = self._build_inputs(prepared)
            ids = batch['input_ids']
            if ids.shape[1] > self.max_length:
                raise ValueError(f'Expanded image/text sequence ({ids.shape[1]}) exceeds seq_len ({self.max_length}); reduce image resolution or increase seq_len')
            labels = ids.clone()
            if self.assistant_only_loss:
                labels.fill_(-100)
                for row, (example, images, _) in enumerate(prepared):
                    for index, message in enumerate(example['messages']):
                        if message['role'] != 'assistant':
                            continue
                        def prefix(messages, generation):
                            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
                            result = self.processor(text=[text], images=[self._images_arg(images)], return_tensors='pt', padding=False)
                            tokens = result['input_ids'][0]
                            if not torch.equal(ids[row, :len(tokens)], tokens):
                                raise ValueError('Chat template does not produce a stable prefix; assistant-only labels cannot be verified')
                            return len(tokens)
                        start = prefix(example['messages'][:index], True)
                        end = prefix(example['messages'][:index + 1], False)
                        labels[row, start:end] = ids[row, start:end]
            labels[batch['attention_mask'] == 0] = -100
            for token_id in self._collect_special_token_ids():
                labels[ids == token_id] = -100
            if any(not (row[1:] != -100).any() for row in labels):
                raise ValueError('A VLM example contains no next-token training targets after masking')
            batch['labels'] = labels
            return batch
    return Dataset, Collator


def apply_adapters(model, config):
    from homellm.training import vlm_common
    if config.get('training_backend') == 'unsloth':
        from unsloth import FastVisionModel
        if config['tuning_method'] != 'full':
            model = FastVisionModel.get_peft_model(model,
                r=config['lora_r'], lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'],
                target_modules=config.get('lora_target_modules') or 'all-linear',
                modules_to_save=config.get('lora_modules_to_save') or None,
                finetune_vision_layers=not config.get('freeze_vision_tower', False),
                finetune_language_layers=not config.get('freeze_language_model', False),
                use_gradient_checkpointing=False,
            )
        FastVisionModel.for_training(model, use_gradient_checkpointing=config['gradient_checkpointing'])
    else:
        model = vlm_common.apply_vlm_lora(model, config)
    vlm_common.apply_vlm_freeze_policy(model, config)
    if not any(param.requires_grad for param in model.parameters()):
        raise ValueError('Freeze/LoRA settings leave no trainable parameters')
    return model


def configure(config):
    from homellm.training import vlm_sft
    vlm_sft.load_vlm_model = load_model
    vlm_sft.VLMJsonlDataset, vlm_sft.VLMDataCollator = runtime_types(config)
    vlm_sft.apply_vlm_lora = apply_adapters
