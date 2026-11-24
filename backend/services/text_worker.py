"""Studio compatibility layer for the installed text training runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The worker runs with the ML workspace as cwd, including packaged installs.
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def configure(worker, config):
    from homellm.models.adapters import HomeAdapter, HFAdapter, detect_model_type
    from backend.services.custom_models import model_classes

    original_resolver = worker.resolve_adapter

    class StudioAdapter(HomeAdapter):
        def __init__(self, kind):
            self.kind = kind

        def init_from_scratch(self, source, tokenizer, settings, trust_remote_code=True):
            import torch
            config_class, model_class = model_classes(self.kind)
            if self.kind == 'blueprint':
                from backend.services.blueprints import with_tokenizer
                blueprint = with_tokenizer(settings['model_blueprint_spec'], len(tokenizer))
                model_config = config_class(vocab_size=len(tokenizer), hidden_size=blueprint['hidden_size'],
                    max_position_embeddings=blueprint['max_position_embeddings'], auto_project=blueprint['auto_project'],
                    blueprint=blueprint, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=False, use_cache=False, studio_graph_version=1)
                model = model_class(model_config)
                dtype = torch.bfloat16 if settings.get('mixed_precision') == 'bf16' else (
                    torch.float16 if settings.get('mixed_precision') == 'fp16' and settings.get('fp16_pure') else torch.float32)
                return model.to(dtype=dtype), model_config
            args = dict(vocab_size=len(tokenizer), hidden_size=settings['hidden_size'],
                        num_hidden_layers=settings['num_layers'], num_attention_heads=settings['n_heads'],
                        max_position_embeddings=settings['seq_len'], dropout=settings.get('dropout', 0),
                        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id, use_sdpa=settings.get('use_flash_attention', True),
                        use_liger=settings.get('use_liger', True))
            if settings.get('intermediate_size'):
                args['intermediate_size'] = settings['intermediate_size']
            if self.kind == 'home_moe':
                args.update({key: settings[key] for key in ('num_experts', 'num_experts_per_tok', 'expert_type')})
            model_config = config_class(**args)
            model = model_class(model_config)
            dtype = torch.bfloat16 if settings.get('mixed_precision') == 'bf16' else (
                torch.float16 if settings.get('mixed_precision') == 'fp16' and settings.get('fp16_pure') else torch.float32)
            return model.to(dtype=dtype), model_config

        def load_for_training(self, base_model_path, stage, tokenizer, config, trust_remote_code=True):
            if not base_model_path:
                return self.init_from_scratch(None, tokenizer, config, trust_remote_code)
            if self.kind == 'home':
                return super().load_for_training(base_model_path, stage, tokenizer, config, trust_remote_code)
            import torch
            config_class, model_class = model_classes(self.kind)
            model_config = config_class.from_pretrained(base_model_path)
            model_config.use_sdpa = config.get('use_flash_attention', True)
            model_config.use_liger = config.get('use_liger', True)
            dtype = torch.bfloat16 if config.get('mixed_precision') == 'bf16' else (
                torch.float16 if config.get('mixed_precision') == 'fp16' and config.get('fp16_pure') else torch.float32)
            model = model_class.from_pretrained(base_model_path, config=model_config, torch_dtype=dtype)
            return model, model.config

    def resolve(settings):
        if settings.get('stage') == 'pretrain':
            kind = settings.get('model_type', 'home')
            return HFAdapter() if kind == 'hf' else StudioAdapter(kind)
        base = Path(settings.get('base_model_path', ''))
        saved = base / 'config.json'
        if saved.is_file() and json.loads(saved.read_text()).get('model_type') == 'homellm_blueprint':
            return StudioAdapter('blueprint')
        kind = detect_model_type(base)
        if kind in ('home', 'home_moe', 'gpt2_home'):
            return StudioAdapter('gpt2' if kind == 'gpt2_home' else kind)
        return original_resolver(settings)

    worker.resolve_adapter = resolve
    if config.get('stage') == 'sft':
        from backend.services.sft import dataset_class
        worker.SFTDataset = dataset_class(worker.SFTDataset, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--metrics', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config.get('training_backend') == 'unsloth':
        # Import before Transformers; never silently substitute another backend.
        from homellm.training.unsloth_sft import is_unsloth_available, run_unsloth_sft
        if not is_unsloth_available():
            raise RuntimeError('Unsloth was selected but is not installed in this Python environment')
        from homellm.app.trainer_worker import MetricsLogger
        run_unsloth_sft(config, MetricsLogger(Path(args.metrics)))
        return
    from homellm.app import trainer_worker
    configure(trainer_worker, config)
    trainer_worker.run_training(config, Path(args.metrics))


if __name__ == '__main__':
    main()
