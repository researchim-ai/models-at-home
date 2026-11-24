"""Offline GPU mechanics tests for modern Qwen3-VL/Qwen3.5 architectures."""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['full', 'lora', 'qlora'], default='full')
    parser.add_argument('--stage', choices=['vlm_sft', 'vlm_pretrain', 'vlm_grpo'], default='vlm_sft')
    parser.add_argument('--family', choices=['qwen3_vl', 'qwen3_5'], default='qwen3_vl')
    parser.add_argument('--backend', choices=['transformers', 'unsloth'], default='transformers')
    parser.add_argument('--algorithm', choices=['grpo', 'dr_grpo', 'dapo', 'sdpo'], default='grpo')
    parser.add_argument('--model-path', help='Optional real pretrained local checkpoint instead of random tiny weights')
    parser.add_argument('--result', help='Write a compact machine-readable proof of the completed smoke')
    parser.add_argument('--reward-target', default='red', help='Reference answer for GRPO smoke')
    parser.add_argument('--reward-fn', choices=['exact_match', 'contains'], default='contains')
    parser.add_argument('--verify-resume', action='store_true', help='Compare uninterrupted native run with resume from its step-2 checkpoint')
    parser.add_argument('--precision', choices=['no', 'bf16', 'fp16'])
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mah-vlm-smoke-') as temporary:
        root = Path(temporary)
        os.environ.update(MODELS_AT_HOME_ROOT=temporary, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        if args.backend == 'unsloth':
            import unsloth
        import torch
        torch.manual_seed(3407)
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from PIL import Image
        from fastapi.testclient import TestClient
        from backend.api import app
        from backend.routes import vlm
        from backend.services import processes
        vocabulary = {word: index for index, word in enumerate([
            '[UNK]', '[PAD]', '[EOS]', '[BOS]', '<image>', '<user>', '<assistant>', 'what', 'color', 'red', 'square', '?',
            '<|image_pad|>', '<|vision_start|>', '<|vision_end|>', '<|video_pad|>'])}
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]',
                    eos_token='[EOS]', bos_token='[BOS]', additional_special_tokens=['<image>', '<user>', '<assistant>',
                        '<|image_pad|>', '<|vision_start|>', '<|vision_end|>', '<|video_pad|>'])
        template = ("{% for message in messages %}{{ '<' + message['role'] + '> ' }}"
            "{% for item in message['content'] %}{% if item['type'] == 'image' %}{{ '<image> ' }}"
            "{% else %}{{ item['text'] + ' ' }}{% endif %}{% endfor %}"
            "{% if message['role'] == 'assistant' %}{{ eos_token + ' ' }}{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}")
        tokenizer.chat_template = template
        from transformers import (Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor,
                                  Qwen2VLImageProcessor, Qwen3VLVideoProcessor)
        template = template.replace('<image> ', '<|vision_start|><|image_pad|><|vision_end|> ')
        tokenizer.chat_template = template
        processor = Qwen3VLProcessor(tokenizer=tokenizer, image_processor=Qwen2VLImageProcessor(
            size={'shortest_edge': 256, 'longest_edge': 256}, patch_size=2, temporal_patch_size=2, merge_size=2),
            video_processor=Qwen3VLVideoProcessor(), chat_template=template)
        vision = {'depth': 1, 'hidden_size': 32, 'intermediate_size': 64, 'out_hidden_size': 32,
                  'num_heads': 4, 'patch_size': 2, 'spatial_merge_size': 2, 'temporal_patch_size': 2,
                  'num_position_embeddings': 64, 'deepstack_visual_indexes': []}
        text = {'vocab_size': len(vocabulary), 'hidden_size': 32, 'intermediate_size': 64,
                'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
                'head_dim': 8, 'max_position_embeddings': 256, 'pad_token_id': 1,
                'eos_token_id': 2, 'bos_token_id': 3,
                'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2]}}
        config_class, model_class = Qwen3VLConfig, Qwen3VLForConditionalGeneration
        if args.family == 'qwen3_5':
            from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
            config_class, model_class = Qwen3_5Config, Qwen3_5ForConditionalGeneration
            vision.pop('deepstack_visual_indexes')
            text.update(layer_types=['linear_attention', 'full_attention'],
                        linear_num_key_heads=2, linear_num_value_heads=2,
                        linear_key_head_dim=8, linear_value_head_dim=8,
                        rope_parameters={'rope_type': 'default', 'mrope_section': [1, 1, 2],
                                         'partial_rotary_factor': 1.0})
        config = config_class(vision_config=vision, text_config=text,
                              image_token_id=12, vision_start_token_id=13, vision_end_token_id=14, video_token_id=15)
        base = root / 'model'
        model_class(config).save_pretrained(base)
        processor.save_pretrained(base)
        if args.model_path:
            base = Path(args.model_path).resolve()
        dataset = root / 'datasets' / 'tiny'
        dataset.mkdir(parents=True)
        Image.new('RGB', (16, 16), 'red').save(dataset / 'red.png')
        data = dataset / 'data.jsonl'
        data.write_text((json.dumps({'image': 'red.png', 'question': 'what color ?', 'answer': args.reward_target if args.stage == 'vlm_grpo' else 'red square'}) + '\n') * 8)
        client = TestClient(app)
        def completed(run_id):
            run = root / '.runs' / run_id
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                result = client.get(f'/api/training/runs/{run_id}').json()
                if result['status'] == 'error':
                    raise AssertionError((run / 'stderr.log').read_text()[-7000:])
                if result['status'] == 'completed' and not processes.is_running(run):
                    metrics = json.loads((run / 'metrics.json').read_text())
                    assert metrics['loss_history'] and all(math.isfinite(loss) for loss in metrics['loss_history']), metrics
                    return result
                time.sleep(.25)
            raise AssertionError((run / 'stderr.log').read_text()[-7000:])
        try:
            settings = {'stage': args.stage, 'model_path': str(base),
                'dataset_path': str(data), 'batch_size': 1, 'gradient_accumulation': 1, 'max_steps': 2,
                'warmup_steps': 0, 'seq_len': 128, 'save_every': 1, 'log_every': 1,
                'mixed_precision': 'bf16' if args.backend == 'unsloth' or args.method == 'qlora' else 'no', 'gpu_ids': [1],
                'training_backend': args.backend, 'grpo_algorithm': args.algorithm,
                'grpo_group_size': 8, 'grpo_max_new_tokens': 8, 'grpo_reward_fn': args.reward_fn, 'grpo_kl_weight': 0.04,
                'tuning_method': args.method, 'use_lora': args.method != 'full', 'lora_r': 2, 'lora_alpha': 4,
                'lora_target_modules': ['q_proj', 'v_proj'], 'gradient_checkpointing': True, 'use_flash_attention': False}
            if args.model_path:
                settings.update(seq_len=1024, min_pixels=4096, max_pixels=4096, mixed_precision='bf16')
            if args.precision:
                settings['mixed_precision'] = args.precision
            preflight = client.post('/api/vlm/preflight', json=settings)
            assert preflight.status_code == 200 and preflight.json()['ready'], preflight.text
            assert preflight.json()['sample']['supervised_tokens'] > 0
            assert preflight.json()['datasets'][0]['valid_rows'] == 8
            assert not list((root / '.runs').glob('*/config.json')), 'Preflight launched a training worker'
            print('PASS processor-only preflight: images, CUDA, masks and targets before training', flush=True)
            if args.preflight_only:
                return
            response = client.post('/api/vlm/start', json=settings)
            assert response.status_code == 200, response.text
            run_id = response.json()['run_id']
            first = completed(run_id)
            assert first['current_step'] == 2, first
            if args.stage == 'vlm_grpo':
                stats = json.loads((root / '.runs' / run_id / 'metrics.json').read_text())
                assert any(value > 0 for value in stats['grad_norm_history']), 'GRPO produced no policy gradient'
                assert any(value > 0 for value in stats['reward_std_history']), f"All group rewards are equal; no reward-driven learning signal: {stats.get('sample_outputs')}"
                assert stats['rl_engine'] == 'homellm_native', stats
                assert stats['native_loss_class'].startswith('homellm.training.rl.loss.'), stats
                assert stats['reward_history'] and stats['kl_history'], stats
                if args.algorithm == 'sdpo':
                    assert stats['sdpo_teacher_forwards'] > 0 and any(value > 0 for value in stats['sdpo_distill_history']), stats
                print(f"PASS policy gradient, group rewards={stats['reward_history']}, KL={stats['kl_history']}", flush=True)
            print(f'PASS {args.stage}/{args.method}: real VLM finite loss, relative image paths, step 2', flush=True)
            response = client.post(f'/api/training/runs/{run_id}/continue', json={'max_steps': 4})
            assert response.status_code == 200, response.text
            result = completed(response.json()['run_id'])
            assert result['current_step'] == 4, result
            print('PASS VLM optimizer/checkpoint resume 2 -> 4', flush=True)
            final = Path(result['config']['output_dir']) / 'final_model'
            from safetensors.torch import load_file
            if args.method == 'full':
                before, after = load_file(str(base / 'model.safetensors')), load_file(str(final / 'model.safetensors'))
                language_keys = [key for key in before if not any(part in key for part in ('vision_tower', 'visual', 'multi_modal_projector'))]
                assert language_keys and any(not torch.equal(before[key], after[key]) for key in language_keys), 'Language weights did not change'
                if args.stage == 'vlm_pretrain':
                    assert all(torch.equal(before[key], after[key]) for key in before if 'vision_tower' in key), 'Frozen vision weights changed'
            else:
                weights = load_file(str(final / 'adapter_model.safetensors'))
                assert any(value.count_nonzero() for key, value in weights.items() if 'lora_B' in key), 'LoRA did not learn'
            print('PASS trainable weights changed; freeze policy preserved', flush=True)
            if args.verify_resume:
                assert args.stage == 'vlm_grpo'
                baseline_request = client.post('/api/vlm/start', json={**settings, 'max_steps': 4})
                assert baseline_request.status_code == 200, baseline_request.text
                baseline = completed(baseline_request.json()['run_id'])
                checkpoint = Path(baseline['config']['output_dir']) / 'checkpoint-2'
                replay_request = client.post('/api/vlm/start', json={**settings, 'max_steps': 4,
                    'resume_from_checkpoint': str(checkpoint)})
                assert replay_request.status_code == 200, replay_request.text
                replay = completed(replay_request.json()['run_id'])
                filename = 'model.safetensors' if args.method == 'full' else 'adapter_model.safetensors'
                baseline_weights = load_file(str(Path(baseline['config']['output_dir']) / 'final_model' / filename))
                replay_weights = load_file(str(Path(replay['config']['output_dir']) / 'final_model' / filename))
                assert baseline_weights.keys() == replay_weights.keys()
                assert all(torch.equal(baseline_weights[key], replay_weights[key]) for key in baseline_weights), 'Native optimizer/RNG/data resume diverged from uninterrupted run'
                print('PASS native uninterrupted/resumed weights are bitwise identical', flush=True)
            response = client.post('/api/vlm/load', json={'model_path': str(final)})
            assert response.status_code == 200, response.text
            response = client.post('/api/vlm/generate', json={'prompt': 'what color ?', 'image_base64':
                base64.b64encode((dataset / 'red.png').read_bytes()).decode(), 'max_tokens': 2, 'temperature': 0})
            assert response.status_code == 200 and isinstance(response.json()['text'], str), response.text
            print('PASS exported VLM/adapter reload and image inference', flush=True)
            if args.result:
                Path(args.result).write_text(json.dumps({'family': args.family, 'method': args.method,
                    'backend': args.backend, 'stage': args.stage, 'algorithm': args.algorithm,
                    'pretrained': args.model_path, 'weights_changed': True, 'resume_step': 4,
                    'image_inference': response.json()['text'],
                    'metrics': json.loads((root / '.runs' / run_id / 'metrics.json').read_text())}, indent=2))
        finally:
            processes.shutdown()
            vlm.unload_vlm()


if __name__ == '__main__':
    main()
