"""Real action-policy train/resume/predict for robotics, games and desktop schemas."""
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
    parser.add_argument('--domain', choices=['robotics', 'simulation', 'desktop'], default='robotics')
    parser.add_argument('--action-type', choices=['continuous', 'discrete', 'hybrid'], default='continuous')
    parser.add_argument('--model-path')
    parser.add_argument('--tiny-backbone', action='store_true', help='Use a tiny random Qwen3-VL with processor from model-path')
    parser.add_argument('--backend', choices=['transformers', 'unsloth'], default='transformers')
    parser.add_argument('--method', choices=['full', 'lora', 'qlora'], default='lora')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--result')
    parser.add_argument('--verify-resume', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mah-vla-smoke-') as temporary:
        os.environ.update(MODELS_AT_HOME_ROOT=temporary, CUDA_DEVICE_ORDER='PCI_BUS_ID')
        from PIL import Image
        from fastapi.testclient import TestClient
        from backend.api import app
        from backend.services import processes
        root = Path(temporary)
        if args.tiny_backbone:
            import torch
            from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
            processor = AutoProcessor.from_pretrained(args.model_path)
            config = Qwen3VLConfig(text_config={'vocab_size': len(processor.tokenizer), 'hidden_size': 32,
                'intermediate_size': 64, 'num_hidden_layers': 1, 'num_attention_heads': 4,
                'num_key_value_heads': 2, 'head_dim': 8, 'max_position_embeddings': 512,
                'rope_parameters': {'rope_type': 'default', 'mrope_section': [1, 1, 2]}},
                vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64,
                    'out_hidden_size': 32, 'num_heads': 4, 'patch_size': 16,
                    'spatial_merge_size': 2, 'temporal_patch_size': 2,
                    'num_position_embeddings': 64, 'deepstack_visual_indexes': []},
                image_token_id=processor.image_token_id, video_token_id=processor.video_token_id,
                vision_start_token_id=processor.vision_start_token_id, vision_end_token_id=processor.vision_end_token_id)
            model_path = root / 'tiny_backbone'
            torch.manual_seed(42)
            Qwen3VLForConditionalGeneration(config).save_pretrained(model_path)
            processor.save_pretrained(model_path)
            args.model_path = str(model_path)
        rows = []
        state_dim = 1 if args.domain == 'robotics' else 0
        classes = [4, 16] if args.domain == 'desktop' else [6] if args.domain == 'simulation' else [2, 3]
        for index in range(16):
            red = index % 2 == 0
            Image.new('RGB', (32, 32), 'red' if red else 'blue').save(root / f'{index}.png')
            value = .8 if red else -.8
            action = [value, -value] if args.action_type == 'continuous' else int(red) if args.action_type == 'discrete' else {'continuous': [value, -value], 'discrete': [int(red) if index == 0 else int(not red) for index in range(len(classes))]}
            rows.append({'image': f'{index}.png', 'instruction': 'go red' if red else 'go blue', 'state': [value] if state_dim else [],
                         'actions': [action] * (1 if index == 0 else 2)})
        data = root / 'data.jsonl'; data.write_text('\n'.join(json.dumps(row) for row in rows))
        validation = root / 'val.jsonl'; validation.write_text('\n'.join(json.dumps(row) for row in rows[:4]))
        settings = {'architecture': 'vlm_action' if args.model_path else 'native_vla',
            'training_backend': args.backend, 'model_path': args.model_path or '', 'dataset_path': str(data), 'val_data_path': str(validation),
            'domain': args.domain, 'action_type': args.action_type, 'action_dim': 2, 'action_horizon': 2, 'state_dim': state_dim,
            'discrete_classes': classes if args.action_type == 'hybrid' else [],
            'batch_size': 2, 'max_steps': 3 if args.model_path else 12, 'save_every': 1,
            'eval_every': 2, 'learning_rate': .0003 if args.model_path else .003,
            'mixed_precision': 'bf16' if args.device == 'cuda' else 'no', 'gpu_ids': [args.gpu_id],
            'device': args.device, 'image_size': 32, 'vision_grid': 4, 'tuning_method': args.method if args.model_path else 'full',
            'lora_r': 2, 'lora_alpha': 4, 'seq_len': 512, 'min_pixels': 4096, 'max_pixels': 4096}
        def finished(client, run_id):
            directory = root / '.runs' / run_id
            deadline = time.monotonic() + 500
            while time.monotonic() < deadline:
                value = client.get(f'/api/training/runs/{run_id}').json()
                if value['status'] == 'error':
                    raise AssertionError((directory / 'stderr.log').read_text()[-6000:])
                if value['status'] == 'completed' and not processes.is_running(directory):
                    return value
                time.sleep(.2)
            raise AssertionError((directory / 'stderr.log').read_text()[-6000:])
        with TestClient(app) as client:
            response = client.post('/api/vla/preflight', json=settings)
            assert response.status_code == 200 and response.json()['ready'], response.text
            assert not list((root / '.runs').glob('*/config.json'))
            print('PASS VLA preflight: schema, masks, normalization, image/processor, no training launched', flush=True)
            response = client.post('/api/vla/start', json=settings)
            assert response.status_code == 200, response.text
            first = finished(client, response.json()['run_id'])
            stats = json.loads((root / '.runs' / first['run_id'] / 'metrics.json').read_text())
            assert all(math.isfinite(v) for v in stats['loss_history'])
            assert any(v > 0 for v in stats['grad_norm_history'])
            assert 'current_val_loss' in stats
            if not args.model_path:
                assert min(stats['loss_history'][-4:]) < stats['loss_history'][0] * .8, stats['loss_history']
            print('PASS real action loss, nonzero gradients, validation and learning', flush=True)
            response = client.post(f"/api/training/runs/{first['run_id']}/continue", json={'max_steps': settings['max_steps'] + 2})
            assert response.status_code == 200, response.text
            resumed = finished(client, response.json()['run_id'])
            assert resumed['current_step'] == settings['max_steps'] + 2
            model = str(Path(resumed['config']['output_dir']) / 'final_model')
            from safetensors.torch import load_file
            artifact = 'action_head.safetensors' if args.model_path else 'policy.safetensors'
            before = load_file(str(Path(first['config']['output_dir']) / 'checkpoint-1' / artifact))
            after = load_file(str(Path(model) / artifact))
            import torch
            assert any(not torch.equal(before[key], after[key]) for key in before), 'Policy/action head weights did not change'
            if args.model_path and args.method != 'full':
                adapter = load_file(str(Path(model) / 'backbone' / 'adapter_model.safetensors'))
                assert any(value.count_nonzero() for key, value in adapter.items() if 'lora_B' in key), 'VLM LoRA backbone did not learn'
            if args.tiny_backbone and args.method == 'full':
                old_backbone = load_file(str(Path(first['config']['output_dir']) / 'checkpoint-1' / 'backbone' / 'model.safetensors'))
                new_backbone = load_file(str(Path(model) / 'backbone' / 'model.safetensors'))
                language = [key for key in old_backbone if 'visual' not in key]
                assert any(not torch.equal(old_backbone[key], new_backbone[key]) for key in language), 'Full VLM backbone did not update'
            if args.verify_resume:
                assert not args.model_path and args.device == 'cpu'
                replay_request = client.post('/api/vla/start', json={**settings,
                    'resume_from_checkpoint': str(Path(first['config']['output_dir']) / 'checkpoint-2')})
                assert replay_request.status_code == 200, replay_request.text
                replay = finished(client, replay_request.json()['run_id'])
                baseline_weights = load_file(str(Path(first['config']['output_dir']) / 'final_model' / 'policy.safetensors'))
                replay_weights = load_file(str(Path(replay['config']['output_dir']) / 'final_model' / 'policy.safetensors'))
                assert all(torch.equal(baseline_weights[key], replay_weights[key]) for key in baseline_weights), 'VLA optimizer/RNG/data resume diverged'
                print('PASS VLA uninterrupted and resumed policy weights are bitwise identical', flush=True)
            print('PASS action optimizer checkpoint resume and learned policy weights' + (' + VLM backbone' if args.model_path else ''), flush=True)
            response = client.post('/api/vla/predict', json={'model_path': model, 'instruction': 'go red', 'state': [.8] if state_dim else [],
                'image_base64': base64.b64encode((root / '0.png').read_bytes()).decode(), 'device': args.device, 'gpu_ids': [args.gpu_id]})
            assert response.status_code == 200, response.text
            result = response.json()
            assert len(result['actions']) == 2 and result['action_type'] == args.action_type
            if args.action_type == 'hybrid':
                assert len(result['actions'][0]['continuous']) == 2 and len(result['actions'][0]['discrete']) == len(classes)
            print('PASS exported policy reload and image/state/instruction -> action chunk', flush=True)
            if args.result:
                Path(args.result).write_text(json.dumps({'settings': settings, 'metrics': stats, 'prediction': result,
                    'resumed_step': resumed['current_step']}, indent=2))


if __name__ == '__main__':
    main()
