"""Run with legacy API Python: modern VLM preflight/train/resume/inference stay isolated."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--backend', choices=['transformers', 'unsloth'], default='transformers')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mah-isolated-vlm-') as temporary:
        os.environ.update(MODELS_AT_HOME_ROOT=temporary, CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_ORDER='PCI_BUS_ID')
        from PIL import Image
        from fastapi.testclient import TestClient
        from backend.api import app
        from backend.services import processes
        from backend.routes import vlm
        from backend.services.vlm_environment import python_command
        assert python_command() != sys.executable, 'Run this script with the legacy API .venv Python'
        root = Path(temporary)
        Image.new('RGB', (32, 32), 'red').save(root / 'red.png')
        data = root / 'data.jsonl'
        data.write_text((json.dumps({'image': 'red.png', 'question': 'What color is this?', 'answer': 'red'}) + '\n') * 8)
        settings = {'stage': 'vlm_sft', 'model_path': args.model_path, 'dataset_path': str(data),
                    'training_backend': args.backend, 'tuning_method': 'lora', 'lora_r': 2, 'lora_alpha': 4,
                    'batch_size': 1, 'gradient_accumulation': 1, 'max_steps': 2, 'warmup_steps': 0,
                    'seq_len': 1024, 'min_pixels': 4096, 'max_pixels': 4096, 'mixed_precision': 'bf16',
                    'use_flash_attention': False, 'log_every': 1, 'save_every': 1}
        def completed(client, run_id):
            run = root / '.runs' / run_id
            until = time.monotonic() + 600
            while time.monotonic() < until:
                result = client.get(f'/api/training/runs/{run_id}').json()
                if result['status'] == 'error':
                    raise AssertionError((run / 'stderr.log').read_text()[-6000:])
                if result['status'] == 'completed' and not processes.is_running(run):
                    return result
                time.sleep(.3)
            raise AssertionError((run / 'stderr.log').read_text()[-6000:])
        with TestClient(app) as client:
            response = client.post('/api/vlm/preflight', json=settings)
            assert response.json()['ready'], response.text
            response = client.post('/api/vlm/start', json=settings)
            assert response.status_code == 200, response.text
            first = completed(client, response.json()['run_id'])
            response = client.post(f"/api/training/runs/{first['run_id']}/continue", json={'max_steps': 4})
            assert response.status_code == 200, response.text
            final = completed(client, response.json()['run_id'])
            assert final['current_step'] == 4
            model = str(Path(final['config']['output_dir']) / 'final_model')
            response = client.post('/api/vlm/load', json={'model_path': model})
            assert response.status_code == 200, response.text
            worker = vlm._vlm_state['worker']
            assert worker.alive and client.get('/api/vlm/status').json()['loaded']
            response = client.post('/api/vlm/generate', json={'prompt': 'What color?', 'max_tokens': 3, 'temperature': 0,
                'image_base64': base64.b64encode((root / 'red.png').read_bytes()).decode()})
            assert response.status_code == 200 and isinstance(response.json()['text'], str), response.text
            assert client.post('/api/vlm/generate', json={'prompt': 'test', 'image_base64': 'invalid!'}).status_code == 400
            client.post('/api/vlm/unload')
            assert not worker.alive and not client.get('/api/vlm/status').json()['loaded']
            import transformers
            assert transformers.__version__.startswith('4.'), transformers.__version__
            print('PASS isolated preflight, actual pretrained LoRA, resume 2 -> 4, persistent inference, input error and GPU worker cleanup; legacy API retains Transformers 4', flush=True)


if __name__ == '__main__':
    main()
