"""Offline CPU end-to-end: one blueprint -> VLM/VLA train, resume, export and inference."""
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with tempfile.TemporaryDirectory(prefix='mah-research-smoke-') as temporary:
        root = Path(temporary)
        os.environ.update(MODELS_AT_HOME_ROOT=temporary, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        from PIL import Image
        import torch
        from safetensors.torch import load_file
        from fastapi.testclient import TestClient
        from backend.api import app
        from backend.services import processes
        for name, color in [('red', 'red'), ('blue', 'blue')]:
            Image.new('RGB', (32, 32), color).save(root / (name + '.png'))
        spec = json.loads((Path(__file__).resolve().parents[1] / 'examples/model-builder/looped-transformer.json').read_text())
        spec['max_position_embeddings'] = 128
        common = {'model_blueprint_spec': spec, 'source_blueprint_spec': spec, 'vision_image_size':32, 'vision_patch_size':8,
            'research_diagnostics':True,'training_backend':'transformers', 'tuning_method':'full', 'device':'cpu', 'mixed_precision':'no',
            'batch_size':2, 'gradient_accumulation':1, 'learning_rate':.001, 'max_steps':2, 'warmup_steps':0, 'save_every':1}
        vlm_data = root / 'captions.jsonl'
        vlm_data.write_text(''.join(json.dumps({'image':str(root/'red.png'),'question':'color?','answer':'red'})+'\n' for _ in range(4)))
        vla_data = root / 'actions.jsonl'
        vla_data.write_text(''.join(json.dumps({'image':str(root/'red.png'),'instruction':'move','state':[0.,0.],
            'actions':[{'continuous':[.1,-.1],'discrete':[1]},{'continuous':[.2,-.2],'discrete':[2]}]})+'\n' for _ in range(4)))
        client = TestClient(app)
        def post(url, value):
            response = client.post(url, json=value)
            assert response.status_code == 200, (url, response.text)
            return response.json()
        def complete(run):
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                value = client.get(f'/api/training/runs/{run}').json()
                if value['status'] == 'completed' and not processes.is_running(root/'.runs'/run):
                    return value
                if value['status'] == 'error':
                    raise AssertionError((root/'.runs'/run/'stderr.log').read_text()[-5000:])
                time.sleep(.2)
            raise AssertionError('Research training exceeded 120s')
        try:
            for task, data in [('vlm', vlm_data), ('vla', vla_data)]:
                config = {**common, 'architecture':'blueprint_'+task, 'dataset_path':str(data), 'stage':task+'_sft'}
                if task == 'vlm':
                    config.update(use_lora=False, assistant_only_loss=True, freeze_vision_tower=False, eval_steps=1)
                else:
                    config.update(action_type='hybrid', action_dim=2, discrete_classes=[3], action_horizon=2, state_dim=2, hidden_dim=32)
                report = post(f'/api/{task}/preflight', config)
                assert report['ready'], report
                run = post(f'/api/{task}/start', config)['run_id']
                result = complete(run)
                assert result['current_step'] == 2
                signals=client.get(f'/api/training/runs/{run}/metrics').json()['metrics'].get('research_signals')
                assert signals and signals['blocks'], f'{task}: missing real-batch signals'
                assert any(row['gradient_rms_sampled'] is not None for row in signals['blocks'])
                output = Path(result['config']['output_dir'])
                filename = 'model.safetensors' if task == 'vlm' else 'policy.safetensors'
                first, second = load_file(str(output/'checkpoint-1'/filename)), load_file(str(output/'checkpoint-2'/filename))
                key = 'core.patch_projection.weight' if task == 'vlm' else 'research_core.patch_projection.weight'
                assert not torch.equal(first[key], second[key]), f'{task}: vision weights did not update'
                continued = post(f'/api/training/runs/{run}/continue', {'max_steps':3})['run_id']
                resumed = complete(continued)
                assert resumed['current_step'] == 3 and resumed['config']['source_blueprint_spec'] == result['config']['source_blueprint_spec']
                exported = str(Path(resumed['config']['output_dir'])/'final_model')
                image = base64.b64encode((root/'red.png').read_bytes()).decode()
                if task == 'vlm':
                    post('/api/vlm/load', {'model_path':exported})
                    generated = post('/api/vlm/generate', {'prompt':'color?','image_base64':image,'max_tokens':3,'temperature':0})
                    assert isinstance(generated['text'], str)
                    post('/api/vlm/unload', {})
                else:
                    prediction = post('/api/vla/predict', {'model_path':exported,'instruction':'move','state':[0.,0.],'image_base64':image,'device':'cpu'})
                    assert len(prediction['actions']) == 2 and len(prediction['actions'][0]['continuous']) == 2
                    assert prediction['action_type'] == 'hybrid'
                print(f'PASS {task.upper()} blueprint preflight, real vision-gradient updates, resume, frozen architecture snapshot, export and inference', flush=True)
        finally:
            processes.shutdown()
            client.post('/api/vlm/unload')


if __name__ == '__main__':
    main()
