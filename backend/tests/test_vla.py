import json
from types import SimpleNamespace

import pytest
from backend.routes import vla


def payload(workspace, **extra):
    path = workspace / 'datasets' / 'actions.jsonl'
    path.write_text('{}\n')
    return {'dataset_path': str(path), **extra}


@pytest.mark.parametrize('extra', [{'action_dim': 0}, {'action_horizon': 0}, {'state_dim': -1},
    {'action_type': 'discrete', 'action_dim': 1}, {'action_type': 'hybrid'},
    {'action_type': 'hybrid', 'discrete_classes': [1]}, {'gpu_ids': [0, 1]},
    {'architecture': 'native_vla', 'tuning_method': 'lora'}, {'architecture': 'vlm_action'},
    {'learning_rate': float('nan')}, {'action_names': ['x']},
    {'action_type': 'hybrid', 'discrete_classes': [2], 'discrete_labels': [['only_one']]}])
def test_invalid_action_contract_rejected(client, workspace, extra):
    # JSON cannot carry NaN on the wire; exercise Pydantic directly for that case.
    if extra.get('learning_rate') != extra.get('learning_rate'):
        with pytest.raises(ValueError):
            vla.VLATrainingRequest(**payload(workspace, **extra))
    else:
        response = client.post('/api/vla/start', json=payload(workspace, **extra))
        assert response.status_code == 422, response.text
    assert not list((workspace / '.runs').glob('*/config.json'))


def test_preflight_does_not_launch_training_and_preserves_gpu(client, workspace, monkeypatch):
    calls = []
    def probe(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout=json.dumps({'ready': True, 'checks': [], 'sample': None}))
    monkeypatch.setattr(vla.subprocess, 'run', probe)
    response = client.post('/api/vla/preflight', json=payload(workspace, gpu_ids=[1]))
    assert response.json()['ready']
    command, options = calls[0]
    assert command[-1] == 'preflight' and options['env']['CUDA_VISIBLE_DEVICES'] == '1'
    assert not list((workspace / '.runs').glob('*/config.json'))


def test_vla_run_is_separate_and_resumes_via_common_api(client, workspace, monkeypatch):
    calls = []
    monkeypatch.setattr(vla.processes, 'launch', lambda run, command, **kwargs: calls.append((run, command)))
    first = client.post('/api/vla/start', json=payload(workspace, max_steps=2)).json()['run_id']
    assert first.startswith('vla_') and calls[0][1][2:4] == ['--mode', 'train']
    config = json.loads((workspace / '.runs' / first / 'config.json').read_text())
    assert config['stage'] == 'vla_sft' and config['model_type'] == 'vla'
    checkpoint = __import__('pathlib').Path(config['output_dir']) / 'checkpoint-2'
    checkpoint.mkdir()
    (checkpoint / 'training_state.pt').write_bytes(b'placeholder')
    (checkpoint / 'complete.json').write_text(json.dumps({'format': 'mah_vla_training_v1', 'step': 2}))
    response = client.post(f'/api/training/runs/{first}/continue', json={'max_steps': 4})
    assert response.status_code == 200, response.text
    second = json.loads((workspace / '.runs' / response.json()['run_id'] / 'config.json').read_text())
    assert second['resume_from_checkpoint'] == str(checkpoint) and second['max_steps'] == 4
    assert len(client.get('/api/vla/runs').json()['runs']) == 2
    assert second['output_dir'] != config['output_dir']


def test_holdout_must_be_a_different_file(client, workspace):
    settings = payload(workspace)
    response = client.post('/api/vla/preflight', json={**settings, 'val_data_path': settings['dataset_path']})
    assert not response.json()['ready']


def test_three_synthetic_domains_are_available(client):
    result = client.get('/api/vla/examples').json()['examples']
    assert {row['domain'] for row in result} == {'robotics', 'simulation', 'desktop'}
    assert all(row['synthetic'] for row in result)


def test_incomplete_checkpoint_is_not_accepted(tmp_path):
    (tmp_path / 'training_state.pt').write_bytes(b'state')
    (tmp_path / 'complete.json').write_text('{broken')
    assert not vla.complete_checkpoint(tmp_path)
