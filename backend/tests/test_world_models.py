import json
from pathlib import Path
from backend.routes import world_models as wm


def test_contracts_and_missing_data(client):
    assert client.post('/api/world-models/preflight', json={'action_type': 'hybrid'}).status_code == 422
    assert client.post('/api/world-models/start', json={'dataset_path': '/missing'}).status_code == 400
    result = client.post('/api/world-models/preflight', json={'dataset_path': '/missing'}).json()
    assert result['ready'] is False
    assert client.post('/api/world-models/rollout', json={'model_path': '/missing', 'image_base64': '', 'actions': []}).status_code == 422
    demo = client.get('/api/world-models/examples').json()
    assert Path(demo['dataset_path']).is_file()


def test_job_resume_filters_incomplete_checkpoints(client, workspace, monkeypatch):
    data = workspace / 'data.jsonl'; data.write_text('{}\n')
    monkeypatch.setattr(wm.processes, 'launch', lambda *a, **k: None)
    r = client.post('/api/world-models/start', json={'dataset_path': str(data), 'max_steps': 2}).json()
    config = json.loads((workspace / '.runs' / r['run_id'] / 'config.json').read_text())
    assert config['stage'] == 'world_model' and config['world_worker_version'] == 1
    run = Path(config['output_dir']) / 'checkpoint-1'; run.mkdir()
    assert client.post(f"/api/training/runs/{r['run_id']}/continue", json={'max_steps': 3}).status_code == 400
    for name in ('model.safetensors', 'world.json', 'training_state.pt'):
        (run / name).write_text('test')
    (run / 'complete.json').write_text(json.dumps({'format': 'mah_world_training_v1', 'step': 1}))
    response = client.post(f"/api/training/runs/{r['run_id']}/continue", json={'max_steps': 3})
    assert response.status_code == 200, response.text
    assert len(client.get('/api/world-models/runs').json()['runs']) == 2


def test_launch_failure_is_visible_in_history(client, workspace, monkeypatch):
    data = workspace / 'data.jsonl'; data.write_text('{}\n')
    def fail(*a, **kw):
        raise OSError('missing ML python')
    monkeypatch.setattr(wm.processes, 'launch', fail)
    assert client.post('/api/world-models/start', json={'dataset_path': str(data)}).status_code == 500
    run = client.get('/api/world-models/runs').json()['runs'][0]
    assert run['status'] == 'error' and not run['process_running']
