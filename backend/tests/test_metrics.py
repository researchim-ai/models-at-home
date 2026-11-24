import json

from backend.paths import atomic_json
from backend.services import metrics


def test_persisted_history_reloads_through_api(client, workspace):
    run = workspace / '.runs' / 'history'
    run.mkdir()
    atomic_json(run / 'config.json', {})
    atomic_json(run / 'metrics.json', {'current_step': 3, 'current_loss': 0.4, 'current_lr': 0.01,
                'steps_history': [1, 2, 3], 'loss_history': [0.8, 0.6, 0.4], 'val_loss_history': [0.7],
                'status': 'completed'})
    data = client.get('/api/training/runs/history/metrics', params={'limit': 2}).json()
    assert data['metrics']['step'] == 3
    assert data['metrics']['loss'] == 0.4
    assert [point['step'] for point in data['history']] == [2, 3]
    assert [point['loss'] for point in data['history']] == [0.6, 0.4]
    assert all('val_loss' not in point for point in data['history'])
    assert client.get('/api/training/runs/history/metrics?limit=0').status_code == 400
    # A partially written legacy metrics file must not turn into a server error.
    (run / 'metrics.json').write_text('{')
    assert client.get('/api/training/runs/history/metrics').json() == {'metrics': None, 'history': []}


def test_nonfinite_metrics_remain_valid_json():
    point = metrics.snapshot({'current_step': 2, 'current_loss': float('nan'),
                              'grad_norm': float('inf'), 'current_lr': 0.1,
                              'gpu_stats': [None, {'memory_used_gb': 1, 'memory_total_gb': 2}]})
    assert point['loss'] is None and point['grad_norm'] is None
    assert point['gpu_stats'][0]['memory_percent'] == 50
    json.dumps(point, allow_nan=False)


def test_sparse_validation_history_uses_its_own_step_numbers():
    data = {'steps_history': [1, 2, 3, 4], 'loss_history': [1, 0.9, 0.8, 0.7],
            'val_steps_history': [2, 4], 'val_loss_history': [0.95, 0.75], 'current_val_loss': 0.75}
    points = metrics.history(data)
    assert 'val_loss' not in points[0] and 'val_loss' not in points[2]
    assert points[1]['val_loss'] == 0.95 and points[3]['val_loss'] == 0.75
    assert metrics.snapshot(data)['val_loss'] == 0.75
