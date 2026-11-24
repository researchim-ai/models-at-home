import json
from types import SimpleNamespace

import pytest

from backend.paths import atomic_json
from backend.routes import training
from backend.services import processes
from backend.services.blueprints import validate, with_tokenizer


def blueprint():
    return {'hidden_size': 32, 'vocab_size': 100, 'max_position_embeddings': 64,
            'blocks': [{'id': 'embed', 'type': 'token_embedding', 'params': {'vocab_size': 100, 'hidden_size': 32}},
                       {'id': 'attn', 'type': 'attention', 'inputs': ['embed'], 'params': {'num_heads': 4}}]}


@pytest.mark.parametrize('edit', [
    lambda b: b['blocks'][1].update(id='embed'),
    lambda b: b['blocks'][0].update(id='input'),
    lambda b: b['blocks'][1].update(inputs=['missing']),
    lambda b: b['blocks'][0].update(inputs=['attn']),
    lambda b: b['blocks'][1]['params'].update(num_heads=3),
    lambda b: b['blocks'][0]['params'].update(vocab_size=99),
    lambda b: b['blocks'][1].update(type='add'),
    lambda b: b['blocks'][1].update(inputs=['embed', 'embed']),
    lambda b: b['blocks'][1]['params'].update(top_k='invalid'),
    lambda b: b['blocks'][1].update(type='positional_embedding', params={'hidden_size': 16}),
    lambda b: b['blocks'][1].update(type='positional_embedding', params={'max_position_embeddings': 16}),
])
def test_invalid_graph_never_saved(client, workspace, edit):
    value = blueprint()
    edit(value)
    response = client.post('/api/blueprints/save', json={'name': 'broken', 'blueprint': value})
    assert response.status_code == 400, response.text
    assert not (workspace / 'blueprints' / 'broken.json').exists()


def test_valid_residual_graph_and_vocabulary_sync():
    value = blueprint()
    value['blocks'].append({'id': 'residual', 'type': 'add', 'inputs': ['embed', 'attn']})
    validate(value)
    synced = with_tokenizer(value, 9)
    assert synced['vocab_size'] == synced['blocks'][0]['params']['vocab_size'] == 9
    assert value['vocab_size'] == value['blocks'][0]['params']['vocab_size'] == 100


def test_blueprint_snapshot_is_immutable_on_resume(client, workspace, monkeypatch):
    monkeypatch.setattr(processes, 'launch', lambda *args, **kwargs: SimpleNamespace(pid=99999999, poll=lambda: None))
    monkeypatch.setattr(training, '_raise_if_exited_early', lambda *args: None)
    value = blueprint()
    saved = client.post('/api/blueprints/save', json={'name': 'tiny', 'blueprint': value})
    assert saved.status_code == 200, saved.text
    data = workspace / 'datasets' / 'tiny.txt'
    data.write_text('tiny training data')
    response = client.post('/api/training/start', json={'model_type': 'blueprint', 'model_blueprint': saved.json()['path'],
                            'dataset_path': str(data), 'seq_len': 64, 'max_steps': 2})
    assert response.status_code == 200, response.text
    run = response.json()['run_id']
    first = json.loads((workspace / '.runs' / run / 'config.json').read_text())
    atomic_json(workspace / 'blueprints' / 'tiny.json', {'broken': True})
    (workspace / 'out' / run / 'checkpoint_step1').mkdir()
    resumed = client.post(f'/api/training/runs/{run}/continue')
    assert resumed.status_code == 200, resumed.text
    config = json.loads((workspace / '.runs' / resumed.json()['run_id'] / 'config.json').read_text())
    assert config['model_blueprint_spec'] == first['model_blueprint_spec']
    assert config['hidden_size'] == 32


@pytest.mark.parametrize('name', ['', ' ', '..', '../escape', 'folder/model'])
def test_blueprint_filename_rejected(client, name):
    assert client.post('/api/blueprints/save', json={'name': name, 'blueprint': blueprint()}).status_code == 400


def test_checkpoint_architecture_alias_accepted():
    assert training.TrainingRequest(stage='sft', model_type='checkpoint').model_type == 'hf'


def test_json_validation_normalizes_moe_without_saving(client, workspace):
    value = blueprint()
    value['blocks'].append({'id': 'experts', 'type': 'moe', 'params': {'num_experts': 4, 'top_k': 1}})
    response = client.post('/api/blueprints/validate', json=value)
    assert response.status_code == 200, response.text
    params = response.json()['blueprint']['blocks'][-1]['params']
    assert params == {'num_experts': 4, 'num_select': 1}
    assert list((workspace / 'blueprints').iterdir()) == []
    value['blocks'][-1]['params']['top_k'] = 5
    assert client.post('/api/blueprints/validate', json=value).status_code == 400


def test_unused_graph_branch_rejected(client):
    value = blueprint()
    value['blocks'].append({'id': 'output', 'type': 'layernorm', 'inputs': ['embed']})
    assert client.post('/api/blueprints/validate', json=value).status_code == 400


def test_unsupported_blueprint_grpo_rejected_before_launch(client, workspace):
    model = workspace / 'models' / 'graph'
    model.mkdir()
    atomic_json(model / 'config.json', {'model_type': 'homellm_blueprint'})
    data = workspace / 'datasets' / 'tiny.jsonl'
    data.write_text('{}\n')
    response = client.post('/api/training/start', json={'stage': 'grpo', 'model_path': str(model), 'dataset_path': str(data)})
    assert response.status_code == 400 and 'not supported' in response.text
    assert list((workspace / '.runs').glob('*/config.json')) == []
