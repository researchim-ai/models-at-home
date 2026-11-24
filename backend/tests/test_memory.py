import sys
from types import SimpleNamespace

import pytest

from backend.paths import atomic_json


@pytest.mark.parametrize('architecture', [
    {'hidden_size': 1024, 'num_hidden_layers': 24, 'num_attention_heads': 16, 'vocab_size': 32000, 'intermediate_size': 2816},
    {'n_embd': 1024, 'n_layer': 24, 'n_head': 16, 'vocab_size': 32000, 'intermediate_size': 2816},
])
@pytest.mark.parametrize('detailed', [False, True])
def test_saved_architecture_overrides_ui_defaults(client, workspace, monkeypatch, architecture, detailed):
    model = workspace / 'models' / 'larger'
    model.mkdir()
    atomic_json(model / 'config.json', architecture)
    received = []
    def estimate(config, **kwargs):
        received.append(config)
        return {'total_gb': 12}
    monkeypatch.setitem(sys.modules, 'homellm.models.memory_estimator', SimpleNamespace(estimate_memory_footprint=estimate))
    config = {'model_type': 'hf', 'model_path': 'models/larger', 'hidden_size': 512, 'num_layers': 8, 'n_heads': 8,
              'vocab_size': 50257, 'stage': 'sft', 'seq_len': 2048, 'sft_max_seq_length': 512}
    response = client.post('/api/models/estimate-memory-detailed' if detailed else '/api/models/estimate-memory',
                           json={'config': config} if detailed else config)
    assert response.status_code == 200, response.text
    assert received[0]['hidden_size'] == 1024
    assert received[0]['num_layers'] == 24
    assert received[0]['n_heads'] == 16
    assert received[0]['vocab_size'] == 32000
    assert received[0]['intermediate_size'] == 2816
    assert received[0]['seq_len'] == 512


def test_arbitrary_graph_estimation_is_explicitly_unsupported(client):
    assert client.post('/api/models/estimate-memory', json={'model_type': 'blueprint'}).status_code == 400
    assert client.post('/api/models/estimate-memory-detailed', json={'config': {'model_type': 'blueprint'}}).status_code == 400
    assert client.post('/api/models/estimate-memory-detailed', json={'config': {}, 'num_gpus': 0}).status_code == 422
