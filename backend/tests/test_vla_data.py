import json
import pytest

torch = pytest.importorskip('torch')
from backend.routes.vla import VLATrainingRequest
from backend.services.vla_data import Demonstrations, batch
from backend.services.vla_training import action_loss
from backend.services.vla_model import ActionPolicy


@pytest.fixture
def demo(tmp_path):
    from PIL import Image
    Image.new('RGB', (16, 16), 'red').save(tmp_path / 'red.png')
    path = tmp_path / 'data.jsonl'
    def make(row=None, **options):
        config = VLATrainingRequest(action_dim=2, action_horizon=3, state_dim=1, **options).model_dump()
        value = {'image': 'red.png', 'instruction': 'move red', 'state': [2.], 'actions': [[3., 4.]]}
        value.update(row or {})
        path.write_text(json.dumps(value) + '\n')
        return Demonstrations(path, config), config
    return make


def test_terminal_chunk_padding_is_masked_and_excluded_from_normalization(demo):
    data, config = demo()
    stats = data.statistics()
    assert stats['action']['mean'] == [3., 4.] and stats['action']['std'] == [1., 1.]
    _, _, targets, mask = batch(data.rows, config, stats)
    assert mask.tolist() == [[True, False, False]]
    predictions = torch.zeros(1, 3, 2, requires_grad=True)
    predictions.data[:, 1:] = 1000
    loss = action_loss(predictions, targets, mask, config)
    assert loss == 0
    loss.backward()
    assert predictions.grad[:, 1:].count_nonzero() == 0


@pytest.mark.parametrize('row', [{'actions': [[1.]]}, {'state': []}, {'actions': []},
    {'actions': [[float('nan'), 2.]]}, {'instruction': ''}, {'image': 'missing.png'}])
def test_bad_demonstration_is_not_silently_skipped(demo, row):
    with pytest.raises(ValueError, match='line 1'):
        demo(row)


def test_hybrid_actions_keep_classes_out_of_continuous_normalization(demo):
    data, config = demo({'actions': [{'continuous': [3., 4.], 'discrete': [1, 2]}]},
                        action_type='hybrid', discrete_classes=[2, 3])
    _, _, targets, mask = batch(data.rows, config, data.statistics())
    assert targets[0, 0].tolist() == [0., 0., 1., 2.]
    prediction = torch.zeros(1, 3, 7, requires_grad=True)
    action_loss(prediction, targets, mask, config).backward()
    assert prediction.grad[0, 0, 2:].abs().sum() > 0
    assert prediction.grad[0, 1:].count_nonzero() == 0


def test_native_policy_uses_vision_language_and_state(demo):
    torch.manual_seed(42)
    data, config = demo()
    model = ActionPolicy(config).eval()
    inputs, states, _, _ = batch(data.rows, config, data.statistics())
    first = model(inputs, states)
    changed_image = model({**inputs, 'images': 1 - inputs['images']}, states)
    changed_text = model({**inputs, 'instructions': ['move blue']}, states)
    changed_state = model(inputs, states + 1)
    assert first.shape == (1, 3, 2)
    for value in (changed_image, changed_text, changed_state):
        assert not torch.allclose(first, value)


def test_preflight_rejects_an_unobserved_future_horizon(demo, monkeypatch):
    import torch
    from backend.services.vla_worker import preflight
    data, config = demo()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    report = preflight({**config, 'data_path': str(data.rows[0]['image']).replace('red.png', 'data.jsonl')})
    assert not report['ready'] and 'full action horizon' in report['checks'][-1]['detail']


def test_native_spatial_grid_distinguishes_object_locations(demo):
    torch.manual_seed(42)
    _, config = demo()
    model = ActionPolicy(config).eval()
    left, right = torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64)
    left[:, :, 16:32, 0:16] = 1
    right[:, :, 16:32, 48:64] = 1
    state = torch.zeros(1, 1)
    a = model({'images': left, 'instructions': ['click target']}, state)
    b = model({'images': right, 'instructions': ['click target']}, state)
    assert not torch.allclose(a, b)
