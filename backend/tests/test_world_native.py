import base64
import json
from pathlib import Path
import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('safetensors')
from backend.routes.world_models import WorldRequest
from backend.services.world_data import Episodes, datasets, action
from backend.services.world_model import WorldModel
from backend.services.world_training import train
from backend.services.world_worker import rollout
from safetensors.torch import load_file

DEMO = Path(__file__).resolve().parents[2] / 'examples' / 'world-models'

def config(tmp_path, **kw):
    return WorldRequest(dataset_path=str(DEMO / 'train.jsonl'), val_data_path=str(DEMO / 'validation.jsonl'),
        state_dim=2, hidden_dim=32, latent_dim=8, batch_size=2, device='cpu', max_steps=3, save_every=1, eval_every=2,
        output_dir=str(tmp_path / 'out'), **kw).model_dump()


def test_episode_contract_and_terminal_boundaries(tmp_path):
    c = config(tmp_path)
    rows = [json.loads((DEMO / 'train.jsonl').read_text().splitlines()[0])]
    rows[0]['frames'] = [str(DEMO / f) for f in rows[0]['frames']]
    rows[0]['dones'][2] = True
    path = tmp_path / 'data.jsonl'; path.write_text(json.dumps(rows[0])+'\n')
    data = Episodes(path, c)
    assert [start for _, start in data.windows] == [3, 4]
    # The transition with terminal target can end a shorter window.
    c['sequence_length'] = 3
    assert (0, 0) in Episodes(path, c).windows
    rows[0]['actions'][0] = [float('nan'), 0]
    path.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match='finite'):
        Episodes(path, c)
    with pytest.raises(ValueError, match='category'):
        action(True, {**c, 'action_type': 'discrete', 'action_dim': 4})
    assert action({'continuous': [1, 2], 'discrete': [2]}, {**c, 'action_type': 'hybrid', 'discrete_classes': [3]}).tolist() == [1, 2, 0, 0, 1]


def test_holdout_overlap_rejected(tmp_path):
    c = config(tmp_path); c['val_data_path'] = c['dataset_path']
    with pytest.raises(ValueError, match='independent'):
        datasets(c)


@pytest.mark.parametrize('architecture', ['latent_gru', 'rssm'])
def test_native_gradients_action_conditioning_and_resume(tmp_path, architecture):
    torch.set_num_threads(2)
    c = config(tmp_path, architecture=architecture)
    data, _ = datasets(c)
    b = data.batch([0, 1], data.statistics(), torch.device('cpu'))
    model = WorldModel(c)
    loss, parts = model.objective(b)
    loss.backward()
    assert torch.isfinite(loss) and sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None) > 0
    assert parts['kl'] > 0 if architecture == 'rssm' else parts['kl'] == 0
    model.eval()
    with torch.no_grad():
        predicted = model.rollout(b['images'][:, 0], b['states'][:, 0], b['actions'])[0]
        changed_actions = b['actions'].clone(); changed_actions[:, 2] += 2
        changed = model.rollout(b['images'][:, 0], b['states'][:, 0], changed_actions)[0]
        assert torch.equal(predicted[:, :2], changed[:, :2])
        assert not torch.equal(predicted[:, 2:], changed[:, 2:])
        altered_future = {**b, 'images': b['images'].clone()}; altered_future['images'][:, 1:] = 1
        # Reported image objective is strictly prior rollout MSE, with no posterior targets in transition.
        _, parts = model.objective(altered_future)
        assert torch.allclose(parts['image_loss'], (predicted - 1).square().mean())
        masks = {**b, 'reward_mask': torch.zeros_like(b['reward_mask']), 'done_mask': torch.zeros_like(b['done_mask'])}
        _, parts = model.objective(masks)
        assert parts['reward_loss'] == 0 and parts['done_loss'] == 0
    result = train(c, tmp_path / 'metrics.json')
    original = load_file(str(Path(c['output_dir']) / 'final_model' / 'model.safetensors'))
    resumed = {**c, 'resume_from_checkpoint': str(Path(c['output_dir']) / 'checkpoint-1'), 'output_dir': str(tmp_path / 'resumed')}
    train(resumed, tmp_path / 'resume-metrics.json')
    repeated = load_file(str(Path(resumed['output_dir']) / 'final_model' / 'model.safetensors'))
    assert all(torch.equal(original[k], repeated[k]) for k in original)
    episode = data.episodes[0]
    request = dict(model_path=result['final_model_dir'], image_base64=base64.b64encode(Path(episode['frames'][0]).read_bytes()).decode(),
        actions=[[2, 0], [0, 2]], state=[10, 10], device='cpu')
    output = rollout(request)
    assert len(output['frames']) == 2 and len(output['states']) == 2
    assert len(output['rewards']) == 2 and all(0 <= p <= 1 for p in output['done_probabilities'])
    changed = {**c, 'resume_from_checkpoint': resumed['resume_from_checkpoint'], 'learning_rate': .002}
    with pytest.raises(ValueError, match='configuration changed'):
        train(changed, tmp_path / 'bad.json')


def test_optional_targets_are_omitted_from_export_prediction(tmp_path):
    row = json.loads((DEMO / 'train.jsonl').read_text().splitlines()[0])
    row['frames'] = [str(DEMO / f) for f in row['frames']]
    for key in ('states', 'rewards', 'dones'):
        row.pop(key)
    path = tmp_path / 'unlabelled.jsonl'; path.write_text(json.dumps(row))
    c = config(tmp_path); c.update(dataset_path=str(path), val_data_path='', state_dim=0, max_steps=1)
    result = train(c, tmp_path / 'metrics.json')
    prediction = rollout(dict(model_path=result['final_model_dir'], image_base64=base64.b64encode(Path(row['frames'][0]).read_bytes()).decode(),
        actions=row['actions'][:2], state=[], device='cpu'))
    assert len(prediction['frames']) == 2
    assert all(prediction[key] is None for key in ('states', 'rewards', 'done_probabilities'))
