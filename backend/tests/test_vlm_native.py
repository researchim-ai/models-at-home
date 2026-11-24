"""Contracts of the VLM bridge to homellm's own RL core."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('homellm.training.rl.loss')
from backend.services.vlm_grpo import native_config, sequence_inputs, selected_log_probs, teacher_messages
from backend.services.vlm_native_trainer import NativeVLMTrainer


def settings(algorithm='grpo'):
    return {'grpo_algorithm': algorithm, 'grpo_group_size': 4, 'grpo_max_new_tokens': 8,
            'grpo_temperature': 1., 'grpo_clip_eps_low': .2, 'grpo_clip_eps_high': .2,
            'grpo_kl_weight': .04}


@pytest.mark.parametrize('algorithm', ['grpo', 'dr_grpo', 'dapo', 'sdpo'])
def test_native_loss_factory_is_reused(algorithm):
    from homellm.training.rl.loss import GRPOLoss, SDPOLoss, create_loss_function
    config = native_config(settings(algorithm))
    loss = create_loss_function(None, config)
    assert type(loss) is (SDPOLoss if algorithm == 'sdpo' else GRPOLoss)
    assert config.use_std_normalization == (algorithm == 'grpo')
    assert config.dynamic_sampling == (algorithm in ('dapo', 'sdpo'))
    assert config.token_level_loss == (algorithm == 'dapo')
    if algorithm == 'dapo':
        assert config.clip_eps_high == .28


def test_vision_tensors_are_preserved_and_only_completion_actions_are_selected():
    class Model(torch.nn.Module):
        def forward(self, input_ids, pixel_values, **kwargs):
            logits = pixel_values.mean() * torch.arange(5.).view(1, 1, 5).expand(1, input_ids.shape[1], 5)
            return SimpleNamespace(logits=logits)
    ids = torch.tensor([[1, 2, 3, 4]])
    pixels = torch.ones(1, 3, 2, 2, requires_grad=True)
    prompt = {'input_ids': ids[:, :2], 'pixel_values': pixels, 'image_grid_thw': torch.ones(1, 3),
              'mm_token_type_ids': torch.tensor([[1, 0]])}
    inputs = sequence_inputs(prompt, ids)
    assert inputs['pixel_values'] is pixels and inputs['image_grid_thw'] is prompt['image_grid_thw']
    assert inputs['mm_token_type_ids'].tolist() == [[1, 0, 0, 0]]
    values = selected_log_probs(Model(), inputs, 2)
    expected = (torch.arange(5.)).log_softmax(0)[torch.tensor([3, 4])]
    assert values.shape == (1, 2) and torch.allclose(values[0], expected)
    values.sum().backward()
    assert pixels.grad.abs().sum() > 0


def test_teacher_reprompt_retains_original_images_and_does_not_mutate_student():
    prompt = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'color?'}]}]
    teacher = teacher_messages(prompt, 'red')
    assert teacher[0]['content'][0] == {'type': 'image'}
    assert 'red' in teacher[-1]['content'][-1]['text']
    assert len(prompt[0]['content']) == 2 and 'red' not in str(prompt)


def test_checkpoint_fingerprint_normalizes_float_defaults_and_checks_data(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text('first\n')
    trainer = NativeVLMTrainer.__new__(NativeVLMTrainer)
    trainer.config = {**settings(), 'data_path': str(path), 'grpo_overlong_penalty': -1}
    original = trainer._signature()
    trainer.config['grpo_overlong_penalty'] = -1.0
    assert trainer._signature() == original
    path.write_text('changed\n')
    assert trainer._signature() != original


def test_refuses_other_engine_checkpoints_before_restoring_optimizer(tmp_path):
    trainer = NativeVLMTrainer.__new__(NativeVLMTrainer)
    trainer.config = {'resume_from_checkpoint': str(tmp_path)}
    with pytest.raises(ValueError, match='TRL/legacy'):
        trainer._restore()


def test_native_loss_does_not_import_trl(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'trl', None)
    from homellm.training.rl.loss import create_loss_function
    loss = create_loss_function(None, native_config(settings('grpo')))
    assert type(loss).__module__ == 'homellm.training.rl.loss'


def test_native_dynamic_sampling_refills_before_optimizer_update():
    class Buffer:
        items = []
        def clear(self): pass
    class Metrics:
        def update(self, **kwargs): pass
    trainer = NativeVLMTrainer.__new__(NativeVLMTrainer)
    trainer.config = {'max_steps': 1, 'batch_size': 1, 'gradient_accumulation': 1}
    trainer.rl = SimpleNamespace(max_refill_rounds=2)
    trainer.buffer, trainer.sidecars, trainer.metrics = Buffer(), {}, Metrics()
    trainer.step, trainer.attempts = 0, 0
    def collect():
        trainer.attempts += 1
        return False, torch.ones(4)
    trainer._collect = collect
    trainer._update = lambda: pytest.fail('Must not optimise a filtered empty buffer')
    with pytest.raises(ValueError, match='exhausted refill'):
        trainer.train()
    assert trainer.attempts == 3 and trainer.step == 0
