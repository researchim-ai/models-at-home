import json
import subprocess
import pytest
from types import SimpleNamespace

from backend.routes import vlm


def payload(workspace):
    path = workspace / 'datasets' / 'vision.jsonl'
    path.write_text('{}\n')
    return {'model_path': 'test/model', 'dataset_path': str(path), 'gpu_ids': [1]}


def test_preflight_isolated_in_selected_gpu_without_creating_run(client, workspace, monkeypatch):
    called = []
    def probe(command, **kwargs):
        called.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({'ready': True, 'checks': [], 'datasets': [], 'sample': None}))
    monkeypatch.setattr(vlm.subprocess, 'run', probe)
    response = client.post('/api/vlm/preflight', json=payload(workspace))
    assert response.status_code == 200 and response.json()['ready']
    command, options = called[0]
    assert command[1].endswith('vlm_preflight_worker.py')
    assert options['env']['CUDA_VISIBLE_DEVICES'] == '1'
    assert json.loads(options['input'])['data_base_dir'] == str(workspace / 'datasets')
    assert options['timeout'] == 90
    assert list((workspace / '.runs').glob('*/config.json')) == []


def test_preflight_timeout_is_actionable(client, workspace, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('preflight', 90)
    monkeypatch.setattr(vlm.subprocess, 'run', timeout)
    report = client.post('/api/vlm/preflight', json=payload(workspace)).json()
    assert not report['ready'] and '90' in report['checks'][0]['detail']


def test_preflight_invalid_config_never_spawns_probe(client, workspace, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid configuration must not spawn a child')
    monkeypatch.setattr(vlm.subprocess, 'run', forbidden)
    report = client.post('/api/vlm/preflight', json={**payload(workspace), 'gpu_ids': [0, 1]}).json()
    assert not report['ready']


def test_recipe_catalog_does_not_claim_untested_scenarios(client):
    result = client.get('/api/vlm/recipes').json()
    qwen = next(item for item in result['recipes'] if item['model_type'] == 'qwen3_vl')
    assert qwen['evidence'] == 'tiny-local-gpu-smoke'
    assert {'stage': 'vlm_pretrain', 'tuning_method': 'lora'} not in qwen['scenarios']
    assert result['tested_profile']['transformers'] == '5.17.0'


def test_recipe_version_drift_and_unknown_families_are_warnings(monkeypatch):
    from backend.services import vlm_recipes as recipes
    monkeypatch.setattr(recipes, 'catalog', lambda: {'installed': {**recipes.PROFILE, 'transformers': '4.57.3'}})
    config = {'stage': 'vlm_sft', 'tuning_method': 'lora'}
    status, detail = recipes.evidence('qwen3_vl', config)
    assert status == 'warning' and '4.57.3' in detail
    assert recipes.evidence('new-family', config)[0] == 'warning'


def test_native_sdpo_reaches_the_multimodal_probe(client, workspace, monkeypatch):
    called = []
    def probe(*args, **kwargs):
        called.append(json.loads(kwargs['input']))
        return SimpleNamespace(returncode=0, stdout=json.dumps({'ready': True, 'checks': [], 'sample': None, 'datasets': []}))
    monkeypatch.setattr(vlm.subprocess, 'run', probe)
    report = client.post('/api/vlm/preflight', json={**payload(workspace),
        'stage': 'vlm_grpo', 'grpo_algorithm': 'sdpo'}).json()
    assert report['ready'] and called[0]['grpo_algorithm'] == 'sdpo'


def test_dapo_defaults_are_explicit():
    config = vlm.VLMTrainingRequest(stage='vlm_grpo', grpo_algorithm='dapo')
    assert config.grpo_kl_weight == 0 and config.grpo_clip_eps_high == .28
    config = vlm.VLMTrainingRequest(stage='vlm_grpo', grpo_algorithm='dapo', grpo_kl_weight=.03)
    assert config.grpo_kl_weight == .03


def test_recipe_does_not_certify_other_policy_objectives():
    from backend.services.vlm_recipes import evidence
    config = {'stage': 'vlm_grpo', 'tuning_method': 'full', 'training_backend': 'transformers', 'grpo_algorithm': 'gspo'}
    assert evidence('qwen3_vl', config)[0] == 'warning'
    config.update(tuning_method='lora', training_backend='unsloth', grpo_algorithm='dr_grpo')
    assert evidence('qwen3_5', config)[0] == 'warning'


def test_legacy_reward_sft_checkpoint_cannot_resume_as_true_grpo(client, workspace):
    run = workspace / '.runs' / 'vlm_legacy'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps({'stage': 'vlm_grpo', 'max_steps': 10}))
    response = client.post('/api/training/runs/vlm_legacy/continue', json={'max_steps': 20})
    assert response.status_code == 400 and 'reward-filtered SFT' in response.text


def test_native_resume_rejects_other_engines_before_model_loading(client, workspace, monkeypatch):
    checkpoint = workspace / 'checkpoint-2'
    checkpoint.mkdir()
    def forbidden(*args, **kwargs):
        raise AssertionError('Incompatible resume must not spawn preflight')
    monkeypatch.setattr(vlm.subprocess, 'run', forbidden)
    report = client.post('/api/vlm/preflight', json={**payload(workspace), 'stage': 'vlm_grpo',
        'resume_from_checkpoint': str(checkpoint)}).json()
    assert not report['ready'] and 'complete native' in report['checks'][0]['detail']


@pytest.mark.parametrize('options', [{'sdpo_ema_rate': .1}, {'sdpo_full_logit_distillation': True},
                                   {'sdpo_distillation_topk': 50}])
def test_native_sdpo_rejects_unimplemented_transport_settings(client, workspace, options):
    report = client.post('/api/vlm/preflight', json={**payload(workspace), 'stage': 'vlm_grpo',
        'grpo_algorithm': 'sdpo', **options}).json()
    assert not report['ready'] and 'selected-token' in report['checks'][0]['detail']
