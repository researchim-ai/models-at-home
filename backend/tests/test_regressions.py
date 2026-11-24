import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, WebSocketDisconnect

from backend.paths import safe_child, atomic_json
from backend.routes import training, datasets, study, notebooks, vlm, chat
from backend.services import processes
from backend import ws


@pytest.fixture
def fake_worker(monkeypatch):
    launches = []
    def launch(run_dir, cmd, **kwargs):
        launches.append((run_dir, cmd, kwargs))
        return SimpleNamespace(pid=99999999, poll=lambda: None)
    monkeypatch.setattr(processes, "launch", launch)
    monkeypatch.setattr(training, "_raise_if_exited_early", lambda *args: None)
    return launches


def start(client, workspace, **config):
    data = workspace / "datasets" / "tiny.txt"
    data.write_text("tiny synthetic training dataset")
    return client.post("/api/training/start", json={"dataset_path": str(data), **config})


def test_runs_are_isolated_and_resume_selects_latest_step(client, workspace, fake_worker):
    first = start(client, workspace).json()["run_id"]
    second = start(client, workspace).json()["run_id"]
    configs = [json.loads((workspace / ".runs" / run / "config.json").read_text()) for run in (first, second)]
    assert configs[0]["output_dir"] != configs[1]["output_dir"]
    assert configs[0]["betas"] == [0.9, 0.95]
    assert fake_worker[0][1][0] == sys.executable
    out = workspace / "out" / first
    (out / "checkpoint_step9").mkdir()
    (out / "checkpoint_step100").mkdir()
    (out / "final_model").mkdir()
    response = client.post(f"/api/training/runs/{first}/continue")
    assert response.status_code == 200, response.text
    resumed = json.loads((workspace / ".runs" / response.json()["run_id"] / "config.json").read_text())
    assert resumed["resume_from_checkpoint"] == str(out / "checkpoint_step100")
    assert resumed["resumed_from_run"] == first
    assert resumed["output_dir"] != str(out)
    runs = client.get("/api/training/runs").json()["runs"]
    old = next(r for r in runs if r["run_id"] == first)
    assert old["final_model_path"] == str(out / "final_model")
    assert old["started_at"] and isinstance(old["started_at"], str)
    assert len(runs) == 3  # agent_sessions must not appear as a run.


def test_resume_without_checkpoint_rejected(client, workspace, fake_worker):
    run = start(client, workspace).json()["run_id"]
    assert client.post(f"/api/training/runs/{run}/continue").status_code == 400


@pytest.mark.parametrize("fields, expected", [
    ({"num_heads": 16}, {"n_heads": 16, "num_heads": 16}),
    ({"flash_attention": False}, {"flash_attention": False, "use_flash_attention": False}),
    ({"lora_rank": 64}, {"lora_r": 64, "lora_rank": 64}),
    ({"gradient_checkpointing": False}, {"grad_checkpoint": False}),
    ({"stage": "sft", "sft_max_seq_length": 4096}, {"seq_len": 4096}),
    ({"num_epochs": 3}, {"epochs": 3}),
])
def test_aliases_honor_explicit_input(fields, expected):
    config = training.TrainingRequest(**fields).model_dump()
    for key, value in expected.items():
        assert config[key] == value


@pytest.mark.parametrize("config", [
    {"batch_size": 0}, {"learning_rate": -1}, {"hidden_size": 512, "n_heads": 7},
    {"betas": "0.9,no"}, {"betas": [0.9, 1]}, {"max_steps": 0}, {"gpu_ids": [-1]},
    {"stage": "not-a-stage"}, {"parallel_mode": "not-a-mode"},
    {"parallel_mode": "multi_gpu"}, {"stage": "grpo", "model_path": "model", "dataset_path": ""},
])
def test_invalid_config_does_not_create_runs(client, workspace, fake_worker, config):
    response = start(client, workspace, **config)
    assert response.status_code in (400, 422), response.text
    assert fake_worker == []
    assert list((workspace / ".runs").glob("*/config.json")) == []


@pytest.mark.parametrize("value", ["", ".", "..", "../outside", "/tmp/outside", "a/../../outside", "..\\outside"])
def test_safe_child_blocks_escape_and_root(workspace, value):
    with pytest.raises(HTTPException):
        safe_child(workspace / "datasets", value)


def test_symlink_escape_rejected(client, workspace):
    outside = workspace / "outside.txt"
    outside.write_text("synthetic secret")
    (workspace / "datasets" / "linked.txt").symlink_to(outside)
    assert client.get("/api/datasets/preview/linked.txt").status_code == 400
    assert client.delete("/api/datasets/linked.txt").status_code == 400
    assert outside.read_text() == "synthetic secret"


def test_preview_and_study_containment(client, workspace):
    sibling = workspace / "study_materials-extra"
    sibling.mkdir()
    (sibling / "secret.md").write_text("synthetic secret")
    assert client.get("/api/study/document", params={"path": "../study_materials-extra/secret.md"}).status_code == 400
    assert client.get("/api/study/index", params={"lang": "../"}).status_code == 400
    assert client.get("/api/datasets/preview/%2E%2E/outside.txt").status_code == 400
    assert client.get("/api/datasets/preview/missing", params={"limit": -1}).status_code == 400
    assert client.delete("/api/models/local/%2E").status_code == 400


@pytest.mark.parametrize("endpoint, body", [
    ("/api/models/download", {"repo_id": "test/model", "save_name": "../escape"}),
    ("/api/datasets/download", {"repo_id": "test/data", "save_as": "../escape"}),
])
def test_download_paths_rejected_before_network(client, endpoint, body):
    response = client.post(endpoint, json=body)
    assert response.status_code == 400, response.text


def test_unknown_origin_cannot_trigger_action(client):
    response = client.post("/api/notebooks/jupyter-start", headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403
    assert client.get("/api/system/health", headers={"Origin": "null"}).status_code == 200


def test_cannot_delete_or_resume_live_unowned_pid(client, workspace):
    run = workspace / ".runs" / "unowned"
    run.mkdir()
    atomic_json(run / "config.json", {"output_dir": "out"})
    (run / "pid").write_text(str(os.getpid()))
    assert client.delete("/api/training/runs/unowned").status_code == 409
    assert client.post("/api/training/runs/unowned/continue").status_code == 409
    assert client.post("/api/training/stop/unowned").status_code == 409
    assert (run / "config.json").exists()


def test_process_manager_stops_owned_worker(workspace):
    run = workspace / ".runs" / "child"
    run.mkdir()
    proc = processes.launch(run, [sys.executable, "-c", "import time; time.sleep(60)"], cwd=workspace, env=os.environ.copy())
    try:
        assert processes.is_running(run)
        processes.stop(run)
        assert proc.poll() is not None
        assert not (run / "pid").exists()
        assert json.loads((run / "metrics.json").read_text())["status"] == "stopped"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_metrics_send_final_status_at_same_step(workspace, monkeypatch):
    run = workspace / ".runs" / "metric-run"
    run.mkdir()
    atomic_json(run / "metrics.json", {"step": 1, "loss": 0.5, "status": "running"})
    sent = []
    class Socket:
        headers = {}
        url = SimpleNamespace(scheme="ws", netloc="localhost")
        async def accept(self, **kwargs): pass
        async def close(self, **kwargs): pass
        async def send_json(self, value): sent.append(value)
    async def tick(_):
        if len(sent) == 1:
            atomic_json(run / "metrics.json", {"step": 1, "loss": 0.5, "status": "completed"})
        else:
            raise WebSocketDisconnect()
    monkeypatch.setattr(ws.asyncio, "sleep", tick)
    asyncio.run(ws.metrics_ws(Socket(), "metric-run"))
    assert [m["status"] for m in sent] == ["running", "completed"]


def test_jupyter_probe_uses_configured_port(monkeypatch):
    monkeypatch.setattr(notebooks, "JUPYTER_PORT", 8899)
    assert notebooks._probe_url(notebooks.JUPYTER_PUBLIC_URL) == "http://127.0.0.1:8899"
    with pytest.raises(HTTPException):
        notebooks._probe_url("file:///tmp/test")


def test_vlm_worker_receives_required_metrics_argument(client, workspace, fake_worker, monkeypatch):
    monkeypatch.setattr(vlm, "_raise_if_exited_early", lambda *args: None)
    data = workspace / "datasets" / "images.jsonl"
    data.write_text('{}\n')
    response = client.post('/api/vlm/start', json={"model_path": "local/model", "dataset_path": str(data)})
    assert response.status_code == 200, response.text
    run = response.json()['run_id']
    assert fake_worker[0][1][-2:] == ['--metrics', str(workspace / '.runs' / run / 'metrics.json')]


def test_grpo_bridge_passes_json_and_ui_run_dir(workspace, monkeypatch):
    from backend.services import grpo_worker
    config = workspace / '.runs' / 'grpo' / 'config.json'
    atomic_json(config, {'stage': 'grpo', 'grpo_algorithm': 'dr_grpo', 'output_dir': str(workspace / 'out')})
    monkeypatch.setattr(sys, 'argv', ['bridge', '--config', str(config)])
    invoked = []
    monkeypatch.setattr(grpo_worker.runpy, 'run_module', lambda *args, **kwargs: invoked.append((args, kwargs)))
    grpo_worker.main()
    assert sys.argv[1] == '--config_json'
    assert json.loads(sys.argv[2])['ui_run_dir'] == str(config.parent)
    assert json.loads(sys.argv[2])['grpo_algorithm'] == 'drgrpo'
    assert invoked[0][0][0] == 'homellm.training.rl.train_rl'


def test_health_remains_responsive_during_blocking_request(client, workspace, monkeypatch):
    import threading
    import types
    entered, release = threading.Event(), threading.Event()
    def load_dataset(**kwargs):
        entered.set()
        release.wait(timeout=3)
        return SimpleNamespace(save_to_disk=lambda path: None)
    monkeypatch.setitem(sys.modules, 'datasets', types.SimpleNamespace(load_dataset=load_dataset))
    result = []
    worker = threading.Thread(target=lambda: result.append(client.post('/api/datasets/download', json={'repo_id': 'fake/data'})))
    worker.start()
    try:
        assert entered.wait(timeout=2)
        before = time.monotonic()
        assert client.get('/api/system/health').status_code == 200
        assert time.monotonic() - before < 1
    finally:
        release.set()
        worker.join(timeout=3)
    assert result[0].status_code == 200


def test_chat_generation_failure_ends_stream_and_reports_error(monkeypatch):
    import contextlib
    import queue
    class Streamer:
        def __init__(self, *args, **kwargs): self.queue = queue.Queue()
        def __next__(self):
            value = self.queue.get(timeout=1)
            if value is None: raise StopIteration
            return value
        def on_finalized_text(self, text, stream_end=False):
            if stream_end: self.queue.put(None)
    class Inputs(dict):
        def to(self, device): return self
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs): return 'prompt'
        def __call__(self, *args, **kwargs):
            assert kwargs['return_token_type_ids'] is False
            return Inputs()
    def fail(**kwargs): raise RuntimeError('synthetic OOM')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(inference_mode=contextlib.nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        TextIteratorStreamer=Streamer, StoppingCriteria=object, StoppingCriteriaList=list))
    monkeypatch.setattr(chat, '_ensure_chat_backend', lambda req: {
        'model': SimpleNamespace(device='cpu', generate=fail), 'tokenizer': Tokenizer()})
    result = list(chat._stream_generate(chat.GenerateRequest(messages=[], model_path='fake')))
    assert 'synthetic OOM' in ''.join(result)
    assert result[-1] == 'data: [DONE]\n\n'
    assert not chat._model_lock.locked()


def test_session_token_required_for_desktop_requests(client, monkeypatch):
    monkeypatch.setenv('MAH_API_TOKEN', 'synthetic-test-token')
    assert client.get('/api/system/health').status_code == 401
    assert client.get('/api/system/health', headers={'X-Studio-Token': 'wrong'}).status_code == 401
    assert client.get('/api/system/health', headers={'X-Studio-Token': 'synthetic-test-token', 'Origin': 'null'}).status_code == 200
    preflight = client.options('/api/training/start', headers={
        'Origin': 'null', 'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type,x-studio-token'})
    assert preflight.status_code == 200


def test_csv_preview_and_gguf_listing(client, workspace):
    (workspace / 'datasets' / 'table.csv').write_text('text,label\nhello,1\nworld,2\n')
    assert client.get('/api/datasets/preview/table.csv', params={'limit': 1}).json()['rows'] == [{'text': 'hello', 'label': '1'}]
    (workspace / 'models' / 'tiny.gguf').write_bytes(b'synthetic')
    assert client.get('/api/models/local').json()['models'][0]['name'] == 'tiny.gguf'


def test_hf_scratch_does_not_load_pretrained_weights(client, workspace, fake_worker):
    response = start(client, workspace, model_type='hf', base_model_path='fake/model')
    assert response.status_code == 200, response.text
    config = json.loads((fake_worker[0][0] / 'config.json').read_text())
    assert config['base_model_path'] == ''
    assert config['model_id'] == 'fake/model'
    assert config['tokenizer_path'] == 'fake/model'


def test_sft_template_reaches_legacy_worker(client, workspace, fake_worker):
    model = workspace / 'base-model'
    model.mkdir()
    response = start(client, workspace, stage='sft', base_model_path=str(model), sft_chat_template='synthetic-template')
    assert response.status_code == 200, response.text
    assert json.loads((fake_worker[0][0] / 'config.json').read_text())['chat_template'] == 'synthetic-template'


@pytest.mark.parametrize('fmt', ['chat', 'instruct'])
def test_sft_format_and_controls_reach_worker(client, workspace, fake_worker, fmt):
    model = workspace / 'base-model'
    model.mkdir()
    response = start(client, workspace, stage='sft', model_path=str(model),
                     sft_data_format=fmt, sft_packing=True, sft_train_on_completions_only=True)
    assert response.status_code == 200, response.text
    config = json.loads((fake_worker[0][0] / 'config.json').read_text())
    assert config['sft_columns'] == {'format': fmt}
    assert config['sft_packing'] and config['sft_train_on_completions_only']
    assert fake_worker[0][1][1].endswith('text_worker.py')


def test_sft_context_overflow_rejected_before_launch(client, workspace, fake_worker):
    model = workspace / 'small-model'
    model.mkdir()
    atomic_json(model / 'config.json', {'model_type': 'homellm', 'max_position_embeddings': 64})
    response = start(client, workspace, stage='sft', model_path=str(model), sft_max_seq_length=128)
    assert response.status_code == 400 and 'context length' in response.text
    assert fake_worker == []


@pytest.mark.parametrize('fields', [
    {'dropout': 1}, {'lora_dropout': -0.1}, {'min_lr_ratio': 2}, {'eps': 0},
    {'mixed_precision': 'unknown'}, {'optimizer': 'typo'}, {'training_backend': 'typo'},
    {'hidden_size': 24, 'n_heads': 8}, {'lr_scheduler': 'typo'},
    {'model_type': 'home_moe', 'num_experts': 2, 'num_experts_per_tok': 3},
])
def test_invalid_training_settings_rejected(client, workspace, fake_worker, fields):
    assert start(client, workspace, **fields).status_code in (400, 422)
    assert fake_worker == []


def test_vlm_grpo_uses_studio_worker_and_persists_policy_settings(client, workspace, fake_worker, monkeypatch):
    monkeypatch.setattr(vlm, '_raise_if_exited_early', lambda *args: None)
    data = workspace / 'datasets' / 'vlm.jsonl'
    data.write_text('{}\n')
    payload = {'stage': 'vlm_grpo', 'dataset_path': str(data), 'model_path': 'test/model',
               'grpo_algorithm': 'sdpo', 'grpo_group_size': 4, 'training_backend': 'unsloth', 'max_steps': 10}
    response = client.post('/api/vlm/start', json=payload)
    assert response.status_code == 200, response.text
    config = json.loads((fake_worker[0][0] / 'config.json').read_text())
    assert config['stage'] == 'vlm_grpo' and config['grpo_algorithm'] == 'sdpo'
    assert config['training_backend'] == 'unsloth' and config['max_steps'] == 10
    assert any('vlm_worker.py' in part for part in fake_worker[0][1])


@pytest.mark.parametrize('fields', [{'max_steps': 0}, {'grpo_group_size': 1},
    {'grpo_algorithm': 'fake_sft'}, {'seq_len': 128, 'grpo_max_new_tokens': 256}, {'gpu_ids': [-1]}])
def test_vlm_grpo_invalid_settings_rejected(client, workspace, fake_worker, fields):
    data = workspace / 'datasets' / 'vlm.jsonl'
    data.write_text('{}\n')
    result = client.post('/api/vlm/start', json={'stage': 'vlm_grpo', 'dataset_path': str(data),
                         'model_path': 'test/model', **fields})
    assert result.status_code in (400, 422), result.text
    assert not fake_worker


def test_vlm_explicit_tuning_method_controls_lora(client, workspace, fake_worker, monkeypatch):
    monkeypatch.setattr(vlm, '_raise_if_exited_early', lambda *args: None)
    data = workspace / 'datasets' / 'vlm.jsonl'
    data.write_text('{}\n')
    response = client.post('/api/vlm/start', json={'dataset_path': str(data), 'model_path': 'test/model',
                                                'use_lora': True, 'tuning_method': 'full'})
    assert response.status_code == 200, response.text
    config = json.loads((fake_worker[0][0] / 'config.json').read_text())
    assert config['use_lora'] is False


def test_finished_plan_requires_explicit_extension(client, workspace, fake_worker):
    run = start(client, workspace, max_steps=2).json()['run_id']
    (workspace / 'out' / run / 'checkpoint_step2').mkdir()
    assert client.post(f'/api/training/runs/{run}/continue').status_code == 400
    assert client.post(f'/api/training/runs/{run}/continue', json={'max_steps': 4}).status_code == 200


def test_websocket_rejects_invalid_session_token(workspace, monkeypatch):
    monkeypatch.setenv('MAH_API_TOKEN', 'test-token')
    closed = []
    class Socket:
        headers = {'origin': 'null', 'sec-websocket-protocol': 'mah.wrong'}
        url = SimpleNamespace(scheme='ws', netloc='localhost')
        async def close(self, code): closed.append(code)
        async def accept(self, **kwargs): raise AssertionError('Invalid token was accepted')
    asyncio.run(ws.metrics_ws(Socket(), 'some-run'))
    assert closed == [1008]


def test_chat_context_overflow_returns_client_error(client, monkeypatch):
    class Inputs(dict):
        def to(self, device): return self
    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return Inputs(input_ids=SimpleNamespace(shape=(1, 8)))
    model = SimpleNamespace(device='cpu', config=SimpleNamespace(model_type='homellm', max_position_embeddings=16))
    monkeypatch.setattr(chat, '_ensure_chat_backend', lambda req: {'model': model, 'tokenizer': Tokenizer()})
    response = client.post('/api/chat/generate', json={'messages': [], 'model_path': 'fake', 'max_tokens': 12, 'stream': False})
    assert response.status_code == 400 and 'context length' in response.text


def test_distributed_launch_matches_selected_gpus_and_precision(client, workspace, fake_worker):
    configs = workspace / 'configs'
    configs.mkdir()
    (configs / 'accelerate_multi_gpu.yaml').write_text('num_processes: 2\n')
    response = start(client, workspace, parallel_mode='multi_gpu', gpu_ids=[0, 2, 3], mixed_precision='no')
    assert response.status_code == 200, response.text
    command = fake_worker[0][1]
    assert command[command.index('--num_processes') + 1] == '3'
    assert command[command.index('--mixed_precision') + 1] == 'no'
    assert fake_worker[0][2]['env']['CUDA_VISIBLE_DEVICES'] == '0,2,3'
    assert start(client, workspace, parallel_mode='multi_gpu', gpu_ids=[0]).status_code == 400
    assert start(client, workspace, gpu_ids=[0, 0]).status_code == 400


def test_unsloth_does_not_offer_unsupported_resume_or_masking(client, workspace, fake_worker):
    model = workspace / 'base'
    model.mkdir()
    payload = {'stage': 'sft', 'training_backend': 'unsloth', 'model_path': str(model)}
    response = start(client, workspace, **payload)
    assert response.status_code == 200, response.text
    run_id = response.json()['run_id']
    assert client.post(f'/api/training/runs/{run_id}/continue').status_code == 400
    assert start(client, workspace, **payload, sft_packing=True).status_code == 400
    assert start(client, workspace, **payload, sft_train_on_completions_only=True).status_code == 400
    assert start(client, workspace, training_backend='unsloth').status_code == 400


def test_vlm_controls_are_preserved_and_paths_resolve(client, workspace, fake_worker, monkeypatch):
    monkeypatch.setattr(vlm, '_raise_if_exited_early', lambda *args: None)
    data = workspace / 'datasets' / 'vlm.jsonl'
    data.write_text('{}\n')
    model = workspace / 'models' / 'vision'
    model.mkdir()
    payload = {'stage': 'vlm_pretrain', 'model_path': 'models/vision', 'dataset_path': 'datasets/vlm.jsonl',
        'gpu_ids': [1], 'mixed_precision': 'no', 'use_lora': False, 'log_every': 1, 'save_every': 2,
        'gradient_checkpointing': False, 'lora_target_modules': ['q_proj'], 'max_pixels': 4096}
    response = client.post('/api/vlm/start', json=payload)
    assert response.status_code == 200, response.text
    config = json.loads((fake_worker[0][0] / 'config.json').read_text())
    assert config['base_model_path'] == str(model)
    assert config['data_base_dir'] == str(data.parent)
    assert config['assistant_only_loss'] is False and config['freeze_vision_tower'] is True
    assert config['gradient_checkpointing'] is False
    assert config['save_every'] == 2 and config['log_every'] == 1
    assert config['lora_target_modules'] == ['q_proj'] and config['max_pixels'] == 4096
    assert fake_worker[0][1][1].endswith('vlm_worker.py')
    assert fake_worker[0][2]['env']['CUDA_VISIBLE_DEVICES'] == '1'
    assert client.post('/api/vlm/start', json={**payload, 'gpu_ids': [0, 1]}).status_code == 400
    assert client.post('/api/vlm/start', json={**payload, 'min_pixels': 8192}).status_code == 400
