"""World model jobs and bounded, non-executing imagined trajectories."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
from typing import Literal
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.paths import get_project_root, safe_id, atomic_json
from backend.services import processes
from backend.services.vlm_environment import python_command

router = APIRouter()
PROJECT_ROOT = get_project_root(Path(__file__).resolve().parents[2] / 'models-at-home')
RUNS_DIR = PROJECT_ROOT / '.runs'
WORKER = Path(__file__).resolve().parents[1] / 'services' / 'world_worker.py'


class WorldRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    stage: Literal['world_model'] = 'world_model'
    architecture: Literal['latent_gru', 'rssm'] = 'latent_gru'
    dataset_path: str = ''
    val_data_path: str = ''
    experiment_name: str = ''
    action_type: Literal['continuous', 'discrete', 'hybrid'] = 'continuous'
    action_dim: int = Field(default=2, ge=1, le=256)
    discrete_classes: list[int] = Field(default_factory=list, max_length=16)
    state_dim: int = Field(default=0, ge=0, le=256)
    image_size: Literal[32, 64, 128] = 32
    sequence_length: int = Field(default=4, ge=1, le=64)
    latent_dim: int = Field(default=32, ge=4, le=256)
    hidden_dim: int = Field(default=128, ge=16, le=512)
    batch_size: int = Field(default=8, ge=1, le=64)
    learning_rate: float = Field(default=.001, gt=0, le=.1)
    max_steps: int = Field(default=500, ge=1, le=10000000)
    save_every: int = Field(default=100, ge=1)
    eval_every: int = Field(default=20, ge=1)
    kl_weight: float = Field(default=.001, ge=0, le=1)
    free_nats: float = Field(default=1., ge=0, le=100)
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    mixed_precision: Literal['no', 'bf16', 'fp16'] = 'no'
    gpu_ids: list[int] = Field(default_factory=list, max_length=1)
    output_dir: str = 'out'
    resume_from_checkpoint: str = ''
    resumed_from_run: str = ''

    @model_validator(mode='after')
    def coherent(self):
        if any(n < 0 for n in self.gpu_ids):
            raise ValueError('GPU id must be nonnegative')
        if self.action_type == 'discrete' and self.action_dim < 2:
            raise ValueError('Discrete actions need at least two classes')
        if self.action_type == 'hybrid' and (not self.discrete_classes or any(not 2 <= n <= 256 for n in self.discrete_classes)):
            raise ValueError('Hybrid actions need discrete_classes in 2..256')
        if self.action_type != 'hybrid' and self.discrete_classes:
            raise ValueError('discrete_classes apply only to hybrid actions')
        if self.batch_size * self.sequence_length * self.image_size ** 2 > 16_777_216:
            raise ValueError('Reduce batch size, sequence length or image size')
        return self


def existing(value, directory=False):
    p = Path(value).expanduser()
    p = (p if p.is_absolute() else PROJECT_ROOT / p).resolve()
    if not value or not (p.is_dir() if directory else p.is_file()):
        raise ValueError(f'Path does not exist: {value}')
    return str(p)


def complete_checkpoint(p):
    try:
        m = json.loads((p / 'complete.json').read_text())
        return m.get('format') == 'mah_world_training_v1' and type(m.get('step')) is int and m['step'] > 0 and all((p / n).stat().st_size > 0 for n in ('training_state.pt', 'model.safetensors', 'world.json'))
    except (OSError, ValueError, AttributeError):
        return False


def normalized(req):
    c = req.model_dump()
    c['dataset_path'] = existing(req.dataset_path)
    if req.val_data_path:
        c['val_data_path'] = existing(req.val_data_path)
        if c['val_data_path'] == c['dataset_path']:
            raise ValueError('Holdout must differ from training data')
    if req.resume_from_checkpoint:
        c['resume_from_checkpoint'] = existing(req.resume_from_checkpoint, True)
        if not complete_checkpoint(Path(c['resume_from_checkpoint'])):
            raise ValueError('Incomplete or incompatible world model checkpoint')
    return c


def env(gpus):
    result = os.environ.copy()
    result['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    result['CUDA_VISIBLE_DEVICES'] = str(gpus[0]) if gpus else result.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    return result


def probe(mode, payload, gpus):
    try:
        result = subprocess.run([python_command(), str(WORKER), '--mode', mode], input=json.dumps(payload), text=True,
            capture_output=True, timeout=180, cwd=PROJECT_ROOT, env=env(gpus))
        if result.returncode:
            raise ValueError(result.stderr[-3000:] or f'Worker exited {result.returncode}')
        response = json.loads(result.stdout)
        if not isinstance(response, dict):
            raise ValueError('Invalid world model worker response')
        return response
    except subprocess.TimeoutExpired as exc:
        raise ValueError('World model operation exceeded 180 seconds') from exc


@router.get('/examples')
def examples():
    root = Path(__file__).resolve().parents[2] / 'examples' / 'world-models'
    return {'dataset_path': str(root / 'train.jsonl'), 'val_data_path': str(root / 'validation.jsonl')}


@router.post('/preflight')
def preflight(req: WorldRequest):
    try:
        c=normalized(req)
        report=probe('preflight',c,req.gpu_ids)
        if report.get('ready') and report.get('data_summary'):
            from backend.services.world_resources import remember_data
            remember_data(c,report['data_summary'])
        return report
    except (ValueError, OSError) as exc:
        return {'ready': False, 'checks': [{'name': 'World Models', 'status': 'error', 'detail': str(exc)}]}


@router.post('/start')
def start(req: WorldRequest):
    try:
        c = normalized(req)
        from backend.services.world_resources import enforce
        enforce(req.architecture,'train',c,PROJECT_ROOT)
        python = python_command()
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    run_id = 'world_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    run = safe_id(RUNS_DIR, run_id)
    output = Path(req.output_dir).expanduser()
    output = output if output.is_absolute() else PROJECT_ROOT / output
    c.update(run_id=run_id, model_type='world_model', world_worker_version=1, started_at=datetime.now().isoformat(), output_dir=str((output / run_id).resolve()))
    run.mkdir(parents=True, exist_ok=False)
    Path(c['output_dir']).mkdir(parents=True, exist_ok=False)
    atomic_json(run / 'config.json', c)
    atomic_json(run / 'metrics.json', {'status': 'initializing', 'current_step': 0})
    try:
        processes.launch(run, [python, str(WORKER), '--mode', 'train', '--config', str(run / 'config.json'), '--metrics', str(run / 'metrics.json')], cwd=PROJECT_ROOT, env=env(req.gpu_ids))
    except Exception as exc:
        atomic_json(run / 'metrics.json', {'status': 'error', 'current_step': 0, 'error': str(exc)})
        raise HTTPException(500, f'Could not launch world model worker: {exc}') from exc
    return {'run_id': run_id}


@router.get('/runs')
def runs():
    from backend.routes.training import list_runs
    return {'runs': [r for r in list_runs()['runs'] if r['stage'] == 'world_model']}


@router.post('/stop/{run_id}')
def stop(run_id: str):
    p = safe_id(RUNS_DIR, run_id)
    if not (p / 'config.json').exists():
        raise HTTPException(404, 'Run not found')
    if json.loads((p / 'config.json').read_text()).get('stage') != 'world_model':
        raise HTTPException(400, 'Not a World Model run')
    processes.stop(p)
    return {'success': True}


class RolloutRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    model_path: str
    image_base64: str = Field(max_length=20_000_000)
    actions: list = Field(min_length=1, max_length=64)
    state: list[float] = Field(default_factory=list, max_length=256)
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    gpu_ids: list[int] = Field(default_factory=list, max_length=1)


@router.post('/rollout')
def rollout(req: RolloutRequest):
    try:
        if any(n < 0 for n in req.gpu_ids):
            raise ValueError('GPU id must be nonnegative')
        payload = req.model_dump()
        payload['model_path'] = existing(req.model_path, True)
        return probe('rollout', payload, req.gpu_ids)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, str(exc))
