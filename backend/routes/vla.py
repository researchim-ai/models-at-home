"""Separate VLA experiments: demonstrations -> action policy -> prediction."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.paths import get_project_root, safe_id, atomic_json
from backend.services import processes
from backend.services.vlm_environment import python_command

router = APIRouter()
PROJECT_ROOT = get_project_root(Path(__file__).resolve().parents[2] / 'models-at-home')
RUNS_DIR = PROJECT_ROOT / '.runs'
WORKER = Path(__file__).resolve().parents[1] / 'services' / 'vla_worker.py'


class VLATrainingRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    stage: Literal['vla_sft'] = 'vla_sft'
    architecture: Literal['native_vla', 'vlm_action'] = 'native_vla'
    training_backend: Literal['transformers', 'unsloth'] = 'transformers'
    domain: Literal['robotics', 'simulation', 'desktop'] = 'robotics'
    action_type: Literal['continuous', 'discrete', 'hybrid'] = 'continuous'
    action_names: list[str] = []
    discrete_classes: list[int] = []
    discrete_names: list[str] = []
    discrete_labels: list[list[str]] = []
    model_path: str = ''
    dataset_path: str = ''
    val_data_path: str = ''
    experiment_name: str = ''
    action_dim: int = Field(default=7, ge=1, le=4096)
    action_horizon: int = Field(default=8, ge=1, le=256)
    state_dim: int = Field(default=7, ge=0, le=1024)
    hidden_dim: int = Field(default=128, ge=16, le=4096)
    image_size: int = Field(default=64, ge=16, le=512)
    vision_grid: int = Field(default=4, ge=1, le=8)
    max_instruction_bytes: int = Field(default=512, ge=1, le=8192)
    batch_size: int = Field(default=8, ge=1, le=128)
    gradient_accumulation: int = Field(default=1, ge=1, le=128)
    learning_rate: float = Field(default=0.0003, gt=0, le=1)
    max_steps: int = Field(default=500, ge=1)
    warmup_steps: int = Field(default=0, ge=0)
    save_every: int = Field(default=100, ge=1)
    eval_every: int = Field(default=20, ge=1)
    mixed_precision: Literal['no', 'bf16', 'fp16'] = 'bf16'
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    gpu_ids: list[int] = []
    output_dir: str = 'out'
    tuning_method: Literal['full', 'lora', 'qlora'] = 'full'
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.0, ge=0, lt=1)
    lora_target_modules: list[str] = []
    lora_modules_to_save: list[str] = []
    freeze_vision_tower: bool = True
    gradient_checkpointing: bool = True
    seq_len: int = Field(default=2048, ge=1)
    min_pixels: int | None = Field(default=None, ge=1)
    max_pixels: int | None = Field(default=None, ge=1)
    resume_from_checkpoint: str = ''
    resumed_from_run: str = ''

    @model_validator(mode='after')
    def coherent(self):
        if len(self.gpu_ids) > 1 or any(value < 0 for value in self.gpu_ids):
            raise ValueError('VLA supports one selected GPU')
        if self.architecture == 'native_vla' and self.vision_grid > self.image_size // 4:
            raise ValueError('vision_grid exceeds encoded image resolution')
        if self.architecture == 'native_vla' and self.training_backend != 'transformers':
            raise ValueError('Unsloth accelerates VLM backbones; native VLA uses our PyTorch model')
        if self.architecture == 'native_vla' and self.tuning_method != 'full':
            raise ValueError('Native VLA trains from scratch; LoRA/QLoRA require VLM action architecture')
        if self.architecture == 'vlm_action' and not self.model_path.strip():
            raise ValueError('Choose a VLM backbone model_path')
        if self.action_type == 'discrete' and self.action_dim < 2:
            raise ValueError('Discrete actions require at least two classes')
        if self.action_type == 'hybrid' and (not self.discrete_classes or len(self.discrete_classes) > 32 or any(not 2 <= n <= 4096 for n in self.discrete_classes)):
            raise ValueError('Hybrid policy needs 1..32 discrete_classes, each in 2..4096')
        if self.discrete_labels and (self.action_type != 'hybrid' or len(self.discrete_labels) != len(self.discrete_classes) or any(len(row) != size for row, size in zip(self.discrete_labels, self.discrete_classes))):
            raise ValueError('discrete_labels must match every hybrid category class count')
        for names in [self.action_names, self.discrete_names, *self.discrete_labels]:
            if any(not name.strip() for name in names) or len(names) != len(set(names)):
                raise ValueError('Action/control labels must be nonempty and unique')
        if self.hidden_dim * self.action_horizon * (self.action_dim + sum(self.discrete_classes)) > 100_000_000:
            raise ValueError('Action head exceeds 100M parameters; reduce dimensions, horizon or hidden_dim')
        if self.action_names and len(self.action_names) != self.action_dim:
            raise ValueError('action_names must match action_dim')
        if self.discrete_names and len(self.discrete_names) != len(self.discrete_classes):
            raise ValueError('discrete_names must match discrete_classes')
        if self.action_type != 'discrete' and self.action_dim > 256:
            raise ValueError('Continuous action vectors are limited to 256 coordinates')
        if self.max_pixels and self.min_pixels and self.min_pixels > self.max_pixels:
            raise ValueError('min_pixels exceeds max_pixels')
        return self


def existing(value, label, *, file=True):
    if not value:
        raise ValueError(f'{label} is required')
    path = Path(value).expanduser()
    path = path if path.is_absolute() else PROJECT_ROOT / path
    path = path.resolve()
    if not (path.is_file() if file else path.is_dir()):
        raise ValueError(f'{label} does not exist: {path}')
    return str(path)


def complete_checkpoint(directory):
    try:
        marker = json.loads((directory / 'complete.json').read_text())
        return (isinstance(marker, dict) and marker.get('format') == 'mah_vla_training_v1'
                and isinstance(marker.get('step'), int) and marker['step'] > 0
                and (directory / 'training_state.pt').stat().st_size > 0)
    except (OSError, ValueError):
        return False


def normalized(req):
    config = req.model_dump()
    config['data_path'] = config['dataset_path'] = existing(req.dataset_path, 'dataset_path')
    if req.val_data_path:
        config['val_data_path'] = existing(req.val_data_path, 'val_data_path')
        if config['val_data_path'] == config['data_path']:
            raise ValueError('Holdout must differ from training dataset')
    if req.architecture == 'vlm_action':
        local = Path(req.model_path).expanduser()
        if not local.is_absolute() and (PROJECT_ROOT / local).exists():
            local = PROJECT_ROOT / local
        if local.exists():
            config['model_path'] = str(local.resolve())
        config['base_model_path'] = config['model_path']
    if req.resume_from_checkpoint:
        directory = Path(existing(req.resume_from_checkpoint, 'checkpoint', file=False))
        if not complete_checkpoint(directory):
            raise ValueError('Resume needs a complete VLA optimizer checkpoint')
        config['resume_from_checkpoint'] = str(directory)
    config['use_flash_attention'] = False
    config['freeze_projector'] = False
    config['freeze_language_model'] = False
    return config


def env(gpu_ids):
    value = os.environ.copy()
    value.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    value['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0]) if gpu_ids else value.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    return value


def probe(mode, payload, gpu_ids):
    try:
        result = subprocess.run([python_command(), str(WORKER), '--mode', mode], input=json.dumps(payload),
            text=True, capture_output=True, cwd=PROJECT_ROOT, env=env(gpu_ids), timeout=180)
        if result.returncode:
            raise ValueError(result.stderr[-3000:] or f'VLA {mode} worker exited {result.returncode}')
        response = json.loads(result.stdout)
        if not isinstance(response, dict) or (mode == 'preflight' and not isinstance(response.get('ready'), bool)):
            raise ValueError('Invalid VLA worker response')
        return response
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f'VLA {mode} exceeded 180 seconds') from exc


@router.get('/examples')
def examples():
    root = Path(__file__).resolve().parents[2] / 'examples' / 'vla'
    return {'examples': [{'domain': domain, 'dataset_path': str(root / domain / 'train.jsonl'),
                           'val_data_path': str(root / domain / 'validation.jsonl'), 'synthetic': True}
                         for domain in ('robotics', 'simulation', 'desktop')
                         if (root / domain / 'train.jsonl').is_file()]}


@router.post('/preflight')
def preflight(req: VLATrainingRequest):
    try:
        return probe('preflight', normalized(req), req.gpu_ids)
    except (OSError, ValueError) as exc:
        return {'ready': False, 'checks': [{'name': 'VLA preflight', 'status': 'error', 'detail': str(exc)}], 'sample': None}


@router.post('/start')
def start(req: VLATrainingRequest):
    try:
        config = normalized(req)
        python = python_command()
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    run_id = 'vla_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    run = safe_id(RUNS_DIR, run_id)
    output = Path(req.output_dir).expanduser()
    output = output if output.is_absolute() else PROJECT_ROOT / output
    config.update(run_id=run_id, model_type='vla', started_at=datetime.now().isoformat(),
                  output_dir=str((output / run_id).resolve()), vla_worker_version=1)
    run.mkdir(parents=True, exist_ok=False)
    Path(config['output_dir']).mkdir(parents=True, exist_ok=False)
    atomic_json(run / 'config.json', config)
    atomic_json(run / 'metrics.json', {'status': 'initializing', 'current_step': 0})
    processes.launch(run, [python, str(WORKER), '--mode', 'train', '--config', str(run / 'config.json'),
                          '--metrics', str(run / 'metrics.json')], cwd=PROJECT_ROOT, env=env(req.gpu_ids))
    return {'run_id': run_id}


@router.get('/runs')
def runs():
    from backend.routes.training import list_runs
    return {'runs': [row for row in list_runs()['runs'] if row['stage'] == 'vla_sft']}


@router.post('/stop/{run_id}')
def stop(run_id: str):
    run = safe_id(RUNS_DIR, run_id)
    if not (run / 'config.json').is_file():
        raise HTTPException(404, 'Run not found')
    if json.loads((run / 'config.json').read_text()).get('stage') != 'vla_sft':
        raise HTTPException(400, 'Not a VLA run')
    processes.stop(run)
    return {'success': True}


class PredictRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    model_path: str
    instruction: str = Field(min_length=1, max_length=8192)
    state: list[float] = Field(default_factory=list, max_length=1024)
    image_base64: str = Field(max_length=20_000_000)
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    gpu_ids: list[int] = Field(default_factory=list, max_length=1)


@router.post('/predict')
def prediction(req: PredictRequest):
    try:
        payload = req.model_dump()
        payload['model_path'] = existing(req.model_path, 'policy', file=False)
        if any(value < 0 for value in req.gpu_ids):
            raise ValueError('GPU id must be nonnegative')
        return probe('predict', payload, req.gpu_ids)
    except (OSError, ValueError) as exc:
        raise HTTPException(400, str(exc))
