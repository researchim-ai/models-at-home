"""Pretrained world models: bounded managed generation/download jobs, not scratch training."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import uuid
from typing import Literal
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.paths import get_project_root, safe_id, atomic_json
from backend.services import processes
from backend.services.world_catalog import MODELS, DOMAINS, model_info
from backend.services.vlm_environment import python_command as default_python
from backend.routes.world_models import env

router = APIRouter()
PROJECT_ROOT = get_project_root(Path(__file__).resolve().parents[2] / 'models-at-home')
RUNS_DIR = PROJECT_ROOT / '.runs'
WORKER = Path(__file__).resolve().parents[1] / 'services' / 'world_foundation_worker.py'


def python_command():
    value = os.environ.get('MAH_WORLD_PYTHON')
    if not value:
        suffix = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
        for root in (PROJECT_ROOT / '.venv-world', PROJECT_ROOT / 'models' / '.world-venv'):
            candidate = root / suffix
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.absolute())
        return default_python()
    p = Path(value).expanduser().absolute()
    if not p.is_file() or not os.access(p, os.X_OK):
        raise ValueError('MAH_WORLD_PYTHON must be an executable Python')
    return str(p)


class FoundationRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    model_id: Literal['cosmos3_edge', 'cosmos3_nano', 'cosmos3_super', 'diamond_csgo', 'diamond_atari'] = 'cosmos3_edge'
    task: Literal['text2video', 'image2video', 'forward_dynamics', 'local_play'] = 'text2video'
    model_path: str = ''
    revision: str = ''
    prompt: str = Field(default='', max_length=16000)
    negative_prompt: str = Field(default='', max_length=16000)
    image_base64: str = Field(default='', max_length=20_000_000)
    num_frames: int = Field(default=49, ge=1, le=241)
    height: Literal[256, 480, 720] = 256
    width: int = Field(default=448, ge=256, le=1280)
    fps: int = Field(default=24, ge=10, le=30)
    inference_steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=6., ge=0, le=20)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    cpu_offload: bool = True
    domain: str = 'umi'
    actions: list[list[float]] = Field(default_factory=list, max_length=240)
    game: Literal['Breakout', 'Pong'] = 'Breakout'
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    upsample: bool = True
    spawn: int = Field(default=0, ge=0, le=6)
    keys: list[Literal['w', 'a', 's', 'd', 'space', 'attack', 'aim', 'look_left', 'look_right', 'look_up', 'look_down']] = Field(default_factory=list, max_length=8)
    gpu_ids: list[int] = Field(default_factory=list, max_length=1)
    @model_validator(mode='after')
    def coherent(self):
        info = model_info(self.model_id)
        if self.task not in info['tasks']:
            raise ValueError('Selected model does not support this task')
        if any(n < 0 for n in self.gpu_ids):
            raise ValueError('GPU id must be nonnegative')
        if self.model_id.startswith('cosmos3'):
            if self.height % 16 or self.width % 16:
                raise ValueError('Cosmos dimensions must be multiples of 16')
            if (self.num_frames - 1) % 4:
                raise ValueError('Cosmos num_frames must be 4k+1')
            if self.task == 'forward_dynamics':
                if self.domain not in DOMAINS:
                    raise ValueError('Unsupported Cosmos embodiment')
                if not self.actions or any(len(row) != DOMAINS[self.domain] for row in self.actions):
                    raise ValueError(f'{self.domain} requires T × {DOMAINS[self.domain]} raw action coordinates')
                if self.height not in (256, 480):
                    raise ValueError('Action-conditioned Cosmos supports resolution tiers 256/480')
                if len(self.actions) % 4:
                    raise ValueError('Cosmos action chunk must have 4k transitions; no silent action padding')
        return self


def local(value, directory=False):
    if not value:
        raise ValueError('Local path is required')
    p = Path(value).expanduser()
    p = (p if p.is_absolute() else PROJECT_ROOT / p).resolve()
    if not (p.is_dir() if directory else p.is_file()):
        raise ValueError(f'Path does not exist: {p}')
    return str(p)


def normalized(req, operation='generate'):
    c = req.model_dump()
    info = model_info(req.model_id)
    c['revision'] = req.revision or info['revision']
    if info['provider'] == 'cosmos3':
        if req.model_path and operation != 'download':
            c['model_path'] = local(req.model_path, True)
        if operation != 'download' and (not req.prompt.strip() or (req.task != 'text2video' and not req.image_base64)):
            raise ValueError('Prompt and a conditioning image for image/action tasks are required')
    elif info['provider'] == 'diamond':
        if req.model_path and operation != 'download':
            c['model_path'] = local(req.model_path, True)
        if operation != 'download' and req.model_id == 'diamond_atari' and req.game == 'Pong' and not req.image_base64:
            raise ValueError('Atari requires an initial screenshot')
    return c


@router.get('/models')
def catalog():
    return {'models': MODELS, 'domains': DOMAINS, 'checked_at': '2026-11-23'}


@router.post('/preflight')
def preflight(req: FoundationRequest):
    try:
        c = normalized(req)
        result = subprocess.run([python_command(), str(WORKER), '--mode', 'preflight'], input=json.dumps(c),
            text=True, capture_output=True, timeout=90, cwd=PROJECT_ROOT, env=env(req.gpu_ids))
        if result.returncode:
            raise ValueError(result.stderr[-3000:] or 'Foundation preflight failed')
        response = json.loads(result.stdout)
        if not isinstance(response, dict) or not isinstance(response.get('ready'), bool):
            raise ValueError('Invalid foundation worker response')
        return response
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {'ready': False, 'checks': [{'name': 'Foundation model', 'status': 'error', 'detail': str(exc)}]}


def launch(req, operation):
    try:
        c = normalized(req, operation)
        if operation == 'generate':
            from backend.services.world_resources import enforce
            enforce(req.model_id,'inference',c,PROJECT_ROOT)
        python = python_command()
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    run_id = f'world_{operation}_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    run = safe_id(RUNS_DIR, run_id)
    c.update(stage='world_foundation', operation=operation, model_type=req.model_id, run_id=run_id,
        started_at=datetime.now().isoformat(), output_dir=str((PROJECT_ROOT / 'out' / run_id).resolve()), max_steps=req.inference_steps)
    run.mkdir(parents=True, exist_ok=False)
    output = Path(c['output_dir']); output.mkdir(parents=True, exist_ok=False)
    # Avoid embedding multi-MB images in the persistent run list response.
    seed_image = c.pop('image_base64', '')
    if seed_image:
        (run / 'input-image.txt').write_text(seed_image)
        c['image_file'] = str(run / 'input-image.txt')
    atomic_json(run / 'config.json', c)
    atomic_json(run / 'metrics.json', {'status': 'initializing', 'current_step': 0, 'operation': operation})
    try:
        processes.launch(run, [python, str(WORKER), '--mode', operation, '--config', str(run / 'config.json'), '--metrics', str(run / 'metrics.json')], cwd=PROJECT_ROOT, env=env(req.gpu_ids))
    except Exception as exc:
        atomic_json(run / 'metrics.json', {'status': 'error', 'error': str(exc)})
        raise HTTPException(500, 'Foundation worker failed to launch') from exc
    return {'run_id': run_id}


@router.post('/generate')
def generate(req: FoundationRequest):
    return launch(req, 'generate')


@router.post('/download')
def download(req: FoundationRequest):
    return launch(req, 'download')


@router.get('/jobs')
def jobs():
    from backend.routes.training import list_runs
    return {'runs': [r for r in list_runs()['runs'] if r['stage'] == 'world_foundation']}


def job(run_id):
    path = safe_id(RUNS_DIR, run_id)
    try:
        c = json.loads((path / 'config.json').read_text())
    except (OSError, ValueError):
        raise HTTPException(404, 'Job not found')
    if c.get('stage') != 'world_foundation':
        raise HTTPException(400, 'Not a foundation job')
    return path, c


@router.post('/stop/{run_id}')
def stop(run_id: str):
    path, c = job(run_id)
    processes.stop(path)
    return {'success': True}


@router.get('/jobs/{run_id}/result')
def result(run_id: str):
    _, c = job(run_id)
    path = Path(c['output_dir']) / 'result.json'
    if not path.is_file():
        raise HTTPException(404, 'Result is not ready')
    return json.loads(path.read_text())


@router.get('/jobs/{run_id}/video')
def video(run_id: str):
    _, c = job(run_id)
    path = Path(c['output_dir']) / 'generated.mp4'
    if not path.is_file():
        raise HTTPException(404, 'Video is not ready')
    return FileResponse(path, media_type='video/mp4', filename=f'{run_id}.mp4')
