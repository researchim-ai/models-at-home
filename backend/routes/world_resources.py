"""Capacity hints for world models; real native training calibration on demand."""
import json
from pathlib import Path
import subprocess
from typing import Literal
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field
from backend.paths import get_project_root
from backend.routes.world_models import WorldRequest,env
from backend.routes.world_foundation import python_command
from backend.services import world_resources as resources

router=APIRouter()
PROJECT_ROOT=get_project_root(Path(__file__).resolve().parents[2]/'models-at-home')
WORKER=Path(__file__).resolve().parents[1]/'services/world_resource_worker.py'

class ResourceRequest(BaseModel):
    model_id:Literal['latent_gru','rssm','diamond_csgo','diamond_atari','cosmos3_edge','cosmos3_nano','cosmos3_super']
    operation:Literal['train','inference']='train'
    config:dict=Field(default_factory=dict)


def config(req):
    if req.model_id in ('latent_gru','rssm'):
        try:c=WorldRequest(**{**req.config,'architecture':req.model_id}).model_dump()
        except ValueError as exc:raise HTTPException(422,str(exc))
        for field in ('dataset_path','val_data_path','resume_from_checkpoint','output_dir'):
            if c.get(field):
                p=Path(c[field]).expanduser();c[field]=str((p if p.is_absolute() else PROJECT_ROOT/p).resolve())
        return c
    # Validate dimensions/precision without requiring images, prompts or model downloads for planning.
    from backend.routes.world_play import StartRequest
    if req.model_id.startswith('diamond'):
        try:
            c=StartRequest(**{**req.config,'model_id':req.model_id}).model_dump()
            if c.get('model_path'):
                path=Path(c['model_path']).expanduser();c['model_path']=str(path if path.is_absolute() else PROJECT_ROOT/path)
            return c
        except ValueError as exc:raise HTTPException(422,str(exc))
    from backend.routes.world_foundation import FoundationRequest
    try:
        c=FoundationRequest(**{**req.config,'model_id':req.model_id,'task':'text2video'}).model_dump()
        if c.get('model_path'):
            path=Path(c['model_path']).expanduser();c['model_path']=str(path if path.is_absolute() else PROJECT_ROOT/path)
        if req.config.get('task')=='forward_dynamics' and c['actions']:c['num_frames']=len(c['actions'])+1
        return c
    except ValueError as exc:raise HTTPException(422,str(exc))


@router.post('/estimate')
def estimate(req:ResourceRequest):
    c=config(req)
    return resources.estimate(req.model_id,req.operation,c,resources.hardware(c.get('output_dir') or PROJECT_ROOT))


def probe(c):
    try:
        result=subprocess.run([python_command(),str(WORKER)],input=json.dumps(c),text=True,capture_output=True,
            cwd=PROJECT_ROOT,env=env(c['gpu_ids']),timeout=150)
        if result.returncode:raise ValueError(result.stderr[-2500:] or f'Probe exited {result.returncode}')
        report=json.loads(result.stdout)
        if not isinstance(report,dict) or report.get('status') not in ('ok','oom','error'):raise ValueError('Invalid resource probe result')
        return report
    except subprocess.TimeoutExpired as exc:raise ValueError('Проба не завершилась за 150 секунд; уменьшите batch/окно/разрешение') from exc


@router.post('/benchmark')
def benchmark(req:ResourceRequest):
    c=config(req)
    if req.model_id not in ('latent_gru','rssm') or req.operation!='train':
        raise HTTPException(400,'Тюнинг этой pretrained-модели не подключён; тренировочный benchmark недоступен')
    if not resources.benchmark_lock.acquire(blocking=False):raise HTTPException(409,'Другая ресурсная проба ещё выполняется')
    try:
        hw=resources.hardware(c['output_dir']);raw=resources.estimate(req.model_id,req.operation,c,hw,use_measurement=False)
        import tempfile
        free=resources._disk(tempfile.gettempdir())
        if free is not None and free<max(.25,raw['disk_required_gib']/max(1,c['max_steps']/c['save_every'])):raise HTTPException(400,'Недостаточно места во временном каталоге для пробы')
        if raw['reasons']:raise HTTPException(400,'; '.join(raw['reasons']))
        if raw['ram_gib'][1]>hw['ram_available_gib']*.7:raise HTTPException(400,'Слишком мал запас RAM для безопасной короткой пробы; уменьшите настройки')
        if raw['device'] in ('cuda','cpu'):c['device']=raw['device']
        try:measured=probe(c)
        except (OSError,ValueError) as exc:raise HTTPException(400,str(exc))
        resources.store_measurement(raw['profile_key'],measured)
    finally:resources.benchmark_lock.release()
    return resources.estimate(req.model_id,req.operation,c,resources.hardware(c['output_dir']))
