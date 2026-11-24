"""Embedded playable world models. Downloads are weights only; all inference is local."""
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from backend.paths import get_project_root
from backend.routes.world_foundation import python_command
from backend.routes.world_models import env
from backend.services import world_play as service

router=APIRouter()
PROJECT_ROOT=get_project_root(Path(__file__).resolve().parents[2]/'models-at-home')

class StartRequest(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    model_id:Literal['diamond_csgo','diamond_atari']='diamond_csgo'
    game:Literal['Breakout','Pong']='Breakout'
    model_path:str=''
    image_base64:str=Field(default='',max_length=20_000_000)
    device:Literal['auto','cpu','cuda']='auto'
    gpu_ids:list[int]=Field(default_factory=list,max_length=1)
    denoising_steps:int=Field(default=3,ge=1,le=10)
    upsample:bool=True
    spawn:int=Field(default=0,ge=0,le=6)
    seed:int=Field(default=42,ge=0,le=2**32-1)

@router.post('/start')
def start(req:StartRequest):
    if any(n<0 for n in req.gpu_ids):raise HTTPException(400,'GPU id must be nonnegative')
    c=req.model_dump()
    if req.model_path:
        p=Path(req.model_path).expanduser();p=(p if p.is_absolute() else PROJECT_ROOT/p).resolve()
        if not p.is_dir():raise HTTPException(400,'Local weights directory does not exist')
        c['model_path']=str(p)
    if req.model_id=='diamond_atari' and req.game=='Pong' and not req.image_base64:raise HTTPException(400,'Upload an Atari initial screenshot')
    try:
        from backend.services.world_resources import enforce
        enforce(req.model_id,'inference',c,PROJECT_ROOT)
        return service.start(python_command(),PROJECT_ROOT,env(req.gpu_ids),c)
    except ValueError as exc:raise HTTPException(400,str(exc))

class SessionRequest(BaseModel):
    session_id:str=Field(min_length=32,max_length=32,pattern='^[a-f0-9]+$')

class StepRequest(SessionRequest):
    model_config=ConfigDict(allow_inf_nan=False)
    keys:list[Literal['w','a','s','d','space','ctrl','shift','1','2','3','r','attack','aim','look_left','look_right','look_up','look_down']]=Field(default_factory=list,max_length=17)
    mouse_x:float=Field(default=0,ge=-1000,le=1000)
    mouse_y:float=Field(default=0,ge=-200,le=200)
    action:int=Field(default=0,ge=0,le=17)

class ResetRequest(SessionRequest):
    spawn:int|None=Field(default=None,ge=0,le=6)

@router.post('/step')
def step(req:StepRequest):return service.request(req.session_id,'step',req.model_dump(exclude={'session_id'}))

@router.post('/reset')
def reset(req:ResetRequest):return service.request(req.session_id,'reset',req.model_dump(exclude={'session_id'}))

@router.post('/stop')
def stop(req:SessionRequest):
    service.close(req.session_id);return {'success':True}

@router.get('/status')
def status():return service.status()
