"""Real embedded pretrained inference with outbound sockets disabled during all model operations."""
import argparse
import base64
import json
import os
from pathlib import Path
import socket
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser();p.add_argument('--model',choices=['diamond_csgo','diamond_atari'],default='diamond_csgo');p.add_argument('--snapshot',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--steps',type=int,default=6);p.add_argument('--result',required=True);p.add_argument('--upsample',action='store_true');args=p.parse_args()
os.environ['HF_HUB_OFFLINE']='1';os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']='0'
from backend.services.world_local import LocalWorld
import torch
c=dict(model_id=args.model,game='Breakout',model_path=args.snapshot,device=args.device,denoising_steps=1 if args.model=='diamond_csgo' else 3,upsample=args.upsample,spawn=0,seed=42,image_base64='')
original_connect=socket.socket.connect
calls=[]
def blocked(self,*a,**k):calls.append(True);raise RuntimeError('Network forbidden during local world inference')
socket.socket.connect=blocked
try:
 t=time.perf_counter();w=LocalWorld(c);load_seconds=time.perf_counter()-t
 first=w.reset();frames=[first['frame']];times=[]
 for i in range(args.steps):
  r=w.step(dict(keys=['w'] if i%2==0 else ['look_left'],mouse_x=0,mouse_y=0,action=1 if i%2==0 else 2));frames.append(r['frame']);times.append(r['seconds'])
 w.reset();a=w.step(dict(keys=['w'],mouse_x=0,mouse_y=0,action=2))
 w.reset();repeat=w.step(dict(keys=['w'],mouse_x=0,mouse_y=0,action=2))
 assert a['frame']==repeat['frame'],'Reset/replay is not deterministic'
 w.reset();other=w.step(dict(keys=['d'],mouse_x=0,mouse_y=0,action=3))
 assert a['frame']!=other['frame'],'Changed actions did not influence predicted frame'
 assert not calls
 result=dict(model_id=args.model,pretrained=True,embedded_architecture=True,device=args.device,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
  parameters=sum(p.numel() for p in w.model.parameters())+ (sum(p.numel() for p in w.upsampler.parameters()) if w.upsampler else 0),steps=args.steps,
  load_seconds=load_seconds,seconds_per_step=times,reset_replay_equal=True,action_influence=True,network_attempts=len(calls),upsampler=args.upsample,
  resolution=first['resolution'],unique_frames=len(set(frames)),peak_vram_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=='cuda' else 0)
 Path(args.result).write_text(json.dumps(result,indent=2)+'\n')
 Path(args.result).with_suffix('.png').write_bytes(base64.b64decode(frames[-1].split(',')[-1]))
 print(json.dumps(result))
finally:socket.socket.connect=original_connect
