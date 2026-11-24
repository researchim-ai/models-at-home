"""In-process architectures for locally playable pretrained DIAMOND worlds."""
import base64
from io import BytesIO
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image, ImageOps
from backend.vendor.diamond.atari.denoiser import Denoiser as AtariDenoiser, DenoiserConfig as AtariConfig
from backend.vendor.diamond.atari.inner_model import InnerModelConfig as AtariInner
from backend.vendor.diamond.atari.diffusion_sampler import DiffusionSampler as AtariSampler, DiffusionSamplerConfig as AtariSampling
from backend.vendor.diamond.csgo.denoiser import Denoiser as CSGODenoiser, DenoiserConfig as CSGOConfig
from backend.vendor.diamond.csgo.inner_model import InnerModelConfig as CSGOInner
from backend.vendor.diamond.csgo.diffusion_sampler import DiffusionSampler as CSGOSampler, DiffusionSamplerConfig as CSGOSampling

REPO = 'eloialonso/diamond'
REVISION = '5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f'
MOUSE_X = [-1000,-500,-300,-200,-100,-60,-30,-20,-10,-4,-2,0,2,4,10,20,30,60,100,200,300,500,1000]
MOUSE_Y = [-200,-100,-50,-20,-10,-4,-2,0,2,4,10,20,50,100,200]
KEYS = ['w','a','s','d','space','ctrl','shift','1','2','3','r']


def download(model_id, game='Breakout', local_dir=None, offline=False):
    from huggingface_hub import snapshot_download
    patterns = ['csgo/*'] if model_id == 'diamond_csgo' else [f'atari_100k/models/{game}.pt']
    return snapshot_download(REPO, revision=REVISION, allow_patterns=patterns, local_dir=local_dir, local_files_only=offline)


def encode_action(keys, mouse_x=0, mouse_y=0):
    # Same 11 keys + 2 clicks + 23/15 mouse bins as upstream CS:GO encoder.
    chosen = set(keys)
    for pair in [('w','s'),('a','d')]:
        if set(pair) <= chosen: chosen.difference_update(pair)
    x = min(MOUSE_X,key=lambda v:abs(v-mouse_x)); y = min(MOUSE_Y,key=lambda v:abs(v-mouse_y))
    out = torch.zeros(51)
    for i,k in enumerate(KEYS): out[i] = k in chosen
    out[11],out[12] = 'attack' in chosen,'aim' in chosen
    out[13+MOUSE_X.index(x)] = 1; out[36+MOUSE_Y.index(y)] = 1
    return out


def image_uri(frame):
    if isinstance(frame,torch.Tensor):
        frame = (frame.detach().float().cpu().clamp(-1,1).add(1).mul(127.5)).to(torch.uint8).permute(1,2,0).numpy()
    image = Image.fromarray(frame.astype('uint8')).convert('RGB')
    buf=BytesIO();image.save(buf,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()


class LocalWorld:
    def __init__(self,c):
        torch.set_num_threads(min(4,torch.get_num_threads()))
        self.config=c
        if c['device']=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable')
        self.device=torch.device('cuda' if c['device']!='cpu' and torch.cuda.is_available() else 'cpu')
        self.is_csgo=c['model_id']=='diamond_csgo'
        self.root=Path(c['model_path']) if c.get('model_path') else Path(download(c['model_id'],c['game'],offline=True))
        path=self.root / ('csgo/model/csgo.pt' if self.is_csgo else f"atari_100k/models/{c['game']}.pt")
        # Official checkpoint contains tensors only; never deserialize arbitrary Python objects.
        sd=torch.load(path,map_location='cpu',weights_only=True)
        def subset(prefix):return {k[len(prefix)+1:]:v for k,v in sd.items() if k.startswith(prefix+'.')}
        weights=subset('denoiser')
        self.upsampler=None
        if self.is_csgo:
            self.model=CSGODenoiser(CSGOConfig(CSGOInner(3,4,2048,[2]*4,[128,256,512,1024],[0,0,1,1],51),.5,.1,True))
            self.model.load_state_dict(weights,strict=True);self.model.to(self.device).eval()
            self.sampler=CSGOSampler(self.model,CSGOSampling(c['denoising_steps'],s_cond=.005))
            if c['upsample']:
                self.upsampler=CSGODenoiser(CSGOConfig(CSGOInner(3,1,2048,[2]*4,[64,64,128,256],[0,0,0,1],51),.5,.1,False,5))
                self.upsampler.load_state_dict(subset('upsampler'),strict=True);self.upsampler.to(self.device).eval()
                self.up_sampler=CSGOSampler(self.upsampler,CSGOSampling(1,sigma_min=1,order=2,s_churn=10,s_tmin=1,s_tmax=5,s_noise=.9))
            self.action_names=KEYS+['attack','aim','look_left','look_right','look_up','look_down']
        else:
            actions=weights['inner_model.act_emb.0.weight'].shape[0]
            self.model=AtariDenoiser(AtariConfig(AtariInner(3,4,256,[2]*4,[64]*4,[0]*4,actions),.5,.3))
            self.model.load_state_dict(weights,strict=True);self.model.to(self.device).eval()
            self.sampler=AtariSampler(self.model,AtariSampling(c['denoising_steps']))
            self.action_names=['NOOP','FIRE','RIGHT','LEFT','RIGHTFIRE','LEFTFIRE'][:actions]
        del sd,weights
        self.reset()

    def reset(self,spawn=None):
        torch.manual_seed(self.config['seed'])
        self.step_number=0
        if self.is_csgo:
            spawn=self.config['spawn'] if spawn is None else spawn
            root=self.root/'csgo/spawn'/str(spawn)
            low=np.load(root/'low_res.npy',allow_pickle=False);full=np.load(root/'full_res.npy',allow_pickle=False)
            act=np.load(root/'act.npy',allow_pickle=False)
            self.obs=torch.tensor(low,device=self.device).float().div(255).mul(2).sub(1).unsqueeze(0)
            self.full=torch.tensor(full,device=self.device).float().div(255).mul(2).sub(1).unsqueeze(0)
            self.act=torch.tensor(act,device=self.device).long().unsqueeze(0)
            self.initial_obs=self.obs.clone();self.initial_full=self.full.clone();self.initial_act=self.act.clone()
            frame=self.full[0,-1] if self.upsampler else self.obs[0,-1]
        else:
            raw=self.config.get('image_base64','')
            if not raw and self.config['game']=='Breakout':
                raw=base64.b64encode((Path(__file__).resolve().parents[2]/'examples/world-local/breakout.png').read_bytes()).decode()
            if not raw: raise ValueError('Pong needs an initial game screenshot; no ROM/emulator is required')
            with Image.open(BytesIO(base64.b64decode(raw.split(',')[-1],validate=True))) as im:
                if im.width*im.height>40_000_000:raise ValueError('Seed image exceeds 40 MP')
                im=ImageOps.exif_transpose(im).convert('RGB').resize((64,64))
                x=torch.tensor(np.array(im).copy()).permute(2,0,1).float().div(255).mul(2).sub(1)
            self.obs=x[None,None].repeat(1,4,1,1,1).to(self.device);self.act=torch.zeros(1,4,dtype=torch.long,device=self.device)
            frame=self.obs[0,-1]
        return dict(frame=image_uri(frame),step=0,action_names=self.action_names,device=str(self.device),resolution=list(frame.shape[-2:]),model_id=self.config['model_id'])

    @torch.inference_mode()
    def step(self,request):
        keys=request['keys'];dx=request['mouse_x'];dy=request['mouse_y']
        t=time.perf_counter()
        if self.is_csgo:
            if 'look_left' in keys:dx=-60
            if 'look_right' in keys:dx=60
            if 'look_up' in keys:dy=-50
            if 'look_down' in keys:dy=50
            a=encode_action(keys,dx,dy).to(self.device)
            self.act[:,-1]=a
        else:
            action=request['action']
            if not 0<=action<len(self.action_names):raise ValueError('Invalid Atari action')
            self.act[:,-1]=action
        # Exactly one autoregressive transition; no future ground-truth frames are read.
        obs,_=self.sampler.sample(self.obs,self.act)
        if not torch.isfinite(obs).all():raise ValueError('Non-finite world prediction')
        output=obs
        if self.upsampler:
            low=torch.nn.functional.interpolate(obs,scale_factor=5,mode='bicubic').unsqueeze(1)
            output,_=self.up_sampler.sample(torch.cat([self.full[:,-1:],low],dim=1),None)
            if not torch.isfinite(output).all():raise ValueError('Non-finite upsampler prediction')
            self.full=self.full.roll(-1,1);self.full[:,-1]=output
        self.obs=self.obs.roll(-1,1);self.obs[:,-1]=obs
        self.act=self.act.roll(-1,1);self.step_number+=1
        return dict(frame=image_uri(output[0]),step=self.step_number,seconds=time.perf_counter()-t,action=a.cpu().tolist() if self.is_csgo else action)
