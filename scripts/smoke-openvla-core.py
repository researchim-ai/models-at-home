"""CPU mechanism test for a reduced original OpenVLA fixture, not a 7B benchmark."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
parser=argparse.ArgumentParser();parser.add_argument('--model-path',required=True);args=parser.parse_args()
import json
from pathlib import Path
import tempfile
import torch
from PIL import Image
import numpy as np
from backend.routes.vla import VLATrainingRequest
from backend.services.vla_openvla import Adapter,action_statistics
from backend.services.vla_policy_runtime import export_config

torch.set_num_threads(2)
for method in ['full','lora']:
 torch.manual_seed(42)
 c=VLATrainingRequest(architecture='openvla',model_path=args.model_path,device='cpu',mixed_precision='no',action_dim=7,action_horizon=1,state_dim=0,tuning_method=method,lora_r=2,lora_alpha=4).model_dump()
 with tempfile.TemporaryDirectory(prefix='mah-openvla-core-') as tmp:
  root=Path(tmp);Image.new('RGB',(32,32),'red').save(root/'i.png')
  rows=[{'image':str(root/'i.png'),'instruction':'reach red','state':np.empty(0,dtype=np.float32),'actions':np.array([[.1,.2,.3,0,0,0,1]],dtype=np.float32)}]
  st={'openvla':action_statistics(rows)}
  obj=Adapter(c);inputs,_,_,_=obj.prepare(rows,c,st)
  opt=torch.optim.AdamW([p for p in obj.parameters() if p.requires_grad],lr=.001)
  old={k:p.detach().clone() for k,p in obj.named_parameters() if p.requires_grad}
  obj.train();loss,_=obj.loss(inputs);loss.backward();opt.step();assert torch.isfinite(loss)
  assert any(not torch.equal(old[k],p) for k,p in obj.named_parameters() if p.requires_grad)
  if method=='lora':assert all(not p.requires_grad for p in obj.policy.vision_backbone.parameters())
  obj.eval()
  with torch.no_grad():actions=obj.predict(inputs)
  assert actions.shape==(1,1,7) and torch.isfinite(actions).all()
  obj.save(root/'export',st,c)
  restored=Adapter(export_config(c,root/'export'));restored.restore(root/'export');restored.eval()
  with torch.no_grad():new_actions=restored.predict(inputs)
  assert torch.equal(actions,new_actions)
  print('PASS real reduced OpenVLA',method,'action-token loss, gradients, update, generation and exported weights reload',float(loss.detach()),flush=True)
