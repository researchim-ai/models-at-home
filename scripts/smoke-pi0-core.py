"""Reduced real upstream PI0: numerical flow objective and export lifecycle, CPU only.

Only configuration sizes are patched; forward/backward and ODE sampling are upstream.
This is not validation of pretrained 3B weights. Run in the isolated LeRobot environment.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from unittest.mock import patch
from lerobot.configs import FeatureType,PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
import lerobot.policies.pi0.modeling_pi0 as mod
from backend.services.vla_lerobot import Adapter
from torch import nn

torch.set_num_threads(2);torch.manual_seed(42)
OriginalPali=mod.PaliGemmaForConditionalGenerationWithPiGemma
OriginalGemma=mod.PiGemmaForCausalLM

def pali(*,config):
 config.vision_config.hidden_size=32;config.vision_config.num_hidden_layers=1;config.vision_config.num_attention_heads=4;config.vision_config.intermediate_size=64;config.vision_config.projection_dim=64;config.vision_config.patch_size=8
 config.text_config.vocab_size=256;config.image_token_index=255
 return OriginalPali(config=config)

def gemma(*,config):
 config.vocab_size=256
 return OriginalGemma(config=config)

with patch.object(mod,'get_gemma_config',lambda _:mod.GemmaConfig(width=64,depth=1,mlp_dim=128,num_heads=8,num_kv_heads=1,head_dim=8)),patch.object(mod,'PaliGemmaForConditionalGenerationWithPiGemma',pali),patch.object(mod,'PiGemmaForCausalLM',gemma):
 pc=PI0Config(device='cpu',chunk_size=2,n_action_steps=2,max_action_dim=2,max_state_dim=2,image_resolution=(32,32),num_inference_steps=2,train_expert_only=True,freeze_vision_encoder=True,
 input_features={'observation.images.camera':PolicyFeature(type=FeatureType.VISUAL,shape=(3,32,32)),'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(1,))},output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(2,))})
 model=mod.PI0Policy(pc)
 obj=Adapter.__new__(Adapter);nn.Module.__init__(obj);obj.kind='pi0';obj.policy=model
 inputs={'observation.images.camera':torch.rand(2,3,32,32),'observation.state':torch.randn(2,1),'observation.language.tokens':torch.randint(1,200,(2,4)),
 'observation.language.attention_mask':torch.ones(2,4,dtype=torch.bool),'action':torch.randn(2,2,2),'action_is_pad':torch.zeros(2,2,dtype=torch.bool)}
 optimizer=torch.optim.AdamW([p for p in obj.parameters() if p.requires_grad],lr=.001)
 obj.train();old={k:p.detach().clone() for k,p in obj.named_parameters() if p.requires_grad}
 loss,details=obj.loss(inputs);loss.backward();optimizer.step()
 assert torch.isfinite(loss)
 assert any(not torch.equal(old[k],p) for k,p in obj.named_parameters() if p.requires_grad)
 obj.eval();torch.manual_seed(123)
 with torch.no_grad():actions=obj.predict(inputs)
 assert actions.shape==(2,2,2) and torch.isfinite(actions).all()
 print('PASS real reduced PI0 flow matching, expert gradients, optimizer update and ODE action generation',float(loss.detach()))

 # Restore the exact official policy tensors while the same reduced config factories are active.
 import tempfile
 from safetensors.torch import save_model,load_model
 with tempfile.TemporaryDirectory(prefix='mah-pi0-core-') as tmp:
  path=Path(tmp)/'model.safetensors';save_model(obj.policy,str(path))
  restored=mod.PI0Policy(pc);load_model(restored,str(path),strict=True)
  obj.policy=restored;obj.eval();torch.manual_seed(123)
  with torch.no_grad():again=obj.predict(inputs)
  assert torch.equal(actions,again),'PI0 weight roundtrip changed action generation'
  print('PASS reduced PI0 safetensors roundtrip and seeded ODE generation')
