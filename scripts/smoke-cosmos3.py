"""Numerical integration test for the *official* Cosmos3 transformer (no video/guardrail bypass).
Default loads the selected pretrained transformer. --tiny explicitly uses random tiny weights.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', default='/tmp/mah-world-pretrained/Cosmos3-Edge')
parser.add_argument('--tiny', action='store_true')
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--result', required=True)
args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'; os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
import torch
from diffusers import Cosmos3OmniTransformer, Cosmos3OmniPipeline

torch.manual_seed(42)
if args.tiny:
    model = Cosmos3OmniTransformer(hidden_size=32, intermediate_size=64, head_dim=16, num_attention_heads=2, num_key_value_heads=1,
        num_hidden_layers=2, latent_channel=4, latent_patch_size=2, patch_latent_dim=16, vocab_size=128,
        rope_axes_dim=[2, 3, 3], hidden_act='relu2', action_dim=8, action_gen=True, qk_norm_for_text=False, use_und_k_norm_for_gen=True).to('cuda', dtype=torch.bfloat16)
else:
    model = Cosmos3OmniTransformer.from_pretrained(args.model_path, subfolder='transformer', torch_dtype=torch.bfloat16).to('cuda')
model.eval(); model.set_attention_backend('native')
# Reuse official token packing, identical to inference, with a tiny synthetic latent grid.
stub = SimpleNamespace(transformer=model, vae_scale_factor_temporal=4)
text = Cosmos3OmniPipeline._prepare_text_segment(stub, [1, 2, 3], 'cuda')
latent = torch.randn(1, model.config.latent_channel, 2, 4, 4, device='cuda', dtype=torch.bfloat16)
vision = Cosmos3OmniPipeline._prepare_vision_segment(stub, latent, True, text['vision_start_temporal_offset'], 24., text['und_len'], 'cuda')
actions = torch.zeros(4, model.config.action_dim, device='cuda', dtype=torch.bfloat16)
act = Cosmos3OmniPipeline._prepare_action_segment(stub, actions, list(range(4)), text['vision_start_temporal_offset'], 24., text['und_len']+vision['num_vision_tokens'], 'cuda')
kwargs = {k: text[k] for k in ('input_ids','text_indexes','und_len')}
kwargs.update({k: vision[k] for k in ('vision_token_shapes','vision_sequence_indexes','vision_mse_loss_indexes','vision_noisy_frame_indexes')})
kwargs.update({k: act[k] for k in ('action_token_shapes','action_sequence_indexes','action_mse_loss_indexes','action_noisy_frame_indexes')})
kwargs.update(position_ids=torch.cat([text['text_mrope_ids'],vision['vision_mrope_ids'],act['action_mrope_ids']],1),
    sequence_length=text['und_len']+vision['num_vision_tokens']+act['action_len'], vision_tokens=[latent],
    vision_timesteps=torch.full((vision['num_noisy_vision_tokens'],),500.,device='cuda'),action_tokens=[actions],
    action_timesteps=torch.empty(0,device='cuda'),action_domain_ids=[torch.tensor(6,device='cuda')],return_dict=False)
with torch.inference_mode():
    prediction = model(**kwargs)[0][0]
    alternate = {**kwargs, 'action_tokens':[actions+1]}
    changed = model(**alternate)[0][0]
assert torch.isfinite(prediction).all() and torch.isfinite(changed).all()
delta = float((prediction.float()-changed.float()).abs().max())
assert delta > 0, 'Official action conditioning did not influence vision prediction'
result = dict(model='Cosmos3-Edge' if not args.tiny else 'tiny random official Cosmos3 class', pretrained=not args.tiny,
    device=torch.cuda.get_device_name(0), parameters=sum(p.numel() for p in model.parameters()),
    prediction_shape=list(prediction.shape), action_influence_max_abs=delta, finite=True,
    peak_memory_gb=torch.cuda.max_memory_allocated()/2**30,
    scope='Numerical denoising/action-conditioning test only; no decoded media or full guarded video generation')
Path(args.result).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
