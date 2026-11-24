"""Standalone policy contracts. No ML imports in the API process."""
import os
from pathlib import Path
import sys

LEROBOT_VERSION = '0.7.0'
POLICIES = {
    'act': {'name':'ACT', 'implementation':'LeRobot', 'default_model':'', 'objective':'L1 reconstruction + CVAE KL',
        'paper':'https://arxiv.org/abs/2304.13705', 'source':'https://huggingface.co/docs/lerobot/act',
        'note':'Image + state policy; does not condition on language. Train from scratch or load an ACT checkpoint. Single camera in this adapter.'},
    'smolvla': {'name':'SmolVLA', 'implementation':'LeRobot', 'default_model':'lerobot/smolvla_base', 'objective':'flow matching',
        'paper':'https://arxiv.org/abs/2506.01844', 'source':'https://huggingface.co/docs/lerobot/smolvla',
        'note':'Pretrained VLM + action expert. Continuous actions; image + instruction + state. Requires compatible dimensions and chunk size.'},
    'pi0': {'name':'π0', 'implementation':'LeRobot / OpenPI port', 'default_model':'lerobot/pi0_base', 'objective':'flow matching',
        'paper':'https://arxiv.org/abs/2410.24164', 'source':'https://huggingface.co/docs/lerobot/pi0',
        'note':'PaliGemma + action expert. Continuous actions, full chunks required; large model. Tokenizer may require HF access.'},
    'openvla': {'name':'OpenVLA', 'implementation':'OpenVLA Hugging Face remote model', 'default_model':'openvla/openvla-7b', 'objective':'autoregressive action-token cross entropy',
        'paper':'https://arxiv.org/abs/2406.09246', 'source':'https://github.com/openvla/openvla',
        'note':'Original 7B policy: one image + instruction → one 7D action. No proprioceptive state input. Last coordinate is gripper in [0,1]. CUDA required.'},
}


def catalog():
    return [{'id':key, **value, 'methods':['full','lora'] if key=='openvla' else ['full'],
             'validation':'CPU ACT and reduced SmolVLA lifecycle; reduced OpenVLA/Pi0 numerical tests; full-size pretrained GPU runs not validated'} for key,value in POLICIES.items()]


def python_for(architecture):
    if architecture not in POLICIES:
        if architecture == 'blueprint_vla':return sys.executable
        from backend.services.vlm_environment import python_command
        return python_command()
    variable = 'MAH_OPENVLA_PYTHON' if architecture == 'openvla' else 'MAH_LEROBOT_PYTHON'
    explicit=os.environ.get(variable)
    root=Path(__file__).resolve().parents[2]
    path=Path(explicit).expanduser().absolute() if explicit else root/('.venv-openvla' if architecture=='openvla' else '.venv-lerobot')/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not path.is_file() or not os.access(path,os.X_OK):
        raise ValueError(f'{architecture}: install the isolated policy environment (docs/VLA_POLICIES.md) and set {variable}; executable missing: {path}')
    return str(path)


def validate_contract(c):
    architecture=c['architecture']
    if architecture not in POLICIES:return
    if c['action_type']!='continuous':raise ValueError(f'{architecture} supports continuous actions only; use Native VLA for discrete/hybrid')
    if c['training_backend']!='transformers':raise ValueError('Standalone policies use their own implementation; Unsloth is not supported')
    if c['tuning_method'] not in (('full','lora') if architecture=='openvla' else ('full',)):
        raise ValueError(f'{architecture}: unsupported tuning method')
    if architecture!='act' and not c['model_path'].strip():raise ValueError(f'{architecture} requires a pretrained policy model_path')
    if architecture=='openvla' and (c['action_dim']!=7 or c['action_horizon']!=1 or c['state_dim']!=0):
        raise ValueError('Original OpenVLA requires action_dim=7, action_horizon=1 and state_dim=0; state is not silently discarded')
    if architecture in ('smolvla','pi0') and (c['state_dim']<1 or c['action_dim']>32 or c['state_dim']>32):
        raise ValueError(f'{architecture}: this adapter requires state_dim=1..32 and action_dim=1..32')
    if c['mixed_precision']=='fp16':raise ValueError('Standalone policies support Float32 or BF16; FP16 is not validated')
