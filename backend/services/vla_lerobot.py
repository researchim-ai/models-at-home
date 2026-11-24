"""LeRobot 0.7 policy adapter; own objectives, never a replacement MLP action head."""
import json
from pathlib import Path
import torch
from torch import nn
from backend.services.vla_policies import LEROBOT_VERSION


def classes(kind):
    from importlib.metadata import version
    if version('lerobot') != LEROBOT_VERSION:
        raise ValueError(f'Use lerobot=={LEROBOT_VERSION} in the isolated policy environment')
    if kind=='act':
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTConfig,ACTPolicy
    if kind=='smolvla':
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        return SmolVLAConfig,SmolVLAPolicy
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    return PI0Config,PI0Policy


def policy_config(c,device='cpu'):
    Config,_=classes(c['architecture'])
    from lerobot.configs import PolicyFeature,FeatureType
    if c['model_path']:
        pc=Config.from_pretrained(c['model_path'])
        if pc.type!=c['architecture']:raise ValueError(f'Checkpoint policy type {pc.type} does not match {c["architecture"]}')
        if pc.n_obs_steps!=1:raise ValueError('Only one observation per JSONL row is supported')
        if c['action_horizon']!=pc.chunk_size:raise ValueError(f'Checkpoint chunk_size={pc.chunk_size}; set action_horizon to this value')
        if c['architecture']=='act' and (pc.action_feature.shape[0]!=c['action_dim'] or pc.robot_state_feature.shape[0]!=c['state_dim']):
            raise ValueError('ACT checkpoint action/state dimensions must match demonstrations')
        if c['architecture']!='act' and (c['action_dim']>pc.max_action_dim or c['state_dim']>pc.max_state_dim):
            raise ValueError('Action/state dimensions exceed pretrained policy capacity')
        if getattr(pc,'adapt_to_pi_aloha',False) or getattr(pc,'use_delta_joint_actions_aloha',False) or getattr(pc,'use_relative_actions',False):
            raise ValueError('This single-image adapter expects explicit action units; ALOHA/relative transforms require a separate embodiment adapter')
        # State/action projection widths remain padded to the pretrained capacity.
        pc.input_features['observation.state']=PolicyFeature(type=FeatureType.STATE,shape=(c['state_dim'],))
        pc.output_features['action']=PolicyFeature(type=FeatureType.ACTION,shape=(c['action_dim'],))
    else:
        pc=Config(input_features={'observation.images.camera':PolicyFeature(type=FeatureType.VISUAL,shape=(3,c['image_size'],c['image_size'])),
            'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(c['state_dim'],))},
            output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(c['action_dim'],))},
            chunk_size=c['action_horizon'],n_action_steps=c['action_horizon'],device=device,
            pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if c.get('act_pretrained_backbone',False) else None,
            dim_model=c.get('act_dim_model',256),n_heads=8,dim_feedforward=c.get('act_dim_model',256)*4,
            n_encoder_layers=c.get('act_encoder_layers',4),n_decoder_layers=1,n_vae_encoder_layers=4)
    pc.device=device
    pc.push_to_hub=False
    if hasattr(pc,'compile_model'):pc.compile_model=False
    if c['architecture']=='act':
        if c['model_path']:pc.pretrained_backbone_weights=None
        if len(pc.image_features)!=1:raise ValueError('ACT JSONL adapter requires a single camera checkpoint')
    else:
        if c['architecture']=='smolvla':
            pc.load_vlm_weights=False  # Whole policy checkpoint already contains VLM weights.
            if c.get('policy_vlm_assets'):pc.vlm_model_name=c['policy_vlm_assets']
        pc.freeze_vision_encoder=c['freeze_vision_tower']
        pc.train_expert_only=c.get('policy_train_expert_only',True)
        if hasattr(pc,'gradient_checkpointing'):pc.gradient_checkpointing=c['gradient_checkpointing']
        # Keep trainable weights FP32; precision is managed by the Studio autocast loop.
        pc.dtype=torch.float32
    return pc


class Adapter(nn.Module):
    def __init__(self,c,*,load_weights=True):
        super().__init__();self.studio_config=c;self.kind=c['architecture']
        device='cuda' if c['device']!='cpu' and torch.cuda.is_available() else 'cpu'
        self.policy_config=pc=policy_config(c,device)
        _,Policy=classes(self.kind)
        self.policy=Policy.from_pretrained(c['model_path'],config=pc,strict=True) if c['model_path'] and load_weights else Policy(pc)
        self.tokenizer=None
        if self.kind!='act':
            from transformers import AutoTokenizer
            source=c.get('policy_tokenizer_path') or (pc.vlm_model_name if self.kind=='smolvla' else pc.text_tokenizer_name)
            self.tokenizer=AutoTokenizer.from_pretrained(source)
        self.camera=c.get('policy_camera_key') or next(iter(pc.image_features),'')
        if self.camera not in pc.image_features:raise ValueError(f'Camera key {self.camera} is not in checkpoint: {list(pc.image_features)}')
        self.backbone=None  # Only used by legacy model serialization dispatch.

    def prepare(self,rows,c,stats):
        from backend.services.vla_data import normalized_targets
        # Existing Studio normalization happens once, using training-set statistics.
        states,targets,mask=normalized_targets(rows,c,stats)
        shape=self.policy_config.image_features[self.camera].shape
        from PIL import Image,ImageOps
        import numpy as np
        images=[]
        for row in rows:
            with Image.open(row['image']) as image:
                image=ImageOps.exif_transpose(image).convert('RGB')
                image=image.resize((shape[2],shape[1])) if self.kind=='act' else ImageOps.pad(image,(shape[2],shape[1]),method=Image.Resampling.BILINEAR,color=0)
                images.append(torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255).permute(2,0,1))
        images=torch.stack(images)
        if self.kind=='act':
            # Studio's ACT visual preprocessing uses fixed ImageNet statistics.
            images=(images-images.new_tensor([.485,.456,.406])[None,:,None,None])/images.new_tensor([.229,.224,.225])[None,:,None,None]
        result={self.camera:images,'observation.state':states,'action':targets,'action_is_pad':~mask}
        if self.tokenizer is not None:
            text=[row['instruction'].rstrip()+'\n' for row in rows]
            encoded=self.tokenizer(text,padding='max_length' if self.kind=='pi0' else True,max_length=self.policy_config.tokenizer_max_length,truncation=False,return_tensors='pt')
            if encoded['input_ids'].shape[1]>self.policy_config.tokenizer_max_length:
                raise ValueError(f'Instruction exceeds policy tokenizer_max_length={self.policy_config.tokenizer_max_length}; no silent truncation')
            result.update({'observation.language.tokens':encoded['input_ids'],'observation.language.attention_mask':encoded['attention_mask'].bool()})
        return result,states,targets,mask

    def loss(self,inputs):
        if self.kind=='pi0' and inputs['action_is_pad'].any():
            raise ValueError('π0 adapter requires full chunks; upstream objective does not mask terminal padding')
        loss,details=self.policy(inputs)
        return loss,details

    def predict(self,inputs):
        self.policy.reset()  # Stateless chunk preview; no stale queued actions from another observation.
        clean={key:value for key,value in inputs.items() if key not in ('action','action_is_pad')}
        return self.policy.predict_action_chunk(clean)

    def save(self,directory,stats,c):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        self.policy.save_pretrained(directory/'policy')
        if self.tokenizer is not None:self.tokenizer.save_pretrained(directory/'tokenizer')
        if self.kind=='smolvla':
            assets=self.policy.model.vlm_with_expert
            assets.vlm.config.save_pretrained(directory/'vlm_assets')
            assets.processor.save_pretrained(directory/'vlm_assets')
        (directory/'policy.json').write_text(json.dumps({'format':'mah_vla_v1','config':c,'statistics':stats,'implementation':f'lerobot=={LEROBOT_VERSION}'},indent=2))

    def restore(self,directory):
        from safetensors.torch import load_model
        load_model(self.policy,str(Path(directory)/'policy'/'model.safetensors'),strict=True)
