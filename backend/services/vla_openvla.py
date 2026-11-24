"""Original OpenVLA action-token objective and generation (Transformers 4.40.1)."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def prompt(instruction):
    return f'In: What action should the robot take to {instruction.strip().lower()}?\nOut:'


def action_statistics(rows):
    values=np.concatenate([row['actions'] for row in rows])
    if np.any((values[:,-1]<0)|(values[:,-1]>1)):
        raise ValueError('OpenVLA gripper (last action coordinate) must be in [0,1]')
    low,high=np.quantile(values,[.01,.99],axis=0)
    return {'q01':low.tolist(),'q99':high.tolist(),'mask':[True]*6+[False]}


def normalize_actions(actions,stats):
    values=np.asarray(actions,dtype=np.float32).copy()
    low,high=np.asarray(stats['q01']),np.asarray(stats['q99'])
    scaled=np.clip(2*(values-low)/np.maximum(high-low,1e-6)-1,-1,1)
    return np.where(stats['mask'],scaled,values).astype(np.float32)


def encode_actions(actions,vocab_size):
    # Same uniform bins and reverse vocabulary mapping as upstream ActionTokenizer.
    return vocab_size-np.digitize(np.clip(actions,-1,1),np.linspace(-1,1,256))


def decode_actions(ids,vocab_size):
    bins=np.linspace(-1,1,256);centers=(bins[:-1]+bins[1:])/2
    return centers[np.clip(vocab_size-np.asarray(ids)-1,0,len(centers)-1)]


def metadata(c):
    from importlib.metadata import version
    if version('transformers')!='4.40.1':raise ValueError('Original OpenVLA requires isolated transformers==4.40.1 (MAH_OPENVLA_PYTHON)')
    from transformers import AutoConfig,AutoProcessor
    cfg=AutoConfig.from_pretrained(c['model_path'],trust_remote_code=True,revision=c.get('policy_revision') or None)
    if cfg.model_type!='openvla':raise ValueError('Choose an original OpenVLA checkpoint, not a generic VLM')
    processor=AutoProcessor.from_pretrained(c.get('policy_processor_path') or c['model_path'],trust_remote_code=True,revision=c.get('policy_revision') or None)
    if not hasattr(processor,'image_processor') or not hasattr(processor,'tokenizer'):
        raise ValueError('OpenVLA checkpoint is missing its Prismatic processor configuration')
    if processor.tokenizer.vocab_size<512:raise ValueError('OpenVLA action vocabulary is invalid')
    return cfg,processor


class Adapter(nn.Module):
    def __init__(self,c):
        super().__init__();self.studio_config=c
        _,self.processor=metadata(c)
        from transformers import AutoModelForVision2Seq
        self.policy=AutoModelForVision2Seq.from_pretrained(c['model_path'],trust_remote_code=True,
            revision=c.get('policy_revision') or None,torch_dtype=torch.bfloat16 if c['tuning_method']=='lora' and c['mixed_precision']=='bf16' else torch.float32,low_cpu_mem_usage=True,attn_implementation='eager')
        if c['freeze_vision_tower']:
            self.policy.vision_backbone.requires_grad_(False)
        if c['tuning_method']=='lora':
            from peft import LoraConfig,get_peft_model
            self.policy=get_peft_model(self.policy,LoraConfig(r=c['lora_r'],lora_alpha=c['lora_alpha'],lora_dropout=c['lora_dropout'],target_modules=c.get('lora_target_modules') or 'all-linear',init_lora_weights='gaussian'))
        if c['freeze_vision_tower']:self.policy.vision_backbone.requires_grad_(False)
        if c['gradient_checkpointing']:
            self.policy.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            if hasattr(self.policy.language_model,'enable_input_require_grads'):self.policy.language_model.enable_input_require_grads()
        self.backbone=None

    def prepare(self,rows,c,stats):
        from PIL import Image,ImageOps
        encoded=[];pixels=[];labels=[];targets=[]
        tok=self.processor.tokenizer
        if tok.eos_token_id is None or tok.pad_token_id is None:raise ValueError('OpenVLA tokenizer needs EOS and PAD IDs')
        for row in rows:
            with Image.open(row['image']) as im:
                inputs=self.processor(prompt(row['instruction']),ImageOps.exif_transpose(im).convert('RGB'),return_tensors='pt')
            prefix=inputs['input_ids'][0].tolist()
            # OpenVLA inference explicitly appends Llama's empty space token 29871.
            if prefix[-1]!=29871:prefix.append(29871)
            target=normalize_actions(row['actions'],stats['openvla'])
            ids=encode_actions(target.reshape(-1),tok.vocab_size).tolist()+[tok.eos_token_id]
            sequence=prefix+ids
            if len(sequence)>c['seq_len']:raise ValueError('OpenVLA prompt/actions exceed seq_len; no silent truncation')
            encoded.append(sequence);labels.append([-100]*len(prefix)+ids);pixels.append(inputs['pixel_values'][0]);targets.append(target)
        length=max(map(len,encoded))
        ids=torch.full((len(rows),length),tok.pad_token_id,dtype=torch.long)
        lab=torch.full_like(ids,-100);mask=torch.zeros_like(ids)
        for i,row in enumerate(encoded):ids[i,:len(row)]=torch.tensor(row);lab[i,:len(row)]=torch.tensor(labels[i]);mask[i,:len(row)]=1
        return {'input_ids':ids,'attention_mask':mask,'pixel_values':torch.stack(pixels),'labels':lab},torch.empty(len(rows),0),torch.tensor(np.stack(targets)),torch.ones(len(rows),1,dtype=torch.bool)

    def loss(self,inputs):
        output=self.policy(**inputs,use_cache=False)
        return output.loss,{}

    def predict(self,inputs):
        # Generate each prompt independently: trailing action labels must never leak into inference.
        outputs=[];device=inputs['input_ids'].device
        for i in range(len(inputs['input_ids'])):
            start=int((inputs['labels'][i]!=-100).nonzero()[0])
            ids=inputs['input_ids'][i:i+1,:start]
            predicted=self.policy.generate(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=inputs['pixel_values'][i:i+1],
                max_new_tokens=7,min_new_tokens=7,do_sample=False,use_cache=True)
            tokens=predicted[0,-7:].detach().cpu().numpy()
            outputs.append(decode_actions(tokens,self.processor.tokenizer.vocab_size))
        return torch.tensor(np.stack(outputs),device=device,dtype=torch.float32).unsqueeze(1)

    def denormalize(self,values,stats):
        st=stats['openvla'];low=values.new_tensor(st['q01']);high=values.new_tensor(st['q99'])
        return torch.where(torch.tensor(st['mask'],device=values.device),(values+1)*.5*(high-low)+low,values)

    def save(self,directory,stats,c):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        base=self.policy.get_base_model() if hasattr(self.policy,'get_base_model') else self.policy
        base.norm_stats={'studio':{'action':stats['openvla']}}
        base.config.norm_stats=base.norm_stats
        self.policy.save_pretrained(directory/'policy',safe_serialization=True)
        type(self.processor.image_processor).register_for_auto_class('AutoImageProcessor')
        type(self.processor).register_for_auto_class('AutoProcessor')
        self.processor.save_pretrained(directory/'processor')
        (directory/'policy.json').write_text(json.dumps({'format':'mah_vla_v1','config':c,'statistics':stats,'implementation':'openvla action tokenizer v1'},indent=2))
        (directory/'dataset_statistics.json').write_text(json.dumps({'studio':{'action':stats['openvla']}},indent=2))

    def restore(self,directory):
        directory=Path(directory)/'policy'
        if self.studio_config['tuning_method']=='lora':
            from peft import set_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights
            result=set_peft_model_state_dict(self.policy,load_peft_weights(str(directory)))
            if result.unexpected_keys:raise ValueError(f'Unexpected OpenVLA adapter weights: {result.unexpected_keys}')
        else:
            from safetensors.torch import load_model
            # Handle sharded full exports through Transformers; checks all missing/unexpected keys.
            from transformers.modeling_utils import load_sharded_checkpoint
            if (directory/'model.safetensors.index.json').is_file():load_sharded_checkpoint(self.policy,str(directory),strict=True,prefer_safe=True)
            else:load_model(self.policy,str(directory/'model.safetensors'),strict=True)
