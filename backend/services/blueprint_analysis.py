"""Research Lab structural analysis and bounded synthetic diagnostics (no pretrained downloads)."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from backend.services.blueprints import validate, hidden_block, execution_summary


class LabProfile(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    task: Literal['llm', 'vlm', 'vla'] = 'llm'
    batch_size: int = Field(default=1, ge=1, le=4)
    seq_len: int = Field(default=16, ge=4, le=64)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    steps: int = Field(default=2, ge=1, le=5)
    learning_rate: float = Field(default=.001, gt=0, le=.1)
    vision_image_size: int = Field(default=32, ge=16, le=128)
    vision_patch_size: int = Field(default=8, ge=2, le=32)
    action_type: Literal['continuous', 'discrete', 'hybrid'] = 'continuous'
    action_dim: int = Field(default=7, ge=1, le=64)
    action_horizon: int = Field(default=4, ge=1, le=16)
    state_dim: int = Field(default=7, ge=0, le=64)
    hidden_dim: int = Field(default=64, ge=16, le=512)
    discrete_classes: list[int] = Field(default_factory=lambda: [4], max_length=4)


def identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def assert_analyzable(spec):
    def visit(block):
        if block['type'] in ('custom_op', 'inline_code', 'permute', 'adaptive_avg_pool'):
            raise ValueError('Advanced/custom operators require a manual forward/backward probe; analysis does not execute custom code or verify sequence-axis changes')
        params = block.get('params', {})
        for child in params.get('body', []):
            visit(child)
        if block['type'] == 'repeater':
            visit({'type':params['block_type'], 'params':params.get('block_params', {})})
    for block in spec['blocks']:
        visit(block)


def build(spec, profile):
    from backend.services.blueprint_model import StudioBlueprintConfig, StudioBlueprintForCausalLM
    if profile.task == 'llm':
        model = StudioBlueprintForCausalLM(StudioBlueprintConfig(vocab_size=spec['vocab_size'], hidden_size=spec['hidden_size'],
            max_position_embeddings=spec['max_position_embeddings'], blueprint=spec, tie_word_embeddings=False, use_cache=False, studio_graph_version=1))
        return model, model.model, spec, {}
    from backend.services.blueprint_multimodal import normalize, BlueprintVisionLM
    config = profile.model_dump()
    config.update(model_blueprint_spec=spec, tuning_method='full', architecture='blueprint_'+profile.task)
    normalize(config, '.', task=profile.task)
    if profile.task == 'vlm':
        model = BlueprintVisionLM(config)
        return model, model.core.graph, config['model_blueprint_spec'], config
    if profile.action_type == 'discrete' and profile.action_dim < 2:
        raise ValueError('Discrete VLA needs at least two classes')
    if profile.action_type == 'hybrid' and (not profile.discrete_classes or any(not 2 <= n <= 64 for n in profile.discrete_classes)):
        raise ValueError('Hybrid VLA needs 1..4 categorical heads, each 2..64 classes')
    from backend.services.vla_model import create
    model, _ = create(config)
    return model, model.research_core.graph, config['model_blueprint_spec'], config


def capabilities(spec,profile=None):
    result = {'llm': {'supported':True, 'modes':['pretrain','SFT'], 'notes':['Choose tokenizer and dataset in Training Studio.']}}
    for task in ('vlm', 'vla'):
        try:
            from backend.services.blueprint_multimodal import normalize
            options=profile.model_dump() if profile is not None else {}
            c = normalize({**options,'model_blueprint_spec':spec, 'tuning_method':'full'}, '.', task=task)
            result[task] = {'supported':True, 'modes':['SFT / full weights'], 'notes':[
                f"Native from-scratch adapter: {c['vision_tokens']} image patches + explicit byte vocabulary (259).",
                'No pretrained vision knowledge; Unsloth/LoRA/QLoRA/GRPO are not implemented for this adapter.']}
        except (ValueError, TypeError) as exc:
            result[task] = {'supported':False, 'modes':[], 'notes':[str(exc)]}
    return result


def macs(module, batch, tokens, *, image_size=32, mode='train'):
    """Dense/attention MAC estimate. Excludes elementwise ops, dispatch and memory traffic."""
    import torch.nn as nn
    from homellm.models.blocks.recurrent import LoopedBlock
    name = type(module).__name__
    if hasattr(module,'research_core') and hasattr(module,'head'):
        return macs(module.research_core,batch,tokens,image_size=image_size,mode=mode)+macs(module.head,batch,1,image_size=image_size,mode=mode)
    if isinstance(module, LoopedBlock):
        count = module.iterations if mode == 'train' else module.eval_iterations
        return count * (macs(module.body, batch, tokens, image_size=image_size, mode=mode) + (macs(module.adapter, batch, tokens) if module.adapter else 0))
    if isinstance(module, nn.Linear):
        return batch * tokens * module.in_features * module.out_features
    if isinstance(module, nn.Conv2d):
        patches = image_size // module.stride[0]
        return batch * patches**2 * module.out_channels * (module.in_channels // module.groups) * math.prod(module.kernel_size)
    if isinstance(module, nn.Conv1d):
        return batch * tokens * module.out_channels * (module.in_channels // module.groups) * module.kernel_size[0]
    if name == 'CausalSelfAttention':
        return sum(macs(child, batch, tokens) for child in module.children()) + 2 * batch * tokens**2 * module.hidden_size
    if name == 'CausalAttention':
        return batch * tokens * (module.mha.in_proj_weight.numel() + module.mha.out_proj.weight.numel()) + 2 * batch * tokens**2 * module.mha.embed_dim
    if name == 'MoE':
        return macs(module.gate, batch, tokens) + sum(macs(e, batch, tokens) for e in module.experts) * module.num_select / module.num_experts
    return sum(macs(child, batch, tokens, image_size=image_size, mode=mode) for child in module.children())


def rows(spec, graph, batch, tokens):
    result = []
    widths = {'input':None}
    previous = 'input'
    def visit(block, module, width, path, parent, multiplier=1):
        output = hidden_block(block['type'], dict(block.get('params', {})), width, path)
        result.append({'path':path, 'parent':parent, 'type':block['type'], 'input_width':width, 'output_width':output,
            'input_shape':[batch,tokens] if block['type']=='token_embedding' or width is None else [batch,tokens,width], 'output_shape':[batch,tokens,output],
            'parameters':sum(p.numel() for p in module.parameters()), 'own_parameters':sum(p.numel() for p in module.parameters(recurse=False)),
            'train_calls_max':multiplier, 'dense_macs_estimate':int(macs(module,batch,tokens)*multiplier)})
        params = block.get('params', {})
        if block['type'] in ('loop','residual'):
            dim = width
            for index, child in enumerate(params['body']):
                dim = visit(child, module.body.layers[index], dim, f'{path}.body[{index}]', path, multiplier * params.get('iterations',4) if block['type']=='loop' else multiplier)
        elif block['type']=='repeater':
            dim = width
            for index, child in enumerate(module.layers):
                dim = visit({'type':params['block_type'],'params':params.get('block_params',{})},child,dim,f'{path}.layers[{index}]',path,multiplier)
        return output
    for block in spec['blocks']:
        parents = block['inputs'] or [previous]
        width = widths[parents[0]]
        kind = block['type']
        if kind in ('token_embedding','positional_embedding'):
            width = block['params'].get('hidden_size',spec['hidden_size'])
        if kind=='concat':
            width = sum(widths[p] for p in parents)
        output = visit(block, graph.modules_dict[block['id']], width, block['id'],None)
        root_row=next(row for row in result if row['path']==block['id'])
        root_row['inputs']=parents
        root_row['input_shapes']=[[batch,tokens] if widths[p] is None else [batch,tokens,widths[p]] for p in parents]
        widths[block['id']] = output; previous = block['id']
    return result


def analyze(spec, profile=None):
    import torch
    import psutil
    profile = LabProfile.model_validate(profile or {}) if not isinstance(profile,LabProfile) else profile
    spec = validate(spec)
    assert_analyzable(spec)
    with torch.device('meta'):
        model, graph, effective, config = build(spec, profile)
    parameter_count = sum(p.numel() for p in model.parameters())
    tokens = min(profile.seq_len, effective['max_position_embeddings']) + config.get('vision_tokens',0)
    if tokens > effective['max_position_embeddings']:
        raise ValueError('Probe text tokens + image patches exceed context; reduce probe seq_len or image size')
    calls, loops = execution_summary(effective)
    trace = rows(effective,graph,profile.batch_size,tokens)
    warnings = ['Parameter memory excludes activations, gradients, optimizer state and temporary attention tensors.',
        'Dense MACs are analytical estimates, not measured FLOPs or a training-time prediction.',
        'Block parameter totals include nested children; use the unique total rather than summing these rows.']
    for loop in loops:
        if loop['gradient_steps'] and loop['input_injection']=='none':
            warnings.append(f"{loop['id']}: truncated gradients without input injection may detach the prelude.")
        if loop['eval_iterations'] > loop['iterations']:
            warnings.append(f"{loop['id']}: additional inference depth does not guarantee improved quality.")
    if any(row['type']=='groupnorm' for row in trace):
        warnings.append('Causality warning: GroupNorm wrapper uses sequence-wide statistics and can leak future tokens.')
    floor = parameter_count * 16 / 2**20
    available = psutil.virtual_memory().available / 2**20
    return {'blueprint':spec, 'effective_blueprint':effective, 'source_hash':identity(spec), 'effective_hash':identity(effective),
        'profile':profile.model_dump(), 'parameters':parameter_count,'parameter_memory_fp32_mib':parameter_count*4/2**20,
        'fp32_adam_parameter_floor_mib':floor,'max_block_applications':calls,'loops':loops,'blocks':trace,
        'dense_macs_estimate':int(macs(model,profile.batch_size,tokens,image_size=profile.vision_image_size)),
        'eval_dense_macs_estimate':int(macs(model,profile.batch_size,tokens,image_size=profile.vision_image_size,mode='eval')),
        'output_tensor_volume_mib':sum(profile.batch_size*tokens*r['output_width']*r['train_calls_max']*4 for r in trace)/2**20,
        'hardware':{'ram_available_mib':available,'torch_cuda_available':torch.cuda.is_available(),
                    'fit_status':'parameter_floor_exceeds_ram' if floor>available*.8 else 'requires_activation_measurement'},
        'contracts':{'task':profile.task, 'text_tokens':tokens-config.get('vision_tokens',0),'vision_tokens':config.get('vision_tokens',0),
            'vocabulary':effective['vocab_size'],'output':'token logits' if profile.task!='vla' else 'action chunks',
            'adapter':'none' if profile.task=='llm' else 'trainable patch projection + positional vectors + byte tokenizer' + (' + action head' if profile.task=='vla' else ' + LM head')},
        'capabilities':capabilities(spec,profile), 'warnings':warnings}


def tensor_stats(tensor):
    tensor = tensor.detach().float()
    finite = torch_isfinite(tensor)
    safe = tensor[finite]
    return {'shape':list(tensor.shape), 'finite_fraction':float(finite.float().mean()),
        'mean':float(safe.mean()) if safe.numel() else None, 'std':float(safe.std(unbiased=False)) if safe.numel() else None,
        'rms':float(safe.square().mean().sqrt()) if safe.numel() else None,
        'abs_max':float(safe.abs().max()) if safe.numel() else None,
        'zero_fraction':float((safe==0).float().mean()) if safe.numel() else None}


def torch_isfinite(value):
    import torch
    return torch.isfinite(value)


def synthetic_probe(spec, profile):
    import torch
    import time
    import sys
    import psutil
    torch.set_num_threads(2)
    profile = LabProfile.model_validate(profile)
    report = analyze(spec,profile)
    if report['fp32_adam_parameter_floor_mib']*2+256>report['hardware']['ram_available_mib']*.8:
        raise ValueError('Insufficient available RAM for the bounded instrumented probe')
    if report['contracts']['text_tokens']<4 or report['contracts']['vocabulary']<2:
        raise ValueError('Synthetic probes need at least four text positions and two vocabulary entries')
    if report['parameters'] > 8_000_000:
        raise ValueError('Synthetic CPU probe is limited to 8M parameters; use a smaller research recipe or run full training on the intended hardware')
    if report['output_tensor_volume_mib'] > 32:
        raise ValueError('Synthetic probe activation volume exceeds its bound; reduce batch/sequence/depth/width')
    torch.manual_seed(profile.seed)
    started = time.perf_counter()
    model, graph, effective, config = build(validate(spec),profile)
    b,t = profile.batch_size,profile.seq_len
    t=min(t,effective['max_position_embeddings']-config.get('vision_tokens',0))
    data_generator=torch.Generator().manual_seed(profile.seed)
    ids = torch.randint(3, effective['vocab_size'], (b,t),generator=data_generator) if effective['vocab_size']>3 else torch.randint(0,effective['vocab_size'],(b,t),generator=data_generator)
    mask = torch.ones_like(ids)
    images = torch.rand(b,3,profile.vision_image_size,profile.vision_image_size,generator=data_generator)
    states = torch.randn(b,profile.state_dim,generator=data_generator)
    if profile.task=='vla':
        if profile.action_type=='discrete':
            targets = torch.randint(profile.action_dim,(b,profile.action_horizon),generator=data_generator)
        else:
            targets = torch.randn(b,profile.action_horizon,profile.action_dim+(len(profile.discrete_classes) if profile.action_type=='hybrid' else 0),generator=data_generator)
            if profile.action_type=='hybrid':
                for index,classes in enumerate(profile.discrete_classes):
                    targets[:,:,profile.action_dim+index]=torch.randint(classes,(b,profile.action_horizon),generator=data_generator)
    data_digest=hashlib.sha256()
    for tensor in (ids,images,states,*([targets] if profile.task=='vla' else [])):
        data_digest.update(tensor.numpy().tobytes())
    torch.manual_seed(profile.seed+1)
    def forward_loss():
        if profile.task=='llm':
            return model(ids,attention_mask=mask,labels=ids).loss
        if profile.task=='vlm':
            return model(images,ids,mask,labels=ids)['loss']
        from backend.services.vla_training import action_loss
        # Direct core use preserves an exactly matched synthetic sequence length.
        hidden,full_mask,_=model.research_core(images,ids,mask)
        features=hidden[:,-1]
        predictions=model.head(torch.cat((features,states),-1)).view(b,profile.action_horizon,model.output_dim)
        return action_loss(predictions,targets,torch.ones(b,profile.action_horizon),config)
    measurements={}; handles=[]
    previous_outputs={}; current_step=0
    module_by_path={}
    def attach(block,module,path):
        module_by_path[path]=module
        value={'path':path,'type':block['type'],'observed_forward_calls':0,'samples':[]}
        measurements[path]=value
        def hook(_,args,out):
            if isinstance(out,torch.Tensor):
                stats=tensor_stats(out)
                stats['optimizer_step']=current_step
                if out.ndim==3 and out.shape[1]>1:
                    stats['token_variation']=float(out.detach().float().std(dim=1,unbiased=False).mean())
                old=previous_outputs.get(path)
                stats['relative_change']=float((out.detach().float()-old).norm()/(old.norm()+1e-8)) if old is not None and old.shape==out.shape else None
                if out.numel()<=16384:
                    previous_outputs[path]=out.detach().float().clone()
                value['observed_forward_calls']+=1; value['last_activation']=stats
                if len(value['samples'])<8: value['samples'].append(stats)
        handles.append(module.register_forward_hook(hook))
        params=block.get('params',{})
        if block['type'] in ('loop','residual'):
            for i,child in enumerate(params['body']): attach(child,module.body.layers[i],f'{path}.body[{i}]')
        if block['type']=='repeater':
            for i,child in enumerate(module.layers): attach({'type':params['block_type'],'params':params.get('block_params',{})},child,f'{path}.layers[{i}]')
    for block in effective['blocks']:
        attach(block,graph.modules_dict[block['id']],block['id'])
    initial={name:p.detach().clone() for name,p in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=profile.learning_rate)
    loss_history=[]
    try:
        model.train()
        from torch.profiler import profile as profiler, ProfilerActivity
        for index in range(profile.steps):
            current_step=index+1; previous_outputs.clear()
            optimizer.zero_grad(set_to_none=True)
            loss=forward_loss()
            if not torch.isfinite(loss): raise ValueError('Probe loss is not finite')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('Probe gradients are not finite')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); loss_history.append(float(loss.detach()))
        gradients=[]
        for path,module in module_by_path.items():
            params=list(module.parameters())
            active=[p.grad.detach().float() for p in params if p.grad is not None]
            measurements[path]['gradient_norm']=float(torch.stack([g.square().sum() for g in active]).sum().sqrt()) if active else None
            measurements[path]['parameters_without_grad']=sum(p.numel() for p in params if p.requires_grad and p.grad is None)
        update=sum(float((p.detach()-initial[name]).square().sum()) for name,p in model.named_parameters())**.5
        for name,p in model.named_parameters():
            gradients.append({'name':name,'parameters':p.numel(),'gradient_norm':float(p.grad.float().norm()) if p.grad is not None else None,
                'update_norm':float((p.detach()-initial[name]).float().norm())})
        for handle in handles: handle.remove()
        handles=[]
        # Profile a separate real forward/backward without diagnostic hooks or an
        # optimizer update, so tensor-statistics code cannot dominate model hotspots.
        optimizer.zero_grad(set_to_none=True)
        with profiler(activities=[ProfilerActivity.CPU],record_shapes=True,with_flops=True) as trace:
            profile_loss=forward_loss()
            profile_loss.backward()
        model.eval()
        changed=ids.clone(); changed[:,t//2:]=(changed[:,t//2:]+1)%effective['vocab_size']
        with torch.no_grad():
            if profile.task=='llm':
                before=model(ids,attention_mask=mask).logits; after=model(changed,attention_mask=mask).logits
            else:
                core=model.core if profile.task=='vlm' else model.research_core
                a,_,prefix=core(images,ids,mask); z,_,_=core(images,changed,mask)
                before=a[:,prefix:]; after=z[:,prefix:]
            causal_delta=float((before[:,:t//2]-after[:,:t//2]).abs().max())
            sensitivity=None
            if profile.task!='llm':
                core=model.core if profile.task=='vlm' else model.research_core
                original,_,prefix=core(images,ids,mask); alternate,_,_=core(1-images,ids,mask)
                sensitivity=float((original[:,-1]-alternate[:,-1]).float().norm())
        ops=sorted(trace.key_averages(),key=lambda row:row.self_cpu_time_total,reverse=True)[:15]
        warnings=['Synthetic initialization probe; losses and timings are not a dataset-quality benchmark or GPU training ETA. Operator timings are a separate forward/backward with diagnostic hooks disabled (no additional optimizer step).',
                  'Nested activation/gradient rows overlap; do not sum them as unique totals. Observed forward calls include checkpoint recomputation.']
        if profile.task!='llm': warnings.append('Multimodal token_embedding hooks observe byte lookup before image-prefix fusion; static shapes describe fused graph states.')
        if causal_delta>1e-5: warnings.append('Causality check failed: changing future tokens changed earlier representations.')
        if sensitivity is not None and sensitivity<1e-8: warnings.append('Image perturbation did not affect the final text feature; inspect cross-token mixing and context.')
        if any(row['parameters_without_grad'] for row in measurements.values()): warnings.append('Some parameters received no gradient in this sample. Check truncation, routing and disconnected branches.')
        try:
            import resource
            peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 1024)
        except ImportError:
            info=psutil.Process().memory_info(); peak_rss=getattr(info,'peak_wset',info.rss)/2**20
        return {'data_fingerprint':data_digest.hexdigest(),'environment':{'data_seed':profile.seed,'training_seed':profile.seed+1,'python':sys.version.split()[0],'torch':str(torch.__version__),'platform':sys.platform,'threads':torch.get_num_threads(),'synthetic_data_version':1},'source_hash':report['source_hash'],'effective_hash':report['effective_hash'],'profile':profile.model_dump(),
            'device':'cpu','precision':'float32','loss_history':loss_history,'optimizer_steps':profile.steps,'weight_update_norm':update,
            'elapsed_ms':(time.perf_counter()-started)*1000,'process_peak_rss_mib':peak_rss,
            'causality':{'passed':causal_delta<=1e-5,'max_prefix_change':causal_delta},'image_sensitivity':sensitivity,
            'blocks':list(measurements.values()),'gradients':sorted(gradients,key=lambda r:r['gradient_norm'] or 0,reverse=True)[:40],
            'operators':[{'name':row.key,'calls':row.count,'self_cpu_ms':row.self_cpu_time_total/1000,'flops':row.flops} for row in ops], 'warnings':warnings}
    finally:
        for handle in handles: handle.remove()
