"""Opt-in periodic signals from a real training batch; retain only bounded scalar samples."""
import torch


class RuntimeSignals:
    def __init__(self, graph, spec, interval=20):
        self.step=0; self.active=False; self.values={}; self.modules={}; self.handles=[]; self.interval=interval
        def attach(block,module,path):
            if len(self.modules)>=64: return
            self.modules[path]=module
            def capture(_,args,out):
                if not self.active or not isinstance(out,torch.Tensor): return
                sample=out.detach().reshape(-1)[:4096].float()
                finite=torch.isfinite(sample); safe=sample[finite]
                self.values[path]={'path':path,'type':block['type'],'shape':list(out.shape),'sampled_values':sample.numel(),
                    'finite_fraction':float(finite.float().mean()) if sample.numel() else 1.,
                    'rms':float(safe.square().mean().sqrt()) if safe.numel() else None,
                    'std':float(safe.std(unbiased=False)) if safe.numel() else None}
            self.handles.append(module.register_forward_hook(capture))
            params=block.get('params',{})
            if block['type'] in ('loop','residual'):
                for index,child in enumerate(params['body']): attach(child,module.body.layers[index],f'{path}.body[{index}]')
            if block['type']=='repeater':
                for index,child in enumerate(module.layers): attach({'type':params['block_type'],'params':params.get('block_params',{})},child,f'{path}.layers[{index}]')
        for block in spec['blocks']: attach(block,graph.modules_dict[block['id']],block['id'])

    def begin(self,step):
        self.step=step; self.active=step==1 or step%self.interval==0
        if self.active: self.values={}

    def collect(self,gradient_stage='unscaled_after_clip'):
        if not self.active: return None
        for path,value in self.values.items():
            samples=[]; left=4096; missing=0
            for parameter in self.modules[path].parameters():
                if parameter.grad is None:
                    if parameter.requires_grad: missing+=parameter.numel()
                elif left:
                    sample=parameter.grad.detach().reshape(-1)[:left].float();samples.append(sample);left-=sample.numel()
            gradient=torch.cat(samples) if samples else None
            finite=gradient[torch.isfinite(gradient)] if gradient is not None else None
            value['gradient_rms_sampled']=float(finite.square().mean().sqrt()) if finite is not None and finite.numel() else None
            value['parameters_without_grad']=missing
        self.active=False
        return {'step':self.step,'device':'runtime','gradient_stage':gradient_stage,'blocks':list(self.values.values()),
            'note':'Real last captured microbatch; max 4096 activation/gradient values per block. Zero sampled gradients do not prove the whole tensor has zero gradient. Nested rows overlap; checkpoint recomputation can supply the last activation.'}

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles=[]
