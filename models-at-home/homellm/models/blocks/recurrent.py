"""Composable recurrent-depth blocks. A loop owns one physical copy of its body."""
from __future__ import annotations

import inspect
import math
import torch
from torch import nn
from homellm.models.blocks import get_block, register_block


def call_block(module, x, attention_mask=None):
    signature = inspect.signature(module.forward).parameters
    if 'attention_mask' in signature or any(p.kind == p.VAR_KEYWORD for p in signature.values()):
        return module(x, attention_mask=attention_mask)
    return module(x)


def auxiliary_losses(module):
    value = getattr(module, 'aux_loss', None)
    result = [value] if isinstance(value, torch.Tensor) else []
    if not getattr(module, '_aggregates_aux_loss', False):
        for child in module.children():
            result.extend(auxiliary_losses(child))
    return result


class BlockSequence(nn.Module):
    def __init__(self, body, width, auto_project):
        super().__init__()
        layers = []
        for spec in body:
            if spec['type'] in ('token_embedding', 'positional_embedding', 'add', 'multiply', 'concat'):
                raise ValueError('Container bodies contain sequential hidden-state blocks, not graph joins or embeddings')
            module, width = get_block(spec['type'])(dict(spec.get('params', {})), width, auto_project)
            layers.append(module)
        if not layers:
            raise ValueError('Container body must not be empty')
        self.layers = nn.ModuleList(layers)
        self.output_width = width

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = call_block(layer, x, attention_mask)
        return x


class LoopedBlock(nn.Module):
    _aggregates_aux_loss = True

    def __init__(self, params, width, auto_project):
        super().__init__()
        self.iterations = params.get('iterations', 4)
        self.eval_iterations = params.get('eval_iterations', self.iterations)
        self.min_iterations = params.get('min_iterations', 1)
        self.random_depth = params.get('random_depth', False)
        self.gradient_steps = params.get('gradient_steps', 0)
        self.checkpoint = params.get('checkpoint', False)
        self.input_injection = params.get('input_injection', 'none')
        self.state_init = params.get('state_init', 'input')
        self.step_embedding = params.get('step_embedding', 'none')
        for value in (self.iterations, self.eval_iterations, self.min_iterations):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 256:
                raise ValueError('Loop depth must be an integer in [1, 256]')
        if not 1 <= self.min_iterations <= self.iterations:
            raise ValueError('min_iterations must not exceed iterations')
        if isinstance(self.gradient_steps, bool) or not isinstance(self.gradient_steps, int) or not 0 <= self.gradient_steps <= self.iterations:
            raise ValueError('gradient_steps must be zero (full BPTT) or in [1, iterations]')
        if self.input_injection not in ('none', 'add', 'concat') or self.state_init not in ('input', 'zeros') or self.step_embedding not in ('none', 'sinusoidal'):
            raise ValueError('Unknown loop injection, state initialization, or step encoding')
        if self.state_init == 'zeros' and self.input_injection == 'none':
            raise ValueError('A zero-initialized loop needs input injection')
        self.body = BlockSequence(params.get('body', []), width, auto_project)
        if self.body.output_width != width:
            raise ValueError('Loop body must preserve hidden width')
        self.adapter = nn.Linear(2 * width, width, bias=False) if self.input_injection == 'concat' else None
        self.last_iterations = 0
        self.aux_loss = None

    def _step(self, state, initial, index, attention_mask):
        if self.input_injection == 'add':
            state = (state + initial) / math.sqrt(2)
        elif self.adapter is not None:
            state = self.adapter(torch.cat((state, initial), dim=-1))
        if self.step_embedding == 'sinusoidal':
            frequency = torch.exp(torch.arange(0, state.shape[-1], 2, device=state.device, dtype=torch.float32) * (-math.log(10000) / state.shape[-1]))
            encoding = torch.zeros(state.shape[-1], device=state.device, dtype=torch.float32)
            encoding[0::2] = torch.sin((index + 1) * frequency)
            encoding[1::2] = torch.cos((index + 1) * frequency[:encoding[1::2].numel()])
            state = state + encoding.to(state.dtype)
        return self.body(state, attention_mask=attention_mask)

    def forward(self, x, attention_mask=None, loop_iterations=None):
        if self.training and (self.gradient_steps or self.checkpoint):
            # A no-grad warmup must not cache detached AMP weight casts for the
            # differentiable tail. Checkpoint recomputation also needs identical
            # cast operations on every pass through the shared parameters.
            with torch.autocast(device_type=x.device.type,
                    enabled=torch.is_autocast_enabled(x.device.type), cache_enabled=False):
                return self._forward(x, attention_mask, loop_iterations)
        return self._forward(x, attention_mask, loop_iterations)

    def _forward(self, x, attention_mask=None, loop_iterations=None):
        if loop_iterations is not None:
            if self.training:
                raise ValueError('loop_iterations is an inference-only override')
            if isinstance(loop_iterations, bool) or not isinstance(loop_iterations, int) or not 1 <= loop_iterations <= 256:
                raise ValueError('loop_iterations must be in [1, 256]')
            depth = loop_iterations
        elif self.training:
            depth = int(torch.randint(self.min_iterations, self.iterations + 1, ()).item()) if self.random_depth else self.iterations
        else:
            depth = self.eval_iterations
        self.last_iterations = depth
        state = x if self.state_init == 'input' else torch.zeros_like(x)
        trained = min(depth, self.gradient_steps or depth)
        losses = []
        for index in range(depth):
            if self.training and index < depth - trained:
                with torch.no_grad():
                    state = self._step(state, x, index, attention_mask)
                state = state.detach()
            else:
                if self.training and self.checkpoint and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    # Bind index now: backward must recompute the same iteration.
                    def step(s, initial, index=index):
                        return self._step(s, initial, index, attention_mask)
                    state = checkpoint(step, state, x, use_reentrant=False)
                else:
                    state = self._step(state, x, index, attention_mask)
                losses.extend(auxiliary_losses(self.body))
        self.aux_loss = torch.stack(losses).mean() if losses else None
        return state


@register_block('loop', 'Shared-weight recurrent depth: execute one editable body repeatedly')
def build_loop(params, in_dim, auto_project):
    return LoopedBlock(params, in_dim, auto_project), in_dim


class ResidualSequence(nn.Module):
    def __init__(self, params, width, auto_project):
        super().__init__()
        self.body = BlockSequence(params.get('body', []), width, auto_project)
        if self.body.output_width != width:
            raise ValueError('Residual body must preserve hidden width')
        scale = torch.tensor(float(params.get('scale', 1.0)))
        if params.get('learnable_scale', False):
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer('scale', scale)

    def forward(self, x, attention_mask=None):
        return x + self.scale.to(x.dtype) * self.body(x, attention_mask=attention_mask)


@register_block('residual', 'Editable residual branch; optional learnable scale (ReZero when initialized to zero)')
def build_residual(params, in_dim, auto_project):
    return ResidualSequence(params, in_dim, auto_project), in_dim
