"""Validate graph structure without importing Torch or executing custom blocks."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.services.blueprint_catalog import SUPPORTED


def hidden_block(kind, params, width, context, depth=0):
    """Validate nested containers without importing or allocating model weights."""
    if depth > 4:
        raise ValueError(f'{context}: container nesting is limited to four levels')
    if not isinstance(kind, str) or kind not in SUPPORTED:
        raise ValueError(f'{context}: unsupported block type {kind}')
    if not isinstance(params, dict):
        raise ValueError(f'{context}: params must be an object')
    def integer(key, default, low=1, high=None):
        value = params.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
            raise ValueError(f'{context}: invalid {key}')
        return value
    def number(key, default, low=None, high=None):
        import math
        value = params.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
            raise ValueError(f'{context}: invalid {key}')
        return value
    for key in ('checkpoint', 'random_depth', 'learnable_scale', 'use_rope', 'use_liger', 'sandwich_norm', 'bias'):
        if key in params and not isinstance(params[key], bool):
            raise ValueError(f'{context}: {key} must be a boolean')
    for key in ('dropout', 'p'):
        if key in params:
            number(key, 0, 0, 1)
    if 'eps' in params and number('eps', 1e-6) <= 0:
        raise ValueError(f'{context}: eps must be positive')
    if kind in ('loop', 'residual'):
        body = params.get('body')
        if not isinstance(body, list) or not 1 <= len(body) <= 64:
            raise ValueError(f'{context}: body must contain 1..64 blocks')
        current = width
        for index, child in enumerate(body):
            if not isinstance(child, dict) or set(child) - {'type', 'params'}:
                raise ValueError(f'{context}: body entries contain only type and params')
            if child.get('type') in ('token_embedding', 'positional_embedding', 'add', 'multiply', 'concat'):
                raise ValueError(f'{context}: body is sequential; embeddings and graph joins belong outside it')
            current = hidden_block(child.get('type'), child.setdefault('params', {}), current, f'{context}.body[{index}]', depth + 1)
        if current != width:
            raise ValueError(f'{context}: body must preserve hidden width ({width}, got {current})')
        if kind == 'loop':
            iterations = integer('iterations', 4, high=256)
            integer('eval_iterations', iterations, high=256)
            integer('min_iterations', 1, high=iterations)
            integer('gradient_steps', 0, low=0, high=iterations)
            if params.get('state_init', 'input') == 'zeros' and params.get('input_injection', 'none') == 'none':
                raise ValueError(f'{context}: zero state needs input injection')
            for key, options, default in [('input_injection', ('none', 'add', 'concat'), 'none'), ('state_init', ('input', 'zeros'), 'input'), ('step_embedding', ('none', 'sinusoidal'), 'none')]:
                if params.get(key, default) not in options:
                    raise ValueError(f'{context}: invalid {key}')
        else:
            number('scale', 1)
        return width
    if kind == 'repeater':
        count = integer('num_repeats', 1, high=256)
        child_type = params.get('block_type')
        if child_type in ('token_embedding', 'positional_embedding', 'add', 'multiply', 'concat'):
            raise ValueError(f'{context}: repeater requires a hidden-state block')
        output = hidden_block(child_type, params.setdefault('block_params', {}), width, f'{context}.block', depth + 1)
        if count > 1 and output != width:
            output = hidden_block(child_type, params['block_params'], output, f'{context}.block', depth + 1)
        return output
    if kind == 'moe':
        if 'top_k' in params:
            if 'num_select' in params and params['num_select'] != params['top_k']:
                raise ValueError(f'{context}: conflicting top_k and num_select')
            params['num_select'] = params.pop('top_k')
        integer('num_select', 2, high=integer('num_experts', 8, high=256))
        if params.get('expert_type', 'mlp') not in ('mlp', 'swiglu'):
            raise ValueError(f'{context}: expert_type must be mlp or swiglu')
    for key in ('num_experts', 'num_select', 'top_k', 'intermediate_size', 'max_position_embeddings'):
        if key in params:
            integer(key, 1)
    if 'attention' in kind or 'llama_block' in kind:
        params.setdefault('num_heads', 8)
        heads = integer('num_heads', 8)
        if width % heads:
            raise ValueError(f'{context}: num_heads must divide input width')
        kv = integer('num_kv_heads', heads)
        if heads % kv:
            raise ValueError(f'{context}: num_kv_heads must divide num_heads')
        if kind == 'attention' and kv != heads:
            raise ValueError(f'{context}: GQA uses causal_self_attention or llama_block')
        if (params.get('use_rope', False) or 'llama_block' in kind) and (width // heads) % 2:
            raise ValueError(f'{context}: RoPE needs an even head width')
        if number('rope_theta', 10000) <= 0:
            raise ValueError(f'{context}: rope_theta must be positive')
    if kind in ('custom_op', 'inline_code'):
        key = 'path' if kind == 'custom_op' else 'code'
        if not isinstance(params.get(key), str) or not params[key].strip():
            raise ValueError(f'{context}: {key} must be a nonempty string')
        if kind == 'custom_op' and not isinstance(params.get('kwargs', {}), dict):
            raise ValueError(f'{context}: kwargs must be an object')
        return integer('out_features', width)
    if kind == 'permute' and (not isinstance(params.get('dims'), list) or any(isinstance(v, bool) or not isinstance(v, int) for v in params['dims']) or sorted(params['dims']) != [0, 1, 2]):
        raise ValueError(f'{context}: dims must be a permutation of [0, 1, 2]')
    if kind == 'adaptive_avg_pool':
        integer('output_size', 1)
    if kind == 'linear':
        if integer('in_features', width) != width:
            raise ValueError(f'{context}: in_features does not match input width')
        return integer('out_features', width)
    if kind == 'causal_conv1d':
        output = integer('out_channels', width)
        groups = integer('groups', 1)
        if width % groups or output % groups:
            raise ValueError(f'{context}: groups must divide input and output channels')
        integer('kernel_size', 3); integer('dilation', 1)
        return output
    if kind == 'groupnorm' and width % integer('num_groups', 32):
        raise ValueError(f'{context}: num_groups must divide hidden width')
    if kind in ('activation', 'mlp', 'residual_mlp') and params.get('activation', params.get('type', 'silu')) not in ('silu', 'gelu', 'relu', 'tanh', 'sigmoid', 'leaky_relu'):
        raise ValueError(f'{context}: unsupported activation')
    return width


class Block(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    id: str = Field(min_length=1, pattern=r'^[A-Za-z0-9_-]+$')
    type: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)


class Blueprint(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    model_type: Literal['homellm_blueprint'] = 'homellm_blueprint'
    vocab_size: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    max_position_embeddings: int = Field(default=2048, gt=0)
    blocks: list[Block] = Field(min_length=1, max_length=256)
    auto_project: bool = True
    dtype: str | None = None
    training: dict[str, Any] | None = None

    @model_validator(mode='after')
    def graph(self):
        seen, dimensions, edges = set(), {'input': None}, {}
        previous = 'input'
        for block in self.blocks:
            if block.id == 'input' or block.id in seen:
                raise ValueError(f'Duplicate or reserved block ID: {block.id}')
            parents = block.inputs or [previous]
            if len(parents) != len(set(parents)):
                raise ValueError(f'{block.id}: duplicate inputs')
            if any(parent not in dimensions for parent in parents):
                raise ValueError(f'{block.id}: inputs must reference earlier blocks or input; cycles are not allowed')
            params = block.params
            for key in ('hidden_size', 'vocab_size', 'max_position_embeddings'):
                if key in params and (isinstance(params[key], bool) or not isinstance(params[key], int) or params[key] <= 0):
                    raise ValueError(f'{block.id}: {key} must be a positive integer')
            dim = dimensions[parents[0]]
            if block.type in ('token_embedding', 'positional_embedding'):
                if block.type == 'token_embedding':
                    if dim is not None:
                        raise ValueError(f'{block.id}: token_embedding requires raw input IDs')
                    if params.get('vocab_size', self.vocab_size) != self.vocab_size:
                        raise ValueError(f'{block.id}: vocab_size must match the blueprint')
                width = params.get('hidden_size', self.hidden_size)
                if block.type == 'positional_embedding':
                    if dim is not None and dim != width:
                        raise ValueError(f'{block.id}: positional embedding width must match its input')
                    if params.get('max_position_embeddings', self.max_position_embeddings) < self.max_position_embeddings:
                        raise ValueError(f'{block.id}: position table is smaller than the blueprint context length')
                dim = width
            elif dim is None:
                raise ValueError(f'{block.id}: add a token_embedding before this block')
            if block.type in ('add', 'multiply'):
                if len(parents) < 2 or (block.type == 'multiply' and len(parents) != 2) or any(dimensions[parent] != dim for parent in parents):
                    raise ValueError(f'{block.id}: {block.type} needs equal-width branches (multiply: exactly two)')
            elif block.type == 'concat':
                if len(parents) < 2 or params.get('dim', -1) != -1 or any(dimensions[parent] is None for parent in parents):
                    raise ValueError(f'{block.id}: concat needs at least two hidden-state inputs and dim=-1')
                dim = sum(dimensions[parent] for parent in parents)
            elif len(parents) > 1:
                raise ValueError(f'{block.id}: multiple inputs require add, multiply, or concat')
            if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                raise ValueError(f'{block.id}: hidden width must be a positive integer')
            dim = hidden_block(block.type, params, dim, block.id)
            seen.add(block.id)
            edges[block.id] = parents
            dimensions[block.id] = dim
            previous = block.id
        needed, pending = set(), [previous]
        while pending:
            node = pending.pop()
            if node in needed or node == 'input':
                continue
            needed.add(node)
            pending.extend(edges[node])
        if seen - needed:
            raise ValueError(f'Blocks do not contribute to the output: {sorted(seen - needed)}')
        if execution_summary(self.model_dump())[0] > 8192:
            raise ValueError('Architecture exceeds 8192 block applications per forward pass')
        return self


def execution_summary(value):
    loops = []
    def cost(kind, params, path):
        if kind in ('loop', 'residual'):
            inner = sum(cost(child['type'], child.get('params', {}), path + f'.body[{i}]') for i, child in enumerate(params['body']))
            if kind == 'loop':
                count = max(params.get('iterations', 4), params.get('eval_iterations', params.get('iterations', 4)))
                loops.append({'id': path, 'iterations': params.get('iterations', 4), 'eval_iterations': params.get('eval_iterations', params.get('iterations', 4)), 'body_blocks': len(params['body']), 'gradient_steps': params.get('gradient_steps', 0), 'input_injection': params.get('input_injection', 'none')})
                return 1 + count * inner
            return 1 + inner
        if kind == 'repeater':
            return 1 + params.get('num_repeats', 1) * cost(params['block_type'], params.get('block_params', {}), path + '.block')
        return 1
    total = sum(cost(b['type'], b.get('params', {}), b['id']) for b in value['blocks'])
    return total, loops


def validate(value: dict) -> dict:
    import json
    json.dumps(value, allow_nan=False)
    return Blueprint.model_validate(json.loads(json.dumps(value, allow_nan=False))).model_dump(exclude_none=True)


def with_tokenizer(value: dict, vocab_size: int) -> dict:
    """Synchronize explicit embedding sizes as well as the top-level vocabulary."""
    import copy
    value = copy.deepcopy(value)
    value['vocab_size'] = vocab_size
    for block in value['blocks']:
        if block['type'] == 'token_embedding':
            block.setdefault('params', {})['vocab_size'] = vocab_size
    return validate(value)
