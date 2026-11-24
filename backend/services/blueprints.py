"""Validate graph structure without importing Torch or executing custom blocks."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    blocks: list[Block] = Field(min_length=1)
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
            if block.type == 'moe' and 'top_k' in params:
                if 'num_select' in params and params['num_select'] != params['top_k']:
                    raise ValueError(f'{block.id}: conflicting top_k and num_select')
                params['num_select'] = params.pop('top_k')
            for key in ('num_experts', 'num_select', 'top_k', 'intermediate_size', 'max_position_embeddings'):
                if key in params and (not isinstance(params[key], int) or isinstance(params[key], bool) or params[key] <= 0):
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
            if block.type == 'add':
                if len(parents) < 2 or any(dimensions[parent] != dim for parent in parents):
                    raise ValueError(f'{block.id}: add requires at least two inputs of the same width')
            elif len(parents) > 1:
                # The installed runtime consumes only args[0] for these blocks.
                raise ValueError(f'{block.id}: this runtime supports multiple inputs only for add')
            if block.type == 'linear':
                if params.get('in_features', dim) != dim:
                    raise ValueError(f'{block.id}: in_features does not match its input width')
                dim = params.get('out_features', dim)
            if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                raise ValueError(f'{block.id}: hidden width must be a positive integer')
            if 'num_heads' in params:
                heads = params['num_heads']
                if not isinstance(heads, int) or isinstance(heads, bool) or heads <= 0 or dim % heads:
                    raise ValueError(f'{block.id}: num_heads must divide the input width')
                if params.get('use_rope') and (dim // heads) % 2:
                    raise ValueError(f'{block.id}: RoPE requires an even head dimension')
            if block.type == 'moe' and not 1 <= params.get('num_select', 2) <= params.get('num_experts', 8):
                raise ValueError(f'{block.id}: num_select must be between 1 and num_experts')
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
        return self


def validate(value: dict) -> dict:
    import json
    json.dumps(value, allow_nan=False)
    return Blueprint.model_validate(value).model_dump(exclude_none=True)


def with_tokenizer(value: dict, vocab_size: int) -> dict:
    """Synchronize explicit embedding sizes as well as the top-level vocabulary."""
    import copy
    value = copy.deepcopy(value)
    value['vocab_size'] = vocab_size
    for block in value['blocks']:
        if block['type'] == 'token_embedding':
            block.setdefault('params', {})['vocab_size'] = vocab_size
    return validate(value)
