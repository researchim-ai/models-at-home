"""Blueprint builder: turns a Blueprint into a torch.nn.Module."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from homellm.models.blueprint import Blueprint
from homellm.models.blocks import get_block


class BlueprintGraphModule(nn.Module):
    """Executes blocks as a DAG based on input dependencies."""

    def __init__(self, modules: Dict[str, nn.Module], inputs_map: Dict[str, List[str]], ordering: List[str], block_types=None):
        super().__init__()
        self.modules_dict = nn.ModuleDict(modules)
        self.inputs_map = inputs_map
        self.ordering = ordering
        self.block_types = block_types or {}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, loop_iterations: int | None = None, inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        # State holds outputs of all blocks: {block_id: tensor}
        # Initial state has 'input' pointing to input_ids (or we could have a dedicated InputBlock)
        state = {}
        
        # Typically the first block is embedding, which takes raw input_ids.
        # We need a convention. Let's say if a block has NO inputs specified, 
        # it takes the output of the PREVIOUS block in ordering.
        # If it's the very first block, it takes input_ids.
        
        last_output = input_ids
        
        for block_id in self.ordering:
            module = self.modules_dict[block_id]
            input_keys = self.inputs_map.get(block_id, [])
            
            # Determine input tensor(s) for this block
            if not input_keys:
                # Sequential mode fallback: take last output
                args = [last_output]
            else:
                # Graph mode: resolve inputs from state
                args = []
                for key in input_keys:
                    if key in state:
                        args.append(state[key])
                    else:
                        # Fallback for 'input' keyword if user explicitly wants raw input
                        if key == "input":
                            args.append(input_ids)
                        else:
                            raise ValueError(f"Block {block_id} depends on {key}, which has not been computed yet.")
            
            # Execute module
            # Logic similar to Sequential: check if module accepts inputs/mask
            if hasattr(module, "forward"):
                module_cls = module.__class__.__name__
                
                if inputs_embeds is not None and self.block_types.get(block_id) == 'token_embedding':
                    if inputs_embeds.shape[:2] != input_ids.shape or inputs_embeds.shape[-1] != module.embedding_dim:
                        raise ValueError('inputs_embeds does not match the blueprint token embedding shape')
                    out = inputs_embeds
                elif module_cls == "PositionalEmbedding":
                    out = module(input_ids=input_ids, attention_mask=attention_mask)
                elif module_cls in ("Add", "Multiply", "Concat"):
                    if len(args) < 2:
                        # Try to find 2 previous if sequential
                        # But in graph mode, args are explicit.
                        if len(args) == 1:
                             # Maybe user provided 1 input, and we need another?
                             # For now strict: Add needs 2 inputs in graph mode
                             raise ValueError(f"Block {block_id} (Add) requires 2 inputs, got {len(args)}")
                    
                    # Explicit sum to ensure broadcasting works as expected
                    # Just calling module(*args) should work, but let's be safe
                    # module(a, b) calls forward(a, b) -> sum((a, b))
                    out = module(*args)
                    
                elif module_cls == "LoopedBlock":
                    out = module(args[0], attention_mask=attention_mask, loop_iterations=loop_iterations)
                elif module_cls == "InlineCodeOp":
                    # Pass context
                    # If sequential (args has 1 item), use it.
                    x_val = args[0] if args else input_ids
                    out = module(x_val, input_ids=input_ids, attention_mask=attention_mask)
                else:
                    # Standard block (Attn, MLP, etc)
                    # Typically takes 1 input + optional mask
                    x_val = args[0]
                    try:
                        out = module(x_val, attention_mask=attention_mask)
                    except TypeError:
                         try:
                             out = module(x_val, attention_mask)
                         except TypeError:
                             out = module(x_val)
            else:
                 # Simple functional module
                 out = module(args[0])
            
            # Save state
            state[block_id] = out
            last_output = out
            
        return last_output


def build_model_from_blueprint(bp: Blueprint) -> Tuple[nn.Module, int]:
    """Builds a torch.nn.Module and returns (module, hidden_size)."""
    modules = {}
    inputs_map = {}
    ordering = []
    
    hidden_sizes = {} # block_id -> out_dim
    # We need to track dimensions to support auto_project in graph mode.
    # This is tricky because we need to know input dim to build the block.
    # Sequential assumption simplifies this: current_dim flows through.
    # For full graph support, we would need a topological dry run.
    
    # SIMPLIFICATION:
    # We still build blocks in the order defined in Blueprint.
    # We assume 'current_dim' logic follows the sequential definition order for defaults,
    # UNLESS we explicitly look up dependencies.
    
    # For robust graph building, we should ideally resolve dimensions.
    # But for MVP, let's assume the user (or UI) orders blocks topologically
    # and we use the 'last defined' dimension or explicit logic.
    
    current_dim = bp.hidden_size
    
    for block in bp.blocks:
        builder = get_block(block.type)
        params: Dict[str, Any] = dict(block.params)
        
        # Resolve input dimension for this block
        # If inputs are specified, try to get dim from the FIRST input
        in_dim = current_dim
        if block.inputs:
            first_parent = block.inputs[0]
            if first_parent in hidden_sizes:
                in_dim = hidden_sizes[first_parent]
        
        # Inject defaults
        if block.type == "token_embedding":
            params.setdefault("vocab_size", bp.vocab_size)
            params.setdefault("hidden_size", bp.hidden_size)
        if block.type == "positional_embedding":
            params.setdefault("hidden_size", bp.hidden_size)
            params.setdefault("max_position_embeddings", bp.max_position_embeddings)
            
        # Build
        module, out_dim = builder(params, in_dim, bp.auto_project)
        
        if block.type == "concat":
            out_dim = sum(hidden_sizes[parent] for parent in block.inputs)
        modules[block.id] = module
        inputs_map[block.id] = block.inputs
        ordering.append(block.id)
        
        hidden_sizes[block.id] = out_dim
        current_dim = out_dim

    model = BlueprintGraphModule(modules, inputs_map, ordering, {block.id: block.type for block in bp.blocks})
    return model, current_dim
