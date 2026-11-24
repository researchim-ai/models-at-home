"""HF-compatible wrapper around blueprint-built models."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from homellm.models.blueprint import Blueprint
from homellm.models.builder import build_model_from_blueprint


class BlueprintLMConfig(PretrainedConfig):
    model_type = "homellm_blueprint"

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 2048,
        auto_project: bool = True,
        blueprint: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.auto_project = auto_project
        self.blueprint = blueprint or {}


class BlueprintForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = BlueprintLMConfig
    base_model_prefix = "blueprint_model"

    def __init__(self, config: BlueprintLMConfig):
        super().__init__(config)
        if not config.blueprint:
            raise ValueError("BlueprintForCausalLM requires config.blueprint to be set")
        self._blueprint = Blueprint.parse_obj(config.blueprint)
        self.model, hidden_size = build_model_from_blueprint(self._blueprint)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if requested (default for causal LM)
        if config.tie_word_embeddings:
            # Try to find the token embedding layer in the built model
            # We look for the first module that is an instance of nn.Embedding
            embedding_layer = None
            for module in self.model.modules():
                if isinstance(module, nn.Embedding) and module.num_embeddings == config.vocab_size:
                    embedding_layer = module
                    break
            
            if embedding_layer is not None:
                if self.lm_head.weight.shape == embedding_layer.weight.shape:
                    self.lm_head.weight = embedding_layer.weight
                else:
                    # Dimensions might mismatch if there is a projection or different hidden size
                    pass

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Add auxiliary loss from MoE layers (load balancing)
            aux_loss = 0.0
            for module in self.model.modules():
                if hasattr(module, "aux_loss") and isinstance(module.aux_loss, torch.Tensor):
                    aux_loss += module.aux_loss
            
            # Standard coefficient for load balancing loss is usually 0.01-0.02
            # We can make it configurable later, but for now 0.01 is a safe research default.
            if isinstance(aux_loss, torch.Tensor) or aux_loss > 0:
                loss += 0.01 * aux_loss

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)


