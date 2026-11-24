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
        vocab_size: int = 50257,
        hidden_size: int = 512,
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
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BlueprintLMConfig):
        super().__init__(config)
        if not config.blueprint:
            raise ValueError("BlueprintForCausalLM requires config.blueprint to be set")
        self._blueprint = Blueprint.parse_obj(config.blueprint)
        self.model, hidden_size = build_model_from_blueprint(self._blueprint)
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)
        
        self.post_init()

    def tie_weights(self):
        # An explicit final projection may change width; only compatible heads tie.
        if self.config.tie_word_embeddings:
            embedding = self.get_input_embeddings()
            if self.lm_head.weight.shape == embedding.weight.shape:
                self.lm_head.weight = embedding.weight

    def get_input_embeddings(self):
        for block in self._blueprint.blocks:
            if block.type == 'token_embedding':
                return self.model.modules_dict[block.id]
        raise ValueError('Blueprint has no token_embedding')

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loop_iterations: Optional[int] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask, loop_iterations=loop_iterations)
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
            from homellm.models.blocks.recurrent import auxiliary_losses
            losses = auxiliary_losses(self.model)
            if losses:
                loss = loss + 0.01 * torch.stack(losses).sum()

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)


