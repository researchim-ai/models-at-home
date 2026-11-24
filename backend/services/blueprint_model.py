"""HF compatibility fixes for the installed runtime's blueprint model."""
from homellm.models.blueprint_model import BlueprintLMConfig, BlueprintForCausalLM
import torch
from torch import nn


class StudioPositionalEmbedding(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embed = embedding

    def forward(self, value, attention_mask=None):
        positions = torch.arange(value.shape[1], device=value.device).unsqueeze(0)
        encoded = self.embed(positions)
        # Raw input IDs request a position-only branch for an explicit Add.
        # In a sequence, positional encoding augments the token representation.
        return encoded if not value.is_floating_point() else value + encoded


class StudioBlueprintConfig(BlueprintLMConfig):
    def __init__(self, vocab_size=50257, hidden_size=512, **kwargs):
        # Transformers constructs a default config while serializing differences.
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size, **kwargs)


class StudioBlueprintForCausalLM(BlueprintForCausalLM):
    config_class = StudioBlueprintConfig
    _tied_weights_keys = ['lm_head.weight']
    _supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, 'studio_graph_version', 0) >= 1:
            widths = {}
            previous_width = config.hidden_size
            for block in self._blueprint.blocks:
                width = widths.get(block.inputs[0], previous_width) if block.inputs else previous_width
                module = self.model.modules_dict[block.id]
                if block.type == 'positional_embedding':
                    self.model.modules_dict[block.id] = StudioPositionalEmbedding(module.embed)
                if block.type in ('token_embedding', 'positional_embedding'):
                    width = block.params.get('hidden_size', config.hidden_size)
                if block.type == 'linear':
                    output = block.params.get('out_features', width)
                    self.model.modules_dict[block.id] = nn.Linear(width, output, bias=block.params.get('bias', False))
                    width = output
                widths[block.id] = previous_width = width

    def get_input_embeddings(self):
        for block in self._blueprint.blocks:
            if block.type == 'token_embedding':
                return self.model.modules_dict[block.id]
        raise ValueError('Blueprint has no token_embedding')

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._studio_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._studio_checkpointing = False

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if self.training and getattr(self, '_studio_checkpointing', False):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(super().forward, input_ids, attention_mask, labels, use_reentrant=False, **kwargs)
        return super().forward(input_ids, attention_mask, labels, **kwargs)
