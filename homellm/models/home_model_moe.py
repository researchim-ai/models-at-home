"""
homellm.models.home_model_moe
-----------------------------
HomeModel с Mixture of Experts (MoE) вместо обычного FFN.

Особенности:
• Базируется на HomeModel (LLaMA-style): RMSNorm, RoPE, SwiGLU experts
• MoE слой: N экспертов, Top-K routing
• Load balancing loss для равномерной нагрузки на экспертов
• Sparse computation: только K экспертов активируются для каждого токена

Формула параметров:
params = embed + layers * (attn + num_experts * mlp + norms + gate) + final_norm
где gate = hidden_size * num_experts
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

# Импортируем компоненты из home_model
from .home_model import (
    RMSNorm,
    create_rmsnorm,
    precompute_freqs_cis,
    apply_rope,
    Attention,
    _get_liger_rmsnorm,
    _get_liger_silumul,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


class HomeMoEConfig(PretrainedConfig):
    """Конфигурация для HomeModel с MoE."""
    model_type = "homellm_moe"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        intermediate_size: Optional[int] = None,
        # MoE specific
        num_experts: int = 8,
        num_experts_per_tok: int = 2,  # Top-K
        expert_type: str = "swiglu",  # "swiglu" or "mlp"
        aux_loss_coef: float = 0.01,  # Коэффициент для load balancing loss
        # Common
        dropout: float = 0.0,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1e4,
        use_sdpa: bool = True,
        use_liger: bool = True,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else int(hidden_size * 4)
        )
        # MoE
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_type = expert_type
        self.aux_loss_coef = aux_loss_coef
        # Common
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_sdpa = bool(use_sdpa)
        self.use_liger = bool(use_liger)


# -----------------------------------------------------------------------------
# MoE Components
# -----------------------------------------------------------------------------


class SwiGLUExpert(nn.Module):
    """
    Один SwiGLU эксперт (как в Mixtral).
    
    Структура: gate(H->I) * silu(up(H->I)) -> down(I->H)
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0, use_liger: bool = True):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Liger optimization
        self._liger_silumul = _get_liger_silumul() if use_liger else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        
        if self._liger_silumul is not None:
            hidden = self._liger_silumul.apply(gate, up)
        else:
            hidden = self.act(gate) * up
        
        return self.dropout(self.w2(hidden))


class MLPExpert(nn.Module):
    """
    Простой MLP эксперт.
    
    Структура: fc1(H->I) -> SiLU -> fc2(I->H)
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class MoELayer(nn.Module):
    """
    Mixture of Experts layer с Top-K routing и load balancing loss.
    
    Особенности:
    - Top-K: каждый токен маршрутизируется к K экспертам
    - Aux loss: штраф за неравномерное распределение токенов
    - Sparse: только активные эксперты вычисляются
    """
    def __init__(self, config: HomeMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.aux_loss_coef = config.aux_loss_coef
        
        # Router (gate)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Create experts
        experts = []
        for _ in range(config.num_experts):
            if config.expert_type == "swiglu":
                experts.append(SwiGLUExpert(
                    config.hidden_size, 
                    config.intermediate_size, 
                    config.dropout,
                    config.use_liger
                ))
            else:
                experts.append(MLPExpert(
                    config.hidden_size, 
                    config.intermediate_size, 
                    config.dropout
                ))
        self.experts = nn.ModuleList(experts)
        
        # For tracking aux loss
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, hidden)
        Returns:
            output: (batch, seq, hidden)
        """
        batch, seq, hidden = x.shape
        x_flat = x.view(-1, hidden)  # (batch*seq, hidden)
        
        # Router logits
        logits = self.gate(x_flat)  # (batch*seq, num_experts)
        probs = F.softmax(logits, dim=-1)
        
        # Top-K selection
        topk_probs, topk_indices = torch.topk(probs, self.num_experts_per_tok, dim=-1)
        
        # --- Load Balancing Loss (Switch Transformer style) ---
        # P_i: average probability mass per expert
        prob_mass_per_expert = probs.mean(dim=0)
        
        # f_i: fraction of tokens routed to each expert
        flat_indices = topk_indices.view(-1)
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        expert_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
        fraction_routed = expert_counts / (batch * seq * self.num_experts_per_tok)
        
        # Aux loss: num_experts * sum(f_i * P_i)
        self.aux_loss = self.num_experts * torch.sum(fraction_routed * prob_mass_per_expert)
        
        # Normalize top-k weights
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Expert routing
        out = torch.zeros_like(x_flat)
        
        for k in range(self.num_experts_per_tok):
            indices = topk_indices[:, k]  # (batch*seq,)
            scores = topk_probs[:, k, None]  # (batch*seq, 1)
            
            for expert_idx in range(self.num_experts):
                mask = (indices == expert_idx)
                if mask.any():
                    selected_x = x_flat[mask]
                    expert_out = self.experts[expert_idx](selected_x)
                    out[mask] += expert_out * scores[mask]
        
        return out.view(batch, seq, hidden)


class HomeMoEBlock(nn.Module):
    """
    Один блок HomeModel MoE.
    
    Pre-norm архитектура: RMSNorm -> Attention -> Add -> RMSNorm -> MoE -> Add
    """
    def __init__(self, config: HomeMoEConfig):
        super().__init__()
        use_liger = getattr(config, 'use_liger', True)
        self.attn_norm = create_rmsnorm(config.hidden_size, use_liger=use_liger)
        self.ffn_norm = create_rmsnorm(config.hidden_size, use_liger=use_liger)
        self.attn = Attention(config)
        self.moe = MoELayer(config)

    def forward(
        self, 
        x: torch.Tensor, 
        cos_sin: Tuple[torch.Tensor, torch.Tensor], 
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ):
        attn_out, present = self.attn(
            self.attn_norm(x), cos_sin, position_ids, past_key_value, use_cache
        )
        x = x + attn_out
        x = x + self.moe(self.ffn_norm(x))
        return x, present


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class HomeMoEModel(PreTrainedModel):
    """
    HomeModel backbone с MoE.
    """
    config_class = HomeMoEConfig
    base_model_prefix = "model"
    
    # vLLM compatibility attributes
    tp_plan = {}  # Tensor parallelism plan (empty for single GPU)
    _supports_attention_backend = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    
    def __init__(self, config: HomeMoEConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HomeMoEBlock(config) for _ in range(config.num_hidden_layers)])
        use_liger = getattr(config, 'use_liger', True)
        self.final_norm = create_rmsnorm(config.hidden_size, use_liger=use_liger)
        self.gradient_checkpointing = False

        # Precompute RoPE
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        
        # Log MoE configuration
        logger.info(
            f"🔀 HomeMoEModel: {config.num_experts} экспертов, "
            f"Top-{config.num_experts_per_tok} routing, "
            f"expert_type={config.expert_type}"
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        # vLLM compatibility arguments
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_instances: Optional[dict] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], float]:
        # Use inputs_embeds if provided, otherwise use input_ids
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
            seq_len = inputs_embeds.shape[1]
            device = inputs_embeds.device
        else:
            hidden_states = self.embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
            device = input_ids.device
        
        if position_ids is None:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                pk, _ = past_key_values[0]
                past_len = int(pk.size(2))
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                device=device
            ).unsqueeze(0)

        presents = [] if use_cache else None
        # Ensure RoPE buffers are on the same device as hidden_states
        cos_sin = (
            self.rope_cos.to(hidden_states.device),
            self.rope_sin.to(hidden_states.device),
        )
        
        # Accumulate aux loss from all MoE layers
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training and not use_cache:
                def make_ckpt_fn(block, cs):
                    def fn(hidden, pos_ids):
                        out, _ = block(hidden, cs, pos_ids, None, False)
                        return out
                    return fn
                
                ckpt_forward = make_ckpt_fn(layer, cos_sin)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    ckpt_forward,
                    hidden_states,
                    position_ids,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = layer(
                    hidden_states,
                    cos_sin,
                    position_ids,
                    past,
                    use_cache,
                )
            
            # Collect aux loss from MoE layer
            if hasattr(layer, 'moe'):
                total_aux_loss += layer.moe.aux_loss
            
            if use_cache:
                presents.append(present)

        hidden_states = self.final_norm(hidden_states)
        return hidden_states, presents, total_aux_loss


class HomeMoEForCausalLM(PreTrainedModel, GenerationMixin):
    """
    HomeModel MoE для causal language modeling.
    
    Aux loss добавляется к основному loss с коэффициентом aux_loss_coef.
    """
    config_class = HomeMoEConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    _supports_gradient_checkpointing = True
    # Атрибуты для совместимости с vLLM TransformersForCausalLM
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_attention_backend = True

    def __init__(self, config: HomeMoEConfig):
        super().__init__(config)
        self.model = HomeMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Включает gradient checkpointing."""
        self.model.gradient_checkpointing = True
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        def ckpt_func(fn, *args):
            return torch.utils.checkpoint.checkpoint(fn, *args, **gradient_checkpointing_kwargs)
        
        self.model._gradient_checkpointing_func = ckpt_func

    def gradient_checkpointing_disable(self):
        """Выключает gradient checkpointing."""
        self.model.gradient_checkpointing = False
        if hasattr(self.model, '_gradient_checkpointing_func'):
            delattr(self.model, '_gradient_checkpointing_func')

    def tie_weights(self):
        """Привязать веса lm_head к embed_tokens."""
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Convert DynamicCache to legacy format if needed
        legacy_past = None
        if past_key_values is not None:
            if hasattr(past_key_values, 'to_legacy_cache'):
                legacy_past = past_key_values.to_legacy_cache()
            elif isinstance(past_key_values, (list, tuple)):
                legacy_past = past_key_values
        
        hidden_states, presents, aux_loss = self.model(
            input_ids,
            past_key_values=legacy_past,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Add auxiliary loss for load balancing
            loss = lm_loss + self.config.aux_loss_coef * aux_loss

        all_hidden_states = None
        if output_hidden_states:
            all_hidden_states = (hidden_states,)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )
