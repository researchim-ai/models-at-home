"""
homellm.models.gpt2_model
-------------------------
Реализация классической GPT-2 архитектуры для сравнения с HomeModel.

Особенности GPT-2:
• LayerNorm (вместо RMSNorm)
• Learned positional embeddings (вместо RoPE)
• GELU MLP с 2 слоями (вместо SwiGLU с 3)
• Bias в проекциях attention и MLP
• Weight tying между embedding и lm_head

Это отличается от HomeModel (LLaMA-style):
- GPT-2 проще и имеет меньше параметров при тех же hidden_size/layers
- GPT-2 использует pre-norm (LayerNorm перед attention/MLP)
- GPT-2 MLP: fc1(H->4H) -> GELU -> fc2(4H->H)
- HomeModel MLP: gate(H->I) * silu(up(H->I)) -> down(I->H) (SwiGLU)
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

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


class GPT2HomeConfig(PretrainedConfig):
    """Конфигурация для GPT-2 модели."""
    model_type = "gpt2_home"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        max_position_embeddings: int = 1024,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # GPT-2 standard: intermediate_size = 4 * hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else int(hidden_size * 4)
        )
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


class GPT2Attention(nn.Module):
    """
    Multi-head self-attention в стиле GPT-2.
    
    Отличия от HomeModel/LLaMA:
    - Fused c_attn (QKV в одной проекции)
    - Bias во всех проекциях
    - Нет RoPE, позиции через learned embeddings
    """
    def __init__(self, config: GPT2HomeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        # Fused QKV projection (GPT-2 style)
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        # Output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        bsz, seq_len, _ = x.size()
        
        # Fused QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        
        # Reshape to (bsz, num_heads, seq_len, head_dim)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Append past KV cache
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v) if use_cache else None
        
        kv_len = k.size(2)
        
        # Attention with causal mask
        if hasattr(F, "scaled_dot_product_attention"):
            # Use PyTorch SDPA
            if past_key_value is None:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True,
                )
            elif seq_len == 1:
                # Generation: no causal mask needed, we're processing one token
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=False,
                )
            else:
                # Chunked decode: need explicit causal mask
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                causal_mask = torch.triu(
                    torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
                    diagonal=kv_len - seq_len + 1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, -1e4)
                attn_probs = F.softmax(attn_weights, dim=-1)
                attn_probs = self.attn_dropout(attn_probs)
                output = torch.matmul(attn_probs, v)
        else:
            # Fallback: manual attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
                diagonal=kv_len - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, -1e4)
            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            output = torch.matmul(attn_probs, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.c_proj(output))
        
        return output, present


class GPT2MLP(nn.Module):
    """
    GPT-2 MLP (FeedForward) block.
    
    Структура: fc1(H -> 4H) -> GELU -> fc2(4H -> H)
    
    Отличие от HomeModel SwiGLU:
    - 2 проекции вместо 3
    - GELU вместо SiLU
    - Bias во всех слоях
    """
    def __init__(self, config: GPT2HomeConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    """
    Один блок GPT-2 трансформера.
    
    Pre-norm архитектура: LayerNorm -> Attention -> Add -> LayerNorm -> MLP -> Add
    """
    def __init__(self, config: GPT2HomeConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GPT2Attention(config)
        self.mlp = GPT2MLP(config)

    def forward(
        self, 
        x: torch.Tensor, 
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        use_cache: bool = False
    ):
        # Pre-norm attention
        attn_out, present = self.attn(self.ln_1(x), past_key_value, use_cache)
        x = x + attn_out
        
        # Pre-norm MLP
        x = x + self.mlp(self.ln_2(x))
        
        return x, present


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class GPT2HomeModel(nn.Module):
    """
    GPT-2 backbone model (без lm_head).
    
    Ключевые отличия от HomeModel:
    - wte: token embeddings
    - wpe: position embeddings (learned, не RoPE!)
    - LayerNorm вместо RMSNorm
    """
    def __init__(self, config: GPT2HomeConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        if position_ids is None:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                pk, _ = past_key_values[0]
                past_len = int(pk.size(2))
            position_ids = torch.arange(
                past_len, past_len + seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(bsz, -1)
        
        # Token + Position embeddings
        token_emb = self.wte(input_ids)
        pos_emb = self.wpe(position_ids)
        hidden_states = self.drop(token_emb + pos_emb)
        
        presents = [] if use_cache else None
        
        for i, block in enumerate(self.h):
            past = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training and not use_cache:
                def make_ckpt_fn(b):
                    def fn(hidden):
                        out, _ = b(hidden, None, False)
                        return out
                    return fn
                
                ckpt_forward = make_ckpt_fn(block)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    ckpt_forward,
                    hidden_states,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = block(hidden_states, past, use_cache)
            
            if use_cache:
                presents.append(present)
        
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents


class GPT2HomeForCausalLM(PreTrainedModel, GenerationMixin):
    """
    GPT-2 для causal language modeling.
    
    Совместим с HuggingFace Transformers API.
    """
    config_class = GPT2HomeConfig
    base_model_prefix = "transformer"
    _tied_weights_keys = ["lm_head.weight"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: GPT2HomeConfig):
        super().__init__(config)
        self.transformer = GPT2HomeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.transformer.wte.weight
        
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Включает gradient checkpointing."""
        self.transformer.gradient_checkpointing = True
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        def ckpt_func(fn, *args):
            return torch.utils.checkpoint.checkpoint(fn, *args, **gradient_checkpointing_kwargs)
        
        self.transformer._gradient_checkpointing_func = ckpt_func

    def gradient_checkpointing_disable(self):
        """Выключает gradient checkpointing."""
        self.transformer.gradient_checkpointing = False
        if hasattr(self.transformer, '_gradient_checkpointing_func'):
            delattr(self.transformer, '_gradient_checkpointing_func')

    def tie_weights(self):
        """Привязать веса lm_head к wte."""
        self.lm_head.weight = self.transformer.wte.weight

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
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
        
        hidden_states, presents = self.transformer(
            input_ids,
            position_ids=position_ids,
            past_key_values=legacy_past,
            use_cache=use_cache,
        )
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        all_hidden_states = None
        if output_hidden_states:
            all_hidden_states = (hidden_states,)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )
