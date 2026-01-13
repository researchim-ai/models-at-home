"""
homellm.models.home_model
-------------------------
Небольшая низкоуровневая реализация Transformer-языковой модели в стиле MiniMind,
но упрощённая и без внешних зависимостей, кроме PyTorch / HF-Transformers.

Особенности:
• RMSNorm вместо LayerNorm;
• Rotary Positional Embedding (RoPE);
• Свёртка QK-KV на модуль FlashAttention (если доступен) или обычное scaled-dot-prod;
• Feed-Forward на SiLU (GELU-подобная);
• Возможность выбора числа голов, слоёв, hidden_size через HomeConfig.

Поддерживаются методы generate (через GenerationMixin).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


class HomeConfig(PretrainedConfig):
    model_type = "homellm"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1e4,
        use_sdpa: bool = True,
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
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_sdpa = bool(use_sdpa)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float):
    """Вычисляет cos/sin для RoPE один раз и кэширует."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    emb = torch.cat([freqs_cos, freqs_cos], dim=-1), torch.cat([freqs_sin, freqs_sin], dim=-1)
    return emb


def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


def apply_rope(q, k, cos, sin, position_ids):
    cos = cos[position_ids][:, None, :, :]
    sin = sin[position_ids][:, None, :, :]
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config: HomeConfig):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.flash = bool(getattr(config, "use_sdpa", True)) and hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = cos_sin
        q, k = apply_rope(q, k, cos, sin, position_ids)

        # Append past
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v) if use_cache else None

        kv_len = k.size(2)  # Полная длина KV (с учётом past_key_value)
        
        if self.flash:
            # PyTorch SDPA сам выберет лучший backend (flash/mem_efficient/math) при fp16/bf16.
            # Важно: для генерации с KV-кэшем обычно seq_len=1, и мы можем безопасно считать attention без causal-маски:
            # в k/v уже только "прошлое", будущих токенов там нет.
            if past_key_value is None:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True,
                )
            elif seq_len == 1:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=0.0,
                    is_causal=False,
                )
            else:
                # Редкий случай: past_key_value есть, но seq_len > 1 (chunked decode) — оставляем безопасный путь ниже.
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                causal_mask = torch.triu(
                    torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
                    diagonal=kv_len - seq_len + 1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, -1e4)
                attn_probs = F.softmax(attn_weights, dim=-1)
                attn_probs = self.dropout(attn_probs)
                output = torch.matmul(attn_probs, v)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # Создаём каузальную маску правильного размера (seq_len × kv_len)
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
                diagonal=kv_len - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, -1e4)
            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_probs = self.dropout(attn_probs)
            output = torch.matmul(attn_probs, v)

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.out_proj(output), present


class FeedForward(nn.Module):
    def __init__(self, config: HomeConfig):
        super().__init__()
        hidden = config.intermediate_size
        self.w1 = nn.Linear(config.hidden_size, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, hidden, bias=False)  # gated-linear
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x)) * self.w3(x)))


class HomeBlock(nn.Module):
    def __init__(self, config: HomeConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, x, cos_sin, position_ids, past_key_value=None, use_cache=False):
        attn_out, present = self.attn(
            self.attn_norm(x), cos_sin, position_ids, past_key_value, use_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.ffn_norm(x))
        return x, present


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class HomeModel(nn.Module):
    def __init__(self, config: HomeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([HomeBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.hidden_size)
        self.gradient_checkpointing = False  # Флаг для checkpointing

        # Precompute RoPE
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        if position_ids is None:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                pk, _ = past_key_values[0]
                # pk: [bs, heads, past_len, head_dim]
                past_len = int(pk.size(2))
            position_ids = torch.arange(
                past_len, past_len + input_ids.shape[1],
                device=input_ids.device
            ).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        
        presents = [] if use_cache else None
        cos_sin = (self.rope_cos, self.rope_sin)
        
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training and not use_cache:
                # Простая обёртка - передаём layer через замыкание с немедленным захватом
                def make_ckpt_fn(block, cs):
                    def fn(hidden, pos_ids):
                        out, _ = block(hidden, cs, pos_ids, None, False)
                        return out
                    return fn
                
                # Создаём функцию с захватом текущего layer
                ckpt_forward = make_ckpt_fn(layer, cos_sin)
                
                # Вызываем checkpoint
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
            
            if use_cache:
                presents.append(present)

        hidden_states = self.final_norm(hidden_states)
        return hidden_states, presents


class HomeForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = HomeConfig
    base_model_prefix = "home_model"
    _tied_weights_keys = ["lm_head.weight"]
    _supports_gradient_checkpointing = True  # Новый формат

    def __init__(self, config: HomeConfig):
        super().__init__(config)
        self.home_model = HomeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.home_model.embed_tokens.weight

        self.post_init()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Включает gradient checkpointing для модели."""
        # Устанавливаем флаг на HomeModel
        self.home_model.gradient_checkpointing = True
        # Сохраняем kwargs для checkpointing
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        def ckpt_func(fn, *args):
            return torch.utils.checkpoint.checkpoint(fn, *args, **gradient_checkpointing_kwargs)
        
        self.home_model._gradient_checkpointing_func = ckpt_func
    
    def gradient_checkpointing_disable(self):
        """Выключает gradient checkpointing."""
        self.home_model.gradient_checkpointing = False
        if hasattr(self.home_model, '_gradient_checkpointing_func'):
            delattr(self.home_model, '_gradient_checkpointing_func')
    
    def tie_weights(self):
        """Привязать веса lm_head к embed_tokens."""
        self.lm_head.weight = self.home_model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.home_model.embed_tokens

    def set_input_embeddings(self, value):
        self.home_model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        **kwargs,  # Для совместимости с новыми версиями transformers (cache_position, etc.)
    ) -> CausalLMOutputWithPast:
        # Конвертируем DynamicCache в список кортежей если нужно
        legacy_past = None
        if past_key_values is not None:
            if hasattr(past_key_values, 'to_legacy_cache'):
                # Это DynamicCache из новых transformers
                legacy_past = past_key_values.to_legacy_cache()
            elif isinstance(past_key_values, (list, tuple)):
                legacy_past = past_key_values
        
        hidden_states, presents = self.home_model(
            input_ids,
            past_key_values=legacy_past,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Сдвиг для предсказания следующего токена
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
        )
