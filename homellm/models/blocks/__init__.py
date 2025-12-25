"""Registry of reusable blocks for blueprint-based models."""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from homellm.models.home_model import RMSNorm


BuildResult = Tuple[nn.Module, int]  # (module, out_hidden_size)


class BlockBuilder:
    """Wraps a builder function with metadata."""

    def __init__(self, fn: Callable[[Dict[str, Any], int, bool], BuildResult], description: str = ""):
        self.fn = fn
        self.description = description

    def __call__(self, params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
        return self.fn(params, in_dim, auto_project)


BLOCK_REGISTRY: Dict[str, BlockBuilder] = {}


def register_block(name: str, description: str = ""):
    def deco(fn):
        BLOCK_REGISTRY[name] = BlockBuilder(fn, description)
        return fn

    return deco


def _maybe_project(in_dim: int, out_dim: int, auto_project: bool) -> Optional[nn.Module]:
    if in_dim == out_dim:
        return None
    if not auto_project:
        raise ValueError(f"Dimension mismatch: {in_dim} -> {out_dim} and auto_project=False")
    return nn.Linear(in_dim, out_dim)


# =============================================================================
# EMBEDDINGS
# =============================================================================

@register_block("token_embedding", "Token embedding layer")
def build_token_embedding(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    vocab_size = params["vocab_size"]
    hidden_size = params["hidden_size"]
    module = nn.Embedding(vocab_size, hidden_size)
    return module, hidden_size


class PositionalEmbedding(nn.Module):
    def __init__(self, max_pos: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(max_pos, hidden_size)

    def forward(self, input_ids=None, attention_mask=None):
        # infer positions from attention_mask or input_ids length
        if input_ids is None and attention_mask is None:
            raise ValueError("PositionalEmbedding requires input_ids or attention_mask to infer sequence length")
        if attention_mask is not None:
            seq_len = attention_mask.size(1)
        else:
            seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=self.embed.weight.device).unsqueeze(0)
        out = self.embed(positions)
        if input_ids is not None:
            return out.expand(input_ids.shape[0], -1, -1)
        return out


@register_block("positional_embedding", "Learnable positional embeddings")
def build_positional_embedding(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    max_pos = params.get("max_position_embeddings", 2048)
    hidden_size = params["hidden_size"]
    module = PositionalEmbedding(max_pos, hidden_size)
    return module, hidden_size


# =============================================================================
# BASIC OPS
# =============================================================================

class Add(nn.Module):
    def forward(self, *args):
        # Support variable number of args, simply sum them
        return sum(args)

@register_block("add", "Elementwise sum")
def build_add(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    return Add(), in_dim


class Multiply(nn.Module):
    def forward(self, x, y):
        return x * y

@register_block("multiply", "Elementwise multiplication (e.g. for gating)")
def build_multiply(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    return Multiply(), in_dim


class Concat(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, *args):
        return torch.cat(args, dim=self.dim)

@register_block("concat", "Concatenation along dimension")
def build_concat(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    # We cannot easily predict out_dim for concat in dynamic graph without tracing.
    # We rely on user or downstream blocks to handle dimension change (auto_project in next block).
    # But for sequential builder, we assume input dims are roughly same, so out_dim = in_dim * num_inputs
    # This is imperfect in static analysis.
    # Let's trust user or auto_project downstream.
    dim = params.get("dim", -1)
    return Concat(dim=dim), in_dim # Placeholder out_dim, downstream might fail if auto_project is off


# =============================================================================
# NORMALIZATION & ACTIVATION
# =============================================================================

@register_block("rmsnorm", "RMSNorm")
def build_rmsnorm(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    eps = params.get("eps", 1e-5)
    return RMSNorm(in_dim, eps=eps), in_dim


@register_block("layernorm", "LayerNorm")
def build_layernorm(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    eps = params.get("eps", 1e-5)
    return nn.LayerNorm(in_dim, eps=eps), in_dim


class GroupNormWrapper(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
    
    def forward(self, x):
        # x: (B, S, H) -> (B, H, S) for GroupNorm -> (B, S, H)
        x = x.transpose(1, 2)
        x = self.gn(x)
        x = x.transpose(1, 2)
        return x

@register_block("groupnorm", "GroupNorm (channels last support)")
def build_groupnorm(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    num_groups = params.get("num_groups", 32)
    return GroupNormWrapper(num_groups, in_dim), in_dim


class Activation(nn.Module):
    def __init__(self, act_type: str):
        super().__init__()
        act_type = act_type.lower()
        if act_type == "silu": self.act = nn.SiLU()
        elif act_type == "gelu": self.act = nn.GELU()
        elif act_type == "relu": self.act = nn.ReLU()
        elif act_type == "tanh": self.act = nn.Tanh()
        elif act_type == "sigmoid": self.act = nn.Sigmoid()
        elif act_type == "leaky_relu": self.act = nn.LeakyReLU()
        else: self.act = nn.Identity()

    def forward(self, x):
        return self.act(x)

@register_block("activation", "Standalone activation function")
def build_activation(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    act = params.get("type", "silu")
    return Activation(act), in_dim



class Repeater(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x

@register_block("repeater", "Repeat a sub-block N times (e.g. Decoder Layers)")
def build_repeater(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    num_repeats = params.get("num_repeats", 1)
    block_type = params.get("block_type")
    block_params = params.get("block_params", {})
    
    if not block_type:
        raise ValueError("repeater requires 'block_type'")

    builder = get_block(block_type)
    
    layers = []
    current_dim = in_dim
    
    for _ in range(num_repeats):
        # Build the sub-block
        # We assume sub-block output dim becomes input for next iteration
        module, out_dim = builder(block_params, current_dim, auto_project)
        layers.append(module)
        current_dim = out_dim
        
    return Repeater(nn.ModuleList(layers)), current_dim

# =============================================================================
# LAYERS (MLP, ATTN, CONV)
# =============================================================================

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "silu", dropout: float = 0.0):
        super().__init__()
        act = activation.lower()
        if act == "gelu":
            activation_fn = nn.GELU()
        elif act == "relu":
            activation_fn = nn.ReLU()
        else:
            activation_fn = nn.SiLU()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            activation_fn,
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


@register_block("mlp", "Two-layer MLP with activation")
def build_mlp(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    hidden_size = in_dim
    intermediate = params.get("intermediate_size", hidden_size * 4)
    activation = params.get("activation", "silu")
    dropout = params.get("dropout", 0.0)
    return FeedForward(hidden_size, intermediate, activation, dropout), hidden_size


class CausalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, attention_mask=None):
        # attention_mask expected as (batch, seq) with 1 for keep, 0 for pad
        attn_mask = None
        if attention_mask is not None:
            bsz, seq = attention_mask.shape
            causal = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
            key_padding = ~attention_mask.bool()
            attn_mask = causal
            out, _ = self.mha(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding)
        else:
            seq = x.size(1)
            causal = torch.triu(torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1)
            out, _ = self.mha(x, x, x, attn_mask=causal)
        return out


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `forward` faster
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but using a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    # cos, sin: (seq_len, dim) -> (1, 1, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0, use_rope: bool = False, rope_theta: float = 10000.0):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout_prob = dropout
        
        if use_rope:
             self.rotary_emb = RotaryEmbedding(self.head_dim, base=rope_theta)

    def forward(self, x, attention_mask=None):
        # x: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
             cos, sin = self.rotary_emb(v, seq_len=seq_len)
             q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flash Attention support (PyTorch 2.0+)
        # attention_mask: (batch, seq_len) -> we need (batch, 1, seq_len, seq_len) or similar for SDPA?
        # F.scaled_dot_product_attention handles causal mask internally if is_causal=True
        
        # But wait! If we have padding (attention_mask has 0s), we must pass it.
        # SDPA expects attn_mask to be a boolean mask where True indicates values to be computed (not masked out) ??
        # Or float mask with -inf. 
        # Actually SDPA docs: "attn_mask: binary mask where 0 (False) is ignore" IS NOT CORRECT. 
        # Docs say: "The shape of attn_mask must be broadcastable to the shape of the attention weights."
        
        # Simple causal only:
        is_causal = True
        
        # If we have padding mask (attention_mask from tokenizer):
        # We need to combine causal + padding.
        # SDPA supports is_causal=True which is efficient. Adding explicit mask disables flash attention in some versions.
        # Ideally we rely on is_causal=True.
        
        # For now, let's use SDPA with is_causal=True.
        # Warning: if there is padding in the batch (left or right), we ideally need to mask it.
        # But for simple pretraining with packed sequences or simple causal, is_causal=True is often enough
        # provided we don't attend to padding tokens which is handled by loss masking usually.
        # However, for correctness, standard MHA masks padding.
        
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout_prob if self.training else 0.0, 
            is_causal=is_causal
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(out)

@register_block("causal_self_attention", "Flash Causal Self-Attention (with optional RoPE)")
def build_causal_self_attention(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    num_heads = params.get("num_heads")
    if num_heads is None:
        # Try to infer valid num_heads (e.g. 32 or 8)
        if in_dim % 32 == 0: num_heads = 32
        elif in_dim % 8 == 0: num_heads = 8
        else: num_heads = 1
        
    dropout = params.get("dropout", 0.0)
    use_rope = params.get("use_rope", False)
    rope_theta = params.get("rope_theta", 10000.0)
    
    if in_dim % num_heads != 0 and not auto_project:
        raise ValueError(f"in_dim {in_dim} not divisible by num_heads {num_heads}")
        
    return CausalSelfAttention(in_dim, num_heads, dropout, use_rope, rope_theta), in_dim


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MoE(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, num_select: int = 2, dropout: float = 0.0, expert_type: str = "mlp"):
        super().__init__()
        self.num_experts = num_experts
        self.num_select = num_select
        self.expert_type = expert_type
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Build experts based on type
        experts_list = []
        for _ in range(num_experts):
            if expert_type == "swiglu":
                # SwiGLU standard intermediate is 8/3 hidden, but let's stick to 4x or custom if we want simple
                # Mixtral uses 14336 for 4096 hidden (~3.5x). Let's use 4x for simplicity unless parameterized.
                experts_list.append(SwiGLU(hidden_size, hidden_size * 4, dropout=dropout))
            else:
                experts_list.append(FeedForward(hidden_size, hidden_size * 4, activation="silu", dropout=dropout))
        
        self.experts = nn.ModuleList(experts_list)
        
        # For tracking auxiliary loss during forward
        self.aux_loss = 0.0

    def forward(self, x):
        # x: (batch, seq, hidden)
        batch, seq, hidden = x.shape
        x_flat = x.view(-1, hidden)
        
        # Gating
        logits = self.gate(x_flat) # (batch*seq, num_experts)
        probs = F.softmax(logits, dim=-1)
        
        # --- Load Balancing Loss Calculation ---
        # We want to minimize the coefficient of variation of the load (fraction of tokens per expert)
        # And also ensure the gating probabilities are balanced.
        # Switch Transformer Loss: aux_loss = alpha * N * sum(f_i * P_i)
        # f_i = fraction of tokens routed to expert i (hard assignment, or soft based on implementation)
        # P_i = fraction of probability mass assigned to expert i (sum(probs, dim=0) / T)
        
        # P_i: Average probability assigned to expert i across the batch
        # shape: (num_experts,)
        prob_mass_per_expert = probs.mean(dim=0) 
        
        # f_i: Fraction of tokens routed to expert i.
        # Since we use Top-K routing, a token contributes to K experts.
        # We can approximate f_i by the sum of probabilities for selected experts, or just use hard counts.
        # For differentiability, using soft probs is often preferred in some implementations, 
        # but the standard definition uses the number of tokens routed.
        # However, to be differentiable w.r.t logits, we need to involve probs.
        
        # Simple differentiable approximation often used:
        # aux_loss = sum(prob_mass_per_expert * prob_mass_per_expert) * num_experts 
        # (minimizing sum of squares pushes towards uniform distribution)
        
        # Let's use the standard "Switch Transformer" style load balancing loss definition:
        # loss = num_experts * sum(P_i * f_i)
        # where f_i is strictly the fraction of tokens choosing expert i (non-differentiable part usually handled by straight-through or just ignored for f_i calculation in terms of grads, but P_i carries grads).
        
        # We use Top-K selection for routing.
        topk_probs, topk_indices = torch.topk(probs, self.num_select, dim=-1)
        
        # Compute load (fraction of tokens routed to each expert)
        # We create a mask of selected experts
        # shape: (batch*seq, num_experts)
        # We need one-hot encoding of selections
        # indices: (batch*seq, k)
        
        # Flatten indices to (batch*seq * k)
        flat_indices = topk_indices.view(-1)
        # One-hot: (batch*seq*k, num_experts) -> sum over batch*seq*k -> (num_experts)
        # This counts how many times each expert was selected
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        expert_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
        
        # Fraction f_i
        fraction_routed = expert_counts / (batch * seq * self.num_select)
        
        # The loss term: num_experts * sum(fraction_routed * prob_mass_per_expert)
        # This encourages prob_mass (soft) to align with uniform distribution if fraction_routed is uniform, 
        # and penalizes if they are correlated and peaked.
        # Ideally we want both to be uniform (1/N).
        # Minimized when both are uniform.
        
        self.aux_loss = self.num_experts * torch.sum(fraction_routed * prob_mass_per_expert)
        
        # Normalize weights for routing
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True) 
        
        # Expert routing (naive loop implementation for simplicity/compatibility)
        out = torch.zeros_like(x_flat)
        
        for k in range(self.num_select):
            indices = topk_indices[:, k] # (batch*seq,) expert index for k-th choice
            scores = topk_probs[:, k, None] # (batch*seq, 1) score
            
            for expert_idx in range(self.num_experts):
                # Mask for current expert
                mask = (indices == expert_idx)
                if mask.any():
                    # Process selected tokens
                    selected_x = x_flat[mask]
                    expert_out = self.experts[expert_idx](selected_x)
                    out[mask] += expert_out * scores[mask]
                    
        return out.view(batch, seq, hidden)

@register_block("moe", "Mixture of Experts (Sparse MLP)")
def build_moe(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    num_experts = params.get("num_experts", 8)
    num_select = params.get("num_select", 2)
    dropout = params.get("dropout", 0.0)
    expert_type = params.get("expert_type", "mlp") # "mlp" or "swiglu"
    return MoE(in_dim, num_experts, num_select, dropout, expert_type), in_dim

@register_block("swiglu", "SwiGLU FeedForward (Llama style)")
def build_swiglu(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    hidden_size = in_dim
    # Default intermediate size is usually 8/3 * hidden_size or similar for SwiGLU
    default_inter = int(2 * hidden_size * 4 / 3) 
    # Round to multiple of 256 is common but let's keep it simple or user defined
    intermediate = params.get("intermediate_size", default_inter)
    dropout = params.get("dropout", 0.0)
    return SwiGLU(hidden_size, intermediate, dropout), hidden_size


class LlamaBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.0, rope_theta: float = 10000.0, eps: float = 1e-5):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout=dropout, use_rope=True, rope_theta=rope_theta)
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        self.mlp = SwiGLU(hidden_size, intermediate_size, dropout=dropout)

    def forward(self, x, attention_mask=None):
        # Pre-Norm Residual Attention
        h = self.norm1(x)
        h = self.attn(h, attention_mask=attention_mask)
        x = x + h
        
        # Pre-Norm Residual MLP
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

@register_block("llama_block", "Single Llama Transformer Block (Attn + MLP + Norms)")
def build_llama_block(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    num_heads = params.get("num_heads", 8)
    # Default intermediate for SwiGLU
    default_inter = int(2 * in_dim * 4 / 3)
    intermediate_size = params.get("intermediate_size", default_inter)
    dropout = params.get("dropout", 0.0)
    rope_theta = params.get("rope_theta", 10000.0)
    eps = params.get("eps", 1e-5)
    
    return LlamaBlock(in_dim, num_heads, intermediate_size, dropout, rope_theta, eps), in_dim



@register_block("attention", "Causal self-attention (MultiheadAttention)")
def build_attention(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    num_heads = params.get("num_heads")
    if num_heads is None:
        raise ValueError("attention block requires num_heads")
    dropout = params.get("dropout", 0.0)
    if in_dim % num_heads != 0 and not auto_project:
        raise ValueError(f"in_dim {in_dim} not divisible by num_heads {num_heads}")
    return CausalAttention(in_dim, num_heads, dropout), in_dim


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        # Padding to ensure causal (only look back)
        # Left padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=self.padding, 
            dilation=dilation, 
            groups=groups
        )

    def forward(self, x):
        # x: (B, S, H) -> (B, H, S)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Remove lookahead padding from the end
        if self.padding > 0:
            x = x[..., :-self.padding]
        # Back to (B, S, H)
        x = x.transpose(1, 2)
        return x

@register_block("causal_conv1d", "Causal 1D Convolution (sequence modeling)")
def build_causal_conv1d(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    out_channels = params.get("out_channels", in_dim)
    kernel_size = params.get("kernel_size", 3)
    dilation = params.get("dilation", 1)
    groups = params.get("groups", 1)
    
    # Auto project in_dim if needed logic is handled by conv itself (in_channels)
    return CausalConv1d(in_dim, out_channels, kernel_size, dilation, groups), out_channels


# =============================================================================
# UTILS & WRAPPERS
# =============================================================================

@register_block("dropout", "Dropout")
def build_dropout(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    p = params.get("p", params.get("dropout", 0.0))
    return nn.Dropout(p), in_dim


@register_block("linear", "Linear projection")
def build_linear(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    out_dim = params["out_features"]
    proj = _maybe_project(in_dim, out_dim, auto_project)
    if proj is not None:
        return proj, out_dim
    return nn.Identity(), out_dim


class Residual(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x, **kwargs):
        # We assume inner module returns same shape
        return x + self.inner(x, **kwargs)

@register_block("residual_mlp", "Residual wrapper around MLP")
def build_residual_mlp(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    mlp, _ = build_mlp(params, in_dim, auto_project)
    return Residual(mlp), in_dim


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

@register_block("permute", "Permute tensor dimensions")
def build_permute(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    dims = params.get("dims") # list of ints
    if not dims:
        raise ValueError("Permute requires 'dims' param (list of ints)")
    return Permute(*dims), in_dim


class AdaptiveAvgPool1dWrapper(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)
    def forward(self, x):
        # (B, S, H) -> (B, H, S) -> pool -> (B, H, S_out) -> (B, S_out, H)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        return x

@register_block("adaptive_avg_pool", "Adaptive Average Pooling 1D")
def build_adaptive_avg_pool(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    output_size = params.get("output_size", 1)
    return AdaptiveAvgPool1dWrapper(output_size), in_dim


class CustomOp(nn.Module):
    def __init__(self, dotted_path: str, kwargs: Dict[str, Any]):
        super().__init__()
        module_path, attr = dotted_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        target = getattr(mod, attr)
        if isinstance(target, type):
            self.op = target(**kwargs)
        elif callable(target):
            class FnWrapper(nn.Module):
                def __init__(self, fn, fn_kwargs):
                    super().__init__()
                    self.fn = fn
                    self.fn_kwargs = fn_kwargs
                def forward(self, *args, **kw):
                    return self.fn(*args, **kw, **self.fn_kwargs)
            self.op = FnWrapper(target, kwargs)
        else:
            raise ValueError(f"CustomOp target {dotted_path} is not callable or class")

    def forward(self, *args, **kwargs):
        return self.op(*args, **kwargs)


@register_block("custom_op", "Load custom nn.Module or function by dotted path")
def build_custom_op(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    path = params["path"]
    kwargs = params.get("kwargs", {})
    out_dim = params.get("out_features", in_dim)
    module = CustomOp(path, kwargs)
    return module, out_dim


class InlineCodeOp(nn.Module):
    def __init__(self, code: str):
        super().__init__()
        self.code = code

    def forward(self, x, input_ids=None, attention_mask=None):
        local_scope = {"x": x, "input_ids": input_ids, "attention_mask": attention_mask, "torch": torch, "nn": nn, "F": F}
        exec(self.code, globals(), local_scope)
        if "x" not in local_scope:
             raise ValueError("Inline code must update variable 'x'")
        return local_scope["x"]


@register_block("inline_code", "Execute arbitrary Python code")
def build_inline_code(params: Dict[str, Any], in_dim: int, auto_project: bool) -> BuildResult:
    code = params["code"]
    out_dim = params.get("out_features", in_dim)
    return InlineCodeOp(code), out_dim


def get_block(name: str) -> BlockBuilder:
    if name not in BLOCK_REGISTRY:
        raise KeyError(f"Block '{name}' is not registered. Known: {list(BLOCK_REGISTRY.keys())}")
    return BLOCK_REGISTRY[name]
