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
        return self.embed(positions)


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
