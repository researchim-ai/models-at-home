"""Editor defaults and categories; importable by an API without Torch installed."""
from copy import deepcopy
from backend.services.blueprint_knowledge import knowledge


def catalog():
    h = 512
    llama = {'num_heads': 8, 'num_kv_heads': 8, 'intermediate_size': 4*h, 'dropout': 0, 'rope_theta': 10000, 'use_liger': False, 'sandwich_norm': False}
    entries = [
        ('token_embedding', 'Embeddings', 'Token embeddings', {'vocab_size': 50257, 'hidden_size': h}),
        ('positional_embedding', 'Embeddings', 'Learned positional embeddings', {'max_position_embeddings': 2048, 'hidden_size': h}),
        ('llama_block', 'Transformers', 'RoPE + MHA/GQA + SwiGLU; pre-norm or sandwich norms', llama),
        ('causal_self_attention', 'Attention', 'Causal SDPA, optional RoPE and grouped-query attention', {'num_heads': 8, 'num_kv_heads': 8, 'use_rope': True, 'rope_theta': 10000, 'dropout': 0}),
        ('attention', 'Attention', 'Causal multi-head attention', {'num_heads': 8, 'dropout': 0}),
        ('loop', 'Composition', 'One shared body, repeated in latent depth', {'iterations': 4, 'eval_iterations': 4, 'min_iterations': 1, 'random_depth': False, 'gradient_steps': 0, 'checkpoint': False, 'input_injection': 'none', 'state_init': 'input', 'step_embedding': 'none', 'body': [{'type': 'llama_block', 'params': llama}]}),
        ('repeater', 'Composition', 'Independent copies of one block (different weights)', {'num_repeats': 4, 'block_type': 'llama_block', 'block_params': llama}),
        ('residual', 'Composition', 'Editable skip branch; learnable scale supports ReZero', {'scale': 1, 'learnable_scale': False, 'body': [{'type': 'rmsnorm', 'params': {'eps': 1e-6}}, {'type': 'swiglu', 'params': {'intermediate_size': 4*h}}]}),
        ('rmsnorm', 'Normalization', 'RMS normalization', {'eps': 1e-6, 'use_liger': False}),
        ('layernorm', 'Normalization', 'Layer normalization', {'eps': 1e-5, 'use_liger': False}),
        ('groupnorm', 'Normalization', 'Channels-last group normalization', {'num_groups': 8}),
        ('swiglu', 'Feed-forward', 'Gated SiLU feed-forward', {'intermediate_size': 4*h, 'dropout': 0, 'use_liger': False}),
        ('mlp', 'Feed-forward', 'Two-layer feed-forward', {'intermediate_size': 4*h, 'activation': 'gelu', 'dropout': 0}),
        ('moe', 'Feed-forward', 'Sparse expert routing with load-balancing loss', {'num_experts': 8, 'num_select': 2, 'expert_type': 'swiglu', 'dropout': 0}),
        ('residual_mlp', 'Feed-forward', 'Residual feed-forward', {'intermediate_size': 4*h, 'activation': 'silu'}),
        ('causal_conv1d', 'Sequence', 'Causal convolution for hybrid sequence models', {'out_channels': h, 'kernel_size': 3, 'dilation': 1, 'groups': 1}),
        ('linear', 'Graph', 'Trainable projection, including equal-width projections', {'in_features': h, 'out_features': h, 'bias': False}),
        ('add', 'Graph', 'Sum at least two equal-width branches', {}),
        ('multiply', 'Graph', 'Multiply two equal-width branches (e.g. a gate)', {}),
        ('concat', 'Graph', 'Concatenate branches along hidden width', {'dim': -1}),
        ('activation', 'Graph', 'Standalone non-linearity; sigmoid enables gating', {'activation': 'silu'}),
        ('dropout', 'Regularization', 'Dropout', {'p': .1}),
        ('custom_op', 'Advanced', 'Local Python module; declare output width and verify its causal contract', {'path': 'torch.nn.Identity', 'kwargs': {}, 'out_features': h}),
        ('inline_code', 'Advanced', 'Local Python forward code; parameter analysis cannot verify this operator', {'code': 'x = x', 'out_features': h}),
        ('permute', 'Advanced', 'Tensor-axis permutation; requires manual sequence-shape validation', {'dims': [0, 1, 2]}),
        ('adaptive_avg_pool', 'Advanced', 'Changes sequence length; manually align labels before training', {'output_size': 1}),
    ]
    for name in ('rmsnorm', 'layernorm', 'swiglu', 'llama_block'):
        base = next(row for row in entries if row[0] == name)
        entries.append(('liger_' + name, 'Liger', 'Optional fused ' + name, base[3]))
    return [dict(type=name, category=group, description=description, defaults=deepcopy(defaults), research=knowledge(name)) for name, group, description, defaults in entries]


SUPPORTED = {item['type'] for item in catalog()}
