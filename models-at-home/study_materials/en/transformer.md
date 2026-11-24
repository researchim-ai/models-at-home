# Transformer: Architecture and How It Works

A study guide on the **Transformer** architecture (“Attention Is All You Need”, Vaswani et al., 2017): what it is made of, how attention works, positional encodings, and variants (encoder, decoder, encoder–decoder). The material is aimed at beginners: terms are explained as they appear, and there is a short glossary at the start.

---

## Who This Is For and How to Use It

- **If you are new to neural networks:** skim the **glossary** below first — it will make the rest easier. You don’t have to memorize everything; you can come back to it while reading.
- **If you already know ML basics:** use the glossary as a refresher; the main value is a coherent description of Transformer blocks and PyTorch code examples.
- **Relation to other files:** LLM internals (tokenization, training, inference) are covered in [LLM.md](LLM.md). Here the focus is on the Transformer architecture itself.

---

## Glossary of Terms

These terms will appear often. Short definitions so you don’t get lost.

| Term | Plain explanation |
|--------|---------------------|
| **Vector** | A fixed-length list of numbers (e.g. 768). In neural networks this represents the “meaning” of a token or position. |
| **Dimension** | How many numbers in a vector. We say “vector of dimension 768” or “768-dimensional vector”. |
| **Sequence** | An ordered list of elements (tokens or vectors). Sequence length is how many elements (e.g. 100 tokens). |
| **Position** | Where an element sits in the sequence: first token is position 1, second is 2, etc. Order matters (“cat eats” ≠ “eats cat”). |
| **Embedding** | Representing an object (token, word) as a vector. Usually from a lookup table: token id → row of matrix = vector. |
| **Layer** | One step of computation in the network: e.g. “multiply by a matrix and apply a function”. Many layers in a row give the model “depth”. |
| **Block** | Several layers grouped into one repeatable module. In a Transformer, one block = attention + feed-forward (and normalization, residuals). |
| **Model parameters** | The numbers the model learns: matrix weights, biases. Parameter count (e.g. “7 billion”) is one measure of model size. |
| **Linear transformation** | Multiplying a vector by a matrix (and adding a bias): $y = xW + b$. Changes dimension and “mixes” the vector components. |
| **Softmax** | Turns a set of numbers into a probability distribution: non-negative, sum to 1. Large values get closer to 1, small to 0. In attention, softmax is applied to scores to get “who to look at” weights. |
| **Residual connection** | Adding the layer output to its input: `output = input + layer(input)`. Helps gradients flow through many layers and stabilizes training. |
| **Normalization** | Scaling the vector to a “normal” range (e.g. by spread). Makes training more stable. |
| **Encoder / Decoder** | **Encoder** — the part that turns input (e.g. source text) into an internal representation. **Decoder** — the part that, from that representation (and already generated text), outputs the next token. LLMs often use only the decoder. |

These terms will appear again below with more context.

---

## What Is a Transformer

**Transformer** is an **architecture** (a “blueprint”) for neural networks that work with sequences: text, speech, action sequences, etc. Main idea: instead of processing the sequence **step by step** (as in older RNNs), the Transformer processes **all positions at once** and links them via **attention** — each position “looks” at others and updates its representation. Order is given by **positional encodings** and (when generating) a “don’t look at the future” mask.

**In simple terms:** take a sentence of 10 words. An RNN processes word 1, then 2, then 3… in order. The Transformer takes all 10 at once, turns each into a vector, then for each word computes “how related is it to each of the others?” From these relations it forms a new representation. This “round” is repeated many times (many blocks). In the end each word “knows” the context of the whole sentence.

Technically: the architecture is based on **self-attention** and **feed-forward** blocks; there are no recurrent (RNN) or convolutional layers for order. All positions are processed **in parallel**, which fits GPUs well.

The original paper described an **encoder–decoder** (source text → internal representation → target text), e.g. for machine translation. Three variants are widely used today:

| Variant | Role | Examples |
|--------|------|---------|
| **Encoder-only** | Only encoder: text → representations; used for classification, search, embeddings. Does not generate tokens one by one. | BERT, RoBERTa |
| **Decoder-only** | Only decoder: given previous tokens, predict the next (autoregression). This is how all modern LLMs are built. | GPT, LLaMA, Mistral |
| **Encoder–decoder** | Two stacks: encoder processes input (e.g. source language), decoder generates output (translation, summary) using the encoder’s “memory”. | T5, original Transformer |

LLMs almost always use a **decoder-only** Transformer (see [LLM.md](LLM.md)).

```
  General idea of the Transformer

  Input: sequence of tokens (or their embeddings)  [x₁, x₂, …, x_L]

  Each layer:
    1. Self-Attention: each position "looks" at others and updates its representation
    2. Feed-Forward: independently per position — nonlinear transformation

  Token order is given by positional encodings (not recurrent connections)
  and (in decoder) by the "don't look at the future" mask.
```

---

## Original Scheme: Encoder–Decoder

In "Attention Is All You Need" the model consists of an **encoder stack** and a **decoder stack**.

### Encoder

- **Input:** sequence of vectors (token embeddings + positional encodings).
- **One encoder layer:**  
  1. **Multi-Head Self-Attention** — each position attends to all others (no mask).  
  2. **Add & Norm** (residual + normalization).  
  3. **Feed-Forward** (per position).  
  4. **Add & Norm**.
- Encoder output is a set of vectors per position — **memory** $[m_1, \ldots, m_L]$ — which the decoder uses via **cross-attention**.

### Decoder

- **Input:** target sequence (shifted one token to the right; "don't see the future" mask).
- **One decoder layer:**  
  1. **Masked Multi-Head Self-Attention** — only on the target sequence, with **causal mask** (position $i$ sees only $1..i$).  
  2. Add & Norm.  
  3. **Multi-Head Cross-Attention**: Query from decoder, Key and Value from encoder output (memory). This is how the decoder "queries" the source text.  
  4. Add & Norm.  
  5. **Feed-Forward**.  
  6. Add & Norm.
- After the decoder stack — a linear layer to vocabulary size (logits for the next token at train/inference time).

```
  Original Transformer (Encoder–Decoder)

  Source sequence                         Target sequence
  [s₁, s₂, …, s_L]                         [t₁, t₂, …, t_M]
        │                                          │
        ▼                                          ▼
  ┌─────────────┐                           ┌─────────────┐
  │ Embedding + │                           │ Embedding + │
  │ Pos Enc     │                           │ Pos Enc     │
  └──────┬──────┘                           └──────┬──────┘
         │                                         │
         ▼                                         ▼
  ┌─────────────┐                           ┌─────────────┐
  │ Enc Layer 1 │  Self-Attn (all-to-all)   │ Dec Layer 1 │  Masked Self-Attn
  │   + FFN     │                           │ + Cross-Attn (Q from dec, K,V from enc)
  └──────┬──────┘                           │   + FFN     │
         │                                  └──────┬──────┘
         ▼                                         │
        ...                                        ...
         │                                         │
         ▼                                         ▼
  ┌─────────────┐                           ┌─────────────┐
  │ Enc Layer N │  ──────────────────────►  │ Dec Layer N │  ──► Linear ──► logits
  └─────────────┘   memory [m₁…m_L]         └─────────────┘
```

---

## Self-Attention

**Attention** in general is a mechanism to "look at other elements and mix their information with weights". **Self-attention** means we look at the **same** sequence (other positions in it). Each element "asks" the others "what do you carry?" and forms its updated representation from the answers.

The core of the Transformer is **Scaled Dot-Product Attention**. For each position, **weights** over all relevant positions (subject to the mask: in the decoder we must not look at the future) are computed — how much to attend to each — and the output is a **weighted sum** of the **Value** vectors of those positions.

### Steps (formulas)

1. **Query, Key, Value (Q, K, V)** — three **linear** transformations of the same input $X$ (three different weight matrices). From each position vector we get three vectors:
   - **Query** — "query": what I want to compare other positions with;
   - **Key** — "key": how others can match me with their queries;
   - **Value** — "value": what I contribute to the weighted sum when "attended to".
   Formally: $Q = X W_Q$, $K = X W_K$, $V = X W_V$.  
   Shapes: $Q, K \in \mathbb{R}^{L \times d_k}$, $V \in \mathbb{R}^{L \times d_v}$ (often $d_k = d_v$).

2. **Scores** — for each pair $(i, j)$ we compute the "similarity" of query $i$ and key $j$ (dot product), and divide by $\sqrt{d_k}$ for stability:
   $$\text{scores} = \frac{Q K^\top}{\sqrt{d_k}}$$
   Matrix of size $L \times L$: entry $(i,j)$ is "how much position $i$ should attend to position $j$".

3. **Scaling by $\sqrt{d_k}$** prevents dot products from growing with $d_k$; otherwise after softmax weights become almost one-hot and gradients shrink. The divisor stabilizes scale and training.

4. **Mask** (in decoder): when generating text the model must not "peek" at future tokens. So for pairs "position $i$ attends to position $j$" with $j > i$ we set the score to $-\infty$. After softmax those entries become 0 — we "don't look" at the future. This is the **causal** mask: only the past is used.

5. **Attention weights:** **softmax** is applied **per row** of the score matrix. Each row becomes a set of non-negative numbers summing to 1 — the weights "how much to take from each position":
   $$A = \mathrm{softmax}(\text{scores}) \quad \Rightarrow \quad \sum_j A_{ij} = 1.$$

6. **Output:** for each position $i$ we take the **weighted sum** of Value vectors over all positions $j$ with weights $A_{ij}$. Result: a new vector per position (its "updated" representation with context):
   $$\mathrm{Attention}(Q,K,V) = A V.$$
   Shape: $L \times d_v$ — one vector of dimension $d_v$ per position.

```
  Self-Attention (one head)

  X [L × H] ──► Q = X W_Q   K = X W_K   V = X W_V
                     │           │           │
                     ▼           ▼           ▼
  scores = Q K^T / √d_k    [L × L]; in decoder lower triangle only, rest -∞
                     │
                     ▼  softmax per row
  A [L × L], each row — distribution of weights over positions
                     │
                     ▼  A · V
  output [L × d_v] — new sequence (contextual representation)
```

### Meaning of Q, K, V

- **Query** — current position's "query": "what do I need from context?"
- **Key** — each position's "key": "how can I be found?"
- **Value** — position's "content": "what I contribute to the weighted sum."

Score $Q_i \cdot K_j$ is how much query $i$ matches key $j$; the $V_j$ values are averaged by these scores.

---

## Multi-Head Attention

**Multi-Head** means running the attention mechanism **several times in parallel** with different weight sets. Each run is a **head**. Each head has its own $W_Q, W_K, W_V$, so each head can learn to focus on different kinds of relations: one on nearby words, another on grammar, another on recurring themes, etc. The outputs of all heads are **concatenated** along the dimension and passed through one more linear layer $W_O$ to mix information from different heads.

Technically: per-head dimension is usually $d_k = d_v = H / h$ (where $h$ is the number of heads) so total compute stays reasonable. Head outputs are concatenated to get a vector of length $H$ again, then $W_O$ is applied:

$$\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W_O,$$
$$\mathrm{head}_i = \mathrm{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}).$$

This lets the model capture different relation types at once (local, long-range, syntax, etc.).

```
  Multi-Head Attention (h heads)

  X [L × H] ──┬──► head_1 (Q₁,K₁,V₁) ──► [L × H/h]
              ├──► head_2 (Q₂,K₂,V₂) ──► [L × H/h]
              ├──► ...
              └──► head_h (Qₕ,Kₕ,Vₕ) ──► [L × H/h]
                              │
                              ▼  Concat
                        [L × H]
                              │
                              ▼  Linear W_O
                        output [L × H]
```

---

## Cross-Attention (in encoder–decoder)

In the decoder **cross-attention** connects the target to the encoder:

- **Query** — from decoder output (current state of the target sequence).
- **Key** and **Value** — from encoder output (memory).

Same formula: $\mathrm{Attention}(Q_{\mathrm{dec}}, K_{\mathrm{enc}}, V_{\mathrm{enc}})$. No mask over encoder positions (decoder can attend to all source positions). The causal mask is only in the decoder's self-attention.

```
  Cross-Attention (one decoder layer)

  Decoder: [d₁, d₂, …, d_M]   Encoder (memory): [m₁, m₂, …, m_L]
       │                              │
       ▼                              ▼
  Q = Dec · W_Q                 K = Enc · W_K    V = Enc · W_V
       │                              │
       └──────────► scores = Q K^T / √d_k  [M × L]  (no causal over L)
                              │
                              ▼  softmax, then A · V
  output [M × d_v] — per decoder position, weighted sum over memory
```

---

## Feed-Forward Network (FFN)

After the attention block in each Transformer layer comes a **Feed-Forward Network (FFN)** — a small standard neural net. Important: it is applied **independently and identically to each position** — no information is exchanged between positions; the same linear layers and nonlinearity are used everywhere. First the dimension is increased (e.g. 4×), a nonlinearity (ReLU or SiLU) is applied, then it is projected back to the original dimension. Attention "gathers" context; the FFN gives the model "room" for nonlinear transforms and storing patterns.

**Classic (original Transformer, GPT-2):**
$$\mathrm{FFN}(x) = \mathrm{ReLU}(x W_1 + b_1) W_2 + b_2.$$

**SwiGLU (many modern LLMs, e.g. LLaMA):**
$$\mathrm{FFN}(x) = \big(\sigma(x W_1) \odot (x W_3)\big) W_2,$$
where $\sigma$ is SiLU (Swish), $\odot$ is element-wise product. The inner dimension (columns of $W_1$, $W_3$) is usually **3–4×** hidden_size (e.g. 11008 for 4096).

FFN provides "room" for patterns and facts; attention provides access to the right positions.

```
  FFN (applied per position independently, weights shared)

  x [H] ──► W₁ [H → 4H] ──► ReLU/SiLU ──► W₂ [4H → H] ──► output

  SwiGLU:  x ──► W₁ ──► σ ──┐
            x ──► W₃ ──►────┴── ⊙ ──► W₂ ──► output
```

---

## Normalization and Residual Connections

Two techniques make deep stacks train well.

- **Residual connection:** the sublayer output is **added** to its input: $x_{\mathrm{out}} = x + \mathrm{Sublayer}(x)$. The sublayer only adds a "correction" to the current representation. Gradients can flow directly through these additions and do not vanish in depth — training is more stable.

- **Normalization:** before (or after) the sublayer the vector is scaled to a "normal" range — e.g. subtract mean and divide by standard deviation along the last axis, or divide by RMS. Most often Transformers use **Pre-LN** — normalize **before** the sublayer: $x_{\mathrm{out}} = x + \mathrm{Sublayer}(\mathrm{Norm}(x))$. Then the sublayer always receives data in a predictable scale.

**LayerNorm:** along the last axis (over vector components) compute mean $\mu$ and variance, then $y = \gamma \cdot (x - \mu) / \sqrt{\sigma^2 + \varepsilon} + \beta$. Parameters $\gamma$ and $\beta$ are learned.

**RMSNorm** (in many LLMs): do not subtract the mean — only scale by **RMS** (root mean square) along the last axis plus one learned scale. Simpler and cheaper; sufficient in LLaMA-style practice.

Typical order in **one block** (decoder-only, Pre-LN):

1. $x_1 = x + \mathrm{Attention}(\mathrm{Norm}(x))$
2. $x_2 = x_1 + \mathrm{FFN}(\mathrm{Norm}(x_1))$

```
  One Transformer block (decoder, Pre-LN)

  x ──► Norm ──► Attention ──► (+) ──► x₁
    │                ▲          │
    └────────────────┘          │
                                 ▼
  x₁ ──► Norm ──► FFN ──► (+) ──► x₂  (block output)
    │              ▲       │
    └──────────────┘       ┘
```

---

## Stack of Blocks: Depth and Number of Layers

A Transformer is **identical blocks** stacked. **Stack** here means "several blocks in a row". **Depth** is the number of such blocks (layers). One block keeps the same shape (input and output `[L × H]`); only the **content** of the representations changes — each block refines contextual information. The number of blocks is fixed before training — a **hyperparameter** (model "design" parameter, unlike learned weights).

### What one block does

- **Attention** in the block: positions exchange information; each position gets a weighted sum over all allowed positions (with causal mask in the decoder).
- **FFN**: per position, independent nonlinear transform; often where patterns and facts are "stored", which attention in other layers can access.

One block = one "round" of position interaction + one "round" of local processing. That is not enough for complex language; so blocks are repeated.

### How many blocks

Number of blocks (**num_layers**, model depth) is a key hyperparameter. Typical for decoder-only LLMs:

| Model scale | num_layers | hidden_size (example) | Examples |
|-------------|------------|------------------------|----------|
| Small (100M–500M) | 12–24 | 768–1024 | GPT-2 small, TinyLlama |
| Medium (1B–7B) | 24–32 | 2048–4096 | LLaMA-7B, Mistral-7B |
| Large (13B–70B) | 32–80 | 5120–8192 | LLaMA-70B, Qwen-72B |

More blocks → more "steps" to pass information between distant tokens and more parameters. Deeper models usually reason better and capture complex dependencies, but training is costlier and needs careful initialization and normalization (Pre-LN, residuals).

### How information flows across layers

- **Lower layers (first ~⅓):** often more local — nearby words, morphology, simple syntax.
- **Middle layers:** more global — agreement, references to earlier parts of the sentence.
- **Upper layers:** abstract representations, "decision" (what to predict next); closer to logits and output.

This is not strict: attention in each head and layer is learned, and role distribution can vary across models.

### Stack sketch

```
  Decoder-only stack (num_layers = N)

  Embeddings + positions  [L × H]
            │
            ▼
  ┌─────────────────┐
  │  Block 1        │  →  [L × H]
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  Block 2        │  →  [L × H]
  └────────┬────────┘
           │
           ⋮   (all blocks same shape)
           │
           ▼
  ┌─────────────────┐
  │  Block N        │  →  [L × H]
  └────────┬────────┘
           ▼
  Final norm → LM head (Linear H → vocab_size) → logits
```

Order inside each block is the same: Norm → Attention → residual → Norm → FFN → residual. Only weights differ from block to block.

### Parameters vs depth

For fixed `hidden_size`, `num_heads`, `intermediate_size`, **parameter count grows linearly with num_layers**: each block adds its Q, K, V, O and FFN matrices. So most parameters in a large model are in the repeated blocks, not in a single embedding or layer.

---

## Positional Encodings

Self-attention has an important property: it processes positions via sums and weighted combinations and **does not encode** "who is first, who is second". If we reorder tokens but keep the same vectors, attention scores and outputs would be the same. So **order is invisible** to the model by itself. To give the model "first word", "second word", etc., we add a **positional encoding** — a vector that depends on position index — to each position's representation. Options:

### 1. Sinusoidal (original paper)

For position $pos$ and dimension $i$:
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}).$$

Pros: fixed, can extrapolate to longer lengths. Cons: model does not always use long context well; embeddings and positions are just added.

### 2. Learned

One vector per position $1, \ldots, N_{\max}$; learned with the model. Simple and effective up to $N_{\max}$, but the model is not trained beyond $N_{\max}$.

### 3. RoPE (Rotary Position Embedding)

Position is added by **rotating** vectors in space. For dimension pairs $(2k, 2k+1)$, Query and Key vectors are rotated by angle $\theta_k \cdot pos$, $\theta_k = 10000^{-2k/d}$. Then the dot product $Q_i \cdot K_j$ depends on **position difference** $i - j$. Used in LLaMA, Mistral, Qwen, etc. Good extrapolation to long contexts.

### 4. ALiBi (Attention with Linear Biases)

Position is not in the embedding but as a **linear bias** on attention logits: add $-m \cdot |i - j|$ to $\mathrm{scores}_{ij}$ ($m$ can depend on head). Simple; often works well on long sequences.

```
  Where positions are applied

  Sinusoidal / Learned:   token_emb(x) + pos_emb(pos)  →  input to blocks

  RoPE:   applied to Q and K inside attention (before Q K^T)

  ALiBi:   scores = Q K^T / √d_k + bias(i, j),  bias = -m·|i-j|
```

---

## Why the Transformer Works

1. **Parallelism:** all positions are processed at once; no sequential dependency as in RNNs. Training on GPU is very efficient.

2. **Long-range dependencies:** in one layer the path between any two positions is one attention step; with many layers information can flow through different "routes". In RNNs the path between distant tokens is long and gradients vanish.

3. **Flexible attention:** weights $A_{ij}$ are learned; the model decides which positions to attend to (local or global).

4. **Scaling:** increasing depth (layers), width (hidden_size, number of heads), and data gives a predictable gain in quality (scaling laws).

---

## Code Examples (PyTorch)

Below are minimal but working PyTorch snippets that match the formulas above. Dimensions in comments: `B` = batch, `L` = seq_len, `H` = hidden_size, `d` = head dimension.

### Scaled Dot-Product Attention (single head, causal mask)

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    q: torch.Tensor,  # [B, L, d]
    k: torch.Tensor,  # [B, L, d]
    v: torch.Tensor,  # [B, L, d]
    causal: bool = True,
) -> torch.Tensor:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  # [B, L, L]

    if causal:
        L = scores.size(-1)
        mask = torch.triu(
            torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)  # [B, L, L]
    out = torch.matmul(attn_weights, v)        # [B, L, d]
    return out
```

### Multi-Head Attention (decoder, causal)

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, causal: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H]
        B, L, H = x.shape
        q = self.w_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, L, d]
        k = self.w_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # merge batch and heads for attention call: [B*h, L, d]
        q = q.reshape(B * self.num_heads, L, self.head_dim)
        k = k.reshape(B * self.num_heads, L, self.head_dim)
        v = v.reshape(B * self.num_heads, L, self.head_dim)
        out = scaled_dot_product_attention(q, k, v, causal=self.causal)  # [B*h, L, d]
        out = out.view(B, self.num_heads, L, self.head_dim).transpose(1, 2).contiguous().view(B, L, H)
        return self.w_o(out)
```

### RMSNorm

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.weight
```

### Feed-Forward: classic ReLU and SwiGLU

```python
class FeedForwardReLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### One Transformer block (decoder-only, Pre-LN)

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.ln1 = RMSNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, causal=True)
        self.ln2 = RMSNorm(hidden_size)
        self.ffn = (
            FeedForwardSwiGLU(hidden_size, intermediate_size)
            if use_swiglu
            else FeedForwardReLU(hidden_size, intermediate_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

### Stack of blocks + embedding and LM head (minimal decoder-only)

```python
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.ln_f = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, L]
        B, L = token_ids.shape
        x = self.embed(token_ids) + self.pos_embed(torch.arange(L, device=token_ids.device))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, L, vocab_size]
        return logits
```

Usage: `logits = model(token_ids)`. For next-token prediction take `logits[:, -1, :]` and apply softmax/sampling. For training compute cross-entropy over `logits[:, :-1]` and target tokens `token_ids[:, 1:]`.

---

## Summary: Dimensions and Hyperparameters

| Parameter | Symbol | Meaning | Typical values |
|-----------|--------|---------|----------------|
| Sequence length | $L$ | Number of input tokens | up to 2k–128k |
| Hidden size | $H$ (hidden_size) | Embedding and block output dimension | 768, 2048, 4096, 8192 |
| Number of heads | $h$ (num_heads) | In multi-head attention | 12, 32, 64; $H$ divisible by $h$ |
| Per-head dimension | $d_k$, $d_v$ | Usually $H / h$ | 64, 128 |
| FFN inner size | intermediate_size | Usually 3–4 × $H$ | 3072, 11008 |
| **Number of blocks (layers)** | $N$ (num_layers) | How many times the block is repeated | 12–24 (small), 32–40 (7B), 80 (70B+) |

**Typical relations:** $d_k = d_v = H/h$; one block has attention matrices (Q, K, V, O) and two (or three with SwiGLU) matrices in the FFN. Most parameters are in the **repeated blocks** (attention + FFN), not in embeddings (for a moderate vocab). Full pipeline from tokens to logits is in the "Code examples (PyTorch)" section.

---

## Relation to Other Materials

- **Decoder-only** Transformer as the basis of LLMs, tokenization, training, and inference — in [LLM.md](LLM.md).
- Here the architecture itself is covered: attention, FFN, positions, encoder/decoder variants, **number and role of blocks**, and **PyTorch examples** (attention, FFN, block, stack to logits).
