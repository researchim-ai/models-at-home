# LLM: What You Need to Know About Structure and Training

A detailed study guide on Large Language Models (LLMs): what they are made of, how they work, and how they are trained. The material is aimed at beginners: key terms are explained on first use, and there is a short glossary at the start. If something is unclear, you can return to the glossary or the Transformer section ([transformer.md](transformer.md)). For models that work with images and text, see [VLM.md](VLM.md).

---

## Glossary of Terms

These terms appear often. You don’t have to memorize them — you can refer back to the table while reading.

| Term | Plain explanation |
|--------|---------------------|
| **Token** | The smallest unit of text the model works with. Usually a **subword** (piece of a word): e.g. “Hel” and “lo” are two tokens for “Hello”. Sometimes a whole short word or one character. |
| **Tokenizer** | The algorithm and vocabulary that split text into tokens and map them to numbers (ids). E.g. “Hello” → [1045, 3921]. |
| **Vocabulary (vocab)** | The list of all allowed tokens. Each token has an id. Vocabulary size is how many tokens (often 32k–128k). |
| **Token id** | Integer — the token’s index in the vocabulary. The model works with numbers; at decode time the id is turned back into text. |
| **Prompt** | The text the user gives the model as input — a question, start of a sentence, context. The model predicts the continuation. |
| **Context** | The text the model “sees” at any moment: prompt plus already generated tokens. Context length is limited (context window). |
| **Autoregression** | Generating one token at a time: each next token depends on all previous ones. The model doesn’t output the whole answer at once. |
| **Model parameters** | The numbers the network learns (matrix weights, biases). “7B” (7 billion) is roughly the parameter count of a model like LLaMA-7B. |
| **Embedding** | Representation of a token as a vector (fixed-length list of numbers). Obtained by lookup: token id → row of a matrix. |
| **OOV (out of vocabulary)** | A token or word not in the vocabulary. With word-level tokenization unknown words are often replaced by [UNK]. With subwords (BPE) they are built from known pieces. |
| **EOS (end of sequence)** | Special token meaning “end of sequence”. The model can output it to signal that the answer is finished. |

---

## What Is an LLM

**LLM (Large Language Model)** is a **large language model**: a neural network trained to predict the **next** piece of text. That piece is a **token** (usually a subword or short word, not a character or full phrase). So the model always answers: “given the text so far, which token comes next with the highest probability?” Formally it is a **conditional probability model**:

$$P(x_{t+1} \mid x_1, x_2, \ldots, x_t)$$

where $x_i$ are tokens. Generation is **autoregressive**: the model outputs exactly **one** next token, that token is appended to the context, and again one token is predicted — until the end of the answer or a stop token (e.g. EOS).

```
  Autoregression: at each step the model outputs exactly ONE token. Token = subword
  (piece of a word or whole short word), not a character or full phrase.

  Text (prompt):   "The capital of France is "
                            │
                            ▼  tokenizer (BPE/SentencePiece) → token sequence
  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      ┌─────────┐
  │ token 1  │ token 2  │ token 3  │ token 4  │ token 5  │  →   │ token 6 │
  │ "The"    │ " capital"│ " of"   │ " France"│ " is"    │      │   ?     │
  └──────────┴──────────┴──────────┴──────────┴──────────┘      └─────────┘
       x₁          x₂          x₃          x₄          x₅           ↑
  ────────────────────────────────────────────────────────────────  model
  Token boundaries may not match spaces: "France" = " France",         predicts
  "capital" = " capital". A token can include a leading space.         one token
                            │
                            ▼
  Step 1: model outputs " Paris"   →  context: ... " is" " Paris"
  Step 2: model outputs "."        →  context: ... " Paris" "."
  Step 3: model outputs [EOS]      →  stop.

  So: each step adds exactly one token (subword); final text is
  the concatenation of tokens.
```

### Why “Large”

- As the number of parameters and amount of training data grow, the model gets better at grammar, facts, reasoning, and style (**scaling laws**).
- “Understanding” and “knowledge” in an LLM are statistical patterns in the weights; the model doesn’t store facts explicitly but generalizes from data.

### Context Window

The model “sees” only the last **N** tokens — the **context length**, or **context window**. Typical values: 2k (2000 tokens), 4k, 8k, 32k, 128k. Here “k” means thousands: 8k = 8000 tokens. Anything before those N tokens is formally unavailable — the model cannot “read” older input. In practice, when trained on long text, the model learns to compress important information into hidden representations, but it cannot explicitly store arbitrarily long history.

```
  Context window (N tokens):

  ←────────── visible part (context window) ──────────→
  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  │ t₁  │ t₂  │ t₃  │ ... │ tₙ₋₂│ tₙ₋₁│ tₙ  │ ??? │
  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
    ↑                                               ↑
  model sees only these N tokens              next token
  (everything left of window “forgotten”)      predicted at this position
```

---

## Tokenization (Splitting Text)

Before being fed to the model, text is turned into a sequence of integers — **token ids** (each token is replaced by its index in the vocabulary). This is done by the **tokenizer**: it knows the splitting rules and the vocabulary. The choice of tokenizer and vocabulary size affects sequence length (how many ids per phrase), quality in different languages, and handling of rare and new words.

### Why Not Characters or Words?

Three ways to choose the token unit:

| Approach     | Pros                         | Cons                                                                 |
|--------------|------------------------------|----------------------------------------------------------------------|
| **Characters** | Small vocab, any word        | Very long sequences, weaker model                                   |
| **Words**      | Short sequences              | Huge vocab, many word forms; OOV words become a single [UNK] token   |
| **Subwords**   | Compromise: 10k–100k vocab, new words as pieces | Need to train the tokenizer and store the vocab |

**Subword tokenization** splits text into pieces: frequent words can be one token, rare ones several subwords. An unknown word is represented as a concatenation of known subwords without losing characters.

```
  Comparison: same sentence under different tokenization

  Text: "The neural network is trained on data."

  By characters (each character = token):
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  │T │h │e │  │n │e │u │r │a │l │  │n │e │t │w │o │r │k │  │...  (many tokens)
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
  Many tokens → expensive and long context.

  By words (vocab 100k):
  ┌──────┬────────┬──────┬────────┬────┬──────┬────┐
  │ The  │ neural │network│ trained│ on │ data │ .  │   unknown word → one [UNK]
  └──────┴────────┴──────┴────────┴────┴──────┴────┘

  By subwords (BPE, vocab ~32k):
  ┌──────┬────────┬──────┬────┬────────┬────┬────┐
  │ The  │ neural │ net  │work│ train  │ ed │ on │ ...  (example; actual split depends on vocab)
  └──────┴────────┴──────┴────┴────────┴────┴────┘
  Balance of length and expressiveness.
```

---

### BPE (Byte Pair Encoding)

Idea: start from atomic units (characters or bytes) and **iteratively merge** the most frequent pair of adjacent tokens into one new token. The vocabulary grows until the target size.

**BPE training algorithm:**

1. **Initialize:** vocabulary = all characters (or all 256 bytes in byte-level BPE). Each character is a separate token.
2. **Count pairs:** over the corpus, count how often each pair of tokens appears consecutively (in the current segmentation).
3. **Merge:** pick the pair with maximum frequency, add it to the vocabulary as one new token, replace that pair with the new token everywhere in the corpus.
4. Repeat steps 2–3 until the target vocabulary size (e.g. 32k–50k).
5. Save the **merge rules** (order matters): at encode time apply merges in this order.

```
  BPE: training loop (overview)

  ┌─────────────┐     current corpus segmentation (first by characters)
  │   Corpus    │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     count frequencies of consecutive pairs (A,B)
  │ Count pairs │     e.g. ("a","n")=5000, ("n","a")=3000, ...
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     pick pair with max frequency, add to vocab,
  │   Merge     │     replace that pair with one token everywhere
  └──────┬──────┘
         │
         ▼
  vocab size < target?  ──No──►  stop, save merge rules
         │
        Yes
         │
         └──────────────────────────► repeat count pairs
```

```
  BPE: training on word "banana" (simplified, only this word in corpus)

  Initial: split by characters.
  ┌─────┬─────┬─────┬─────┬─────┬─────┐
  │  b  │  a  │  n  │  a  │  n  │  a  │
  └─────┴─────┴─────┴─────┴─────┴─────┘
  Pairs: (b,a)=1, (a,n)=2, (n,a)=2. Most frequent (a,n) → merge to "an".

  After 1st merge (a + n → "an"):
  ┌─────┬─────┬─────┬─────┬─────┐
  │  b  │ an  │  a  │  n  │  a  │
  └─────┴─────┴─────┴─────┴─────┘
  Pairs: (an,a)=2, ... Merge (an,a) → "ana", etc.

  After several merges:
  ┌─────┬─────────┬─────┐
  │ ban │   ana   │     │   or   │ ba │ nana │  (depends on corpus and merge order)
  └─────┴─────────┴─────┘
  Result: word represented by fewer tokens.
```

**Encoding new text:** split text into characters, then apply merge rules **in order** (as they were added during training). Each rule: if the current token sequence contains that pair consecutively — replace it with one token.

```
  BPE: encoding (encode)

  Text: "banana"
  Step 0:  [ b ] [ a ] [ n ] [ a ] [ n ] [ a ]

  Apply rules in order (as in training):
  Rule 1: (a, n) → "an"   →  [ b ] [ an ] [ a ] [ n ] [ a ]
  Rule 2: (an, a) → "ana" →  [ b ] [ ana ] [ n ] [ a ]
  ... (continue with rule list until no more apply)

  Final token sequence → map to ids via vocabulary.
```

**Decoding:** from the id list recover token strings (each id maps to a string). Concatenate them to get the original text. Special tokens (EOS, pad) are not decoded to characters.

```
  BPE: decoding (decode)

  id sequence:  [ 101, 204, 305 ]
        │
        ▼  via vocab:  101 → "Hello",  204 → ",",  305 → " world"
  Tokens:  "Hello"  ","  " world"
        │
        ▼  concatenation (no spaces between pieces unless special rules)
  Text:   "Hello, world"
```

**Byte-level BPE (as in GPT-2):** base units are **bytes** (0–255). Any string is a sequence of bytes. Plus: no “unknown character”; emoji and rare Unicode are split into bytes. Minus: for non-ASCII text you get many small tokens; part of the vocab is “spent” on bytes.

**Vocabulary size is a tradeoff:** small (8k–16k) → longer sequences, more compute; large (64k–128k) → shorter sequences but more parameters in embedding and LM head, and many tokens are rarer in training.

### SentencePiece and similar

**SentencePiece** is a framework, not a single algorithm: text is first turned into a sequence of characters (or bytes), then one of several segmentation methods is applied. Important: **spaces are encoded too** (e.g. space is replaced by `_` and included in tokens), so decoding recovers text and spaces unambiguously.

- **BPE inside SentencePiece:** same idea of merging frequent pairs; often implemented at Unicode or byte level.
- **Unigram:** each token in the vocab has a probability; segmentation is chosen to maximize the product of token probabilities (or sum of log probs). The vocab is trained iteratively (EM-like). Often gives more even quality across languages.
- Modern LLMs often use SentencePiece (Llama, Mistral) or BPE (GPT-2; LLaMA uses BPE-style in SentencePiece).

```
  Unigram (sketch): segment by maximum probability

  Word: "training"
  Possible splits (vocab has subwords with probabilities):

  Option A:  [ train ] [ ing ]   →  log P(train)+log P(ing)
  Option B:  [ tra ] [ in ] [ ing ]  →  ...
  Option C:  [ training ]  →  log P(training)

  Pick the split with maximum sum of log P(token). Vocab and probs
  are trained on the corpus (EM-like procedure).
```

```
  Tokenization pipeline (SentencePiece-style)

  "Hello, world!"         preprocessing (normalize spaces, etc.)
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Optional: pre-tokenization (split on spaces,            │
  │  punctuation) → "Hello" "," "world" "!"                 │
  └─────────────────────────────────────────────────────────┘
        │
        ▼  apply BPE / Unigram using vocab rules
  ┌─────┬─────┬─────┬─────┬─────┐
  │Hello│  ,  │     │world│  !  │   (example split)
  └─────┴─────┴─────┴─────┴─────┘
        │
        ▼  vocab lookup: string → id
  [ 1045, 3921, 11, 482, 0 ]
```

### Special tokens

The vocabulary includes tokens that are not “normal” text:

- **BOS / EOS** (beginning/end of sequence) — sequence boundaries; generation often stops when EOS is output.
- **Pad** — pad sequences to the same length in a batch; pad positions are usually masked out when computing loss.
- In chat models: `<|im_start|>`, `<|im_end|>`, `<|user|>`, `<|assistant|>` etc. to mark roles in the dialogue.

The tokenizer is usually **trained** on a representative text sample before training the model, then frozen. Tokenizer choice and vocabulary size affect sequence length and quality in different languages.

```
  Tokenization: final form (text → ids)

  "Hello, world!"                 vocabulary (vocab)
        │                                │
        ▼                                │
  ┌───────────┐   BPE / SentencePiece    │
  │ Tokenizer │ ────────────────────────►│  "Hello" → 1045
  └───────────┘                          │  "," → 3921
        │                                │  " world" → 482
        ▼                                │  "!" → 0
  [ 1045, 3921, 11, 482, 0 ]  ◄──────────┘
   id₁   id₂   id₃  id₄  id₅
   (seq_len = 5)
```

---

## Embedding: From Tokens to Vectors

Inside the model we work not with integers (token ids) but with **vectors** — fixed-length lists of numbers. That length is **hidden_size** (typically 768, 1024, 2048, 4096, 8192). The first step is to **turn each token id into a vector** of that dimension. This step is **embedding**: we map the discrete token into a continuous vector space the network can work with.

### Token embedding

- It is a **table (matrix)** of size **vocab_size × hidden_size**: one row per token, each row is a vector of length hidden_size.
- Token with id $i$ is the **$i$-th row** of this matrix. This is a **lookup**: we fetch the vector by index.
- These weights (the whole matrix) are **trained** with the rest of the network — token vectors adapt to the task.

### Positional embedding: why positions matter

The Transformer structure **does not** encode token order: if you shuffle tokens, the “raw” attention input would be the same. Order is added via **positional encodings**.

**Options:**

1. **Sinusoidal** — as in the original Transformer and GPT-2. Position $pos$ is encoded with $\sin(pos/10000^{2i/d})$, $\cos(\ldots)$. Plus: can extrapolate to unseen lengths. Minus: model often uses long context less well.

2. **Learned** — a separate vector per position up to max_length. Simple and effective at trained lengths, but the model is not trained beyond max_length.

3. **RoPE (Rotary Position Embedding)** — used in LLaMA, Mistral, Qwen, etc. Position is added by **rotating** Query and Key vectors with position. Simplified: for (Q, K) at positions $m$ and $n$ a relative angle $\theta(m-n)$ is used. Good extrapolation to longer lengths and natural encoding of “distance” between tokens.

4. **ALiBi (Attention with Linear Biases)** — position is not in the embedding but as a **linear bias** on attention scores: add $-m \cdot |i - j|$ to the attention logits ($m$ depends on the head). Simple and often extrapolates well.

**RoPE in more detail:** for each dimension pair $(2k, 2k+1)$, Q and K vectors are rotated by angle $\theta \cdot \mathrm{pos}$, $\theta = 10000^{-2k/d}$. Then the dot product $Q_i \cdot K_j$ depends only on **position difference** $i-j$, giving shift invariance and good extrapolation. In LLaMA and similar, RoPE is applied to Q and K **before** computing attention scores.

Result: the input to Transformer blocks is the **sum** (or concatenation, depending on the scheme) of token and positional representations; dimension at each position is **hidden_size**.

```
  Embedding: id → vectors (hidden_size)

  [ id₁   id₂   id₃  ...  id_L ]     (L = seq_len)
       │     │     │          │
       ▼     ▼     ▼          ▼
  ┌─────────────────────────────────────────┐
  │  Token Embedding  [vocab_size × H]       │   lookup by id
  └─────────────────────────────────────────┘
       │     │     │          │
       ▼     ▼     ▼          ▼
  [ v₁   v₂   v₃  ...  v_L ]   each v ∈ ℝ^H
       │     │     │          │
       │     +     +          +
       │     │     │          │
  ┌────┴─────┴─────┴──────────┴────┐
  │  Positional Embedding / RoPE   │
  └────────────────────────────────┘
       │     │     │          │
       ▼     ▼     ▼          ▼
  [ x₁   x₂   x₃  ...  x_L ]   input to Transformer  [L × H]
```

---

## Transformer Block: Attention and FFN

The core of an LLM is a **repeated block** (Transformer layer). Many such blocks in a row form the model body. Inside one block there are two main sublayers: **Self-Attention** (each position looks at others and updates its representation) and **Feed-Forward Network (FFN)** (nonlinear transform per position). Around them: **normalization** and **residual connections** for stable training. For block and attention details see [transformer.md](transformer.md).

### Self-Attention

For **each** position the model “looks” at all **allowed** tokens (in an LLM during generation — only itself and previous positions, not future) and forms a new vector as a **weighted sum** of their representations. The model learns the weights “whom to look at”.

#### Steps (formulas)

1. From input $X$ (shape `[batch, seq_len, hidden_size]`) get three matrices by linear maps: $Q = X W_Q$, $K = X W_K$, $V = X W_V$.

2. **Attention scores** for each pair $(i, j)$: $\text{scores}_{ij} = \frac{Q_i \cdot K_j^\top}{\sqrt{d_k}}$ where $d_k$ is key dimension (usually hidden_size / num_heads). The $\sqrt{d_k}$ **scaling** avoids peaky softmax and gradient issues when $d_k$ is large, and improves numerical stability.

3. **Causal mask**: in an LLM a token must not see the future. So for $j > i$ set $\text{scores}_{ij} = -\infty$. After softmax those positions get weight 0.

4. **Attention weights**: softmax over rows: $A = \mathrm{softmax}(\text{scores})$.

5. **Output**: weighted sum of values: $\text{Attention}(Q,K,V) = A \cdot V$. Output dimension equals $V$ (per head: hidden_size / num_heads).

```
  Attention (one head): from X to weighted sum over V

  Positions:  1     2     3    ...    L
               │     │     │           │
  X ────────► Q,K,V for each position
  scores:   [ s11   -∞    -∞   ...  -∞  ]   causal: j>i → -∞
            [ s21  s22   -∞   ...  -∞  ]
            [ ...  ...  ...  ...  ... ]
              │
              ▼ softmax over rows
  weights A: each row sums to 1
              │
              ▼  A · V
  output:    new vector per position (contextual representation)
```

#### Multi-Head Attention

Instead of one Q, K, V set we use **num_heads** parallel “heads”, each with its own $W_Q, W_K, W_V$. Outputs are **concatenated** and passed through a linear layer $W_O$: $\text{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h) W_O$. So the model can capture different types of relations (e.g. syntax and local context).

```
  Multi-Head Attention (num_heads = h)

  X [L × H] ──┬──► head_1 (Q₁,K₁,V₁) ──► out_1 [L × H/h]
              ├──► head_2 ... ──► out_2 [L × H/h]
              └──► head_h ──► out_h [L × H/h]
                              │
                              ▼  Concat → Linear W_O
                        output [L × H]
```

#### Dimensions

- **hidden_size** must be divisible by **num_heads**.
- Often $d_k = d_v = \text{hidden\_size} / \text{num\_heads}$ (e.g. 128 for hidden_size=4096, num_heads=32).

### Feed-Forward Network (FFN)

After attention comes a two-layer network applied **independently per position** (same weights for all positions). **GPT-2 style:** $\text{FFN}(x) = \mathrm{ReLU}(x W_1 + b_1) W_2 + b_2$. **SwiGLU (LLaMA, many recent):** $\text{FFN}(x) = (\sigma(x W_1) \odot (x W_3)) W_2$ with $\sigma$ = SiLU, $\odot$ = element-wise product. Inner dimension is **intermediate_size**, usually **3–4× hidden_size** (e.g. 11008 for hidden_size=4096). FFN stores facts and patterns; attention gives access to context.

```
  FFN (same network for all positions):

  x [H] ──► Linear ──► [intermediate] ──► SiLU/ReLU ──► Linear ──► [H]
  SwiGLU:  x ──► W₁ ──► σ ──┐
            x ──► W₃ ──►─────┴── * (element-wise) ──► W₂ ──► output
```

### Normalization and residuals

- **Residual**: each sublayer is added to its input: $x_{\mathrm{out}} = x + \mathrm{Sublayer}(x)$. Helps gradients flow through many layers.
- **Normalization**: usually **Pre-LN** — normalize **before** the sublayer: $\mathrm{Sublayer}(\mathrm{Norm}(x))$. LLMs often use **RMSNorm** instead of LayerNorm (no mean subtraction; scale by RMS). **Pre-LN** gives more stable training at great depth.

Typical block order (LLaMA-style): $x_1 = x + \mathrm{Attention}(\mathrm{RMSNorm}(x))$, $x_2 = x_1 + \mathrm{FFN}(\mathrm{RMSNorm}(x_1))$. The number of such blocks is **num_layers** (12, 24, 32, 40, 80 depending on model size).

---

## Output Layer (LM head)

After all blocks each position has a **hidden_size** vector. Final step: predict a distribution over the vocabulary — linear layer **hidden_size → vocab_size** giving **logits** per token. Often this layer is **tied** with the token embedding matrix (saves parameters and can help training). **Training:** cross-entropy between predicted distribution (softmax(logits)) and target token, averaged over positions and batch. **Inference:** from the last position’s logits we get a distribution and **sample** the next token (see below).

```
  Full model pipeline (tokens → logits):

  [id₁ ... id_L] ──► Embedding ──► [x₁ ... x_L] ──► Block 1 ──► ... ──► Block N
         ▼
  [h₁ ... h_L]  ──► take last position h_L ──► LM head ──► logits ──► softmax ──► sample
```

---

## Key Hyperparameters and Model Size

| Parameter | Meaning | Example values |
|----------|--------|-------------------|
| **vocab_size** | Token vocabulary size | 32k, 50k, 128k |
| **hidden_size** | Hidden representation size (embeddings, attention, FFN) | 768, 2048, 4096, 8192 |
| **num_layers** | Number of Transformer blocks | 12, 24, 32, 80 |
| **num_heads** | Number of attention heads (hidden_size divisible by num_heads) | 12, 32, 64 |
| **num_kv_heads** | In GQA: number of heads for K,V (smaller than num_heads) | 8, 32 |
| **intermediate_size** | Inner dimension in FFN | 3–4 × hidden_size |
| **max_position_embeddings** | Maximum context length | 2048, 8192, 128k |

**Rough parameter count (LLaMA-style):** Embedding + LM head ≈ $2 V H$ with weight tying. Per block: attention (Q,K,V,O) + FFN ≈ $8 H^2 + 3 H \cdot \text{intermediate\_size}$. For 7B parameters typical: hidden_size ≈ 4096, num_layers ≈ 32, intermediate_size ≈ 11008.

**GQA (Grouped-Query Attention):** some heads share the same K, V (num_kv_heads < num_heads). E.g. num_heads=32, num_kv_heads=8 → 8 (K,V) sets, each used by 4 Q heads. Cuts KV-cache memory at inference and speeds generation with small quality cost. **MQA** is the extreme: one shared (K,V) for all heads.

**Example configs:**

| Model  | Parameters | hidden | layers | heads | intermediate | context |
|---------|-----------|--------|--------|-------|---------------|----------|
| LLaMA 7B  | ~7B   | 4096 | 32 | 32 | 11008 | 2k–8k   |
| LLaMA 70B | ~70B  | 8192 | 80 | 64 | 28672 | 4k–8k   |
| Mistral 7B| ~7B   | 4096 | 32 | 32 (8 kv) | 14336 | 32k    |
| GPT-2 small | 124M | 768  | 12 | 12 | 3072  | 1024   |

---

## How LLMs Are Trained: Three Stages

### Stage 1: Pretraining (training from scratch)

**Goal:** teach the model to predict the next token in “raw” text so it learns language, facts, and style.

**Data:** Large corpora (web e.g. Common Crawl, books, code, articles, forums). **Cleaning and quality:** remove junk (repeated characters, broken markup), language filters, quality classifiers. Quality often matters more than volume beyond a point. **Deduplication:** exact (identical strings) and approximate (n-grams, MinHash) reduce memorization and improve generalization. **Tokenization:** either pre-encode to ids (saves CPU) or tokenize on the fly when loading a batch. Split into fixed-length sequences (e.g. 2048 tokens) or by document boundaries with padding/packing. **Packing** = concatenate short documents into one long sequence with separators to avoid wasting positions on pad.

```
  Pretraining batch:

  Raw text (documents) → chunk by seq_len → [Sample 1 ... Sample B]
  For each position i=1..L-1: target = token at i+1 (causal: model sees only t_1..t_i)
```

#### Loss (Cross-Entropy)

**Per position:** model outputs **logits** $z \in \mathbb{R}^{V}$. Softmax gives $p_k$. True token has id $y$. **Cross-Entropy:** $L_{\mathrm{CE}} = -\log p_y$. Higher probability on the correct token → lower loss. **Over batch/sequence:** average CE over all non-padded positions → **next-token prediction loss**. Minimizing CE is equivalent to MLE. **Masking:** pad positions are usually excluded from loss; in SFT, loss is only on assistant reply tokens.

#### Optimizers

Parameters $\theta$ are updated by the gradient of the loss. **SGD:** $\theta_{t+1} = \theta_t - \eta \nabla_\theta L$. **Momentum:** keep a velocity $v$ and update with $v$. **Adam:** keeps first moment $m$ and second moment $v$ of gradients; step is adaptive per parameter. Typical $\beta_1=0.9$, $\beta_2=0.999$. **AdamW:** Adam + decoupled weight decay $\theta \leftarrow \theta - \eta \lambda \theta$ after the Adam step. Standard for pretrain/SFT. Adam/AdamW use ~2× parameter memory for $m$ and $v$ (often in fp32).

#### Learning rate: warmup and decay

**Warmup:** at the start, LR is increased linearly from 0 (or a small value) to the target over several thousand steps to avoid huge steps when gradients are unstable. **Decay:** after warmup, LR is reduced, often **cosine decay** to a minimum (e.g. 10% of peak). Typical peak LR: 1e-4 … 3e-4 for pretraining; 1e-5 … 5e-5 for SFT.

#### Gradient clipping

Gradients can **explode**. To limit the step: compute global norm $\|g\|$; if $\|g\| > \mathrm{max\_norm}$ (e.g. 1.0), scale all gradients by $\mathrm{max\_norm} / \|g\|$. Standard in LLM training.

#### Summary: pretraining optimization

Loss = averaged CE with pad masking (and in SFT, mask non-reply tokens). Optimizer: AdamW, weight decay 0.01–0.1. LR: warmup + cosine/linear decay. Gradient clipping max_norm ≈ 1.0. **Mixed precision** (FP16/BF16) and **gradient checkpointing** to save memory. **Gradient accumulation:** run several small forward/backward without updating weights, sum grads, then one optimizer step → effective batch = batch_size × accumulation_steps. **Fused Cross-Entropy** computes loss without materializing the full logit matrix, saving VRAM. **Scale:** train on trillions of tokens; models from hundreds of millions to hundreds of billions of parameters. **Scaling laws** (Chinchilla): quality grows with data and model size; optimal tokens-per-parameter ~15–20. **Perplexity** $\mathrm{PPL} = \exp(L)$ where $L$ is average CE per token; lower loss → lower PPL. After pretraining the model can continue text and “knows” language but may not yet follow dialogue or instructions.

---

### Stage 2: SFT (Supervised Fine-Tuning)

**Goal:** teach dialogue/instruction format: input = user request (and optionally system prompt), output = assistant reply. **Data:** pairs (instruction/question → answer) or full dialogues (user/assistant/system); often a mix (general instructions, code, reasoning, safety). **Format:** text is assembled with a chat template (e.g. `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...`). **Loss** is computed only on **assistant reply** tokens (system and user tokens are masked). **Risks:** less data than pretraining → **overfitting** and **catastrophic forgetting**. Common mitigations: small LR, few epochs, **LoRA**/QLoRA (train only low-rank matrices, freeze the rest). After SFT the model is useful as a chatbot and follows instructions but may not yet reason step-by-step or satisfy complex criteria.

---

### Stage 3: RL and alignment (GRPO, PPO, DPO, etc.)

**Goal:** improve behavior on criteria hard to specify with labels only (e.g. “correct answer”, “good reasoning”, “safe answer”). **Idea:** model generates **several answers** per prompt; each gets a **reward** $r$ (from a reward model, rule, human, or combination). Training increases probability of high-reward answers and decreases low-reward (policy gradient). **GRPO:** process answers in groups; advantage $A_i = r_i - \bar{r}$ (or normalized). Policy gradient with clipping (PPO-style). **Regularization:** **KL penalty** to a reference (often SFT) model; **clip range** on policy ratio. Other methods: **DPO**, **KTO** (preference-based, no explicit reward model). Outcome: models that reason better (chain-of-thought), pass tests, write code, etc.

```
  One optimization step (general):

  Batch → Forward → Loss → Backward → gradients → (optional) clip → Optimizer.step → θ updated
```

---

## Inference: How the Model Generates Output

Generation proceeds one token at a time. At each step:

1. The current sequence (prompt + already generated tokens) is fed into the model.
2. Logits for the **last** position are computed (or for all positions, but only the last is used).
3. The next token is **sampled** from the distribution.
4. The token is appended to the sequence; repeat until EOS or length limit.

### Sampling

- **Greedy:** take the token with maximum logit. Deterministic; outputs are often “flat”.
- **Temperature:** divide logits by $T$ before softmax. $T \to 0$ → almost greedy; large $T$ → more uniform, more randomness. Typical 0.7–1.0 for creative tasks, 0.1–0.3 for precise answers.
- **Top-k:** keep only the $k$ tokens with highest logits, set the rest to $-\infty$, then softmax and sample. Limits “silly” rare tokens.
- **Top-p (nucleus):** sort tokens by probability and take the smallest set whose cumulative probability is $\ge p$, renormalize and sample. Adapts to the shape of the distribution.

**Repetition penalty:** during generation, logits of already generated tokens (in a window or the full reply) are penalized (reduced or zeroed) so the model repeats less. Typical values 1.1–1.2.

**Stop sequences:** list of strings (or token ids) at which generation stops (e.g. `["\n\n", "<|im_end|>"]`). Avoids wasting length on trailing text.

**Max new tokens:** upper bound on the number of tokens generated after the prompt; together with stop and EOS it limits output length.

Often temperature is combined with top-p or top-k for a balance of diversity and coherence.

```
  Step-by-step generation (with KV-cache):

  Step 1:  [prompt tokens]        → model → logits → sample → tok₁
  Step 2:  [prompt ... tok₁]     → model (reuse KV for prompt!) → sample → tok₂
  Step 3:  [prompt ... tok₁ tok₂] → model → ...
  Memory: KV-cache grows by one position per step (only for the new token).
```

### KV-cache

In autoregression, for the same prefix (prompt + generated so far), attention would recompute keys and values for **all** previous positions every time. To avoid that, **keys and values** for all layers and heads are stored in a cache (KV-cache). On the next step only the key and value for the new token are added. This greatly speeds up generation and reduces compute.

### Context length at inference

Memory and time grow with sequence length (quadratically for attention without optimizations, linearly with Flash Attention). So context is limited (e.g. 4k, 8k, 32k tokens). Models with long context (128k+) often use special tricks (sparse attention, grouping, etc.).

---

## Memory and Speed: What Matters in Practice

- **KV-cache:** size grows linearly with sequence length and number of layers; with long context and large batches it is the main consumer of VRAM at inference. Per layer: 2 (K and V) × batch × num_heads × seq_len × head_dim × sizeof(dtype). For 7B, 32 layers, seq=2048, bf16: on the order of gigabytes per batch.
- **Flash Attention:** reformulates attention so the full [L×L] scores matrix is not materialized in memory — block-wise computation and overwriting. Reduces memory traffic and uses the GPU better. Gives speedup and lower memory peaks; almost essential for long context.
- **Quantization:** weights (and optionally activations) stored in lower precision (INT8, INT4, NF4). Shrinks model size (2–4×) and speeds inference with small quality loss. QLoRA: base in 4-bit, LoRA in fp16 — train large models on limited VRAM.
- **Gradient checkpointing** (during training): do not store all intermediate activations in the forward pass; recompute them during backward as needed. Saves memory (sometimes significantly) in exchange for ~20–30% longer training.

**Rough VRAM estimate for training (fp16/bf16):** model ~2 bytes/parameter, optimizer (Adam) ~8 bytes/parameter, gradients ~2 bytes/parameter, activations depend on batch × seq × hidden × layers. For 7B with batch=1, seq=2048 without checkpoint: tens of GB; with gradient checkpointing and a small batch it can fit in 24 GB.

---

## Architecture Families (brief)

- **GPT-2 / GPT-3 style:** decoder-only Transformer, learned or sinusoidal positions, classic MLP in FFN. Base for many commercial and open models.
- **LLaMA / Mistral:** decoder-only, **RoPE**, **SwiGLU**, **RMSNorm**, **GQA**. De facto standard for open LLMs.
- **Qwen, Yi, others:** broadly similar to LLaMA with variations (vocab size, context length, normalization details).

Differences are mainly in positional encodings, FFN activations, normalization, and attention setup (GQA etc.). The overall pattern “embeddings → N blocks (attention + FFN) → LM head” stays the same.

---

## Typical Training Issues

- **Loss spike:** learning rate too high, “bad” batch, gradient explosion. Fix: lower LR, check data, enable or tighten gradient clipping.
- **Out of Memory (OOM):** not enough VRAM. Fix: reduce batch size, enable gradient checkpointing, use LoRA/QLoRA, reduce seq_len, use FSDP/DeepSpeed for multi-GPU.
- **Loss not decreasing / plateau:** LR too low, not enough data, or model already converged. Fix: increase LR (or restart with warmup), check data quality and size.
- **Overfitting (SFT):** validation loss rises while train loss falls. Fix: fewer epochs, stronger weight decay, more data, LoRA with small rank.
- **Catastrophic forgetting (SFT):** model “forgot” base knowledge after fine-tuning. Fix: smaller LR, fewer steps, LoRA instead of full fine-tune, mix in pretrain-like data.

---

## Summary

- **LLM** — autoregressive model over tokens; core is the **Transformer** (causal multi-head attention + FFN, normalization, residuals).
- **Tokenization** (BPE, SentencePiece) maps text to ids; **embeddings** to vectors; **positions** (RoPE, ALiBi, etc.) encode order.
- **Training:** **pretraining** (next-token prediction, cross-entropy loss) → **SFT** (instructions/dialogues, mask by reply) → **RL/alignment** (rewards, preferences). Optimizer usually AdamW, LR with warmup and decay, gradient clipping.
- **Inference:** sampling (temperature, top-k, top-p), repetition penalty, stop sequences, KV-cache, context length limit.
- To save resources: **LoRA** / **QLoRA**, **quantization**, **Flash Attention**, **gradient checkpointing**, **gradient accumulation**.

**Metrics:** loss (cross-entropy), perplexity = exp(loss). For SFT/RL also human ratings, reward, test pass rates.

This file gives a solid base; individual topics (RoPE details, attention implementation, specific RL algorithms, data and cleaning, model evaluation) can be covered in follow-up materials.
