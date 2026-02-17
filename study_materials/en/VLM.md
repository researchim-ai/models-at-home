# VLM: What You Need to Know About Structure and Training

A detailed study guide on Vision-Language Models (VLM): what they are made of, how they process images and text, how they are trained, and how to use them. The material is aimed at **complete beginners**: all terms are explained on first use, there is an extended glossary at the start, and step-by-step examples with numbers and diagrams throughout. If something is unclear, you can return to the glossary or related sections. The text “half” of VLM (how the answer is generated) is covered in [LLM.md](LLM.md); the basics of the attention architecture are in [transformer.md](transformer.md).

---

## What You Need to Know in Advance (Minimum)

To follow the material from scratch, this picture is enough:

- **A language model (LLM)** takes text (a prompt) and continues it one “chunk” at a time — a token (subword). Internally it works with **vectors** — fixed-length lists of numbers — not with letters. For more, see [LLM.md](LLM.md).
- **Vector** — a list of numbers, e.g. 768 or 4096. Networks take in and output such numbers; the “meaning” of a vector is learned during training.
- **Training a neural network** — adjusting parameters (weights) so that on examples “input → correct output” the loss gets smaller. Lower loss means better predictions (e.g. caption for an image).

If you have never worked with LLMs or neural networks, that’s fine: all needed concepts are introduced and explained step by step below. You can re-read the glossary as you go.

---

## Glossary of Terms

These terms appear often. You don’t have to memorize them — you can refer back to this table while reading.

| Term | Plain explanation |
|--------|---------------------|
| **VLM (Vision-Language Model)** | A model that takes **image(s) and/or text** and outputs **text** (answer, caption, description). It combines “vision” and “language” in one pipeline. |
| **Vision Encoder** | A network that turns an **image** into a set of **vectors** (embeddings). Usually ViT (Vision Transformer) or CNN; output is a sequence of “visual tokens”. |
| **Vector** | A fixed-length list of numbers (e.g. 768 or 4096). In VLM: representation of one image patch or one text token; the model always works with vectors. |
| **Patch** | A square piece of the image (e.g. 14×14 or 16×16 pixels) treated as one input “unit” for ViT. The image is split into a grid of patches; each becomes one vector. |
| **Image token (visual token)** | One vector after the vision encoder (or after projection). For the LLM inside the VLM it is the same kind of “token” as text, but it encodes a piece of the image. Count depends on resolution and patch size. |
| **Projection** | A layer (linear or MLP) that maps **vectors from the vision encoder** into the **LLM embedding space**. Needed because the encoder and LLM were trained in different spaces (different dimension and “meaning” of coordinates). |
| **Caption** | Text description of an image. Training data often uses (image, caption) pairs; the model learns to predict the caption from the image. |
| **Loss** | A single number measuring how far the model’s prediction is from the correct answer. Training = minimizing loss over many examples. VLM usually use token-level cross-entropy (see the training section and [LLM.md](LLM.md)). |
| **Freeze** | Not updating some part of the model during training. E.g. “freeze the encoder” — keep its weights fixed and train only the projection and LLM. Saves memory and avoids “breaking” a good encoder. |
| **Multimodal** | Using several input types: image + text, video + text, etc. VLM is a multimodal model. |
| **Q-Former** | A module (e.g. in BLIP-2): a small Transformer with learnable query vectors that “squeezes” the image encoder output into a **fixed** number of tokens for the LLM. Reduces visual token count and compute cost. |
| **Interleaved** | Alternating sequence: [image_tokens, text_tokens, image_tokens, text_tokens, ...]. Allows multiple images and text in one context. |
| **Image-text alignment** | Aligning images and text in a shared representation space. Achieved at pretraining (learning to predict captions from images). |
| **LoRA** | A way to fine-tune: add small low-rank matrices to some layers and train only those; freeze the main weights. Lets you adapt a large model with few parameters and less memory. See [LLM.md](LLM.md). |

---

## What Is a VLM

**VLM (Vision-Language Model)** is a model that takes **image(s)** and **text** (question, instruction, context) as input and outputs **text** (answer, caption, reasoning). So it’s an “LLM with eyes”: a module turns the image into a sequence of vectors that are fed into the same model as text tokens. Answer generation stays **autoregressive** — one token at a time, like a normal LLM.

```
  VLM overview: image + text → text

  Image                         Text (question / prompt)
     │                                │
     ▼                                ▼
  ┌─────────────┐             Tokenizer
  │   Vision    │                   │
  │   Encoder   │                   ▼
  └──────┬──────┘             [ tok₁  tok₂  ...  tok_m ]
         │                                │
         ▼                                │
  [ img₁  img₂  ...  img_n ]              │
         │                                │
         ▼                                │
  ┌─────────────┐                         │
  │  Projection │                         │
  └──────┬──────┘                         │
         │                                │
         ▼                                ▼
  [ v₁   v₂   ...  v_n ]  +  [ t₁   t₂   ...  t_m ]
         │                                │
         └──────────────┬─────────────────┘
                        ▼
              Combined sequence
              [ v₁ ... v_n  t₁ ... t_m ]  (or interleaved)
                        │
                        ▼
              ┌─────────────────┐
              │   LLM (as in    │  ← same Transformer blocks as in LLM
              │   LLM.md)       │
              └────────┬────────┘
                       ▼
              Generate answer one token at a time (autoregression)
```

### Why VLMs Are Used

- **Image QA:** “What’s in the photo?”, “How many objects?”, “Describe the details.”
- **OCR and tables:** extract text from images, answer from charts and diagrams.
- **Visual reasoning:** explain or solve from a graph, diagram, or code screenshot.
- **Robotics and driving:** understand the scene from a camera frame.

VLMs don’t replace text-only LLMs but extend them to the visual world (and video if frames are treated as images).

### Why Not Feed the Image Directly Into the LLM?

One might ask: the LLM handles token sequences, so why not turn each pixel (or each image patch) into a “token” and feed that?

Reasons:

1. **Size.** A 224×224 RGB image is 224×224×3 = **150,528 numbers**. LLMs are built for sequences of thousands of tokens (e.g. 2k–32k), each token being one vector (e.g. 4096 numbers). Treating each pixel as a position would make the sequence huge: the LLM isn’t trained for that length, and memory/time would explode.

2. **Different “language”.** The LLM is trained on **text** embeddings: vector coordinates encode meaning of words and subwords. Raw pixels (R, G, B per point) live in a different space with no “vocabulary” the LLM knows. So even if we fed pixels in, the model wouldn’t use them like text.

3. **Feature hierarchy.** The image should first be “understood”: edges, objects, layout. That’s what the **vision encoder** does — it turns pixels into higher-level representations (patch vectors). Those can then be aligned to language via **projection** and fed to the LLM as “image tokens”.

So: we first map the image to a limited number of vectors (encoder + optional compression), then map those to the LLM’s “language” (projection), and only then feed them to the LLM together with text.

---

## Architecture: What a VLM Is Made Of

Three main parts:

1. **Vision Encoder** — turns the image into a sequence of vectors.
2. **Projection** — maps those vectors into the LLM embedding space.
3. **LLM** — same architecture as a language model (Transformer decoder): takes the combined “visual” + text tokens and generates text.

Below we go through each part.

---

### Vision Encoder: From Pixels to Vectors

**Goal:** from a pixel matrix (e.g. H×W×3) get a sequence of vectors of fixed dimension (e.g. 768 or 1024). These will become “visual tokens” for the LLM. Reminder: a **vector** is just a list of numbers of fixed length; the network transforms vectors at each step.

#### Why Use Patches Instead of the Whole Image?

- A 224×224×3 image gives **150,528** numbers. Processing such a long sequence in a Transformer is expensive (memory and time grow quadratically with length). Splitting into 16×16 patches gives **196 pieces** of 16×16×3 = 768 numbers each — 196 “elements” instead of 150k. Each patch is then turned into one vector (e.g. length 1024). So we get **196 vectors** — a length the model can handle.
- Patches keep **locality**: one vector describes a small region (corner of an eye, part of a wheel, etc.). The Transformer then links these via self-attention into a full picture.

#### Approach 1: Vision Transformer (ViT)

The most common choice in modern VLMs.

**Steps in order:**

1. **Split into patches.** The image is cut into square patches of size P×P (typically 14×14 or 16×16). E.g. 224×224 with P=16 gives a 14×14 grid = **196 patches**. Each patch is a small 16×16×3 image (768 numbers if flattened).

2. **Patch linear projection (Patch Embedding).** Each patch is flattened to a vector of length P×P×3 (e.g. 768), then passed through a linear layer to dimension `encoder_hidden_size` (e.g. 768 or 1024). Result: one vector per patch.

3. **Positional embeddings.** As in the Transformer, order matters. Learnable positional embeddings (or 2D positions) are added to the patch vectors.

4. **Optional [CLS] token.** In classic ViT one special token is prepended; its final vector is often used for classification. In VLMs, sometimes all patch tokens are used without [CLS].

5. **Transformer blocks.** The sequence is passed through several self-attention + FFN layers (see [transformer.md](transformer.md)). Output has the same length.

```
  ViT: image → sequence of vectors

  Image 224×224×3
        │
        ▼  split into 16×16 patches
  ┌────┬────┬────┬───┐
  │ p₁ │ p₂ │ p₃ │...│  14×14 = 196 patches
  ├────┼────┼────┼───┤
  │    │    │    │   │
  └────┴────┴────┴───┘
        │
        ▼  Linear: each p_i (768) → vector (1024)
  [ v₁   v₂   v₃  ...  v₁₉₆ ]  + positional embeddings
        │
        ▼  Transformer blocks (attention + FFN)
  [ h₁   h₂   h₃  ...  h₁₉₆ ]   vision encoder output
```

**Resolution and token count:** at 336×336 and P=14 you get (336/14)² = 576 tokens. Higher resolution means more visual tokens and higher cost for the LLM (memory and time grow with sequence length). So 224–336 on the short side or adaptive splitting is common.

**Concrete numbers:** image 224×224, patches 16×16 → 196 patches. After the linear layer each patch → vector of length 1024 (encoder_hidden_size). After Transformer blocks the encoder output is **196 vectors of 1024 numbers** — the “raw” image representation. These are then projected into the LLM space and placed at the start of the sequence instead of “text” tokens.

#### Approach 2: CNN (rare in new VLMs)

Early models sometimes used ResNet or another CNN: the feature map was either globally pooled (one vector per image) or split into regions. Most current open VLMs use ViT (or variants: ViT-L, ViT-g, SigLIP, etc.).

#### Approach 3: Encoder from CLIP / SigLIP

Many VLMs don’t train the encoder from scratch but use a **frozen** vision encoder from CLIP or SigLIP — models already trained for image–text alignment. Pros: strong visual representations, fewer parameters to train. Cons: the encoder isn’t tailored to the VLM task; it can be unfrozen and fine-tuned if needed.

**Summary:** the vision encoder outputs a sequence of N vectors of dimension `encoder_hidden_size` (e.g. N=256, dim 1024). These are the raw visual representations.

---

### Projection: Visual Vectors Into the LLM Space

**Why not feed the vision encoder output straight into the LLM?**

- **Different dimension.** The encoder often outputs vectors of length 768, 1024, or 1280 (its `encoder_hidden_size`). The LLM uses vectors of length e.g. 4096 (its `hidden_size`). Every input position must have that length — otherwise attention and linear layers don’t match.
- **Different “language”.** Even with matching dimensions, encoder vectors encode visual features (edges, textures, objects) while LLM embeddings encode **word** meaning. To let the LLM “understand” the image in the same context as text, we map visual vectors into the same space as text embeddings. That’s what **projection** does: a linear layer (or MLP) maps each encoder vector to a vector of the LLM’s `hidden_size`. After projection, visual tokens look “native” to the LLM — same size and, after training, aligned meaning.

**Projection** is a layer (or small network) that maps each encoder vector to the LLM’s **hidden_size**. Then visual tokens can be placed at the start of the sequence with text tokens.

**Options:**

1. **Single linear layer:** `proj(x) = x W + b`, W of size `encoder_hidden_size × hidden_size`. Fast and few parameters.

2. **MLP (2–3 layers):** several linear layers with activation (GELU, SiLU). More flexible, a few more parameters.

3. **Q-Former (BLIP-2):** not just linear. A set of **learnable query vectors** (e.g. 32) is introduced. They go through a Transformer with cross-attention to the vision encoder output. So N visual vectors are compressed to a **fixed** number of vectors (one per query), then projected into the LLM space. Pros: strong compression (many patches → few tokens), lower cost. Cons: more complex and more parameters.

4. **Resampler (in some recent models):** similar idea to Q-Former — compress many visual tokens to a fixed number via attention with learnable queries.

```
  Projection: two options

  Option A (linear / MLP):
  [ h₁  h₂  ...  h_N ]  (encoder_hidden_size)
         │
         ▼  for each h_i: Linear or MLP
  [ v₁  v₂  ...  v_N ]  (LLM hidden_size)   ← N visual tokens

  Option B (Q-Former / Resampler):
  [ h₁  h₂  ...  h_N ]  (encoder output)
         │
         ▼  learnable queries + cross-attention to h₁..h_N
  [ q₁  q₂  ...  q_K ]  (K fixed, e.g. 32)
         │
         ▼  Linear
  [ v₁  v₂  ...  v_K ]  (LLM hidden_size)   ← fewer tokens (K < N)
```

After projection, visual tokens are **formally the same** for the LLM as text tokens: a sequence of hidden_size-dimensional vectors. They get positions (often consecutive at the start), and then take part in self-attention and FFN like any other token.

---

### LLM: Shared With the Language Model

The text part of a VLM is the **same architecture** as a normal LLM: tokenizer, token + positional embedding, stack of Transformer blocks (causal self-attention + FFN), LM head. Details in [LLM.md](LLM.md).

**The only difference is where the input sequence comes from:**

- In a pure LLM: only text tokens (embeddings from the vocabulary).
- In a VLM: first **visual tokens** (from projection), then **text** (question, instruction, system prompt, etc.). The model “sees” both image and text in one context and generates the answer autoregressively.

```
  Input to the LLM inside a VLM:

  Positions:  1    2   ...   N    N+1   N+2   ...   N+M
  Content:    v₁   v₂  ...  v_N   t₁    t₂    ...  t_M
              ↑________________↑   ↑____________________↑
              visual tokens      text tokens (question)
              (projection)       (vocab embedding)

  Then — as in an LLM: causal attention, FFN, LM head; generate one token at a time.
```

**Special tokens for images:** in chat models, tokens like `<image>` or `<img>` are often used; in the sequence they are replaced by the corresponding visual tokens (several positions per image). That keeps the dialog format uniform: the user can write “Describe this: <image>” and the model knows where to put the image embeddings.

---

## How the Image Becomes Tokens (Full Pipeline)

1. **Resize and normalize.** The image is resized to what the encoder expects (e.g. 224×224 or 336×336). **Normalization** — scale pixel values (usually 0–255) to the range the encoder was trained on: subtract mean and divide by std per channel (R, G, B), often using ImageNet constants. So the encoder gets input in its expected range.

2. **Patches + ViT.** Split into patches → patch embedding → positions → Transformer blocks. Output: N vectors (N = (H/P)×(W/P) at one scale).

3. **Projection.** N vectors → N (or K with Q-Former) vectors of the LLM’s hidden_size.

4. **Merge with text.** Text is tokenized; text embeddings come from the token embedding table. In the sequence, visual tokens come first, then text (or interleaved if there are multiple images). Position indices are assigned in order (visual 1..N, text N+1..N+M, etc.).

5. **Positional embeddings.** RoPE (or similar) is applied to the combined sequence by position — same as in the LLM.

Then — standard LLM forward and answer generation.

```
  Full VLM pipeline (one forward pass)

  Image [H×W×3]
        │
        ▼  resize, normalize
  ┌─────────────┐
  │ Vision Enc │  ViT: patches → [h₁..h_N]
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Projection  │  [h₁..h_N] → [v₁..v_K]  (K=N or K fixed)
  └──────┬──────┘
         │
  Text "Describe the image."
         │
         ▼  tokenizer → token embedding
         [ t₁  t₂  ...  t_m ]
         │
         ▼  concatenation
  [ v₁  ...  v_K  t₁  ...  t_m ]
         │
         ▼  + positional embedding (RoPE etc.)
  ┌─────────────────────────────────┐
  │  Transformer blocks (LLM)       │
  └─────────────────────────────────┘
         │
         ▼  LM head at last position
  Logits → sample → next answer token → autoregress until EOS
```

---

## Walkthrough: One Example Step by Step

To tie everything together, we follow one request from start to finish: one image and one question.

**Input:** image of a cat on a sofa (file `cat.jpg`) and the user question: “What’s in the photo?”

1. **Prepare the image.** Load the image, resize to 224×224 (or whatever the model expects), normalize pixels (subtract mean, divide by std — often ImageNet stats). Output: matrix 224×224×3.

2. **Vision encoder.** Split image into 16×16 patches → 196 patches. Each patch → vector of 1024 (patch embedding + positions), run through ViT Transformer blocks. We get **196 vectors of length 1024** — the “visual” representations of image regions.

3. **Projection.** Each of the 196 vectors goes through a linear layer (or MLP): 1024 → 4096 (if LLM hidden_size = 4096). We get **196 vectors of length 4096** — same space as the LLM’s text embeddings.

4. **Question text.** The string “What’s in the photo?” is tokenized (e.g. “What” “ ’s” “ in” “ the” “ photo” “?”). Each token gets an id from the vocab; the token embedding table gives vectors of length 4096. Say we get 4 tokens → 4 vectors.

5. **Build the sequence.** Concatenate: 196 visual tokens first, then 4 text tokens. **200 positions** in total, each a 4096-dim vector. Assign position indices 1…200 and apply positional embeddings (e.g. RoPE), as in a normal LLM.

6. **LLM forward.** These 200 vectors go through the Transformer stack (causal self-attention + FFN). At the **last** position (200 — end of “What’s in the photo?”) we get the contextual representation. The LM head produces logits over the full vocabulary (tens of thousands of numbers).

7. **Generate the answer.** Convert logits to a token distribution (softmax), sample the first answer token — e.g. “The”. Add it to the sequence (now 201 positions), run forward again (or use KV-cache and compute only for the new position), get the next token, and so on until EOS or length limit. Final answer: “The photo shows a cat sleeping on a sofa.” (or similar).

Important: at steps 6–7 the model “sees” all 196 visual tokens, the question text, and the already generated answer tokens — everything in one sequence. Attention lets each position look at all previous ones, including image tokens, so the answer can rely on the image content.

```
  One request: numbers for reference

  Image 224×224  →  ViT  →  196 vectors × 1024
                                →  projection  →  196 vectors × 4096
  "What's in the photo?" → tokens → 4 vectors × 4096

  Concatenation: [ 196 visual | 4 text ] = 200 positions × 4096
       ↓
  Transformer (LLM)  →  at position 200: logits over vocab
       ↓
  Sample token  →  "The"  →  append to sequence  →  repeat
       ↓
  Answer: "The photo shows a cat sleeping on a sofa."
```

---

## How VLMs Are Trained

Training usually has several stages, similar to LLMs: first image–text alignment, then instructions/dialogue, and optionally RL/alignment from preferences.

---

### Stage 1: Pretraining (image–text alignment)

**Goal:** teach the model to link image and text. Usually **caption prediction**: input image, output caption. The model learns to predict the next caption token given the image and previous caption tokens. **Loss** is cross-entropy over caption tokens: at each position we compare the model’s predicted distribution over the vocab with the true next token; the closer the prediction to the correct token, the lower the loss. Sum over all caption positions and all batch examples, then average — one number. Training = adjust weights to decrease this number. For cross-entropy and next-token prediction see [LLM.md](LLM.md).

**What one training example looks like:**

- A pair: **image file** (e.g. `train_001.jpg`) and **text caption** (e.g. “A cat sleeping on a red sofa by the window”). The dataset is a large list of such pairs. A batch might have 32 pairs: 32 images and 32 captions. Each image goes through the vision encoder and projection; each caption is tokenized. The sequence for the LLM is: [visual tokens for this image] + [caption tokens]. The model learns to predict the next caption token; loss is computed **only on caption positions** (not on visual tokens — we are not predicting the image from text).

**Data:**

- Large (image, text) datasets: COCO, LAION, Conceptual Captions, in-house captions, etc.
- Text can be short captions or longer descriptions. Quality and diversity of captions matter a lot.

**What is trained (what is frozen):**

- **Option A:** train only the **projection**; vision encoder and LLM **frozen**. Fast and memory-cheap; the model only learns to “translate” visual vectors into the LLM space. Often used when encoder and LLM are already strong (e.g. CLIP + LLaMA) and only need to be “connected”.
- **Option B:** train projection and **LLM**; vision encoder frozen. The LLM is fine-tuned to predict captions from visual tokens. More data and steps, but usually better quality.
- **Option C:** train all three (encoder + projection + LLM). Maximum flexibility but more overfitting risk; needs more compute and a smaller learning rate for the encoder.

**Input format:**

- Image → vision encoder → projection → visual tokens.
- Caption text is tokenized. Sequence: [visual tokens] + [caption tokens]. Loss is **only on caption tokens** (causal mask: when predicting the i-th caption token the model sees visual tokens and previous caption tokens). Same masking idea as in LLM SFT: positions before the caption don’t contribute to loss.

```
  Pretraining VLM: one example

  Image → [ v₁  v₂  ...  v_K ]
  Caption:     "A cat sleeping on a sofa."
  Tokens:        t₁   t₂   t₃   t₄   t₅   t₆   t₇

  LLM input:   [ v₁ ... v_K  t₁  t₂  t₃  t₄  t₅  t₆  t₇ ]
  Target (loss):         t₂  t₃  t₄  t₅  t₆  t₇  t₈(EOS?)
  (mask: loss only on caption positions, not on v_i)
```

After this stage the model can generate a caption from an image but may not yet answer in “Question: … Answer: …” or chat format.

---

### Stage 2: SFT (Supervised Fine-Tuning) on instructions / dialogue

**Goal:** teach “image question → answer” or multi-turn dialogue with images. After pretraining the model can caption but may not follow “User: … Assistant: …”. SFT teaches that.

**What one example looks like:**

- Triple: **image** + **question/instruction** (user text) + **correct answer** (assistant text). E.g. street photo, question “Are there people in the photo?”, answer “Yes, two people on the sidewalk.” In code/datasets this is often one row with image path and two text fields. When building the batch, the image goes through the encoder and projection; text is assembled into one sequence with the model’s **chat template** (system prompt, user, assistant, special tokens). Where the template has an image placeholder (e.g. `<image>`), the visual tokens are inserted.

**Data:**

- Sets of (image, question/instruction, answer) or full dialogues with embedded images. E.g. LLaVA-style data, VQA datasets, description/reasoning/OCR instructions.

**Sequence format and loss:**

- Sequence is built like in a chat model: e.g. `<image>` (replaced by visual tokens) + user text + assistant answer. **Loss is only on assistant answer tokens** — system prompt and user question are masked out (0 loss), otherwise the model would learn to “predict” the question from the image instead of answering. Same as in text-only chat SFT — see [LLM.md](LLM.md).

**Hyperparameters:**

- Learning rate usually lower than in pretraining (e.g. 1e-5 … 2e-5).
- Often only the projection and part of the LLM are fine-tuned, or **LoRA** on the LLM, to preserve language ability and save memory.

After SFT the model is useful as an “image chat”: it answers questions about images, describes, reasons.

---

### Stage 3: RL and alignment (optional)

Same idea as for LLMs: GRPO, DPO, PPO, etc. on (image, question, several answers, preference or reward). Goal: improve style, completeness, safety. Details are as in [LLM.md](LLM.md) (stage 3); the only difference is that the input also includes the image.

---

## Inference: How to Use a VLM

**In short:** prepare image(s) and text prompt → get visual tokens → build sequence [visual + text tokens] → run the LLM and generate the answer one token at a time until EOS or length limit.

**Step by step: what the user does and what happens inside**

1. **User:** uploads an image (file or camera) and enters text, e.g. “Describe what’s in the image” or “How many people are here?”
2. **System:** resizes and normalizes the image, runs it through the vision encoder → N vectors (e.g. 196). Runs them through the projection → N vectors in the LLM space. Text is tokenized; token embeddings are taken from the vocabulary. One sequence is built: first N visual tokens, then the user prompt tokens (and optionally system prompt and chat special tokens — depends on the model format).
3. **First forward:** this sequence is fed to the LLM. At the **last** position (end of prompt) the model outputs logits over the vocabulary. From these, a probability distribution (softmax) is computed and, with parameters (temperature, top-p, etc.), **one token is sampled** — the first word (or subword) of the answer.
4. **Generation loop:** this token is appended to the sequence. The LLM is run again (often only for the new position, reusing KV-cache for all previous ones). The next token is obtained. Repeat until the end-of-sequence token (EOS) or max length. The answer token sequence is decoded back to text and shown to the user.

Sampling (temperature, top-k, top-p), repetition penalty, stop sequences work as in a normal LLM (see [LLM.md](LLM.md)). **KV-cache** stores keys and values for all already processed positions (including visual and prompt text) so they are not recomputed during generation — this greatly speeds up decoding.

**Multiple images:** in interleaved format the sequence is e.g. [img1_tokens] [text1] [img2_tokens] [text2] … The model distinguishes images by position and context. Support for multiple images depends on the trained format and chat template (see the model’s documentation).

---

## Fine-Tuning and Adapting a VLM

If you want to **fine-tune** a pretrained VLM for your task (e.g. your image domain or answer format):

### Full fine-tuning

All parameters (vision encoder + projection + LLM) are trained. Maximum flexibility but needs many GPUs, lots of data, and risks catastrophic forgetting. Rarely done without a large dataset.

### LoRA / QLoRA on the LLM

As with LLMs: add low-rank LoRA matrices to attention (and optionally FFN) layers; train only those, freeze the base. Vision encoder and projection can stay frozen or be fine-tuned with a small LR. This is the standard way to adapt a VLM to your data with limited VRAM.

### Projection only

If the base model already understands images reasonably well, sometimes it’s enough to fine-tune only the projection layer on your examples (image + desired answer). Very cheap in parameters and memory.

### Unfreezing the vision encoder

For domain-specific data (medical, satellite, diagrams) it can help to unfreeze the vision encoder and fine-tune it together with the projection (and optionally LoRA on the LLM) with a small learning rate. Then the encoder adapts to your visual domain.

**Practical tips:**

- Use **gradient checkpointing** during training to fit the batch in memory.
- **Gradient accumulation** increases effective batch size without raising VRAM per step.
- Save checkpoints and monitor validation loss; if overfitting, reduce epochs or increase regularization.

---

## Memory and Speed

- **Vision encoder:** one forward per image; with a batch of images the encoder runs for each. Memory grows with resolution (number of patches) and encoder size (ViT-L, ViT-g are heavier).
- **Projection:** few parameters compared to the LLM; cost is small.
- **LLM:** main cost is as in a text-only LLM: sequence length = visual tokens + text tokens. More visual tokens (higher resolution, more images) → more memory and time. **KV-cache** at generation stores keys/values for the full sequence, including visual positions.
- **Quantization:** as with LLMs, weights can be quantized (INT8, INT4, QLoRA) for faster inference and less memory, including when fine-tuning.

---

## Model Families and Examples

- **LLaVA:** ViT (CLIP) + linear projection + LLaMA. Simple design; many variants (LLaVA 1.5, LLaVA-NeXT) with different resolutions and sizes.
- **BLIP-2:** frozen vision encoder + **Q-Former** + frozen LLM (at first stage); then the LLM is unfrozen. Strong compression of visual tokens.
- **Qwen-VL, Yi-VL:** commercial/open VLMs with multi-image support, variable resolution, and chat format.
- **InternVL, CogVLM:** heavier encoders and/or projections, long context.
- **PaliGemma, SmolVLM:** small VLMs for cheap inference and experiments.

Differences are mainly in: choice of vision encoder (ViT, SigLIP, etc.), use of Q-Former/Resampler, LLM size, and trained format (single vs multiple images, resolution, chat template).

---

## Common Issues in Training and Use

- **Model “ignores” the image, answers from text only:** weak projection or insufficient image–text training. Fix: ensure visual tokens are actually fed to the LLM (no bug skipping the image); fine-tune the projection on data where the correct answer clearly depends on the image; increase number of visual tokens or pretraining quality.
- **OOM (Out of Memory):** not enough VRAM. Fix: reduce batch size, input resolution, number of visual tokens; enable gradient checkpointing; use LoRA and quantization (INT8/INT4, QLoRA when fine-tuning).
- **Poor quality on your data:** model was trained on general images, yours are specific (medical, satellite, diagrams). Fix: fine-tune on in-domain examples (LoRA + projection or unfreeze encoder); check caption quality and data format (correct chat template, proper `<image>` placement).
- **Slow inference:** reduce input resolution, use a quantized model, or a model with fewer visual tokens (Q-Former/Resampler); batch requests when serving.

---

## How to Tell If the Model Really Uses the Image

When you start using a VLM or have fine-tuned one, it helps to check that answers depend on **image content**, not only on the question text.

- **Swap test:** ask the same question (“What is in the image?”) for two different images (e.g. cat vs car). Answers should differ. If the model gives the same or very generic answers for both, it may be underusing the image.
- **Detail questions:** “What color is the person’s shirt in the photo?”, “How many windows on the building?” — if the model gives plausible answers, it is using visual tokens. If it always deflects or is wrong, check that the image is fed correctly and consider fine-tuning.
- **When fine-tuning:** on a validation set, look not only at loss but at sample answers for the same images across epochs — answers should become more relevant to the image.

---

## FAQ

**Do I need to know LLMs and Transformers before reading this?**  
It helps to have a rough idea: an LLM generates text token by token, using vectors and attention blocks. If this is your first time with VLMs, you can still read in order: the glossary and “What is a VLM” give the minimum; you can refer to [LLM.md](LLM.md) and [transformer.md](transformer.md) when needed.

**I don’t program / don’t know Python. Can I use VLMs?**  
You can use ready-made VLMs (web UIs, APIs, apps) without programming. To **fine-tune** or read code you need basics: Python, handling data (images + text), running scripts and editing config. This material explains what goes on inside; implementation differs by framework (transformers, LLaMA-Factory, etc.) — see their docs.

**How much data do I need to fine-tune a VLM?**  
It depends. Projection only on a good base: sometimes a few thousand (image, caption/answer) pairs suffice. LoRA on the LLM for your answer format: often a few thousand to tens of thousands of examples. Full fine-tuning or encoder tuning for a narrow domain usually needs more data and care to avoid overfitting.

**How is a VLM different from “plain” image recognition (classification, detection)?**  
Classification outputs a label from a fixed set (e.g. “cat”, “dog”); detection outputs boxes and classes. A VLM outputs **free-form text**: description, answer to a question, reasoning. So one model can describe a scene, answer questions about an image, or do OCR — depending on the prompt.

**Why do some models “compress” the image to 32 tokens (Q-Former) instead of 196?**  
To shorten the sequence for the LLM: fewer positions → less memory and time (attention cost grows quadratically with length). The Q-Former learns to “squeeze” 196 (or more) visual vectors into 32 informative ones. For many tasks that’s enough; for fine detail (small text, many objects) people sometimes prefer models with more visual tokens or higher resolution.

---

## Summary

- **VLM** = **Vision Encoder** (usually ViT) + **Projection** + **LLM**. The image becomes a sequence of vectors (visual tokens), concatenated with text tokens and fed into the same Transformer as in a language model.
- **Training:** pretraining (caption from image, cross-entropy on caption tokens) → SFT (image QA, loss on answer) → optional RL/alignment. Often the encoder or LLM is frozen at the first stage.
- **Use:** image + text → visual tokens + text tokens → one context → autoregressive answer generation. Generation settings (temperature, top-p, stop) are as in an LLM.
- **Fine-tuning:** in practice, LoRA on the LLM + optionally projection or unfreezing the encoder; full fine-tuning when you have enough compute and data.

This material is written so that someone with no prior VLM experience can understand what the model is, how it’s built, how it’s trained, and how to use and fine-tune it. If something is still unclear, return to the glossary or the relevant section (architecture, training, inference); for fine-tuning and using specific models, rely on their documentation and code examples.
