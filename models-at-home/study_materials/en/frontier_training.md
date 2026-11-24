# Training Frontier LLMs: A Detailed Guide to Building Thinking AI and Autonomous Agents

The landscape of large language models (LLMs) has shifted from simple chatbots to thinking models and autonomous agents. Models such as **GLM-5**, **OLMo 3**, **Kimi K2.5**, **DeepSeek-R1**, and **gpt-oss-120b** illustrate the transition from the *vibe coding* paradigm (writing code from step-by-step prompts) to *agentic engineering*—where AI autonomously plans architecture, writes code, fixes errors in the terminal, and tests solutions in real execution environments.

This in-depth study material describes the full lifecycle (Model Flow) of building frontier LLMs: from architectural choices and optimizers to multi-stage data curation, RLVR (reinforcement learning with verifiable rewards), and building scalable infrastructures for simulating agentic environments.

The material draws on technical reports from **GLM-5**, **OLMo 3**, **Kimi K2.5**, **Hermes 4**, **SmolLM3**, **Intellect-3**, and other open projects.

---

## Table of Contents
1. [Architecture and Stability](#1-architecture-and-stability)
    - [Dense vs MoE and Load Balancing](#dense-vs-moe-and-load-balancing)
    - [Attention Optimization (MHA, GQA, MLA, DSA)](#attention-optimization-mha-gqa-mla-dsa)
    - [Positional Encodings and Long Context](#positional-encodings-and-long-context)
    - [Logit Stabilization and Loss Terms](#logit-stabilization-and-loss-terms)
2. [Optimizers and Hyperparameters](#2-optimizers-and-hyperparameters)
    - [AdamW vs Muon](#adamw-vs-muon)
    - [Learning Rate Schedule (WSD) and Batch Size](#learning-rate-schedule-wsd-and-batch-size)
3. [Stage 1: Pre-Training](#3-stage-1-pre-training)
    - [Data Mixing and Filtering](#data-mixing-and-filtering)
    - [Quality-Aware Upsampling](#quality-aware-upsampling)
4. [Stage 2: Mid-Training](#4-stage-2-mid-training)
    - [Context Extension](#context-extension)
    - [Synthetic Data and Reasoning](#synthetic-data-and-reasoning)
    - [Decontamination (Leak Prevention)](#decontamination-leak-prevention)
5. [Stage 3: Post-Training](#5-stage-3-post-training)
    - [Supervised Fine-Tuning (SFT) and Chat Formats](#supervised-fine-tuning-sft-and-chat-formats)
    - [Preference Optimization (DPO) and Delta Learning](#preference-optimization-dpo-and-delta-learning)
    - [Reinforcement Learning (RLVR and GRPO)](#reinforcement-learning-rlvr-and-grpo)
    - [RL Infrastructure: Inflight Updates and Continuous Batching](#rl-infrastructure-inflight-updates-and-continuous-batching)
    - [Cross-Stage Distillation](#cross-stage-distillation)
6. [Agentic Engineering and Environment Scaling](#6-agentic-engineering-and-environment-scaling)
    - [Software Engineering (SWE) and Terminals](#software-engineering-swe-and-terminals)
    - [Context Management for Agents (Keep-recent-k)](#context-management-for-agents-keep-recent-k)
7. [Training Ops: Engineering Pitfalls](#7-training-ops-engineering-pitfalls)
8. [Additional Details from Technical Reports](#additional-details-from-technical-reports)

---

## 1. Architecture and Stability

Architectural decisions form the foundation of model performance. Mistakes at this stage are costly, so the industry favors proven setups and changes configuration only after rigorous ablation studies.

### Dense vs MoE and Load Balancing
*   **Dense:** The default when GPU memory is limited or timelines are tight. Dense models are easier to train and exhibit fewer unpredictable loss spikes.
*   **MoE (Mixture of Experts):** Used for extreme scale (e.g. **GLM-5** with 744B parameters or **Kimi K2** with 1T). MoE activates only a small subset of experts per token (e.g. 8 out of 256), reducing inference FLOPs.
*   **Sparsity:** Models increase the number of experts (e.g. 384 in Kimi K2) while shrinking expert size (granularity), improving efficiency leverage. **Shared Experts** are always active and handle basic language patterns; routed experts specialize on narrower tasks.
*   **Load Balancing:** Uneven expert usage degrades the model. Historically, *auxiliary loss* was used. Modern approaches include:
    1. **Sequence-wise auxiliary loss:** Balancing over the full token sequence, not just a single batch.
    2. **Auxiliary-loss free load balancing:** In **DeepSeek-V3** and **Kimi K2**, dynamic bias vectors for router logits are updated instead of adding loss terms. Underused experts get higher bias to attract more tokens.
    3. **SMEBU (Sequence-wise MoE Balancing with Uniformity):** An advanced bias-update scheme using momentum and `tanh` for stable balancing.

### Attention Optimization (MHA, GQA, MLA, DSA)
*   **MHA (Multi-Head Attention)** gives strong quality but produces a large KV cache (memory-heavy at inference).
*   **GQA (Grouped Query Attention):** A compromise. Keys and values are grouped (e.g. 2, 4, or 8). In ablations **GQA-8** often beats MHA on cost-quality tradeoff.
*   **MLA (Multi-Latent Attention):** A DeepSeek innovation that compresses the KV cache into a latent space (e.g. 576 dims). **GLM-5** found vanilla MLA underperformed GQA-8; the fix was **Muon Split** (per-head block orthogonalization).
*   **DSA (DeepSeek Sparse Attention):** For very long contexts (128K+), even GQA is expensive. GLM-5 uses DSA, which uses an indexer to dynamically select top-k important tokens, cutting attention cost by ~1.5–2× without accuracy loss (e.g. on NIAH—Needle In A Haystack).
*   **Gated Attention:** Adding gating to attention outputs reduces “attention sinks” and sharply reduces loss spikes.

### Positional Encodings and Long Context
Transformers have no built-in notion of token order.
*   **RoPE (Rotary Position Embedding):** De facto standard. Encodes position by rotating Q and K in 2D planes.
*   **YaRN and ABF (Adaptive Base Frequency):** To extend context (4K → 128K), the rotation base frequency is scaled up.
*   **RNoPE (Rotary + No Position Embedding):** A hybrid (e.g. in SmolLM3): some layers use RoPE (local context), others use NoPE for global sequence perception.
*   **Document Masking:** When packing multiple short documents into one long batch (sequence packing), tokens from one document must not attend to another. This is **critical** when extending the window to 64K+.

### Logit Stabilization and Loss Terms
Deep networks are prone to gradient explosion.
*   **Logit Softcapping:** First widely used in Gemma 2 and now an industry standard. Pre-softmax logits (attention and LM head) are compressed with `tanh`:
    $$ \text{logits} \leftarrow \text{soft\_cap} \cdot \tanh(\text{logits} / \text{soft\_cap}) $$
    This smoothly bounds logits (e.g. in [-50, 50]), preventing fatal spikes.
*   **No Weight Decay on Embeddings:** Regularizing embeddings shrinks their norm, which can cause gradient explosion in early layers due to LayerNorm/RMSNorm. Best practice is to disable weight decay (L2) for embedding parameters.

---

## 2. Optimizers and Hyperparameters

### AdamW vs Muon
*   **AdamW** with $\beta_1=0.9$, $\beta_2=0.95$ remains the default.
*   **Muon (matrix update algorithm):** A newer optimizer that updates parameters at the matrix level via a Newton–Schulz approximation of the matrix sign. It reduces axis-aligned bias and improves sample efficiency on large batches. **GLM-5** uses **Muon Split** for MLA (block-wise orthogonalization). Muon requires heavy engineering (e.g. All-to-All communication) because it needs full gradient matrices, not shards as in ZeRO/FSDP. Kimi K2 adds **MuonClip** to cap exploding logits.

### Learning Rate Schedule (WSD) and Batch Size
Classical cosine decay is often replaced by **WSD (Warmup–Stable–Decay)**. In WSD, the learning rate stays on a high plateau for most of training and drops sharply only in the last 10–20% of steps. WSD makes ablations easy: you can retrain only the final decay phase on different data mixes, saving huge compute.

*Batch size:* As the model stabilizes, gradients become less noisy. **Batch size warmup** is used: start with a smaller batch and increase it toward the end of pre-training.

---

## 3. Stage 1: Pre-Training

Pre-training consumes 80–90% of compute. Success is driven by the data mixture. Models like OLMo 3 and GLM-5 are trained on trillions of tokens (GLM-5: 28.5T).

### Data Mixing and Filtering
Raw data (Common Crawl, DCLM) is heavily cleaned:
1. **Heuristics and spam filters:** Remove HTML, lists, adult content; score by length vs unique-word ratio.
2. **Exact and fuzzy deduplication:** Remove exact clones (hashing) and near-duplicates via **MinHash**. Suffix-array–style methods remove boilerplate and footers.
3. **Decontamination:** Crucial. Remove all 8-grams that overlap with held-out benchmarks (MMLU, HumanEval, GSM8K). Otherwise, later RL can simply surface memorized answers.
4. **Multi-stage training:** The most valuable domains (code, math) are added later in training so that high-quality patterns are reinforced in the model’s “memory.”

### Quality-Aware Upsampling
Filtering to only the top 5% of texts leaves too little data for trillions of tokens. Instead, **quality-aware upsampling** is used: top 5% of texts are repeated 5–7 times in the final dataset; documents in the top 40% appear once; everything below is dropped.

---

## 4. Stage 2: Mid-Training

Mid-training bridges basic language acquisition (pre-training) and fine-tuning (post-training). Volume is typically 50–500B tokens.

### Context Extension
The context window is extended in stages, e.g. `4K → 32K → 128K → 200K`. The mix includes books, long scientific articles (e.g. via olmOCR-style parsers), and synthetic reasoning trajectories. *Interleaved packing*—shuffling independent texts within one window—helps mitigate the “lost-in-the-middle” effect.

### Synthetic Data and Reasoning
In OLMo 3, mid-training heavily uses **distilled reasoning tokens**.
Hundreds of thousands of synthetic prompts are created:
* **Program-verifiable data:** Solutions to hard algorithmic problems generated by a strong model (e.g. QwQ-32B or GPT-4) and verified by a Python interpreter.
* **Meta-reasoning:** Data that teaches cognitive patterns: *backtracking*, self-checking, and code debugging.

**Takeaway:** Introducing reasoning chains at mid-training sharply boosts the model’s base mathematical and algorithmic ability before full RL.

---

## 5. Stage 3: Post-Training

Here the base “sponge” is turned into an autonomous thinking agent (Thinking model) or a helpful assistant (Instruct model). In GLM-5 and OLMo 3 this pipeline has three steps.

### Supervised Fine-Tuning (SFT) and Chat Formats
The model learns instructions, chat formats (`<|user|>`, `<|assistant|>`), and the basic thought-chain pattern (`<think>...</think>`).
In GLM-5 and top models, advanced modes are used:
* **Interleaved Thinking:** The model must think before every response or tool call.
* **Preserved Thinking:** In complex coding tasks, thought history is kept across all dialogue rounds; the model builds on prior reasoning instead of starting from scratch each turn.

### Preference Optimization (DPO) and Delta Learning
Further SFT on strong-model answers quickly saturates. The model must learn to distinguish *good* from *bad*.
**DPO (Direct Preference Optimization)** is used with **Delta Learning**:
1. Pairs of responses (Chosen and Rejected) are built.
2. **Maximizing delta:** Chosen from a strong model (e.g. 32B), Rejected from a deliberately weak one (e.g. 0.5B).
3. **Length bias control:** DPO tends to reward longer answers. To avoid verbosity, the length difference between Chosen and Rejected is capped (e.g. ≤100 tokens). This yields concise, clear answers, which is critical for Instruct models.

### Reinforcement Learning (RLVR and GRPO)
The stage that pushes the “intelligence ceiling.” There is no human or GPT-4 teacher—the model searches for solutions and gets reward only from objective outcomes (**RL with Verifiable Rewards**, RLVR).
*   **GRPO (Group Relative Policy Optimization):** The standard, replacing PPO. For one prompt, a group of responses (e.g. 16) is sampled; rewards are computed and weights are updated to favor responses above the group mean.
*   **No KL penalty:** GRPO drops the per-token KL term used in PPO, which discouraged long outputs. Without it, the model can “think” for thousands of tokens without being penalized for length.
*   **Verifiers:** For math—numeric match (e.g. SymPy). For code—running unit tests (Fail-to-Pass). For instructions—scripts checking JSON format or word count.

### RL Infrastructure: Inflight Updates and Continuous Batching
RL training is bottlenecked by inference (generation) speed.
*   **Continuous Batching:** Unlike static batching (where everyone waits for the longest response), continuous batching fills freed GPU slots with new prompts immediately, saving up to ~50% compute.
*   **Inflight Updates:** Generation and weight updates are decoupled. The learner updates weights without stopping the inference (actor) servers.
*   **TITO (Token-in-Token-out):** Generators send token IDs directly to the learner, skipping decode→text→retokenize. This avoids critical bugs (e.g. misaligned indentation in generated code).
*   **Off-policy correction (IcePop):** Because weights are updated asynchronously, actors may use slightly stale policies. Double-sided importance sampling (IcePop) masks tokens when the ratio between “old” and “new” policy is too large.

### Cross-Stage Distillation
Long RL on math and code causes **catastrophic forgetting**—the model gets smarter but loses empathy, languages, and general chat (alignment tax). To avoid this, GLM-5 runs **on-policy distillation** as a final stage: checkpoints from earlier stages (e.g. right after SFT) serve as teachers; the current RL model is penalized for deviating from their logit distribution on general dialogue prompts. The model becomes “human” again while keeping its algorithmic gains.

---

## 6. Agentic Engineering and Environment Scaling

Training agents for complex, autonomous tasks requires more than question–answer datasets. Models like GLM-5 and Intellect-3 are trained and evaluated in isolated sandboxes.

### Software Engineering (SWE) and Terminals
*   **SWE Environments:** Tens of thousands of verifiable environments are built from real GitHub bug reports (SWE-bench Verified). The model must find the right file, write a patch, run the build, and pass unit tests. Reward = 1 only for a successful build and passing tests.
*   **Terminal environments:** Systems spawn Docker containers with terminal access. The agent must install dependencies, configure a database, or write bash scripts. LLMs are used to generate Dockerfiles and tests from web tutorials.
*   **UI / Slide generation:** The model learns to produce HTML/CSS. RL reward is based on *runtime rendering*—code is rendered in a headless browser (Playwright), and layout, aspect ratio (e.g. 16:9), and visual quality are scored.

### Context Management for Agents (Keep-recent-k)
During long web search (browsing agents) or log debugging, context can grow to 100K+ tokens and the model “forgets” instructions. GLM-5 uses a **Keep-recent-k** strategy: when tool-call history exceeds $k$ steps, older logs and full HTML pages are discarded or summarized, keeping only the last $k$ steps in full. Above a hard limit (e.g. 32K tokens), the full tool context is reset while preserving the global goal. This lets agents run for hours without drowning in noise.

---

## 7. Training Ops: Engineering Pitfalls

Large-scale training often fails due to infrastructure, not algorithm math:

1. **Throughput vanishing:** Storage clusters struggle with terabytes of small reads. If data is evicted to slow disks (e.g. S3), GPUs idle. Mitigation: local caching and dataloaders with prefetch (e.g. Tokenizedbytes).
2. **Noisy loss and Dataloader:** If the dataloader reads a long code repo sequentially, the batch is all code and gradients spike. Strong shuffling of tokenized sequences is needed (e.g. **RSDB**—Random Sequential Document Buffer).
3. **Tensor Parallelism bugs:** A classic mistake: the same `random seed` on all GPUs in a TP group. The model produces identical features and loss stops improving. Careful seed and initialization control can save months of training.

---
**Summary:** Building frontier models today is a synthesis of high-throughput infrastructure, strict filtering of trillions of tokens, and advanced RL algorithms. Simple text generators have evolved into agentic systems that autonomously write code, render pages, and think before every action.

---

## Additional Details from Technical Reports

Below are concrete numbers, recipes, and practices from OLMo 3 reports, Alex Wa’s frontier methodologies blog, GLM-5, and other sources—to make the article as complete as possible.

### Minimal Playbook and General Principles (Alex Wa)
* **Lock in evals early** for knowledge, math, code, long context, and instruction following; finish eval implementation before the base model is done.
* **Base architecture choice:** dense + GQA + RoPE/RNoPE by default; MoE only if inference efficiency is needed and infrastructure for load balancing exists.
* **Tokenizer** tuned to target language and domain mix; vocab and special tokens are fixed early.
* **Data pipeline:** deduplication, filtering, contamination checks; explicit data-quality metrics.
* **Ablations:** one variable at a time; fast and reliable (with good discriminative power).
* **Multi-stage mix:** best and “reasoning” data shifted toward the end of training.
* **Stability:** logit softcapping (preferred, per Gemma) or z-loss/QK-norm, gradient clipping, precision policy, alerts on loss spikes.
* **Throughput validation** on long runs and dataloader behavior (packing, shuffling, random access).
* **Seeds:** consistent seeds, especially under tensor parallelism.

### Tokenizer and Embeddings
* **Vocab size:** for English, ~50K is often enough; for multilingual, 100K+. Too large a vocab inflates the embedding matrix (up to ~20% of params in small models).
* **Tied embeddings (input/output weight sharing):** in Hugging Face 1.2B experiments, tied gave comparable quality with 18% fewer parameters; untied at the same param budget gave worse loss and evals.
* **BPE** remains standard; “fertility” and “proportion of continued words” help evaluate tokenizer efficiency.

### Pre-training: Numbers and Pipeline (OLMo 3)
* **Scale:** pretrain on **Dolma3 mix** — ~5.93T tokens (pool ~9T); midtrain — **100B** tokens (**Dolmino-2**); long-context extension — **50B** (7B) or **100B** (32B).
* **Pretrain composition (6T mix):** Common Crawl ~76%, olmOCR PDF ~13.6%, Stack-Edu (code) ~6.9%, arXiv ~0.86%, FineMath 3+ ~2.56%, Wikipedia/Wikibooks ~0.04%.
* **Three-stage deduplication:** (1) exact document hash — removes ~67% duplicates; (2) fuzzy MinHash (Jaccard) — ~23% more; (3) substring/suffix-array — repeated substrings (500+ bytes) — ~14% bytes removed. **Duplodocus** (Rust) for exact + MinHash; overall document count drops by ~75%.
* **Topic & quality:** 24-topic split (WebOrganizer) and quality ventiles (5-percentile bins); per-topic **upsampling curves:** discard bottom 40% by quality, upsample top 5% up to 7× (curve integral sets target token volume).
* **Token-constrained mixing and conditional mixing:** swarm search over mixes (many small proxy models with Dirichlet weights), BPB regression on tasks, conditional mix updates when new domains appear without restarting the full swarm.

### Mid-training: Methodology (OLMo 3)
* **Microanneals:** 5B target-dataset tokens + 5B web tokens; comparison with 10B web-only for quick impact on base evals.
* **Integration runs:** full 100B runs on candidate mixes; checkpoints then go through SFT and posttrain evals. Five integration rounds; final round with **decontamination**.
* **Decontamination:** **decon** package — n-gram search from benchmarks in documents; above-threshold match removes document. All benchmark splits considered (including train), as some evals use expanded sets.
* **Meta-reasoning:** seven cognitive categories (self-awareness, evaluation, goal management, hierarchical organization, backward chaining, backtracking, conceptual reasoning); tasks like “answer-to-question” or “code debugging”; trace generation with GPT-4.1 / o4-mini.
* **Program-verifiable data:** tasks with deterministic Python verification; filtered by verifier. ~250M such tokens in a 5B microanneal give +1–2 points on GSM8K and MBPP.

### Base Architecture and Training (OLMo 3)
* **Context:** 8192 tokens in pretrain and midtrain (OLMo 2 used 4096).
* **Sliding Window Attention (SWA):** in three of every four layers — window 4096; last layer always full attention.
* **Learning rate:** cosine schedule; 7B: peak 3e-4, warmup 2000 steps; 32B: peak 6e-4, warmup 2000. Final LR ~10% of peak.
* **Throughput:** 7B — 7700 TPS/GPU, 32B — 1960 TPS/GPU at length 8192, bfloat16; MFU ~43% and ~41%. **OLMo-core** stack, FlashAttention-2, async checkpointing.
* **Tokenizer:** same as OLMo 2 — derived from **cl100k** (OpenAI).

### Post-training Hyperparameters (OLMo 3)
* **SFT:** batch 1M tokens (7B) or 4M (32B), 2 epochs, packing, max length 32K; sometimes **model souping** — linear blend of two checkpoints with different LR.
* **DPO:** length-normalized DPO loss; sweep over learning rate and dataset size (early stopping important — quality drops past optimum).
* **RL (Thinking 7B/32B):** LR 1e-6 / 2e-6, constant schedule; 32B: 750 steps, group size 8, max response 32K, clip 0.2/0.272; actors and learner on separate nodes (32B: 64 GPU learner, 160 GPU actors). **RL-Zero:** 13.3K math prompts after dedup and offline filtering; simple prompt without `<think>`; check for **spurious rewards** (random rewards must not improve evals — else contamination).

### Evals and Scaling
* **Benchmarks from reports:** knowledge — MMLU, GPQA Diamond, SimpleQA; math — MATH, AIME, Minerva; code — HumanEval, MBPP, LiveCodeBench, SWE-bench Verified; long context — RULER, HELMET, MRCR; instruction following — IFEval, IFBench, MultiChallenge; alignment — AlpacaEval, LMArena.
* **Scaling laws:** Chinchilla-style C ≈ 6·N·D; many teams “overtrain” in tokens relative to optimum (e.g. Qwen 3 — 36T). Final choice often driven by inference cost and sparsity.
* **Critical batch size** grows over training; batch size warmup — smaller batch early, larger toward the end.

### Alternatives to Pure RL
* **Online DPO:** online preferences (candidate generation + LLM judge); more stable than RL but depends on label quality and coverage.
* **On-policy distillation:** student samples responses, teacher provides logits; train on KL between student and teacher. Cheaper than GRPO (one sample per prompt), in reports (Qwen3, Thinking Machines) gives strong boost and reduces catastrophic forgetting when fine-tuning.

### Safety and “Usual Suspects”
* **gpt-oss-120b:** pretrain filtering for CBRN (chem/bio/rad/nuclear); Preparedness evals; jailbreak tests (StrongReject), instruction hierarchy (system > developer > user > assistant > tool).
* **Usual suspects when training fails:** too high LR; “bad” data batches; MoE load imbalance; storage/network issues; poor initialization (in OLMo 2, N(0, 0.02) improved stability); fp16 without care. On spikes — skip bad batches or increase gradient clipping.

### GLM-5 Infrastructure (Brief)
* **Multi-Token Prediction (MTP):** separate parameters for three MTP layers during training for inference consistency (2 tokens per step); higher accept length than DeepSeek-V3.2.
* **Slime framework:** custom rollouts (multi-step agent scenarios), PD disaggregation (prefill and decode on different resources), FP8 for rollouts, heartbeat-based fault tolerance.
* **General RL:** three dimensions — foundational correctness, emotional intelligence, task-specific quality; hybrid rewards — rule-based + outcome reward models (ORM) + generative reward models (GRM); human answers as style anchors.
* **Memory and parallelism:** flexible MTP placement, Pipeline ZeRO2 for gradients, zero-redundant communication for Muon, activation offloading, sequence-chunked output projection; INT4 QAT at SFT stage.
* **GLM-5 scale and DSA:** Training scale up to **28.5T** tokens (744B parameters). **DSA (DeepSeek Sparse Attention):** Replaces dense O(L²) attention with dynamic selection of important tokens; continued pre-training from a dense base: 1000-step warmup (indexer only), then 20B tokens of joint adaptation; DSA reduces attention compute by ~1.5–2× for long sequences, enabling 128K context at much lower GPU cost.

### Thinking SFT: Data Sources and Filtering (OLMo 3)
* **Prompts and traces:** Math — OpenThoughts3 (16× repeat, complete solutions) and SYNTHETIC-2 (verified); incomplete traces regenerated via QwQ-32B up to 32K tokens. Code — AceCoder, The Algorithms (Python), Nemotron Post-training, OpenCodeReasoning; up to 16 responses per prompt from QwQ-32B, filtered by synthetic test cases from GPT-4.1. Chat and safety — WildChat (Tulu 3 and beyond), OpenAssistant; traces from DeepSeek R1. Instruction following — Tulu 3 + verifiable constraints (Pyatkin), Persona IF with Nemotron-Personas; response verification. Science and other — OpenThoughts3 science, TableGPT, Aya; regeneration of incomplete and generation with DeepSeek R1.
* **SFT filtering:** (1) Licenses (non-commercial/unclear dropped); (2) Incomplete reasoning chains; (3) Domain accuracy (constraint checks for IF, test execution for code); (4) Mentions of other model developers and dates; (5) Excessive repetition; (6) Excessive Chinese characters or politically charged phrasing in traces. **Topic filtering:** Classification by OpenAI query taxonomy; dropping/downsampling irrelevant topics (image generation, greetings) improves behavior. Details and script links — in open-instruct and report appendix.
* **Mixing and decontamination:** Methodology as in midtraining — parallel collection, shared mix standards, multiple integration rounds; “base” mix 100K examples from OpenThoughts3, then add up to 100K per category for ablations. Posttrain decontamination: Tulu 3 procedure, 8-grams, overlap threshold 0.5; heuristics against false positives (ignore common phrases, in math ignore n-grams of short tokens). **Model souping:** Final Thinking SFT checkpoint — linear weighted merge of two checkpoints with different LR via mergekit.

### Delta Learning and DPO for Reasoning (OLMo 3)
* **Idea:** Preference quality is driven primarily by the *delta* between chosen and rejected response, not absolute quality of each. Pairs (x, y_c, y_r) with clear capability contrast help even when SFT on y_c no longer helps or hurts.
* **Dolci-Think-DPO:** Chosen — Qwen 3 32B (thinking), rejected — Qwen 3 0.6B (thinking); one strong and one weak model give stable contrast. Prompt pool — from Dolci-Instruct SFT plus DaringAnteater and UltraFeedback from OLMo 2 7B preference set. For Instruct-DPO they add delta-aware GPT-judge pairs and multi-turn preferences (self-talk and synthetic-context), while controlling length bias (cap difference at 100 tokens for chat and multi-turn).

### Instruct: Function-Calling and Preferences (OLMo 3)
* **Real tool trajectories:** Science QA (ASTA/ASC MCP, Semantic Scholar); Web search QA (DR Tulu, Serper API, HotpotQA, TaskCraft, WebWalkerQA, SearchArena, OpenScholar); query filtering via GPT-5 — keep only those that require search and allow verifiable long-form answers. **SimFC:** Synthetic trajectories with LLM-simulated environment over API pool (xLAM, ToolACE, public MCPs); diversity — multi-turn, multi-step, refusals due to insufficient information.
* **Format:** OpenAPI for tool definitions, function calls as pythonic code blocks; specs in system prompt, environment outputs in a dedicated role; vocabulary extended with special tokens for tags. Evaluation: BFCLv3 (intrinsic function calling), LitQA2 (ASC), SimpleQA (search/browsing); No-Tools setting for comparison.
* **Starting from Thinking SFT:** Training Instruct SFT from a “warm start” on Thinking SFT significantly improves metrics while keeping short answers without thinking traces. **Preferences:** Combining delta-learning heuristic and delta-aware GPT-judge pairs beats either alone; DPO vs data size gives a U-shaped curve — quality drops past the optimum (early stopping and sweep over LR and dataset size matter). RL for Instruct: less challenging math/code datasets, no offline difficulty filtering; max response 8K (7B) or 16K (32B); two DPO candidates (best average and best “vibe test”), final by average, length, and vibe test.

### Long-Context Extension (OLMo 3)
* **Scale and mix:** **longmino** pool 600B+ tokens; training uses 34% long-context and 66% short from Dolmino-2. For 7B — 50B extension tokens, for 32B — 100B. Comparison: SmolLM3/GLM 4.5/DeepSeek V3 ~100–123B, Apertus 225B, Kimi K2 400B, Llama 3.1 800B; AFM and Nemotron Nano 2 — under 20B tokens to 64K/128K.
* **Data:** Base — olmOCR PDF; filtering by gzip compressibility (drop top and bottom 20%). Synthetic augmentation à la CLIPPER: split 32K–65K docs into 8K–32K sections, tf-idf over noun phrases, 8 snippets per phrase, generate aggregation tasks (CWE — count occurrences, REX — rewrite in vignettes) via OLMo 2 Instruct 32B.
* **Recipe:** YaRN applied only to full-attention layers (not sliding window). **Document packing:** Best-fit document packing. **Intra-document masking:** Tokens from one document not mixed with others in the same sequence. **Infrastructure:** 8-way context parallelism (8K per device), all-gather CP attention. **Model souping (32B):** Merge of last three extension checkpoints (steps 10K, 11K, 11,921). Evals: RULER — main dev metric; HELMET — held-out.

### RL-Zero: Data, Prompts, Active Sampling (OLMo 3)
* **Data:** Math — aggressive filtering of DAPO Math, Klear-Reasoner Math, Open-Reasoner-Zero, Omega; DAPO dedup, English only; semantic clustering over Klear/Orz/Omega, one representative per cluster + DAPO. Decontamination from pretrain and evals; offline filtering — prompts fully solved in 8/8 samples by final base are removed. **13.3K** math prompts total. Code, IF, and general chat — subsampled from Dolci-Think-RL.
* **Prompt and evals:** Simple template without `<think>` strongly outperforms standard posttrain templates when training from a pure base (Dolmino had no special markup). Evals cleaned of special formatting (e.g. \boxed{}) to match training prompts. Response length 16K at train, 32K at eval with temperature 1.0 for pass@k.
* **Active sampling:** Continuously pulling prompt–completion pairs from the queue after filtering for non-zero advantage keeps the batch full and stabilizes training (lower loss variance). **Spurious rewards:** Training on random binary rewards must not improve evals; if it does, that indicates contamination. In OLMo 3 RL-Zero, random rewards do not improve — decontamination confirmed.

### Quality-Aware Upsampling Formula (OLMo 3)
* Family of **truncated power-exponential curves** f(x): zero below threshold a, above — C(x−a)^p·e^{λ(x−a)}. Constraints: integral over [0,1] equals target ratio Z/X; maximum average upsampling per bucket ≤ M; monotonicity. In practice: M=7, discard bottom 40% by quality (a=0.4); for each WebOrganizer topic solve numerically for p, λ, C. Combined with token-constrained mixing: truncated power-exponential family performed best vs mixing-only, upsampling-only, or arithmetic/geometric mean of targets.

### Decontamination (decon): Implementation
* N-gram search from evals into documents; at each traversal step get step set of matching documents, intersect with **active set**. A document leaves the active set after **11 misses** in a row. Result — map of document id to set of unique matched n-grams. Details — [allenai/decon](https://github.com/allenai/decon).

### Posttrain Evals: Variance (OLMo 3)
* **High variance:** GPQA, AlpacaEval 3, IFEval. **Stable:** ZebraLogic, Omega, AIME 24 (Avg@32), HumanEvalPlus, AgiEval, BigBenchHard. **Very stable:** LiveCodeBench (Avg@10), MBPPPlus, MATH, MMLU, PopQA. Eval cost during 7B recipe development — about 10–20% of compute budget.

---

## References and Technical Reports

This article is based on the following technical reports and publications:

1. **GLM-5: from Vibe Coding to Agentic Engineering** (Zhipu AI & Tsinghua University, 2026) — [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
2. **Kimi K2.5: visual agentic intelligence** (Moonshot AI, 2026) — [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)
3. **OLMo 3 Technical Report** (Allen Institute for AI, 2026) — Training recipes for OLMo 3 Base, Instruct, Thinking, and RLZero.
4. **Frontier model training methodologies** (Alex Wa’s Blog, Jan 2026) — A detailed survey of training open-weight frontier models, including Hugging Face **SmolLM3**, Prime Intellect **Intellect-3**, Nous Research **Hermes 4**, OpenAI **gpt-oss-120b**, and Arcee **Trinity series**.
5. **DeepSeek-V3.2: pushing the frontier of open large language models** (DeepSeek-AI, 2025) — [arXiv:2512.02556](https://arxiv.org/abs/2512.02556)
6. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (DeepSeek-AI, 2025) — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
