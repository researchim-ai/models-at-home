"""Curated primary references. A composition or primitive is not assigned a fictitious inventor."""
from copy import deepcopy


def paper(title, authors, year, url, relationship='research'):
    return dict(title=title, authors=authors, year=year, url=url, relationship=relationship)


TRANSFORMER = paper('Attention Is All You Need', 'Vaswani et al.', 2017, 'https://arxiv.org/abs/1706.03762')
ROPE = paper('RoFormer: Enhanced Transformer with Rotary Position Embedding', 'Su et al.', 2021, 'https://arxiv.org/abs/2104.09864', 'component')
GQA = paper('GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints', 'Ainslie et al.', 2023, 'https://arxiv.org/abs/2305.13245', 'component')
SWIGLU = paper('GLU Variants Improve Transformer', 'Noam Shazeer', 2020, 'https://arxiv.org/abs/2002.05202')
RMS = paper('Root Mean Square Layer Normalization', 'Biao Zhang; Rico Sennrich', 2019, 'https://arxiv.org/abs/1910.07467')
RESIDUAL = paper('Deep Residual Learning for Image Recognition', 'He et al.', 2015, 'https://arxiv.org/abs/1512.03385', 'background')
REZERO = paper('ReZero is All You Need: Fast Convergence at Large Depth', 'Bachlechner et al.', 2020, 'https://arxiv.org/abs/2003.04887', 'variant')
UT = paper('Universal Transformers', 'Dehghani et al.', 2018, 'https://arxiv.org/abs/1807.03819', 'background')
HUGINN = paper('Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach', 'Geiping et al.', 2025, 'https://arxiv.org/abs/2502.05171', 'inspiration')
VISION = paper('An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', 'Dosovitskiy et al.', 2020, 'https://arxiv.org/abs/2010.11929', 'background')


def entry(ru, en, formula, origin_ru, origin_en, references, tradeoffs, experiments, contract='[B,T,H] → [B,T,H]'):
    return dict(summary={'ru':ru, 'en':en}, formula=formula, shape_contract=contract,
        origin={'ru':origin_ru, 'en':origin_en}, references=references, tradeoffs=tradeoffs, experiments=experiments)


KNOWLEDGE = {
    'token_embedding': entry('Преобразует ID токена в обучаемый вектор. Отдельные строки таблицы получают градиент при использовании соответствующих токенов.',
        'Looks up a trainable vector for each token ID. Rows receive gradients when their tokens are used.', 'x = E[token_id]',
        'Обучаемые представления появились раньше современных LLM. Здесь — ранний пример их применения в нейронной языковой модели, а не утверждение об изобретении embeddings в 2003 году.',
        'Learned representations predate modern LLMs. This is an early neural-language-model reference, not an invention claim.',
        [paper('A Neural Probabilistic Language Model', 'Bengio, Ducharme, Vincent, Jauvin', 2003, 'https://jmlr.org/papers/v3/bengio03a.html', 'background')],
        ['Vocabulary × hidden width can dominate parameter memory.', 'Token IDs must fit the declared vocabulary.'], ['Compare vocabulary sizes while keeping model width fixed.'], '[B,T] IDs → [B,T,H]'),
    'positional_embedding': entry('Добавляет обучаемый вектор позиции к представлению токена; в отдельной ветке выдаёт только позиции.',
        'Adds learned position vectors to token features, or emits a position-only branch for an explicit join.', 'h_t = x_t + P_t',
        'Статья Transformer рассматривает позиционное кодирование и обучаемые варианты. Наш блок использует таблицу, не синусоиды и не RoPE.',
        'The Transformer paper discusses positional representations, including learned alternatives. This block is a learned table, not RoPE.',
        [TRANSFORMER], ['Position table limits context length.', 'Combining learned positions and RoPE changes the experiment.'], ['Ablate learned positions versus RoPE.']),
    'attention': entry('Causal self-attention смешивает информацию только из текущей и предыдущих позиций, с учётом padding.',
        'Causal self-attention mixes only current and preceding positions and respects padding.', 'Attention(Q,K,V) = softmax(QKᵀ / √d + mask)V',
        'Ссылка на multi-head attention в Transformer; это не утверждение, что внимание впервые появилось в 2017 году.',
        'Reference for Transformer multi-head attention; attention itself predates 2017.', [TRANSFORMER],
        ['Dense attention compute grows quadratically with context.', 'The mask must protect both future and padding tokens.'], ['Compare head count at fixed hidden width.']),
    'causal_self_attention': entry('Проекции Q/K/V, causal SDPA, опциональные RoPE и GQA. Меньше KV-heads уменьшает параметры K/V.',
        'Q/K/V projections and causal SDPA with optional RoPE and grouped-query attention.', 'softmax(RoPE(Q)RoPE(K)ᵀ / √d + mask)V',
        'Композиция нескольких опубликованных методов; наши tensor shapes и fallback kernels не означают точную копию чужой модели.',
        'A composition of published methods, not an exact reproduction of a particular model.', [TRANSFORMER, ROPE, GQA],
        ['num_kv_heads must divide num_heads.', 'This implementation expands KV groups for SDPA; parameter savings do not guarantee a faster kernel.'], ['Compare MHA, GQA and MQA (one KV head).']),
    'llama_block': entry('Полный causal-блок: RMSNorm, RoPE attention, residual-связи и SwiGLU. Можно включить GQA и sandwich-нормализации.',
        'A complete causal block with RMSNorm, RoPE attention, residual paths and SwiGLU; optional GQA and sandwich norms.',
        'x ← x + Attn(Norm(x)); x ← x + SwiGLU(Norm(x))',
        'Llama-style реализация студии. Ссылка на семейство LLaMA и первоисточники компонентов, не на тождество его checkpoint.',
        'A Studio Llama-style implementation; family and component references do not imply checkpoint compatibility.',
        [paper('LLaMA: Open and Efficient Foundation Language Models', 'Touvron et al.', 2023, 'https://arxiv.org/abs/2302.13971', 'architecture'), RMS, SWIGLU, ROPE, GQA],
        ['Intermediate size often dominates block parameters.', 'Sandwich norms alter residual dynamics and require measurement.'], ['Compare pre-norm and sandwich-norm in a shared core.']),
    'loop': entry('Одно тело с общими весами многократно уточняет скрытое состояние. Число вычислений растёт без копирования параметров.',
        'One shared body repeatedly refines hidden states. Compute increases without copying its parameters.', 's_(r+1) = Core(s_r, input; θ_shared)',
        'Инженерный контейнер студии, опирающийся на recurrent-depth исследования. Это не полная реализация Universal Transformer или Huginn.',
        'A Studio container inspired by recurrent-depth research, not a complete Universal Transformer or Huginn reproduction.',
        [UT, HUGINN], ['More loops increase latency and activation storage.', 'TBPTT without input injection may detach the prelude.', 'Extra inference depth does not guarantee better quality.'],
        ['Hold parameters fixed and compare 1/2/4/8 iterations.', 'Compare full BPTT and a short gradient tail.']),
    'repeater': entry('Создаёт независимые копии блока. Копии имеют разные параметры; это глубина стека, не shared-weight loop.',
        'Constructs independent block copies with separate weights. This is a deep stack, not a shared-weight loop.', 'h = Block_N(...Block_2(Block_1(x)))',
        'Инженерный контейнер; у класса repeater нет отдельной научной статьи или уникального изобретателя.',
        'An engineering container; the class itself has no separate research invention attribution.', [TRANSFORMER],
        ['Parameters and optimizer state grow with repeat count.'], ['Compare an independent stack against a loop at matched executed depth.']),
    'residual': entry('Прибавляет выход редактируемой ветки к входу. Обучаемый scale с нулевым стартом задаёт ReZero-вариант.',
        'Adds an editable branch to its input. A learnable zero-initialized scale provides a ReZero-style gate.', 'y = x + α · F(x)',
        'Ссылки разделяют residual-обучение и вариант ReZero. Сам контейнер — реализация студии.',
        'References distinguish residual learning from the ReZero variant; the container is implemented by Studio.', [RESIDUAL, REZERO],
        ['Branch output must match input width.', 'At scale=0 the branch initially receives zero gradient while the scale can learn.'], ['Compare fixed α=1 with learnable α initialized to zero.']),
    'rmsnorm': entry('Нормализует RMS признаков без вычитания среднего; содержит обучаемый масштаб.',
        'Normalizes feature RMS without centering and includes a learned scale.', 'y = γ · x / √(mean(x²) + ε)',
        'RMSNorm предложен Biao Zhang и Rico Sennrich; реализация следует этому типу нормализации.',
        'RMSNorm was proposed by Biao Zhang and Rico Sennrich.', [RMS], ['Epsilon affects small-magnitude activations.'], ['Compare activation RMS and gradient norms against LayerNorm.']),
    'layernorm': entry('Нормализует признаки каждой позиции с вычитанием среднего и делением на стандартное отклонение.',
        'Centers and scales the features at each position.', 'y = γ · (x − mean(x)) / √(var(x)+ε) + β',
        'Layer Normalization: Ba, Kiros и Hinton, 2016.', 'Layer Normalization: Ba, Kiros and Hinton, 2016.',
        [paper('Layer Normalization', 'Ba, Kiros, Hinton', 2016, 'https://arxiv.org/abs/1607.06450')], ['Centering changes the feature geometry relative to RMSNorm.'], ['Compare norms at matched parameter budget.']),
    'groupnorm': entry('Нормализует группы каналов. В этой реализации статистики затрагивают временную ось: блок не является безопасным causal-normalization для авторегрессии.',
        'Normalizes channel groups. This implementation includes the sequence axis in statistics and is not a safe causal normalization for autoregression.',
        'Normalize channel groups over their spatial/sequence extent',
        'Group Normalization: Yuxin Wu и Kaiming He, 2018. Ограничение причинности относится к нашему sequence wrapper.',
        'Group Normalization: Yuxin Wu and Kaiming He, 2018. The causality caveat concerns this sequence wrapper.',
        [paper('Group Normalization', 'Yuxin Wu; Kaiming He', 2018, 'https://arxiv.org/abs/1803.08494')],
        ['Can leak future-token information in this sequence wrapper.', 'Research multimodal adapters reject this block.'], ['Use only in a context with an explicitly non-causal contract.']),
    'swiglu': entry('Две ветки проекции перемножаются после SiLU-gating; затем результат проецируется обратно.',
        'Projects into two branches, applies SiLU gating and projects the product back.', 'SwiGLU(x) = W_o(SiLU(W_g x) ⊙ W_u x)',
        'SwiGLU исследован в GLU Variants Improve Transformer, Noam Shazeer, 2020.',
        'SwiGLU is studied in GLU Variants Improve Transformer by Noam Shazeer, 2020.', [SWIGLU],
        ['Three dense matrices; compare intermediate widths fairly.'], ['Compare SwiGLU and GELU MLP at matched parameter count.']),
    'mlp': entry('Позиционный feed-forward: расширение признаков, нелинейность, обратная проекция.',
        'Position-wise feed-forward: expansion, nonlinearity and output projection.', 'FFN(x) = W₂ φ(W₁x+b₁)+b₂',
        'MLP не приписывается Transformer. Статья здесь — источник его position-wise FFN применения.',
        'The MLP is not attributed to the Transformer; this reference concerns its position-wise FFN use.', [TRANSFORMER],
        ['Intermediate width controls capacity and compute.'], ['Change activation while keeping widths fixed.']),
    'moe': entry('Router выбирает несколько экспертов для каждого токена; auxiliary loss помогает распределять нагрузку.',
        'A router selects a few experts per token; an auxiliary objective encourages balanced routing.', 'y = Σ_(e∈top-k) gate_e(x) · Expert_e(x)',
        'Ссылка на sparsely-gated MoE 2017 года. Концепция mixture-of-experts существовала раньше.',
        'Reference for sparsely-gated MoE (2017); mixture-of-experts predates this paper.',
        [paper('Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer', 'Shazeer et al.', 2017, 'https://arxiv.org/abs/1701.06538', 'variant')],
        ['All experts occupy parameter memory even when only top-k execute.', 'This local dispatcher is a simple implementation, not distributed expert parallelism.'], ['Compare expert count, top-k and observed router usage.']),
    'causal_conv1d': entry('Свёртка по последовательности использует только доступное прошлое; dilation расширяет receptive field.',
        'A sequence convolution uses only the available past; dilation expands its receptive field.', 'y_t = Σ_i W_i x_(t−i·dilation)',
        'WaveNet — известный источник causal/dilated convolution, а не утверждение об изобретении самой свёртки.',
        'WaveNet is a causal/dilated convolution reference, not an invention claim for convolution.',
        [paper('WaveNet: A Generative Model for Raw Audio', 'van den Oord et al.', 2016, 'https://arxiv.org/abs/1609.03499', 'background')], ['Local receptive field differs from attention.', 'Grouped convolution must respect channel divisibility.'], ['Compare a convolution/attention hybrid at a fixed compute budget.']),
    'dropout': entry('Случайно отключает элементы во время обучения, с компенсацией масштаба. При eval отключён.',
        'Randomly removes activations during training and rescales survivors; disabled at evaluation.', 'y = x · Bernoulli(1−p)/(1−p)',
        'Ссылка на публикацию Dropout в JMLR (2014); идеи и предварительные работы появились раньше.',
        'Reference to the JMLR Dropout publication (2014); ideas and preliminary work appeared earlier.',
        [paper('Dropout: A Simple Way to Prevent Neural Networks from Overfitting', 'Srivastava et al.', 2014, 'https://jmlr.org/papers/v15/srivastava14a.html')], ['Stochastic probes require matched seeds.', 'p=1 removes all signal through the branch.'], ['Compare train/eval activation statistics.']),
}

PRIMITIVES = {
    'linear': ('Обучаемая аффинная проекция признаков.', 'A trainable affine feature projection.', 'y = xWᵀ + b'),
    'add': ('Складывает ветки одинаковой формы.', 'Adds equal-shaped branches.', 'y = x₁ + x₂ + …'),
    'multiply': ('Перемножает две ветки одинаковой формы; можно использовать для gating.', 'Multiplies two equal-shaped branches; useful for gating.', 'y = x₁ ⊙ x₂'),
    'concat': ('Объединяет ветки по скрытой размерности.', 'Concatenates branches along hidden width.', 'H_out = Σ H_in'),
    'activation': ('Поэлементная нелинейность. Название и поведение определяет параметр activation.', 'An element-wise nonlinearity selected by activation.', 'y = φ(x)'),
    'custom_op': ('Локальный Python-модуль с явно объявленной выходной шириной.', 'A local Python module with declared output width.', 'y = CustomModule(x)'),
    'inline_code': ('Пользовательский Python forward; форму, причинность и градиенты проверяет автор.', 'User-defined Python forward; its author must verify shapes, causality and gradients.', 'x ← user code(x)'),
    'permute': ('Меняет порядок осей тензора; требует ручной проверки sequence-контракта.', 'Reorders tensor axes; requires manual sequence-contract validation.', 'y = permute(x, dims)'),
    'adaptive_avg_pool': ('Меняет длину последовательности усреднением; это влияет на согласование labels.', 'Changes sequence length by averaging; label alignment must be handled explicitly.', 'T_out = output_size'),
}


def knowledge(kind):
    base = kind.removeprefix('liger_')
    if base == 'residual_mlp':
        result = deepcopy(KNOWLEDGE['residual'])
        result['summary'] = {'ru':'Residual-ветка с MLP: y = x + MLP(x).', 'en':'An MLP residual branch: y = x + MLP(x).'}
        result['references'].append(TRANSFORMER)
    elif base in KNOWLEDGE:
        result = deepcopy(KNOWLEDGE[base])
    else:
        ru, en, formula = PRIMITIVES[base]
        refs = [paper('PyTorch reference implementation', 'PyTorch contributors', None, 'https://docs.pytorch.org/docs/stable/nn.html', 'implementation')]
        result = entry(ru, en, formula, 'Базовая операция или инженерное расширение. Единственный изобретатель и статья не назначаются.',
            'A primitive or engineering extension; no unique inventor or invention paper is assigned.', refs,
            ['Validate tensor shape, sequence alignment and gradients for the selected use.'], ['Change one factor at a time and record a baseline.'])
    if kind.startswith('liger_'):
        result['references'].append(paper('Liger Kernel', 'LinkedIn / Liger contributors', None, 'https://github.com/linkedin/Liger-Kernel', 'implementation'))
        result['tradeoffs'].append('Optional fused implementation; hardware support and numerical behavior need validation.')
    return result
