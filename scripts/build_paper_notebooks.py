"""Original educational PyTorch implementations of mechanisms from cited papers.

No upstream repository code or paper text is copied. These CPU notebooks reproduce
specific mechanisms, not benchmark results or complete production architectures.
"""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1] / 'models-at-home' / 'notebooks' / 'papers'
PAPERS = []


def paper(slug, title, year, authors, url, topic, section, name, goal, context, math, steps, interpretation, exercises, answers, scope, prerequisite):
    PAPERS.append(dict(slug=slug, title=title, year=year, authors=authors, url=url, topic=topic, section=section,
                       name=name, goal=goal, context=context, math=math, steps=steps, interpretation=interpretation,
                       exercises=exercises, answers=answers, scope=scope, prerequisite=prerequisite))


def step(ru, en, code):
    return ([ru, en], dedent(code).strip())


paper('01-attention', 'Attention Is All You Need', 2017, 'Vaswani et al.', 'https://arxiv.org/abs/1706.03762', 'llm', '/builder',
 ['Attention: от Q/K/V до causal mask', 'Attention: from Q/K/V to a causal mask'],
 ['Реализовать scaled dot-product attention без готового attention-модуля и проверить, что будущее не меняет прошлое.',
  'Implement scaled dot-product attention without a ready-made attention module and verify that future tokens cannot alter the past.'],
 ['До Transformer обмен информацией в последовательности часто зависел от рекуррентного прохода. Attention позволяет каждой позиции собрать взвешенную смесь признаков других позиций. Q описывает запрос позиции, K — признаки для сопоставления, V — передаваемую информацию. Это обучаемые проекции, а не заранее заданные роли слов.\n\nВектор результата — смесь V, но веса смеси зависят от Q и K. Несколько голов работают в разных подпространствах. Сам по себе attention не знает порядок: нужны позиционные признаки. В causal LM отдельно запрещают доступ к будущим позициям.',
  'Before Transformers, sequence communication often depended on recurrence. Attention lets each position collect a weighted mixture of other positions. Q represents a query, K matching features, and V the information to transfer; these are learned projections, not fixed linguistic roles.\n\nThe output mixes V, with mixing weights determined by Q and K. Multiple heads operate in different subspaces. Attention alone does not encode order: position information is needed. A causal LM must separately prohibit future access.'],
 ['`S = Q @ K.transpose(-2,-1) / sqrt(d_head)`; `P = softmax(S + mask)`; `O = P @ V`.\n\nПри форме Q/K/V `[B,H,T,D]` scores имеют форму `[B,H,T,T]`. Softmax идёт по ключам (последняя ось); сумма каждой строки равна 1. Масштаб уменьшает рост дисперсии dot product. В mask разрешённые элементы равны 0, запрещённые — `-inf`. Полностью запрещённая строка даст NaN, поэтому диагональ оставляем доступной.',
  '`S = Q @ K.transpose(-2,-1) / sqrt(d_head)`; `P = softmax(S + mask)`; `O = P @ V`.\n\nFor Q/K/V `[B,H,T,D]`, scores are `[B,H,T,T]`. Softmax runs over keys, the final axis, so each row sums to 1. Scaling controls dot-product variance. An additive mask uses 0 for allowed positions and `-inf` for forbidden positions. Fully masked rows produce NaNs; keep the diagonal available.'],
 [step('Шаг 1. Явно считаем scores и веса. Посмотрите на causal треугольник: верхняя часть должна быть нулевой после softmax.',
       'Step 1. Compute scores and weights explicitly. Above the causal diagonal, probabilities must be zero.', '''
 def attention(q, k, v, causal=True):
     scores = q @ k.transpose(-2, -1) / q.shape[-1]**0.5
     if causal:
         future = torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool), diagonal=1)
         scores = scores.masked_fill(future, float('-inf'))
     probabilities = scores.softmax(dim=-1)
     return probabilities @ v, probabilities
 q, k, v = [torch.randn(2, 2, 6, 4) for _ in range(3)]
 out, probabilities = attention(q, k, v)
 assert out.shape == q.shape
 assert torch.allclose(probabilities.sum(-1), torch.ones(2, 2, 6))
 assert torch.equal(probabilities.triu(1), torch.zeros_like(probabilities))
 plt.imshow(probabilities[0, 0].detach(), cmap='Reds', vmin=0, vmax=1)
 plt.xlabel('key position'); plt.ylabel('query position'); plt.colorbar(); plt.show()
 '''),
  step('Шаг 2. Одна голова становится модулем с обучаемыми Q/K/V. Градиент должен доходить до всех трёх проекций.',
       'Step 2. Add learned Q/K/V projections. Gradients should reach all three projections.', '''
 class AttentionBlock(nn.Module):
     def __init__(self, width=8):
         super().__init__()
         self.q = nn.Linear(width, width, bias=False)
         self.k = nn.Linear(width, width, bias=False)
         self.v = nn.Linear(width, width, bias=False)
     def forward(self, x):
         return attention(self.q(x), self.k(x), self.v(x))[0]
 block = AttentionBlock(); x = torch.randn(2, 6, 8)
 target = torch.randn_like(x)
 F.mse_loss(block(x), target).backward()
 for name, parameter in block.named_parameters():
     print(name, 'gradient norm:', parameter.grad.norm().item())
     assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
 '''),
  step('Шаг 3. Меняем только будущую часть входа. Причинность проверяется выходами, а не названием класса.',
       'Step 3. Change only future inputs. Verify causality through outputs, not class names.', '''
 changed = x.clone(); changed[:, 3:] += 10 * torch.randn_like(changed[:, 3:])
 with torch.no_grad():
     delta = (block(x)[:, :3] - block(changed)[:, :3]).abs().max()
     unmasked_a = attention(block.q(x), block.k(x), block.v(x), False)[0]
     unmasked_b = attention(block.q(changed), block.k(changed), block.v(changed), False)[0]
 print('causal delta:', delta.item(), 'unmasked delta:', (unmasked_a[:, :3] - unmasked_b[:, :3]).abs().max().item())
 assert delta < 1e-6
 ''')],
 ['Нулевая верхняя часть attention matrix и неизменность раннего выхода — две независимые проверки. Ненулевой градиент показывает наличие пути обучения, но не качество language modeling. Heatmap относится к случайным входам: нельзя приписывать ей языковой смысл.',
  'Zero future probabilities and unchanged early outputs are separate checks. Nonzero gradients establish a learning path, not language-model quality. This heatmap uses random inputs and has no linguistic interpretation.'],
 ['1. Уберите sqrt(D) и сравните entropy весов при D=4 и D=256.\n2. Уберите mask: какой assert должен упасть?\n3. Добавьте две головы и выходную Linear, сохранив ширину 8.',
  '1. Remove sqrt(D) and compare weight entropy at D=4 and D=256.\n2. Remove the mask: which assertion should fail?\n3. Add two heads and an output Linear while preserving width 8.'],
 ['Для нормальных независимых Q/K variance dot product растёт с D, поэтому без масштаба softmax часто становится резче. Без mask ранний output зависит от будущего. Две головы: `[B,T,8] → [B,2,T,4]`, затем concatenate и проекция 8→8.',
  'For independent normal Q/K, dot-product variance grows with D; unscaled softmax often sharpens. Without masking, early outputs depend on future inputs. Two heads reshape `[B,T,8] → [B,2,T,4]`, followed by concatenation and an 8→8 projection.'],
 ['Только attention-механизм. Нет полного encoder–decoder, positional encoding, translation corpus и воспроизведения BLEU из статьи.',
  'Attention mechanism only: no complete encoder–decoder, positional encoding, translation corpus, or paper BLEU reproduction.'],
 ['Базовые тензоры PyTorch, матричное умножение, softmax и backward. Сначала пройдите «Как модель учится».',
  'PyTorch tensors, matrix multiplication, softmax, and backward. Start with “How a model learns”.'])

paper('02-rope', 'RoFormer: Enhanced Transformer with Rotary Position Embedding', 2021, 'Su et al.', 'https://arxiv.org/abs/2104.09864', 'builder', '/builder',
 ['RoPE: позиция как поворот', 'RoPE: position as a rotation'],
 ['Написать rotary embedding и проверить сохранение нормы и зависимость dot product от относительного смещения.',
  'Implement rotary embeddings and test norm preservation and relative-position dot products.'],
 ['RoPE кодирует позицию, поворачивая пары координат Q и K. Это отличается от прибавления position embedding к токену. Разные пары используют разные частоты, поэтому близкие и далёкие сдвиги выражаются через несколько масштабов. V в обычном применении RoPE не вращается.\n\nВажно различать математическое свойство относительной позиции и способность обученной модели работать на большем контексте. Первое проверяется точно, второе требует эксперимента; смена частот и context scaling не гарантируют качество.',
  'RoPE encodes position by rotating coordinate pairs of Q and K rather than adding a position vector to a token. Different pairs use different frequencies, representing offsets at multiple scales. In the usual application, V is not rotated.\n\nDistinguish the mathematical relative-position property from a trained model’s ability to handle longer contexts. The former can be tested directly; the latter needs evaluation. Frequency changes and context scaling do not guarantee quality.'],
 ['Для пары `(a,b)` угол `phi = position * theta_i`: `(a*cos(phi)-b*sin(phi), a*sin(phi)+b*cos(phi))`. Частота `theta_i = base^(-2i/D)`.\n\n`R(m)q · R(n)k = q · R(n-m)k`. Поэтому общий сдвиг m и n не меняет dot product. Ширина D должна быть чётной. Здесь используется соседнее разбиение на пары; другие layouts допустимы только при согласованной перестановке весов и координат.',
  'For pair `(a,b)` and angle `phi = position * theta_i`, rotate to `(a*cos(phi)-b*sin(phi), a*sin(phi)+b*cos(phi))`, with `theta_i = base^(-2i/D)`.\n\n`R(m)q · R(n)k = q · R(n-m)k`, so a common shift preserves the dot product. D must be even. We use adjacent coordinate pairs; other layouts need consistent coordinate and weight permutations.'],
 [step('Шаг 1. Считаем вращение явно в float64, чтобы численная проверка была точной.', 'Step 1. Implement rotations in float64 for a precise numerical check.', '''
 def rope(x, positions, base=10000.):
     width = x.shape[-1]
     assert width % 2 == 0
     frequency = base ** (-torch.arange(0, width, 2, dtype=x.dtype) / width)
     angle = positions.to(x.dtype)[..., None] * frequency
     even, odd = x[..., 0::2], x[..., 1::2]
     result = torch.stack([even * angle.cos() - odd * angle.sin(),
                           even * angle.sin() + odd * angle.cos()], dim=-1)
     return result.flatten(-2)
 x = torch.randn(8, dtype=torch.float64)
 rotated = rope(x, torch.tensor(7))
 print('norm error:', (x.norm() - rotated.norm()).abs().item())
 assert torch.allclose(x.norm(), rotated.norm(), atol=1e-12)
 '''),
  step('Шаг 2. Проверяем относительный сдвиг и одновременно сдвигаем обе позиции.', 'Step 2. Test the relative-position identity and a common position shift.', '''
 q = torch.randn(8, dtype=torch.float64); k = torch.randn_like(q)
 m, n, offset = 5, 11, 20
 left = rope(q, torch.tensor(m)) @ rope(k, torch.tensor(n))
 right = q @ rope(k, torch.tensor(n - m))
 shifted = rope(q, torch.tensor(m + offset)) @ rope(k, torch.tensor(n + offset))
 print('relative error:', abs(left - right).item(), 'common-shift error:', abs(left - shifted).item())
 assert torch.allclose(left, right, atol=1e-12)
 assert torch.allclose(left, shifted, atol=1e-12)
 '''),
  step('Шаг 3. Смотрим, как dot product меняется с расстоянием; проверяем differentiability.', 'Step 3. Plot the offset dependence and check differentiability.', '''
 positions = torch.arange(128)
 scores = torch.stack([rope(q, torch.tensor(0)) @ rope(k, p) for p in positions])
 plt.plot(positions, scores, color='#d73535'); plt.xlabel('relative offset'); plt.ylabel('Q dot K'); plt.show()
 differentiable = torch.randn(2, 6, 8, requires_grad=True)
 result = rope(differentiable, torch.arange(6))
 result.mul(torch.randn_like(result)).sum().backward()
 assert torch.isfinite(differentiable.grad).all()
 print('gradient norm:', differentiable.grad.norm().item())
 ''')],
 ['Одна пара Q/K может давать осциллирующую кривую, а не монотонное затухание. Тест нормы проверяет вращение; тест общего сдвига проверяет позиционную структуру. Оба нужны: сохранение нормы само по себе ещё не подтверждает корректность RoPE.',
  'A single Q/K pair can produce oscillations, not monotonic decay. Norm preservation checks the rotation; common-shift invariance checks positional structure. Both matter: preserving norms alone does not establish correct RoPE.'],
 ['1. Поверните только Q: сохранится ли общий сдвиг?\n2. Смените base с 10000 на 100.\n3. Передайте D=7 и объясните ошибку.',
  '1. Rotate Q only: does common-shift invariance survive?\n2. Change base from 10000 to 100.\n3. Use D=7 and explain the error.'],
 ['Если K не вращается согласованно, dot product зависит от абсолютного положения Q. Base меняет сетку частот. Нечётная ширина не разбивается на полные пары; partial rotary требует явно выделить чётную подчасть.',
  'Without consistently rotating K, dot products depend on Q’s absolute position. Base changes the frequency grid. Odd width cannot form complete pairs; partial rotary requires an explicitly selected even-sized subspace.'],
 ['Реализовано вращение и его свойства, не RoFormer LM и не метод расширения контекста.', 'Rotation and its properties only; no RoFormer LM or context-extension method.'],
 ['Разбор Attention, тригонометрия поворота и формы тензоров.', 'Attention lesson, 2D rotations, and tensor shapes.'])

paper('03-lora', 'LoRA: Low-Rank Adaptation of Large Language Models', 2021, 'Hu et al.', 'https://arxiv.org/abs/2106.09685', 'llm', '/',
 ['LoRA: адаптер, градиенты и merge', 'LoRA: adapters, gradients, and merging'],
 ['Создать низкоранговое обновление, обучить его и проверить эквивалентность merged весов.', 'Build and train a low-rank update, then verify merged-weight equivalence.'],
 ['При fine-tuning не обязательно менять каждую матрицу целиком. LoRA ограничивает обновление низким рангом, сохраняя базовые веса замороженными. Это ограничение класса обновлений, а не просто сжатие уже обученной матрицы.\n\nДве меньшие матрицы экономят trainable parameters и optimizer states. Но память базовой модели и активаций остаётся. В этой работе разбираем один Linear; подбор target modules в LLM и качество downstream задач — следующий уровень эксперимента.',
  'Fine-tuning need not update every full matrix. LoRA constrains updates to low rank while freezing base weights. This constrains the update family rather than merely compressing an already trained matrix.\n\nTwo smaller matrices reduce trainable parameters and optimizer states, but base-model and activation memory remain. We study one Linear; target-module selection and downstream quality require further experiments.'],
 ['`y = x W0^T + (alpha/r) x A^T B^T`, где `A:[r,in]`, `B:[out,r]`. Trainable count `r*(in+out)`.\n\nПри B=0 начальный output совпадает с базой. В первый backward A обычно получает нулевой градиент, а B — ненулевой; это ожидаемо. Если обе матрицы равны нулю, оба градиента исчезают. Merge: `Wmerged = W0 + (alpha/r) B A`.',
  '`y = x W0^T + (alpha/r) x A^T B^T`, with `A:[r,in]`, `B:[out,r]`. Trainable count is `r*(in+out)`.\n\nWith B=0, initial output matches the base. On the first backward A normally has zero gradient and B a nonzero one. If both start at zero, both gradients vanish. Merge uses `Wmerged = W0 + (alpha/r) B A`.'],
 [step('Шаг 1. Создаём слой и проверяем исходную функцию и асимметрию первых градиентов.', 'Step 1. Build the layer; verify initial equivalence and first-step gradient asymmetry.', '''
 class LoRALinear(nn.Module):
     def __init__(self, inputs=16, outputs=12, rank=3, alpha=6):
         super().__init__()
         self.base = nn.Linear(inputs, outputs, bias=False).requires_grad_(False)
         self.A = nn.Parameter(torch.randn(rank, inputs) * 0.02)
         self.B = nn.Parameter(torch.zeros(outputs, rank))
         self.scale = alpha / rank
     def forward(self, x):
         return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
 layer = LoRALinear(); x = torch.randn(64, 16)
 target = torch.randn(64, 12)
 assert torch.equal(layer(x), layer.base(x))
 F.mse_loss(layer(x), target).backward()
 print('first A/B grad:', layer.A.grad.norm().item(), layer.B.grad.norm().item())
 assert layer.A.grad.norm() == 0 and layer.B.grad.norm() > 0
 assert layer.base.weight.grad is None
 '''),
  step('Шаг 2. Обучаем только адаптер на контролируемом low-rank изменении. Target специально имеет низкий ранг — это проверка реализации, а не сравнение с full tuning.',
       'Step 2. Fit a controlled low-rank target using only adapters. The target is deliberately low-rank; this verifies implementation, not superiority to full tuning.', '''
 frozen = layer.base.weight.detach().clone()
 target_update = torch.randn(12, 3) @ torch.randn(3, 16) * 0.1
 target = F.linear(x, frozen + target_update)
 optimizer = torch.optim.AdamW([layer.A, layer.B], lr=0.03, weight_decay=0)
 losses = []
 for step in range(160):
     optimizer.zero_grad(set_to_none=True)
     loss = F.mse_loss(layer(x), target); loss.backward(); optimizer.step()
     losses.append(loss.item())
 assert torch.equal(layer.base.weight, frozen)
 assert losses[-1] < losses[0]
 print('trainable:', layer.A.numel() + layer.B.numel(), 'full matrix:', frozen.numel())
 plt.plot(losses, color='#d73535'); plt.xlabel('step'); plt.ylabel('adapter MSE'); plt.show()
 '''),
  step('Шаг 3. Merge должен сохранять функцию. После merge нельзя повторно добавлять адаптер к уже изменённой базе.',
       'Step 3. Merging must preserve the function. Do not add the adapter again on top of merged weights.', '''
 with torch.no_grad():
     merged_weight = layer.base.weight + layer.scale * (layer.B @ layer.A)
     adapter_output = layer(x)
     merged_output = F.linear(x, merged_weight)
 error = (adapter_output - merged_output).abs().max().item()
 print('merge error:', error)
 assert torch.allclose(adapter_output, merged_output, atol=1e-5)
 bad_A = nn.Parameter(torch.zeros(3, 16)); bad_B = nn.Parameter(torch.zeros(12, 3))
 (x @ bad_A.T @ bad_B.T - target).square().mean().backward()
 assert bad_A.grad.norm() == 0 and bad_B.grad.norm() == 0
 ''')],
 ['Снижение loss здесь ожидаемо: target построен как rank-3 update. Проверка замороженной базы выявляет случайное full tuning. Merge сравнивается с tolerance, поскольку порядок операций меняет округление float32.',
  'Loss should fall because the target is a rank-3 update. The frozen-base assertion detects accidental full tuning. Merge uses a tolerance because operation order changes float32 rounding.'],
 ['1. Поставьте rank=1 при rank-3 target.\n2. Увеличьте alpha, оставив learning rate прежним.\n3. Выполните merge дважды и сравните outputs.',
  '1. Use rank=1 for the rank-3 target.\n2. Increase alpha without changing learning rate.\n3. Merge twice and compare outputs.'],
 ['Rank-1 адаптер в общем случае не представит rank-3 update. Alpha масштабирует эффективное обновление. Повторный merge удваивает delta и меняет функцию: merge должен быть управляемой операцией состояния.',
  'A rank-1 adapter generally cannot represent a rank-3 update. Alpha scales the effective update. Merging twice doubles the delta and changes the function: merging must be state-aware.'],
 ['Один Linear без dropout/quantization. Не QLoRA, не PEFT integration и не воспроизведение benchmark статьи.',
  'One Linear without dropout or quantization; not QLoRA, PEFT integration, or paper benchmarks.'],
 ['Основы обучения и умножение матриц; вводный урок LoRA.', 'Training fundamentals, matrix products, and introductory LoRA.'])

paper('04-universal', 'Universal Transformers', 2018, 'Dehghani et al.', 'https://arxiv.org/abs/1807.03819', 'builder', '/builder',
 ['Universal Transformer: повторное ядро', 'Universal Transformer: a recurrent core'],
 ['Проверить sharing, depth signals и разницу полного и обрезанного backward.', 'Verify sharing, depth signals, and full versus truncated backpropagation.'],
 ['Universal Transformer повторяет общее преобразование по глубине, сохраняя параллельную обработку позиций. Модель может получать сигнал позиции и шага вычисления. В статье также рассматривается адаптивная остановка для разных позиций.\n\nПовторное использование одного объекта nn.Module — sharing; создание нескольких одинаково инициализированных объектов — нет. Глубина вычисления, число параметров и длина backward — разные величины. Ниже исследуем fixed-depth вариант: adaptive computation time намеренно не реализуется.',
  'Universal Transformers recurrently apply a shared transformation over depth while processing positions in parallel. Position and computation-step signals can be supplied. The paper also studies adaptive per-position halting.\n\nReusing one nn.Module establishes sharing; separately instantiated modules with equal initial values do not. Computation depth, parameter count, and backward depth are different quantities. We study fixed-depth recurrence and deliberately omit adaptive computation time.'],
 ['`h_(s+1) = Core_theta(h_s + depth_signal(s))`. Theta остаётся одной и той же на каждом шаге.\n\nFull BPTT суммирует пути градиента через все применения. `detach()` отсекает предыдущую историю; это другой estimator градиента. Checkpointing вместо detach может экономить activation memory через пересчёт, сохраняя математический градиент при согласованном RNG.',
  '`h_(s+1) = Core_theta(h_s + depth_signal(s))`, reusing theta at every step.\n\nFull BPTT combines gradient paths through all applications. detach cuts prior history and changes the gradient estimator. Checkpointing can instead trade recomputation for activation memory while preserving mathematical gradients with consistent RNG.'],
 [step('Шаг 1. Одно causal ядро получает детерминированный depth signal. Он не добавляет обучаемых параметров на каждый проход.',
       'Step 1. Supply one causal core with deterministic depth signals, adding no trainable parameters per iteration.', '''
 class SharedCore(nn.Module):
     def __init__(self, width=12):
         super().__init__()
         self.core = nn.TransformerEncoderLayer(width, 3, 24, dropout=0, batch_first=True)
         self.width = width
     def forward(self, x, depth, truncate=False):
         h = x
         causal = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), 1)
         for index in range(depth):
             if truncate and index:
                 h = h.detach()
             signal = torch.sin(torch.arange(self.width) / self.width + index) * 0.1
             h = self.core(h + signal, src_mask=causal)
         return h
 model = SharedCore(); x = torch.randn(2, 5, 12)
 count = sum(p.numel() for p in model.parameters())
 for depth in [1, 2, 4]:
     print('depth:', depth, 'parameters:', count, 'output:', model(x, depth).shape)
 assert len(list(model.children())) == 1
 '''),
  step('Шаг 2. Сравниваем градиенты при абсолютно одинаковом forward. Отличается только обратный проход.',
       'Step 2. Keep the forward computation identical and compare only backward behavior.', '''
 target = torch.randn_like(x)
 def gradient(truncate):
     model.zero_grad(set_to_none=True)
     output = model(x, 4, truncate)
     F.mse_loss(output, target).backward()
     return output.detach(), model.core.linear1.weight.grad.clone()
 full_output, full_gradient = gradient(False)
 truncated_output, truncated_gradient = gradient(True)
 assert torch.equal(full_output, truncated_output)
 print('gradient difference:', (full_gradient - truncated_gradient).norm().item())
 assert not torch.allclose(full_gradient, truncated_gradient)
 '''),
  step('Шаг 3. Учим на глубине 2, измеряем на нескольких глубинах. Большая глубина не обязана улучшать результат.',
       'Step 3. Train at depth 2, then measure several depths. More computation need not improve the result.', '''
 optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
 for step in range(40):
     optimizer.zero_grad(set_to_none=True)
     loss = F.mse_loss(model(x, 2), target); loss.backward(); optimizer.step()
 with torch.no_grad():
     depths = [1, 2, 3, 4, 6]
     losses = [F.mse_loss(model(x, d), target).item() for d in depths]
 print(dict(zip(depths, losses)))
 plt.plot(depths, losses, marker='o', color='#d73535'); plt.xlabel('inference depth'); plt.ylabel('training-batch MSE'); plt.show()
 ''')],
 ['Loss измеряется на том же маленьком batch: это проверка влияния глубины, не generalization benchmark. Равные outputs при разных градиентах показывают, почему detach нельзя считать безвредной оптимизацией памяти.',
  'Loss uses the same tiny training batch: this probes depth sensitivity, not generalization. Identical outputs with different gradients show why detach is not a harmless memory optimization.'],
 ['1. Замените общее ядро на четыре независимых.\n2. Уберите depth signal.\n3. Сделайте train depth случайной от 1 до 4 и сравните по отдельной выборке.',
  '1. Replace the shared core with four independent cores.\n2. Remove depth signals.\n3. Randomize training depth from 1 to 4 and compare on held-out data.'],
 ['Независимые ядра увеличивают число параметров. Без depth signal шаг вычисления известен только через evolving state. Random depth меняет распределение задач обучения; нужны одинаковый compute budget и несколько seed.',
  'Independent cores increase parameter count. Without depth signals, iteration information comes only through the evolving state. Random depth changes the training distribution; compare equal compute budgets and multiple seeds.'],
 ['Fixed-depth shared Transformer; нет adaptive halting, полного оригинального position/depth encoding и задач статьи.',
  'Fixed-depth shared Transformer only; no adaptive halting, exact original position/depth encoding, or paper tasks.'],
 ['Attention, residual blocks и вводный урок looped transformers.', 'Attention, residual blocks, and introductory looped transformers.'])

paper('05-clip', 'Learning Transferable Visual Models From Natural Language Supervision', 2021, 'Radford et al.', 'https://arxiv.org/abs/2103.00020', 'vlm', '/vlm',
 ['CLIP: совместить изображения и подписи', 'CLIP: align images and captions'],
 ['Обучить два encoder с симметричным contrastive loss и проверить retrieval и перепутанные пары.', 'Train two encoders with symmetric contrastive loss and test retrieval and shuffled pairs.'],
 ['CLIP учится сопоставлять изображение с текстовым описанием, а не генерировать описание токен за токеном. Два encoder дают векторы в одном пространстве; правильные пары сближаются относительно других пар batch.\n\nОтрицательные примеры здесь — остальные элементы batch. Если разные подписи обозначают одно и то же изображение, обычный single-positive loss может считать их ложными отрицательными. В маленьком эксперименте используем уникальные синтетические пары. Это позволяет изучить objective, не скачивая pretrained CLIP.',
  'CLIP aligns images and descriptions rather than generating captions token by token. Two encoders produce vectors in a shared space; matching pairs are favored relative to other batch pairs.\n\nOther batch items supply negatives. If multiple captions describe the same image, a single-positive objective may introduce false negatives. We use unique synthetic pairs to study the objective without downloading pretrained CLIP.'],
 ['`u = normalize(image_encoder(x))`, `v = normalize(text_encoder(t))`. `S = exp(log_scale) * u @ v.T`.\n\n`L = (CE(S, arange(B)) + CE(S.T, arange(B))) / 2`. Диагональ содержит совпадающие пары; строки — image→text, столбцы — text→image. Нормализация отделяет направление вектора от его длины. Temperature управляет резкостью softmax, но не заменяет хорошие representations.',
  '`u = normalize(image_encoder(x))`, `v = normalize(text_encoder(t))`, and `S = exp(log_scale) * u @ v.T`.\n\n`L = (CE(S, arange(B)) + CE(S.T, arange(B))) / 2`. Diagonal entries are positives; rows perform image→text and columns text→image retrieval. Normalization separates direction from magnitude. Temperature controls softmax sharpness but cannot replace good representations.'],
 [step('Шаг 1. Делаем две модальности из общего скрытого сигнала. Linear encoder заменяют большие vision/text towers.',
       'Step 1. Generate two modalities from shared latent information. Linear encoders replace large vision/text towers.', '''
 batch = 24
 latent = torch.randn(batch, 6)
 image_features = latent @ torch.randn(6, 12) + 0.01 * torch.randn(batch, 12)
 text_features = latent @ torch.randn(6, 10) + 0.01 * torch.randn(batch, 10)
 image_encoder = nn.Linear(12, 16); text_encoder = nn.Linear(10, 16)
 log_scale = nn.Parameter(torch.tensor(2.0))
 labels = torch.arange(batch)
 def similarities():
     u = F.normalize(image_encoder(image_features), dim=-1)
     v = F.normalize(text_encoder(text_features), dim=-1)
     return log_scale.exp().clamp(max=100) * (u @ v.T)
 scores = similarities()
 print('similarity shape:', scores.shape)
 assert scores.shape == (batch, batch)
 '''),
  step('Шаг 2. Обновляем оба encoder. Retrieval accuracy здесь относится только к memorized batch.',
       'Step 2. Update both encoders. Retrieval accuracy here measures only the memorized batch.', '''
 optimizer = torch.optim.AdamW([*image_encoder.parameters(), *text_encoder.parameters(), log_scale], lr=0.02)
 history = []
 for step in range(120):
     scores = similarities()
     loss = (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
     optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
     history.append(loss.item())
 with torch.no_grad():
     scores = similarities()
     accuracy = (scores.argmax(-1) == labels).float().mean().item()
 print('training-batch retrieval:', accuracy)
 assert history[-1] < history[0]
 plt.imshow(scores.detach(), cmap='Reds'); plt.xlabel('text index'); plt.ylabel('image index'); plt.colorbar(); plt.show()
 '''),
  step('Шаг 3. Переставляем подписи, но оставляем старые labels. Это контролируемая проверка того, что loss зависит от pairing.',
       'Step 3. Permute captions while keeping the old labels: a controlled pairing diagnostic.', '''
 permutation = torch.roll(torch.arange(batch), 1)
 aligned_loss = F.cross_entropy(scores, labels)
 wrong_pair_loss = F.cross_entropy(scores[:, permutation], labels)
 print('aligned loss:', aligned_loss.item(), 'wrong-pair loss:', wrong_pair_loss.item())
 assert wrong_pair_loss > aligned_loss
 # Correct labels restore equivalence after a column permutation.
 inverse = torch.argsort(permutation)
 corrected = F.cross_entropy(scores[:, permutation], inverse)
 assert torch.allclose(corrected, aligned_loss, atol=1e-6)
 ''')],
 ['Диагональная структура показывает выученное соответствие обучающих пар. Она не доказывает zero-shot качество на новых изображениях. Перестановка и обратная перестановка labels помогают найти ошибки индексации.',
  'A strong diagonal establishes alignment of the training pairs, not zero-shot performance on new images. Permutations and inverse label permutations help expose indexing bugs.'],
 ['1. Уберите normalize и следите за нормами encoder outputs.\n2. Продублируйте одну пару: что будет с single-positive labels?\n3. Сделайте отдельный test latent и оцените retrieval.',
  '1. Remove normalization and track output norms.\n2. Duplicate a pair: what happens to single-positive labels?\n3. Generate held-out latent samples and evaluate retrieval.'],
 ['Без normalize модель может менять scores через длину векторов. Дубликаты создают неоднозначные positives и требуют другого target/loss. Test должен быть с новыми latent при тех же генераторах модальностей; иначе меняется сама задача.',
  'Without normalization, vector magnitude can change scores. Duplicates create ambiguous positives and need appropriate targets/loss. For testing, use new latent samples with the same modality generators; otherwise the task itself changes.'],
 ['Реализован CLIP-style objective на синтетических признаках. Нет image/text towers, web-scale data или zero-shot benchmark.',
  'CLIP-style objective on synthetic features; no image/text towers, web-scale data, or zero-shot benchmark.'],
 ['Softmax, cross entropy, базовый урок VLM.', 'Softmax, cross entropy, and introductory VLM.'])

paper('06-dpo', 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model', 2023, 'Rafailov et al.', 'https://arxiv.org/abs/2305.18290', 'rl', '/',
 ['DPO: учиться по парам предпочтений', 'DPO: learning from preference pairs'],
 ['Связать sequence log-probabilities с DPO objective, проверить masking, reference и знак обновления.', 'Connect sequence log-probabilities to DPO and test masking, the reference, and update direction.'],
 ['DPO превращает предпочтения chosen/rejected в loss без отдельного обучения reward model в этом objective. Нужны policy и фиксированная reference. Отношение их вероятностей задаёт implicit reward; chosen должен выигрывать относительно rejected именно по этому отношению.\n\nПросто увеличить вероятность chosen недостаточно для проверки: важна разница с reference. Dataset предпочтений может быть шумным и смещённым. Ниже используем три искусственные пары и маленькую categorical policy, чтобы видеть каждое число.',
  'DPO converts chosen/rejected preferences into a loss without separately fitting a reward model in this objective. It uses a policy and fixed reference. Their probability ratio defines an implicit reward: chosen should beat rejected in that relative measure.\n\nIncreasing chosen probability alone is not the full diagnostic; the difference from reference matters. Preference datasets can be noisy and biased. We use three synthetic pairs and a tiny categorical policy so every quantity is visible.'],
 ['`margin = beta * [(logpi_ch - logpi_rej) - (logref_ch - logref_rej)]`; `loss = -logsigmoid(margin)`.\n\nДля LM logpi — сумма log probabilities ответа с causal shift и token mask. Среднее по токенам вместо суммы меняет objective и length effects. Reference заморожена; beta масштабирует margin. Логарифмы считаем через log_softmax, а не log(softmax), чтобы избежать потери устойчивости.',
  '`margin = beta * [(logpi_ch - logpi_rej) - (logref_ch - logref_rej)]`; `loss = -logsigmoid(margin)`.\n\nFor an LM, logpi sums response-token log-probabilities after causal shifting and masking. Token means change the objective and length effects. Freeze the reference; beta scales the margin. Use log_softmax instead of log(softmax) for stability.'],
 [step('Шаг 1. Проверяем вычисление sequence log-probabilities. Изменение masked token не должно менять сумму.',
       'Step 1. Check sequence log-probabilities. Modifying a masked token must not change the sum.', '''
 def sequence_logp(logits, targets, mask):
     token_logp = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
     return (token_logp * mask).sum(-1)
 logits = torch.randn(2, 4, 7)
 targets = torch.randint(0, 7, (2, 4)); mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
 changed = logits.clone(); changed[mask == 0] = torch.randn_like(changed[mask == 0]) * 10
 assert torch.equal(sequence_logp(logits, targets, mask), sequence_logp(changed, targets, mask))
 print('sequence log-probs:', sequence_logp(logits, targets, mask))
 '''),
  step('Шаг 2. У reference и policy в начале одинаковые logits: margin=0 и loss=log(2), даже если raw probabilities неравны.',
       'Step 2. Initialize policy and reference identically: margin=0 and loss=log(2), even with unequal raw probabilities.', '''
 policy = nn.Parameter(torch.tensor([[0.2, -0.3], [-0.2, 0.8], [0., 0.]]))
 reference = policy.detach().clone().log_softmax(-1)
 beta = 0.5
 def dpo():
     logp = policy.log_softmax(-1)
     margins = beta * ((logp[:, 0] - logp[:, 1]) - (reference[:, 0] - reference[:, 1]))
     return -F.logsigmoid(margins).mean(), margins
 loss, initial_margin = dpo()
 assert torch.allclose(initial_margin, torch.zeros(3))
 assert abs(loss.item() - math.log(2)) < 1e-6
 loss.backward()
 print('chosen/rejected gradients:', policy.grad)
 assert (policy.grad[:, 0] < 0).all() and (policy.grad[:, 1] > 0).all()
 '''),
  step('Шаг 3. Reference остаётся неизменной, margins растут. График показывает objective, а не полезность ответов.',
       'Step 3. Keep the reference fixed and observe growing margins. This plots the objective, not answer usefulness.', '''
 frozen_reference = reference.clone()
 optimizer = torch.optim.SGD([policy], lr=0.5)
 history = []
 for step in range(40):
     optimizer.zero_grad(set_to_none=True)
     loss, margins = dpo(); loss.backward(); optimizer.step()
     history.append(loss.item())
 assert torch.equal(reference, frozen_reference)
 assert (dpo()[1] > initial_margin).all()
 plt.plot(history, color='#d73535'); plt.ylabel('DPO loss'); plt.xlabel('step'); plt.show()
 print('final relative margins:', dpo()[1].detach())
 ''')],
 ['Loss=log(2) на старте — хороший regression check для policy=reference. Неверная token mask или обновление reference изменит смысл обучения, даже если loss всё ещё падает. В настоящей задаче предпочтения нужно оценивать на held-out prompts.',
  'Initial log(2) is a useful policy=reference regression check. Incorrect masks or an updated reference change training semantics even if loss falls. Real preference quality needs held-out prompt evaluation.'],
 ['1. Пересчитывайте reference после каждого шага: что произойдёт с margin?\n2. Поменяйте chosen/rejected местами.\n3. Сравните сумму и среднее log-probs для ответов разной длины.',
  '1. Recompute the reference after every step: what happens to the margin?\n2. Swap chosen and rejected.\n3. Compare sums and means for different response lengths.'],
 ['Если reference постоянно совпадает с policy, относительный margin снова обнуляется. Swap меняет знак сигнала. Mean меняет нормировку по длине; это отдельный objective, а не безусловное исправление DPO.',
  'A reference constantly matched to the policy resets the relative margin. Swapping preference labels reverses the signal. Means alter length normalization; that is a different objective, not a universal DPO correction.'],
 ['DPO objective и masking. Нет реальной language generation, preference corpus или reproduction human eval статьи.',
  'DPO objective and masking only; no real language generation, preference corpus, or paper human-evaluation reproduction.'],
 ['Causal LM, log_softmax, response masks; вводный урок предпочтений.', 'Causal LM, log_softmax, response masks, and introductory preferences.'])

paper('07-grpo', 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models', 2024, 'Shao et al.', 'https://arxiv.org/abs/2402.03300', 'rl', '/',
 ['DeepSeekMath: разобрать GRPO objective', 'DeepSeekMath: dissecting the GRPO objective'],
 ['Разделить group advantages, old policy, reference KL и response masks; проверить clipping и постоянные rewards.',
  'Separate group advantages, old policy, reference KL, and response masks; test clipping and constant rewards.'],
 ['GRPO использует группу ответов на один prompt для оценки относительного качества без отдельного value model. Важно не смешивать prompts при нормировке rewards. Old policy — распределение, собравшее rollout; reference — политика для регуляризации, если KL включена. Это разные роли.\n\nReward может быть формальным и проверяемым, но его всё равно можно эксплуатировать. Падающий policy loss не означает улучшение reasoning. Здесь фиксированные искусственные responses; generation, проверка математических ответов и dataset статьи не воспроизводятся.',
  'GRPO uses a response group per prompt to estimate relative quality without a separate value model. Do not mix prompts when normalizing rewards. Old policy collected the rollout; the reference supplies regularization when KL is enabled. These roles differ.\n\nEven formal, verifiable rewards can be exploited. Falling policy loss does not establish improved reasoning. We use fixed synthetic responses and do not reproduce generation, mathematical verification, or the paper dataset.'],
 ['`A_i = (r_i - mean(group)) / max(std(group), eps)`. Для reward одинаковых внутри группы advantage=0. Здесь явно выбираем population std (`unbiased=False`); trainer conventions могут отличаться.\n\n`ratio = exp(logpi - logpi_old)`; surrogate=`min(ratio*A, clip(ratio,1-e,1+e)*A)`. Token KL estimator: `exp(logref-logpi) - (logref-logpi) - 1`. Сначала усредняем по валидным токенам каждого ответа, затем по ответам. Old/reference log-probs detached.',
  '`A_i = (r_i - mean(group)) / max(std(group), eps)`. Constant within-group rewards give zero advantages. We explicitly use population std (`unbiased=False`); trainer conventions can differ.\n\n`ratio = exp(logpi - logpi_old)`; surrogate=`min(ratio*A, clip(ratio,1-e,1+e)*A)`. The token KL estimator is `exp(logref-logpi) - (logref-logpi) - 1`. Average valid tokens per response, then responses. Detach old/reference log-probabilities.'],
 [step('Шаг 1. Два prompts, по три ответа. Вторая группа имеет одинаковую награду и не даёт относительного сигнала.',
       'Step 1. Two prompts with three responses each. The second group has constant rewards and no relative signal.', '''
 rewards = torch.tensor([[0., 1., 3.], [2., 2., 2.]])
 advantage = (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
 assert torch.isfinite(advantage).all()
 assert torch.equal(advantage[1], torch.zeros(3))
 assert torch.allclose(advantage.mean(-1), torch.zeros(2), atol=1e-6)
 print('advantages:', advantage)
 mask = torch.tensor([[[1, 1, 0], [1, 1, 1], [1, 0, 0]], [[1, 1, 0], [1, 1, 1], [1, 0, 0]]], dtype=torch.float)
 '''),
  step('Шаг 2. Считаем objective по token log-probs. Это числовой unit experiment: свободные log-probs не являются полноценной нормированной policy.',
       'Step 2. Compute a token-level objective. Free log-probability variables are a numerical unit experiment, not a fully normalized policy.', '''
 old_logp = torch.full((2, 3, 3), -2.)
 reference_logp = old_logp.clone()
 current_logp = old_logp.clone().requires_grad_()
 def objective(current):
     ratio = (current - old_logp).exp()
     A = advantage[..., None]
     surrogate = torch.minimum(ratio * A, ratio.clamp(0.8, 1.2) * A)
     difference = reference_logp - current
     kl = difference.exp() - difference - 1
     token_objective = surrogate - 0.04 * kl
     loss = -((token_objective * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
     return loss, kl
 loss, kl = objective(current_logp); loss.backward()
 assert torch.equal(kl, torch.zeros_like(kl))
 assert torch.equal(current_logp.grad[mask == 0], torch.zeros_like(current_logp.grad[mask == 0]))
 assert torch.equal(current_logp.grad[1], torch.zeros_like(current_logp.grad[1]))
 print('initial loss:', loss.item(), 'gradient:', current_logp.grad)
 '''),
  step('Шаг 3. Визуализируем clipping для положительного и отрицательного advantage, отдельно от KL.',
       'Step 3. Plot clipping for positive and negative advantages, independently of KL.', '''
 ratios = torch.linspace(0.4, 1.6, 121)
 for A, color in [(1., '#d73535'), (-1., '#777777')]:
     clipped = torch.minimum(ratios * A, ratios.clamp(0.8, 1.2) * A)
     plt.plot(ratios, clipped, label=f'advantage={A}', color=color)
 plt.axvline(0.8, ls=':', color='gray'); plt.axvline(1.2, ls=':', color='gray')
 plt.xlabel('policy ratio'); plt.ylabel('clipped surrogate'); plt.legend(); plt.show()
 varied = old_logp.clone(); varied[0, 0, 2] += 5
 assert torch.allclose(objective(varied)[0], objective(old_logp)[0])
 ''')],
 ['Для A>0 plateau появляется справа от 1+epsilon; для A<0 — слева от 1-epsilon. Clipping не запрещает любые большие изменения параметров и не гарантирует ограниченный KL. Нулевой advantage не означает отсутствие KL-сигнала, если policy ушла от reference.',
  'For A>0 the plateau is above 1+epsilon; for A<0 it is below 1-epsilon. Clipping neither prevents every large parameter change nor guarantees bounded KL. Zero advantage does not remove KL pressure if policy differs from reference.'],
 ['1. Нормируйте rewards по всему batch и найдите ошибку смысла.\n2. Измените current_logp второй группы и проверьте KL gradient.\n3. Сравните token-average и response-average при разной длине.',
  '1. Normalize across the full batch and identify the semantic error.\n2. Change the second group’s current log-probs and inspect KL gradients.\n3. Compare token averages with response averages for different lengths.'],
 ['Batch-нормировка сравнивает ответы разных задач. При отличии от reference KL даёт сигнал даже без reward advantage. Token-average сильнее взвешивает длинные ответы; response-average даёт одинаковый вес каждому ответу.',
  'Batch normalization compares responses to different tasks. Reference divergence can supply a KL signal even with zero reward advantage. Token averages weight longer answers more; response averages give each response equal weight.'],
 ['Objective и диагностические инварианты. Не полный GRPO trainer, не обучение DeepSeekMath и не reproduction математических результатов.',
  'Objective and diagnostic invariants only; not a full GRPO trainer, DeepSeekMath training, or mathematical benchmark reproduction.'],
 ['Вводный GRPO, log probabilities, masks и градиенты.', 'Introductory GRPO, log-probabilities, masks, and gradients.'])

paper('08-ddpm', 'Denoising Diffusion Probabilistic Models', 2020, 'Ho, Jain & Abbeel', 'https://arxiv.org/abs/2006.11239', 'world-models', '/world-models',
 ['DDPM: от шума к выборке', 'DDPM: from noise to samples'],
 ['Написать forward noising, epsilon predictor и полный маленький reverse chain на одномерных данных.',
  'Implement forward noising, an epsilon predictor, and a complete tiny reverse chain on one-dimensional data.'],
 ['Diffusion постепенно разрушает данные шумом и учится обращать этот процесс. Модель получает x_t и t, предсказывая добавленный noise. На inference настоящего x0 уже нет: генерация начинается с Gaussian noise.\n\nМеханизм полезен для понимания diffusion world models, но сам по себе не моделирует действие и время среды. Здесь каждый объект — одно число из смеси двух Gaussians. Это осознанно маленькая постановка для проверки формул, не генератор изображений или DIAMOND.',
  'Diffusion gradually corrupts data and learns to reverse that process. The model receives x_t and t and predicts added noise. At inference, true x0 is unavailable: generation starts from Gaussian noise.\n\nThis mechanism helps explain diffusion world models but does not itself condition on actions or environment time. Each sample here is one scalar from a two-Gaussian mixture, a small formula-checking task rather than image generation or DIAMOND.'],
 ['`alpha_t = 1-beta_t`; `abar_t = product(alpha_0...alpha_t)`. `x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*epsilon`. Loss=`MSE(epsilon_theta(x_t,t), epsilon)`.\n\nReverse mean: `(x_t - beta_t/sqrt(1-abar_t)*eps_hat)/sqrt(alpha_t)`. Здесь используем fixed posterior variance `beta_t*(1-abar_prev)/(1-abar_t)`. На последнем шаге t=0 шум не добавляем. Время diffusion и время rollout среды — разные оси.',
  '`alpha_t = 1-beta_t`; `abar_t = product(alpha_0...alpha_t)`. `x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*epsilon`. Loss is `MSE(epsilon_theta(x_t,t), epsilon)`.\n\nReverse mean is `(x_t - beta_t/sqrt(1-abar_t)*eps_hat)/sqrt(alpha_t)`. We use fixed posterior variance `beta_t*(1-abar_prev)/(1-abar_t)`. Add no noise at final t=0. Diffusion time and environment rollout time are distinct axes.'],
 [step('Шаг 1. Проверяем schedule и точное восстановление x0, если известен настоящий noise. Это oracle sanity check, не генерация.',
       'Step 1. Check the schedule and recover x0 using known noise. This is an oracle sanity check, not generation.', '''
 T = 80
 beta = torch.linspace(1e-4, 0.2, T)
 alpha = 1 - beta; abar = alpha.cumprod(0)
 previous = torch.cat([torch.ones(1), abar[:-1]])
 posterior_variance = beta * (1 - previous) / (1 - abar)
 assert (abar[1:] < abar[:-1]).all() and posterior_variance[0] == 0
 x0 = torch.randn(128, 1); epsilon = torch.randn_like(x0); t = torch.randint(T, (128,))
 xt = abar[t, None].sqrt() * x0 + (1 - abar[t, None]).sqrt() * epsilon
 recovered = (xt - (1 - abar[t, None]).sqrt() * epsilon) / abar[t, None].sqrt()
 assert torch.allclose(recovered, x0, atol=1e-4)
 print('terminal alpha_bar:', abar[-1].item())
 '''),
  step('Шаг 2. Учим маленькую MLP предсказывать epsilon. Данные генерируются заново на каждом шаге.',
       'Step 2. Train a tiny MLP to predict epsilon, resampling training data each step.', '''
 def data(batch):
     return (torch.randint(0, 2, (batch, 1)).float() * 2 - 1) * 2 + 0.3 * torch.randn(batch, 1)
 predictor = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
 def predict(x, time):
     return predictor(torch.cat([x, time[:, None].float() / (T - 1)], -1))
 optimizer = torch.optim.Adam(predictor.parameters(), lr=0.003)
 history = []
 for step in range(400):
     clean = data(128); time = torch.randint(T, (128,)); noise = torch.randn_like(clean)
     noisy = abar[time, None].sqrt() * clean + (1 - abar[time, None]).sqrt() * noise
     loss = F.mse_loss(predict(noisy, time), noise)
     optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); history.append(loss.item())
 assert math.isfinite(history[-1])
 plt.plot(history, color='#d73535', alpha=0.6); plt.xlabel('step'); plt.ylabel('epsilon MSE'); plt.show()
 '''),
  step('Шаг 3. Reverse chain не видит clean data. Сравнение гистограмм диагностическое: маленькая сеть может давать плохие выборки.',
       'Step 3. The reverse chain never sees clean data. Histograms are diagnostic: this small network may generate poor samples.', '''
 with torch.no_grad():
     samples = torch.randn(1000, 1)
     for index in reversed(range(T)):
         time = torch.full((len(samples),), index, dtype=torch.long)
         epsilon_hat = predict(samples, time)
         mean = (samples - beta[index] / (1 - abar[index]).sqrt() * epsilon_hat) / alpha[index].sqrt()
         samples = mean + posterior_variance[index].sqrt() * torch.randn_like(samples) if index else mean
 assert torch.isfinite(samples).all()
 plt.hist(data(1000).flatten(), bins=50, density=True, alpha=0.4, color='gray', label='data')
 plt.hist(samples.flatten(), bins=50, density=True, alpha=0.5, color='#d73535', label='generated')
 plt.legend(); plt.xlabel('scalar value'); plt.show()
 ''')],
 ['Train loss усредняет разные noise levels; отдельно полезно смотреть MSE по t. Ошибка индекса schedule или дополнительный noise на t=0 может испортить sampling при рабочем backward. Совпадение средних не доказывает совпадение распределений.',
  'Training loss averages different noise levels; inspect MSE by t separately. Schedule indexing or noise added at t=0 can break sampling despite valid backward. Matching means does not establish matching distributions.'],
 ['1. Уберите t из predictor.\n2. Сделайте terminal abar близким к 1: корректен ли Gaussian старт?\n3. Добавьте conditioning на действие и объясните, откуда взять transition dataset.',
  '1. Remove t from the predictor.\n2. Make terminal abar near 1: is Gaussian initialization justified?\n3. Add action conditioning and specify the required transition dataset.'],
 ['Без t сеть смешивает разные задачи denoising. Если terminal data остаётся заметной, standard normal старт хуже соответствует forward endpoint. Для action-conditioned world model нужны observation/action/next-observation triples и независимые эпизоды для оценки.',
  'Without t, the network mixes denoising tasks. If terminal samples retain strong data signal, normal initialization mismatches the forward endpoint. Action-conditioned world models need observation/action/next-observation triples and held-out episodes.'],
 ['Одномерный DDPM с tiny MLP и коротким schedule. Нет U-Net, image benchmarks, classifier-free guidance или world-model rollout.',
  'One-dimensional DDPM with a tiny MLP and short schedule; no U-Net, image benchmarks, classifier-free guidance, or world-model rollout.'],
 ['Gaussian noise, MSE, оптимизатор и вводный world-model урок.', 'Gaussian noise, MSE, optimization, and introductory world models.'])

paper('09-flow', 'Flow Matching for Generative Modeling', 2022, 'Lipman et al.', 'https://arxiv.org/abs/2210.02747', 'vla', '/vla',
 ['Flow Matching: обучить поле скоростей', 'Flow Matching: learn a velocity field'],
 ['Обучить conditional velocity objective и получить выборки численным решением ODE.', 'Train a conditional velocity objective and sample by numerically solving an ODE.'],
 ['Flow matching учит поле скоростей, которое переносит начальное распределение в распределение данных. На training можно выбрать точки пути и их целевые скорости, не решая ODE на каждом шаге. На sampling интегрируем обученное поле.\n\nДля action policies поле дополнительно зависит от наблюдения и инструкции. Здесь conditioning отсутствует: разбираем одномерное распределение. Берём независимые пары noise/data и линейную интерполяцию с нулевой конечной дисперсией. Это частный conditional path, а не решение глобальной optimal transport задачи.',
  'Flow matching learns a velocity field transporting a base distribution to data. Training can sample path points and target velocities without solving an ODE on every step; sampling integrates the learned field.\n\nAction policies additionally condition the field on observations and instructions. We omit conditioning and study a one-dimensional distribution using independent noise/data pairs and linear interpolation with zero endpoint noise. This is a particular conditional path, not a global optimal-transport solution.'],
 ['`x_t = (1-t)*x0 + t*x1`; target velocity=`x1-x0`; `L = E[||v_theta(x_t,t) - (x1-x0)||²]`.\n\nВ каждой паре скорость постоянна, но conditional paths пересекаются: оптимальная сеть предсказывает условное среднее velocity по доступным x_t,t. На sampling: `x ← x + dt*v_theta(x,t)`, t растёт от 0 до 1. Обратное направление требует согласованно поменять знак/определение времени.',
  '`x_t = (1-t)*x0 + t*x1`; target velocity is `x1-x0`; `L = E[||v_theta(x_t,t) - (x1-x0)||²]`.\n\nEach pair has constant velocity, but conditional paths overlap: the network predicts conditional mean velocity given x_t,t. Euler sampling uses `x ← x + dt*v_theta(x,t)` as t goes from 0 to 1. Reversing time requires consistently changing the sign/time convention.'],
 [step('Шаг 1. Проверяем endpoints и производную линейного пути через finite differences.',
       'Step 1. Verify endpoints and the path derivative with finite differences.', '''
 x0 = torch.randn(32, 1); x1 = 2.0 + 0.5 * torch.randn(32, 1)
 def path(t):
     return (1 - t) * x0 + t * x1
 assert torch.equal(path(0), x0) and torch.equal(path(1), x1)
 dt = 1e-3
 derivative = (path(0.4 + dt) - path(0.4)) / dt
 assert torch.allclose(derivative, x1 - x0, atol=3e-4)
 print('endpoint means:', x0.mean().item(), x1.mean().item())
 '''),
  step('Шаг 2. Сеть видит x_t и t, а не пару endpoints. Это отличие training target от входа модели.',
       'Step 2. The model sees x_t and t, not both endpoints. Training targets and model inputs are different.', '''
 velocity = nn.Sequential(nn.Linear(2, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 1))
 optimizer = torch.optim.Adam(velocity.parameters(), lr=0.003)
 history = []
 for step in range(300):
     base = torch.randn(128, 1); data = 2.0 + 0.5 * torch.randn(128, 1)
     t = torch.rand(128, 1); point = (1 - t) * base + t * data
     loss = F.mse_loss(velocity(torch.cat([point, t], -1)), data - base)
     optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); history.append(loss.item())
 assert math.isfinite(history[-1])
 plt.plot(history, color='#d73535', alpha=0.6); plt.xlabel('step'); plt.ylabel('velocity MSE'); plt.show()
 '''),
  step('Шаг 3. Сравниваем шаги Euler при одинаковом начальном шуме. Различие отражает discretization и ошибку выученного поля.',
       'Step 3. Compare Euler step counts from identical initial noise. Differences reflect discretization and field approximation.', '''
 initial = torch.randn(1000, 1)
 def sample(steps):
     x = initial.clone()
     with torch.no_grad():
         for index in range(steps):
             t = torch.full_like(x, index / steps)
             x = x + velocity(torch.cat([x, t], -1)) / steps
     return x
 coarse, fine = sample(5), sample(50)
 assert torch.isfinite(fine).all()
 print('Euler 5/50 means:', coarse.mean().item(), fine.mean().item(), 'target mean: 2')
 print('fine std:', fine.std().item(), 'target std: 0.5')
 plt.hist(fine.flatten(), bins=40, density=True, alpha=0.6, color='#d73535', label='generated')
 plt.hist((2 + 0.5 * torch.randn(1000)).flatten(), bins=40, density=True, alpha=0.3, color='gray', label='target')
 plt.legend(); plt.show()
 ''')],
 ['Training MSE не обязан идти к нулю: для одного x_t,t возможны разные пары и velocities. Больше sampling steps уменьшает discretization error, но не исправляет ошибочное поле. Для VLA нужны ещё conditioning, action normalization и dataset-specific masks.',
  'Training MSE need not reach zero: multiple pairs can share x_t,t but have different velocities. More sampling steps can reduce discretization error, not fix a wrong field. VLA also needs conditioning, action normalization, and dataset-specific masks.'],
 ['1. Поменяйте знак velocity на sampling.\n2. Сравните 1, 5, 20, 100 шагов при одном seed.\n3. Добавьте бинарное conditioning: target mean -2 или +2.',
  '1. Reverse the sampling velocity sign.\n2. Compare 1, 5, 20, 100 steps with the same seed.\n3. Add binary conditioning for target means -2 and +2.'],
 ['Неправильный знак переносит распределение не в ту сторону. Сравнивать нужно одинаковые initial samples. Conditional label должен присутствовать и при training, и при sampling; иначе сеть смешивает две задачи.',
  'The wrong sign transports samples in the wrong direction. Compare identical initial samples. Supply conditioning labels at both training and sampling; otherwise the network mixes the tasks.'],
 ['Простой linear conditional flow matching. Нет полного π0/SmolVLA, vision encoder, робота и benchmark статьи.',
  'Simple linear conditional flow matching; no complete π0/SmolVLA, vision encoder, robot, or paper benchmark.'],
 ['Производная, MSE, Gaussian samples; вводный VLA.', 'Derivatives, MSE, Gaussian samples, and introductory VLA.'])

paper('10-act', 'Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware', 2023, 'Zhao et al.', 'https://arxiv.org/abs/2304.13705', 'vla', '/vla',
 ['ACT: chunks, CVAE и temporal ensemble', 'ACT: chunks, CVAE, and temporal ensembling'],
 ['Написать маленькую conditional VAE для chunks действий, отличить posterior training от inference и объединить перекрывающиеся predictions.',
  'Build a tiny conditional VAE for action chunks, separate posterior training from inference, and ensemble overlapping predictions.'],
 ['ACT предсказывает последовательность будущих действий вместо одного следующего действия. CVAE позволяет моделировать вариативность демонстраций: posterior во время обучения видит target actions, decoder использует latent вместе с наблюдением. На inference target actions недоступны.\n\nTemporal ensembling объединяет предсказания для одного абсолютного момента, сделанные в разные моменты. Это не среднее произвольных позиций chunk. Здесь state-only MLP заменяет visual Transformer, чтобы детально проверить objective и индексацию на CPU.',
  'ACT predicts future action chunks rather than a single next action. Its CVAE models demonstration variability: the training posterior sees target actions, while the decoder uses a latent and observation. Target actions are unavailable at inference.\n\nTemporal ensembling combines predictions for the same absolute time made at different times, not arbitrary chunk positions. A state-only MLP replaces the visual Transformer so we can inspect the objective and indexing on CPU.'],
 ['`z = mu + exp(0.5*logvar)*epsilon`; `KL(q||N(0,I)) = 0.5*sum(mu² + exp(logvar) - 1 - logvar)`.\n\n`loss = masked_L1(actions_hat, actions) + beta*KL`. Reconstruction усредняется только по валидным координатам; KL — по batch после суммы latent dimensions. В этом примере inference берёт z=0. Ensemble ниже использует `weight=exp(-k*age)`, где age больше у более старого chunk: это явно выбранная recency-weighted convention, не утверждение о точной convention оригинального кода.',
  '`z = mu + exp(0.5*logvar)*epsilon`; `KL(q||N(0,I)) = 0.5*sum(mu² + exp(logvar) - 1 - logvar)`.\n\n`loss = masked_L1(actions_hat, actions) + beta*KL`. Average reconstruction over valid coordinates; sum latent dimensions before averaging KL over the batch. Inference here uses z=0. Our ensemble uses `weight=exp(-k*age)`, with larger age for older chunks: an explicit recency-weighted convention, not a claim about the original code’s exact weighting convention.'],
 [step('Шаг 1. Posterior получает state и demonstration chunk; inference decoder получает только state и z=0.',
       'Step 1. The posterior receives state and the demonstration chunk; inference decoding uses only state and z=0.', '''
 B, K, A, Z = 48, 4, 2, 3
 state = torch.randn(B, 3)
 actions = state[:, None, :2] * 0.3 + torch.arange(K)[None, :, None] * 0.1
 valid = torch.ones(B, K, dtype=torch.bool); valid[:12, -1] = False
 class ChunkCVAE(nn.Module):
     def __init__(self):
         super().__init__()
         self.posterior = nn.Sequential(nn.Linear(3 + K*A, 32), nn.ReLU(), nn.Linear(32, 2*Z))
         self.decoder = nn.Sequential(nn.Linear(3 + Z, 32), nn.ReLU(), nn.Linear(32, K*A))
     def decode(self, observation, z):
         return self.decoder(torch.cat([observation, z], -1)).reshape(-1, K, A)
     def forward(self, observation, targets, mask):
         # Zero padded targets before feeding the posterior as well as masking loss.
         encoded = torch.cat([observation, (targets * mask[..., None]).flatten(1)], -1)
         mu, logvar = self.posterior(encoded).chunk(2, -1)
         z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
         return self.decode(observation, z), mu, logvar
 model = ChunkCVAE()
 assert model.decode(state, torch.zeros(B, Z)).shape == actions.shape
 '''),
  step('Шаг 2. Проверяем нулевой KL стандартного нормального posterior и обучаем objective с маской.',
       'Step 2. Verify zero KL for a standard-normal posterior, then train the masked objective.', '''
 def kl_normal(mu, logvar):
     return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
 assert kl_normal(torch.zeros(B, Z), torch.zeros(B, Z)) == 0
 optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
 history = []
 for step in range(100):
     prediction, mu, logvar = model(state, actions, valid)
     reconstruction = ((prediction - actions).abs() * valid[..., None]).sum() / (valid.sum() * A)
     loss = reconstruction + 0.01 * kl_normal(mu, logvar)
     optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); history.append(loss.item())
 with torch.no_grad():
     predicted = model.decode(state, torch.zeros(B, Z))
 print('z=0 inference L1 on train:', (predicted - actions).abs().mean().item())
 assert torch.isfinite(predicted).all()
 plt.plot(history, color='#d73535'); plt.xlabel('step'); plt.ylabel('L1 + beta KL'); plt.show()
 '''),
  step('Шаг 3. В момент t=2 доступны a[0,2], a[1,1], a[2,0]. Даже нулевой action — валидное предсказание, а не признак отсутствия данных.',
       'Step 3. At t=2 combine a[0,2], a[1,1], a[2,0]. A zero action remains a valid prediction, not missing data.', '''
 chunks = torch.tensor([[[0., 0.], [1., 1.], [2., 2.], [3., 3.]],
                        [[10., 10.], [11., 11.], [12., 12.], [13., 13.]],
                        [[20., 20.], [21., 21.], [22., 22.], [23., 23.]]])
 now = 2; starts = torch.arange(3)
 candidates = chunks[starts, now - starts]
 ages = (now - starts).float(); weights = torch.exp(-0.5 * ages); weights /= weights.sum()
 ensemble = (candidates * weights[:, None]).sum(0)
 print('same-time candidates:', candidates.tolist(), 'weights:', weights.tolist(), 'ensemble:', ensemble.tolist())
 assert torch.equal(candidates[:, 0], torch.tensor([2., 11., 20.]))
 assert ensemble.shape == (A,) and torch.allclose(weights.sum(), torch.tensor(1.))
 ''')],
 ['KL контролирует posterior, но не гарантирует полезный latent; слишком большой beta может вызвать posterior collapse. Низкий reconstruction loss с target-conditioned posterior не равен качеству z=0 inference. Ensemble требует явных timestamps и validity masks.',
  'KL regularizes the posterior but does not guarantee useful latents; excessive beta can cause posterior collapse. Low target-conditioned reconstruction loss is not equivalent to good z=0 inference. Ensembling requires explicit timestamps and validity masks.'],
 ['1. Увеличьте beta в 100 раз и сравните KL/reconstruction отдельно.\n2. Подайте actions на inference: почему это leakage?\n3. Удалите старый chunk и пересчитайте ensemble. Не используйте action==0 как validity.',
  '1. Increase beta 100-fold and track KL/reconstruction separately.\n2. Supply actions at inference: why is that leakage?\n3. Remove an old chunk and recompute the ensemble. Do not use action==0 as validity.'],
 ['Большой beta прижимает posterior к prior и может лишить decoder информации latent. На deployment target actions неизвестны. Chunk validity определяется наличием предсказания и его временем, а не числовым значением action.',
  'Large beta pushes the posterior toward the prior and may remove latent information. Deployment has no target actions. Chunk validity depends on prediction availability and timestamps, not the numerical action value.'],
 ['Учебная state-only CVAE и same-time ensembling. Не официальный ACT visual Transformer, не LeRobot trainer и не проверка реального робота.',
  'Educational state-only CVAE and same-time ensembling; not the official ACT visual Transformer, LeRobot trainer, or a real-robot validation.'],
 ['Вводный VLA, action chunks, Gaussian latent и KL.', 'Introductory VLA, action chunks, Gaussian latents, and KL.'])


def build():
    for index, item in enumerate(PAPERS):
        for language_index, language in enumerate(('ru', 'en')):
            ru = language_index == 0
            def text(field):
                return item[field][language_index]
            cells = []
            def add(kind, source):
                cell = {'id': f"{item['slug']}-{len(cells)}", 'cell_type': kind, 'metadata': {}, 'source': source}
                if kind == 'code':
                    cell.update(outputs=[], execution_count=None)
                cells.append(cell)
            add('markdown', f"# {text('name')}\n\n**{item['title']}** · {item['authors']} · {item['year']}\n\n[{'Оригинальная статья' if ru else 'Original paper'}]({item['url']})\n\n" + ('**Цель урока:** ' if ru else '**Learning goal:** ') + text('goal'))
            add('markdown', ('## Перед началом\n\n' if ru else '## Before starting\n\n') + text('prerequisite') + '\n\n' + ('Около 40–60 минут с заданиями. CPU, PyTorch и Matplotlib; без скачивания весов и датасетов. Время — ориентир для чтения и работы, а не benchmark выполнения.' if ru else 'Allow roughly 40–60 minutes including exercises. CPU, PyTorch, and Matplotlib; no model or dataset downloads. This is a study-time estimate, not a runtime benchmark.'))
            add('markdown', ('## Задача и идея\n\n' if ru else '## Problem and idea\n\n') + text('context'))
            add('markdown', ('## Граница воспроизведения\n\n' if ru else '## Reproduction boundary\n\n') + text('scope') + '\n\n' + ('Код написан для этого урока по математическим идеям статьи. Это не скопированный авторский репозиторий. Успех assertions подтверждает указанные свойства реализации, но не результаты статьи.' if ru else 'The code is an original educational implementation of mathematical ideas from the paper, not a copy of an author repository. Passing assertions verifies the stated implementation properties, not the paper’s results.'))
            add('markdown', ('## Формула → тензоры\n\n' if ru else '## Formula → tensors\n\n') + text('math'))
            add('code', dedent('''
                %matplotlib inline
                import math
                import torch
                from torch import nn
                from torch.nn import functional as F
                import matplotlib.pyplot as plt
                torch.set_num_threads(1)
                torch.manual_seed(42)
                plt.rcParams.update({'figure.figsize': (7, 3), 'axes.spines.top': False, 'axes.spines.right': False})
                print('PyTorch:', torch.__version__, '| CPU | seed=42')
            ''').strip())
            for explanation, code in item['steps']:
                heading, body = explanation[language_index].split('. ', 1)
                add('markdown', '## ' + heading + '\n\n' + body)
                add('code', code)
            add('markdown', ('## Как читать результат\n\n' if ru else '## Reading the result\n\n') + text('interpretation'))
            add('markdown', ('## Самостоятельные эксперименты\n\n' if ru else '## Independent experiments\n\n') + text('exercises') + '\n\n' + ('Сначала запишите прогноз, затем меняйте по одному фактору. Для честного сравнения сбрасывайте ядро и выполняйте все ячейки с начала. Сохраните копию для своих изменений.' if ru else 'Write a prediction first, then change one factor at a time. Reset the kernel and rerun all cells for a controlled comparison. Save a copy for your changes.'))
            add('markdown', ('## Проверка рассуждений — после эксперимента\n\n' if ru else '## Reasoning check — after experimenting\n\n') + text('answers'))
            add('markdown', ('## Что записать в лабораторный журнал\n\n' if ru else '## What to record in your lab notes\n\n') + ('- Гипотеза и изменённый параметр.\n- Seed, размер данных, число шагов и используемая precision.\n- Значения проверок до и после, график и объяснение расхождения с прогнозом.\n- Что этот эксперимент **не** позволяет заключить.\n\nСвязанный раздел доступен по кнопке «Открыть раздел студии» над уроком.' if ru else '- Hypothesis and changed parameter.\n- Seed, data size, step budget, and precision.\n- Before/after diagnostics, plot, and explanation of unexpected results.\n- What this experiment **does not** establish.\n\nUse “Open Studio workspace” above the lesson to continue in the related tools.'))
            metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                        'language_info': {'name': 'python'}, 'studio': {'lesson': True, 'language': language,
                        'track': 'papers', 'title': text('name'), 'description': text('goal'), 'topic': item['topic'],
                        'minutes': 50, 'level': 'intermediate', 'section': item['section'],
                        'paper': {'title': item['title'], 'authors': item['authors'], 'year': item['year'], 'url': item['url']}}}
            path = ROOT / language / f"{item['slug']}.ipynb"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'nbformat': 4, 'nbformat_minor': 5, 'metadata': metadata, 'cells': cells}, ensure_ascii=False, indent=2) + '\n')
    print(f'Built {len(PAPERS) * 2} paper notebooks')


if __name__ == '__main__':
    build()
