"""Rebuild the bilingual, offline CPU curriculum shipped with Studio.

Run from the repository root. Generated lessons contain no saved execution state.
Every lesson has a goal, explanation, runnable experiment, exercise and sources.
"""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1] / 'models-at-home' / 'notebooks' / 'lessons'
LESSONS = []


def lesson(slug, topic, section, titles, descriptions, theory, code, exercises, sources):
    LESSONS.append(dict(slug=slug, topic=topic, section=section, titles=titles,
                        descriptions=descriptions, theory=theory, code=dedent(code).strip(),
                        exercises=exercises, sources=sources))


lesson('01-training', 'basics', '/',
 ['Как модель учится', 'How a model learns'],
 ['Loss, градиенты, шаг оптимизатора и проверка на новых данных.', 'Loss, gradients, optimizer steps, and held-out data.'],
 ['Модель — функция с обучаемыми параметрами. Loss измеряет ошибку; backward вычисляет производные, а оптимизатор меняет параметры. Здесь мы восстанавливаем линейную зависимость из шумных наблюдений.\n\nДелим данные до обучения. Validation не участвует в backward: она помогает заметить переобучение. Низкая ошибка на искусственных данных подтверждает работу примера, но не качество на вашей задаче.',
  'A model is a function with trainable parameters. Loss measures error, backward computes derivatives, and an optimizer updates parameters. We recover a linear relation from noisy observations.\n\nSplit the data before training. Validation never participates in backward; it helps detect overfitting. Low error on synthetic data verifies this exercise, not performance on your task.'],
 '''
 x = torch.linspace(-2, 2, 120).unsqueeze(1)
 y = 2.5 * x - 0.7 + 0.15 * torch.randn_like(x)
 order = torch.randperm(len(x)); train, val = order[:90], order[90:]
 model = nn.Linear(1, 1)
 optimizer = torch.optim.AdamW(model.parameters(), lr=0.08)
 history = []
 for step in range(100):
     optimizer.zero_grad(set_to_none=True)
     loss = F.mse_loss(model(x[train]), y[train])
     loss.backward(); optimizer.step()
     history.append(loss.item())
 with torch.no_grad():
     print('validation MSE:', F.mse_loss(model(x[val]), y[val]).item())
 print('weight, bias:', model.weight.item(), model.bias.item())
 plt.plot(history, color='#d73535'); plt.xlabel('step'); plt.ylabel('train loss'); plt.show()
 ''',
 ['Измените learning rate на 0.001 и 1.0. Сравните кривые и validation loss. Уменьшите train до 10 примеров: что меняется?',
  'Try learning rates of 0.001 and 1.0. Compare the curves and validation loss. Reduce training to 10 examples: what changes?'],
 [('PyTorch: optimization', 'https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html')])

lesson('02-data', 'data', '/',
 ['Данные, токены и маски', 'Data, tokens, and masks'],
 ['Как подготовить текст и не обучаться на padding.', 'Prepare text without learning from padding.'],
 ['Батч содержит последовательности разной длины. Padding заполняет пустые позиции, attention mask запрещает использовать их как контекст, а labels=-100 исключает их из cross entropy. Это разные механизмы.\n\nДля SFT важно сохранять границы ролей и отвечать на вопрос, какие токены участвуют в loss. Для изображений, действий и эпизодов дополнительно нужны согласованные размеры, единицы и временные границы. Не смешивайте кадры одного эпизода между train и validation.',
  'Batches contain sequences of different lengths. Padding fills missing positions, attention masks exclude them from context, and labels=-100 excludes them from cross entropy. These are different mechanisms.\n\nSFT requires role boundaries and a clear choice of supervised tokens. Images, actions, and episodes also need consistent shapes, units, and time boundaries. Keep an episode entirely within one data split.'],
 '''
 samples = ['model', 'learn']
 tokens = [list(s.encode('utf-8')) for s in samples]
 width = max(map(len, tokens)) + 3
 ids = torch.zeros(len(tokens), width, dtype=torch.long)
 mask = torch.zeros_like(ids)
 for row, sequence in enumerate(tokens):
     ids[row, :len(sequence)] = torch.tensor(sequence)
     mask[row, :len(sequence)] = 1
 labels = ids.clone(); labels[mask == 0] = -100
 logits = torch.randn(*ids.shape, 256)
 loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
 print('input_ids:', ids.tolist()); print('attention_mask:', mask.tolist())
 print('labels:', labels.tolist()); print('masked loss:', loss.item())
 assert (labels == -100).sum() == (mask == 0).sum()
 ''',
 ['Добавьте русский текст и посмотрите, почему число UTF-8 байтов отличается от числа символов. Это учебный byte tokenizer; настоящие модели используют свой tokenizer и chat template.',
  'Add non-ASCII text and compare UTF-8 bytes with character counts. This byte tokenizer is educational; real models require their own tokenizer and chat template.'],
 [('PyTorch: CrossEntropyLoss', 'https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html')])

lesson('03-sft', 'llm', '/',
 ['LLM: pretraining и SFT', 'LLM: pretraining and SFT'],
 ['Предсказание следующего токена и обучение только на ответе.', 'Next-token prediction and response-only supervision.'],
 ['Causal LM предсказывает следующий токен по предыдущим. В pretraining loss обычно считается на всех непустых токенах; в response-only SFT — на ответе ассистента. Сдвиг logits и labels должен быть сделан ровно один раз.\n\nНиже небольшая модель с causal attention, а не готовая языковая модель. Данные синтетические. В Training Studio используйте свой train/validation, chat template модели и сначала короткий проверочный запуск.',
  'A causal LM predicts each next token from preceding tokens. Pretraining usually supervises every non-padding token; response-only SFT supervises the assistant response. Shift logits and labels exactly once.\n\nThis is a tiny causal-attention model, not a pretrained language model. Its data is synthetic. In Training Studio, supply your own splits and model chat template, then start with a short check run.'],
 '''
 class TinyLM(nn.Module):
     def __init__(self):
         super().__init__()
         self.embedding = nn.Embedding(16, 24)
         self.position = nn.Embedding(8, 24)
         self.block = nn.TransformerEncoderLayer(24, 4, 48, dropout=0, batch_first=True)
         self.head = nn.Linear(24, 16)
     def forward(self, ids):
         length = ids.shape[1]
         h = self.embedding(ids) + self.position(torch.arange(length))
         causal = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
         return self.head(self.block(h, src_mask=causal))
 model = TinyLM(); optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
 batch = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]).repeat(8, 1)
 labels = batch[:, 1:].clone(); labels[:, :3] = -100
 losses = []
 for step in range(35):
     optimizer.zero_grad(set_to_none=True)
     loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1), labels.flatten())
     loss.backward(); optimizer.step(); losses.append(loss.item())
 print('supervised tokens per sample:', (labels[0] != -100).sum().item())
 plt.plot(losses, color='#d73535'); plt.ylabel('response-only CE'); plt.show()
 ''',
 ['Уберите маску prompt и сравните loss. Меняйте длину prompt, сохраняя корректное соответствие сдвинутым labels. Почему memorization одного примера не означает полезный SFT?',
  'Remove the prompt mask and compare loss. Change prompt length while preserving shifted labels. Why does memorizing one sequence not establish useful SFT?'],
 [('Attention Is All You Need', 'https://arxiv.org/abs/1706.03762')])

lesson('04-lora', 'llm', '/',
 ['LoRA и полное обучение', 'LoRA and full fine-tuning'],
 ['Что обучается в адаптере и почему экономия памяти не равна размеру весов.', 'What an adapter trains and why memory includes more than weights.'],
 ['LoRA добавляет к замороженной матрице W низкоранговое обновление B×A. Обычно A инициализируют случайно, B — нулями: стартовая функция остаётся прежней, а градиент B уже ненулевой. Масштаб alpha/r влияет на обновление.\n\nЗаморозка уменьшает память градиентов и оптимизатора, но базовые веса и активации остаются. QLoRA дополнительно квантует базу и требует совместимого backend. Этот CPU-пример показывает LoRA без квантизации и Unsloth.',
  'LoRA adds a low-rank update B×A to frozen weights W. A commonly starts random and B starts at zero, preserving the initial function while allowing a nonzero B gradient. The alpha/r scaling affects the update.\n\nFreezing reduces gradient and optimizer memory; base weights and activations remain. QLoRA also quantizes the base and needs a compatible backend. This CPU exercise demonstrates LoRA without quantization or Unsloth.'],
 '''
 base = nn.Linear(12, 8); base.requires_grad_(False)
 rank, alpha = 2, 4
 A = nn.Parameter(torch.randn(rank, 12) * 0.02)
 B = nn.Parameter(torch.zeros(8, rank))
 optimizer = torch.optim.AdamW([A, B], lr=0.03)
 x = torch.randn(32, 12); target = torch.randn(32, 8)
 frozen = base.weight.detach().clone()
 history = []
 for step in range(50):
     prediction = base(x) + (x @ A.T @ B.T) * alpha / rank
     loss = F.mse_loss(prediction, target)
     optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
     history.append(loss.item())
 assert torch.equal(base.weight, frozen)
 print('trainable:', A.numel() + B.numel(), 'base:', sum(p.numel() for p in base.parameters()))
 plt.plot(history, color='#d73535'); plt.ylabel('LoRA MSE'); plt.show()
 ''',
 ['Поставьте rank=1 и rank=8, повторив ячейку с начала. Сравните число параметров и итоговую ошибку. Что произойдёт, если и A, и B изначально нулевые?',
  'Try ranks 1 and 8, rerunning the whole cell. Compare parameter counts and final error. What happens if both A and B start at zero?'],
 [('LoRA', 'https://arxiv.org/abs/2106.09685'), ('QLoRA', 'https://arxiv.org/abs/2305.14314')])

lesson('05-grpo', 'rl', '/',
 ['GRPO: от reward к обновлению', 'GRPO: from reward to an update'],
 ['Групповые advantages, policy ratio и clipping.', 'Group-relative advantages, policy ratios, and clipping.'],
 ['GRPO сравнивает несколько ответов на один prompt. Reward нормируется внутри группы, чтобы получить advantage. Policy ratio сопоставляет новую политику с политикой, собравшей примеры; clipping ограничивает часть изменений.\n\nВ настоящем trainer нужны generation, корректные token masks, сохранённые old log-probs, reference/KL при её использовании и проверенный reward. Ниже только ядро clipped objective без KL и генерации. Это не замена нашему trainer и не запуск TRL. Нулевая дисперсия reward не должна создавать NaN.',
  'GRPO compares several responses to the same prompt. Rewards are normalized within each group to form advantages. The policy ratio compares the current policy with the sampling policy; clipping limits part of the update.\n\nA real trainer also needs generation, token masks, saved old log-probs, reference/KL when used, and a validated reward. This exercise isolates the clipped objective without KL or generation. It neither replaces Studio’s trainer nor runs TRL. Constant rewards must not produce NaNs.'],
 '''
 rewards = torch.tensor([[0., 1., 2., 3.], [1., 1., 1., 1.]])
 advantages = (rewards - rewards.mean(-1, keepdim=True)) / rewards.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
 logits = nn.Parameter(torch.zeros(2, 4))
 old_logp = logits.detach().log_softmax(-1).clone()
 optimizer = torch.optim.Adam([logits], lr=0.08)
 for step in range(8):
     logp = logits.log_softmax(-1)
     ratio = (logp - old_logp).exp()
     objective = torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)
     loss = -objective.mean()
     optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
 print('advantages:', advantages)
 print('new probabilities:', logits.softmax(-1).detach())
 assert torch.isfinite(loss) and torch.equal(advantages[1], torch.zeros(4))
 ''',
 ['Измените reward, включая полностью одинаковые группы. Какие ответы получают большую вероятность? Объясните, почему old_logp нельзя пересчитывать после каждого optimizer.step.',
  'Change the rewards, including constant groups. Which answers gain probability? Explain why old_logp must not be recomputed after each optimizer step.'],
 [('DeepSeekMath / GRPO', 'https://arxiv.org/abs/2402.03300')])

lesson('06-preferences', 'rl', '/',
 ['Предпочтения и дистилляция', 'Preferences and distillation'],
 ['Разница между парными предпочтениями, наградой и ответами учителя.', 'Pairwise preferences, rewards, and teacher distributions.'],
 ['SFT учится по правильным ответам. DPO использует пары chosen/rejected и отношение policy к фиксированной reference. Distillation переносит распределение учителя через KL. Это разные источники сигнала, а не взаимозаменяемые названия.\n\nВ примере считаем DPO на sequence log-probabilities и distillation на logits. Для текста нужны суммы log-probs только по нужным токенам, одинаковый tokenizer для прямого logit distillation и независимая оценка качества. Доступность конкретного режима в студии зависит от выбранного семейства модели.',
  'SFT uses target answers. DPO uses chosen/rejected pairs and the policy-to-reference ratio. Distillation transfers a teacher distribution through KL. These are different learning signals.\n\nWe compute DPO from sequence log-probabilities and distillation from logits. Text training needs log-prob sums over the supervised tokens, matching tokenizers for direct logit distillation, and separate evaluation. Available Studio modes depend on the model family.'],
 '''
 chosen = torch.tensor([-2.0, -1.4], requires_grad=True)
 rejected = torch.tensor([-3.0, -2.0], requires_grad=True)
 ref_chosen = torch.tensor([-2.5, -1.7]); ref_rejected = torch.tensor([-2.8, -2.1])
 beta = 0.1
 dpo_loss = -F.logsigmoid(beta * ((chosen - rejected) - (ref_chosen - ref_rejected))).mean()
 dpo_loss.backward()
 print('DPO:', dpo_loss.item(), 'chosen gradient:', chosen.grad)
 teacher = torch.tensor([[3., 1., -1.]])
 student = torch.tensor([[0., 0., 0.]], requires_grad=True)
 temperature = 2.
 kd = F.kl_div((student / temperature).log_softmax(-1), (teacher / temperature).softmax(-1), reduction='batchmean') * temperature**2
 kd.backward(); print('distillation KL:', kd.item(), 'student gradient:', student.grad)
 ''',
 ['Поменяйте chosen и rejected местами. Проверьте знак градиента. Затем меняйте temperature: какие различия в распределении учителя становятся заметнее?',
  'Swap chosen and rejected and inspect the gradient sign. Change temperature: which differences in the teacher distribution become more visible?'],
 [('DPO', 'https://arxiv.org/abs/2305.18290'), ('Distillation', 'https://arxiv.org/abs/1503.02531')])

lesson('07-exploration', 'rl', '/vla',
 ['Исследование среды и NGU', 'Exploration and NGU'],
 ['Эпизодическая новизна, intrinsic reward и ограничения упрощения.', 'Episodic novelty, intrinsic reward, and what the simplification omits.'],
 ['Never Give Up (2020) — алгоритм исследования среды, а не новая версия GRPO. В оригинале сочетаются episodic novelty, lifelong novelty, семейство политик с разным весом intrinsic reward и off-policy recurrent обучение.\n\nЗдесь воспроизводим только интуицию эпизодического бонуса через расстояния до памяти состояний. Это не полный NGU и не проверка полного алгоритма. Intrinsic reward не равен успешности задачи: агент может предпочесть новизну полезным действиям. Память эпизода нужно сбрасывать на reset.',
  'Never Give Up (2020) is an exploration algorithm, not a new GRPO variant. The original combines episodic novelty, lifelong novelty, policies with different intrinsic-reward weights, and off-policy recurrent learning.\n\nHere we illustrate only an episodic bonus using distances to remembered states. This is neither full NGU nor its validation. Intrinsic reward is not task success: novelty can distract an agent. Reset episodic memory at episode boundaries.'],
 '''
 states = torch.tensor([[0., 0.], [1., 0.], [0., 0.], [2., 0.], [1., 0.], [5., 0.]])
 memory, bonuses = [], []
 for state in states:
     if not memory:
         bonus = 1.0
     else:
         distances = torch.cdist(state[None], torch.stack(memory)).square()[0]
         nearest = distances.topk(min(3, len(memory)), largest=False).values
         similarity = (0.1 / (nearest + 0.1)).sum()
         bonus = float(1 / torch.sqrt(1 + similarity))
     bonuses.append(bonus); memory.append(state)
 task_reward = torch.tensor([0., 0., 0., 0., 0., 1.])
 mixed = task_reward + 0.2 * torch.tensor(bonuses)
 print('novelty:', bonuses); print('mixed reward:', mixed.tolist())
 plt.plot(bonuses, color='#d73535', marker='o'); plt.ylabel('illustrative novelty bonus'); plt.show()
 ''',
 ['Повторите одно состояние 20 раз. Затем добавьте далёкое состояние. Почему бонус должен зависеть от обучаемого представления, а не сырых пикселей?',
  'Repeat one state 20 times, then add a distant one. Why should novelty use a learned representation instead of raw pixels?'],
 [('Never Give Up', 'https://arxiv.org/abs/2002.06038')])

lesson('08-vlm', 'vlm', '/vlm',
 ['VLM: изображение и текст', 'VLM: images and text'],
 ['Патчи изображения, проектор и проверка зависимости от картинки.', 'Image patches, a projector, and image-dependence checks.'],
 ['VLM связывает визуальное представление с текстовым. В простом варианте изображение разбивается на патчи, кодируется и проецируется в пространство признаков. Реальные готовые VLM требуют родной processor, формат диалога и правила image tokens.\n\nНиже маленькая fusion-модель на искусственной задаче: ответ зависит от яркости картинки и текстового признака. Это классификатор для понимания механизма, а не генеративная VLM. В VLM Studio можно обучать поддерживаемые готовые модели или собственную архитектуру из Research Lab; проверяйте доступность SFT/GRPO и backend до запуска.',
  'A VLM connects visual representations to text. A simple approach encodes image patches and projects them into a feature space. Real pretrained VLMs require their native processor, conversation format, and image-token conventions.\n\nThis tiny fusion classifier solves a synthetic task using image brightness and a text feature; it is not a generative VLM. VLM Studio supports its listed pretrained models and Research Lab architectures. Check SFT/GRPO and backend compatibility before training.'],
 '''
 images = torch.rand(64, 3, 8, 8)
 instruction = torch.randint(0, 2, (64,))
 # Shift brightness strongly enough that the task is visible in a tiny example.
 images[:32] *= 0.2; images[32:] = 0.8 + images[32:] * 0.2
 labels = ((images.mean((1, 2, 3)) > 0.5).long() ^ instruction)
 class Fusion(nn.Module):
     def __init__(self):
         super().__init__()
         self.patch = nn.Conv2d(3, 8, kernel_size=4, stride=4)
         self.text = nn.Embedding(2, 4)
         self.head = nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 2))
     def forward(self, image, text):
         visual = self.patch(image).mean((2, 3))
         return self.head(torch.cat([visual, self.text(text)], -1))
 model = Fusion(); optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
 for step in range(80):
     loss = F.cross_entropy(model(images, instruction), labels)
     optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
 with torch.no_grad():
     accuracy = (model(images, instruction).argmax(-1) == labels).float().mean()
     image_effect = (model(images, instruction) - model(torch.zeros_like(images), instruction)).abs().mean()
 print('training accuracy:', accuracy.item(), 'image ablation delta:', image_effect.item())
 ''',
 ['Обнулите изображение до обучения. Сравните accuracy и image ablation delta. Затем сделайте отдельную validation выборку: обучение и оценка на одном наборе скрывают переобучение.',
  'Zero the image before training and compare accuracy and ablation delta. Create a held-out set: evaluating training data hides overfitting.'],
 [('ViT', 'https://arxiv.org/abs/2010.11929')])

lesson('09-vla', 'vla', '/vla',
 ['VLA: от наблюдения к действию', 'VLA: from observations to actions'],
 ['Action chunks, маски, нормализация и семейства готовых политик.', 'Action chunks, masks, normalization, and supported policy families.'],
 ['VLA получает наблюдение, инструкцию и, если архитектура поддерживает, состояние. Выход может быть токенами действий, вектором или chunk нескольких будущих действий. Loss зависит от политики: OpenVLA — action-token CE; ACT — reconstruction + KL; SmolVLA и π0 — flow matching.\n\nЭти готовые политики в студии обучаются по демонстрациям, а не через универсальный GRPO/Unsloth. Поддержка RL собственных VLA — отдельный путь. Ниже только непрерывная masked imitation задача; это не ACT и не π0. Терминальный padding нельзя выдавать за реальные действия. Для робота дополнительно обязательны проверка единиц, calibration и ограничения действий.',
  'A VLA consumes observations, instructions, and state where supported. Outputs may be action tokens, a vector, or a chunk of future actions. Objectives depend on the policy: OpenVLA uses action-token CE, ACT reconstruction plus KL, and SmolVLA/π0 flow matching.\n\nThese standalone policies use demonstrations in Studio, not universal GRPO/Unsloth. RL for Studio’s native VLA is a separate route. This exercise is only masked continuous imitation, not ACT or π0. Terminal padding is not real action data. Hardware use additionally requires verified units, calibration, and action limits.'],
 '''
 batch, horizon, action_dim = 24, 4, 3
 state = torch.randn(batch, 6)
 target = state[:, :3, None].transpose(1, 2).repeat(1, horizon, 1) * 0.2
 target += torch.arange(horizon)[None, :, None] * 0.05
 valid = torch.ones(batch, horizon, dtype=torch.bool); valid[:8, -2:] = False
 mean = target[valid].mean(0); std = target[valid].std(0).clamp_min(1e-6)
 normalized = (target - mean) / std
 policy = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, horizon * action_dim))
 optimizer = torch.optim.AdamW(policy.parameters(), lr=0.02)
 for step in range(60):
     prediction = policy(state).reshape(batch, horizon, action_dim)
     loss = ((prediction - normalized).square() * valid[..., None]).sum() / (valid.sum() * action_dim)
     optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
 action = prediction.detach() * std + mean
 print('valid steps:', valid.sum().item(), 'loss:', loss.item())
 print('first chunk in original units:', action[0])
 ''',
 ['Замените horizon на 1 и 8. Уберите маску: как изменится смысл loss? Статистики нормализации в настоящей задаче считают только на train и сохраняют вместе с политикой.',
  'Try horizons 1 and 8. Remove the mask: how does the objective change? In real training, compute normalization on training data only and export it with the policy.'],
 [('OpenVLA', 'https://arxiv.org/abs/2406.09246'), ('ACT', 'https://arxiv.org/abs/2304.13705'), ('π0', 'https://arxiv.org/abs/2410.24164')])

lesson('10-world', 'world-models', '/world-models',
 ['World models: предсказывать переходы', 'World models: predicting transitions'],
 ['Action-conditioned динамика, rollout и накопление ошибки.', 'Action-conditioned dynamics, rollouts, and compounding error.'],
 ['World model учится предсказывать следующее наблюдение или скрытое состояние с учётом действия. При обучении она часто видит настоящие предыдущие состояния; при rollout получает собственные предсказания, поэтому ошибка накапливается.\n\nНиже контролируемая линейная среда и небольшая нейросеть. Это учебная динамика, не DIAMOND или Genie. В World Models Studio собственные модели имеют путь обучения; у foundation-моделей смотрите отдельные флаги inference/training. Доступные веса не означают наличие fine-tuning. Не делите переходы одного эпизода случайно между train/validation.',
  'A world model predicts the next observation or latent state conditioned on actions. Training often uses real previous states; rollouts feed predictions back into the model, allowing errors to accumulate.\n\nThis is a controlled linear environment and a tiny neural predictor, not DIAMOND or Genie. Studio’s native world models have training paths; foundation models expose separate inference/training support. Available weights do not imply fine-tuning support. Keep episodes separate across train and validation.'],
 '''
 states = torch.randn(128, 2); actions = torch.randn(128, 1)
 next_states = 0.9 * states + actions * torch.tensor([[0.2, -0.1]])
 model = nn.Sequential(nn.Linear(3, 24), nn.Tanh(), nn.Linear(24, 2))
 optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
 for step in range(100):
     prediction = model(torch.cat([states, actions], -1))
     loss = F.mse_loss(prediction, next_states)
     optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
 real = torch.tensor([[1., -1.]]); imagined = real.clone(); errors = []
 with torch.no_grad():
     for step in range(20):
         action = torch.tensor([[0.3]])
         real = 0.9 * real + action * torch.tensor([[0.2, -0.1]])
         imagined = model(torch.cat([imagined, action], -1))
         errors.append(F.mse_loss(imagined, real).item())
 print('one-step train MSE:', loss.item())
 plt.plot(errors, color='#d73535'); plt.xlabel('rollout step'); plt.ylabel('rollout MSE'); plt.show()
 ''',
 ['Увеличьте rollout до 100 шагов и задайте действия вне диапазона train. Низкий one-step loss гарантирует ли устойчивую симуляцию?',
  'Extend the rollout to 100 steps and use actions outside the training range. Does low one-step loss guarantee stable simulation?'],
 [('DIAMOND', 'https://arxiv.org/abs/2405.12399'), ('DreamerV3', 'https://arxiv.org/abs/2301.04104')])

lesson('11-architecture', 'builder', '/builder',
 ['Архитектура как набор блоков', 'Architectures as blocks'],
 ['Формы тензоров, residual, normalization и проверка градиентов.', 'Tensor shapes, residuals, normalization, and gradient checks.'],
 ['Блок задаёт преобразование тензора. Residual складывает вход и результат блока: формы должны совпадать. LayerNorm нормализует признаки; MLP меняет представление. Внимание добавляет обмен информацией между позициями и требует правильной causal mask.\n\nВ Model Builder можно собрать граф, открыть справку по блокам, проверить формы и передать поддерживаемую архитектуру в обучение LLM/VLM/VLA. Не каждый произвольный граф подходит всем задачам: важны modality, causal свойства и выходная размерность. Ниже проверяем простой residual MLP, не заменяя валидатор конструктора.',
  'A block transforms a tensor. A residual adds the input to the block output, so shapes must match. LayerNorm normalizes features; an MLP transforms representations. Attention lets positions exchange information and requires the correct causal mask.\n\nModel Builder supports graph editing, block references, shape analysis, and training handoffs for supported LLM/VLM/VLA architectures. Arbitrary graphs are not universally compatible: modality, causality, and output shapes matter. This exercise probes a residual MLP, not the complete Builder validator.'],
 '''
 class ResidualMLP(nn.Module):
     def __init__(self, width):
         super().__init__()
         self.norm = nn.LayerNorm(width)
         self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
     def forward(self, x):
         return x + self.mlp(self.norm(x))
 block = ResidualMLP(16)
 x = torch.randn(2, 8, 16, requires_grad=True)
 y = block(x); y.square().mean().backward()
 print('input:', tuple(x.shape), 'output:', tuple(y.shape))
 print('parameters:', sum(p.numel() for p in block.parameters()))
 print('input gradient norm:', x.grad.norm().item())
 assert x.shape == y.shape and torch.isfinite(x.grad).all()
 ''',
 ['Поменяйте последнюю Linear на выход ширины 8 и прочитайте ошибку residual. Затем восстановите форму и сравните gradient norm с LayerNorm и без неё.',
  'Change the final Linear to output width 8 and inspect the residual error. Restore the shape and compare gradient norms with and without LayerNorm.'],
 [('Residual Learning', 'https://arxiv.org/abs/1512.03385'), ('Layer Normalization', 'https://arxiv.org/abs/1607.06450')])

lesson('12-loops', 'builder', '/builder',
 ['Looped transformers: общие веса', 'Looped transformers: shared weights'],
 ['Повторное применение ядра, глубина вычислений и backward через циклы.', 'Reusing a core, computation depth, and backpropagation through loops.'],
 ['Looped transformer применяет одно и то же ядро несколько раз. Число уникальных параметров не растёт с количеством проходов, но вычисления и память активаций обычно растут. Списки из независимых копий блока не дают sharing.\n\nНиже общее causal-attention ядро. Full backprop проходит через все итерации. Detach между проходами обрезает градиент и меняет обучение; checkpointing пересчитывает активации и при корректной реализации сохраняет градиент. Изменение loop depth на inference — эксперимент, а не гарантированное улучшение. В Builder эти параметры задаются у looped блока.',
  'A looped transformer applies the same core repeatedly. Unique parameter count stays fixed while computation and activation memory generally grow. A list of independent block copies is not weight sharing.\n\nThis exercise reuses a causal-attention core. Full backprop traverses every iteration. Detaching between iterations truncates gradients and changes training; correctly applied checkpointing recomputes activations while preserving gradients. Increasing inference depth is an experiment, not a guaranteed improvement. Configure these choices on the Builder looped block.'],
 '''
 class Looped(nn.Module):
     def __init__(self):
         super().__init__()
         self.core = nn.TransformerEncoderLayer(16, 4, 32, dropout=0, batch_first=True)
     def forward(self, h, loops):
         mask = torch.triu(torch.ones(h.shape[1], h.shape[1], dtype=torch.bool), diagonal=1)
         for _ in range(loops):
             h = self.core(h, src_mask=mask)
         return h
 model = Looped(); x = torch.randn(2, 6, 16)
 parameters = sum(p.numel() for p in model.parameters())
 for depth in [1, 2, 4]:
     model.zero_grad(set_to_none=True)
     output = model(x, depth)
     # LayerNorm makes output.square().mean() nearly constant; use a nonuniform target.
     loss = F.mse_loss(output, x * 0.3)
     loss.backward()
     print('loops:', depth, 'unique parameters:', parameters, 'core gradient:', model.core.linear1.weight.grad.norm().item())
 ''',
 ['Сравните с ModuleList из четырёх независимых ядер. Проверьте число параметров и id(weight). Почему равные начальные значения весов ещё не означают общие параметры?',
  'Compare with a ModuleList of four independent cores. Check parameter counts and id(weight). Why do equal initial values not establish weight sharing?'],
 [('Universal Transformers', 'https://arxiv.org/abs/1807.03819'), ('ALBERT', 'https://arxiv.org/abs/1909.11942')])

lesson('13-moe', 'builder', '/builder',
 ['MoE: маршрутизация между экспертами', 'MoE: routing between experts'],
 ['Top-k routing, дисбаланс и отличие общего размера от активного.', 'Top-k routing, imbalance, and total versus active parameters.'],
 ['Mixture of Experts выбирает часть экспертов для каждого токена. Общий размер включает всех экспертов, а вычисления зависят от числа активных. Реальная экономия зависит от реализации dispatcher, batch, коммуникаций и железа.\n\nНиже наглядный top-1 router. Дискретный argmax не даёт градиента через выбор эксперта: production MoE использует подходящую схему взвешивания и балансировки. Пример нужен для изучения маршрутов и не выдаёт себя за полноценный MoE trainer.',
  'Mixture of Experts selects a subset of experts for each token. Total size includes every expert; computation depends on active experts. Actual efficiency depends on dispatching, batch size, communication, and hardware.\n\nThis is an illustrative top-1 router. Discrete argmax does not propagate a gradient through expert selection; production MoE needs suitable weighting and balancing. This exercise studies routes, not a complete MoE trainer.'],
 '''
 tokens = torch.randn(32, 8)
 router = nn.Linear(8, 4)
 experts = nn.ModuleList([nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)) for _ in range(4)])
 probabilities = router(tokens).softmax(-1)
 routes = probabilities.argmax(-1)
 output = torch.zeros_like(tokens)
 for index, expert in enumerate(experts):
     selected = routes == index
     if selected.any():
         output[selected] = expert(tokens[selected])
 counts = torch.bincount(routes, minlength=4)
 print('tokens per expert:', counts.tolist())
 print('all expert parameters:', sum(p.numel() for p in experts.parameters()))
 print('output shape:', output.shape)
 plt.bar(range(4), counts.tolist(), color='#d73535'); plt.xlabel('expert'); plt.show()
 ''',
 ['Сместите bias router к одному эксперту и посмотрите на распределение. Почему почти неиспользуемые эксперты ухудшают обучение? Какие величины нужно логировать кроме общего loss?',
  'Bias the router toward one expert and inspect usage. Why do underused experts hurt learning? What should be logged besides total loss?'],
 [('Switch Transformers', 'https://arxiv.org/abs/2101.03961')])

lesson('14-research', 'research', '/builder',
 ['Research Lab: измерять, а не угадывать', 'Research Lab: measure instead of guessing'],
 ['Activation и gradient norms, causal leakage и сравнение изменений.', 'Activation/gradient norms, causal leakage, and controlled comparisons.'],
 ['Падающий train loss не доказывает корректность модели. Проверяйте конечность градиентов, зависимость от входа, отсутствие доступа к будущему и качество на validation. Сравнивайте изменения при одинаковых данных, seed и числе шагов.\n\nНиже тест причинности: меняем будущие токены и проверяем, изменился ли ранний output. Для causal модели изменение должно быть близко к нулю. Research Lab выполняет дополнительные probes и анализ архитектуры; маленький CPU probe не заменяет полноценный benchmark.',
  'Falling training loss does not establish correctness. Check finite gradients, input dependence, future-token leakage, and held-out performance. Compare changes using the same data, seed, and step budget.\n\nThe test below modifies future tokens and checks an earlier output. A causal model should keep that output unchanged within numerical tolerance. Research Lab provides additional probes and architecture analysis; a tiny CPU probe is not a full benchmark.'],
 '''
 block = nn.TransformerEncoderLayer(16, 4, 32, dropout=0, batch_first=True).eval()
 original = torch.randn(2, 6, 16)
 modified = original.clone(); modified[:, 3:] += torch.randn_like(modified[:, 3:]) * 5
 causal = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
 with torch.no_grad():
     masked_delta = (block(original, src_mask=causal)[:, :3] - block(modified, src_mask=causal)[:, :3]).abs().max()
     unmasked_delta = (block(original)[:, :3] - block(modified)[:, :3]).abs().max()
 print('causal future-token delta:', masked_delta.item())
 print('unmasked future-token delta:', unmasked_delta.item())
 assert masked_delta < 1e-5
 block.train(); loss = block(original, src_mask=causal).mul(original).mean(); loss.backward()
 for name, parameter in block.named_parameters():
     if parameter.grad is not None:
         print(name, 'gradient norm:', parameter.grad.norm().item())
 ''',
 ['Уберите causal mask и убедитесь, что тест на masked_delta больше не проходит. Добавьте dropout: почему для сравнения выходов нужен eval или контролируемый RNG?',
  'Remove the causal mask and verify the assertion fails. Add dropout: why do output comparisons require eval mode or controlled RNG?'],
 [('PyTorch profiler', 'https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html')])

lesson('15-resources', 'resources', '/',
 ['Память и время обучения', 'Training memory and runtime'],
 ['Бюджет памяти, активации и оценка по измеренному шагу.', 'Memory budgets, activations, and estimates from measured steps.'],
 ['Память обучения включает веса, gradients, optimizer states, активации и временные buffers. FP32 AdamW часто требует примерно 16 байт на обучаемый параметр только для весов/градиентов/моментов; mixed precision и fused optimizers меняют эту сумму. Attention и длина контекста могут определять пик.\n\nНиже считаем нижнюю оценку для FP32 и измеряем CPU шаг. Это не прогноз для GPU и не обещание, что модель влезет. Для реального запуска используйте preflight и короткий benchmark на выбранном железе, с теми же batch, resolution, precision и optimizer. Начальные шаги включают прогрев, загрузку данных и compilation.',
  'Training memory includes weights, gradients, optimizer states, activations, and temporary buffers. FP32 AdamW commonly needs about 16 bytes per trainable parameter for weights, gradients, and moments alone; mixed precision and fused optimizers change this total. Attention and context length can dominate peak memory.\n\nWe compute a lower-bound FP32 estimate and time CPU steps. This does not predict GPU throughput or guarantee fit. Use preflight and a short benchmark on the target hardware with matching batch, resolution, precision, and optimizer. Startup includes warmup, data loading, and compilation.'],
 '''
 import time, statistics
 model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 16))
 optimizer = torch.optim.AdamW(model.parameters())
 x = torch.randn(16, 64); target = torch.randn(16, 16)
 count = sum(p.numel() for p in model.parameters())
 print('FP32 weights + gradients + Adam moments, MiB:', count * 16 / 2**20)
 durations = []
 for step in range(15):
     begin = time.perf_counter()
     optimizer.zero_grad(set_to_none=True); loss = F.mse_loss(model(x), target)
     loss.backward(); optimizer.step()
     if step >= 5:
         durations.append(time.perf_counter() - begin)
 median = statistics.median(durations)
 print('measured CPU step, ms:', median * 1000)
 print('1000 steps, compute only, seconds:', median * 1000)
 ''',
 ['Увеличьте batch в 4 раза и измерьте заново. Почему время не обязано вырасти ровно в 4 раза? Добавьте к бюджету памяти запас на активации и загрузку checkpoint.',
  'Increase batch size fourfold and remeasure. Why might runtime scale differently? Add activation and checkpoint-loading headroom to the memory budget.'],
 [('PyTorch CUDA memory', 'https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management')])

lesson('16-inference', 'inference', '/agent',
 ['Inference, checkpoint и воспроизводимость', 'Inference, checkpoints, and reproducibility'],
 ['Сохранение весов, eval, no_grad и проверка загрузки.', 'Saving weights, eval, no_grad, and reload checks.'],
 ['Веса без конфигурации, tokenizer/processor и нормализации могут быть недостаточны для восстановления модели. Для продолжения обучения нужны также optimizer/scheduler и RNG state. eval меняет dropout и некоторые нормализации; no_grad отдельно отключает запись графа.\n\nНиже сохраняем безопасный state_dict в памяти, загружаем в такую же архитектуру и сравниваем выходы. Это проверяет round-trip весов, а не полный resume training. Agent Studio использует инструменты и модели для inference; обучение происходит в профильных разделах.',
  'Weights may be insufficient without configuration, tokenizer/processor, and normalization. Resuming training also needs optimizer/scheduler and RNG state. eval changes dropout and some normalization behavior; no_grad independently disables graph recording.\n\nWe serialize a state_dict in memory, reload it into the same architecture, and compare outputs. This verifies a weight round trip, not full training resume. Agent Studio uses tools and models for inference; training belongs in the respective workspaces.'],
 '''
 import io
 model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 2)).eval()
 x = torch.randn(3, 4)
 with torch.no_grad():
     before = model(x)
 buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
 restored = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 2))
 restored.load_state_dict(torch.load(buffer, weights_only=True)); restored.eval()
 with torch.no_grad():
     after = restored(x)
 print('max reload error:', (before - after).abs().max().item())
 assert torch.equal(before, after)
 restored.train()
 with torch.no_grad():
     print('train-mode dropout delta:', (restored(x) - restored(x)).abs().max().item())
 ''',
 ['Составьте список артефактов для экспорта VLA: архитектура, веса, нормализация состояния/действий, порядок координат, image processor и horizon. Что случится, если перепутать порядок действий?',
  'List the artifacts needed for VLA export: architecture, weights, state/action normalization, coordinate order, image processor, and horizon. What happens if action order is wrong?'],
 [('PyTorch: saving and loading', 'https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html')])


def build():
    for item in LESSONS:
        for idx, language in enumerate(['ru', 'en']):
            intro = ('Все примеры этого урока работают на CPU, не скачивают модели и используют искусственные данные. Они объясняют механизм; качество настоящего обучения проверяется отдельно.' if idx == 0 else
                     'All examples run on CPU without model downloads and use synthetic data. They explain mechanisms; real training quality must be evaluated separately.')
            setup = dedent('''
                %matplotlib inline
                import torch
                from torch import nn
                from torch.nn import functional as F
                import matplotlib.pyplot as plt
                torch.set_num_threads(1)
                torch.manual_seed(42)
                plt.rcParams.update({'figure.figsize': (7, 3), 'axes.spines.top': False, 'axes.spines.right': False})
                print('PyTorch:', torch.__version__, '| device: CPU')
            ''').strip()
            sources = '\n'.join(f'- [{name}]({url})' for name, url in item['sources'])
            texts = [
                ('markdown', f"# {item['titles'][idx]}\n\n{item['descriptions'][idx]}\n\n> {intro}"),
                ('markdown', f"## {'Что важно понять' if idx == 0 else 'What to understand'}\n\n{item['theory'][idx]}"),
                ('code', setup),
                ('markdown', f"## {'Запустите эксперимент' if idx == 0 else 'Run the experiment'}\n\n" + ('Запускайте ячейки по порядку. Код можно менять; Shift + Enter выполняет текущую ячейку.' if idx == 0 else 'Run cells in order. Edit the code; Shift + Enter runs the current cell.')),
                ('code', item['code']),
                ('markdown', f"## {'Попробуйте сами' if idx == 0 else 'Try it yourself'}\n\n{item['exercises'][idx]}\n\n" + ('Сохраните копию ноутбука и запишите, что изменили, чего ожидали и что измерили.' if idx == 0 else 'Save a notebook copy and record what you changed, expected, and measured.')),
                ('markdown', f"## {'Продолжить в студии' if idx == 0 else 'Continue in Studio'}\n\n" + ('Кнопка «Открыть раздел студии» над уроком ведёт к связанным инструментам. Учебный код сам не запускает большие тренировки. Перед реальным запуском проверьте данные, ресурсы и поддержку метода у выбранной модели.' if idx == 0 else 'The workspace link above the lesson opens related tools. This notebook does not launch large training jobs. Before real training, check data, resources, and method support for the chosen model.') + f"\n\n## {'Источники' if idx == 0 else 'Sources'}\n\n{sources}"),
            ]
            cells = []
            for number, (kind, source) in enumerate(texts):
                cell = {'id': f"{item['slug']}-{number}", 'cell_type': kind, 'source': source, 'metadata': {}}
                if kind == 'code':
                    cell.update(outputs=[], execution_count=None)
                cells.append(cell)
            notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
                'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python'}, 'studio': {
                    'lesson': True, 'language': language, 'title': item['titles'][idx],
                    'description': item['descriptions'][idx], 'topic': item['topic'],
                    'minutes': 15, 'level': 'introductory', 'section': item['section']}}, 'cells': cells}
            path = ROOT / language / f"{item['slug']}.ipynb"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    print(f'Built {len(LESSONS) * 2} notebooks')


if __name__ == '__main__':
    build()
