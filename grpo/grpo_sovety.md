По структуре это **в целом рабочий GRPO/“PPO-without-critic” пайплайн**, и твои “Dr. GRPO” модификации **в основном совпадают** с тем, что описано у Wolfe (GRPO++). Но есть несколько **важных расхождений/дыр**, из-за которых на масштабе очень легко поймать нестабильность, нулевые градиенты и/или энтропийный коллапс.

Ниже — разбор **что соответствует**, **что не соответствует**, и **что бы я поправил в первую очередь** (с конкретными правками).

---

## 1) Что в коде соответствует vanilla GRPO

### ✅ Групповое “relative” преимущество

Ты считаешь advantage внутри группы с общим промптом: `A_i = r_i - mean(r)` (+ деление на std для обычного GRPO). Это соответствует определению GRPO. ([Cameron R. Wolfe][1])

### ✅ PPO-style ratio + clipping

`ratio = exp(log_probs - old_log_probs)` и `min(ratio*A, clip(ratio)*A)` — это стандартный GRPO surrogate с клиппингом. ([Cameron R. Wolfe][1])

### ✅ Маскирование токенов completion’а

Ты применяешь `action_mask` и фактически обучаешься только на токенах ответа — это правильно.

---

## 2) Твой “DrGRPO” — насколько совпадает с описанием

Wolfe пересказывает Dr. GRPO как **две модификации**:

1. **фиксированная нормализация** loss по константе `MAX_TOKENS`, а не по длине ответа
2. **убрать std** из advantage (только вычесть mean). ([Cameron R. Wolfe][1])

### ✅ Убрать std — у тебя сделано

`critic_type="drgrpo"` → только `returns - mean`, без `/ std`. Это совпадает. ([Cameron R. Wolfe][1])

### ✅ Фиксированная нормализация — у тебя почти сделано

Ты делаешь `masked_sum(..., constant_normalizer=generate_max_length).mean()`, что по смыслу совпадает с “делим на константу” (token-level вклад перестаёт зависеть от длины ответа). ([Cameron R. Wolfe][1])

### ⚠️ Но: у тебя `generate_max_length` сейчас **не тот “MAX_TOKENS”, который нужен**

Ты используешь `max_length=1024` в `generate`, а это **total length (prompt+completion)**, и он ещё и зависит от длины промпта. Для DrGRPO/DAPO-подобных фиксов важно, чтобы константа соответствовала **лимиту на число *генерируемых* токенов** (max_new_tokens), чтобы нормализация была “честной” и стабильной.

**Как лучше:**

* В генерации используй `max_new_tokens=MAX_NEW_TOKENS`
* В loss используй `constant_normalizer=MAX_NEW_TOKENS`

Пример (минимальная правка):

```python
MAX_NEW_TOKENS = 768  # пример

generation_config = GenerationConfig(
    do_sample=True,
    top_p=top_p,
    temperature=temperature,
    max_new_tokens=MAX_NEW_TOKENS,
    pad_token_id=pad_token_id,
)
...
grpo_loss_fn = GRPOLoss(..., critic_type="drgrpo", generate_max_length=MAX_NEW_TOKENS)
```

---

## 3) Ключевые несоответствия “GRPO++ best practices” (важно)

### (A) Нет **clip higher** (decoupled upper bound)

В DAPO/GRPO++ один из главных практических фиксов — поднять **верхнюю** границу клипа: `[1-ε_low, 1+ε_high]` с `ε_low≈0.2`, `ε_high≈0.28` (пример из статьи), чтобы уменьшить энтропийный коллапс. ([Cameron R. Wolfe][1])

У тебя сейчас симметрично: `clip_eps=0.2` → `[0.8, 1.2]`.

**Как поправить:**

```python
class GRPOLoss(nn.Module):
    def __init__(..., eps_low=0.2, eps_high=0.28, ...):
        ...
        self.eps_low = eps_low
        self.eps_high = eps_high

    def forward(...):
        ratio = (log_probs - old_log_probs).exp()
        ratio_clip = ratio.clamp(1 - self.eps_low, 1 + self.eps_high)
        surr1 = ratio * advantages
        surr2 = ratio_clip * advantages
        ...
```

### (B) Нет **dynamic sampling / zero-gradient filtering**

Статья прямо говорит: когда в группе **все ответы одинаково “идеальны”** (или вообще все одинаковый reward), advantage становится нулевым → фактически batch уменьшается → градиент шумнее → обучение хуже. Решение: **оверсэмплинг + фильтрация + добор**, пока batch не наполнится. ([Cameron R. Wolfe][1])

У тебя сейчас такие группы **всё равно попадают в replay_buffer** и съедают compute.

**Минимальный фикс (без полного active sampling):**
просто **пропускай** группы, где rewards константные:

```python
if returns.max() == returns.min():
    continue  # zero-gradient group
```

**Правильнее (как в GRPO++/DAPO):**

* берёшь batch побольше,
* фильтруешь zero-gradient,
* добираешь промпты пока не наберёшь нужное число “полезных” промптов. ([Cameron R. Wolfe][1])

### (C) “Token-level loss” в DAPO и “No KL” в reasoning-RL

GRPO++ отмечает, что в современных reasoning-RL пайплайнах **часто убирают KL penalty** вообще. ([Cameron R. Wolfe][1])
У тебя KL включён (`kl_weight=0.01`). Это не “неправильно”, но если твоя цель — R1-Zero-like reasoning RLVR, то это **расходится с типичной практикой** из статьи.

Также DAPO рекомендует token-level агрегацию (вес каждого токена одинаковый), чтобы убрать length bias. ([Cameron R. Wolfe][1])
Ты это частично делаешь через DrGRPO (фиксированная нормализация), но если ты запускаешь `critic_type="grpo"`, у тебя остаётся sample-level mean → length bias.

### (D) Нет “overlong reward shaping” (наказание/обработка truncation)

В DAPO подчёркивается, что **как именно** штрафовать “обрезанные” (truncated) ответы — важно; наивное наказание может ломать обучение. ([Cameron R. Wolfe][1])
У тебя вообще нет детекта “дошли до лимита и не закончили” → такие rollouts получают обычный reward.

Минимально: логируй флаг truncated и вводи аккуратный shaping.

---

## 4) Чисто инженерные/корректностные моменты (тоже важные)

### ⚠️ PAD == EOS ломает обучение “заканчивать ответ”

Ты ставишь `tokenizer.pad_token = eos_token`, а потом маскируешь `sequence_ids == pad_token_id` → ты **выкидываешь EOS-токены** из loss (и как EOS, и как pad). Это часто нежелательно: EOS — это *действие*, которому модель должна учиться.

Лучше:

* добавить отдельный PAD токен (и `resize_token_embeddings`), или
* маскировать “пэддинг после первого EOS”, а не все EOS.

### ⚠️ Оптимизатор на `model.parameters()` вместо только trainable

С LoRA надо так:

```python
optimizer = AdamW32bit((p for p in model.parameters() if p.requires_grad), lr=lr, is_paged=True)
```

Иначе лишний оверхед.

### ⚠️ Метрики “здоровья” RL

В статье прямо рекомендуют мониторить **длину ответа и энтропию** как ранние индикаторы проблем. ([Cameron R. Wolfe][1])
У тебя логируются returns/KL/grad_norm, но нет:

* avg completion length
* token entropy / next-token entropy (хотя бы на rollout’ах)

Это реально помогает поймать “entropy collapse” раньше, чем всё поедет.

---

## Итоговый вердикт

**Да**, это “похоже на GRPO” и **твоя DrGRPO-идея реализована по смыслу верно** (mean-only advantage + fixed normalizer). ([Cameron R. Wolfe][1])
Но чтобы соответствовать “GRPO++ / DAPO-уровню практики”, тебе критично не хватает как минимум:

1. **clip higher** (ε_high > ε_low) ([Cameron R. Wolfe][1])
2. **dynamic sampling / zero-gradient filtering + active sampling** ([Cameron R. Wolfe][1])
3. аккуратно определённого `MAX_TOKENS` через `max_new_tokens` (особенно для DrGRPO) ([Cameron R. Wolfe][1])
4. решения PAD/EOS маскирования
5. (по желанию, но часто нужно) убрать/сильно ослабить KL в reasoning-RLVR ([Cameron R. Wolfe][1])

Если хочешь — скидывай, как именно ты хочешь “режимы” в библиотеке (grpo / dapo-like / drgrpo), и я набросаю тебе аккуратный **единый интерфейс loss + sampler** (с правильными масками, clip higher, фильтрацией нулевого градиента и max_new_tokens), чтобы это легло в твою agents-fabric как модуль.

[1]: https://cameronrwolfe.substack.com/p/grpo-tricks "GRPO++: Tricks for Making RL Actually Work"
