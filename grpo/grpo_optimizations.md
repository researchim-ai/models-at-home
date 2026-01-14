Ниже — практическая «как сделать правильно» схема имплементации **GRPO (Group Relative Policy Optimization)** так, чтобы она **максимально эффективно** работала и с **DeepSpeed ZeRO-3** (без/с CPU offload), и с **PyTorch FSDP**. Я опираюсь на первоисточники (DeepSeekMath / DeepSeek-R1) и на промышленные реализации (TRL/OpenRLHF/veRL), потому что именно там уже “вшиты” многие performance-грабли. ([arXiv][1])

---

## 0) Что именно оптимизирует GRPO (формула, которую надо реализовать)

GRPO = PPO-подобная оптимизация, но **без критика/value-модели**: baseline оценивается **по группе сэмплов** для одного и того же prompt.

Для каждого вопроса `q` вы сэмплируете группу `G` ответов `{o_i}` из “старой” политики `π_{θ_old}` и оптимизируете `π_θ` по clipped-objective + KL к референсу: ([arXiv][1])

* **Advantage (групповой):**
  [
  A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}
  ]
  ([arXiv][1])

* **Objective (в DeepSeek-R1):** clipped PPO-ratio + KL penalty к `π_ref`. ([arXiv][1])
  TRL в документации также расписывает 4 шага: generation → advantage → KL estimate (аппроксимация Schulman 2020) → loss. ([Hugging Face][2])

---

## 1) Архитектура пайплайна “как в больших системах” (самое важное для эффективности)

### Почему “наивно” делать generate() прямо внутри ZeRO-3/FSDP — плохо

* При ZeRO-3/FSDP параметры шардингованы; autoregressive generation приводит к частым **all-gather** / перемещениям параметров и резко падает throughput.
* В FSDP ещё и `model.generate()` исторически ломался/требовал костылей вроде `summon_full_params`, что дорого. ([GitHub][3])

### Рекомендуемая топология (реально быстрая)

**Разделяйте тренинг и rollout-инференс:**

**(A) Rollout engine (инференс)**

* vLLM (идеально) или DeepSpeed Hybrid Engine.
* Генерирует `G` completions на prompt, возвращает:

  * `completion_ids`
  * `old_logprobs` по токенам (или хотя бы по выбранным токенам)
  * опционально `ref_logprobs` (если делаете KL к референсу)

**(B) Reward stage**

* Rule-based (верифицируемые задачи) или reward model.
* Желательно батчево и параллельно (CPU для правил, GPU для RM).

**(C) Trainer (ZeRO-3 или FSDP)**

* Делает forward на (prompt+completion) teacher-forcing’ом, считает `logprobs_new`,
* считает ratio/clipping/KL и делает backprop.

Эта схема — “каноническая” для ускорения RLHF: inference часто доминирует по времени (особенно длинный CoT), поэтому vLLM-rollout критичен. ([vLLM Blog][4])

---

## 2) Данные и семплинг: как разложить “group of G” по distributed так, чтобы не было боли

### Ключевое правило

**Все `G` сэмплов одного prompt должны жить на одном rank (или на одном “group” ranks).**
Иначе вам придётся делать неудобный all-gather rewards, чтобы посчитать mean/std по группе.

На практике делают “repeat sampler”: один индекс датасета повторяется `G` раз для генерации разных completions. Похожие самплеры есть в экосистеме (пример — GRPO sampler под sequence parallel). ([docs.axolotl.ai][5])

---

## 3) Роллаут (generation): что именно сохранять, чтобы GRPO считался дёшево

Для каждого completion вам нужно:

1. `completion_ids` (+ `prompt_ids`)
2. `old_logprobs` **только по сгенерированным токенам** (не храните полный logits)
3. (опционально) `ref_logprobs` по тем же токенам для KL
4. masks: где prompt, где completion, где padding

**Важно для памяти:** храните `old_logprobs` в fp16/bf16 (или fp32 если ловите нестабильность), но без logits.

---

## 4) Reward и advantage: как делать устойчиво (и почему это важно на длинном CoT)

Базовая формула advantage — mean/std внутри группы. ([arXiv][1])

Но в реальных раннах есть 2 частых улучшения (оба есть в TRL):

* **Отключить деление на std** (иногда даёт bias по “сложности вопроса”). ([Hugging Face][2])
* **Считать mean локально по группе, а std глобально по батчу** (робастнее shaping). ([Hugging Face][2])

На длинных chain-of-thought ещё всплывает проблема “length bias” → поэтому многие перешли на **token-level нормализацию (DAPO)** или варианты вроде Dr.GRPO. TRL это прямо документирует и даёт переключатели `loss_type`. ([Hugging Face][2])

---

## 5) Loss: как считать в коде (минимум лишних операций)

### Что вы оптимизируете (практический вариант)

На токенах completion (не на prompt):

* `logp_new_t = log π_θ(a_t | s_t)`
* `logp_old_t` сохранён из rollout
* ratio `r_t = exp(logp_new_t - logp_old_t)`
* clipped surrogate:

  * `pg_t = min(r_t * A, clip(r_t, 1-ε, 1+ε) * A)`
* (опционально) KL term, обычно через аппроксимацию Schulman:

  * `kl_t ≈ (π_new/π_ref) - log(π_new/π_ref) - 1` ([Hugging Face][2])
* итог: `loss = -(pg_t - β * kl_t)` агрегируете по токенам/seq согласно выбранному `loss_type`.

### Псевдокод (ядро)

```python
# tensors: [B*G, T] for completion tokens (after padding/packing)
ratio = torch.exp(logp_new - logp_old)              # no logits stored
ratio_clipped = torch.clamp(ratio, 1-eps, 1+eps)
pg = torch.minimum(ratio * adv, ratio_clipped * adv)

if beta > 0:
    # kl approx via log-ratio to reference; store ref_logp during rollout or recompute
    log_ratio_ref = (logp_new - logp_ref)
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
else:
    kl = 0.0

loss_token = -(pg - beta * kl) * completion_mask
loss = loss_token.sum() / completion_mask.sum()
```

---

## 6) DeepSpeed ZeRO-3: как сделать максимально быстро

### 6.1 Без CPU offload (лучший speed)

**Цель:** держать training model на GPU, сделать generation либо:

* через **vLLM server mode** на выделенных GPU/ноде, либо
* через **DeepSpeed Hybrid Engine** (если хотите всё “внутри DeepSpeed”). ([vLLM][6])

**Практически самый простой high-throughput паттерн**: выделить 1 GPU под vLLM, остальные под тренинг (как в гайдах). ([rocm.blogs.amd.com][7])

**В TRL это уже поддержано:** `use_vllm=True`, `vllm_mode="server"` или `"colocate"`. Server-mode обычно стабильнее по NCCL/памяти. ([Hugging Face][2])

### Важная ZeRO-3 настройка для generation

TRL имеет `ds3_gather_for_generation`:

* `True`: веса **собираются** для генерации → быстрее generation,
* `False`: можно тренировать модели, не вмещающиеся целиком в VRAM одного GPU, но generation медленнее, и это **не совместимо с vLLM generation** в TRL. ([Hugging Face][2])

**Вывод:** для “самого быстрого” режима — либо model помещается под gather, либо generation уезжает на отдельный inference-контур, где модель загружается полноценно (TP) и обновляется весами батчево.

---

### 6.2 ZeRO-3 + CPU offload (когда иначе не влезает)

Если вы включаете `offload_param`/`offload_optimizer`, вы выигрываете память, но платите latency на перемещения. DeepSpeed это официально поддерживает в stage-3. ([deepspeed.readthedocs.io][8])

**Чтобы не убиться по скорости:**

1. **Generation обязательно выносите** в отдельный inference-engine (vLLM/DS-inference), иначе training-GPU будет постоянно ждать подкачки.
2. **Увеличивайте “rollout per sync”**: генерируйте много (например, тысячи completions) на одну синхронизацию весов — DeepSeek-R1 описывает крупные роллауты и обучение мини-батчами. ([arXiv][1])
3. Рассмотрите `beta=0` (без KL) на ранних этапах — во многих современных GRPO-схемах KL не обязателен (TRL тоже по умолчанию β=0). ([Hugging Face][2])
   Если KL нужен — лучше получать `ref_logprobs` на inference-стороне (два policy в inference) или реже обновлять `π_ref`.

---

## 7) PyTorch FSDP: как сделать правильно (и не потерять скорость)

### 7.1 Конфигурация FSDP для LLM-тренинга

База из best practices:

* правильная wrapping policy (обычно по transformer block),
* mixed precision,
* backward prefetch,
* activation checkpointing. ([docs.pytorch.org][9])

CPU offload в FSDP существует (offload params на CPU), но как и в ZeRO-3 — это tradeoff speed/memory. ([Hugging Face][10])

### 7.2 Самый больной момент: generation под FSDP

`model.generate()` под FSDP часто либо ломается, либо требует `FSDP.summon_full_params`, что дорого. ([GitHub][3])

**Решение для эффективности то же:** выносите rollout в vLLM/инференс-процесс, а FSDP-тренер делает только teacher-forcing forward/backward на уже готовых (prompt+completion).

---

## 8) Синхронизация весов в rollout engine (важно и для ZeRO-3, и для FSDP)

Это “скрытая” стоимость GRPO.

### Лучшие практики

* **Обновляйте веса в inference не каждый step**, а раз в `K` steps (или `steps_per_generation`).
* Генерируйте большой пул completions между апдейтами (амортизация).
* Если inference = vLLM:

  * server mode хорошо масштабируется (отдельные GPU/ноды). ([Hugging Face][2])
* Если замечаете training/inference mismatch:

  * используйте importance sampling коррекцию (в TRL есть TIS/MIS). ([Hugging Face][2])

---

## 9) Оптимизации, которые реально дают x-ы (а не “косметику”)

### 9.1 Убираем повторный прогон общего префикса (самая GRPO-специфичная боль)

GRPO генерирует `G` продолжений для одного prompt → префикс один и тот же, а forward по префиксу повторяется `G` раз.

Есть свежая работа **Prefix Grouper**: “shared-prefix forward” — кодирует общий префикс один раз и переиспользует, сохраняя эквивалентность градиентов. Это прямой удар по основному compute-оверхеду GRPO на длинных prompt’ах. ([arXiv][11])

Если вы пишете свой стек, это одна из самых “вкусных” оптимизаций.

### 9.2 Loss type для длинных CoT

На практике часто берут `loss_type="dapo"` (token-level) или Dr.GRPO, чтобы не было странных эффектов длины. TRL даёт готовые формулировки и объясняет почему. ([Hugging Face][2])

---

## 10) Конкретные “рецепты” под ваши два бэкенда

### Рецепт A: “максимальный throughput” (если есть куда вынести inference)

* Trainer: ZeRO-3 **без** CPU offload (bf16), максимум overlap/сборка градиентов.
* Rollout: vLLM server на отдельном GPU/ноде.
* Reward: параллельный (async) rule-based + (опционально) RM.
* GRPO: `G=8..16`, `loss_type="dapo"`, `scale_rewards="group"` или `"batch"`.
* KL: `beta=0` сначала, потом включать если нужно (или держать ref, но обновлять редко).

Основания: TRL прямо рекомендует vLLM server/colocate, а OpenRLHF/vLLM показывают, что вынесенный inference — главный ускоритель. ([Hugging Face][2])

### Рецепт B: “модель не влезает без offload”

* Trainer: ZeRO-3 + `offload_param/offload_optimizer` (CPU), но:

  * inference **всегда отдельно** (иначе всё в latency).
* Rollout: vLLM на нескольких GPU (tensor_parallel) с редким обновлением весов.
* Делайте большие rollout-пулы между апдейтами (амортизация), как в DeepSeek-R1 (там описаны крупные роллауты и мини-батчи). ([arXiv][1])

### Рецепт C: FSDP-тренер

* FULL_SHARD, wrapping по transformer block, mixed precision, checkpointing. ([docs.pytorch.org][9])
* Никакого `generate()` внутри FSDP — rollout снаружи. ([GitHub][3])

---

## 11) “Скелет” процесса по компонентам (чтобы имплементировать самому)

1. **Sampler**: выдаёт batch prompts, каждый повторяется `G` раз на rank.
2. **Rollout**:

   * отправить prompts → inference engine,
   * получить `completion_ids`, `old_logprobs` (и `ref_logprobs` если надо).
3. **Reward**:

   * посчитать `r_i` для каждого completion (async).
4. **Advantage**:

   * сгруппировать по prompt, посчитать mean/std, получить `A_i`.
5. **Train step**:

   * teacher forcing forward по (prompt+completion),
   * собрать `logp_new` по токенам completion,
   * посчитать GRPO loss (clipped ratio + KL),
   * backprop/step через ZeRO-3 или FSDP.
6. **Weight sync** (каждые K шагов):

   * обновить inference weights (и ref, если используете “скользящий” reference).

---

Если хочешь, я могу в следующем сообщении накидать **два готовых “каркаса” кода**:

1. чистый PyTorch loop + DeepSpeed ZeRO-3 engine (с местами, где подключается vLLM server),
2. аналогичный loop под FSDP,

…с акцентом на то, чтобы **не хранить лишние logits**, правильно делать masks/packing, и чтобы группировка `G` никогда не расползалась по ranks.

[1]: https://arxiv.org/pdf/2501.12948 "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
[2]: https://huggingface.co/docs/trl/main/en/grpo_trainer "GRPO Trainer"
[3]: https://github.com/pytorch/pytorch/issues/100069?utm_source=chatgpt.com "Issue with FSDP + HuggingFace generate #100069"
[4]: https://blog.vllm.ai/2025/04/23/openrlhf-vllm.html "Accelerating RLHF with vLLM, Best Practice from OpenRLHF | vLLM Blog"
[5]: https://docs.axolotl.ai/docs/api/core.trainers.grpo.sampler.html?utm_source=chatgpt.com "core.trainers.grpo.sampler"
[6]: https://docs.vllm.ai/en/latest/training/trl/ "Transformers Reinforcement Learning - vLLM"
[7]: https://rocm.blogs.amd.com/software-tools-optimization/llm-grpo-rocm/README.html "Fine-Tuning LLMs with GRPO on AMD MI300X: Scalable RLHF with Hugging Face TRL and ROCm — ROCm Blogs"
[8]: https://deepspeed.readthedocs.io/en/latest/zero3.html?utm_source=chatgpt.com "ZeRO — DeepSpeed 0.18.5 documentation - Read the Docs"
[9]: https://docs.pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html?utm_source=chatgpt.com "Advanced Model Training with Fully Sharded Data Parallel ..."
[10]: https://huggingface.co/docs/peft/en/accelerate/fsdp?utm_source=chatgpt.com "Fully Sharded Data Parallel"
[11]: https://arxiv.org/abs/2506.05433 "[2506.05433] Prefix Grouper: Efficient GRPO Training through Shared-Prefix Forward"
