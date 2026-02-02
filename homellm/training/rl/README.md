# GRPO/RL Module for HomeLLM

Модуль обучения с подкреплением (RL) для развития reasoning способностей LLM.

## Поддерживаемые алгоритмы

### GRPO (Group Relative Policy Optimization)
Стандартная реализация из DeepSeek-R1:
- Групповая нормализация advantages (mean + std)
- PPO-style клиппинг
- Без отдельного критика

### Dr.GRPO (GRPO Done Right)
Улучшенная версия с исправлением biases:
- Только вычитание среднего (без деления на std)
- Фиксированная нормализация по длине
- Устраняет length bias и difficulty bias

### DAPO (Decoupled Clip + Dynamic Sampling)
Самая продвинутая версия:
- Асимметричный клиппинг (eps_high=0.28)
- Dynamic sampling для фильтрации zero-gradient групп
- Token-level loss (вместо sample-level)
- Мягкий штраф за слишком длинные ответы

## Быстрый старт

### Установка зависимостей

```bash
pip install -e ".[rl]"
```

### Обучение на GSM8K

```python
from homellm.training.rl import GRPOConfig, GRPOTrainer
from homellm.training.rl.data import load_gsm8k

# Загружаем датасет
dataset = load_gsm8k(split="train", max_samples=1000)

# Создаём конфигурацию
config = GRPOConfig.from_preset("reasoning_small")

# Создаём trainer
trainer = GRPOTrainer(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    config=config,
)

# Обучаем
trainer.train(dataset)
```

### Через командную строку

```bash
# Базовое обучение
python -m homellm.training.rl.train_rl --model Qwen/Qwen2.5-0.5B-Instruct

# С Dr.GRPO и W&B
python -m homellm.training.rl.train_rl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --algorithm drgrpo \
    --use_wandb \
    --max_samples 5000

# С 4-bit квантизацией для 3B модели
python -m homellm.training.rl.train_rl \
    --model Qwen/Qwen2.5-3B-Instruct \
    --use_4bit \
    --batch_size 2 \
    --group_size 4
```

## Конфигурация

### Предустановки (presets)

```python
# Стандартный GRPO
config = GRPOConfig.from_preset("grpo")

# Dr.GRPO
config = GRPOConfig.from_preset("drgrpo")

# DAPO (все улучшения)
config = GRPOConfig.from_preset("dapo")

# Для маленьких моделей (0.5-3B)
config = GRPOConfig.from_preset("reasoning_small")

# Для больших моделей (7B+)
config = GRPOConfig.from_preset("reasoning_large")
```

### Важные параметры

```python
config = GRPOConfig(
    # Алгоритм
    algorithm="drgrpo",  # "grpo", "drgrpo", "dapo"
    
    # Batch размеры
    group_size=8,        # Генераций на промпт
    batch_size=4,        # Промптов на шаг
    train_batch_size=2,  # Для обучения
    
    # Генерация
    max_new_tokens=512,
    temperature=0.7,
    
    # Клиппинг
    clip_eps_low=0.2,
    clip_eps_high=0.28,  # Для DAPO
    
    # KL (обычно 0 для reasoning)
    kl_weight=0.0,
    
    # LoRA
    use_lora=True,
    lora_r=16,
    
    # Квантизация
    use_4bit=True,
)
```

## Reward функции

### Встроенные

```python
from homellm.training.rl.rewards import (
    FormatReward,         # Проверка формата reasoning тегов
    ReasoningQualityReward,  # Качество chain-of-thought
    GSM8KReward,          # Для GSM8K математики
    CombinedReward,       # Комбинация нескольких
)

# Комбинированный reward
reward_fn = CombinedReward([
    FormatReward(weight=1.0),
    ReasoningQualityReward(weight=0.5),
    GSM8KReward(weight=2.0),
])
```

### Кастомный reward

```python
from homellm.training.rl.rewards.base import RewardFunction

class MyReward(RewardFunction):
    def __call__(self, completion, reference_answer, **kwargs):
        # Ваша логика
        if "correct" in completion.lower():
            return 1.0
        return 0.0

# Использование
trainer = GRPOTrainer(
    model_name="...",
    reward_fn=MyReward(),
)
```

## Форматы reasoning

### DeepSeek формат (по умолчанию)
```
<think>
Шаги решения...
</think>
<answer>
42
</answer>
```

### Simple формат
```
<reasoning>
Шаги решения...
</reasoning>
<answer>
42
</answer>
```

## Структура модуля

```
homellm/training/rl/
├── __init__.py          # Экспорты
├── config.py            # GRPOConfig
├── trainer.py           # GRPOTrainer
├── loss.py              # Loss функции
├── experience.py        # Experience буфер
├── rollout.py           # Генерация rollout'ов
├── train_rl.py       # Скрипт обучения
├── algorithms/          # (для будущих алгоритмов)
├── rewards/             # Reward функции
│   ├── base.py
│   ├── format.py
│   └── math.py
└── data/                # Датасеты
    ├── base.py
    └── gsm8k.py
```

## Мониторинг обучения

### W&B метрики

При включённом `use_wandb` логируются:
- `rollouts/returns_mean` - средний reward
- `training/loss` - loss
- `training/kl` - KL дивергенция
- `training/grad_norm` - норма градиента
- `val/accuracy` - точность на валидации

### Ключевые индикаторы здоровья

1. **Reward** должен расти
2. **Длина ответов** должна медленно расти (модель учится рассуждать)
3. **KL** должен быть небольшим (или 0 для reasoning)
4. **Энтропия** не должна резко падать (entropy collapse)

## Лучшие практики

1. **Начните с маленькой модели** (0.5B-1.5B) для отладки
2. **Используйте LoRA** для эффективного обучения
3. **Batch size важен** - слишком маленький = нестабильность
4. **Group size ≥ 8** для надёжной оценки advantages
5. **KL = 0** для reasoning задач (модель должна отклоняться от исходной)
6. **Dr.GRPO или DAPO** вместо vanilla GRPO для стабильности

## Ссылки

- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) - оригинальная статья GRPO
- [GRPO++](https://cameronrwolfe.substack.com/p/grpo-tricks) - обзор улучшений
- [GSM8K](https://github.com/openai/grade-school-math) - датасет
