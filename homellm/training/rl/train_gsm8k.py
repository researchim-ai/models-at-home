#!/usr/bin/env python
"""
Скрипт для обучения reasoning на GSM8K с GRPO.

Пример использования:
    # Базовое обучение
    python -m homellm.training.rl.train_gsm8k --model Qwen/Qwen2.5-0.5B-Instruct

    # С кастомными параметрами
    python -m homellm.training.rl.train_gsm8k \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --algorithm drgrpo \
        --batch_size 4 \
        --group_size 8 \
        --max_samples 1000 \
        --output_dir ./output/grpo_gsm8k

    # С W&B логированием
    python -m homellm.training.rl.train_gsm8k \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --use_wandb \
        --wandb_project my-grpo-experiments
"""
import argparse
import logging
from pathlib import Path
from datetime import datetime

from .config import GRPOConfig, RLAlgorithm
from .trainer import GRPOTrainer
from .data.gsm8k import load_gsm8k
from .rewards.base import CombinedReward, UniversalRuleReward
from .rewards.math import GSM8KReward
from .rewards.format import FormatReward, ReasoningQualityReward

logger = logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Обучение LLM reasoning на GSM8K с GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Модель
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Название модели HuggingFace или путь",
    )
    
    # Алгоритм
    parser.add_argument(
        "--algorithm",
        type=str,
        default="grpo",
        choices=["grpo", "drgrpo", "dapo"],
        help="Алгоритм RL",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["grpo", "drgrpo", "dapo", "reasoning_small", "reasoning_large"],
        help="Использовать предустановленную конфигурацию",
    )
    
    # Датасет
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Максимальное количество примеров (None = весь датасет)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split датасета",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="Путь к JSONL файлу вместо GSM8K (поля: prompt, answer)",
    )
    
    # Batch размеры
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Количество промптов на шаг",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Количество генераций на промпт",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size для обучения",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Шагов накопления градиента",
    )
    
    # Генерация
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Максимум токенов в ответе",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Температура сэмплирования",
    )
    
    # Обучение
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Количество эпох",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Максимум шагов (None = по эпохам)",
    )
    
    # GRPO параметры
    parser.add_argument(
        "--clip_eps",
        type=float,
        default=0.2,
        help="Epsilon для PPO клиппинга",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.0,
        help="Вес KL штрафа (0 для reasoning)",
    )
    
    # LoRA
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Использовать LoRA",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Не использовать LoRA",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    
    # Квантизация
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Использовать 4-bit квантизацию",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Использовать 8-bit квантизацию",
    )
    
    # Логирование и сохранение
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Директория для сохранения (auto если не указано)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Сохранять каждые N шагов",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Логировать каждые N шагов",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Использовать W&B",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="homellm-grpo",
        help="Название проекта W&B",
    )
    
    # Формат
    parser.add_argument(
        "--reasoning_format",
        type=str,
        default="deepseek",
        choices=["deepseek", "simple", "russian"],
        help="Формат reasoning тегов",
    )
    
    # Разное
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # JSON конфигурация (для запуска из Streamlit UI)
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="JSON строка с полной конфигурацией (перезаписывает другие параметры)",
    )
    
    return parser.parse_args()


def main():
    """Основная функция."""
    import json
    
    args = parse_args()
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("GRPO Training на GSM8K")
    logger.info("=" * 60)
    
    # Если передана JSON конфигурация (из Streamlit UI)
    ui_config = None
    reward_rules = None
    
    if args.config_json:
        logger.info("Загрузка конфигурации из JSON...")
        ui_config = json.loads(args.config_json)
        
        # Извлекаем reward правила
        reward_rules = ui_config.get("grpo_reward_rules", [])
        if reward_rules:
            logger.info(f"Загружено {len(reward_rules)} reward правил из UI")
    
    # Создаём конфигурацию
    if args.preset:
        logger.info(f"Используем preset: {args.preset}")
        config = GRPOConfig.from_preset(args.preset)
    else:
        algorithm = args.algorithm
        if ui_config:
            algorithm = ui_config.get("grpo_algorithm", algorithm)
        config = GRPOConfig(
            algorithm=RLAlgorithm(algorithm),
        )
    
    # Переопределяем параметры из UI конфига
    # ВАЖНО: Все параметры должны быть явно переданы из UI, без fallback на args
    # Это гарантирует что мы точно знаем откуда берутся значения
    if ui_config:
        # GRPO параметры (обязательные из UI)
        if "grpo_prompt_batch_size" not in ui_config:
            raise ValueError("❌ Не задан grpo_prompt_batch_size (prompts/step) из UI.")
        config.batch_size = ui_config["grpo_prompt_batch_size"]

        config.group_size = ui_config["grpo_group_size"]
        if config.group_size < 8:
            raise ValueError("❌ group_size должен быть >= 8 для стабильного GRPO.")

        config.train_batch_size = ui_config["grpo_train_batch_size"]
        config.gradient_accumulation_steps = ui_config["gradient_accumulation"]
        config.max_new_tokens = ui_config["grpo_max_new_tokens"]
        config.temperature = ui_config["grpo_temperature"]
        config.learning_rate = ui_config["grpo_learning_rate"]
        config.max_steps = ui_config.get("grpo_max_steps")
        config.clip_eps_low = ui_config["grpo_clip_eps_low"]
        config.clip_eps_high = ui_config.get("grpo_clip_eps_high", config.clip_eps_low)
        config.kl_weight = ui_config["grpo_kl_weight"]
        config.epochs_per_step = ui_config.get("grpo_epochs_per_step", 1)
        config.reasoning_format = ui_config.get("grpo_reasoning_format", config.reasoning_format)
        
        # LoRA параметры (обязательные из UI, если use_lora=True)
        config.use_lora = ui_config.get("use_lora", config.use_lora)
        if config.use_lora:
            # ВАЖНО: Если use_lora=True, все LoRA параметры должны быть явно указаны в UI
            if "lora_r" not in ui_config or ui_config["lora_r"] is None:
                raise ValueError(
                    "❌ use_lora=True но lora_r не указан в UI конфиге! "
                    "Укажите lora_r в render_grpo_sidebar_config() или в model_config."
                )
            config.lora_r = ui_config["lora_r"]
            
            if "lora_alpha" not in ui_config or ui_config["lora_alpha"] is None:
                raise ValueError(
                    "❌ use_lora=True но lora_alpha не указан в UI конфиге! "
                    "Укажите lora_alpha в render_grpo_sidebar_config() или в model_config."
                )
            config.lora_alpha = ui_config["lora_alpha"]
            
            # lora_dropout и lora_target_modules могут быть None (используются дефолты из GRPOConfig)
            config.lora_dropout = ui_config.get("lora_dropout", config.lora_dropout)
            config.lora_target_modules = ui_config.get("lora_target_modules", config.lora_target_modules)
        
        # Квантизация (обязательные из UI)
        config.use_4bit = ui_config.get("use_4bit", False)
        config.use_8bit = ui_config.get("use_8bit", False)
        config.quantize_reference_model = ui_config.get("quantize_reference_model", config.quantize_reference_model)
        
        # Output директория
        config.output_dir = ui_config.get("output_dir", f"./output/grpo_gsm8k/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        config.batch_size = args.batch_size
        config.group_size = args.group_size
        config.train_batch_size = args.train_batch_size
        config.gradient_accumulation_steps = args.gradient_accumulation
        config.max_new_tokens = args.max_new_tokens
        config.temperature = args.temperature
        config.learning_rate = args.learning_rate
        config.num_epochs = args.num_epochs
        config.max_steps = args.max_steps
        config.clip_eps_low = args.clip_eps
        config.clip_eps_high = args.clip_eps if args.algorithm != "dapo" else 0.28
        config.kl_weight = args.kl_weight
        config.use_lora = args.use_lora and not args.no_lora
        config.lora_r = args.lora_r
        config.use_4bit = args.use_4bit
        config.use_8bit = args.use_8bit
        config.reasoning_format = args.reasoning_format
        
        # Output директория
        if args.output_dir:
            config.output_dir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.output_dir = f"./output/grpo_gsm8k/{timestamp}"
    
    config.save_steps = args.save_steps
    config.log_steps = args.log_steps
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    config.seed = args.seed
    
    # Определяем модель
    model_name = args.model
    if ui_config:
        model_name = ui_config.get("base_model_path") or ui_config.get("model_name", model_name)
    
    logger.info(f"Модель: {model_name}")
    logger.info(f"Алгоритм: {config.algorithm.value}")
    logger.info(f"Output: {config.output_dir}")
    
    # Загружаем датасет
    dataset_file = args.dataset_file
    max_samples = args.max_samples
    dataset_language = "en"  # По умолчанию английский
    
    if ui_config:
        dataset_source = ui_config.get("grpo_dataset_source", "")
        dataset_key = ui_config.get("grpo_dataset_key", "gsm8k_en")
        dataset_language = ui_config.get("grpo_dataset_language", "en")
        
        if "GSM8K" in dataset_source or dataset_key in ("gsm8k_en", "gsm8k_ru"):
            dataset_file = None  # Используем GSM8K из HuggingFace
            max_samples = ui_config.get("grpo_max_samples", max_samples)
            # Определяем язык по ключу датасета
            if dataset_key == "gsm8k_ru":
                dataset_language = "ru"
        else:
            dataset_file = ui_config.get("grpo_dataset_path") or ui_config.get("data_path")
    
    # Определяем ключ датасета
    dataset_key = None
    if ui_config:
        dataset_key = ui_config.get("grpo_dataset_key")
    
    if dataset_file:
        logger.info(f"Загрузка датасета из файла: {dataset_file}")
        from .data.base import RLDataset, RLSample
        from .data.gsm8k import extract_gsm8k_final_answer
        
        samples = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    # Если это GSM8K/GSM8K-RU стиль: answer содержит решение с "#### финал"
                    raw_answer = data.get("answer", data.get("response", ""))
                    ref_answer = raw_answer
                    if isinstance(raw_answer, str) and "####" in raw_answer:
                        ref_answer = extract_gsm8k_final_answer(raw_answer)
                    samples.append(RLSample(
                        prompt=data.get("prompt", data.get("question", "")),
                        reference_answer=ref_answer,
                        metadata={
                            **(data.get("metadata", {}) or {}),
                            "full_answer": raw_answer,
                        },
                    ))
        if max_samples:
            samples = samples[:max_samples]
        train_dataset = RLDataset(samples)
    else:
        # Загружаем датасет из HuggingFace
        dataset_names = {
            "gsm8k_en": "GSM8K (English)",
            "gsm8k_ru": "GSM8K-RU (d0rj/gsm8k-ru)",
            "math_ru": "MATH-RU (d0rj/competition_math_ru)",
            "mgsm_ru": "MGSM (juletxara/mgsm)",
        }
        ds_name = dataset_names.get(dataset_key, f"GSM8K ({dataset_language})")
        logger.info(f"Загрузка датасета: {ds_name}...")
        
        train_dataset = load_gsm8k(
            split=args.split,
            max_samples=max_samples,
            reasoning_format=config.reasoning_format,
            language=dataset_language,
            dataset_key=dataset_key,
        )
    logger.info(f"Загружено {len(train_dataset)} примеров")
    if len(train_dataset) <= 0:
        raise ValueError(
            "❌ Датасет пустой (0 примеров). "
            "Проверьте, что выбран reasoning-датасет (GSM8K/GSM8K-RU/MATH-RU), "
            "и что в JSONL есть поля question/prompt и answer/response."
        )
    
    # Создаём reward функцию
    if reward_rules:
        # Используем универсальные правила из UI
        logger.info(f"Создание UniversalRuleReward из {len(reward_rules)} правил")
        for rule in reward_rules:
            logger.info(f"  - {rule.get('name')}: weight={rule.get('weight')}")
        reward_fn = UniversalRuleReward.from_config(reward_rules)
    else:
        # Стандартная конфигурация для GSM8K
        logger.info("Используем стандартную reward функцию для GSM8K")
        reward_fn = CombinedReward([
            FormatReward(format_reward=0.2, weight=1.0),
            ReasoningQualityReward(max_reward=0.2, weight=0.5),
            GSM8KReward(correct_reward=1.0, close_reward=0.3, weight=2.0),
        ])
    
    # Создаём trainer
    trainer = GRPOTrainer(
        model_name=model_name,
        config=config,
        reward_fn=reward_fn,
    )
    
    # Запускаем обучение
    logger.info("Начинаем обучение...")
    trainer.train(train_dataset)
    
    logger.info("=" * 60)
    logger.info("Обучение завершено!")
    logger.info(f"Модель сохранена в: {config.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
