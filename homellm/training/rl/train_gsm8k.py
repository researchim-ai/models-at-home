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
from .rewards.base import CombinedReward
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
    
    return parser.parse_args()


def main():
    """Основная функция."""
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
    
    # Создаём конфигурацию
    if args.preset:
        logger.info(f"Используем preset: {args.preset}")
        config = GRPOConfig.from_preset(args.preset)
    else:
        config = GRPOConfig(
            algorithm=RLAlgorithm(args.algorithm),
        )
    
    # Переопределяем параметры из аргументов
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
    config.save_steps = args.save_steps
    config.log_steps = args.log_steps
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    config.reasoning_format = args.reasoning_format
    config.seed = args.seed
    
    # Output директория
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = f"./output/grpo_gsm8k/{timestamp}"
    
    logger.info(f"Модель: {args.model}")
    logger.info(f"Алгоритм: {config.algorithm.value}")
    logger.info(f"Output: {config.output_dir}")
    
    # Загружаем датасет
    if args.dataset_file:
        logger.info(f"Загрузка датасета из файла: {args.dataset_file}")
        from .data.base import RLDataset, RLSample
        import json
        
        samples = []
        with open(args.dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    samples.append(RLSample(
                        prompt=data.get("prompt", data.get("question", "")),
                        reference_answer=data.get("answer", data.get("response", "")),
                        metadata=data.get("metadata", {}),
                    ))
        if args.max_samples:
            samples = samples[:args.max_samples]
        train_dataset = RLDataset(samples)
    else:
        logger.info("Загрузка GSM8K датасета...")
        train_dataset = load_gsm8k(
            split=args.split,
            max_samples=args.max_samples,
            reasoning_format=args.reasoning_format,
        )
    logger.info(f"Загружено {len(train_dataset)} примеров")
    
    # Создаём reward функцию
    reward_fn = CombinedReward([
        FormatReward(format_reward=0.2, weight=1.0),
        ReasoningQualityReward(max_reward=0.2, weight=0.5),
        GSM8KReward(correct_reward=1.0, close_reward=0.3, weight=2.0),
    ])
    
    # Создаём trainer
    trainer = GRPOTrainer(
        model_name=args.model,
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
