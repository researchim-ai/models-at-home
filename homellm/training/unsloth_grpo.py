"""
Unsloth backend для GRPO тренировки.
Использует оптимизации Unsloth для ускорения и экономии памяти.

Особенности:
- 2x быстрее обучение
- До 70% меньше VRAM
- Triton ядра (RMSNorm, RoPE, SwiGLU)
- Smart Gradient Checkpointing
- Оптимизированные RL функции
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

import torch

logger = logging.getLogger(__name__)


def is_unsloth_available() -> bool:
    """Проверяет доступность Unsloth."""
    try:
        import unsloth
        return True
    except ImportError:
        return False


def run_unsloth_grpo(
    config: Dict[str, Any],
    metrics_logger: Any,
    dataset: Any = None,
    reward_fn: Optional[Callable] = None,
) -> None:
    """
    Запуск GRPO тренировки с использованием Unsloth backend.
    
    Args:
        config: Конфигурация тренировки
        metrics_logger: Логгер метрик для UI
        dataset: Датасет (опционально, иначе загружается по config)
        reward_fn: Функция награды (опционально)
    """
    # ВАЖНО: Unsloth должен импортироваться ПЕРВЫМ, до transformers/trl
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
        from unsloth.models.rl import PatchFastRL
    except ImportError as e:
        raise ImportError(
            "Unsloth не установлен. Установите через: pip install unsloth\n"
            f"Ошибка: {e}"
        )
    
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError(
            "trl не установлен или версия не поддерживает GRPO. "
            "Установите: pip install trl>=0.9.0"
        )
    
    from datasets import load_dataset, Dataset
    
    metrics_logger.update(status="loading_model", backend="unsloth")
    
    # === Параметры из UI ===
    base_model_path = config.get("base_model_path")
    if not base_model_path:
        raise ValueError("base_model_path required for GRPO")
    
    max_seq_length = config.get("seq_len", 2048)
    dtype = None  # Auto-detect
    
    # Precision
    mixed_precision = config.get("mixed_precision", "bf16")
    if mixed_precision == "bf16" and is_bfloat16_supported():
        dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # === Метод тюнинга из UI ===
    # tuning_method: "full" / "lora" / "qlora"
    tuning_method = config.get("tuning_method", "full")
    
    # Определяем режим на основе tuning_method
    if tuning_method == "qlora":
        use_lora = True
        load_in_4bit = True
        logger.info("🦥 Mode: QLoRA (4-bit quantization + LoRA)")
    elif tuning_method == "lora":
        use_lora = True
        load_in_4bit = False
        logger.info("🦥 Mode: LoRA (16-bit + LoRA)")
    else:  # "full"
        use_lora = False
        load_in_4bit = False
        logger.info("🦥 Mode: Full fine-tuning (all weights trainable)")
    
    # Переопределение из конфига (если явно указано)
    if "use_lora" in config:
        use_lora = config["use_lora"]
    if "use_4bit" in config:
        load_in_4bit = config["use_4bit"]
    if "load_in_4bit" in config:
        load_in_4bit = config["load_in_4bit"]
    
    # Валидация: 4-bit требует LoRA
    if load_in_4bit and not use_lora:
        logger.warning("⚠️ 4-bit квантизация требует LoRA — включаем LoRA!")
        use_lora = True
    
    # LoRA параметры
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.0)
    lora_target_modules = config.get("lora_target_modules")
    
    logger.info(f"🦥 Final settings: use_lora={use_lora}, load_in_4bit={load_in_4bit}")
    if use_lora:
        logger.info(f"🦥 LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # === Загрузка модели через Unsloth ===
    logger.info(f"🦥 Unsloth GRPO: Loading model from {base_model_path}")
    logger.info(f"   max_seq_length={max_seq_length}, dtype={dtype}, load_in_4bit={load_in_4bit}")
    
    # Для multi-GPU: указываем device_map на текущий GPU процесса
    # Каждый процесс accelerate/DDP работает со своим GPU
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": f"cuda:{local_rank}"}
    logger.info(f"🦥 Device map: {device_map} (LOCAL_RANK={local_rank})")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        device_map=device_map,
    )
    
    # === Добавляем LoRA адаптеры ===
    if use_lora:
        logger.info(f"🦥 Unsloth: Adding LoRA adapters (r={lora_r}, alpha={lora_alpha})")
        
        # Target modules из конфига или дефолтные для трансформеров
        if lora_target_modules:
            target_modules = lora_target_modules
        else:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        
        logger.info(f"🦥 LoRA target modules: {target_modules}")
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth smart checkpointing
            random_state=42,
            max_seq_length=max_seq_length,
        )
    
    # Pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # === Применяем RL патчи Unsloth ===
    logger.info("🦥 Applying Unsloth RL patches...")
    try:
        PatchFastRL("unsloth/Llama-3.2-1B-Instruct", FastLanguageModel)  # dummy call to patch
    except Exception as e:
        logger.warning(f"⚠️ Could not apply Unsloth RL patches: {e}")
    
    metrics_logger.update(status="loading_dataset")
    
    # === Загрузка датасета ===
    if dataset is None:
        # Пробуем разные ключи из UI config
        data_path = (
            config.get("grpo_dataset_path") or 
            config.get("data_path") or 
            config.get("dataset_path")
        )
        dataset_key = config.get("grpo_dataset_key", "")
        dataset_source = config.get("grpo_dataset_source", "")
        
        # GSM8K из HuggingFace
        if "GSM8K" in dataset_source or dataset_key in ("gsm8k_en", "gsm8k_ru"):
            logger.info(f"🦥 Loading GSM8K dataset (key={dataset_key})...")
            if dataset_key == "gsm8k_ru":
                # Русский GSM8K
                dataset = load_dataset("d0rj/gsm8k-ru", split="train")
            else:
                # Английский GSM8K
                dataset = load_dataset("openai/gsm8k", "main", split="train")
        elif data_path:
            # Загружаем датасет из файла
            logger.info(f"🦥 Loading dataset from: {data_path}")
            if data_path.endswith((".json", ".jsonl")):
                dataset = load_dataset("json", data_files=data_path, split="train")
            else:
                # HuggingFace dataset
                hf_config = config.get("hf_dataset_config")
                if hf_config:
                    dataset = load_dataset(data_path, hf_config, split="train")
                else:
                    dataset = load_dataset(data_path, split="train")
        else:
            # Default: GSM8K English
            logger.info("🦥 No dataset specified, defaulting to GSM8K (English)")
            dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    # === Форматирование датасета для GRPO ===
    prompt_col = config.get("grpo_prompt_column", "question")
    answer_col = config.get("grpo_answer_column", "answer")
    
    # Детектируем колонки в датасете
    cols = dataset.column_names
    logger.info(f"🦥 Dataset columns: {cols}")
    
    # Для GSM8K колонки: question, answer
    if "question" in cols and prompt_col not in cols:
        prompt_col = "question"
    if "answer" in cols and answer_col not in cols:
        answer_col = "answer"
    
    def format_for_grpo(example):
        """Форматирует пример для GRPO."""
        question = example.get(prompt_col, "")
        answer = example.get(answer_col, "")
        
        # Формируем промпт
        prompt = f"Question: {question}\n\nLet's think step by step.\n\n"
        
        return {
            "prompt": prompt,
            "ground_truth": answer,
        }
    
    dataset = dataset.map(format_for_grpo, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x.get("prompt", "")) > 0)
    
    logger.info(f"🦥 Dataset prepared: {len(dataset)} examples")
    
    metrics_logger.update(
        status="training",
        num_train_examples=len(dataset),
        backend="unsloth",
    )
    
    # === GRPO Config ===
    output_dir = Path(config.get("output_dir", "out/unsloth_grpo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = config.get("batch_size", 1)
    gradient_accumulation = config.get("gradient_accumulation", 8)
    num_generations = config.get("grpo_num_generations", 4)
    max_new_tokens = config.get("grpo_max_new_tokens", 512)
    
    # Базовые параметры GRPOConfig (совместимо с разными версиями trl)
    grpo_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=config.get("epochs", 1),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=config.get("learning_rate", 5e-6),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 50),
        lr_scheduler_type=config.get("lr_schedule", "cosine"),
        logging_steps=config.get("log_every", 10),
        save_steps=config.get("save_every", 500),
        save_total_limit=3,
        bf16=mixed_precision == "bf16" and is_bfloat16_supported(),
        fp16=mixed_precision == "fp16",
        optim="adamw_8bit",
        seed=42,
        max_grad_norm=config.get("max_grad_norm", 1.0),
        report_to="none",
        
        # GRPO specific
        num_generations=num_generations,
        temperature=config.get("grpo_temperature", 0.7),
        beta=config.get("grpo_beta", 0.1),  # KL coefficient
    )
    
    # Добавляем max_completion_length если поддерживается (новый API trl)
    # или max_new_tokens (старый API)
    import inspect
    grpo_sig = inspect.signature(GRPOConfig.__init__)
    if "max_completion_length" in grpo_sig.parameters:
        grpo_kwargs["max_completion_length"] = max_new_tokens
    elif "max_new_tokens" in grpo_sig.parameters:
        grpo_kwargs["max_new_tokens"] = max_new_tokens
    
    grpo_config = GRPOConfig(**grpo_kwargs)
    
    # === Reward Function ===
    if reward_fn is None:
        # Создаём дефолтную reward function
        def default_reward_fn(completions: List[str], ground_truths: List[str]) -> List[float]:
            """Простая reward function на основе match."""
            rewards = []
            for completion, gt in zip(completions, ground_truths):
                # Извлекаем финальный ответ
                if "####" in completion:
                    pred = completion.split("####")[-1].strip()
                else:
                    pred = completion.strip().split()[-1] if completion.strip() else ""
                
                gt_clean = str(gt).strip()
                
                # Точное совпадение
                if pred == gt_clean:
                    rewards.append(1.0)
                # Частичное совпадение
                elif gt_clean in pred or pred in gt_clean:
                    rewards.append(0.5)
                else:
                    rewards.append(0.0)
            
            return rewards
        
        reward_fn = default_reward_fn
    
    # === Trainer ===
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=grpo_config,
        reward_funcs=reward_fn,
    )
    
    # === Callback для метрик ===
    from transformers import TrainerCallback
    
    class MetricsCallback(TrainerCallback):
        def __init__(self, metrics_logger, start_time):
            self.metrics_logger = metrics_logger
            self.start_time = start_time
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            
            step = state.global_step
            loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            reward = logs.get("reward", 0.0)
            
            elapsed = time.time() - self.start_time
            
            self.metrics_logger.log_step(
                step=step,
                loss=loss,
                lr=lr,
            )
            
            # Дополнительные GRPO метрики
            self.metrics_logger.update(
                current_reward=reward,
                kl_divergence=logs.get("kl", 0.0),
            )
        
        def on_save(self, args, state, control, **kwargs):
            ckpt_path = str(output_dir / f"checkpoint-{state.global_step}")
            self.metrics_logger.log_checkpoint(ckpt_path)
    
    trainer.add_callback(MetricsCallback(metrics_logger, time.time()))
    
    # === Запуск тренировки ===
    logger.info("🦥 Unsloth: Starting GRPO training...")
    start_time = time.time()
    
    # Используем unsloth train если доступен
    try:
        from unsloth import unsloth_train
        unsloth_train(trainer)
    except ImportError:
        trainer.train()
    
    total_time = time.time() - start_time
    
    # === Сохранение модели ===
    metrics_logger.update(status="saving_model")
    
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    if use_lora:
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"🦥 Saved LoRA adapters to {final_dir}")
        
        if config.get("merge_lora", False):
            merged_dir = output_dir / "merged_model"
            merged_dir.mkdir(parents=True, exist_ok=True)
            
            model = model.merge_and_unload()
            model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            logger.info(f"🦥 Saved merged model to {merged_dir}")
    else:
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"🦥 Saved full model to {final_dir}")
    
    # === Финальные метрики ===
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    duration_str = f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"
    
    metrics_logger.update(
        status="completed",
        total_time_seconds=total_time,
        training_duration=duration_str,
        final_model_path=str(final_dir),
        backend="unsloth",
    )
    
    logger.info(f"🦥 Unsloth GRPO completed in {duration_str}")
