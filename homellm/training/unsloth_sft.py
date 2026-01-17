"""
Unsloth backend для SFT тренировки.
Использует оптимизации Unsloth для ускорения и экономии памяти.

Особенности:
- 2x быстрее обучение
- До 70% меньше VRAM
- Triton ядра (RMSNorm, RoPE, SwiGLU)
- Smart Gradient Checkpointing
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)


def is_unsloth_available() -> bool:
    """Проверяет доступность Unsloth."""
    try:
        import unsloth
        return True
    except ImportError:
        return False


def run_unsloth_sft(
    config: Dict[str, Any],
    metrics_logger: Any,
) -> None:
    """
    Запуск SFT тренировки с использованием Unsloth backend.
    
    Args:
        config: Конфигурация тренировки
        metrics_logger: Логгер метрик для UI
    """
    # ВАЖНО: Unsloth должен импортироваться ПЕРВЫМ, до transformers/trl
    try:
        from unsloth import FastLanguageModel
        from unsloth import is_bfloat16_supported
    except ImportError as e:
        raise ImportError(
            "Unsloth не установлен. Установите через: pip install unsloth\n"
            f"Ошибка: {e}"
        )
    
    from trl import SFTTrainer
    try:
        from trl import SFTConfig as TrainingArguments
    except ImportError:
        from transformers import TrainingArguments
    
    from datasets import load_dataset, Dataset
    from transformers import DataCollatorForSeq2Seq
    
    metrics_logger.update(status="loading_model", backend="unsloth")
    
    # === Параметры ===
    base_model_path = config.get("base_model_path")
    if not base_model_path:
        raise ValueError("base_model_path required for SFT")
    
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
    
    # LoRA config
    use_lora = config.get("use_lora", True)
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.0)
    
    # Quantization
    load_in_4bit = config.get("load_in_4bit", True)
    
    # === Загрузка модели через Unsloth ===
    logger.info(f"🦥 Unsloth: Loading model from {base_model_path}")
    logger.info(f"   max_seq_length={max_seq_length}, dtype={dtype}, load_in_4bit={load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    
    # === Добавляем LoRA адаптеры ===
    if use_lora:
        logger.info(f"🦥 Unsloth: Adding LoRA adapters (r={lora_r}, alpha={lora_alpha})")
        
        # Определяем target modules на основе архитектуры
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth smart checkpointing
            random_state=42,
            max_seq_length=max_seq_length,
        )
    
    # Pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    metrics_logger.update(status="loading_dataset")
    
    # === Загрузка датасета ===
    data_path = config.get("data_path")
    if not data_path:
        raise ValueError("data_path required for SFT")
    
    # Определяем формат данных
    sft_columns = config.get("sft_columns", {})
    sft_template = config.get("sft_template", {})
    
    # Загружаем датасет
    if data_path.endswith((".json", ".jsonl")):
        dataset = load_dataset("json", data_files=data_path, split="train")
    else:
        # HuggingFace dataset
        dataset = load_dataset(data_path, split="train")
    
    # === Форматирование данных для SFT ===
    def format_instruction(example):
        """Форматирует пример в instruction-response формат."""
        fmt = sft_columns.get("format", "instruct")
        
        if fmt == "chat":
            # Chat format с messages
            messages_col = sft_columns.get("messages", "messages")
            messages = example.get(messages_col, [])
            if not messages:
                return {"text": ""}
            
            # Применяем chat template если есть
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                # Fallback к простому форматированию
                text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        text += f"### User:\n{content}\n\n"
                    elif role == "assistant":
                        text += f"### Assistant:\n{content}\n\n"
                    elif role == "system":
                        text += f"{content}\n\n"
            
            return {"text": text}
        else:
            # Instruct format
            instr_col = sft_columns.get("instruction", "instruction")
            input_col = sft_columns.get("input", "input")
            output_col = sft_columns.get("output", "output")
            
            instruction = example.get(instr_col, "")
            inp = example.get(input_col, "")
            output = example.get(output_col, "")
            
            # Шаблон
            system = sft_template.get("system", "You are a helpful assistant.")
            user_tag = sft_template.get("user_tag", "### User:")
            bot_tag = sft_template.get("bot_tag", "### Assistant:")
            separator = sft_template.get("separator", "\n\n")
            
            # Формируем текст
            text = f"{system}{separator}"
            if inp:
                text += f"{user_tag}{separator}{instruction}\n\nInput: {inp}{separator}"
            else:
                text += f"{user_tag}{separator}{instruction}{separator}"
            text += f"{bot_tag}{separator}{output}"
            
            return {"text": text}
    
    # Применяем форматирование
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Фильтруем пустые
    dataset = dataset.filter(lambda x: len(x.get("text", "")) > 0)
    
    logger.info(f"🦥 Dataset prepared: {len(dataset)} examples")
    
    metrics_logger.update(
        status="training",
        num_train_examples=len(dataset),
        backend="unsloth",
    )
    
    # === Training Arguments ===
    output_dir = Path(config.get("output_dir", "out/unsloth_sft"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Вычисляем шаги
    batch_size = config.get("batch_size", 4)
    gradient_accumulation = config.get("gradient_accumulation", 4)
    epochs = config.get("epochs", 1)
    max_steps = config.get("max_steps", -1)  # -1 = все эпохи
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=config.get("learning_rate", 2e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 100),
        lr_scheduler_type=config.get("lr_schedule", "cosine"),
        logging_steps=config.get("log_every", 10),
        save_steps=config.get("save_every", 500),
        save_total_limit=3,
        bf16=mixed_precision == "bf16" and is_bfloat16_supported(),
        fp16=mixed_precision == "fp16",
        optim="adamw_8bit",  # Unsloth рекомендует 8bit optimizer
        seed=42,
        max_grad_norm=config.get("max_grad_norm", 1.0),
        report_to="none",  # Мы сами логируем
    )
    
    # === Создаём Trainer ===
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,  # Без packing для простоты
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
            
            # Время шага
            elapsed = time.time() - self.start_time
            samples_per_sec = (step * batch_size * gradient_accumulation) / max(1, elapsed)
            
            self.metrics_logger.log_step(
                step=step,
                loss=loss,
                lr=lr,
                samples_per_sec=samples_per_sec,
            )
        
        def on_save(self, args, state, control, **kwargs):
            ckpt_path = str(output_dir / f"checkpoint-{state.global_step}")
            self.metrics_logger.log_checkpoint(ckpt_path)
    
    trainer.add_callback(MetricsCallback(metrics_logger, time.time()))
    
    # === Запуск тренировки ===
    logger.info("🦥 Unsloth: Starting training...")
    start_time = time.time()
    
    # Unsloth train
    try:
        from unsloth import unsloth_train
        unsloth_train(trainer)
    except ImportError:
        # Fallback если unsloth_train недоступен
        trainer.train()
    
    total_time = time.time() - start_time
    
    # === Сохранение финальной модели ===
    metrics_logger.update(status="saving_model")
    
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем LoRA адаптеры (если use_lora=True) или всю модель
    if use_lora:
        # Сохраняем только LoRA адаптеры
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"🦥 Saved LoRA adapters to {final_dir}")
        
        # Опционально: merge и сохранение полной модели
        if config.get("merge_lora", False):
            merged_dir = output_dir / "merged_model"
            merged_dir.mkdir(parents=True, exist_ok=True)
            
            # Merge LoRA в базовую модель
            model = model.merge_and_unload()
            model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            logger.info(f"🦥 Saved merged model to {merged_dir}")
    else:
        # Full fine-tuning - сохраняем всё
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
    
    logger.info(f"🦥 Unsloth SFT completed in {duration_str}")
