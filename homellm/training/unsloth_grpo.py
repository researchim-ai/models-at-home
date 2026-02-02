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
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

# Подавляем warning от Unsloth про порядок импортов
# (мы намеренно импортируем его после установки env variables)
warnings.filterwarnings("ignore", message=".*Unsloth should be imported before.*")

import torch

logger = logging.getLogger(__name__)


def is_unsloth_available() -> bool:
    """Проверяет доступность Unsloth."""
    try:
        import unsloth
        return True
    except ImportError:
        return False


def _patch_unsloth_left_pack_padding():
    """
    Monkey-patch для фикса бага в Unsloth:
    torch.argsort не поддерживает bool dtype на CUDA с stable=True.
    
    Unsloth компилирует код в кэш, поэтому нужно патчить torch.argsort глобально.
    """
    try:
        import torch
        
        # Сохраняем оригинальный argsort
        _original_argsort = torch.argsort
        
        def _patched_argsort(input, dim=-1, descending=False, stable=False):
            """Patched argsort: converts bool to int for CUDA compatibility."""
            # Если bool на CUDA с stable=True — конвертируем в int
            if input.dtype == torch.bool and input.is_cuda and stable:
                input = input.int()
            return _original_argsort(input, dim=dim, descending=descending, stable=stable)
        
        # Заменяем глобально
        torch.argsort = _patched_argsort
        
        logger.info("🦥 Applied global fix for torch.argsort (bool on CUDA)")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not patch torch.argsort: {e}")
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
    import os
    
    # === Проверка и инициализация multi-GPU ===
    # Unsloth имеет экспериментальную поддержку multi-GPU
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    
    if world_size > 1:
        logger.warning("=" * 60)
        logger.warning("⚠️ Unsloth multi-GPU (DDP) — экспериментальная поддержка!")
        logger.warning(f"   WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
        logger.warning("   При проблемах используйте 'models-at-home backend'")
        logger.warning("=" * 60)
        
        # ВАЖНО: Инициализируем torch.distributed РАНЬШЕ загрузки модели
        # чтобы можно было использовать барьеры для последовательной загрузки
        if not torch.distributed.is_initialized():
            logger.info(f"🦥 Rank {rank}: Initializing torch.distributed for sequential model loading...")
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )
            torch.cuda.set_device(local_rank)
            logger.info(f"🦥 Rank {rank}: torch.distributed initialized!")
        
        # ВАЖНО: Барьер чтобы все процессы дождались инициализации друг друга
        logger.info(f"🦥 Rank {rank}: Waiting for all processes to initialize...")
        torch.distributed.barrier()
        logger.info(f"🦥 Rank {rank}: All processes ready!")
    
    # === Определяем режим тюнинга ДО импорта Unsloth ===
    tuning_method = config.get("tuning_method", "full")
    
    # ВАЖНО: Unsloth GRPO НЕ поддерживает full fine-tuning (баг в их коде)
    # Форсируем LoRA если выбран full
    if tuning_method == "full":
        logger.warning("⚠️ Unsloth GRPO не поддерживает full fine-tuning! Автоматически переключаюсь на LoRA.")
        logger.warning("   Для full fine-tuning используйте models-at-home backend.")
        tuning_method = "lora"  # Форсируем LoRA
        config["tuning_method"] = "lora"  # Обновляем config
    
    os.environ["UNSLOTH_ENABLE_FULL_FINETUNING"] = "0"
    
    # Сначала применяем патч для бага с bool argsort на CUDA
    _patch_unsloth_left_pack_padding()
    
    # ВАЖНО: Unsloth должен импортироваться ПОСЛЕ установки env variables
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
    
    # === Метод тюнинга из UI (уже обработан выше — full → lora) ===
    # tuning_method теперь: "lora" или "qlora"
    
    # Unsloth GRPO поддерживает только LoRA/QLoRA
    if tuning_method == "qlora":
        use_lora = True
        load_in_4bit = True
        full_finetuning = False
        logger.info("🦥 Mode: QLoRA (4-bit quantization + LoRA)")
    else:  # "lora" (full был автоматически переключён на lora выше)
        use_lora = True
        load_in_4bit = False
        full_finetuning = False
        logger.info("🦥 Mode: LoRA (16-bit + LoRA)")
    
    # Переопределение из конфига (если явно указано)
    if "use_4bit" in config and config["use_4bit"]:
        load_in_4bit = True
    if "load_in_4bit" in config and config["load_in_4bit"]:
        load_in_4bit = True
    
    # LoRA параметры
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.0)
    lora_target_modules = config.get("lora_target_modules")
    
    # ВАЖНО: Unsloth требует dropout = 0 для быстрых патчей!
    # Иначе он не применяет оптимизации к QKV, O, MLP слоям
    if lora_dropout != 0.0:
        logger.warning(f"⚠️ Unsloth требует dropout=0 для максимальной производительности!")
        logger.warning(f"   Меняю dropout с {lora_dropout} на 0.0")
        lora_dropout = 0.0
    
    logger.info(f"🦥 Final settings: use_lora={use_lora}, load_in_4bit={load_in_4bit}, full_finetuning={full_finetuning}")
    if use_lora:
        logger.info(f"🦥 LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # === Загрузка модели через Unsloth ===
    logger.info(f"🦥 Unsloth GRPO: Loading model from {base_model_path}")
    logger.info(f"   max_seq_length={max_seq_length}, dtype={dtype}, load_in_4bit={load_in_4bit}, full_finetuning={full_finetuning}")
    
    # Для multi-GPU: указываем device_map на текущий GPU процесса
    # Каждый процесс accelerate/DDP работает со своим GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": f"cuda:{local_rank}"}
    logger.info(f"🦥 Device map: {device_map} (LOCAL_RANK={local_rank})")
    
    # vLLM GPU utilization из UI config
    gpu_memory_utilization = config.get("grpo_vllm_gpu_util", 0.4)
    
    # Параметры для FastLanguageModel.from_pretrained
    load_kwargs = dict(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
    )
    
    # fast_inference=True включает vLLM для быстрой генерации (ВАЖНО для скорости!)
    # НО: fast_inference НЕ совместим с trust_remote_code!
    # НО: fast_inference требует LoRA, не работает с full_finetuning
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_multi_gpu = world_size > 1
    rank = int(os.environ.get("RANK", 0))
    
    if use_lora:
        # LoRA mode: включаем fast_inference для скорости
        load_kwargs["max_lora_rank"] = lora_r
        load_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        
        if is_multi_gpu:
            # Multi-GPU: vLLM конфликтует при DDP, отключаем fast_inference
            # Unsloth официально: "Unsloth currently does not support multi GPU setups"
            # Но DDP для training работает, просто без vLLM для генерации
            load_kwargs["fast_inference"] = False
            logger.warning("=" * 60)
            logger.warning("⚠️ Multi-GPU + fast_inference (vLLM) НЕ поддерживается Unsloth!")
            logger.warning("   Отключаю fast_inference для стабильной работы.")
            logger.warning("   Для максимальной скорости используйте single GPU:")
            logger.warning("   CUDA_VISIBLE_DEVICES=0 python -m homellm.training.rl.train_rl")
            logger.warning("=" * 60)
            logger.info(f"🦥 Multi-GPU ({world_size} GPUs): Training with DDP (без vLLM)")
        else:
            load_kwargs["fast_inference"] = True  # Включаем vLLM только для single GPU!
            logger.info(f"🦥 Single GPU: Enabling fast_inference (vLLM) with max_lora_rank={lora_r}, gpu_util={gpu_memory_utilization}")
    else:
        # Full fine-tuning: vLLM не поддерживается
        load_kwargs["full_finetuning"] = full_finetuning
        load_kwargs["trust_remote_code"] = True
        logger.info("🦥 Full fine-tuning mode: fast_inference disabled (not supported)")
    
    # === MULTI-GPU: Последовательная загрузка моделей ===
    # При DDP каждый процесс загружает свою копию модели.
    # vLLM + компиляция могут конфликтовать при одновременной загрузке.
    # Решение: загружаем модели ПОСЛЕДОВАТЕЛЬНО — один за одним.
    
    if is_multi_gpu and torch.distributed.is_initialized():
        # Каждый rank ждёт своей очереди
        for loading_rank in range(world_size):
            if rank == loading_rank:
                logger.info(f"🦥 Rank {rank}/{world_size}: Loading model NOW...")
                model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
                logger.info(f"🦥 Rank {rank}/{world_size}: Model loaded successfully!")
            # Синхронизация после каждой загрузки
            torch.distributed.barrier()
            if rank != loading_rank:
                logger.info(f"🦥 Rank {rank}: Rank {loading_rank} finished loading, continuing...")
        logger.info(f"🦥 Rank {rank}: All {world_size} models loaded!")
    else:
        # Single GPU — просто загружаем
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    
    # === Добавляем LoRA адаптеры ===
    if use_lora:
        # Unsloth рекомендует lora_alpha = lora_r * 2 для ускорения обучения
        # Если пользователь не задал alpha, используем r*2
        if lora_alpha is None or lora_alpha == lora_r:
            lora_alpha = lora_r * 2
            logger.info(f"🦥 Using optimized lora_alpha = lora_r * 2 = {lora_alpha} (speeds up training)")
        
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
            lora_dropout=lora_dropout,  # Unsloth требует 0 для оптимизаций
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth smart checkpointing
            random_state=42,
            max_seq_length=max_seq_length,
        )
    
    # Барьер после добавления LoRA адаптеров
    if is_multi_gpu and torch.distributed.is_initialized():
        logger.info(f"🦥 Rank {rank}: LoRA adapters added, syncing all processes...")
        torch.distributed.barrier()
    
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
    
    # ===== System prompt на основе reasoning_format (как в обычном бэкенде!) =====
    reasoning_format = config.get("grpo_reasoning_format", "deepseek")
    custom_system_prompt = config.get("grpo_system_prompt", None)
    
    # Формируем system prompt в зависимости от формата (как в rollout.py build_reasoning_prompt)
    if custom_system_prompt:
        # Если пользователь задал кастомный промпт в UI — используем его
        system_prompt = custom_system_prompt
    elif reasoning_format == "deepseek":
        # Формат DeepSeek с <think> тегами
        system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>"""
    elif reasoning_format == "simple":
        # Простой формат с <reasoning> тегами
        system_prompt = """Отвечай строго в формате:
<reasoning>
(Шаги решения)
</reasoning>
<answer>
(Короткий итоговый ответ)
</answer>"""
    elif reasoning_format == "russian":
        # Русский формат
        system_prompt = """Ты — умный помощник. Решай задачи пошагово.
Сначала подробно рассуждай в теге <reasoning>...</reasoning>,
затем дай краткий ответ в теге <answer>...</answer>.

Пример:
<reasoning>
Дано: ...
Нужно найти: ...
Решение: ...
</reasoning>
<answer>
42
</answer>"""
    elif reasoning_format == "gsm8k":
        # Формат GSM8K с ####
        system_prompt = """You are a helpful assistant that solves math problems step by step.
Show your reasoning process, then provide the final numerical answer after ####.

Example format:
Let me solve this step by step.
Step 1: ...
Step 2: ...
Therefore, the answer is X.
#### X"""
    else:
        # Fallback: используем дефолтный промпт для deepseek
        system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>"""
    
    logger.info(f"🦥 Reasoning format: {reasoning_format}")
    logger.info(f"🦥 System prompt preview: {system_prompt[:100]}...")
    
    def format_for_grpo(example):
        """Форматирует пример для GRPO с chat messages (как в примере Unsloth)."""
        question = example.get(prompt_col, "")
        raw_answer = example.get(answer_col, "")
        
        # TRL ожидает prompt как список chat messages!
        # Именно так работает в примере Unsloth qwen3_grpo.py
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        # Для GSM8K извлекаем число после ####, для остальных датасетов берём as-is
        if "####" in str(raw_answer):
            answer = str(raw_answer).split("####")[-1].strip()
        else:
            answer = str(raw_answer).strip()
        
        return {
            "prompt": prompt,  # Chat messages!
            "answer": answer,  # TRL передаёт это в reward_funcs как kwarg
        }
    
    dataset = dataset.map(format_for_grpo, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x.get("prompt", [])) > 0)
    
    # DEBUG: Проверяем структуру датасета
    logger.info(f"🦥 Dataset prepared: {len(dataset)} examples")
    logger.info(f"🦥 Dataset columns after formatting: {dataset.column_names}")
    if len(dataset) > 0:
        first_prompt = dataset[0]['prompt']
        prompt_preview = first_prompt[-1]["content"][:80] if isinstance(first_prompt, list) else str(first_prompt)[:80]
        logger.info(f"🦥 First example: prompt={prompt_preview}..., answer={dataset[0].get('answer', 'MISSING!')}")
    
    metrics_logger.update(
        status="training",
        num_train_examples=len(dataset),
        backend="unsloth",
    )
    
    # === GRPO Config ===
    output_dir = Path(config.get("output_dir", "out/unsloth_grpo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Параметры из UI (с правильными именами!)
    batch_size = config.get("grpo_train_batch_size", config.get("batch_size", 1))
    gradient_accumulation = config.get("gradient_accumulation", 4)
    num_generations = config.get("grpo_group_size", config.get("grpo_num_generations", 8))
    max_new_tokens = config.get("grpo_max_new_tokens", 512)
    learning_rate = config.get("grpo_learning_rate", config.get("learning_rate", 5e-5))
    temperature = config.get("grpo_temperature", 0.7)
    kl_weight = config.get("grpo_kl_weight", config.get("grpo_beta", 0.1))
    clip_eps = config.get("grpo_clip_eps_low", 0.2)
    algorithm = config.get("grpo_algorithm", "grpo")  # grpo, dapo, dr_grpo
    
    logger.info(f"🦥 GRPO Config from UI:")
    logger.info(f"   learning_rate={learning_rate}, batch_size={batch_size}, grad_accum={gradient_accumulation}")
    logger.info(f"   num_generations={num_generations}, temperature={temperature}, kl_weight={kl_weight}")
    logger.info(f"   algorithm={algorithm}, clip_eps={clip_eps}")
    
    # Базовые параметры GRPOConfig (совместимо с разными версиями trl)
    grpo_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=config.get("epochs", 1),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=config.get("weight_decay", 0.001),  # Как в примере Unsloth
        warmup_steps=config.get("warmup_steps", 50),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type=config.get("lr_schedule", "linear"),  # linear как в примере Unsloth
        logging_steps=config.get("log_every", 1),  # Чаще логируем для наглядности
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
        temperature=temperature,
        beta=kl_weight,  # KL coefficient
        
    )
    
    # Loss type: grpo, bnpo (для DAPO используется bnpo), dr_grpo
    if algorithm == "dapo":
        grpo_kwargs["loss_type"] = "dapo"  # или "bnpo" для DAPO
    elif algorithm == "dr_grpo":
        grpo_kwargs["loss_type"] = "dr_grpo"
    else:
        grpo_kwargs["loss_type"] = "grpo"  # дефолт
    
    # Проверяем какие параметры поддерживаются в GRPOConfig
    import inspect
    grpo_sig = inspect.signature(GRPOConfig.__init__)
    
    # Epsilon для PPO clipping
    if "epsilon" in grpo_sig.parameters:
        grpo_kwargs["epsilon"] = clip_eps
    elif "epsilon_low" in grpo_sig.parameters:
        grpo_kwargs["epsilon_low"] = clip_eps
        grpo_kwargs["epsilon_high"] = config.get("grpo_clip_eps_high", clip_eps)
    
    # max_prompt_length и max_completion_length (как в примере Unsloth)
    # Важно: max_prompt_length + max_completion_length <= max_seq_length
    max_prompt_length = max_seq_length // 2  # Половина для промпта
    
    if "max_prompt_length" in grpo_sig.parameters:
        grpo_kwargs["max_prompt_length"] = max_prompt_length
    
    if "max_completion_length" in grpo_sig.parameters:
        grpo_kwargs["max_completion_length"] = min(max_new_tokens, max_seq_length - max_prompt_length)
    elif "max_new_tokens" in grpo_sig.parameters:
        grpo_kwargs["max_new_tokens"] = max_new_tokens
    
    # vLLM SamplingParams (как в примере Unsloth qwen3_grpo.py)
    if "vllm_sampling_params" in grpo_sig.parameters:
        try:
            from vllm import SamplingParams
            vllm_sampling_params = SamplingParams(
                min_p=0.1,
                top_p=1.0,
                top_k=-1,
                seed=42,
                stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
                include_stop_str_in_output=True,
            )
            grpo_kwargs["vllm_sampling_params"] = vllm_sampling_params
            logger.info("🦥 Added vLLM SamplingParams to GRPOConfig")
        except ImportError:
            logger.debug("vLLM not available, skipping sampling params")
    
    grpo_config = GRPOConfig(**grpo_kwargs)
    
    # === Reward Function с логированием сэмплов ===
    # Получаем reward rules из конфига UI
    reward_rules = config.get("grpo_reward_rules", [])
    reasoning_format = config.get("grpo_reasoning_format", "reasoning_answer")
    
    import re
    import json
    
    # === Хелпер для извлечения content из TRL completions ===
    # TRL передаёт completions как список chat messages: [{"role": "assistant", "content": "..."}]
    # А не как простые строки!
    def _get_content(item) -> str:
        """Извлекает текст из completion (может быть строка или chat message)."""
        if isinstance(item, str):
            return item
        elif isinstance(item, list) and len(item) > 0:
            # [{"role": "assistant", "content": "..."}]
            if isinstance(item[0], dict) and "content" in item[0]:
                return item[0]["content"]
            elif isinstance(item[0], str):
                return item[0]
        elif isinstance(item, dict) and "content" in item:
            return item["content"]
        return str(item)
    
    def _get_question(prompt) -> str:
        """Извлекает вопрос из prompt (может быть строка или chat messages).
        
        TRL передаёт prompts в формате chat messages:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "вопрос"}]
        """
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list) and len(prompt) > 0:
            # Ищем последний user message (как в примере Unsloth: prompts[0][-1]["content"])
            for msg in reversed(prompt):
                if isinstance(msg, dict):
                    if msg.get("role") == "user" and "content" in msg:
                        return msg["content"]
            # Fallback: берём content последнего сообщения
            if isinstance(prompt[-1], dict) and "content" in prompt[-1]:
                return prompt[-1]["content"]
        elif isinstance(prompt, dict) and "content" in prompt:
            return prompt["content"]
        return str(prompt)
    
    # Глобальные переменные для логирования (как в оригинальном Unsloth примере)
    global UNSLOTH_PRINTED_TIMES
    UNSLOTH_PRINTED_TIMES = 0
    
    # Интервал логирования
    log_completions = config.get("grpo_log_completions", True)
    completion_log_interval = config.get("grpo_completion_log_interval", 10)
    
    # UI run dir для samples.jsonl
    ui_run_dir = config.get("ui_run_dir")
    samples_file = output_dir / "samples.jsonl"
    ui_samples_file = Path(ui_run_dir) / "samples.jsonl" if ui_run_dir else None
    
    def _save_sample_to_file(
        step: int, 
        prompt_messages: List,  # Chat messages формат
        completion: str, 
        reward: float, 
        reference_answer: str = "", 
        extracted: str = "",
        all_completions: Optional[List[str]] = None,
        all_rewards: Optional[List[float]] = None,
    ):
        """Сохраняет сэмпл в samples.jsonl для UI (как в обычном бэкенде)."""
        try:
            # Форматируем промпт для отображения
            if isinstance(prompt_messages, list):
                # Применяем chat template для красивого отображения
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        prompt_messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except:
                    # Fallback: просто конкатенируем сообщения
                    formatted_prompt = "\n".join(
                        f"[{m.get('role', 'unknown')}]: {m.get('content', '')}"
                        for m in prompt_messages if isinstance(m, dict)
                    )
            else:
                formatted_prompt = str(prompt_messages)
            
            # Формируем full_texts (промпт + completion) как в обычном бэкенде
            completions_list = all_completions if all_completions else [completion]
            rewards_list = all_rewards if all_rewards else [reward]
            
            full_texts = [formatted_prompt + comp for comp in completions_list]
            
            sample_entry = {
                "step": step,
                "prompt": formatted_prompt,
                "reference_answer": reference_answer,
                "completions": completions_list,
                "full_texts": full_texts,  # Промпт + completion для UI отображения
                "rewards": rewards_list,
                "extracted": extracted,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Сохраняем в output_dir/samples.jsonl
            with open(samples_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")
            
            # Дублируем в UI run_dir
            if ui_samples_file:
                ui_samples_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ui_samples_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")
            
            logger.debug(f"📝 Saved sample to samples.jsonl (step={step})")
        except Exception as e:
            logger.warning(f"⚠️ Could not save sample: {e}")
    
    def _print_sample(
        step: int, 
        prompt_messages: List,  # Chat messages формат
        answer: str, 
        response: str, 
        extracted: str, 
        reward: float
    ):
        """Выводит сэмпл в консоль с полным промптом (как в обычном бэкенде)."""
        # Форматируем промпт для отображения
        if isinstance(prompt_messages, list):
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    prompt_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except:
                formatted_prompt = "\n".join(
                    f"[{m.get('role', 'unknown')}]: {m.get('content', '')}"
                    for m in prompt_messages if isinstance(m, dict)
                )
        else:
            formatted_prompt = str(prompt_messages)
        
        print("\n" + "=" * 80)
        print(f"📝 SAMPLE AT STEP {step}")
        print("=" * 80)
        print(f"\n{'─'*40} PROMPT {'─'*40}")
        print(formatted_prompt[:1000])
        if len(formatted_prompt) > 1000:
            print(f"... (truncated, total {len(formatted_prompt)} chars)")
        print(f"\n{'─'*40} REFERENCE ANSWER {'─'*40}")
        print(f"✅ {answer}")
        print(f"\n{'─'*40} MODEL RESPONSE {'─'*40}")
        print(response[:1500])
        if len(response) > 1500:
            print(f"... (truncated, total {len(response)} chars)")
        print(f"\n{'─'*40} EVALUATION {'─'*40}")
        print(f"🎯 Extracted: {extracted}")
        print(f"⭐ Reward: {reward:.4f}")
        print("=" * 80 + "\n")
    
    def create_trl_reward_fn_from_rules(rules: List[Dict], reasoning_fmt: str):
        """
        Создаёт TRL-совместимую reward функцию из UI правил (Reward Designer).
        
        Использует UniversalRuleReward из обычного бэкенда для полной совместимости!
        
        TRL вызывает: reward_fn(prompts, completions, answer, **kwargs)
        """
        from homellm.training.rl.rewards.base import UniversalRuleReward
        
        # Фильтруем только активные правила
        active_rules = [r for r in rules if r.get("enabled", True)]
        if not active_rules:
            return []
        
        # Создаём UniversalRuleReward из правил
        universal_reward = UniversalRuleReward.from_config(active_rules)
        logger.info(f"🦥 Created UniversalRuleReward from {len(active_rules)} UI rules")
        
        for rule in active_rules:
            logger.info(f"🦥   - {rule.get('name', 'unnamed')} (weight={rule.get('weight', 1.0)})")
        
        # Создаём TRL-совместимую обёртку
        def ui_rules_reward_fn(
            prompts: List,
            completions: List,
            answer: Optional[List[str]] = None,
            **kwargs
        ) -> List[float]:
            """TRL-совместимая обёртка для UniversalRuleReward."""
            rewards = []
            
            # Получаем ground truth
            gt_list = answer or []
            if isinstance(gt_list, str):
                gt_list = [gt_list]
            
            for i, completion in enumerate(completions):
                # Извлекаем текст из chat messages
                response = _get_content(completion)
                
                # Получаем промпт
                prompt_text = _get_question(prompts[i]) if i < len(prompts) else ""
                
                # Получаем reference answer
                reference = gt_list[i] if i < len(gt_list) else ""
                
                # Вызываем UniversalRuleReward
                reward = universal_reward(
                    completion=response,
                    reference_answer=str(reference),
                    prompt=prompt_text,
                )
                rewards.append(reward)
            
            return rewards
        
        # Возвращаем как список с одной функцией (TRL ожидает список)
        return [ui_rules_reward_fn]
    
    if reward_rules and len(reward_rules) > 0:
        # Используем правила из UI
        logger.info(f"🦥 Creating reward functions from {len(reward_rules)} UI rules")
        reward_fn = create_trl_reward_fn_from_rules(reward_rules, reasoning_format)
        if not reward_fn:
            logger.warning("🦥 No valid reward rules, using defaults")
            reward_fn = None
    
    if reward_fn is None:
        # Дефолтные reward функции (адаптируются под reasoning_format)
        
        # Паттерны для проверки формата в зависимости от reasoning_format
        if reasoning_format in ("deepseek", "simple", "russian"):
            format_pattern = re.compile(r'<answer>\s*.+?\s*</answer>', re.DOTALL | re.IGNORECASE)
            format_name = "<answer>...</answer>"
            use_answer_tags = True
        else:  # gsm8k или другие
            format_pattern = re.compile(r'####\s*-?\d+')
            format_name = "#### <number>"
            use_answer_tags = False
        
        def default_format_fn(
            prompts: List,
            completions: List,
            answer: Optional[List[str]] = None,
            **kwargs
        ) -> List[float]:
            """Проверяет соответствие формату ответа."""
            rewards = []
            for completion in completions:
                response = _get_content(completion)
                if format_pattern.search(response):
                    rewards.append(1.0)  # Формат соблюдён
                else:
                    # Частичный reward если есть хоть какой-то ответ
                    if use_answer_tags:
                        if "<answer>" in response.lower():
                            rewards.append(0.3)
                        else:
                            rewards.append(0.0)
                    else:
                        if "####" in response:
                            rewards.append(0.3)
                        else:
                            rewards.append(0.0)
            return rewards
        
        def default_correctness_fn(
            prompts: List,
            completions: List,
            answer: Optional[List[str]] = None,
            **kwargs
        ) -> List[float]:
            """Проверяет правильность ответа (сравнение с ground truth)."""
            rewards = []
            gt_list = answer or []
            
            for i, completion in enumerate(completions):
                response = _get_content(completion)
                extracted = extract_answer_from_response(response)
                true_answer = gt_list[i] if i < len(gt_list) else None
                
                if extracted is None:
                    rewards.append(-1.0)  # Штраф за отсутствие ответа
                    continue
                
                if true_answer is None:
                    rewards.append(0.0)
                    continue
                
                try:
                    # Пробуем числовое сравнение
                    true_val = float(str(true_answer).strip().replace(",", ""))
                    guess_val = float(str(extracted).strip().replace(",", ""))
                    if guess_val == true_val:
                        rewards.append(3.0)  # Правильный ответ
                    else:
                        # Частичный reward за близкий ответ
                        ratio = guess_val / true_val if true_val != 0 else 0
                        if 0.9 <= ratio <= 1.1:
                            rewards.append(1.0)
                        else:
                            rewards.append(-0.5)
                except (ValueError, TypeError):
                    # Строковое сравнение
                    if str(extracted).strip().lower() == str(true_answer).strip().lower():
                        rewards.append(3.0)
                    else:
                        rewards.append(-0.5)
            
            return rewards
        
        reward_fn = [default_format_fn, default_correctness_fn]
        logger.info(f"🦥 Using default reward functions for format={reasoning_format} (pattern: {format_name})")
    
    # === Wrapper для логирования сэмплов (как в оригинальном Unsloth примере) ===
    # Создаём последнюю reward функцию которая логирует промпт/ответ/reward
    
    # Паттерны для извлечения ответа в зависимости от формата
    # reasoning_format уже определён выше
    if reasoning_format in ("deepseek", "simple", "russian"):
        # Формат с <answer>...</answer> тегами
        answer_tag_pattern = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.DOTALL | re.IGNORECASE)
        use_hash_format = False
        logger.info(f"🦥 Using <answer> tag pattern for extraction (format={reasoning_format})")
    else:
        # Формат с #### (GSM8K style)
        answer_tag_pattern = None
        use_hash_format = True
        logger.info(f"🦥 Using #### pattern for extraction (format={reasoning_format})")
    
    answer_hash_pattern = re.compile(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)')
    
    def extract_answer_from_response(response: str) -> Optional[str]:
        """Извлекает ответ из response в зависимости от формата."""
        # Сначала пробуем <answer> теги (если формат поддерживает)
        if answer_tag_pattern:
            match = answer_tag_pattern.search(response)
            if match:
                return match.group(1).strip()
        
        # Затем пробуем #### формат
        match = answer_hash_pattern.search(response)
        if match:
            return match.group(1).replace(",", "")
        
        # Fallback: ищем после ####
        if "####" in response:
            after = response.split("####")[-1].strip()
            numbers = re.findall(r'-?\d+(?:\.\d+)?', after)
            if numbers:
                return numbers[0]
        
        # Последний fallback: последнее число в тексте
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        return numbers[-1] if numbers else None
    
    def logging_reward_fn(
        prompts: List,
        completions: List,
        answer: Optional[List[str]] = None,  # ground truth от TRL (позиционный!)
        **kwargs
    ) -> List[float]:
        """Reward функция с логированием (как check_numbers в Unsloth примере)."""
        global UNSLOTH_PRINTED_TIMES
        
        # DEBUG: Логируем что получили от TRL (только на первом вызове)
        if UNSLOTH_PRINTED_TIMES == 0:
            logger.info(f"🔍 DEBUG: answer count={len(answer) if answer else 0}, completions count={len(completions)}")
        
        # Извлекаем текст из completions (TRL передаёт как chat messages!)
        responses = [_get_content(c) for c in completions]
        
        # Извлекаем ответы из responses (используем универсальную функцию)
        extracted_responses = [extract_answer_from_response(r) for r in responses]
        
        # Получаем ground truth
        # TRL передаёт answer как ПОЗИЦИОННЫЙ аргумент (не через kwargs!)
        gt_list = answer or []
        
        # Убедимся что это список
        if gt_list is None:
            gt_list = []
        elif isinstance(gt_list, str):
            gt_list = [gt_list]
        
        # Эта функция только для ЛОГИРОВАНИЯ, не влияет на итоговый reward
        # Основные rewards идут из UI-заданных функций (format_check, exact_match и т.д.)
        scores = [0.0] * len(completions)  # Всегда возвращаем 0 — не влияем на обучение
        
        # Вычисляем "информационный" reward только для отображения в логах
        display_rewards = []
        for i, (guess, response) in enumerate(zip(extracted_responses, responses)):
            true_answer = gt_list[i] if i < len(gt_list) else None
            
            if guess is None:
                display_rewards.append(0.0)
                continue
            
            try:
                if true_answer is not None:
                    true_val = float(str(true_answer).strip().replace(",", ""))
                    guess_val = float(guess)
                    display_rewards.append(1.0 if guess_val == true_val else 0.0)
                else:
                    display_rewards.append(0.0)
            except:
                display_rewards.append(0.0)
        
        # Логируем периодически (как в оригинальном Unsloth примере)
        if log_completions and UNSLOTH_PRINTED_TIMES % completion_log_interval == 0:
            # Получаем первый промпт (chat messages)
            first_prompt = prompts[0] if prompts else []
            
            gt_str = str(gt_list[0]) if gt_list else "N/A"
            response_text = responses[0] if responses else "N/A"
            extracted = extracted_responses[0] if extracted_responses else "N/A"
            # Используем display_reward для логирования (informational only)
            display_reward = display_rewards[0] if display_rewards else 0.0
            
            # Выводим в консоль с полным промптом
            _print_sample(
                step=UNSLOTH_PRINTED_TIMES,
                prompt_messages=first_prompt,  # Передаём chat messages
                answer=gt_str,
                response=response_text,
                extracted=str(extracted),
                reward=display_reward if isinstance(display_reward, float) else 0.0,
            )
            
            # Сохраняем в файл для UI (с полным промптом и всеми completions)
            _save_sample_to_file(
                step=UNSLOTH_PRINTED_TIMES,
                prompt_messages=first_prompt,  # Chat messages
                completion=response_text,
                reward=display_reward if isinstance(display_reward, float) else 0.0,
                reference_answer=gt_str,
                extracted=str(extracted),
                all_completions=responses,  # Все completions из batch
                all_rewards=display_rewards,  # Все rewards
            )
        
        UNSLOTH_PRINTED_TIMES += 1
        return scores
    
    # Добавляем logging_reward_fn к списку
    if isinstance(reward_fn, list):
        reward_fn.append(logging_reward_fn)
    else:
        reward_fn = [reward_fn, logging_reward_fn] if reward_fn else [logging_reward_fn]
    
    logger.info(f"🦥 Added logging reward function (log_every={completion_log_interval} steps)")
    
    # === Патч для DDP: monkeypatch для unwrap model ===
    # Unsloth использует model.config напрямую, но при DDP модель обёрнута
    # Создаём обёртку которая автоматически unwrap'ит модель
    if world_size > 1:
        from accelerate.utils import extract_model_from_parallel
        
        # Сохраняем оригинальный getattr для DDP
        original_ddp_getattr = None
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            original_ddp_getattr = DDP.__getattr__
            
            def patched_getattr(self, name):
                if name == 'config' and hasattr(self, 'module'):
                    return getattr(self.module, 'config', None)
                return original_ddp_getattr(self, name)
            
            DDP.__getattr__ = patched_getattr
            logger.info("🦥 Patched DDP.__getattr__ for .config access")
        except Exception as e:
            logger.warning(f"⚠️ Could not patch DDP: {e}")
    
    # === Trainer ===
    # GRPOTrainer — используем processing_class как в примере Unsloth
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,  # Новый API TRL (не tokenizer!)
        train_dataset=dataset,
        args=grpo_config,
        reward_funcs=reward_fn,
    )
    
    # === Callback для метрик ===
    from transformers import TrainerCallback
    
    class MetricsCallback(TrainerCallback):
        def __init__(self, metrics_logger, start_time, total_steps, tokenizer):
            self.metrics_logger = metrics_logger
            self.start_time = start_time
            self.total_steps = total_steps
            self.tokenizer = tokenizer
            self.last_log_step = -1
            self.sample_log_interval = 50  # Логировать сэмплы каждые N шагов
        
        def on_train_begin(self, args, state, control, **kwargs):
            self.metrics_logger.update(
                status="training",
                total_steps=self.total_steps,
                backend="unsloth",
            )
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            
            step = state.global_step
            
            # Избегаем дублирования логов для одного шага
            if step == self.last_log_step:
                return
            self.last_log_step = step
            
            loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", 0.0)
            
            # GRPO специфичные метрики
            reward = logs.get("reward", logs.get("rewards/mean", 0.0))
            kl = logs.get("kl", logs.get("kl_divergence", 0.0))
            
            # Дополнительные метрики из TRL
            policy_loss = logs.get("loss/policy", logs.get("policy_loss", None))
            value_loss = logs.get("loss/value", logs.get("value_loss", None))
            entropy = logs.get("loss/entropy", logs.get("entropy", None))
            
            elapsed = time.time() - self.start_time
            samples_per_sec = step / elapsed if elapsed > 0 else 0
            
            # Логируем шаг со всеми метриками
            self.metrics_logger.log_step(
                step=step,
                loss=loss,
                lr=lr,
                samples_per_sec=samples_per_sec,
                reward=reward,
                kl=kl,
            )
            
            # Дополнительные метрики для UI
            eta_seconds = (self.total_steps - step) / samples_per_sec if samples_per_sec > 0 else 0
            self.metrics_logger.update(
                elapsed_time=elapsed,
                eta_seconds=eta_seconds,
                samples_per_second=samples_per_sec,
            )
            
            # Красивый лог в консоль
            log_msg = f"🦥 Step {step}/{self.total_steps} | Loss: {loss:.4f} | Reward: {reward:.4f} | KL: {kl:.4f} | LR: {lr:.2e}"
            if policy_loss is not None:
                log_msg += f" | Policy: {policy_loss:.4f}"
            logger.info(log_msg)
            
            # Показываем completions из логов если есть
            completions = logs.get("completions", None)
            if completions and step % self.sample_log_interval == 0:
                self._log_sample_completions(step, completions)
        
        def _log_sample_completions(self, step, completions):
            """Логирует примеры сгенерированных ответов."""
            if not completions:
                return
            
            logger.info("=" * 80)
            logger.info(f"📝 Sample completions at step {step}:")
            logger.info("=" * 80)
            
            # Показываем до 3 примеров
            samples_to_show = completions[:3] if isinstance(completions, list) else [completions]
            
            for i, completion in enumerate(samples_to_show):
                if isinstance(completion, dict):
                    prompt = completion.get("prompt", "N/A")[:200]
                    response = completion.get("response", completion.get("completion", "N/A"))[:500]
                    reward = completion.get("reward", "N/A")
                else:
                    prompt = "N/A"
                    response = str(completion)[:500]
                    reward = "N/A"
                
                logger.info(f"\n--- Sample {i+1} ---")
                logger.info(f"Prompt: {prompt}...")
                logger.info(f"Response: {response}...")
                logger.info(f"Reward: {reward}")
            
            logger.info("=" * 80)
        
        def on_save(self, args, state, control, **kwargs):
            ckpt_path = str(output_dir / f"checkpoint-{state.global_step}")
            self.metrics_logger.log_checkpoint(ckpt_path)
        
        def on_train_end(self, args, state, control, **kwargs):
            total_time = time.time() - self.start_time
            self.metrics_logger.update(
                status="completed",
                total_training_time=total_time,
                final_step=state.global_step,
            )
    
    # Рассчитываем total_steps
    total_steps = grpo_config.max_steps if grpo_config.max_steps > 0 else (
        len(dataset) // (grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps * max(1, world_size))
        * grpo_config.num_train_epochs
    )
    
    trainer.add_callback(MetricsCallback(metrics_logger, time.time(), total_steps, tokenizer))
    
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
    
    # merge_lora по умолчанию True — для единообразия с models-at-home backend
    # final_model/ содержит merged модель для удобного inference
    merge_lora = config.get("merge_lora", True)
    
    if use_lora:
        if merge_lora:
            # Сначала сохраняем LoRA адаптеры отдельно (до merge)
            lora_dir = output_dir / "lora_adapters"
            lora_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(lora_dir)
            tokenizer.save_pretrained(lora_dir)
            logger.info(f"🦥 Saved LoRA adapters to {lora_dir}")
            
            # Merge LoRA в базовую модель и сохранить в final_model/
            logger.info("🦥 Merging LoRA adapters into base model...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            logger.info(f"🦥 Saved merged model to {final_dir}")
        else:
            # Сохранить только LoRA адаптеры
            model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            logger.info(f"🦥 Saved LoRA adapters to {final_dir}")
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
