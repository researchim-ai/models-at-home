"""
Фоновый worker для тренировки модели.
Записывает метрики в JSON файл для чтения Streamlit приложением.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)

from homellm.models.adapters import resolve_adapter
from homellm.training.pretrain import StreamingTextDataset
from homellm.training.sft import SFTDataset

logger = logging.getLogger(__name__)


def get_gpu_stats():
    """
    Получить статистику GPU через nvidia-smi (работает правильно при DDP).
    
    ВАЖНО: Фильтрует и ремапит GPU по CUDA_VISIBLE_DEVICES, чтобы показывать
    только выбранные GPU с правильными индексами (0..N-1).
    """
    gpu_stats = []
    
    # Получаем список видимых GPU из окружения
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = None
    if visible:
        try:
            visible_ids = [int(x.strip()) for x in visible.split(",") if x.strip() != ""]
        except Exception:
            visible_ids = None
    
    try:
        import subprocess
        # Получаем данные со всех GPU через nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu_id = int(parts[0])
                        memory_used = float(parts[1]) / 1024  # MiB -> GiB
                        memory_total = float(parts[2]) / 1024
                        utilization = int(parts[3]) if parts[3].isdigit() else None
                        
                        gpu_stats.append({
                            "id": gpu_id,
                            "memory_used_gb": round(memory_used, 2),
                            "memory_total_gb": round(memory_total, 2),
                            "memory_percent": round(memory_used / memory_total * 100, 1) if memory_total > 0 else 0,
                            "utilization": utilization,
                        })
            
            # Фильтруем и ремапим по CUDA_VISIBLE_DEVICES
            if visible_ids is not None:
                # Оставляем только видимые GPU
                gpu_stats = [g for g in gpu_stats if g["id"] in visible_ids]
                # Ремапим физические ID -> логические (0..N-1)
                remap = {phys: i for i, phys in enumerate(visible_ids)}
                for g in gpu_stats:
                    g["id"] = remap.get(g["id"], g["id"])
    except Exception:
        # Fallback на torch.cuda если nvidia-smi недоступен
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    
                    gpu_stats.append({
                        "id": i,
                        "memory_used_gb": round(memory_allocated, 2),
                        "memory_total_gb": round(memory_total, 2),
                        "memory_percent": round(memory_allocated / memory_total * 100, 1),
                        "utilization": None,
                    })
                except:
                    pass
    
    return gpu_stats


class MetricsLogger:
    """Логгер метрик в JSON файл для визуализации.
    
    ВАЖНО: В Multi-GPU режиме должен использоваться только на main process (rank 0),
    чтобы избежать гонок при записи в один файл.
    """
    
    def __init__(self, log_path: Path, enabled: bool = True):
        self.log_path = log_path
        self.enabled = enabled
        self.start_timestamp = time.time()
        self.metrics = {
            "status": "initializing",
            "start_time": datetime.now().isoformat(),
            "current_step": 0,
            "total_steps": 0,
            "epoch": 0,
            "loss_history": [],
            "lr_history": [],
            "steps_history": [],
            "current_loss": 0.0,
            "current_lr": 0.0,
            "samples_per_second": 0.0,
            "eta_seconds": 0,
            "error": None,
            "checkpoints": [],
            "gpu_stats": [],
            "current_val_loss": None,
            "val_loss_history": [],
            "val_steps_history": [],
        }
        self._save()
    
    def _save(self):
        """Атомарная запись через временный файл для избежания гонок."""
        if not self.enabled:
            return
        
        # Создаём родительскую директорию если нужно
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Атомарная запись через временный файл
        tmp_path = self.log_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(self.metrics, f, indent=2)
            os.replace(tmp_path, self.log_path)
        except Exception as e:
            # Если не удалось записать, пытаемся удалить tmp файл
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except:
                pass
            # Логируем ошибку, но не падаем
            logger.warning(f"Failed to save metrics: {e}")
    
    def update(self, **kwargs):
        if not self.enabled:
            return
        self.metrics.update(kwargs)
        self._save()
    
    def log_step(self, step: int, loss: float, lr: float, samples_per_sec: float = 0, step_time: float = 0):
        if not self.enabled:
            return
        
        self.metrics["current_step"] = step
        self.metrics["current_loss"] = loss
        self.metrics["current_lr"] = lr
        self.metrics["samples_per_second"] = samples_per_sec
        self.metrics["loss_history"].append(loss)
        self.metrics["lr_history"].append(lr)
        self.metrics["steps_history"].append(step)
        
        # Elapsed
        self.metrics["elapsed_seconds"] = time.time() - self.start_timestamp
        
        # GPU stats
        self.metrics["gpu_stats"] = get_gpu_stats()
        
        # ETA на основе времени шага
        if step > 0 and step_time > 0:
            remaining_steps = max(0, self.metrics["total_steps"] - step)
            self.metrics["eta_seconds"] = int(remaining_steps * step_time)
        
        self._save()
    
    def log_checkpoint(self, path: str):
        if not self.enabled:
            return
        
        self.metrics["checkpoints"].append({
            "path": path,
            "step": self.metrics["current_step"],
            "time": datetime.now().isoformat()
        })
        self._save()
    
    def log_eval(self, step: int, val_loss: float):
        """Логировать результат валидации."""
        if not self.enabled:
            return
        
        self.metrics["current_val_loss"] = val_loss
        self.metrics["val_loss_history"].append(val_loss)
        self.metrics["val_steps_history"].append(step)
        self._save()


def run_training(config: Dict[str, Any], metrics_path: Path):
    """Запуск тренировки с записью метрик."""
    
    # Mixed precision
    mixed_precision = config.get("mixed_precision", "no")
    stage = config.get("stage", "pretrain") # pretrain | continual_pretrain | sft
    
    # ВАЖНО: Убираем dispatch_batches=True, так как это вызывает проблемы с NoneType при broadcast
    # Вместо этого будем использовать явное шардирование внутри StreamingTextDataset (shard=True)
    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=None,
        even_batches=False, 
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation"],
        mixed_precision=mixed_precision,
        dataloader_config=dataloader_config,
    )
    
    # ВАЖНО: MetricsLogger только на main process для избежания гонок при Multi-GPU
    metrics = MetricsLogger(metrics_path, enabled=accelerator.is_main_process)
    
    try:
        # Определяем адаптер для работы с моделью
        adapter = resolve_adapter(config)
        logger.info(f"Using adapter: {adapter.__class__.__name__}")
        
        metrics.update(status="loading_tokenizer", stage=stage)
        
        # Загружаем токенизатор через адаптер
        tokenizer_path = config.get("tokenizer_path")
        if not tokenizer_path and stage in ("sft", "continual_pretrain") and config.get("base_model_path"):
            # Для SFT/continual используем токенизатор из базовой модели
            tokenizer_path = config["base_model_path"]
        elif not tokenizer_path:
            # Для pretrain используем model_id если указан, иначе gpt2
            tokenizer_path = config.get("model_id", "gpt2")
        
        tokenizer = adapter.load_tokenizer(tokenizer_path, trust_remote_code=True)
        tokenizer = adapter.prepare_tokenizer(tokenizer)  # pad_token = eos_token
        
        metrics.update(status=f"loading_dataset ({stage})")
        
        # ВАЖНО: Вычисляем val_ratio ДО создания train_dataset, чтобы избежать data leakage
        val_ratio = float(config.get("val_ratio", 0.0))
        eval_every = int(config.get("eval_every", 0) or 0)
        eval_batches = int(config.get("eval_batches", 20) or 20)
        
        # Держим holdout только если реально используем eval
        holdout_ratio = val_ratio if (val_ratio > 0.0 and eval_every > 0) else 0.0
        
        # Dataset Selection based on Stage
        # ВАЖНО: Шардирование делает accelerate.prepare() через DataLoaderShard/Dispatcher
        # Датасет только делает train/val split, но НЕ шардирует между процессами (shard=False)
        if stage == "sft":
            train_dataset = SFTDataset(
                config["data_path"],
                tokenizer,
                seq_len=config["seq_len"],
                sft_columns=config.get("sft_columns"),
                sft_template=config.get("sft_template"),
                num_replicas=accelerator.num_processes,  # ✅ Явное шардирование
                rank=accelerator.process_index,  # ✅ Явное шардирование
                split="train",
                val_ratio=holdout_ratio,
                shard=True,  # ✅ Включаем шардирование внутри датасета
            )
            
            # Получаем пример сформированного промпта для отображения (для SFT)
            try:
                if hasattr(train_dataset, 'get_sample_prompt'):
                    sample_prompt = train_dataset.get_sample_prompt(max_samples=20)
                    if sample_prompt:
                        metrics.update(sample_prompt=sample_prompt)
                        logger.info("Sample prompt saved to metrics")
            except Exception as e:
                logger.warning(f"Failed to get sample prompt: {e}")
            
            # Логируем статистику датасета (после первого прохода будет обновлена)
            if hasattr(train_dataset, 'num_replicas'):
                logger.info(f"Dataset initialized: num_replicas={train_dataset.num_replicas}, rank={train_dataset.rank}")
            
            # Для SFT data collator не нужен специальный masking, 
            # но нужно просто паддить тензоры в батче.
            # SFTDataset уже возвращает тензоры, DataCollatorForLanguageModeling с mlm=False подойдет
            # или default_data_collator если labels уже есть (а они есть в SFTDataset)
            from transformers import default_data_collator
            collate_fn = default_data_collator
        else:
            # Pretrain или Continual Pretrain - используем StreamingTextDataset
            train_dataset = StreamingTextDataset(
                config["data_path"], 
                tokenizer, 
                seq_len=config["seq_len"],
                num_replicas=accelerator.num_processes,  # ✅ Явное шардирование
                rank=accelerator.process_index,  # ✅ Явное шардирование
                split="train",
                val_ratio=holdout_ratio,
                shard=True,  # ✅ Включаем шардирование внутри датасета
            )
            
            # Получаем пример текста для отображения (для pretrain/continual_pretrain)
            try:
                if hasattr(train_dataset, 'get_sample_prompt'):
                    sample_prompt = train_dataset.get_sample_prompt(max_samples=20)
                    if sample_prompt:
                        metrics.update(sample_prompt=sample_prompt)
                        logger.info("Sample prompt saved to metrics")
            except Exception as e:
                logger.warning(f"Failed to get sample prompt: {e}")
            
            # ВАЖНО: Для pretrain используем кастомный collator, который маскирует по attention_mask,
            # а не по pad_token_id. Это критично, если pad_token = eos_token (EOS не должен маскироваться)
            def causal_lm_collator(batch):
                """Кастомный collator для pretrain, маскирует labels по attention_mask."""
                # Фильтруем битые примеры (если dataset вернул None)
                batch = [x for x in batch if x is not None]
                if not batch:
                     # Пустой батч не должен крашить accelerate, вернем пустые тензоры или пустой dict
                     # Но accelerate может не переварить пустой dict при broadcast, поэтому вернем dummy batch
                     # чтобы просто пропустить шаг
                     dummy = torch.zeros((1, 1), dtype=torch.long)
                     return {"input_ids": dummy, "attention_mask": dummy, "labels": dummy}

                input_ids = torch.stack([x["input_ids"] for x in batch])
                attention_mask = torch.stack([x["attention_mask"] for x in batch])
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100  # Маскируем только padding, не EOS
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            
            collate_fn = causal_lm_collator
        
        # Создаем validation loader если нужно
        val_loader = None
        
        if holdout_ratio > 0.0:
            # ВАЖНО: Шардирование делает accelerate.prepare() через DataLoaderShard/Dispatcher
            # Датасет только делает train/val split, но НЕ шардирует между процессами (shard=False)
            if stage == "sft":
                from homellm.training.sft import SFTDataset
                val_dataset = SFTDataset(
                    config["data_path"],
                    tokenizer,
                    seq_len=config["seq_len"],
                    sft_columns=config.get("sft_columns"),
                    sft_template=config.get("sft_template"),
                    num_replicas=accelerator.num_processes,  # ✅ Явное шардирование
                    rank=accelerator.process_index,  # ✅ Явное шардирование
                    split="val",
                    val_ratio=holdout_ratio,
                    shard=True,  # ✅ Включаем шардирование внутри датасета
                )
                from transformers import default_data_collator
                val_collate = default_data_collator
            else:
                # Используем уже импортированный StreamingTextDataset
                val_dataset = StreamingTextDataset(
                    config["data_path"],
                    tokenizer,
                    seq_len=config["seq_len"],
                    num_replicas=accelerator.num_processes,  # ✅ Явное шардирование
                    rank=accelerator.process_index,  # ✅ Явное шардирование
                    split="val",
                    val_ratio=holdout_ratio,
                    shard=True,  # ✅ Включаем шардирование внутри датасета
                )
                # Используем тот же кастомный collator для val, что и для train
                def causal_lm_collator(batch):
                    """Кастомный collator для pretrain, маскирует labels по attention_mask."""
                    batch = [x for x in batch if x is not None]
                    if not batch:
                        dummy = torch.zeros((1, 1), dtype=torch.long)
                        return {"input_ids": dummy, "attention_mask": dummy, "labels": dummy}

                    input_ids = torch.stack([x["input_ids"] for x in batch])
                    attention_mask = torch.stack([x["attention_mask"] for x in batch])
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100  # Маскируем только padding, не EOS
                    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                
                val_collate = causal_lm_collator
            
            # ВАЖНО: num_workers=0 для val, чтобы избежать дубликатов при shard=False
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                collate_fn=val_collate,
                num_workers=0,  # Без workers для валидации, чтобы избежать дубликатов
            )
            
        # ВАЖНО: num_workers=0 для IterableDataset, чтобы избежать дублирования данных
        # drop_last=True для стабильности DDP (все процессы выполнят одинаковое число итераций)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            collate_fn=collate_fn,
            num_workers=0,  # IterableDataset + num_workers>0 = дублирование данных
            drop_last=True,  # ✅ Критично для DDP: избегаем рассинхрона по числу шагов между процессами
        )
        
        metrics.update(status="building_model")
        
        # Загружаем модель через адаптер
        resume_from_checkpoint = None
        base_model_path = config.get("base_model_path")
        
        # Проверяем, является ли это accelerate checkpoint (для resume)
        # ВАЖНО: pytorch_model.bin.index.json может быть и у обычных шардированных HF-сейвов,
        # поэтому проверяем наличие accelerator_state.json - это точный признак accelerate checkpoint
        def is_accelerate_checkpoint(p: Path) -> bool:
            """Проверяет, является ли путь accelerate checkpoint'ом."""
            return (p / "accelerator_state.json").exists()
        
        if base_model_path:
            base_path = Path(base_model_path)
            is_checkpoint = is_accelerate_checkpoint(base_path)
            
            if is_checkpoint and stage == "continual_pretrain":
                resume_from_checkpoint = str(base_path)
                logger.info(f"Continual pretraining: will resume from accelerate checkpoint {base_path}")
            elif is_checkpoint and stage != "continual_pretrain":
                logger.warning(
                    f"Found accelerate checkpoint at {base_path}, but stage is {stage}. "
                    f"Resume from checkpoint is only supported for continual_pretrain. "
                    f"Will load as regular model instead."
                )
        
        # Загружаем модель через адаптер
        model, model_config = adapter.load_for_training(
            base_model_path=base_model_path,
            stage=stage,
            tokenizer=tokenizer,
            config=config,
            trust_remote_code=True,
        )
        
        # Подготавливаем модель для обучения (resize, LoRA, use_cache, etc.)
        model = adapter.prepare_for_training(model, tokenizer, config)
        
        # Подсчёт параметров
        num_params = sum(p.numel() for p in model.parameters())
        metrics.update(num_parameters=num_params)
        
        if config.get("grad_checkpoint", False):
            model.gradient_checkpointing_enable()
        
        # Scheduler - определяем количество шагов
        # Если указан max_steps - используем его
        if config.get("max_steps"):
            max_train_steps = config["max_steps"]
        else:
            # Пробуем получить длину датасета
            # ВАЖНО: len(train_dataset) возвращает полную длину для выбранного split
            # Но при DDP/FSDP/ZeRO accelerate.prepare() шардирует данные между процессами
            # Каждый процесс увидит примерно dataset_len / num_processes примеров
            try:
                dataset_len = len(train_dataset)  # Полная длина для выбранного split
                # Учитываем распределение по процессам (для IterableDataset это критично)
                per_proc_len = math.ceil(dataset_len / accelerator.num_processes)
                
                # batch_size здесь per-device, gradient_accumulation уже учтен
                steps_per_epoch = math.ceil(per_proc_len / config["batch_size"])
                num_update_steps_per_epoch = math.ceil(steps_per_epoch / config["gradient_accumulation"])
                max_train_steps = config["epochs"] * num_update_steps_per_epoch
            except (TypeError, AttributeError):
                # Для streaming dataset без __len__ - используем save_every * 10 как оценку
                max_train_steps = config.get("save_every", 5000) * 2
        
        metrics.update(total_steps=max_train_steps, max_steps_estimated=config.get("max_steps") is None)
        
        # ВАЖНО: Для LoRA/QLoRA оптимизатор должен брать только trainable параметры
        # Это критично для QLoRA, где базовые веса заморожены и огромные
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found in model. Check LoRA/QLoRA configuration.")
        
        logger.info(f"Optimizing {len(trainable_params)} trainable parameters "
                   f"(total: {sum(p.numel() for p in model.parameters())})")
        
        optimizer = torch.optim.AdamW(
            trainable_params,  # ✅ Только trainable параметры (критично для LoRA/QLoRA)
            lr=config["learning_rate"], 
            betas=(0.9, 0.95), 
            eps=1e-8,
            weight_decay=config.get("weight_decay", 0.1)
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=max_train_steps,
        )
        
        if val_loader is not None:
            model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_loader, val_loader, lr_scheduler
            )
        else:
            model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_loader, lr_scheduler
            )
        
        # Resume из checkpoint для continual_pretrain (если указан checkpoint)
        starting_step = 0
        if resume_from_checkpoint and stage == "continual_pretrain":
            try:
                logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
                accelerator.load_state(resume_from_checkpoint)
                
                # Пытаемся извлечь global_step из checkpoint
                checkpoint_meta = Path(resume_from_checkpoint) / "checkpoint_metadata.json"
                if checkpoint_meta.exists():
                    with open(checkpoint_meta) as f:
                        meta = json.load(f)
                        starting_step = meta.get("global_step", 0)
                else:
                    # Пытаемся извлечь из имени папки (например checkpoint_step1000)
                    import re
                    match = re.search(r'step(\d+)', str(resume_from_checkpoint))
                    if match:
                        starting_step = int(match.group(1))
                
                logger.info(f"Resumed from step {starting_step}")
                metrics.update(status="resumed", resumed_from_step=starting_step)
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint {resume_from_checkpoint}: {e}")
                logger.warning("Continuing without resume (starting from step 0)")
                metrics.update(status="resume_failed", resume_error=str(e))
        
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics.update(status="training")
        
        # ВАЖНО: Синхронизируем global_step с starting_step после resume
        # Это критично для правильной работы LR scheduler и checkpointing
        global_step = starting_step
        
        # Функция для валидации (использует eval_batches из замыкания)
        def evaluate(val_loader):
            """
            Выполнить валидацию на val_loader.
            
            ВАЖНО: val_loader зашардирован для всех процессов, поэтому каждый процесс
            видит свою часть validation данных. reduce() корректно усредняет loss по всем процессам.
            """
            model.eval()
            losses = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= eval_batches:
                        break
                    out = model(**batch)
                    loss = out.loss.detach()
                    # Усредняем loss по всем процессам (каждый процесс видит свою часть val данных)
                    loss = accelerator.reduce(loss, reduction="mean")
                    # Сохраняем только на main process, чтобы избежать дублирования
                    if accelerator.is_main_process:
                        losses.append(loss.item())
            model.train()
            if not losses:
                return None
            # Возвращаем средний loss (уже усредненный по процессам)
            return sum(losses) / len(losses) if losses else None
        
        # ВАЖНО: global_step уже инициализирован выше как starting_step (для resume)
        # Если resume не было, starting_step = 0, поэтому global_step = 0
        start_time = time.time()
        
        # Debug: проверка дублей между процессами (первые N шагов)
        # ВАЖНО: Включается только если в конфиге есть debug_check_duplicates=True
        debug_check_duplicates = config.get("debug_check_duplicates", False)
        debug_sample_ids = []  # Собираем sample_id для первых шагов
        debug_max_samples = 20  # Проверяем первые 20 примеров
        
        # Loss tracking:
        # - micro_* для проверки divergence на КАЖДОМ update-step
        # - log_* для сглаженного лога (усреднение по update-step-ам между логами)
        micro_loss_sum = 0.0
        micro_count = 0
        log_loss_sum = 0.0
        log_updates = 0
        update_start_time = time.time()

        training_complete = False
        
        for epoch in range(config["epochs"]):
            if training_complete:
                break
                
            metrics.update(epoch=epoch + 1)
            model.train()
            
            for step, batch in enumerate(train_loader):
                # Debug: проверка дублей (только первые N примеров)
                # ВАЖНО: При dispatch_batches=True нужно собирать хэши со всех процессов через gather
                if debug_check_duplicates and global_step < debug_max_samples:
                    # Для проверки используем hash от input_ids
                    if "input_ids" in batch:
                        # Берем первый пример из батча для проверки
                        sample_hash = hash(batch["input_ids"][0].cpu().numpy().tobytes())
                        debug_sample_ids.append((global_step, accelerator.process_index, sample_hash))
                
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    loss_val = loss.detach().float().item()
                    
                    # КРИТИЧНО: если loss NaN/Inf - пропускаем этот шаг и останавливаемся
                    if math.isnan(loss_val) or math.isinf(loss_val):
                        logger.error(f"Loss is NaN or Inf at step {global_step}. Stopping training to prevent bad model.")
                        metrics.update(status="error", error=f"Training Diverged: Loss is NaN or Infinity at step {global_step}. Try lowering learning rate or changing precision.")
                        raise ValueError(f"Training Diverged: Loss is NaN or Infinity at step {global_step}.")
                    
                    # micro-аккумулятор: только для текущего update-step
                    micro_loss_sum += loss_val
                    micro_count += 1
                    
                    accelerator.backward(loss)
                    
                    # ВАЖНО: шаг оптимизатора только на update-step
                    if accelerator.sync_gradients:
                        # Gradient clipping для стабильности
                        max_grad_norm = config.get("max_grad_norm", 1.0)
                        if max_grad_norm > 0:
                            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                
                # Проверяем, произошел ли шаг оптимизатора (синхронизация градиентов)
                if accelerator.sync_gradients:
                    global_step += 1
                    
                    # Время выполнения одного шага обновления (update step)
                    update_time = time.time() - update_start_time
                    update_start_time = time.time()
                    
                    # Проверка на случай если все микро-шаги были пропущены (не должно случиться после фикса выше)
                    if micro_count == 0:
                        logger.warning(f"No valid loss values at step {global_step}, skipping update")
                        continue

                    update_loss = micro_loss_sum / micro_count
                    # ВАЖНО: усредняем loss по всем процессам в Multi-GPU
                    update_loss_t = torch.tensor(update_loss, device=accelerator.device)
                    update_loss = accelerator.reduce(update_loss_t, reduction="mean").item()
                    
                    micro_loss_sum = 0.0
                    micro_count = 0

                    # накопление для сглаженного лога
                    log_loss_sum += update_loss
                    log_updates += 1

                    # Debug: проверка дублей после первых N шагов
                    # ВАЖНО: При dispatch_batches=True нужно собирать хэши со всех процессов через gather
                    if debug_check_duplicates and global_step == debug_max_samples:
                        # Собираем количество хэшей с каждого процесса
                        hash_counts = torch.tensor([len(debug_sample_ids)], device=accelerator.device, dtype=torch.long)
                        all_counts = accelerator.gather(hash_counts)
                        
                        if accelerator.is_main_process:
                            total_hashes = all_counts.sum().item()
                            logger.info(f"Debug: collected {total_hashes} sample hashes across all processes")
                            
                            # Собираем все хэши со всех процессов для проверки дублей
                            # Создаем тензоры с хэшами (каждый процесс отправляет свои хэши)
                            if len(debug_sample_ids) > 0:
                                local_hashes = torch.tensor([h for _, _, h in debug_sample_ids], device=accelerator.device, dtype=torch.long)
                                # Pad до максимальной длины для gather
                                max_len = all_counts.max().item()
                                if len(local_hashes) < max_len:
                                    padding = torch.full((max_len - len(local_hashes),), -1, device=accelerator.device, dtype=torch.long)
                                    local_hashes = torch.cat([local_hashes, padding])
                                
                                all_hashes = accelerator.gather(local_hashes.unsqueeze(0))
                                # Убираем padding (-1) и проверяем дубли
                                all_hashes_flat = all_hashes.flatten()
                                valid_hashes = all_hashes_flat[all_hashes_flat != -1].cpu().numpy()
                                
                                unique_hashes = set(valid_hashes)
                                if len(valid_hashes) != len(unique_hashes):
                                    duplicates = len(valid_hashes) - len(unique_hashes)
                                    logger.warning(f"Debug: Found {duplicates} duplicate samples in first {debug_max_samples} steps across all processes!")
                                else:
                                    logger.info(f"Debug: No duplicates found in first {debug_max_samples} steps (all {len(valid_hashes)} samples unique across all processes)")
                            else:
                                logger.warning("Debug: No sample hashes collected on main process")
                    
                    # Логирование
                    if global_step % config.get("log_every", 10) == 0 or global_step == 1:
                        avg_loss = log_loss_sum / max(1, log_updates)
                        
                        # Samples per second (Effective Batch / Time)
                        # Учитываем multi-GPU: реальный глобальный batch = effective_batch * num_processes
                        effective_batch = config["batch_size"] * config["gradient_accumulation"]
                        global_batch = effective_batch * accelerator.num_processes
                        samples_per_sec = global_batch / update_time if update_time > 0 else 0
                        
                        metrics.log_step(
                            step=global_step,
                            loss=avg_loss,
                            lr=lr_scheduler.get_last_lr()[0],
                            samples_per_sec=samples_per_sec,
                            step_time=update_time
                        )
                        
                        log_loss_sum = 0.0
                        log_updates = 0
                    
                    # Checkpoint
                    if global_step % config.get("save_every", 5000) == 0:
                        ckpt_path = output_dir / f"checkpoint_step{global_step}"
                        accelerator.save_state(ckpt_path)
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            model_config.save_pretrained(ckpt_path)
                            metrics.log_checkpoint(str(ckpt_path))
                    
                    # Validation
                    # ВАЖНО: evaluate() вызывается на всех процессах, т.к. val_loader зашардирован
                    # Каждый процесс видит свою часть val данных, reduce() усредняет результаты
                    if val_loader is not None and eval_every > 0 and (global_step % eval_every == 0):
                        val_loss = evaluate(val_loader)  # Вызывается на всех процессах
                        if accelerator.is_main_process and val_loss is not None:
                            metrics.log_eval(global_step, float(val_loss))
                            logger.info(f"Validation at step {global_step}: val_loss={val_loss:.4f}")
                    
                    # Проверяем достигли ли лимита шагов
                    if global_step >= max_train_steps:
                        training_complete = True
                        break
        
        # Final save - только на main process
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metrics.update(status="saving_model")
            final_dir = output_dir / "final_model"
            final_dir.mkdir(parents=True, exist_ok=True)
            
            # --- (опционально) сохраняем chat_template для SFT ---
            if stage == "sft" and config.get("sft_template"):
                tmpl = config["sft_template"]
                
                default_sys = tmpl.get("system", "You are a helpful assistant.")
                u_tag = tmpl.get("user_tag", "### User:")
                b_tag = tmpl.get("bot_tag", "### Assistant:")
                
                default_sys_escaped = default_sys.replace("'", "\\'")
                u_tag_escaped = u_tag.replace("'", "\\'")
                b_tag_escaped = b_tag.replace("'", "\\'")
                
                tokenizer.chat_template = (
                    "{% if messages and messages[0]['role'] == 'system' %}"
                    "{{ messages[0]['content'] }}\n\n"
                    "{% else %}"
                    f"{{{{ '{default_sys_escaped}' }}}}\n\n"
                    "{% endif %}"
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}"
                    f"{u_tag_escaped}\n{{{{ message['content'] }}}}\n\n"
                    "{% elif message['role'] == 'assistant' %}"
                    f"{b_tag_escaped}\n{{{{ message['content'] }}}}\n\n"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    f"{b_tag_escaped}\n"
                    "{% endif %}"
                )
                logger.info(f"Saved chat_template: user_tag='{u_tag}', bot_tag='{b_tag}'")
            
            # --- ВСЕГДА сохраняем финальную модель (и для pretrain тоже) ---
            # Используем адаптер для универсального сохранения (работает для Home и HF моделей)
            adapter.save_final(accelerator, model, tokenizer, final_dir)
            
            total_time = time.time() - start_time
            hours, rem = divmod(total_time, 3600)
            minutes, seconds = divmod(rem, 60)
            duration_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
            
            metrics.update(
                status="completed",
                total_time_seconds=total_time,
                training_duration=duration_str,
                final_model_path=str(final_dir),
            )
        
        accelerator.wait_for_everyone()
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Training failed: {e}\n{tb}")
        metrics.update(status="error", error=f"{str(e)}\n\n{tb}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics JSON")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    run_training(config, Path(args.metrics))


if __name__ == "__main__":
    main()
