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
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from safetensors.torch import load_file

from homellm.models.home_model import HomeConfig, HomeForCausalLM
from homellm.training.pretrain import StreamingTextDataset
from homellm.training.sft import SFTDataset

logger = logging.getLogger(__name__)


def get_gpu_stats():
    """Получить статистику GPU через nvidia-smi (работает правильно при DDP)."""
    gpu_stats = []
    
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
    stage = config.get("stage", "pretrain") # pretrain | sft
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation"],
        mixed_precision=mixed_precision,
    )
    
    # ВАЖНО: MetricsLogger только на main process для избежания гонок при Multi-GPU
    metrics = MetricsLogger(metrics_path, enabled=accelerator.is_main_process)
    
    try:
        metrics.update(status="loading_tokenizer", stage=stage)
        
        # Tokenizer
        # Для SFT важно использовать токенизатор базовой модели (если он там сохранён)
        tokenizer_path = config.get("tokenizer_path", "gpt2")
        
        if stage == "sft" and config.get("base_model_path"):
            bp = Path(config["base_model_path"])
            # Проверяем наличие файлов токенизатора в базовой модели
            if (bp / "tokenizer.json").exists() or (bp / "tokenizer_config.json").exists() or (bp / "vocab.json").exists():
                tokenizer_path = str(bp)
                logger.info(f"Using tokenizer from base model: {tokenizer_path}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception as e:
            # Если путь к токенизатору кривой или это путь к чекпоинту без токенизатора
            logger.warning(f"Failed to load tokenizer from {tokenizer_path}, falling back to gpt2: {e}")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # pad token: лучше иметь, иначе паддинг/коллатор могут чудить
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        
        metrics.update(status=f"loading_dataset ({stage})")
        
        # Dataset Selection based on Stage
        if stage == "sft":
            train_dataset = SFTDataset(
                config["data_path"],
                tokenizer,
                seq_len=config["seq_len"],
                sft_columns=config.get("sft_columns"),
                sft_template=config.get("sft_template"),
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index
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
            # Pretrain
            train_dataset = StreamingTextDataset(
                config["data_path"], 
                tokenizer, 
                seq_len=config["seq_len"],
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index
            )
            
            # Получаем пример текста для отображения (для pretrain)
            try:
                if hasattr(train_dataset, 'get_sample_prompt'):
                    sample_prompt = train_dataset.get_sample_prompt(max_samples=20)
                    if sample_prompt:
                        metrics.update(sample_prompt=sample_prompt)
                        logger.info("Sample prompt saved to metrics")
            except Exception as e:
                logger.warning(f"Failed to get sample prompt: {e}")
            
            collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
        # Создаем validation loader если нужно
        val_ratio = float(config.get("val_ratio", 0.0))
        eval_every = int(config.get("eval_every", 0) or 0)
        eval_batches = int(config.get("eval_batches", 20) or 20)
        val_loader = None
        
        if val_ratio > 0.0 and eval_every > 0:
            if stage == "sft":
                from homellm.training.sft import SFTDataset
                val_dataset = SFTDataset(
                    config["data_path"],
                    tokenizer,
                    seq_len=config["seq_len"],
                    sft_columns=config.get("sft_columns"),
                    sft_template=config.get("sft_template"),
                    num_replicas=1,  # важно!
                    rank=0,
                    split="val",
                    val_ratio=val_ratio,
                    shard=False,     # важно!
                )
                from transformers import default_data_collator
                val_collate = default_data_collator
            else:
                # Используем уже импортированный StreamingTextDataset
                val_dataset = StreamingTextDataset(
                    config["data_path"],
                    tokenizer,
                    seq_len=config["seq_len"],
                    num_replicas=1,
                    rank=0,
                    split="val",
                    val_ratio=val_ratio,
                    shard=False,
                )
                val_collate = DataCollatorForLanguageModeling(tokenizer, mlm=False)
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                collate_fn=val_collate,
                num_workers=2,
            )
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            collate_fn=collate_fn,
            num_workers=2,
        )
        
        metrics.update(status="building_model")
        
        # Model Configuration
        # Если SFT, мы должны загрузить веса
        if stage == "sft" and config.get("base_model_path"):
            base_model_path = Path(config["base_model_path"])
            
            # Пытаемся загрузить конфиг
            if (base_model_path / "config.json").exists():
                model_config = HomeConfig.from_pretrained(str(base_model_path))
            else:
                 # Пытаемся найти конфиг в родительской папке запуска
                 parent_config = base_model_path.parent / "run_config.json"
                 if parent_config.exists():
                     with open(parent_config) as f:
                        run_cfg = json.load(f)
                     model_config = HomeConfig(
                        vocab_size=len(tokenizer),
                        hidden_size=run_cfg.get("hidden_size", 512),
                        num_hidden_layers=run_cfg.get("num_layers", 8),
                        num_attention_heads=run_cfg.get("n_heads", 8),
                        max_position_embeddings=run_cfg.get("seq_len", 512),
                     )
                 else:
                     raise ValueError(f"Cannot find config.json in {base_model_path}")
                     
            model = HomeForCausalLM(model_config)
            
            # Загрузка весов
            # Ищем .safetensors или .bin
            if (base_model_path / "model.safetensors").exists():
                 state_dict = load_file(str(base_model_path / "model.safetensors"))
                 model.load_state_dict(state_dict, strict=False) # strict=False т.к. может измениться размер vocab
            elif (base_model_path / "pytorch_model.bin").exists():
                 state_dict = torch.load(base_model_path / "pytorch_model.bin", map_location="cpu")
                 model.load_state_dict(state_dict, strict=False)
            else:
                # Accelerate checkpoint format?
                try:
                    # Попробуем загрузить через load_state_dict если это папка чекпоинта accelerate
                    # Обычно accelerate сохраняет random_states и model.safetensors внутри
                    pass
                except:
                    pass
            
            logger.info(f"Loaded base model from {base_model_path}")
            
        else:
            # Pretrain from scratch
            model_config = HomeConfig(
                vocab_size=len(tokenizer),
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["n_heads"],
                max_position_embeddings=config["seq_len"],
                dropout=config.get("dropout", 0.0),
            )
            model = HomeForCausalLM(model_config)
        
        # Если vocab токенизатора изменился (например добавили pad_token) — расширяем эмбеддинги
        # Это важно для обоих случаев: pretrain (vocab может измениться после добавления pad_token)
        # и SFT (vocab базовой модели может отличаться от токенизатора)
        if model.config.vocab_size != len(tokenizer):
            logger.info(f"Resizing token embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
            model.config.vocab_size = len(tokenizer)
            if hasattr(model, "tie_weights"):
                model.tie_weights()
        
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
            try:
                dataset_len = len(train_dataset)  # Количество примеров (если доступно)
                steps_per_epoch = math.ceil(dataset_len / config["batch_size"])
                num_update_steps_per_epoch = math.ceil(steps_per_epoch / config["gradient_accumulation"])
                max_train_steps = config["epochs"] * num_update_steps_per_epoch
            except (TypeError, AttributeError):
                # Для streaming dataset без __len__ - используем save_every * 10 как оценку
                max_train_steps = config.get("save_every", 5000) * 2
        
        metrics.update(total_steps=max_train_steps, max_steps_estimated=config.get("max_steps") is None)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
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
        
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics.update(status="training")
        
        # Функция для валидации (использует eval_batches из замыкания)
        def evaluate(val_loader):
            """Выполнить валидацию на val_loader."""
            model.eval()
            losses = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= eval_batches:
                        break
                    out = model(**batch)
                    loss = out.loss.detach()
                    # на всякий: усреднить по ранкам
                    loss = accelerator.reduce(loss, reduction="mean")
                    losses.append(loss.item())
            model.train()
            if not losses:
                return None
            return sum(losses) / len(losses)
        
        global_step = 0
        start_time = time.time()
        
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
                
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    loss_val = loss.detach().float().item()
                    
                    # micro-аккумулятор: только для текущего update-step
                    if not (math.isnan(loss_val) or math.isinf(loss_val)):
                        micro_loss_sum += loss_val
                        micro_count += 1
                    
                    accelerator.backward(loss)
                    
                    # ВАЖНО: шаг оптимизатора только на update-step
                    if accelerator.sync_gradients:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                
                # Проверяем, произошел ли шаг оптимизатора (синхронизация градиентов)
                if accelerator.sync_gradients:
                    global_step += 1
                    
                    # Время выполнения одного шага обновления (update step)
                    update_time = time.time() - update_start_time
                    update_start_time = time.time()
                    
                    # Если на ВСЕХ микро-шагах update-step был NaN/Inf — останавливаемся
                    if micro_count == 0:
                        logger.error(f"Loss is NaN or Inf at step {global_step}. Stopping training to prevent bad model.")
                        metrics.update(status="error", error="Training Diverged: Loss is NaN or Infinity. Try lowering learning rate or changing precision.")
                        raise ValueError("Training Diverged: Loss is NaN or Infinity.")

                    update_loss = micro_loss_sum / micro_count
                    micro_loss_sum = 0.0
                    micro_count = 0

                    # накопление для сглаженного лога
                    log_loss_sum += update_loss
                    log_updates += 1

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
                    if val_loader is not None and eval_every > 0 and (global_step % eval_every == 0):
                        val_loss = evaluate(val_loader)
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
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(final_dir, save_function=accelerator.save)
            model_config.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            
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
