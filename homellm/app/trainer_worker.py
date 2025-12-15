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

from homellm.models.home_model import HomeConfig, HomeForCausalLM
from homellm.training.pretrain import StreamingTextDataset

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Логгер метрик в JSON файл для визуализации."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
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
        }
        self._save()
    
    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def update(self, **kwargs):
        self.metrics.update(kwargs)
        self._save()
    
    def log_step(self, step: int, loss: float, lr: float, samples_per_sec: float = 0):
        self.metrics["current_step"] = step
        self.metrics["current_loss"] = loss
        self.metrics["current_lr"] = lr
        self.metrics["samples_per_second"] = samples_per_sec
        self.metrics["loss_history"].append(loss)
        self.metrics["lr_history"].append(lr)
        self.metrics["steps_history"].append(step)
        
        # ETA
        if step > 0 and samples_per_sec > 0:
            remaining_steps = self.metrics["total_steps"] - step
            self.metrics["eta_seconds"] = int(remaining_steps / samples_per_sec)
        
        self._save()
    
    def log_checkpoint(self, path: str):
        self.metrics["checkpoints"].append({
            "path": path,
            "step": self.metrics["current_step"],
            "time": datetime.now().isoformat()
        })
        self._save()


def run_training(config: Dict[str, Any], metrics_path: Path):
    """Запуск тренировки с записью метрик."""
    
    metrics = MetricsLogger(metrics_path)
    
    try:
        metrics.update(status="loading_tokenizer")
        
        # Mixed precision
        mixed_precision = config.get("mixed_precision", "no")
        
        accelerator = Accelerator(
            gradient_accumulation_steps=config["gradient_accumulation"],
            mixed_precision=mixed_precision,
        )
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        
        metrics.update(status="loading_dataset")
        
        # Dataset
        train_dataset = StreamingTextDataset(
            config["data_path"], 
            tokenizer, 
            seq_len=config["seq_len"]
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            collate_fn=data_collator,
            num_workers=2,
        )
        
        metrics.update(status="building_model")
        
        # Model
        model_config = HomeConfig(
            vocab_size=len(tokenizer),
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["n_heads"],
            max_position_embeddings=config["seq_len"],
            dropout=config.get("dropout", 0.0),
        )
        model = HomeForCausalLM(model_config)
        
        # Подсчёт параметров
        num_params = sum(p.numel() for p in model.parameters())
        metrics.update(num_parameters=num_params)
        
        if config.get("grad_checkpoint", False):
            model.gradient_checkpointing_enable()
        
        # Scheduler
        try:
            dataset_len = len(train_loader)
        except TypeError:
            dataset_len = config.get("save_every", 5000) * config["epochs"]
        
        num_update_steps_per_epoch = math.ceil(dataset_len / config["gradient_accumulation"])
        max_train_steps = config["epochs"] * num_update_steps_per_epoch
        
        metrics.update(total_steps=max_train_steps)
        
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
        
        model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, lr_scheduler
        )
        
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics.update(status="training")
        
        global_step = 0
        start_time = time.time()
        
        for epoch in range(config["epochs"]):
            metrics.update(epoch=epoch + 1)
            model.train()
            
            for step, batch in enumerate(train_loader):
                step_start = time.time()
                
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                # Логирование
                if global_step % config.get("log_every", 10) == 0 or global_step == 1:
                    step_time = time.time() - step_start
                    samples_per_sec = config["batch_size"] / step_time if step_time > 0 else 0
                    
                    metrics.log_step(
                        step=global_step,
                        loss=loss.detach().float().item(),
                        lr=lr_scheduler.get_last_lr()[0],
                        samples_per_sec=samples_per_sec
                    )
                
                # Checkpoint
                if global_step % config.get("save_every", 5000) == 0:
                    ckpt_path = output_dir / f"checkpoint_step{global_step}"
                    accelerator.save_state(ckpt_path)
                    metrics.log_checkpoint(str(ckpt_path))
        
        # Final save
        metrics.update(status="saving_model")
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)
        
        total_time = time.time() - start_time
        metrics.update(
            status="completed",
            total_time_seconds=total_time,
            final_model_path=str(final_dir)
        )
        
    except Exception as e:
        metrics.update(status="error", error=str(e))
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

