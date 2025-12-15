#!/usr/bin/env python
"""
homellm.training.pretrain
=========================
Скрипт языкового претрейна «с нуля» на корпусе из JSONL/текст-файла.

Главные особенности:
1. Полностью независим от minimind: собственные конфиги, загрузка корпуса, модель.
2. Использует HuggingFace `transformers` + `accelerate` – автоматически масштабируется на CPU / одну или несколько GPU.
3. Поддерживаются аргументы CLI для:
   • пути к датасету (`--data_path`),
   • каталога для чекпойнтов (`--output_dir`),
   • размера модели (`--hidden_size`, `--num_layers`, `--n_heads`),
   • длины последовательности (`--seq_len`),
   • размеров батча, lr, warmup.
4. Простой текстовый датасет (каждая строка – UTF-8 текст) либо JSONL c ключом "text". Потоковое чтение – не держит весь корпус в RAM.
5. Ротационные позиции (RoPE) опционно, gradient checkpointing, fp16/bf16, cosine scheduler с warmup.
6. Сохранение чекпойнтов через `accelerate` → совместимо с HF Hub.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Iterable, List, Dict

import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)

# Локальная модель
from homellm.models.home_model import HomeConfig, HomeForCausalLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """Поточно читает текст/JSONL файл построчно, не загружая всё в память."""

    def __init__(self, file_path: str, tokenizer, seq_len: int, num_replicas: int = 1, rank: int = 0):
        super().__init__()
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_replicas = num_replicas
        self.rank = rank

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset {self.file_path} not found")

        # Определяем формат
        self._is_jsonl = self.file_path.suffix.lower() in {".jsonl", ".json"}

    def _read_lines(self) -> Iterable[str]:
        mode = "r" if self.file_path.suffix != ".gz" else "rt"
        with open(self.file_path, mode, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if self._is_jsonl:
                    try:
                        obj = json.loads(line)
                        line = obj.get("text", "")
                    except json.JSONDecodeError:
                        continue
                yield line

    def __iter__(self):
        # Получаем информацию о воркере DataLoader
        worker_info = torch.utils.data.get_worker_info()
        
        # Общее количество "читателей" = num_replicas * num_workers
        total_workers = self.num_replicas
        current_worker_id = self.rank
        
        if worker_info is not None:
            total_workers *= worker_info.num_workers
            current_worker_id = self.rank * worker_info.num_workers + worker_info.id
        
        for idx, text in enumerate(self._read_lines()):
            # Шардинг данных между процессами и воркерами
            if idx % total_workers != current_worker_id:
                continue
            
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",  # Всегда дополняем до seq_len
                add_special_tokens=True,
            )["input_ids"]
            
            yield torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        """При первом обращении быстро подсчитывает количество строк в файле. Может быть медленно для очень больших файлов."""
        if not hasattr(self, "_length"):
            logger.info("Подсчёт строк в датасете %s — может занять время...", self.file_path)
            cnt = 0
            with open(self.file_path, "r", encoding="utf-8") as f:
                for _ in f:
                    cnt += 1
            self._length = cnt
        return self._length


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="HomeLLM pretraining script")
    # Files & dirs
    parser.add_argument("--data_path", type=str, required=True, help="Путь к .txt/.jsonl корпусу")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Куда сохранять модель")
    parser.add_argument("--tokenizer_path", type=str, default="gpt2", help="Хуггингфейс-токенизатор или путь")

    # Model size
    parser.add_argument("--hidden_size", type=int, default=512, help="Размер эмбеддинга/скрытого слоя")
    parser.add_argument("--num_layers", type=int, default=8, help="Количество слоёв Transformer")
    parser.add_argument("--n_heads", type=int, default=8, help="Количество голов внимания")
    parser.add_argument("--seq_len", type=int, default=512, help="Максимальная длина входа")

    parser.add_argument("--arch", type=str, default="gpt2", choices=["gpt2", "home"], help="Какая архитектура используется")

    # Training params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay для AdamW")
    parser.add_argument("--fp16", action="store_true", help="Использовать fp16")
    parser.add_argument("--bf16", action="store_true", help="Использовать bf16 (Ampere+)")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Включить torch.gradient_checkpointing для экономии VRAM")
    parser.add_argument("--save_every", type=int, default=10000, help="Сохранять чекпойнт каждые N шагов")
    parser.add_argument("--log_every", type=int, default=500, help="Логировать каждый N шаг")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Нельзя одновременно задавать --fp16 и --bf16")

    # Настройка увеличить mixed_precision согласно accelerate>=0.20
    mixed_precision = (
        "fp16" if args.fp16 else "bf16" if args.bf16 else "no"
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=mixed_precision,
    )
    is_main = accelerator.is_main_process

    if is_main:
        logger.info("Arguments: %s", vars(args))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Dataset / Dataloader
    train_dataset = StreamingTextDataset(args.data_path, tokenizer, seq_len=args.seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=2,
    )

    # Выбор архитектуры
    if getattr(args, "arch", "gpt2") == "home":
        config = HomeConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.n_heads,
            max_position_embeddings=args.seq_len,
        )
        model = HomeForCausalLM(config)
    else:
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_embd=args.hidden_size,
            n_layer=args.num_layers,
            n_head=args.n_heads,
            n_positions=args.seq_len,
        )
        model = GPT2LMHeadModel(config)

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    # LR scheduler
    try:
        dataset_len = len(train_loader)
    except TypeError:
        # Для IterableDataset без __len__ — оцениваем через аргумент epochs=1 => steps = save_every*??
        dataset_len = args.save_every * args.epochs
    num_update_steps_per_epoch = math.ceil(dataset_len / args.gradient_accumulation)
    max_train_steps = args.epochs * num_update_steps_per_epoch

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        betas=(0.9, 0.95), 
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        if is_main:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            if is_main and ((step + 1) % args.log_every == 0 or step == 0):
                pbar.set_postfix({"loss": f"{loss.detach().float().item():.6f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"})
                # обновляем прогресс на фактическое количество обработанных батчей
                pbar.update(args.log_every if (step + 1) % args.log_every == 0 else 1)

            # Save
            if global_step % args.save_every == 0:
                if is_main:
                    ckpt_path = output_dir / f"checkpoint_step{global_step}.pt"
                    accelerator.save_state(ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")

        if is_main:
            pbar.close()

    # Final save (merged weights -> HF format)
    if is_main:
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Training finished, model saved to {final_dir}")


if __name__ == "__main__":
    main() 