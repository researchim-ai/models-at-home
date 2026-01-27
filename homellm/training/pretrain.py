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
import random
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

# Локальные модели
from homellm.models.home_model import HomeConfig, HomeForCausalLM
from homellm.models.gpt2_model import GPT2HomeConfig, GPT2HomeForCausalLM
from homellm.models.home_model_moe import HomeMoEConfig, HomeMoEForCausalLM

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

    def __init__(
        self,
        file_path: str,
        tokenizer,
        seq_len: int,
        num_replicas: int = 1,
        rank: int = 0,
        split: str = "train",      # "train" | "val"
        val_ratio: float = 0.0,    # если >0, часть строк уходит в val
        shard: bool = True,        # для val поставим False
        # Strict resume: если задано, можем мгновенно восстановить позицию в .jsonl по byte offset.
        # ВАЖНО: для .gz строгий seek невозможен без отдельного индексированного формата.
        resume_byte_offset: int = 0,
        resume_global_idx: int = 0,
        strict_resume: bool = True,
    ):
        super().__init__()
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_replicas = num_replicas
        self.rank = rank
        self.split = split
        self.val_ratio = float(val_ratio or 0.0)
        self.shard = bool(shard)
        self.resume_byte_offset = int(resume_byte_offset or 0)
        self.resume_global_idx = int(resume_global_idx or 0)
        self.strict_resume = bool(strict_resume)

        # Будем обновлять при итерации: позиция (следующий offset/idx) для точного resume
        self._resume_state = {
            "byte_offset": self.resume_byte_offset,
            "global_idx": self.resume_global_idx,
        }

        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset {self.file_path} not found")

        # Определяем формат (правильно обрабатываем .jsonl.gz)
        suffixes = self.file_path.suffixes  # например [".jsonl", ".gz"]
        self._is_gz = bool(suffixes) and suffixes[-1] == ".gz"
        
        # определяем "базовое" расширение без .gz
        base_suffix = suffixes[-2].lower() if self._is_gz and len(suffixes) >= 2 else self.file_path.suffix.lower()
        self._is_jsonl = (base_suffix == ".jsonl")  # НЕ поддерживаем ".json" для pretrain (это массив, не JSONL)

    def _read_lines(self) -> Iterable[str]:
        # Поддержка .gz файлов (включая .jsonl.gz)
        if self._is_gz:
            import gzip
            f = gzip.open(self.file_path, "rt", encoding="utf-8")
        else:
            f = open(self.file_path, "r", encoding="utf-8")
        
        try:
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
        finally:
            f.close()

    def get_resume_state(self) -> dict:
        """
        Возвращает состояние потока для строгого resume.
        Для .gz resume возможен только через slow-skip (или отдельный индексированный формат).
        """
        return {
            "file_path": str(self.file_path),
            "is_gz": bool(self._is_gz),
            "is_jsonl": bool(self._is_jsonl),
            "split": self.split,
            "val_ratio": self.val_ratio,
            "shard": self.shard,
            "num_replicas": int(self.num_replicas),
            "rank": int(self.rank),
            "byte_offset": int(self._resume_state.get("byte_offset", 0) or 0),
            "global_idx": int(self._resume_state.get("global_idx", 0) or 0),
        }

    def __iter__(self):
        """
        Итератор с опциональным шардированием.
        
        ВАЖНО: По умолчанию шардирование делает accelerate.prepare() через DataLoaderShard/Dispatcher.
        Явное шардирование в датасете включается только если shard=True.
        
        Логика:
        1. Читаем все строки с глобальным индексом
        2. Применяем train/val split по глобальному индексу
        3. Опционально шардируем между процессами и workers (только если shard=True)
        """
        from torch.utils.data import get_worker_info
        
        # Train/val split параметры
        scale = 10000
        threshold = int(self.val_ratio * scale)

        # Базовый итератор всех строк с глобальным индексом.
        # Для строгого resume на .jsonl используем бинарное чтение + seek по byte_offset.
        # Это сохраняет ПОЛНУЮ детерминированность при resume (при условии, что файл не менялся).
        if self.strict_resume and not self._is_gz and self._is_jsonl:
            start_off = int(self.resume_byte_offset or 0)
            start_idx = int(self.resume_global_idx or 0)

            def iter_with_offsets():
                idx = start_idx
                with open(self.file_path, "rb") as f:
                    if start_off > 0:
                        f.seek(start_off)
                    # Обновим состояние "мы на этой позиции"
                    self._resume_state = {"byte_offset": int(f.tell()), "global_idx": int(idx)}

                    while True:
                        off = int(f.tell())
                        raw = f.readline()
                        if not raw:
                            break

                        # Декодируем и чистим
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue

                        # Парсим JSONL (text поле), полностью повторяя старую семантику _read_lines()
                        try:
                            obj = json.loads(line)
                            line = obj.get("text", "")
                        except json.JSONDecodeError:
                            continue

                        line = (line or "").strip()
                        if not line:
                            continue

                        # ВАЖНО: idx — это глобальный индекс ПОСЛЕ фильтрации пустых/битых строк,
                        # т.е. ровно то, что раньше давал enumerate(self._read_lines()).
                        # Сохраняем next-state на "следующую валидную строку"
                        next_off = int(f.tell())
                        self._resume_state = {"byte_offset": next_off, "global_idx": int(idx + 1)}
                        yield idx, line
                        idx += 1

            base_iter = iter_with_offsets()
        else:
            # Фолбэк: старый путь (для .txt или .gz). Строгий seek тут не гарантируется.
            base_iter = enumerate(self._read_lines())
        
        # 1. Применяем train/val split по глобальному индексу
        def apply_split(idx_text_pair):
            idx, text = idx_text_pair
            is_val = (threshold > 0) and ((idx % scale) < threshold)
            if self.split == "val" and not is_val:
                return None
            if self.split == "train" and is_val:
                return None
            return idx_text_pair
        
        # Фильтруем по split
        filtered_iter = (pair for pair in base_iter if apply_split(pair) is not None)
        
        # 2. Шардирование между DDP процессами (только если shard=True)
        # ВАЖНО: Если shard=False, шардирование делает accelerate.prepare()
        if self.shard and self.num_replicas > 1:
            filtered_iter = ((idx, text) for idx, text in filtered_iter if idx % self.num_replicas == self.rank)
        
        # 3. Шардирование между DataLoader workers внутри каждого процесса (только если shard=True)
        worker_info = get_worker_info()
        if self.shard and worker_info is not None and worker_info.num_workers > 1:
            total_workers = self.num_replicas * worker_info.num_workers
            worker_rank = self.rank * worker_info.num_workers + worker_info.id
            filtered_iter = ((idx, text) for idx, text in filtered_iter if idx % total_workers == worker_rank)
        
        # Итерируем по данным (зашардированным или нет, в зависимости от shard)
        for idx, text in filtered_iter:
            
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.seq_len,
                padding="max_length",  # Всегда дополняем до seq_len
                add_special_tokens=True,
                return_attention_mask=True,  # ✅ Нужно для правильного masking labels
            )
            
            # Возвращаем dict с attention_mask для правильного masking
            # ВАЖНО: labels будут маскироваться по attention_mask, а не по pad_token_id,
            # чтобы не маскировать "настоящий EOS" если pad_token = eos_token
            yield {
                "input_ids": torch.tensor(tokens["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(tokens["attention_mask"], dtype=torch.long),
            }

    def get_sample_prompt(self, max_samples: int = 10) -> str | None:
        """Получить пример текста из датасета для отображения в UI.
        
        Использует reservoir sampling для больших файлов.
        """
        try:
            samples = []
            # Поддержка .gz файлов
            if self._is_gz:
                import gzip
                f = gzip.open(self.file_path, "rt", encoding="utf-8")
            else:
                f = open(self.file_path, "r", encoding="utf-8")
            
            try:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Парсим JSONL если нужно
                    if self._is_jsonl:
                        try:
                            obj = json.loads(line)
                            line = obj.get("text", "")
                        except json.JSONDecodeError:
                            continue
                    
                    if not line:
                        continue
                    
                    # Reservoir sampling для больших файлов
                    if len(samples) < max_samples:
                        samples.append(line)
                    else:
                        # С вероятностью max_samples/(idx+1) заменяем случайный элемент
                        r = random.randint(0, idx)
                        if r < max_samples:
                            samples[r] = line
                    
                    # Останавливаемся после проверки достаточного количества строк
                    if idx >= max_samples * 10:
                        break
            finally:
                f.close()
            
            if samples:
                # Возвращаем первый непустой семпл
                for sample in samples:
                    if sample.strip():
                        # Обрезаем до разумной длины для отображения
                        if len(sample) > 500:
                            return sample[:500] + "..."
                        return sample
        except Exception as e:
            logger.warning(f"Failed to get sample prompt: {e}")
            return None
        
        return None

    def __len__(self):
        """При первом обращении быстро подсчитывает количество строк в файле.
        
        Учитывает val_ratio и split для корректной оценки длины на один rank.
        """
        if not hasattr(self, "_length"):
            logger.info("Подсчёт строк в датасете %s — может занять время...", self.file_path)
            cnt = 0
            # Поддержка .gz файлов
            if self._is_gz:
                import gzip
                f = gzip.open(self.file_path, "rt", encoding="utf-8")
            else:
                f = open(self.file_path, "r", encoding="utf-8")
            
            try:
                for _ in f:
                    cnt += 1
            finally:
                f.close()
            
            # Учитываем val_ratio и split
            if self.val_ratio > 0:
                val_cnt = int(cnt * self.val_ratio)
                train_cnt = cnt - val_cnt
            else:
                train_cnt = cnt
                val_cnt = 0
            
            # Выбираем нужный split
            if self.split == "train":
                effective = train_cnt
            else:  # val
                effective = val_cnt if self.val_ratio > 0 else 0
            
            # ВАЖНО: Не делим на num_replicas, т.к. шардирование делает accelerate.prepare()
            # Возвращаем полную длину для выбранного split
            self._length = effective
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

    parser.add_argument("--arch", type=str, default="home", choices=["gpt2", "gpt2_home", "home", "home_moe"], 
                        help="Архитектура: home (LLaMA-style), gpt2_home (GPT-2 Classic), home_moe (MoE), gpt2 (HF GPT-2)")
    
    # MoE параметры
    parser.add_argument("--num_experts", type=int, default=8, help="Количество экспертов для MoE архитектуры")
    parser.add_argument("--num_experts_per_tok", type=int, default=2, help="Top-K: сколько экспертов активируется на токен")
    parser.add_argument("--expert_type", type=str, default="swiglu", choices=["swiglu", "mlp"], help="Тип экспертов: swiglu или mlp")

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
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        help="Включить SDPA (scaled_dot_product_attention), чтобы PyTorch мог использовать flash-attention kernels при fp16/bf16",
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Отключить SDPA/flash attention (использовать обычный attention)",
    )
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
    if args.flash_attention and args.no_flash_attention:
        raise ValueError("Нельзя одновременно задавать --flash_attention и --no_flash_attention")

    # Настройка увеличить mixed_precision согласно accelerate>=0.20
    mixed_precision = (
        "fp16" if args.fp16 else "bf16" if args.bf16 else "no"
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=mixed_precision,
    )
    is_main = accelerator.is_main_process

    # Настройка CUDA SDPA kernels (влияет на HomeModel, т.к. он использует scaled_dot_product_attention)
    use_flash_attention = True
    if args.no_flash_attention:
        use_flash_attention = False
    elif args.flash_attention:
        use_flash_attention = True
    if torch.cuda.is_available():
        try:
            if use_flash_attention:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            else:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            if is_main:
                logger.info(
                    "SDPA kernels: flash=%s mem_efficient=%s math=%s (use_flash_attention=%s)",
                    getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: "N/A")(),
                    getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: "N/A")(),
                    getattr(torch.backends.cuda, "math_sdp_enabled", lambda: "N/A")(),
                    use_flash_attention,
                )
        except Exception as e:
            if is_main:
                logger.warning(f"Could not configure CUDA SDPA kernels: {e}")

    if is_main:
        logger.info("Arguments: %s", vars(args))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Dataset / Dataloader
    train_dataset = StreamingTextDataset(args.data_path, tokenizer, seq_len=args.seq_len)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # ВАЖНО: num_workers=0 для IterableDataset, чтобы избежать дублирования данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=0,  # IterableDataset + num_workers>0 = дублирование данных
    )

    # Выбор архитектуры
    arch = getattr(args, "arch", "home")
    
    if arch == "home":
        # HomeModel (LLaMA-style): RMSNorm, RoPE, SwiGLU
        config = HomeConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.n_heads,
            max_position_embeddings=args.seq_len,
            use_sdpa=use_flash_attention,
        )
        model = HomeForCausalLM(config)
        if is_main:
            logger.info("🏠 Архитектура: HomeModel (LLaMA-style) — RMSNorm, RoPE, SwiGLU")
    
    elif arch == "gpt2_home":
        # GPT-2 Classic: LayerNorm, Learned Pos, GELU
        config = GPT2HomeConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.n_heads,
            max_position_embeddings=args.seq_len,
        )
        model = GPT2HomeForCausalLM(config)
        if is_main:
            logger.info("🤖 Архитектура: GPT-2 Classic — LayerNorm, Learned Pos, GELU")
    
    elif arch == "home_moe":
        # HomeModel MoE: RMSNorm, RoPE, SwiGLU experts
        config = HomeMoEConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.n_heads,
            max_position_embeddings=args.seq_len,
            use_sdpa=use_flash_attention,
            num_experts=args.num_experts,
            num_experts_per_tok=args.num_experts_per_tok,
            expert_type=args.expert_type,
        )
        model = HomeMoEForCausalLM(config)
        if is_main:
            logger.info(f"🔀 Архитектура: HomeModel MoE — {args.num_experts} экспертов, Top-{args.num_experts_per_tok}")
    
    else:
        # Fallback: HuggingFace GPT-2
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_embd=args.hidden_size,
            n_layer=args.num_layers,
            n_head=args.n_heads,
            n_positions=args.seq_len,
        )
        model = GPT2LMHeadModel(config)
        if is_main:
            logger.info("🤗 Архитектура: HuggingFace GPT-2 (стандартная)")

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
                with accelerator.autocast():
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