"""
VLM SFT: Fine-tuning Vision-Language Models (e.g. LLaVA) on image + text data.

Data format (JSONL):
  - {"image": "<path or URL>", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  - or {"image": "<path>", "caption": "..."}

Entry point: python -m homellm.training.vlm_sft --config <path> --metrics <path>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MetricsLogger (same JSON shape as trainer_worker for UI compatibility)
# ---------------------------------------------------------------------------

def _get_gpu_stats() -> List[Dict]:
    out = []
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        out.append({
                            "id": int(parts[0]),
                            "memory_used_gb": round(float(parts[1]) / 1024, 2),
                            "memory_total_gb": round(float(parts[2]) / 1024, 2),
                        })
    except Exception:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    out.append({
                        "id": i,
                        "memory_used_gb": round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                        "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    })
                except Exception:
                    pass
    return out


class MetricsLogger:
    """Writes metrics to JSON for VLM Studio UI (same format as trainer_worker)."""

    def __init__(self, log_path: Path, enabled: bool = True):
        self.log_path = Path(log_path)
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
        }
        self._save()

    def _save(self):
        if not self.enabled:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.log_path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(self.metrics, f, indent=2)
            os.replace(tmp, self.log_path)
        except Exception as e:
            logger.warning("Failed to save metrics: %s", e)
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

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
        self.metrics["loss_history"] = self.metrics.get("loss_history", []) + [loss]
        self.metrics["lr_history"] = self.metrics.get("lr_history", []) + [lr]
        self.metrics["steps_history"] = self.metrics.get("steps_history", []) + [step]
        self.metrics["elapsed_seconds"] = time.time() - self.start_timestamp
        gpu_stats = _get_gpu_stats()
        self.metrics["gpu_stats"] = gpu_stats
        if gpu_stats:
            self.metrics["gpu_memory_used_mb"] = int(gpu_stats[0].get("memory_used_gb", 0) * 1024)
        if step > 0 and step_time > 0:
            remaining = max(0, self.metrics["total_steps"] - step)
            self.metrics["eta_seconds"] = int(remaining * step_time)
        self._save()


# ---------------------------------------------------------------------------
# Dataset: JSONL with image path/URL + conversations or caption
# ---------------------------------------------------------------------------

def _load_image(image_path: str, base_dir: Optional[Path] = None) -> Optional[Image.Image]:
    """Load image from path or URL. base_dir is used to resolve relative paths."""
    path = image_path.strip()
    if path.startswith(("http://", "https://")):
        try:
            from urllib.request import urlopen
            from io import BytesIO
            with urlopen(path, timeout=10) as resp:
                return Image.open(BytesIO(resp.read())).convert("RGB")
        except Exception as e:
            logger.warning("Failed to load image from URL %s: %s", path, e)
            return None
    if base_dir is not None and not os.path.isabs(path):
        path = str(base_dir / path)
    if not os.path.isfile(path):
        logger.warning("Image file not found: %s", path)
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning("Failed to load image %s: %s", path, e)
        return None


class VLMSFTDataset(Dataset):
    """
    Dataset for VLM SFT from JSONL: each line has "image" (path/URL) and
    "conversations" (list of {role, content}) or "caption" (single string).
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        processor: Any,
        max_length: int = 2048,
        image_key: str = "image",
        base_dir: Optional[Union[str, Path]] = None,
    ):
        self.file_path = Path(file_path)
        self.processor = processor
        self.max_length = max_length
        self.image_key = image_key
        self.base_dir = Path(base_dir) if base_dir else self.file_path.parent
        self.examples: List[Dict[str, Any]] = []
        self._load_examples()

    def _load_examples(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if self.image_key not in data:
                    continue
                if "conversations" in data and isinstance(data["conversations"], list) and len(data["conversations"]) > 0:
                    self.examples.append(data)
                elif "caption" in data and data["caption"]:
                    # Convert caption to single-turn conversation
                    self.examples.append({
                        self.image_key: data[self.image_key],
                        "conversations": [
                            {"role": "user", "content": "Describe this image."},
                            {"role": "assistant", "content": str(data["caption"])},
                        ],
                    })
                else:
                    continue
        logger.info("VLMSFTDataset loaded %d examples from %s", len(self.examples), self.file_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        ex = self.examples[idx]
        image_path = ex[self.image_key]
        img = _load_image(image_path, self.base_dir)
        if img is None:
            return None
        conversations = ex["conversations"]
        # Build text for chat template or plain concatenation
        if hasattr(self.processor, "apply_chat_template") and getattr(
            self.processor, "chat_template", None
        ):
            text = self.processor.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            parts = []
            for msg in conversations:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        x.get("text", str(x)) for x in content if isinstance(x, dict)
                    ).strip() or str(content)
                parts.append(f"{role}: {content}")
            text = "\n".join(parts)
        try:
            # Processor may be LlavaProcessor: (images, text) -> input_ids, attention_mask, pixel_values
            if hasattr(self.processor, "image_processor") and hasattr(self.processor, "tokenizer"):
                from transformers import LlavaProcessor
                if isinstance(self.processor, LlavaProcessor):
                    inputs = self.processor(
                        images=img,
                        text=text,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                    )
                else:
                    inputs = self.processor(
                        images=img,
                        text=text,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                    )
            else:
                inputs = self.processor(
                    images=img,
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                )
        except Exception as e:
            logger.warning("Processor failed for idx %d: %s", idx, e)
            return None
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)
        # Labels: -100 on prompt (non-assistant) tokens; train only on last assistant reply
        labels = input_ids.clone()
        last_assistant_idx = None
        for i in range(len(conversations) - 1, -1, -1):
            if conversations[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        tok = getattr(self.processor, "tokenizer", None)
        if last_assistant_idx is not None and tok is not None and hasattr(self.processor, "apply_chat_template"):
            # Prompt = everything before the last assistant message (with generation prompt)
            prompt_text = self.processor.apply_chat_template(
                conversations[:last_assistant_idx],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tok(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
            prompt_end = min(len(prompt_ids), len(labels))
            labels[:prompt_end] = -100
        # Mask padding
        if tok is not None and getattr(tok, "pad_token_id", None) is not None:
            labels[labels == tok.pad_token_id] = -100
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            out["pixel_values"] = pixel_values
        return out


def _collate_vlm_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack batch; filter None."""
    batch = [x for x in batch if x is not None]
    if not batch:
        raise ValueError("Empty batch after filtering None")
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# ---------------------------------------------------------------------------
# run_vlm_sft
# ---------------------------------------------------------------------------

def run_vlm_sft(config: Dict[str, Any], metrics_logger: MetricsLogger) -> None:
    """Run VLM SFT training. Uses LLaVA by default; model_name_or_path from config."""
    if config.get("stage") not in (None, "vlm_sft"):
        raise ValueError(f"vlm_sft worker supports only stage=vlm_sft, got {config.get('stage')}")
    model_name_or_path = config.get("base_model_path") or config.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("base_model_path or model_name_or_path required")
    data_path = config.get("data_path")
    if not data_path or not Path(data_path).exists():
        raise ValueError("data_path must point to an existing JSONL file")
    output_dir = config.get("output_dir", "out/vlm_sft")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seq_len = int(config.get("seq_len", 2048))
    batch_size = int(config.get("batch_size", 2))
    num_epochs = float(config.get("num_epochs", 3))
    learning_rate = float(config.get("learning_rate", 2e-5))
    tuning_method = (config.get("tuning_method") or "lora").lower()
    save_every = int(config.get("save_every", 500))
    log_every = int(config.get("log_every", 10))
    gradient_accumulation = int(config.get("gradient_accumulation", 4))
    warmup_steps = int(config.get("warmup_steps", 0))

    metrics_logger.update(status="loading_model", stage="vlm_sft")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available, training on CPU (slow)")

    # Load processor and model (LLaVA)
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
    except ImportError:
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        AutoProcessor = LlavaProcessor
        AutoModelForVision2Seq = LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,
    }
    if tuning_method == "qlora":
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        except ImportError:
            logger.warning("bitsandbytes not found, falling back to LoRA without 4bit")
            tuning_method = "lora"
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        )
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **model_kwargs)
    if device == "cuda" and model.device.type != "cuda":
        model = model.to(device)

    if tuning_method in ("lora", "qlora"):
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("peft required for LoRA/QLoRA. pip install peft")
        lora_r = int(config.get("lora_r", 32))
        lora_alpha = int(config.get("lora_alpha", 32))
        target_modules = config.get("lora_target_modules") or [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ]
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Dataset
    metrics_logger.update(status="loading_dataset")
    dataset = VLMSFTDataset(
        file_path=data_path,
        processor=processor,
        max_length=seq_len,
        base_dir=config.get("data_base_dir"),
    )
    if len(dataset) == 0:
        raise ValueError("No valid examples in dataset")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_vlm_batch,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    total_steps = len(dataloader) * int(num_epochs) // max(1, gradient_accumulation)
    metrics_logger.update(total_steps=total_steps, status="training")

    # Trainer
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=log_every,
        save_steps=save_every,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and device == "cuda",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=_collate_vlm_batch,
    )
    from transformers import TrainerCallback
    class _MetricsCallbackClass(TrainerCallback):
        def __init__(self, ml, log_every):
            self.ml = ml
            self.log_every = log_every
            self.step_t0 = time.time()
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step % self.log_every != 0:
                return
            loss = logs.get("loss")
            lr = logs.get("learning_rate", 0)
            if loss is not None:
                step_time = time.time() - self.step_t0
                self.ml.log_step(
                    state.global_step,
                    loss,
                    lr,
                    samples_per_sec=args.per_device_train_batch_size * args.gradient_accumulation_steps / max(step_time, 1e-6),
                    step_time=step_time,
                )
            self.step_t0 = time.time()
    trainer.add_callback(_MetricsCallbackClass(metrics_logger, log_every))

    try:
        trainer.train()
        final_dir = Path(output_dir) / "final_model"
        trainer.save_model(str(final_dir))
        processor.save_pretrained(str(final_dir))
        elapsed = time.time() - metrics_logger.start_timestamp
        metrics_logger.update(status="completed", training_duration=f"{int(elapsed // 60)} min")
        metrics_logger.log_checkpoint(str(final_dir), loss=metrics_logger.metrics.get("current_loss"))
    except Exception as e:
        import traceback
        metrics_logger.update(status="error", error=str(e) + "\n" + traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    metrics_path = Path(args.metrics)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    metrics_logger = MetricsLogger(metrics_path, enabled=True)
    run_vlm_sft(config, metrics_logger)


if __name__ == "__main__":
    main()
