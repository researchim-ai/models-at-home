"""
VLM SFT: modern supervised fine-tuning for small VLMs.

Supports Qwen 3.5, Qwen2/2.5-VL, LLaVA-NeXT and similar HF image-text-to-text models.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import torch

from .vlm_common import (
    MetricsLogger,
    apply_vlm_freeze_policy,
    apply_vlm_lora,
    configure_sdpa_kernels,
    ensure_cuda_available,
    load_vlm_model,
    load_vlm_processor,
    mark_training_completed,
    resolve_resume_checkpoint,
)
from .vlm_data import VLMDataCollator, VLMJsonlDataset

logger = logging.getLogger(__name__)

VLMSFTDataset = VLMJsonlDataset


def run_vlm_sft(config: Dict[str, Any], metrics_logger: MetricsLogger) -> None:
    """Public entrypoint: runs SFT and records any failure into metrics.json.

    Wrapping the whole pipeline (not just trainer.train) means model/dataset
    loading errors surface in the UI instead of leaving it stuck on
    ``loading_model`` forever.
    """
    try:
        _run_vlm_sft_impl(config, metrics_logger)
    except Exception as exc:
        import traceback

        metrics_logger.update(status="error", error=str(exc) + "\n" + traceback.format_exc())
        raise


def _run_vlm_sft_impl(config: Dict[str, Any], metrics_logger: MetricsLogger) -> None:
    if config.get("stage") not in (None, "vlm_sft"):
        raise ValueError(f"vlm_sft worker supports only stage=vlm_sft, got {config.get('stage')}")

    model_name_or_path = config.get("base_model_path") or config.get("model_name_or_path")
    if not model_name_or_path:
        raise ValueError("base_model_path or model_name_or_path required")

    data_path = config.get("data_path")
    if not data_path or not Path(data_path).exists():
        raise ValueError("data_path must point to an existing JSONL file")

    output_dir = Path(config.get("output_dir", "out/vlm_sft"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_len = int(config.get("seq_len", 2048))
    batch_size = int(config.get("batch_size", 2))
    num_epochs = float(config.get("num_epochs", 1))
    max_steps = int(config["max_steps"]) if config.get("max_steps") else -1
    learning_rate = float(config.get("learning_rate", 2e-5))
    gradient_accumulation = int(config.get("gradient_accumulation", 4))
    warmup_steps = int(config.get("warmup_steps", 0))
    log_every = int(config.get("log_every", 10))
    save_every = int(config.get("save_every", 500))
    save_total_limit = int(config.get("save_total_limit", 2))
    eval_steps = int(config.get("eval_steps", 0))
    lr_schedule = str(config.get("lr_schedule", "cosine"))
    # Map UI/legacy optimizer names to valid HF TrainingArguments.optim values.
    optim_aliases = {
        "adamw": "adamw_torch",
        "adam": "adamw_torch",
        "adamw_8bit": "adamw_bnb_8bit",
        "muon": "adamw_torch",
        "magma_adamw": "adamw_torch",
    }
    valid_optims = {
        "adamw_torch", "adamw_torch_fused", "adamw_hf", "adamw_apex_fused",
        "adamw_anyprecision", "adamw_bnb_8bit", "adafactor", "sgd", "adagrad", "rmsprop",
    }
    optimizer = str(config.get("optimizer", "adamw"))
    optimizer = optim_aliases.get(optimizer, optimizer)
    if optimizer not in valid_optims:
        optimizer = "adamw_torch"
    mixed_precision = str(config.get("mixed_precision", "") or "").lower()
    use_bf16 = device == "cuda" and mixed_precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and ((mixed_precision == "fp16") or (mixed_precision not in {"bf16", "fp16", "no"} and not torch.cuda.is_bf16_supported()))

    ensure_cuda_available()
    metrics_logger.update(status="loading_model", stage="vlm_sft", model_name_or_path=model_name_or_path)
    configure_sdpa_kernels(config)
    processor = load_vlm_processor(model_name_or_path, config)
    model = load_vlm_model(model_name_or_path, config, device)
    metrics_logger.update(attn_implementation=getattr(getattr(model, "config", None), "_attn_implementation", None))
    apply_vlm_freeze_policy(model, config)
    model = apply_vlm_lora(model, config)
    try:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        metrics_logger.update(trainable_params=int(trainable), num_parameters=int(total))
    except Exception:
        pass

    metrics_logger.update(status="loading_dataset")
    field_map = config.get("vlm_columns") or None
    train_dataset = VLMJsonlDataset(
        data_path,
        prompt_for_caption=str(config.get("caption_prompt", "Describe this image.")),
        field_map=field_map,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "No valid examples in dataset. Проверьте, что датасет содержит изображения "
            "и что поля (image/question/answer/caption/messages) указаны верно."
        )

    eval_dataset = None
    val_path = config.get("val_data_path")
    if val_path and Path(val_path).exists():
        eval_dataset = VLMJsonlDataset(
            val_path,
            prompt_for_caption=str(config.get("caption_prompt", "Describe this image.")),
            field_map=field_map,
        )

    collator = VLMDataCollator(
        processor,
        max_length=seq_len,
        assistant_only_loss=bool(config.get("assistant_only_loss", True)),
        base_dir=config.get("data_base_dir"),
    )
    steps_per_epoch = max(1, len(train_dataset) // max(1, batch_size))
    total_steps = max(1, int(steps_per_epoch * num_epochs) // max(1, gradient_accumulation))
    if max_steps > 0:
        total_steps = max_steps
    metrics_logger.update(total_steps=total_steps, status="training")

    from transformers import Trainer, TrainerCallback, TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, int(config.get("eval_batch_size", batch_size))),
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=log_every,
        save_steps=save_every,
        eval_steps=eval_steps if eval_dataset and eval_steps > 0 else None,
        eval_strategy="steps" if eval_dataset and eval_steps > 0 else "no",
        save_total_limit=save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=int(config.get("num_workers", 0)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        lr_scheduler_type=lr_schedule,
        optim=optimizer,
        max_steps=max_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    class MetricsCallback(TrainerCallback):
        def __init__(self, ml: MetricsLogger, every: int):
            self.ml = ml
            self.every = max(1, every)
            self.step_t0 = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            # Обновляем шаг/время КАЖДЫЙ шаг, чтобы прогресс-бар в UI не «замирал»
            # между логами loss (loss пишется раз в log_every шагов).
            step = int(state.global_step)
            if step <= 0:
                return
            self.ml.update(
                current_step=step,
                elapsed_seconds=time.time() - self.ml.start_timestamp,
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            step = int(state.global_step)
            if step <= 0 or step % self.every != 0:
                return
            loss = logs.get("loss")
            lr = logs.get("learning_rate", 0.0)
            if loss is not None:
                step_time = time.time() - self.step_t0
                self.ml.log_step(
                    step,
                    float(loss),
                    float(lr),
                    samples_per_sec=args.per_device_train_batch_size * args.gradient_accumulation_steps / max(step_time, 1e-6),
                    step_time=step_time,
                )
            if "eval_loss" in logs:
                self.ml.update(current_val_loss=float(logs["eval_loss"]))
            self.step_t0 = time.time()

        def on_save(self, args, state, control, **kwargs):
            ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ckpt_dir.exists():
                self.ml.log_checkpoint(str(ckpt_dir), step=state.global_step, loss=self.ml.metrics.get("current_loss"))

    trainer.add_callback(MetricsCallback(metrics_logger, log_every))

    trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(config))
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    mark_training_completed(metrics_logger, str(final_dir))
    metrics_logger.log_checkpoint(str(final_dir), loss=metrics_logger.metrics.get("current_loss"))


def main() -> None:
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
