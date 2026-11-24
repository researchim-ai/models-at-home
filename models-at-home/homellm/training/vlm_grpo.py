from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

from .vlm_common import (
    MetricsLogger,
    apply_vlm_freeze_policy,
    apply_vlm_lora,
    load_vlm_model,
    load_vlm_processor,
    mark_training_completed,
)
from .vlm_data import VLMDataCollator, VLMJsonlDataset, load_image

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _compute_reward(response: str, target: str) -> float:
    response_norm = _normalize_text(response)
    target_norm = _normalize_text(target)
    if not response_norm:
        return -1.0
    reward = 0.0
    if target_norm:
        if response_norm == target_norm:
            reward += 1.0
        if target_norm in response_norm:
            reward += 0.5
    reward += min(len(response_norm.split()), 64) / 256.0
    return reward


def _prompt_messages(messages: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return messages[:idx]
    return messages


def _prepare_generation_batch(processor: Any, examples: Sequence[Dict[str, Any]], base_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    texts: List[str] = []
    images: List[Any] = []
    valid_examples: List[Dict[str, Any]] = []
    for example in examples:
        prompt_messages = _prompt_messages(example["messages"])
        text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        loaded_images = [load_image(image_ref, base_dir) for image_ref in example["image_refs"]]
        loaded_images = [img for img in loaded_images if img is not None]
        if not loaded_images:
            continue
        texts.append(text)
        images.append(loaded_images[0] if len(loaded_images) == 1 else loaded_images)
        valid_examples.append(example)
    if not texts:
        raise ValueError("No valid multimodal examples in batch")
    return valid_examples, processor(text=texts, images=images, padding=True, return_tensors="pt")


def run_vlm_grpo(config: Dict[str, Any], metrics_logger: MetricsLogger) -> None:
    if config.get("stage") not in (None, "vlm_grpo"):
        raise ValueError(f"vlm_grpo worker supports only stage=vlm_grpo, got {config.get('stage')}")

    model_name_or_path = config.get("base_model_path") or config.get("model_name_or_path")
    data_path = config.get("data_path")
    if not model_name_or_path:
        raise ValueError("base_model_path or model_name_or_path required")
    if not data_path or not Path(data_path).exists():
        raise ValueError("data_path must point to an existing JSONL file")

    output_dir = Path(config.get("output_dir", "out/vlm_grpo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(config.get("data_base_dir") or Path(data_path).parent)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = VLMJsonlDataset(data_path, prompt_for_caption=str(config.get("caption_prompt", "Describe this image.")))
    if len(train_dataset) == 0:
        raise ValueError("No valid examples in dataset")

    processor = load_vlm_processor(model_name_or_path, config)
    model = load_vlm_model(model_name_or_path, config, device)
    apply_vlm_freeze_policy(model, config)
    model = apply_vlm_lora(model, config)
    model.train()

    batch_size = int(config.get("batch_size", 1))
    num_epochs = int(config.get("num_epochs", 1))
    learning_rate = float(config.get("learning_rate", 5e-6))
    max_new_tokens = int(config.get("max_new_tokens", 128))
    temperature = float(config.get("temperature", 0.7))
    log_every = int(config.get("log_every", 10))
    save_every = int(config.get("save_every", 100))

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    collator = VLMDataCollator(
        processor,
        max_length=int(config.get("seq_len", 2048)),
        assistant_only_loss=True,
        base_dir=base_dir,
    )
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    total_steps = max(1, math.ceil(len(dataloader) * max(1, num_epochs)))
    metrics_logger.update(status="training", stage="vlm_grpo", total_steps=total_steps)
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device

    global_step = 0
    for _epoch in range(num_epochs):
        for batch_examples in dataloader:
            global_step += 1
            step_t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if step_t0 is not None:
                step_t0.record()

            valid_examples, generation_inputs = _prepare_generation_batch(processor, batch_examples, base_dir)
            generation_inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in generation_inputs.items()
            }
            with torch.no_grad():
                generated_ids = model.generate(
                    **generation_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 0.1),
                )

            completions = processor.batch_decode(
                [
                    out_ids[int(attn_mask.sum().item()) :]
                    for out_ids, attn_mask in zip(generated_ids, generation_inputs["attention_mask"])
                ],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            rewards = [
                _compute_reward(response, example.get("target_text", ""))
                for example, response in zip(valid_examples, completions)
            ]
            best_examples = []
            for example, response, reward in zip(valid_examples, completions, rewards):
                tuned_example = {
                    "messages": list(_prompt_messages(example["messages"]))
                    + [{"role": "assistant", "content": [{"type": "text", "text": response}]}],
                    "image_refs": example["image_refs"],
                    "target_text": response,
                    "metadata": example.get("metadata", {}),
                }
                if reward > 0:
                    best_examples.append(tuned_example)
                metrics_logger.log_sample_output(
                    processor.apply_chat_template(_prompt_messages(example["messages"]), tokenize=False, add_generation_prompt=True)[:300],
                    response,
                    reward=reward,
                )

            if not best_examples:
                metrics_logger.log_step(global_step, 0.0, learning_rate, reward=sum(rewards) / max(1, len(rewards)), step_time=0.0)
                continue

            optimizer.zero_grad(set_to_none=True)
            train_batch = collator(best_examples)
            train_batch = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in train_batch.items()
            }
            outputs = model(**train_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            step_time_sec = 0.0
            if step_t0 is not None:
                step_t1 = torch.cuda.Event(enable_timing=True)
                step_t1.record()
                torch.cuda.synchronize()
                step_time_sec = step_t0.elapsed_time(step_t1) / 1000.0

            avg_reward = float(sum(rewards) / max(1, len(rewards)))
            metrics_logger.log_step(
                global_step,
                float(loss.detach().cpu().item()),
                learning_rate,
                reward=avg_reward,
                samples_per_sec=len(best_examples) / max(step_time_sec, 1e-6) if step_time_sec else 0.0,
                step_time=step_time_sec,
            )

            if global_step % save_every == 0:
                ckpt_dir = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(str(ckpt_dir))
                processor.save_pretrained(str(ckpt_dir))
                metrics_logger.log_checkpoint(str(ckpt_dir), step=global_step, loss=float(loss.detach().cpu().item()))

            if global_step % log_every == 0:
                logger.info("VLM GRPO step=%s loss=%.4f reward=%.4f", global_step, float(loss), avg_reward)

    final_dir = output_dir / "final_model"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    mark_training_completed(metrics_logger, str(final_dir))
    metrics_logger.log_checkpoint(str(final_dir), step=global_step, loss=metrics_logger.metrics.get("current_loss"))


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
    run_vlm_grpo(config, metrics_logger)


if __name__ == "__main__":
    main()
