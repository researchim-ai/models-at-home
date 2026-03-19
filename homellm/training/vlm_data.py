from __future__ import annotations

import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.request import urlopen

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_image(image_ref: str, base_dir: Optional[Path] = None) -> Optional[Image.Image]:
    path = str(image_ref).strip()
    if not path:
        return None
    if path.startswith(("http://", "https://")):
        try:
            with urlopen(path, timeout=15) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load image URL %s: %s", path, exc)
            return None
    if path.startswith("file://"):
        path = path[7:]
    if base_dir is not None and not os.path.isabs(path):
        path = str(base_dir / path)
    if not os.path.exists(path):
        logger.warning("Image not found: %s", path)
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to open image %s: %s", path, exc)
        return None


def _normalize_content(content: Any) -> Any:
    if isinstance(content, list):
        normalized = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    normalized.append({"type": "text", "text": str(item["text"])})
                elif item.get("type") == "image" or "image" in item or "image_url" in item:
                    image_value = item.get("image") or item.get("image_url")
                    normalized.append({"type": "image", "image": image_value})
                else:
                    normalized.append(item)
            else:
                normalized.append({"type": "text", "text": str(item)})
        return normalized
    if isinstance(content, dict):
        return [content]
    return str(content)


def _assistant_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif not isinstance(item, dict):
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
    return str(content)


def _build_messages_from_record(record: Dict[str, Any], prompt_for_caption: str) -> List[Dict[str, Any]]:
    if isinstance(record.get("messages"), list):
        messages = record["messages"]
    elif isinstance(record.get("conversations"), list):
        messages = record["conversations"]
    elif record.get("caption"):
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_for_caption}]},
            {"role": "assistant", "content": [{"type": "text", "text": str(record["caption"])}]},
        ]
    elif record.get("question") and (record.get("answer") or record.get("response") or record.get("output")):
        answer = record.get("answer") or record.get("response") or record.get("output")
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": str(record["question"])}]},
            {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]},
        ]
    elif record.get("prompt") and (record.get("completion") or record.get("response") or record.get("answer")):
        answer = record.get("completion") or record.get("response") or record.get("answer")
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": str(record["prompt"])}]},
            {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]},
        ]
    else:
        messages = []

    normalized_messages: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _normalize_content(message.get("content", ""))
        normalized_messages.append({"role": role, "content": content})
    return normalized_messages


def _extract_image_refs(record: Dict[str, Any]) -> List[str]:
    images: List[str] = []
    if record.get("images"):
        if isinstance(record["images"], list):
            images.extend(str(item) for item in record["images"] if item)
    if record.get("image"):
        images.append(str(record["image"]))
    if record.get("image_path"):
        images.append(str(record["image_path"]))
    if record.get("input_image_path"):
        value = record["input_image_path"]
        if isinstance(value, list):
            images.extend(str(item) for item in value if item)
        else:
            images.append(str(value))
    deduped: List[str] = []
    seen = set()
    for image_ref in images:
        if image_ref not in seen:
            seen.add(image_ref)
            deduped.append(image_ref)
    return deduped


def _extract_target_text(record: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    for key in ("answer", "target", "response", "output", "caption", "completion"):
        if record.get(key):
            return str(record[key])
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _assistant_text_from_content(message.get("content"))
    return ""


class VLMJsonlDataset(Dataset):
    """Schema-aware VLM dataset with support for single and multi-image examples."""

    def __init__(
        self,
        file_path: Union[str, Path],
        *,
        prompt_for_caption: str = "Describe this image.",
    ):
        self.file_path = Path(file_path)
        self.base_dir = self.file_path.parent
        self.prompt_for_caption = prompt_for_caption
        self.examples: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image_refs = _extract_image_refs(record)
                messages = _build_messages_from_record(record, self.prompt_for_caption)
                if not image_refs or not messages:
                    continue
                self.examples.append(
                    {
                        "messages": messages,
                        "image_refs": image_refs,
                        "target_text": _extract_target_text(record, messages),
                        "metadata": {
                            "id": record.get("id"),
                            "task_name": record.get("task_name"),
                            "schema": record.get("schema") or "auto",
                        },
                    }
                )
        logger.info("Loaded %d VLM examples from %s", len(self.examples), self.file_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class VLMDataCollator:
    """Processor-driven collator that preserves every multimodal tensor returned by the processor."""

    def __init__(
        self,
        processor: Any,
        *,
        max_length: int = 2048,
        assistant_only_loss: bool = True,
        base_dir: Optional[Union[str, Path]] = None,
    ):
        self.processor = processor
        self.max_length = max_length
        self.assistant_only_loss = assistant_only_loss
        self.base_dir = Path(base_dir) if base_dir else None

    def _build_text(self, messages: Sequence[Dict[str, Any]]) -> str:
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=False,
            )
        parts = []
        for message in messages:
            parts.append(f"{message.get('role', 'user')}: {_assistant_text_from_content(message.get('content'))}")
        return "\n".join(parts)

    def _prompt_messages(self, messages: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "assistant":
                return messages[:idx]
        return messages

    def _build_inputs(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        images_batch: List[Any] = []

        for example in examples:
            images = [load_image(image_ref, self.base_dir) for image_ref in example["image_refs"]]
            images = [img for img in images if img is not None]
            if not images:
                continue
            texts.append(self._build_text(example["messages"]))
            images_batch.append(images[0] if len(images) == 1 else images)

        if not texts:
            raise ValueError("All images failed to load")

        return self.processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def _collect_special_token_ids(self) -> List[int]:
        token_ids = set()
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return []
        for token_name in (
            "boi_token",
            "eoi_token",
            "image_token",
            "image_pad_token",
            "vision_start_token",
            "vision_end_token",
            "video_token",
            "video_pad_token",
        ):
            token = getattr(tokenizer, token_name, None)
            if token:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and token_id >= 0:
                    token_ids.add(int(token_id))
        for literal in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<image>", "<|video_pad|>"):
            token_id = tokenizer.convert_tokens_to_ids(literal)
            if token_id is not None and token_id >= 0 and token_id != tokenizer.unk_token_id:
                token_ids.add(int(token_id))
        return sorted(token_ids)

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        examples = [example for example in examples if example]
        if not examples:
            raise ValueError("Empty batch after filtering None")

        batch = self._build_inputs(examples)
        labels = batch["input_ids"].clone()
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        for token_id in self._collect_special_token_ids():
            labels[labels == token_id] = -100
        # Common Qwen image placeholder id used in HF examples.
        labels[labels == 262144] = -100

        if self.assistant_only_loss and tokenizer is not None and hasattr(self.processor, "apply_chat_template"):
            for row_idx, example in enumerate(examples):
                prompt_messages = self._prompt_messages(example["messages"])
                prompt_text = self.processor.apply_chat_template(
                    list(prompt_messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_ids = tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].squeeze(0)
                prompt_len = min(len(prompt_ids), labels[row_idx].shape[0])
                labels[row_idx, :prompt_len] = -100

        batch["labels"] = labels
        return batch
