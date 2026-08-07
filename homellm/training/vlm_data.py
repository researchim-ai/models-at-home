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


def _coerce_text(value: Any) -> str:
    """Normalize a text field that may arrive as a multilingual dict, a list, a
    stringified dict, or a scalar into a single readable string.

    Some HF datasets (e.g. DocVQA) store ``question``/``answer`` as a Python-dict
    string like ``"{'en': '...', 'de': '...'}"`` — we prefer the English value.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("en", "english", "text", "value"):
            if value.get(key):
                return str(value[key]).strip()
        for v in value.values():
            if v:
                return str(v).strip()
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _coerce_text(item)
            if text:
                return text
        return ""
    text = str(value).strip()
    if text.startswith(("{", "[")):
        try:
            import ast

            parsed = ast.literal_eval(text)
            if isinstance(parsed, (dict, list, tuple)):
                return _coerce_text(parsed)
        except Exception:
            pass
    return text


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


def _build_messages_with_field_map(
    record: Dict[str, Any], prompt_for_caption: str, field_map: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Build the conversation using an explicit UI-provided field mapping."""
    fmt = field_map.get("format")
    if fmt == "messages":
        field = field_map.get("messages_field") or "messages"
        raw = record.get(field)
        return list(raw) if isinstance(raw, list) else []
    if fmt == "caption":
        caption = _coerce_text(record.get(field_map.get("caption_field") or "caption"))
        if not caption:
            return []
        prompt = field_map.get("caption_prompt") or prompt_for_caption
        return [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ]
    # default: VQA / instruction pair
    question = _coerce_text(record.get(field_map.get("question_field") or "question"))
    answer = _coerce_text(record.get(field_map.get("answer_field") or "answer"))
    if not question or not answer:
        return []
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def _build_messages_from_record(
    record: Dict[str, Any], prompt_for_caption: str, field_map: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    if field_map and field_map.get("format"):
        messages = _build_messages_with_field_map(record, prompt_for_caption, field_map)
        normalized_messages: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = _normalize_content(message.get("content", ""))
            normalized_messages.append({"role": role, "content": content})
        return normalized_messages
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


def _extract_image_refs(record: Dict[str, Any], field_map: Optional[Dict[str, Any]] = None) -> List[str]:
    if field_map and field_map.get("image_field"):
        value = record.get(field_map["image_field"])
        if isinstance(value, list):
            refs = [str(item) for item in value if item]
        elif value:
            refs = [str(value)]
        else:
            refs = []
        if refs:
            return refs
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
        field_map: Optional[Dict[str, Any]] = None,
    ):
        self.file_path = Path(file_path)
        self.base_dir = self.file_path.parent
        self.prompt_for_caption = prompt_for_caption
        self.field_map = field_map or None
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
                image_refs = _extract_image_refs(record, self.field_map)
                messages = _build_messages_from_record(record, self.prompt_for_caption, self.field_map)
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
        # Right padding keeps the prompt at the front so assistant-only masking
        # (labels[:prompt_len] = -100) is correct.
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right"

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

    def _load_example_images(self, example: Dict[str, Any]) -> List[Image.Image]:
        images = [load_image(image_ref, self.base_dir) for image_ref in example["image_refs"]]
        return [img for img in images if img is not None]

    @staticmethod
    def _images_arg(images: List[Image.Image]) -> Any:
        return images[0] if len(images) == 1 else images

    def _build_inputs(self, prepared: Sequence[tuple]) -> Dict[str, torch.Tensor]:
        # NOTE: no truncation here. Multimodal processors expand the <image>
        # placeholder into many tokens; truncating would desync text vs pixel
        # values and raise "Mismatch in image token count".
        texts = [text for (_example, _images, text) in prepared]
        images_batch = [self._images_arg(images) for (_example, images, _text) in prepared]
        return self.processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
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

        prepared: List[tuple] = []
        for example in examples:
            images = self._load_example_images(example)
            if not images:
                continue
            prepared.append((example, images, self._build_text(example["messages"])))
        if not prepared:
            raise ValueError("All images failed to load")

        batch = self._build_inputs(prepared)
        labels = batch["input_ids"].clone()
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        for token_id in self._collect_special_token_ids():
            labels[labels == token_id] = -100
        # Common Qwen image placeholder id used in HF examples.
        labels[labels == 262144] = -100

        if self.assistant_only_loss and tokenizer is not None and hasattr(self.processor, "apply_chat_template"):
            self._mask_prompt_tokens(labels, prepared)

        batch["labels"] = labels
        return batch

    def _mask_prompt_tokens(self, labels: torch.Tensor, prepared: Sequence[tuple]) -> None:
        """Mask prompt tokens (set -100) so loss is computed on the answer only.

        The prompt length is measured by re-running the *processor* on the
        prompt-only turn together with the same image(s); this accounts for the
        expanded image placeholder tokens (a plain tokenizer would undercount).
        Right padding guarantees the prompt sits at the front of each row.
        """
        for row_idx, (example, images, _text) in enumerate(prepared):
            prompt_messages = self._prompt_messages(example["messages"])
            if not prompt_messages:
                continue
            try:
                prompt_text = self.processor.apply_chat_template(
                    list(prompt_messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_inputs = self.processor(
                    text=[prompt_text],
                    images=[self._images_arg(images)],
                    return_tensors="pt",
                    padding=False,
                )
                prompt_len = int(prompt_inputs["input_ids"].shape[1])
            except Exception as exc:  # be resilient to odd templates
                logger.warning("assistant-only masking failed, training on full text: %s", exc)
                continue
            prompt_len = min(prompt_len, labels[row_idx].shape[0])
            labels[row_idx, :prompt_len] = -100
