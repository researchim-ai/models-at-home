import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class SFTDataset(IterableDataset):
    """
    Датасет для Supervised Fine-Tuning с гибкой настройкой.
    Поддерживает Chat и Instruct форматы.
    
    Возвращает dict: input_ids, attention_mask, labels
    labels замаскированы так, чтобы обучаться только на ответах ассистента.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        seq_len: int = 2048,
        sft_columns: Optional[dict] = None,
        sft_template: Optional[dict] = None,
        num_replicas: int = 1,
        rank: int = 0,
        split: str = "train",      # "train" | "val"
        val_ratio: float = 0.0,    # если >0, часть строк уходит в val
        shard: bool = True,        # для val поставим False
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.split = split
        self.val_ratio = float(val_ratio or 0.0)
        self.shard = bool(shard)
        
        # Определяем, является ли файл gzip-архивом
        suffixes = self.file_path.suffixes
        self._is_gz = suffixes and suffixes[-1] == ".gz"

        # Дефолтные настройки
        self.cols = sft_columns or {"format": "instruct", "instruction": "instruction", "output": "output"}
        self.tmpl = sft_template or {
            "system": "You are a helpful assistant.",
            "user_tag": "### User:",
            "bot_tag": "### Assistant:",
            "separator": "\n\n",
        }

        # Убедимся что есть pad token
        if self.tokenizer.pad_token is None:
            # лучше падать eos, чем None
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Статистика для отладки
        self.stats = {
            "total_lines": 0,
            "parsed": 0,
            "skipped": 0,
            "valid_examples": 0,
        }

    # ---------------------------------------------------------------------
    # Path helper (совместим с UI путями)
    # ---------------------------------------------------------------------

    def _get_nested(self, data: Any, path: str):
        """
        Получить значение из вложенной структуры по пути.
        
        Поддерживает:
        - 'a.b.c'
        - 'messages[0].content'
        - 'messages[].content' -> список значений
        - суффиксы вида ' [список ...]' -> игнорируются
        """
        if not path:
            return None

        # убрать суффиксы типа " [список из N эл.]" / " [список]"
        path = re.sub(r" \[список.*?\]$", "", path).strip()
        if not path:
            return None

        # messages[].content
        if "[]." in path:
            list_path, rest = path.split("[].", 1)
            lst = self._get_nested(data, list_path)
            if isinstance(lst, list):
                out = []
                for item in lst:
                    out.append(self._get_nested(item, rest))
                return out
            return None

        # messages[3].content
        m = re.search(r"\[(\d+)\]", path)
        if m:
            idx = int(m.group(1))
            base = path[: m.start()]
            tail = path[m.end() :]
            if tail.startswith("."):
                tail = tail[1:]
            base_val = self._get_nested(data, base)
            if isinstance(base_val, list) and 0 <= idx < len(base_val):
                return self._get_nested(base_val[idx], tail) if tail else base_val[idx]
            return None

        # обычный dot path
        cur = data
        for key in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                return None
            if cur is None:
                return None
        return cur

    def _as_text(self, v: Any) -> str:
        """Нормализует значение в строку (списки склеивает)."""
        if v is None:
            return ""
        if isinstance(v, list):
            # склеиваем непустые
            parts = []
            for x in v:
                s = "" if x is None else str(x)
                if s.strip():
                    parts.append(s)
            return "\n".join(parts)
        return str(v)

    # ---------------------------------------------------------------------
    # Prompt formatting (+ spans)
    # ---------------------------------------------------------------------

    def _segments_to_text_and_spans(self, segments: List[Tuple[str, bool]]) -> Tuple[str, List[Tuple[int, int]]]:
        """Склеивает сегменты в текст и возвращает char-spans для trainable частей."""
        spans: List[Tuple[int, int]] = []
        cur = 0
        parts: List[str] = []
        for seg_text, trainable in segments:
            if not seg_text:
                continue
            parts.append(seg_text)
            if trainable:
                spans.append((cur, cur + len(seg_text)))
            cur += len(seg_text)
        return "".join(parts), spans

    def _format_prompt_chat(self, data: Dict[str, Any]) -> Optional[str]:
        """Обратная совместимость: возвращает только текст."""
        text, _ = self._format_prompt_chat_with_spans(data)
        return text

    def _format_prompt_chat_with_spans(self, data: Dict[str, Any]) -> Tuple[Optional[str], List[Tuple[int, int]]]:
        msg_path = self.cols.get("messages_path") or self.cols.get("messages", "messages")
        msgs = self._get_nested(data, msg_path)
        if not msgs:
            return None, []
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                return None, []
        if not isinstance(msgs, list) or not msgs:
            return None, []

        role_field = self.cols.get("role_field", "role")
        content_field = self.cols.get("content_field", "content")
        role_system = self.cols.get("role_system", "system")
        role_user = self.cols.get("role_user", "user")
        role_assistant = self.cols.get("role_assistant", "assistant")

        sep = self.tmpl.get("separator", "\n\n")
        default_system = self.tmpl.get("system", "You are a helpful assistant.")
        sys_text = default_system

        # ищем системное сообщение
        for m in msgs:
            if isinstance(m, dict) and str(m.get(role_field, "")) == role_system:
                c = self._as_text(m.get(content_field, ""))
                if c.strip():
                    sys_text = c
                break

        # segments: (text, trainable?)
        segments: List[Tuple[str, bool]] = [(f"{sys_text}{sep}", False)]

        has_asst = False
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get(role_field, ""))
            content = self._as_text(m.get(content_field, ""))

            if role == role_system:
                continue
            if role == role_user:
                segments.append((f"{self.tmpl['user_tag']}\n{content}{sep}", False))
            elif role == role_assistant:
                # тег ассистента НЕ учим (без \n)
                segments.append((f"{self.tmpl['bot_tag']}", False))
                # учим только сам ответ (+ \n перед ним и sep после)
                # ВАЖНО: \n переносим в trainable часть, чтобы первый токен ответа не маскировался
                segments.append((f"\n{content}{sep}", True))
                has_asst = True

        if not has_asst:
            return None, []

        # EOS — обучаемый
        segments.append((self.tokenizer.eos_token, True))
        text, spans = self._segments_to_text_and_spans(segments)
        return text, spans

    def _format_prompt_instruct(self, data: Dict[str, Any]) -> Optional[str]:
        """Обратная совместимость: возвращает только текст."""
        text, _ = self._format_prompt_instruct_with_spans(data)
        return text

    def _format_prompt_instruct_with_spans(self, data: Dict[str, Any]) -> Tuple[Optional[str], List[Tuple[int, int]]]:
        instr_path = self.cols.get("instruction", "instruction")
        out_path = self.cols.get("output", "output")
        instr = self._as_text(self._get_nested(data, instr_path))
        out = self._as_text(self._get_nested(data, out_path))

        if not instr.strip() and not out.strip():
            return None, []

        default_system = self.tmpl.get("system", "You are a helpful assistant.")
        sys_val = default_system

        sys_field = self.cols.get("system_field")
        if sys_field:
            v = self._as_text(self._get_nested(data, sys_field))
            if v.strip():
                sys_val = v

        sep = self.tmpl.get("separator", "\n\n")
        if not out.strip():
            return None, []

        segments: List[Tuple[str, bool]] = [
            (f"{sys_val}{sep}", False),
            (f"{self.tmpl['user_tag']}\n{instr}{sep}", False),
            (f"{self.tmpl['bot_tag']}", False),  # тег без \n
            (f"\n{out}{sep}", True),  # \n переносим в trainable часть
            (self.tokenizer.eos_token, True),
        ]
        text, spans = self._segments_to_text_and_spans(segments)
        return text, spans

    def _format_prompt(self, data: Dict[str, Any]) -> Optional[str]:
        fmt = self.cols.get("format", "instruct")
        if fmt == "chat":
            return self._format_prompt_chat(data)
        return self._format_prompt_instruct(data)

    def _format_prompt_with_spans(self, data: Dict[str, Any]) -> Tuple[Optional[str], List[Tuple[int, int]]]:
        fmt = self.cols.get("format", "instruct")
        if fmt == "chat":
            return self._format_prompt_chat_with_spans(data)
        return self._format_prompt_instruct_with_spans(data)

    # ---------------------------------------------------------------------
    # Public helper for UI metrics
    # ---------------------------------------------------------------------

    def get_sample_prompt(self, max_samples: int = 10) -> Optional[str]:
        """Получить пример промпта из датасета для отображения в UI.
        
        Использует reservoir sampling для больших файлов (безопасно для RAM).
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
                    
                    try:
                        data = json.loads(line)
                        prompt = self._format_prompt(data)
                        if not prompt or len(prompt.strip()) < 10:
                            continue
                        
                        # Reservoir sampling для больших файлов
                        if len(samples) < max_samples:
                            samples.append(prompt)
                        else:
                            # С вероятностью max_samples/(idx+1) заменяем случайный элемент
                            r = random.randint(0, idx)
                            if r < max_samples:
                                samples[r] = prompt
                        
                        # Останавливаемся после проверки достаточного количества строк
                        if idx >= max_samples * 10:
                            break
                    except (json.JSONDecodeError, Exception):
                        continue
            
            finally:
                f.close()
            
            if samples:
                # Возвращаем первый валидный семпл
                return samples[0]
        except Exception as e:
            logger.warning(f"Failed to get sample prompt: {e}")
            return None
        return None

    # ---------------------------------------------------------------------
    # IterableDataset
    # ---------------------------------------------------------------------

    def __iter__(self):
        # ВАЖНО: Шардирование делает accelerate.prepare(), мы только делаем train/val split
        # детерминированный split по глобальному idx
        scale = 10000
        threshold = int(self.val_ratio * scale)

        # Поддержка .gz файлов (включая .jsonl.gz)
        if self._is_gz:
            import gzip
            f = gzip.open(self.file_path, "rt", encoding="utf-8")
        else:
            f = open(self.file_path, "r", encoding="utf-8")
        
        try:
            for idx, line in enumerate(f):
                self.stats["total_lines"] += 1
                
                is_val = (threshold > 0) and ((idx % scale) < threshold)
                
                # Фильтруем по split (train/val)
                if self.split == "val" and not is_val:
                    continue
                if self.split == "train" and is_val:
                    continue

                try:
                    data = json.loads(line)
                    self.stats["parsed"] += 1
                except Exception:
                    self.stats["skipped"] += 1
                    continue

                fmt = self.cols.get("format", "instruct")
                text, train_spans = self._format_prompt_with_spans(data)

                if not text or len(text.strip()) < 10:
                    self.stats["skipped"] += 1
                    continue

                # ВАЖНО: корректный masking делаем через offsets (fast tokenizer)
                # Требуем fast tokenizer для SFT, иначе masking будет некорректным
                use_offsets = bool(getattr(self.tokenizer, "is_fast", False))
                if fmt in ("chat", "instruct") and not use_offsets:
                    raise ValueError(
                        f"SFTDataset requires a *fast* tokenizer for correct assistant-only masking. "
                        f"Current tokenizer: {type(self.tokenizer).__name__}. "
                        f"Please use AutoTokenizer.from_pretrained(..., use_fast=True) or a fast tokenizer class."
                    )
                enc = self.tokenizer(
                    text,
                    max_length=self.seq_len,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=False,
                    return_attention_mask=True,
                    return_offsets_mapping=use_offsets,
                )

                input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
                attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
                labels = torch.full_like(input_ids, fill_value=-100)

                # ---------------------------------------------------------
                # Masking: учим только assistant output
                # ---------------------------------------------------------

                if use_offsets:
                    offsets = enc.get("offset_mapping", [])
                    if offsets:
                        # offsets может быть списком списков или тензором, нормализуем
                        if isinstance(offsets, list) and len(offsets) > 0:
                            offsets_list = offsets[0] if isinstance(offsets[0], list) else offsets
                        else:
                            offsets_list = []
                        
                        # helper: токен trainable если он ПОЛНОСТЬЮ внутри любого train span
                        def is_trainable_token(start: int, end: int) -> bool:
                            if start >= end:
                                return False
                            for a, b in train_spans:
                                if start >= a and end <= b:
                                    return True
                            return False

                        for i, (s, e) in enumerate(offsets_list):
                            if i >= len(labels):
                                break
                            if is_trainable_token(int(s), int(e)):
                                labels[i] = input_ids[i]
                # Если use_offsets=False, то мы уже выбросили ValueError выше

                # не учим паддинг
                labels = labels.masked_fill(attention_mask == 0, -100)
                
                # ВАЖНО: EOS токен часто имеет offset (0,0) у fast tokenizers,
                # поэтому он не попадает в train_spans через offset_mapping.
                # Добавляем явную обработку EOS токена.
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None:
                    # Ищем последний валидный токен (не padding)
                    last_valid_idx = int(attention_mask.sum().item()) - 1
                    if last_valid_idx >= 0 and input_ids[last_valid_idx].item() == eos_id:
                        labels[last_valid_idx] = input_ids[last_valid_idx]

                # Если после масок не осталось обучаемых токенов — пропускаем
                if not (labels != -100).any():
                    self.stats["skipped"] += 1
                    continue

                # дошли сюда => пример валиден
                self.stats["valid_examples"] += 1

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
        finally:
            f.close()

    def __len__(self):
        """
        Возвращает приблизительное количество примеров, доступных ЭТОМУ ранку.
        (важно для расчёта max_train_steps в trainer_worker).
        
        ВАЖНО: Это приблизительная оценка, так как реальное количество валидных примеров
        может быть меньше из-за:
        - битых JSON строк
        - пустых/коротких примеров
        - примеров, где после masking не осталось обучаемых токенов
        - truncation, который может удалить весь trainable контент
        
        Учитывает val_ratio и split для корректной оценки длины на один rank.
        """
        if not hasattr(self, "_length"):
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
            # 
            # ПРИМЕЧАНИЕ: Реальное количество валидных примеров может быть меньше из-за skipped.
            # Если нужно точное значение - используйте max_steps вместо epochs.
            self._length = effective
            
            # Логируем предупреждение если есть статистика о skipped примерах
            if hasattr(self, 'stats') and self.stats.get("total_lines", 0) > 0:
                skipped_ratio = self.stats.get("skipped", 0) / max(1, self.stats.get("total_lines", 1))
                if skipped_ratio > 0.1:  # Если >10% примеров скипается
                    logger.warning(
                        f"SFTDataset: {skipped_ratio*100:.1f}% примеров пропущено. "
                        f"Реальная длина может быть меньше __len__()={self._length}. "
                        f"Рекомендуется использовать max_steps вместо epochs."
                    )
        
        return self._length
    
    def debug_masking(self, data: Dict[str, Any], max_tokens: int = 50) -> Optional[str]:
        """
        Отладочная функция для проверки корректности маскирования.
        
        Возвращает строку с декодированным текстом и пометками:
        - [MASKED] для токенов с labels=-100
        - [TRAIN] для токенов с labels!=-100 (обучаемые)
        
        Полезно для быстрой проверки, что обучается только ответ ассистента.
        """
        try:
            fmt = self.cols.get("format", "instruct")
            text, train_spans = self._format_prompt_with_spans(data)
            
            if not text:
                return None
            
            use_offsets = bool(getattr(self.tokenizer, "is_fast", False))
            if not use_offsets:
                return "⚠️ Fast tokenizer required for masking debug"
            
            enc = self.tokenizer(
                text,
                max_length=self.seq_len,
                truncation=True,
                padding="max_length",
                add_special_tokens=False,
                return_attention_mask=True,
                return_offsets_mapping=True,
            )
            
            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
            labels = torch.full_like(input_ids, fill_value=-100)
            
            offsets = enc.get("offset_mapping", [])
            if offsets:
                offsets_list = offsets[0] if isinstance(offsets[0], list) else offsets
                
                def is_trainable_token(start: int, end: int) -> bool:
                    if start >= end:
                        return False
                    for a, b in train_spans:
                        if start >= a and end <= b:
                            return True
                    return False
                
                for i, (s, e) in enumerate(offsets_list):
                    if i >= len(labels):
                        break
                    if is_trainable_token(int(s), int(e)):
                        labels[i] = input_ids[i]
            
            labels = labels.masked_fill(attention_mask == 0, -100)
            
            # EOS обработка
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                last_valid_idx = int(attention_mask.sum().item()) - 1
                if last_valid_idx >= 0 and input_ids[last_valid_idx].item() == eos_id:
                    labels[last_valid_idx] = input_ids[last_valid_idx]
            
            # Формируем отладочный вывод
            valid_len = int(attention_mask.sum().item())
            trainable_count = int((labels != -100).sum().item())
            
            result = []
            result.append(f"Total tokens: {valid_len}, Trainable: {trainable_count} ({trainable_count/valid_len*100:.1f}%)\n")
            result.append("=" * 60 + "\n")
            
            for i in range(min(valid_len, max_tokens)):
                token_id = input_ids[i].item()
                token_text = self.tokenizer.decode([token_id])
                is_trainable = labels[i].item() != -100
                
                marker = "[TRAIN]" if is_trainable else "[MASKED]"
                result.append(f"{i:3d} {marker:8s} {token_text!r}\n")
            
            if valid_len > max_tokens:
                result.append(f"... ({valid_len - max_tokens} more tokens)\n")
            
            result.append("=" * 60 + "\n")
            result.append(f"Trainable spans (char positions): {train_spans}\n")
            
            return "".join(result)
        except Exception as e:
            return f"Error in debug_masking: {e}"
