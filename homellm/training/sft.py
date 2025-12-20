import json
import math
import random
import re
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer


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
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.split = split
        self.val_ratio = float(val_ratio or 0.0)
        self.shard = bool(shard)

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
    # Prompt formatting
    # ---------------------------------------------------------------------

    def _format_prompt_chat(self, data: Dict[str, Any]) -> Optional[str]:
        msg_path = self.cols.get("messages_path") or self.cols.get("messages", "messages")
        msgs = self._get_nested(data, msg_path)
        if not msgs:
            return None
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                return None
        if not isinstance(msgs, list) or not msgs:
            return None

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

        parts: List[str] = [f"{sys_text}{sep}"]

        has_asst = False
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get(role_field, ""))
            content = self._as_text(m.get(content_field, ""))

            if role == role_system:
                continue
            if role == role_user:
                parts.append(f"{self.tmpl['user_tag']}\n{content}{sep}")
            elif role == role_assistant:
                parts.append(f"{self.tmpl['bot_tag']}\n{content}{sep}")
                has_asst = True

        if not has_asst:
            return None

        parts.append(self.tokenizer.eos_token)
        return "".join(parts)

    def _format_prompt_instruct(self, data: Dict[str, Any]) -> Optional[str]:
        instr_path = self.cols.get("instruction", "instruction")
        out_path = self.cols.get("output", "output")
        instr = self._as_text(self._get_nested(data, instr_path))
        out = self._as_text(self._get_nested(data, out_path))

        if not instr.strip() and not out.strip():
            return None

        default_system = self.tmpl.get("system", "You are a helpful assistant.")
        sys_val = default_system

        sys_field = self.cols.get("system_field")
        if sys_field:
            v = self._as_text(self._get_nested(data, sys_field))
            if v.strip():
                sys_val = v

        sep = self.tmpl.get("separator", "\n\n")
        # Важно: добавляем \n после bot_tag, чтобы граница ответа была однозначной
        text = (
            f"{sys_val}{sep}"
            f"{self.tmpl['user_tag']}\n{instr}{sep}"
            f"{self.tmpl['bot_tag']}\n{out}{sep}"
            f"{self.tokenizer.eos_token}"
        )
        return text

    def _format_prompt(self, data: Dict[str, Any]) -> Optional[str]:
        fmt = self.cols.get("format", "instruct")
        if fmt == "chat":
            return self._format_prompt_chat(data)
        return self._format_prompt_instruct(data)

    # ---------------------------------------------------------------------
    # Public helper for UI metrics
    # ---------------------------------------------------------------------

    def get_sample_prompt(self, max_samples: int = 10) -> Optional[str]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return None

            attempts = min(max_samples, len(lines))
            for idx in random.sample(range(len(lines)), attempts):
                try:
                    data = json.loads(lines[idx])
                    prompt = self._format_prompt(data)
                    if prompt and len(prompt.strip()) >= 10:
                        return prompt
                except Exception:
                    continue
        except Exception:
            return None
        return None

    # ---------------------------------------------------------------------
    # IterableDataset
    # ---------------------------------------------------------------------

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # shard only if enabled
        total_readers = self.num_replicas
        reader_id = self.rank
        
        if self.shard and worker_info is not None:
            total_readers *= num_workers
            reader_id = self.rank * num_workers + worker_id
        elif not self.shard:
            total_readers = 1
            reader_id = 0

        # детерминированный split по глобальному idx
        scale = 10000
        threshold = int(self.val_ratio * scale)
        seen = 0  # счетчик только выбранного split (важно!)

        with open(self.file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                self.stats["total_lines"] += 1
                
                is_val = (threshold > 0) and ((idx % scale) < threshold)
                
                if self.split == "val" and not is_val:
                    continue
                if self.split == "train" and is_val:
                    continue

                # shard по seen (а не по idx), чтобы равномернее
                if self.shard:
                    if (seen % total_readers) != reader_id:
                        seen += 1
                        continue
                    seen += 1

                try:
                    data = json.loads(line)
                    self.stats["parsed"] += 1
                except Exception:
                    self.stats["skipped"] += 1
                    continue

                fmt = self.cols.get("format", "instruct")
                text = self._format_prompt(data)

                if not text or len(text.strip()) < 10:
                    self.stats["skipped"] += 1
                    continue

                # Токенизация полного текста
                # ВАЖНО: add_special_tokens=False, т.к. мы уже вручную добавили eos_token в текст
                tokens = self.tokenizer(
                    text,
                    max_length=self.seq_len,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=False,
                    return_tensors="pt",
                )

                input_ids = tokens["input_ids"].squeeze(0)
                attention_mask = tokens["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                
                # Инициализируем labels: сначала всё маскируем, потом размаскируем обучаемые части
                labels.fill_(-100)

                # ---------------------------------------------------------
                # Masking: учим только assistant output
                # ---------------------------------------------------------

                if fmt == "chat":
                    # Для chat-формата мы строим более точную маску:
                    # замаскировать всё, а затем "размаскировать" только ассистентские сегменты.
                    # Для этого пересоберем сегменты как в форматере.

                    msg_path = self.cols.get("messages_path") or self.cols.get("messages", "messages")
                    msgs = self._get_nested(data, msg_path)
                    if isinstance(msgs, str):
                        try:
                            msgs = json.loads(msgs)
                        except Exception:
                            self.stats["skipped"] += 1
                            continue

                    if not isinstance(msgs, list) or not msgs:
                        self.stats["skipped"] += 1
                        continue

                    role_field = self.cols.get("role_field", "role")
                    content_field = self.cols.get("content_field", "content")
                    role_system = self.cols.get("role_system", "system")
                    role_user = self.cols.get("role_user", "user")
                    role_assistant = self.cols.get("role_assistant", "assistant")

                    sep = self.tmpl.get("separator", "\n\n")
                    default_system = self.tmpl.get("system", "You are a helpful assistant.")
                    sys_text = default_system

                    for m in msgs:
                        if isinstance(m, dict) and str(m.get(role_field, "")) == role_system:
                            c = self._as_text(m.get(content_field, ""))
                            if c.strip():
                                sys_text = c
                            break

                    # сегменты (text, trainable)
                    segments: List[tuple[str, bool]] = [(f"{sys_text}{sep}", False)]

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
                            # ТЕГ ассистента — это часть промпта, не обучаем
                            segments.append((f"{self.tmpl['bot_tag']}\n", False))
                            # Обучаем только сам ответ ассистента
                            segments.append((f"{content}{sep}", True))
                            has_asst = True

                    if not has_asst:
                        self.stats["skipped"] += 1
                        continue
                    
                    # Добавляем EOS токен как trainable (чтобы модель училась завершаться)
                    segments.append((self.tokenizer.eos_token, True))

                    # теперь размаскируем только trainable сегменты по токен-границам
                    cur = 0
                    for seg_text, trainable in segments:
                        seg_ids = self.tokenizer(seg_text, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
                        seg_len = int(seg_ids.numel())
                        if seg_len <= 0:
                            continue
                        end = min(cur + seg_len, labels.numel())
                        if trainable and cur < labels.numel():
                            # обучаемся на тех же токенах, что и в input_ids
                            labels[cur:end] = input_ids[cur:end]
                        cur += seg_len
                        if cur >= labels.numel():
                            break
                    
                    # Не обучаемся на паддинге
                    labels = labels.masked_fill(attention_mask == 0, -100)

                else:
                    # -------- Instruct masking --------
                    instr_path = self.cols.get("instruction", "instruction")
                    out_path = self.cols.get("output", "output")
                    instr = self._as_text(self._get_nested(data, instr_path))
                    out = self._as_text(self._get_nested(data, out_path))
                    
                    # если ответа нет — смысла учить нет
                    if not out.strip():
                        self.stats["skipped"] += 1
                        continue
                    
                    default_system = self.tmpl.get("system", "You are a helpful assistant.")
                    sys_val = default_system
                    
                    sys_field = self.cols.get("system_field")
                    if sys_field:
                        v = self._as_text(self._get_nested(data, sys_field))
                        if v.strip():
                            sys_val = v
                    
                    sep = self.tmpl.get("separator", "\n\n")
                    
                    # ВАЖНО: маску строим в токен-границах так же, как собирали prompt в _format_prompt_instruct
                    # сегменты (text, trainable)
                    # ТЕГ ассистента не обучаем, обучаем только сам out (+ eos)
                    segments: List[tuple[str, bool]] = [
                        (f"{sys_val}{sep}", False),
                        (f"{self.tmpl['user_tag']}\n{instr}{sep}", False),
                        (f"{self.tmpl['bot_tag']}\n", False),
                        (f"{out}{sep}", True),
                        (self.tokenizer.eos_token, True),
                    ]
                    
                    cur = 0
                    for seg_text, trainable in segments:
                        seg_ids = self.tokenizer(
                            seg_text,
                            add_special_tokens=False,
                            return_tensors="pt",
                        )["input_ids"].squeeze(0)
                        seg_len = int(seg_ids.numel())
                        if seg_len <= 0:
                            continue
                        end = min(cur + seg_len, labels.numel())
                        if trainable and cur < labels.numel():
                            labels[cur:end] = input_ids[cur:end]
                        cur += seg_len
                        if cur >= labels.numel():
                            break
                    
                    # Не обучаемся на паддинге
                    labels = labels.masked_fill(attention_mask == 0, -100)

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

    def __len__(self):
        """
        Возвращает приблизительное количество примеров, доступных ЭТОМУ ранку.
        (важно для расчёта max_train_steps в trainer_worker).
        """
        if not hasattr(self, "_length"):
            cnt = 0
            with open(self.file_path, "r", encoding="utf-8") as f:
                for _ in f:
                    cnt += 1
            # распределяем по replicas примерно поровну
            self._length = int(math.ceil(cnt / self.num_replicas))
        return self._length
