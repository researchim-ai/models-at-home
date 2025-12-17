import json
import re
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

class SFTDataset(IterableDataset):
    """
    Датасет для Supervised Fine-Tuning с гибкой настройкой.
    Поддерживает Chat и Instruct форматы.
    """
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, seq_len: int = 2048, 
                 sft_columns=None, sft_template=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Дефолтные настройки
        self.cols = sft_columns or {"format": "instruct", "instruction": "instruction", "output": "output"}
        self.tmpl = sft_template or {
            "system": "You are a helpful assistant.",
            "user_tag": "### User:",
            "bot_tag": "### Assistant:",
            "separator": "\n\n"
        }
        
        # Убедимся что есть pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_nested(self, data: dict, path: str):
        """Получить значение из вложенного словаря по пути.
        
        Поддерживает:
        - 'key.subkey' - вложенные словари
        - 'messages [список из N эл.]' - списки (убирает суффикс)
        """
        if not path:
            return None
        
        # Убираем суффиксы типа " [список из 3 эл.]"
        path = re.sub(r' \[список.*?\]$', '', path)
        
        keys = path.split('.')
        curr = data
        try:
            for k in keys:
                if isinstance(curr, dict):
                    curr = curr.get(k)
                elif isinstance(curr, list) and k.isdigit():
                    curr = curr[int(k)]
                else:
                    return None
                
                if curr is None:
                    return None
            return curr
        except Exception:
            return None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # Шардинг
                if worker_info is not None:
                    if idx % worker_info.num_workers != worker_info.id:
                        continue
                
                try:
                    data = json.loads(line)
                    text = ""
                    
                    # Определяем формат
                    fmt = self.cols.get("format", "instruct")
                    
                    # 1. Chat Format (Messages List)
                    if fmt == "chat":
                        # Получаем путь к списку сообщений
                        msg_path = self.cols.get("messages_path") or self.cols.get("messages", "messages")
                        msgs = self._get_nested(data, msg_path)
                        
                        if not msgs:
                            continue
                        
                        # Если messages это строка с JSON, парсим её
                        if isinstance(msgs, str):
                            try:
                                msgs = json.loads(msgs)
                            except:
                                continue
                        
                        if not isinstance(msgs, list) or not msgs:
                            continue
                        
                        # Получаем настройки маппинга полей
                        role_field = self.cols.get("role_field", "role")
                        content_field = self.cols.get("content_field", "content")
                        role_system = self.cols.get("role_system", "system")
                        role_user = self.cols.get("role_user", "user")
                        role_assistant = self.cols.get("role_assistant", "assistant")
                        
                        # Формируем текст
                        sys_text = self.tmpl.get("system", "")
                        user_texts = []
                        assistant_texts = []
                        
                        for m in msgs:
                            if not isinstance(m, dict):
                                continue
                            role = str(m.get(role_field, ""))
                            content = str(m.get(content_field, ""))
                            
                            if role == role_system:
                                sys_text = content
                            elif role == role_user:
                                user_texts.append(content)
                            elif role == role_assistant:
                                assistant_texts.append(content)
                        
                        if not user_texts and not assistant_texts:
                            continue
                        
                        # Строим промпт
                        sep = self.tmpl.get("separator", "\n\n")
                        text = f"{sys_text}{sep}" if sys_text else ""
                        
                        # Чередуем user/assistant
                        for i in range(max(len(user_texts), len(assistant_texts))):
                            if i < len(user_texts):
                                text += f"{self.tmpl['user_tag']}\n{user_texts[i]}{sep}"
                            if i < len(assistant_texts):
                                text += f"{self.tmpl['bot_tag']}\n{assistant_texts[i]}{sep}"
                        
                        text = text.rstrip(sep) + self.tokenizer.eos_token
                    
                    # 2. Instruct Format (отдельные поля)
                    else:
                        instr = self._get_nested(data, self.cols.get("instruction", "instruction")) or ""
                        out = self._get_nested(data, self.cols.get("output", "output")) or ""
                        
                        # System Prompt
                        sys_field = self.cols.get("system_field")
                        sys_val = self.tmpl.get("system", "")
                        if sys_field:
                            val_in_data = self._get_nested(data, sys_field)
                            if val_in_data:
                                sys_val = str(val_in_data)
                        
                        if not instr and not out:
                            continue
                        
                        sep = self.tmpl.get("separator", "\n\n")
                        text = f"{sys_val}{sep}" if sys_val else ""
                        text += f"{self.tmpl['user_tag']}\n{instr}{sep}"
                        text += f"{self.tmpl['bot_tag']}\n{out}{self.tokenizer.eos_token}"
                    
                    if not text or len(text.strip()) < 10:
                        continue
                        
                    # Токенизация
                    tokens = self.tokenizer(
                        text, 
                        max_length=self.seq_len, 
                        truncation=True, 
                        padding="max_length",
                        return_tensors="pt"
                    )
                    
                    input_ids = tokens["input_ids"].squeeze(0)
                    attention_mask = tokens["attention_mask"].squeeze(0)
                    
                    # Labels: маскируем паддинг
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                    
                except Exception as e:
                    # Пропускаем битые записи
                    continue
