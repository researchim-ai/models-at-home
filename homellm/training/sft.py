import json
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

class SFTDataset(IterableDataset):
    """
    Датасет для Supervised Fine-Tuning с гибкой настройкой.
    """
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, seq_len: int = 2048, 
                 sft_columns=None, sft_template=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Дефолтные настройки
        self.cols = sft_columns or {"format": "alpaca", "instruction": "instruction", "input": "input", "output": "output"}
        self.tmpl = sft_template or {
            "system": "",
            "user_tag": "### User:",
            "bot_tag": "### Assistant:",
            "separator": "\n\n"
        }
        
        # Убедимся что есть pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_nested(self, data: dict, path: str):
        """Получить значение из вложенного словаря по пути 'key.subkey'."""
        if not path:
            return None
        
        keys = path.split('.')
        curr = data
        try:
            for k in keys:
                if isinstance(curr, dict):
                    curr = curr.get(k)
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
                    
                    # Определяем формат: Chat или Alpaca
                    fmt = self.cols.get("format", "alpaca")
                    
                    # 1. Chat Format (Messages List)
                    if fmt == "chat":
                        msg_col = self.cols.get("messages", "messages")
                        msgs = self._get_nested(data, msg_col)
                        
                        if msgs:
                            # Если messages это строка с JSON, парсим её
                            if isinstance(msgs, str):
                                try:
                                    msgs = json.loads(msgs)
                                except:
                                    continue # Skip broken JSON
                            
                            if isinstance(msgs, list):
                                if self.tokenizer.chat_template:
                                    text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                                else:
                                    # Fallback на простой формат
                                    text = self.tmpl["system"] + self.tmpl["separator"]
                                    for m in msgs:
                                        role = m.get("role")
                                        content = m.get("content")
                                        if role == "user":
                                            text += f"{self.tmpl['user_tag']}\n{content}{self.tmpl['separator']}"
                                        elif role == "assistant":
                                            text += f"{self.tmpl['bot_tag']}\n{content}{self.tmpl['separator']}"
                                        elif role == "system":
                                            text = f"{content}{self.tmpl['separator']}" + text
                    
                    # 2. Alpaca Format (Columns)
                    else:
                        instr = self._get_nested(data, self.cols.get("instruction", "instruction")) or ""
                        inp = self._get_nested(data, self.cols.get("input", "input")) or ""
                        out = self._get_nested(data, self.cols.get("output", "output")) or ""
                        
                        # System Prompt: может быть задан в данных или статически в шаблоне
                        sys_field = self.cols.get("system_field")
                        sys_val = self.tmpl["system"]
                        if sys_field:
                            # Если поле есть в данных, берем оттуда
                            val_in_data = self._get_nested(data, sys_field)
                            if val_in_data:
                                sys_val = str(val_in_data)
                        
                        if not instr and not out:
                            continue
                            
                        # Формируем промпт вручную
                        # System -> User (Instr + Input) -> Bot (Output)
                        
                        prompt = f"{sys_val}{self.tmpl['separator']}"
                        
                        user_content = str(instr)
                        if inp:
                            user_content += f"\n{inp}"
                            
                        prompt += f"{self.tmpl['user_tag']}\n{user_content}{self.tmpl['separator']}"
                        prompt += f"{self.tmpl['bot_tag']}\n{out}{self.tokenizer.eos_token}"
                        
                        text = prompt
                    
                    if not text:
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
                    
                except Exception:
                    continue
