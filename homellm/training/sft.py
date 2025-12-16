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
        self.cols = sft_columns or {"instruction": "instruction", "input": "input", "output": "output"}
        self.tmpl = sft_template or {
            "system": "",
            "user_tag": "### User:",
            "bot_tag": "### Assistant:",
            "separator": "\n\n"
        }
        
        # Убедимся что есть pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
                    
                    # 1. Если есть поле 'messages' (ChatML), используем chat_template
                    if "messages" in data:
                        if self.tokenizer.chat_template:
                            text = self.tokenizer.apply_chat_template(data["messages"], tokenize=False, add_generation_prompt=False)
                        else:
                            # Fallback на простой формат
                            msgs = data["messages"]
                            text = self.tmpl["system"] + self.tmpl["separator"]
                            for m in msgs:
                                role = m["role"]
                                content = m["content"]
                                if role == "user":
                                    text += f"{self.tmpl['user_tag']}\n{content}{self.tmpl['separator']}"
                                elif role == "assistant":
                                    text += f"{self.tmpl['bot_tag']}\n{content}{self.tmpl['separator']}"
                                elif role == "system":
                                    text = f"{content}{self.tmpl['separator']}" + text
                    
                    # 2. Иначе используем маппинг колонок (Alpaca style)
                    else:
                        instr = data.get(self.cols.get("instruction", "instruction"), "")
                        inp = data.get(self.cols.get("input", "input"), "")
                        out = data.get(self.cols.get("output", "output"), "")
                        
                        if not instr and not out:
                            continue
                            
                        # Формируем промпт вручную
                        # System -> User (Instr + Input) -> Bot (Output)
                        
                        prompt = f"{self.tmpl['system']}{self.tmpl['separator']}"
                        
                        user_content = instr
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
