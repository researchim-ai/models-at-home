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
    
    def _format_prompt(self, data: dict) -> str:
        """Формирует промпт из данных (без токенизации). Возвращает None если не удалось."""
        try:
            fmt = self.cols.get("format", "instruct")
            
            # 1. Chat Format
            if fmt == "chat":
                msg_path = self.cols.get("messages_path") or self.cols.get("messages", "messages")
                msgs = self._get_nested(data, msg_path)
                
                if not msgs:
                    return None
                
                if isinstance(msgs, str):
                    try:
                        msgs = json.loads(msgs)
                    except:
                        return None
                
                if not isinstance(msgs, list) or not msgs:
                    return None
                
                role_field = self.cols.get("role_field", "role")
                content_field = self.cols.get("content_field", "content")
                role_system = self.cols.get("role_system", "system")
                role_user = self.cols.get("role_user", "user")
                role_assistant = self.cols.get("role_assistant", "assistant")
                
                default_system = self.tmpl.get("system", "You are a helpful assistant.")
                sys_text = default_system
                user_texts = []
                assistant_texts = []
                
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get(role_field, ""))
                    content = str(m.get(content_field, ""))
                    
                    if role == role_system and content.strip():
                        sys_text = content
                    elif role == role_user:
                        user_texts.append(content)
                    elif role == role_assistant:
                        assistant_texts.append(content)
                
                if not user_texts and not assistant_texts:
                    return None
                
                sep = self.tmpl.get("separator", "\n\n")
                text = f"{sys_text}{sep}"
                
                for i in range(max(len(user_texts), len(assistant_texts))):
                    if i < len(user_texts):
                        text += f"{self.tmpl['user_tag']}\n{user_texts[i]}{sep}"
                    if i < len(assistant_texts):
                        text += f"{self.tmpl['bot_tag']}\n{assistant_texts[i]}{sep}"
                
                text += self.tokenizer.eos_token
                return text
            
            # 2. Instruct Format
            else:
                instr = self._get_nested(data, self.cols.get("instruction", "instruction")) or ""
                out = self._get_nested(data, self.cols.get("output", "output")) or ""
                
                default_system = self.tmpl.get("system", "You are a helpful assistant.")
                sys_val = default_system
                
                sys_field = self.cols.get("system_field")
                if sys_field:
                    val_in_data = self._get_nested(data, sys_field)
                    if val_in_data and str(val_in_data).strip():
                        sys_val = str(val_in_data)
                
                if not instr and not out:
                    return None
                
                sep = self.tmpl.get("separator", "\n\n")
                text = f"{sys_val}{sep}"
                text += f"{self.tmpl['user_tag']}\n{instr}{sep}"
                text += f"{self.tmpl['bot_tag']}\n{out}{sep}{self.tokenizer.eos_token}"
                return text
                
        except Exception:
            return None
    
    def get_sample_prompt(self, max_samples: int = 10) -> str:
        """Получить пример сформированного промпта из случайного семпла.
        
        Args:
            max_samples: Максимальное количество семплов для проверки
            
        Returns:
            Строка с примером промпта или None если не удалось найти валидный семпл
        """
        import random
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # Пробуем найти валидный семпл
        attempts = min(max_samples, len(lines))
        random_indices = random.sample(range(len(lines)), attempts)
        
        for idx in random_indices:
            try:
                data = json.loads(lines[idx])
                prompt = self._format_prompt(data)
                if prompt and len(prompt.strip()) >= 10:
                    return prompt
            except Exception:
                continue
        
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
                        # Дефолтный system prompt из конфига (всегда используем для консистентности с инференсом)
                        default_system = self.tmpl.get("system", "You are a helpful assistant.")
                        sys_text = default_system  # Начинаем с дефолтного
                        user_texts = []
                        assistant_texts = []
                        
                        for m in msgs:
                            if not isinstance(m, dict):
                                continue
                            role = str(m.get(role_field, ""))
                            content = str(m.get(content_field, ""))
                            
                            if role == role_system and content.strip():
                                sys_text = content  # Переопределяем если есть в данных
                            elif role == role_user:
                                user_texts.append(content)
                            elif role == role_assistant:
                                assistant_texts.append(content)
                        
                        if not user_texts and not assistant_texts:
                            continue
                        
                        # Строим промпт
                        # Формат: {system}\n\n{user_tag}\n{user}\n\n{bot_tag}\n{assistant}\n\n<eos>
                        sep = self.tmpl.get("separator", "\n\n")
                        text = f"{sys_text}{sep}"  # System всегда есть (как при инференсе)
                        
                        # Чередуем user/assistant
                        for i in range(max(len(user_texts), len(assistant_texts))):
                            if i < len(user_texts):
                                text += f"{self.tmpl['user_tag']}\n{user_texts[i]}{sep}"
                            if i < len(assistant_texts):
                                text += f"{self.tmpl['bot_tag']}\n{assistant_texts[i]}{sep}"
                        
                        text += self.tokenizer.eos_token
                    
                    # 2. Instruct Format (отдельные поля)
                    else:
                        instr = self._get_nested(data, self.cols.get("instruction", "instruction")) or ""
                        out = self._get_nested(data, self.cols.get("output", "output")) or ""
                        
                        # System Prompt (всегда используем для консистентности с инференсом)
                        default_system = self.tmpl.get("system", "You are a helpful assistant.")
                        sys_val = default_system  # Начинаем с дефолтного
                        
                        sys_field = self.cols.get("system_field")
                        if sys_field:
                            val_in_data = self._get_nested(data, sys_field)
                            if val_in_data and str(val_in_data).strip():
                                sys_val = str(val_in_data)  # Переопределяем если есть в данных
                        
                        if not instr and not out:
                            continue
                        
                        # Формат: {system}\n\n{user_tag}\n{instr}\n\n{bot_tag}\n{out}\n\n<eos>
                        sep = self.tmpl.get("separator", "\n\n")
                        text = f"{sys_val}{sep}"  # System всегда есть (как при инференсе)
                        text += f"{self.tmpl['user_tag']}\n{instr}{sep}"
                        text += f"{self.tmpl['bot_tag']}\n{out}{sep}{self.tokenizer.eos_token}"
                    
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    # ВАЖНО: В SFT мы обучаем ТОЛЬКО на ответе ассистента
                    # Нужно найти позицию начала ответа ассистента в тексте
                    bot_tag = self.tmpl.get('bot_tag', '### Assistant:')
                    
                    # Находим позицию тега ассистента в тексте (до токенизации)
                    bot_tag_pos = text.find(bot_tag)
                    
                    if bot_tag_pos == -1:
                        # Если не нашли тег, пропускаем этот пример
                        continue
                    
                    # Текст до ответа ассистента (системный промпт + вопрос + тег ассистента)
                    text_before_assistant = text[:bot_tag_pos + len(bot_tag)]
                    
                    # Токенизация всего текста
                    tokens = self.tokenizer(
                        text, 
                        max_length=self.seq_len, 
                        truncation=True, 
                        padding="max_length",
                        return_tensors="pt"
                    )
                    
                    input_ids = tokens["input_ids"].squeeze(0)
                    attention_mask = tokens["attention_mask"].squeeze(0)
                    
                    # Токенизируем часть до ответа ассистента с теми же параметрами
                    # чтобы точно определить позицию начала ответа
                    tokens_before = self.tokenizer(
                        text_before_assistant,
                        max_length=self.seq_len,
                        truncation=True,
                        padding=False,  # Без паддинга для точного подсчета
                        return_tensors="pt"
                    )
                    assistant_start_token_idx = len(tokens_before["input_ids"].squeeze(0))
                    
                    # Labels: маскируем все до начала ответа ассистента
                    labels = input_ids.clone()
                    # Маскируем системный промпт, вопрос пользователя и тег ассистента
                    # Обучаем ТОЛЬКО на самом ответе ассистента
                    if assistant_start_token_idx < len(labels):
                        labels[:assistant_start_token_idx] = -100
                    
                    # Маскируем паддинг
                    labels[attention_mask == 0] = -100
                    
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                    
                except Exception as e:
                    # Пропускаем битые записи
                    continue
