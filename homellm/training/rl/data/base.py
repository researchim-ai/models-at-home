"""
Базовый класс для RL датасетов.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


@dataclass
class RLSample:
    """
    Один пример для RL обучения.
    
    Attributes:
        prompt: Промпт (вопрос/задача)
        reference_answer: Эталонный ответ
        metadata: Дополнительные данные
    """
    prompt: str
    reference_answer: str
    metadata: Optional[Dict[str, Any]] = None


class RLDataset(Dataset):
    """
    Базовый класс для датасетов RL обучения.
    
    Наследники могут реализовать:
    - load_data: Загрузка данных в self.samples
    - format_prompt: Форматирование промпта
    
    Или передать samples напрямую в конструктор.
    """
    
    def __init__(
        self,
        samples: Optional[List[RLSample]] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
        reasoning_format: str = "deepseek",
    ):
        """
        Args:
            samples: Готовый список примеров (опционально)
            split: "train" или "test"
            max_samples: Максимальное количество примеров
            reasoning_format: Формат reasoning тегов
        """
        self.split = split
        self.max_samples = max_samples
        self.reasoning_format = reasoning_format
        self.samples: List[RLSample] = samples or []
    
    def load_data(self) -> None:
        """Загружает данные в self.samples. Переопределите в наследниках."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> RLSample:
        return self.samples[idx]
    
    def format_prompt(self, sample: RLSample) -> str:
        """
        Форматирует промпт с системным сообщением.
        
        По умолчанию использует формат вопрос-ответ.
        Переопределите для кастомных форматов.
        """
        return sample.prompt
    
    def get_batch(self, indices: List[int]) -> List[RLSample]:
        """Получает батч примеров по индексам."""
        return [self.samples[i] for i in indices]
    
    def iter_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[List[RLSample]]:
        """Итератор по батчам."""
        indices = list(range(len(self.samples)))
        if shuffle:
            import random
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield self.get_batch(batch_indices)
    
    def get_prompts_and_answers(self) -> tuple:
        """Возвращает списки промптов и ответов."""
        prompts = [self.format_prompt(s) for s in self.samples]
        answers = [s.reference_answer for s in self.samples]
        return prompts, answers


class SimplePromptDataset(RLDataset):
    """
    Простой датасет из списка пар (prompt, answer).
    
    Удобно для создания кастомных датасетов.
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        prompt_key: str = "question",
        answer_key: str = "answer",
        **kwargs,
    ):
        """
        Args:
            data: Список словарей с промптами и ответами
            prompt_key: Ключ для промпта
            answer_key: Ключ для ответа
        """
        super().__init__(**kwargs)
        self.data = data
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.load_data()
    
    def load_data(self) -> None:
        """Загружает данные из списка."""
        for item in self.data:
            sample = RLSample(
                prompt=item[self.prompt_key],
                reference_answer=item[self.answer_key],
                metadata=item,
            )
            self.samples.append(sample)
        
        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]


class JSONLDataset(RLDataset):
    """
    Датасет из JSONL файла.
    """
    
    def __init__(
        self,
        file_path: str,
        prompt_key: str = "question",
        answer_key: str = "answer",
        **kwargs,
    ):
        """
        Args:
            file_path: Путь к JSONL файлу
            prompt_key: Ключ для промпта
            answer_key: Ключ для ответа
        """
        super().__init__(**kwargs)
        self.file_path = file_path
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.load_data()
    
    def load_data(self) -> None:
        """Загружает данные из JSONL файла."""
        import json
        from pathlib import Path
        
        file_path = Path(self.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                sample = RLSample(
                    prompt=item.get(self.prompt_key, ""),
                    reference_answer=item.get(self.answer_key, ""),
                    metadata=item,
                )
                self.samples.append(sample)
                
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
