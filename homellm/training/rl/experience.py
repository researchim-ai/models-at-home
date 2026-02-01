"""
Experience буфер для GRPO.

Хранит опыт от rollout'ов для последующего обучения.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional, List
import torch
import torch.nn.functional as F


def zero_pad_sequences(
    sequences: List[torch.Tensor], 
    side: str = "left",
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Паддинг последовательностей до одинаковой длины.
    
    Args:
        sequences: Список тензоров разной длины
        side: "left" или "right" паддинг
        pad_value: Значение для паддинга
        
    Returns:
        Батч тензоров одинаковой длины
    """
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        if pad_len > 0:
            if side == "left":
                padding = (pad_len, 0)
            else:
                padding = (0, pad_len)
            padded = F.pad(seq, padding, value=pad_value)
        else:
            padded = seq
        padded_sequences.append(padded)
        
    return torch.stack(padded_sequences, dim=0)


@dataclass
class Experience:
    """
    Один опыт взаимодействия (completion для промпта).
    
    Attributes:
        sequences: Полная последовательность (prompt + completion) [seq_len]
        prompt_length: Длина промпта (для разделения)
        action_log_probs: Log-вероятности токенов completion по текущей политике [seq_len-1]
        log_probs_ref: Log-вероятности по референсной модели (для KL) [seq_len-1]
        returns: Накопленная награда для этого completion [1]
        advantages: Вычисленное преимущество (advantage) [1] или [seq_len-1]
        attention_mask: Маска внимания [seq_len]
        action_mask: Маска действий (только completion) [seq_len-1]
        kl: KL дивергенция [seq_len-1]
        prompt_id: ID промпта (для группировки)
    """
    sequences: torch.Tensor
    prompt_length: int
    action_log_probs: torch.Tensor
    log_probs_ref: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    action_mask: Optional[torch.Tensor] = None
    kl: Optional[torch.Tensor] = None
    prompt_id: Optional[int] = None
    prompt_ids: Optional[List[int]] = None  # Список prompt_ids для batch (для SDPO)
    completion_text: Optional[str] = None  # Для отладки
    
    def to(self, device: torch.device) -> "ReplayBuffer":
        """Перемещает все тензоры на указанное устройство."""
        members = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[f.name] = v
        return Experience(**members)
    
    @property
    def completion_length(self) -> int:
        """Длина completion (ответа)."""
        return self.sequences.size(-1) - self.prompt_length
    
    @property
    def total_length(self) -> int:
        """Полная длина последовательности."""
        return self.sequences.size(-1)


def split_experience_batch(experience: Experience) -> List[Experience]:
    """Разбивает батч Experience на список отдельных Experience."""
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]
    
    tensor_keys = (
        "sequences", "action_log_probs", "log_probs_ref",
        "returns", "advantages", "attention_mask", "action_mask", "kl"
    )
    
    for key in tensor_keys:
        value = getattr(experience, key)
        if value is None:
            vals = [None] * batch_size
        else:
            vals = torch.unbind(value)
        for i, v in enumerate(vals):
            batch_data[i][key] = v
    
    # Скалярные значения (копируем как есть)
    for i in range(batch_size):
        batch_data[i]["prompt_length"] = experience.prompt_length
        batch_data[i]["prompt_id"] = experience.prompt_id
        batch_data[i]["completion_text"] = None
    
    return [Experience(**data) for data in batch_data]


def join_experience_batch(items: List[Experience]) -> Experience:
    """Объединяет список Experience в один батч с паддингом."""
    batch_data = {}
    
    tensor_keys = (
        "sequences", "action_log_probs", "log_probs_ref",
        "returns", "advantages", "attention_mask", "action_mask", "kl"
    )
    
    for key in tensor_keys:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            # Определяем значение паддинга
            pad_value = 0
            if key == "action_mask" or key == "attention_mask":
                pad_value = 0  # Маски паддим нулями
            
            data = zero_pad_sequences(vals, side="left", pad_value=pad_value)
        else:
            data = None
        batch_data[key] = data
    
    # Для prompt_length берём первый (должны быть одинаковые в группе)
    batch_data["prompt_length"] = items[0].prompt_length if items else 0
    batch_data["prompt_id"] = items[0].prompt_id if items else None
    # 🎓 SDPO: сохраняем все prompt_ids для batch
    batch_data["prompt_ids"] = [item.prompt_id for item in items] if items else None
    batch_data["completion_text"] = None
    
    return Experience(**batch_data)


class ReplayBuffer:
    """
    Буфер для хранения опыта.
    
    Поддерживает:
    - Добавление опыта (батч или по одному)
    - Итерацию по батчам
    - Фильтрацию zero-gradient групп (для dynamic sampling)
    """
    
    def __init__(self, limit: int = 0) -> None:
        """
        Args:
            limit: Максимальный размер буфера (0 = без ограничения)
        """
        self.limit = limit
        self.items: List[Experience] = []
        
        # Для группировки по prompt_id
        self._prompt_groups: dict = {}
    
    def append(self, experience: Experience) -> None:
        """Добавляет опыт (батч разбивается на отдельные элементы)."""
        if experience.sequences.dim() > 1:
            # Батч -> разбиваем на отдельные элементы
            items = split_experience_batch(experience)
            self.items.extend(items)
        else:
            self.items.append(experience)
        
        # Применяем лимит
        if self.limit > 0 and len(self.items) > self.limit:
            samples_to_remove = len(self.items) - self.limit
            self.items = self.items[samples_to_remove:]
    
    def append_group(
        self, 
        experiences: List[Experience], 
        prompt_id: int,
        filter_zero_gradient: bool = False,
    ) -> bool:
        """
        Добавляет группу опытов для одного промпта.
        
        Args:
            experiences: Список опытов (G completions для одного промпта)
            prompt_id: ID промпта
            filter_zero_gradient: Фильтровать группы с нулевым градиентом
            
        Returns:
            True если группа добавлена, False если отфильтрована
        """
        if filter_zero_gradient:
            # Проверяем, что не все rewards одинаковые
            returns = torch.stack([exp.returns for exp in experiences])
            if returns.max() == returns.min():
                return False  # Zero-gradient группа
        
        for exp in experiences:
            exp.prompt_id = prompt_id
            self.items.append(exp)
        
        self._prompt_groups[prompt_id] = len(experiences)
        return True
    
    def clear(self) -> None:
        """Очищает буфер."""
        self.items.clear()
        self._prompt_groups.clear()
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]
    
    def get_stats(self) -> dict:
        """Возвращает статистику буфера."""
        if not self.items:
            return {"size": 0, "num_groups": 0}
        
        returns = torch.stack([item.returns for item in self.items if item.returns is not None])
        
        return {
            "size": len(self.items),
            "num_groups": len(self._prompt_groups),
            "returns_mean": returns.mean().item() if len(returns) > 0 else 0,
            "returns_std": returns.std().item() if len(returns) > 0 else 0,
            "returns_max": returns.max().item() if len(returns) > 0 else 0,
            "returns_min": returns.min().item() if len(returns) > 0 else 0,
        }
