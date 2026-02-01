"""
DAPO (Dynamic sampling Asymmetric clipping Policy Optimization) конфигурация.

DAPO (GRPO++) — улучшенная версия GRPO из ByteDance:
- Clip Higher: асимметричный клиппинг [0.8, 1.28] вместо [0.8, 1.2]
- Dynamic Sampling: фильтрация zero-gradient групп + добор новых
- Token-level Loss: каждый токен имеет равный вес
- Overlong Penalty: штраф за слишком длинные ответы

Paper: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
"""
from dataclasses import dataclass
from typing import Dict, Any

from .base import RLAlgorithm
from .drgrpo import DrGRPOConfig


@dataclass
class DAPOConfig(DrGRPOConfig):
    """
    Конфигурация для DAPO алгоритма.
    
    DAPO расширяет Dr.GRPO:
    - Асимметричный клиппинг (clip_eps_high = 0.28)
    - Dynamic sampling для фильтрации zero-gradient групп
    - Token-level loss агрегация
    - Overlong penalty
    
    Attributes:
        dynamic_sampling: Фильтровать группы где все rewards одинаковые (zero gradient)
        max_refill_rounds: Максимум попыток добора новых групп при фильтрации
        token_level_loss: Использовать token-level агрегацию вместо sample-level
        overlong_penalty: Штраф за ответы превышающие max_new_tokens
        overlong_buffer: Буферная зона перед штрафом
    """
    
    # ============================================================
    # DAPO СПЕЦИФИЧНЫЕ ПАРАМЕТРЫ
    # ============================================================
    
    # Clip Higher: увеличенный верхний порог клиппинга
    clip_eps_high: float = 0.28
    
    # Dynamic Sampling: фильтрация zero-gradient групп
    dynamic_sampling: bool = True
    max_refill_rounds: int = 3  # 0 = без добора, 8 = как в статье
    
    # Token-level Loss: каждый токен имеет равный вес
    token_level_loss: bool = True
    
    # Overlong Penalty: штраф за длинные ответы
    overlong_penalty: float = -1.0
    overlong_buffer: int = 0  # 4000 в оригинальном DAPO
    
    # Liger loss type для DAPO
    liger_grpo_loss_type: str = "dapo"
    
    @property
    def algorithm(self) -> RLAlgorithm:
        return RLAlgorithm.DAPO
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфиг в словарь."""
        result = super().to_dict()
        result.update({
            "dynamic_sampling": self.dynamic_sampling,
            "max_refill_rounds": self.max_refill_rounds,
            "token_level_loss": self.token_level_loss,
            "overlong_penalty": self.overlong_penalty,
            "overlong_buffer": self.overlong_buffer,
        })
        return result
