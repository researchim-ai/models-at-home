"""
Dr.GRPO (Understanding R1-Zero) конфигурация.

Dr.GRPO — улучшенная версия GRPO с фиксами:
- Без деления на std (устраняет reward hacking)
- Фиксированный length normalizer (устраняет length bias)

Формула advantage:
    A_i = r_i - mean(r)  (без деления на std!)

Paper: "Understanding R1-Zero-Like Training: A Critical Perspective"
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .base import RLAlgorithm
from .grpo import GRPOConfig


@dataclass
class DrGRPOConfig(GRPOConfig):
    """
    Конфигурация для Dr.GRPO алгоритма.
    
    Dr.GRPO расширяет GRPO:
    - use_std_normalization = False (не делим на std)
    - fixed_length_normalizer для устранения length bias
    
    Attributes:
        fixed_length_normalizer: Фиксированный делитель для loss нормализации.
            По умолчанию = max_new_tokens. Устраняет bias в сторону коротких ответов.
    """
    
    # ============================================================
    # DR.GRPO СПЕЦИФИЧНЫЕ ПАРАМЕТРЫ
    # ============================================================
    
    # Фиксированный делитель для нормализации loss
    # Вместо деления на фактическую длину ответа делим на константу
    fixed_length_normalizer: Optional[int] = None
    
    # Override: Dr.GRPO НЕ делит на std
    use_std_normalization: bool = False
    
    # Liger loss type для Dr.GRPO
    liger_grpo_loss_type: str = "dr_grpo"
    
    def __post_init__(self):
        """Устанавливает fixed_length_normalizer если не задан."""
        if self.fixed_length_normalizer is None:
            self.fixed_length_normalizer = self.max_new_tokens
    
    @property
    def algorithm(self) -> RLAlgorithm:
        return RLAlgorithm.DRGRPO
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфиг в словарь."""
        result = super().to_dict()
        result.update({
            "fixed_length_normalizer": self.fixed_length_normalizer,
        })
        return result
