"""
GRPO (Group Relative Policy Optimization) конфигурация.

GRPO — базовый алгоритм из DeepSeek-R1:
- PPO-style surrogate loss с клиппингом
- Групповая нормализация advantages (mean ± std)
- KL регуляризация (опционально)

Формула advantage:
    A_i = (r_i - mean(r)) / std(r)
"""
from dataclasses import dataclass
from typing import Dict, Any

from .base import BaseRLConfig, RLAlgorithm


@dataclass
class GRPOConfig(BaseRLConfig):
    """
    Конфигурация для стандартного GRPO алгоритма.
    
    GRPO использует:
    - PPO-style clipped surrogate loss
    - Нормализацию advantages по std (деление на стандартное отклонение)
    - Опциональную KL регуляризацию
    
    Attributes:
        clip_eps_low: Нижняя граница клиппинга ratio
        clip_eps_high: Верхняя граница клиппинга ratio
        kl_weight: Вес KL-штрафа (0 для reasoning-RL)
        use_std_normalization: Делить advantages на std (True для GRPO)
    """
    
    # ============================================================
    # КЛИППИНГ (PPO-style)
    # ============================================================
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2
    
    # ============================================================
    # KL РЕГУЛЯРИЗАЦИЯ
    # ============================================================
    kl_weight: float = 0.0  # Для reasoning-RL обычно 0
    
    # ============================================================
    # НОРМАЛИЗАЦИЯ ADVANTAGES
    # ============================================================
    use_std_normalization: bool = True  # True для GRPO, False для DrGRPO
    
    # ============================================================
    # LIGER FUSED LOSS
    # ============================================================
    # Для GRPO используем стандартный GRPO loss type
    liger_fused_grpo: bool = True
    liger_grpo_loss_type: str = "grpo"
    
    @property
    def algorithm(self) -> RLAlgorithm:
        return RLAlgorithm.GRPO
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфиг в словарь."""
        result = super().to_dict()
        result.update({
            "clip_eps_low": self.clip_eps_low,
            "clip_eps_high": self.clip_eps_high,
            "kl_weight": self.kl_weight,
            "use_std_normalization": self.use_std_normalization,
            "liger_fused_grpo": self.liger_fused_grpo,
            "liger_grpo_loss_type": self.liger_grpo_loss_type,
        })
        return result
