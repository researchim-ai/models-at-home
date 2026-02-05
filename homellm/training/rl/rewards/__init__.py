"""
Reward функции для GRPO/SDPO.

Модульная система для разных типов задач:
- math: Математические задачи (GSM8K, MATH)
- format: Проверка формата reasoning
- code: Задачи на код
- universal: Универсальные правила из Reward Designer

🔥 SDPO Support: RewardResult содержит feedback для self-distillation.
"""

from .base import RewardFunction, CombinedReward, UniversalRuleReward, RewardResult
from .math import MathReward, GSM8KReward, MathExpressionReward
from .format import FormatReward, ReasoningQualityReward

__all__ = [
    # Core
    "RewardFunction",
    "RewardResult",
    "CombinedReward",
    "UniversalRuleReward",
    # Math
    "MathReward",
    "GSM8KReward",
    "MathExpressionReward",
    # Format
    "FormatReward",
    "ReasoningQualityReward",
]
