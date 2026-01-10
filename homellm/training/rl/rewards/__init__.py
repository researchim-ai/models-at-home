"""
Reward функции для GRPO.

Модульная система для разных типов задач:
- math: Математические задачи (GSM8K, MATH)
- format: Проверка формата reasoning
- code: Задачи на код
- universal: Универсальные правила из Reward Designer
"""

from .base import RewardFunction, CombinedReward, UniversalRuleReward
from .math import MathReward, GSM8KReward
from .format import FormatReward, ReasoningQualityReward

__all__ = [
    "RewardFunction",
    "CombinedReward",
    "UniversalRuleReward",
    "MathReward",
    "GSM8KReward", 
    "FormatReward",
    "ReasoningQualityReward",
]
