"""
homellm.training.rl
===================
Модуль обучения с подкреплением (RL) для LLM.

Поддерживаемые алгоритмы:
- GRPO (Group Relative Policy Optimization)
- Dr.GRPO (GRPO Done Right) - без деления на std, фиксированная нормализация
- DAPO (Decoupled Clip + Dynamic Sampling)

Основные компоненты:
- GRPOTrainer: Основной класс для обучения
- GRPOConfig: Конфигурация для настройки всех параметров
- RewardFunction: Базовый класс для reward функций
"""

from .config import GRPOConfig, RLAlgorithm
from .experience import Experience, ReplayBuffer
from .loss import GRPOLoss, compute_advantages
from .trainer import GRPOTrainer
from .rollout import Rollout, generate_rollouts

__all__ = [
    # Config
    "GRPOConfig",
    "RLAlgorithm",
    # Experience
    "Experience", 
    "ReplayBuffer",
    # Loss
    "GRPOLoss",
    "compute_advantages",
    # Trainer
    "GRPOTrainer",
    # Rollout
    "Rollout",
    "generate_rollouts",
]
