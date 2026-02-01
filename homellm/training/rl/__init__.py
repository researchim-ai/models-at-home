"""
homellm.training.rl
===================
Модуль обучения с подкреплением (RL) для LLM.

Поддерживаемые алгоритмы:
- GRPO (Group Relative Policy Optimization)
- Dr.GRPO (GRPO Done Right) - без деления на std, фиксированная нормализация
- DAPO (Decoupled Clip + Dynamic Sampling)
- SDPO (Self-Distilled Policy Optimization) - self-distillation + reprompting

Основные компоненты:
- GRPOTrainer: Основной класс для обучения
- Конфиги: GRPOConfig, DrGRPOConfig, DAPOConfig, SDPOConfig
- RewardFunction: Базовый класс для reward функций

Использование (новый стиль - рекомендуется):
    from homellm.training.rl.config import SDPOConfig, DAPOConfig
    config = SDPOConfig(learning_rate=5e-6, alpha=0.5)

Использование (старый стиль - backward compatible):
    from homellm.training.rl import GRPOConfig, RLAlgorithm
    config = GRPOConfig(algorithm=RLAlgorithm.SDPO)
"""

# Новая модульная система конфигов
from .config import (
    RLAlgorithm,
    BaseRLConfig,
    GRPOConfig as GRPOConfigNew,
    DrGRPOConfig,
    DAPOConfig,
    SDPOConfig,
    create_config,
    get_config_class,
    ALGORITHM_CONFIGS,
)

# Backward compatible GRPOConfig (универсальный конфиг со всеми параметрами)
# Используем legacy_config.py для обратной совместимости со старым кодом
from .legacy_config import GRPOConfig

from .experience import Experience, ReplayBuffer
from .loss import GRPOLoss, SDPOLoss, compute_advantages, create_loss_function
from .trainer import GRPOTrainer
from .rollout import Rollout, generate_rollouts

__all__ = [
    # Algorithm enum
    "RLAlgorithm",
    
    # New modular configs (рекомендуется)
    "BaseRLConfig",
    "GRPOConfigNew",
    "DrGRPOConfig",
    "DAPOConfig",
    "SDPOConfig",
    "create_config",
    "get_config_class",
    "ALGORITHM_CONFIGS",
    
    # Legacy config (backward compatible)
    "GRPOConfig",
    
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
