"""
Модульная система конфигураций для RL алгоритмов.

Каждый алгоритм имеет свой конфиг класс:
- GRPOConfig: стандартный GRPO
- DrGRPOConfig: Dr.GRPO с фиксами
- DAPOConfig: DAPO (GRPO++) с dynamic sampling
- SDPOConfig: SDPO с self-distillation

Пример использования:
    from homellm.training.rl.config import SDPOConfig, GRPOConfig
    
    # Создаём SDPO конфиг с кастомными параметрами
    config = SDPOConfig(
        batch_size=8,
        learning_rate=5e-6,
        success_threshold=0.7,
        alpha=0.5,
    )
    
    # Или используем preset
    config = create_config("sdpo")
    config = create_config("reasoning_small")
"""

from .base import BaseRLConfig, RLAlgorithm
from .grpo import GRPOConfig
from .drgrpo import DrGRPOConfig
from .dapo import DAPOConfig
from .sdpo import SDPOConfig

from typing import Union, Dict, Any

# Type alias для любого RL конфига
RLConfig = Union[GRPOConfig, DrGRPOConfig, DAPOConfig, SDPOConfig]

# Маппинг алгоритмов на классы конфигов
ALGORITHM_CONFIGS: Dict[RLAlgorithm, type] = {
    RLAlgorithm.GRPO: GRPOConfig,
    RLAlgorithm.DRGRPO: DrGRPOConfig,
    RLAlgorithm.DAPO: DAPOConfig,
    RLAlgorithm.SDPO: SDPOConfig,
}


def create_config(preset: str, **overrides) -> RLConfig:
    """
    Создаёт конфигурацию из предустановки.
    
    Args:
        preset: Название предустановки:
            - "grpo": Стандартный GRPO
            - "drgrpo": Dr.GRPO с фиксами
            - "dapo": DAPO с dynamic sampling
            - "sdpo": SDPO с self-distillation
            - "reasoning_small": Для маленьких моделей (0.5-3B)
            - "reasoning_large": Для больших моделей (7B+)
        **overrides: Параметры для переопределения
        
    Returns:
        Соответствующий конфиг
        
    Example:
        config = create_config("sdpo", learning_rate=1e-5)
        config = create_config("reasoning_small", use_4bit=True)
    """
    # Определяем preset defaults
    preset_defaults: Dict[str, Dict[str, Any]] = {
        # Базовые алгоритмы - без дополнительных defaults
        "grpo": {},
        "drgrpo": {},
        "dapo": {},
        "sdpo": {},
        
        # Presets для разных размеров моделей
        "reasoning_small": {
            "group_size": 8,
            "batch_size": 4,
            "train_batch_size": 2,
            "max_new_tokens": 512,
            "learning_rate": 5e-6,
            "use_lora": True,
            "use_4bit": True,
        },
        "reasoning_large": {
            "group_size": 16,
            "batch_size": 8,
            "train_batch_size": 4,
            "max_new_tokens": 2048,
            "learning_rate": 1e-6,
            "use_lora": True,
            "use_4bit": True,
        },
        
        # SDPO presets
        "sdpo_math": {
            "success_threshold": 0.5,
            "alpha": 0.5,
            "include_feedback": True,
            "max_new_tokens": 1024,
        },
        "sdpo_code": {
            "success_threshold": 0.8,
            "alpha": 0.5,
            "include_feedback": True,
            "max_new_tokens": 2048,
        },
    }
    
    # Маппинг preset -> config class
    preset_classes: Dict[str, type] = {
        "grpo": GRPOConfig,
        "drgrpo": DrGRPOConfig,
        "dapo": DAPOConfig,
        "sdpo": SDPOConfig,
        "reasoning_small": DrGRPOConfig,
        "reasoning_large": DAPOConfig,
        "sdpo_math": SDPOConfig,
        "sdpo_code": SDPOConfig,
    }
    
    if preset not in preset_classes:
        available = list(preset_classes.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")
    
    # Комбинируем: preset defaults + user overrides (overrides имеют приоритет)
    config_class = preset_classes[preset]
    defaults = preset_defaults.get(preset, {})
    final_kwargs = {**defaults, **overrides}
    
    return config_class(**final_kwargs)


def get_config_class(algorithm: Union[str, RLAlgorithm]) -> type:
    """
    Возвращает класс конфига для указанного алгоритма.
    
    Args:
        algorithm: Название алгоритма или RLAlgorithm enum
        
    Returns:
        Класс конфига (GRPOConfig, DAPOConfig, etc.)
    """
    if isinstance(algorithm, str):
        algorithm = RLAlgorithm(algorithm.lower())
    
    if algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return ALGORITHM_CONFIGS[algorithm]


# Экспорт для обратной совместимости
# Старый код может использовать GRPOConfig как универсальный конфиг
__all__ = [
    # Enum
    "RLAlgorithm",
    
    # Config classes
    "BaseRLConfig",
    "GRPOConfig",
    "DrGRPOConfig", 
    "DAPOConfig",
    "SDPOConfig",
    
    # Type alias
    "RLConfig",
    
    # Factory functions
    "create_config",
    "get_config_class",
    "ALGORITHM_CONFIGS",
]
