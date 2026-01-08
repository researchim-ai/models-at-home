"""
Базовые классы для reward функций.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
import torch


class RewardFunction(ABC):
    """
    Базовый класс для reward функций.
    
    Наследники должны реализовать метод __call__ для вычисления reward.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: Вес этой reward функции при комбинировании
        """
        self.weight = weight
    
    @abstractmethod
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        reasoning_format: str = "deepseek",
        is_truncated: bool = False,
        **kwargs,
    ) -> float:
        """
        Вычисляет reward для одного completion.
        
        Args:
            completion: Сгенерированный текст
            reference_answer: Эталонный ответ
            reasoning_format: Формат reasoning тегов
            is_truncated: Был ли ответ обрезан
            **kwargs: Дополнительные параметры
            
        Returns:
            Значение reward (обычно 0-1)
        """
        pass
    
    def batch_call(
        self,
        completions: List[str],
        reference_answers: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """
        Вычисляет rewards для батча.
        
        Args:
            completions: Список completions
            reference_answers: Список эталонных ответов
            
        Returns:
            Tensor rewards [batch_size]
        """
        rewards = []
        for completion, ref in zip(completions, reference_answers):
            reward = self(completion, ref, **kwargs)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)


class CombinedReward(RewardFunction):
    """
    Комбинация нескольких reward функций.
    
    Итоговый reward = sum(weight_i * reward_i) / sum(weights)
    или взвешенная сумма без нормализации.
    """
    
    def __init__(
        self,
        reward_functions: List[RewardFunction],
        normalize: bool = True,
    ):
        """
        Args:
            reward_functions: Список reward функций
            normalize: Нормализовать ли на сумму весов
        """
        super().__init__(weight=1.0)
        self.reward_functions = reward_functions
        self.normalize = normalize
        
        self._total_weight = sum(rf.weight for rf in reward_functions)
    
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        **kwargs,
    ) -> float:
        """Вычисляет комбинированный reward."""
        total_reward = 0.0
        
        for rf in self.reward_functions:
            reward = rf(completion, reference_answer, **kwargs)
            total_reward += rf.weight * reward
        
        if self.normalize and self._total_weight > 0:
            return total_reward / self._total_weight
        return total_reward
    
    def get_component_rewards(
        self,
        completion: str,
        reference_answer: str,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Возвращает rewards по компонентам (для логирования).
        """
        component_rewards = {}
        for rf in self.reward_functions:
            name = rf.__class__.__name__
            reward = rf(completion, reference_answer, **kwargs)
            component_rewards[name] = reward
        return component_rewards


class ConstantReward(RewardFunction):
    """
    Константный reward (для тестирования).
    """
    
    def __init__(self, value: float = 0.0, weight: float = 1.0):
        super().__init__(weight)
        self.value = value
    
    def __call__(self, completion: str, reference_answer: str, **kwargs) -> float:
        return self.value


class BinaryReward(RewardFunction):
    """
    Бинарный reward: 1 если условие выполнено, 0 иначе.
    """
    
    def __init__(
        self, 
        condition_fn,
        weight: float = 1.0,
        true_value: float = 1.0,
        false_value: float = 0.0,
    ):
        """
        Args:
            condition_fn: Функция (completion, reference) -> bool
            true_value: Reward если условие True
            false_value: Reward если условие False
        """
        super().__init__(weight)
        self.condition_fn = condition_fn
        self.true_value = true_value
        self.false_value = false_value
    
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        **kwargs,
    ) -> float:
        if self.condition_fn(completion, reference_answer):
            return self.true_value
        return self.false_value
