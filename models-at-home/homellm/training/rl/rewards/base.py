"""
Базовые классы для reward функций.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Callable
import torch
import re


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


class UniversalRuleReward(RewardFunction):
    """
    Универсальная reward функция на основе правил из Reward Designer.
    
    Поддерживает:
    - Regex экстракторы для извлечения значений
    - Условия с разными операторами (contains, matches, equals, etc.)
    - Логику AND/OR для комбинирования условий
    - Python формулы для вычисления reward
    """
    
    def __init__(
        self,
        rules: List[Dict[str, Any]],
        normalize: bool = False,
    ):
        """
        Args:
            rules: Список правил из Reward Designer
            normalize: Нормализовать ли на сумму весов
        """
        super().__init__(weight=1.0)
        self.rules = rules
        self.normalize = normalize
        self._total_weight = sum(r.get("weight", 1.0) for r in rules if r.get("enabled", True))
    
    def _substitute_vars(
        self,
        text: str,
        response: str,
        reference: str,
        prompt: str,
        extracted: Dict[str, str],
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Подставляет переменные в строку."""
        if not isinstance(text, str):
            return text
        text = text.replace("{{response}}", response)
        text = text.replace("{{reference}}", reference)
        text = text.replace("{{prompt}}", prompt)
        for k, v in extracted.items():
            text = text.replace(f"{{{{extracted.{k}}}}}", str(v) if v else "")
        # Подставляем metadata поля
        if metadata:
            for k, v in metadata.items():
                text = text.replace(f"{{{{metadata.{k}}}}}", str(v) if v is not None else "")
        return text
    
    def _evaluate_condition(
        self,
        cond: Dict[str, Any],
        response: str,
        reference: str,
        prompt: str,
        extracted: Dict[str, str],
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Вычисляет одно условие."""
        cond_type = cond.get("type", "contains")
        
        target = self._substitute_vars(
            cond.get("target") or cond.get("left", "{{response}}"),
            response, reference, prompt, extracted, metadata
        )
        
        try:
            if cond_type == "contains":
                value = self._substitute_vars(cond.get("value", ""), response, reference, prompt, extracted, metadata)
                return value in target
            
            elif cond_type == "not_contains":
                value = self._substitute_vars(cond.get("value", ""), response, reference, prompt, extracted, metadata)
                return value not in target
            
            elif cond_type == "matches":
                pattern = cond.get("pattern", "")
                return bool(re.search(pattern, target))
            
            elif cond_type == "not_matches":
                pattern = cond.get("pattern", "")
                return not bool(re.search(pattern, target))
            
            elif cond_type == "equals":
                value = self._substitute_vars(cond.get("value", ""), response, reference, prompt, extracted, metadata)
                return target.strip() == value.strip()
            
            elif cond_type == "equals_numeric":
                right = self._substitute_vars(cond.get("right", ""), response, reference, prompt, extracted, metadata)
                tolerance = float(cond.get("tolerance", 0.01))
                try:
                    left_num = float(re.sub(r"[^\d.\-]", "", str(target)))
                    right_num = float(re.sub(r"[^\d.\-]", "", str(right)))
                    return abs(left_num - right_num) <= tolerance
                except:
                    return False
            
            elif cond_type == "greater":
                value = float(cond.get("value", 0))
                try:
                    num = float(re.sub(r"[^\d.\-]", "", str(target)))
                    return num > value
                except:
                    return False
            
            elif cond_type == "less":
                value = float(cond.get("value", 0))
                try:
                    num = float(re.sub(r"[^\d.\-]", "", str(target)))
                    return num < value
                except:
                    return False
            
            elif cond_type == "length_between":
                length = len(target)
                min_len = int(cond.get("min", 0))
                max_len = int(cond.get("max", 99999))
                return min_len <= length <= max_len
            
            elif cond_type == "length_min":
                return len(target) >= int(cond.get("min", 0))
            
            elif cond_type == "length_max":
                return len(target) <= int(cond.get("max", 99999))
            
        except Exception:
            return False
        
        return False
    
    def _evaluate_formula(
        self,
        formula: str,
        response: str,
        reference: str,
        prompt: str,
        extracted: Dict[str, str],
        metadata: Dict[str, Any] = None,
    ) -> float:
        """Вычисляет формулу reward."""
        # Подстановка переменных
        formula = formula.replace("{{response}}", f"'''{response}'''")
        formula = formula.replace("{{reference}}", f"'''{reference}'''")
        formula = formula.replace("{{prompt}}", f"'''{prompt}'''")
        for k, v in extracted.items():
            safe_v = str(v).replace("'", "\\'") if v else ""
            formula = formula.replace(f"{{{{extracted.{k}}}}}", f"'''{safe_v}'''")
        # Подстановка metadata полей
        if metadata:
            for k, v in metadata.items():
                safe_v = str(v).replace("'", "\\'") if v is not None else ""
                formula = formula.replace(f"{{{{metadata.{k}}}}}", f"'''{safe_v}'''")
        
        try:
            safe_globals = {
                "__builtins__": {
                    "len": len, "min": min, "max": max, "abs": abs,
                    "float": float, "int": int, "str": str, "bool": bool,
                }
            }
            return float(eval(formula, safe_globals))
        except Exception:
            return 0.0
    
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        prompt: str = "",
        **kwargs,
    ) -> float:
        """
        Вычисляет reward на основе правил.
        
        Args:
            completion: Ответ модели
            reference_answer: Эталонный ответ
            prompt: Промпт
            **kwargs: Дополнительные данные, включая metadata из датасета
        """
        total_reward = 0.0
        
        # Извлекаем metadata из kwargs (передаётся из RLSample)
        metadata = kwargs.get("metadata", {})
        
        for rule in self.rules:
            if not rule.get("enabled", True):
                continue
            
            weight = rule.get("weight", 1.0)
            extracted = {}
            
            # 1. Экстракторы
            for ext in rule.get("extractors", []):
                source = ext.get("source", "{{response}}")
                source_text = self._substitute_vars(source, completion, reference_answer, prompt, {}, metadata)
                
                pattern = ext.get("pattern", "")
                flags = 0
                if "DOTALL" in ext.get("flags", ""):
                    flags |= re.DOTALL
                if "IGNORECASE" in ext.get("flags", ""):
                    flags |= re.IGNORECASE
                
                try:
                    match = re.search(pattern, source_text, flags)
                    if match:
                        if match.groups():
                            extracted[ext["name"]] = match.group(1)
                        else:
                            extracted[ext["name"]] = match.group(0)
                    else:
                        extracted[ext["name"]] = ""
                except re.error:
                    extracted[ext["name"]] = ""
            
            # 2. Условия
            conditions = rule.get("conditions", [])
            logic = rule.get("condition_logic", "all")
            
            if not conditions:
                all_conditions_true = True
            else:
                results = [
                    self._evaluate_condition(c, completion, reference_answer, prompt, extracted, metadata)
                    for c in conditions
                ]
                if logic == "all":
                    all_conditions_true = all(results)
                else:
                    all_conditions_true = any(results)
            
            # 3. Формула reward
            if all_conditions_true:
                formula = rule.get("reward_formula", "1.0")
            else:
                formula = rule.get("else_reward", "0.0")
            
            rule_reward = self._evaluate_formula(formula, completion, reference_answer, prompt, extracted, metadata)
            total_reward += weight * rule_reward
        
        if self.normalize and self._total_weight > 0:
            return total_reward / self._total_weight
        return total_reward
    
    @classmethod
    def from_config(cls, reward_rules: List[Dict[str, Any]]) -> "UniversalRuleReward":
        """
        Создаёт UniversalRuleReward из конфигурации Reward Designer.
        """
        return cls(rules=reward_rules, normalize=False)
