"""
Reward функции для проверки формата reasoning.
"""
import re
from typing import Optional
from .base import RewardFunction


class FormatReward(RewardFunction):
    """
    Проверяет соблюдение формата reasoning.
    
    Ожидаемый формат:
    - DeepSeek: <think>...</think><answer>...</answer>
    - Simple: <reasoning>...</reasoning><answer>...</answer>
    
    Reward:
    - Правильный формат (оба тега): +format_reward
    - Частичный формат (один тег): +partial_reward
    - Нет формата: 0
    """
    
    def __init__(
        self,
        format_reward: float = 0.2,
        partial_reward: float = 0.1,
        weight: float = 1.0,
    ):
        """
        Args:
            format_reward: Reward за полный формат
            partial_reward: Reward за частичный формат
            weight: Вес в комбинированном reward
        """
        super().__init__(weight)
        self.format_reward = format_reward
        self.partial_reward = partial_reward
    
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        reasoning_format: str = "deepseek",
        **kwargs,
    ) -> float:
        """Проверяет формат completion."""
        if reasoning_format == "deepseek":
            reasoning_open = "<think>"
            reasoning_close = "</think>"
        else:
            reasoning_open = "<reasoning>"
            reasoning_close = "</reasoning>"
        
        answer_open = "<answer>"
        answer_close = "</answer>"
        
        # Считаем теги
        has_reasoning_open = reasoning_open in completion
        has_reasoning_close = reasoning_close in completion
        has_answer_open = answer_open in completion
        has_answer_close = answer_close in completion
        
        # Проверяем количество (должен быть ровно один)
        reasoning_count = min(
            completion.count(reasoning_open),
            completion.count(reasoning_close)
        )
        answer_count = min(
            completion.count(answer_open),
            completion.count(answer_close)
        )
        
        # Правильный формат: ровно по одному тегу каждого типа
        if reasoning_count == 1 and answer_count == 1:
            # Проверяем порядок: reasoning должен быть перед answer
            reasoning_end = completion.find(reasoning_close)
            answer_start = completion.find(answer_open)
            if reasoning_end < answer_start:
                return self.format_reward
        
        # Частичный формат
        if (has_reasoning_open and has_reasoning_close) or (has_answer_open and has_answer_close):
            return self.partial_reward
        
        return 0.0


class ReasoningQualityReward(RewardFunction):
    """
    Оценивает качество reasoning (chain-of-thought).
    
    Критерии:
    - Минимальная длина reasoning
    - Наличие шагов решения (нумерованные, маркеры)
    - Наличие ключевых слов (следовательно, потому что, значит)
    """
    
    def __init__(
        self,
        min_words: int = 10,
        step_bonus: float = 0.1,
        keyword_bonus: float = 0.05,
        max_reward: float = 0.3,
        weight: float = 1.0,
    ):
        """
        Args:
            min_words: Минимальное количество слов для бонуса
            step_bonus: Бонус за наличие шагов
            keyword_bonus: Бонус за ключевые слова
            max_reward: Максимальный суммарный reward
            weight: Вес
        """
        super().__init__(weight)
        self.min_words = min_words
        self.step_bonus = step_bonus
        self.keyword_bonus = keyword_bonus
        self.max_reward = max_reward
        
        # Паттерны для детекции шагов
        self.step_patterns = [
            r"\d+[\.\)]\s",  # 1. или 1)
            r"[-•]\s",  # маркированный список
            r"(?:шаг|step)\s*\d+",  # "шаг 1" или "step 1"
            r"(?:во-первых|во-вторых|в-третьих|firstly|secondly)",
        ]
        
        # Ключевые слова рассуждения
        self.keywords_ru = [
            "следовательно", "потому что", "значит", "таким образом",
            "поэтому", "отсюда", "получаем", "вычисляем", "находим",
            "подставим", "решим", "равно",
        ]
        self.keywords_en = [
            "therefore", "because", "thus", "hence", "so",
            "we get", "we find", "calculate", "solve", "equals",
        ]
    
    def _extract_reasoning(self, completion: str, reasoning_format: str) -> Optional[str]:
        """Извлекает текст reasoning из completion."""
        if reasoning_format == "deepseek":
            pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        else:
            pattern = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
        
        match = pattern.search(completion)
        return match.group(1).strip() if match else None
    
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        reasoning_format: str = "deepseek",
        **kwargs,
    ) -> float:
        """Оценивает качество reasoning."""
        reasoning = self._extract_reasoning(completion, reasoning_format)
        
        if not reasoning:
            return 0.0
        
        reward = 0.0
        words = reasoning.split()
        
        # Бонус за минимальную длину
        if len(words) >= self.min_words:
            reward += 0.1
        
        # Бонус за шаги
        for pattern in self.step_patterns:
            if re.search(pattern, reasoning, re.IGNORECASE):
                reward += self.step_bonus
                break
        
        # Бонус за ключевые слова
        reasoning_lower = reasoning.lower()
        for keyword in self.keywords_ru + self.keywords_en:
            if keyword in reasoning_lower:
                reward += self.keyword_bonus
                break
        
        return min(reward, self.max_reward)


class TruncationPenalty(RewardFunction):
    """
    Штраф за обрезанные (truncated) ответы.
    
    Мягкий штраф из DAPO: если ответ обрезан, но почти уложился -
    меньший штраф, чем за сильное превышение.
    """
    
    def __init__(
        self,
        penalty: float = -0.5,
        hard_penalty: float = -1.0,
        weight: float = 1.0,
    ):
        """
        Args:
            penalty: Штраф за обрезку
            hard_penalty: Жёсткий штраф (для очень длинных)
            weight: Вес
        """
        super().__init__(weight)
        self.penalty = penalty
        self.hard_penalty = hard_penalty
    
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        is_truncated: bool = False,
        **kwargs,
    ) -> float:
        """Возвращает штраф если ответ обрезан."""
        if is_truncated:
            # Проверяем, есть ли хоть какой-то answer тег
            if "<answer>" in completion and "</answer>" not in completion:
                # Начал отвечать, но не закончил - мягкий штраф
                return self.penalty
            elif "</answer>" in completion:
                # Закончил ответ, но потом продолжил - тоже мягкий
                return self.penalty
            else:
                # Вообще не дошёл до ответа - жёсткий штраф
                return self.hard_penalty
        return 0.0
