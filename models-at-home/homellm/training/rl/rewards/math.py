"""
Reward функции для математических задач.
"""
import re
from typing import Optional, List
from .base import RewardFunction


def extract_number_from_text(text: str) -> Optional[float]:
    """
    Извлекает число из текста.
    
    Поддерживает:
    - Целые числа: 42
    - Дробные: 3.14
    - Отрицательные: -5
    - С запятой (русский формат): 3,14
    - С разделителями тысяч: 1,000 или 1 000
    """
    if not text:
        return None
    
    # Очищаем текст
    text = text.strip()
    
    # Убираем пробелы между цифрами (разделители тысяч)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    
    # Заменяем запятую на точку (русский формат)
    # Но только если это дробный разделитель (за ней идут цифры, но не 3+)
    text = re.sub(r',(\d{1,2})(?!\d)', r'.\1', text)
    
    # Убираем запятые-разделители тысяч
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    
    # Ищем число
    patterns = [
        r'[-+]?\d+\.?\d*',  # 42, -42, 3.14
        r'[-+]?\.\d+',      # .5
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                continue
    
    return None


def extract_answer_number(completion: str, reasoning_format: str = "deepseek") -> Optional[float]:
    """
    Извлекает число из тега <answer>.
    """
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match = pattern.search(completion)
    
    if not match:
        return None
    
    answer_text = match.group(1).strip()
    return extract_number_from_text(answer_text)


def normalize_math_answer(text: str) -> str:
    """
    Нормализует математический ответ для сравнения.
    
    - Убирает пробелы
    - Приводит к нижнему регистру
    - Нормализует дроби (1/2 -> 0.5)
    - Удаляет единицы измерения
    """
    if not text:
        return ""
    
    text = text.strip().lower()
    
    # Убираем единицы измерения
    units = ["руб", "рублей", "рубля", "$", "€", "долларов", "meters", "km", "kg", "г", "кг", "м"]
    for unit in units:
        text = text.replace(unit, "").strip()
    
    # Нормализуем дроби
    fraction_match = re.match(r"(\d+)\s*/\s*(\d+)", text)
    if fraction_match:
        num = int(fraction_match.group(1))
        denom = int(fraction_match.group(2))
        if denom != 0:
            text = str(num / denom)
    
    # Убираем лишние нули после точки
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    
    return text


class MathReward(RewardFunction):
    """
    Базовый reward для математических задач.
    
    Сравнивает числовой ответ с эталоном.
    """
    
    def __init__(
        self,
        tolerance: float = 1e-4,
        partial_tolerance: float = 0.5,
        exact_reward: float = 1.0,
        partial_reward: float = 0.5,
        weight: float = 1.0,
    ):
        """
        Args:
            tolerance: Допуск для точного совпадения
            partial_tolerance: Допуск для частичного совпадения
            exact_reward: Reward за точное совпадение
            partial_reward: Reward за частичное совпадение
            weight: Вес
        """
        super().__init__(weight)
        self.tolerance = tolerance
        self.partial_tolerance = partial_tolerance
        self.exact_reward = exact_reward
        self.partial_reward = partial_reward
    
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        reasoning_format: str = "deepseek",
        **kwargs,
    ) -> float:
        """Сравнивает числовые ответы."""
        # Извлекаем числа
        pred_num = extract_answer_number(completion, reasoning_format)
        ref_num = extract_number_from_text(reference_answer)
        
        if pred_num is None or ref_num is None:
            return 0.0
        
        # Сравниваем
        diff = abs(pred_num - ref_num)
        
        if diff < self.tolerance:
            return self.exact_reward
        elif diff < self.partial_tolerance:
            return self.partial_reward
        else:
            return 0.0


class GSM8KReward(RewardFunction):
    """
    Reward функция специально для GSM8K датасета.
    
    GSM8K особенности:
    - Ответы всегда целые числа
    - Формат: "#### answer" или просто число
    - Иногда есть пояснения типа "The answer is 42"
    """
    
    def __init__(
        self,
        correct_reward: float = 1.0,
        close_reward: float = 0.3,
        format_bonus: float = 0.1,
        weight: float = 1.0,
    ):
        """
        Args:
            correct_reward: Reward за правильный ответ
            close_reward: Reward за близкий ответ (±1)
            format_bonus: Бонус за правильный формат reasoning
            weight: Вес
        """
        super().__init__(weight)
        self.correct_reward = correct_reward
        self.close_reward = close_reward
        self.format_bonus = format_bonus
    
    def _extract_gsm8k_answer(self, text: str) -> Optional[int]:
        """Извлекает ответ в формате GSM8K."""
        # Сначала пробуем формат #### answer
        hash_match = re.search(r"####\s*(-?\d+)", text)
        if hash_match:
            return int(hash_match.group(1))
        
        # "The answer is X" или "Ответ: X"
        answer_patterns = [
            r"(?:the answer is|answer is|ответ[:\s]+)[\s]*(-?\d+)",
            r"(?:итого|всего|получаем)[:\s]*(-?\d+)",
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Из тега <answer>
        answer_tag = re.search(r"<answer>\s*(-?\d+)\s*</answer>", text)
        if answer_tag:
            return int(answer_tag.group(1))
        
        # Последнее число в тексте
        numbers = re.findall(r"(-?\d+)", text)
        if numbers:
            return int(numbers[-1])
        
        return None
    
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        reasoning_format: str = "deepseek",
        **kwargs,
    ) -> float:
        """Вычисляет reward для GSM8K."""
        # Извлекаем ответы
        pred_answer = self._extract_gsm8k_answer(completion)
        
        # Reference может быть в формате "#### 42" или просто "42"
        ref_answer = self._extract_gsm8k_answer(reference_answer)
        if ref_answer is None:
            ref_num = extract_number_from_text(reference_answer)
            ref_answer = int(ref_num) if ref_num is not None else None
        
        if pred_answer is None or ref_answer is None:
            return 0.0
        
        reward = 0.0
        
        # Основной reward за правильный ответ
        if pred_answer == ref_answer:
            reward = self.correct_reward
        elif abs(pred_answer - ref_answer) == 1:
            # Близкий ответ (ошибка на 1)
            reward = self.close_reward
        
        # Бонус за правильный формат
        if reasoning_format == "deepseek":
            if "<think>" in completion and "</think>" in completion:
                reward += self.format_bonus
        else:
            if "<reasoning>" in completion and "</reasoning>" in completion:
                reward += self.format_bonus
        
        return min(reward, 1.0)


class MathExpressionReward(RewardFunction):
    """
    Reward для математических выражений и уравнений.
    
    Может сравнивать:
    - Числа
    - Простые выражения (2+2)
    - Списки чисел (корни уравнения)
    """
    
    def __init__(
        self,
        tolerance: float = 1e-4,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        self.tolerance = tolerance
    
    def _parse_roots(self, text: str) -> List[float]:
        """Парсит список корней уравнения."""
        # x1 = 1, x2 = -1
        pairs = re.findall(r"x\d*\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
        if pairs:
            return sorted([float(p) for p in pairs])
        
        # Просто числа через запятую или пробел
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
        if numbers:
            return sorted([float(n) for n in numbers])
        
        return []
    
    def __call__(
        self,
        completion: str,
        reference_answer: str,
        reasoning_format: str = "deepseek",
        **kwargs,
    ) -> float:
        """Сравнивает математические выражения."""
        # Извлекаем ответ из completion
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        match = pattern.search(completion)
        
        if not match:
            return 0.0
        
        pred_text = match.group(1).strip()
        
        # Пробуем как одно число
        pred_num = extract_number_from_text(pred_text)
        ref_num = extract_number_from_text(reference_answer)
        
        if pred_num is not None and ref_num is not None:
            if abs(pred_num - ref_num) < self.tolerance:
                return 1.0
            return 0.0
        
        # Пробуем как список корней
        pred_roots = self._parse_roots(pred_text)
        ref_roots = self._parse_roots(reference_answer)
        
        if pred_roots and ref_roots:
            if len(pred_roots) != len(ref_roots):
                return 0.0
            
            matches = 0
            for p, r in zip(pred_roots, ref_roots):
                if abs(p - r) < self.tolerance:
                    matches += 1
            
            return matches / len(ref_roots)
        
        # Fallback: строковое сравнение
        pred_norm = normalize_math_answer(pred_text)
        ref_norm = normalize_math_answer(reference_answer)
        
        return 1.0 if pred_norm == ref_norm else 0.0
