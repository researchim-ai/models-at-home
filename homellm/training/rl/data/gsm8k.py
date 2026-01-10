"""
GSM8K датасет для обучения reasoning.

GSM8K (Grade School Math 8K) - датасет математических задач
уровня начальной школы с пошаговыми решениями.

Формат:
- question: Текст задачи
- answer: Пошаговое решение с ответом в формате "#### NUMBER"
"""
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .base import RLDataset, RLSample

logger = logging.getLogger(__name__)


def extract_gsm8k_final_answer(answer_text: str) -> str:
    """
    Извлекает финальный ответ из GSM8K формата.
    
    GSM8K формат: "Пошаговое решение\n#### 42"
    
    Args:
        answer_text: Полный текст ответа
        
    Returns:
        Числовой ответ как строка
    """
    # Ищем формат #### NUMBER
    match = re.search(r"####\s*(-?\d+(?:,\d+)?(?:\.\d+)?)", answer_text)
    if match:
        # Убираем запятые (разделители тысяч)
        return match.group(1).replace(",", "")
    
    # Fallback: последнее число в тексте
    numbers = re.findall(r"-?\d+(?:,\d+)?(?:\.\d+)?", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return answer_text.strip()


class GSM8KDataset(RLDataset):
    """
    Датасет GSM8K для обучения математическому reasoning.
    
    Поддерживает:
    - Загрузку из HuggingFace datasets (GSM8K, GSM8K-RU, MATH-RU, MGSM)
    - Загрузку из локального JSONL файла
    - Фильтрацию по сложности
    - Форматирование с reasoning тегами
    """
    
    # Поддерживаемые датасеты
    SUPPORTED_DATASETS = {
        "gsm8k": {
            "hf_name": "gsm8k",
            "config": "main",
            "question_field": "question",
            "answer_field": "answer",
        },
        "gsm8k_ru": {
            "hf_name": "d0rj/gsm8k-ru",
            "config": None,
            "question_field": "question",
            "answer_field": "answer",
        },
        "math_ru": {
            "hf_name": "d0rj/competition_math_ru",
            "config": None,
            "question_field": "problem",
            "answer_field": "solution",
        },
        "mgsm": {
            "hf_name": "juletxara/mgsm",
            "config": "ru",  # Русская часть
            "question_field": "question",
            "answer_field": "answer_number",  # MGSM имеет числовой ответ
        },
    }
    
    # Системные промпты для разных форматов
    SYSTEM_PROMPTS = {
        "deepseek": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>""",
        
        "simple": """Реши математическую задачу. Сначала приведи рассуждения, затем дай ответ.

Формат ответа:
<reasoning>
(Пошаговое решение)
</reasoning>
<answer>
(Числовой ответ)
</answer>""",
        
        "russian": """Ты — помощник по математике. Решай задачи пошагово.

Отвечай в формате:
<reasoning>
Шаг 1: ...
Шаг 2: ...
...
</reasoning>
<answer>
(только число)
</answer>""",
    }
    
    def __init__(
        self,
        file_path: Optional[str] = None,
        use_huggingface: bool = True,
        split: str = "train",
        max_samples: Optional[int] = None,
        reasoning_format: str = "deepseek",
        language: str = "en",
        dataset_key: Optional[str] = None,
        min_answer: Optional[int] = None,
        max_answer: Optional[int] = None,
    ):
        """
        Args:
            file_path: Путь к локальному JSONL файлу (опционально)
            use_huggingface: Загружать из HuggingFace
            split: "train" или "test"
            max_samples: Максимальное количество примеров
            reasoning_format: "deepseek", "simple" или "russian"
            language: Язык датасета ("en" или "ru")
            dataset_key: Ключ датасета из SUPPORTED_DATASETS (gsm8k, gsm8k_ru, math_ru, mgsm)
            min_answer: Минимальное значение ответа (для фильтрации)
            max_answer: Максимальное значение ответа (для фильтрации)
        """
        super().__init__(split=split, max_samples=max_samples, reasoning_format=reasoning_format)
        
        self.file_path = file_path
        self.use_huggingface = use_huggingface
        self.language = language
        self.dataset_key = dataset_key
        self.min_answer = min_answer
        self.max_answer = max_answer
        
        self.load_data()
    
    def load_data(self) -> None:
        """Загружает данные GSM8K."""
        if self.file_path:
            self._load_from_file()
        elif self.use_huggingface:
            self._load_from_huggingface()
        else:
            raise ValueError("Укажите file_path или use_huggingface=True")
    
    def _load_from_huggingface(self) -> None:
        """Загружает из HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Установите datasets: pip install datasets")
        
        # Определяем какой датасет загружать
        if self.dataset_key and self.dataset_key in self.SUPPORTED_DATASETS:
            ds_info = self.SUPPORTED_DATASETS[self.dataset_key]
        elif self.language == "ru":
            ds_info = self.SUPPORTED_DATASETS["gsm8k_ru"]
        else:
            ds_info = self.SUPPORTED_DATASETS["gsm8k"]
        
        dataset_name = ds_info["hf_name"]
        config = ds_info.get("config")
        question_field = ds_info.get("question_field", "question")
        answer_field = ds_info.get("answer_field", "answer")
        
        logger.info(f"Загрузка датасета из HuggingFace: {dataset_name} (split={self.split})...")
        
        try:
            if config:
                dataset = load_dataset(dataset_name, config, split=self.split)
            else:
                dataset = load_dataset(dataset_name, split=self.split)
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета {dataset_name}: {e}")
            # Fallback на английский GSM8K
            logger.info("Fallback на английский GSM8K...")
            dataset = load_dataset("gsm8k", "main", split=self.split)
            question_field = "question"
            answer_field = "answer"
        
        for item in dataset:
            # Адаптируем поля для разных датасетов
            adapted_item = {
                "question": item.get(question_field, ""),
                "answer": str(item.get(answer_field, "")),
            }
            
            sample = self._process_item(adapted_item)
            if sample:
                self.samples.append(sample)
                
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
        
        logger.info(f"Загружено {len(self.samples)} примеров из {dataset_name}")
    
    def _load_from_file(self) -> None:
        """Загружает из локального файла."""
        file_path = Path(self.file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.file_path}")
        
        logger.info(f"Загрузка GSM8K из {file_path}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                sample = self._process_item(item)
                
                if sample:
                    self.samples.append(sample)
                    
                    if self.max_samples and len(self.samples) >= self.max_samples:
                        break
        
        logger.info(f"Загружено {len(self.samples)} примеров GSM8K")
    
    def _process_item(self, item: Dict[str, Any]) -> Optional[RLSample]:
        """Обрабатывает один пример датасета."""
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        if not question or not answer:
            return None
        
        # Извлекаем финальный ответ
        final_answer = extract_gsm8k_final_answer(answer)
        
        # Фильтрация по значению ответа
        try:
            answer_num = float(final_answer.replace(",", ""))
            if self.min_answer is not None and answer_num < self.min_answer:
                return None
            if self.max_answer is not None and answer_num > self.max_answer:
                return None
        except ValueError:
            pass
        
        return RLSample(
            prompt=question,
            reference_answer=final_answer,
            metadata={
                "full_answer": answer,
                "question": question,
            }
        )
    
    def format_prompt(self, sample: RLSample) -> str:
        """
        Форматирует промпт с системным сообщением.
        
        Использует системный промпт в зависимости от reasoning_format.
        """
        system_prompt = self.SYSTEM_PROMPTS.get(
            self.reasoning_format,
            self.SYSTEM_PROMPTS["deepseek"]
        )
        
        # Можно вернуть просто вопрос или с системным промптом
        # Системный промпт будет применён в build_reasoning_prompt
        return sample.prompt
    
    def get_system_prompt(self) -> str:
        """Возвращает системный промпт для текущего формата."""
        return self.SYSTEM_PROMPTS.get(
            self.reasoning_format,
            self.SYSTEM_PROMPTS["deepseek"]
        )


def load_gsm8k(
    split: str = "train",
    max_samples: Optional[int] = None,
    reasoning_format: str = "deepseek",
    language: str = "en",
    dataset_key: Optional[str] = None,
    **kwargs,
) -> GSM8KDataset:
    """
    Удобная функция для загрузки математических датасетов.
    
    Args:
        split: "train" или "test"
        max_samples: Максимальное количество примеров
        reasoning_format: Формат reasoning тегов
        language: "en" для английского, "ru" для русского
        dataset_key: Ключ датасета:
            - "gsm8k" — английский GSM8K
            - "gsm8k_ru" — d0rj/gsm8k-ru (русский перевод)
            - "math_ru" — d0rj/competition_math_ru (олимпиадные задачи)
            - "mgsm" — juletxara/mgsm (многоязычный)
        
    Returns:
        GSM8KDataset
        
    Example:
        >>> # Английский датасет
        >>> dataset = load_gsm8k(split="train", max_samples=1000)
        
        >>> # Русский датасет (d0rj/gsm8k-ru)
        >>> dataset_ru = load_gsm8k(split="train", max_samples=500, language="ru")
        
        >>> # Конкретный датасет по ключу
        >>> dataset_math = load_gsm8k(dataset_key="math_ru", max_samples=500)
    """
    return GSM8KDataset(
        use_huggingface=True,
        split=split,
        max_samples=max_samples,
        reasoning_format=reasoning_format,
        language=language,
        dataset_key=dataset_key,
        **kwargs,
    )


# Пример создания русского GSM8K датасета
class GSM8KRussianDataset(GSM8KDataset):
    """
    Русская версия GSM8K (если доступна).
    
    Или автоматический перевод через API.
    """
    
    def __init__(
        self,
        file_path: Optional[str] = None,
        **kwargs,
    ):
        kwargs["reasoning_format"] = kwargs.get("reasoning_format", "russian")
        kwargs["language"] = "ru"
        
        super().__init__(file_path=file_path, use_huggingface=False, **kwargs)
    
    def _load_from_file(self) -> None:
        """Загружает русскую версию."""
        if not self.file_path:
            logger.warning("Для русского GSM8K укажите путь к файлу")
            return
        
        super()._load_from_file()
