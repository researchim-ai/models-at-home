"""
SDPO (Self-Distilled Policy Optimization) конфигурация.

Реализация SDPO согласно статье "Reinforcement Learning via Self-Distillation":
- Self-distillation: модель с feedback в контексте становится teacher
- Reprompting: модель получает успешное решение и/или feedback как контекст
- Environment feedback: ошибки компиляции/runtime, результаты тестов
- Top-K Distillation с tail bucket для экономии памяти

Paper: https://arxiv.org/abs/2601.20802
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .base import RLAlgorithm
from .drgrpo import DrGRPOConfig


@dataclass
class SDPOConfig(DrGRPOConfig):
    """
    Конфигурация для SDPO алгоритма.
    
    SDPO расширяет GRPO с self-distillation:
    - Модель с feedback/solution в контексте становится "teacher"
    - Teacher переоценивает каждый токен исходного ответа
    - KL divergence между student и teacher распределениями
    
    Ключевая идея: plотный credit assignment на уровне токенов
    вместо скалярного reward на весь ответ.
    
    Attributes:
        # Core SDPO
        success_threshold: Порог reward для "успешной" траектории
        alpha: Параметр для KL divergence (0=forward, 1=reverse, 0.5=JSD)
        
        # Distillation
        full_logit_distillation: Использовать полное распределение
        distillation_topk: Top-k логитов для distillation (None = всё)
        distillation_add_tail: Добавлять tail bucket для top-k
        is_clip: Importance Sampling clip для стабильности
        
        # Reprompting
        include_environment_feedback: Включать feedback от среды
        environment_feedback_only_without_solution: Feedback только если нет solution
        dont_reprompt_on_self_success: Не reprompt успешную траекторию саму на себя
        remove_thinking_from_demonstration: Убирать <think> из демонстраций
        max_reprompt_len: Максимальная длина reprompt контекста
        reprompt_truncation: Метод обрезки ("left", "right", "error")
        
        # Teacher
        ema_rate: EMA rate для teacher модели (0 = без EMA)
        loss_weight: Вес SDPO loss относительно GRPO loss
    """
    
    # ============================================================
    # SDPO CORE PARAMETERS
    # ============================================================
    
    # Порог успешности траектории
    # Траектории с reward >= threshold используются как демонстрации
    success_threshold: float = 1.0
    
    # Alpha для KL divergence:
    # - 0.0: Forward KL (student → teacher) — mode-seeking, рекомендуется
    # - 1.0: Reverse KL (teacher → student) — mode-covering  
    # - 0.5: Jensen-Shannon Divergence
    alpha: float = 0.0
    
    # ============================================================
    # DISTILLATION PARAMETERS
    # ============================================================
    
    # Full-logit vs selected-token distillation
    # True: KL по распределению [vocab_size] или [top-k] — точнее
    # False: KL только по selected tokens — экономит память
    full_logit_distillation: bool = True
    
    # Top-k логитов для distillation
    # None = использовать все логиты (дорого по памяти!)
    # 50-100 = только top-k (рекомендуется для экономии памяти)
    distillation_topk: Optional[int] = 100
    
    # 🔥 Добавлять "tail" bucket для top-k distillation
    # Tail = log(1 - sum(top_k_probs)) — учитывает оставшуюся вероятность
    # True: более точная дистилляция, рекомендуется
    # False: только top-k без tail
    distillation_add_tail: bool = True
    
    # Importance Sampling clip для стабильности
    # Ограничивает влияние сильно отличающихся траекторий
    # None = без IS weighting
    is_clip: Optional[float] = 2.0
    
    # ============================================================
    # REPROMPTING PARAMETERS
    # ============================================================
    
    # 🔥 Включать environment feedback в reprompt
    # True: ошибки компиляции, runtime errors, failed tests добавляются
    include_environment_feedback: bool = True
    
    # Использовать feedback только если нет успешного solution
    # True: feedback OR solution, но не оба вместе
    # False: feedback И solution могут быть вместе
    environment_feedback_only_without_solution: bool = True
    
    # 🔥 Не использовать успешную траекторию как teacher для самой себя
    # True: выбираем другую успешную траекторию из группы
    # False: траектория может быть своим же teacher
    dont_reprompt_on_self_success: bool = True
    
    # 🔥 Убирать <think>...</think> из демонстраций
    # True: убираем chain-of-thought, оставляем только ответ
    # False: демонстрация содержит полный reasoning
    remove_thinking_from_demonstration: bool = True
    
    # Максимальная длина reprompted контекста (в токенах)
    max_reprompt_len: int = 10240
    
    # Метод обрезки при превышении max_reprompt_len
    # "right": обрезаем справа (начало сохраняется)
    # "left": обрезаем слева (конец сохраняется)
    # "error": выбрасываем ошибку
    reprompt_truncation: str = "right"
    
    # ============================================================
    # TEACHER MODEL PARAMETERS
    # ============================================================
    
    # EMA rate для teacher модели
    # 0.0: без EMA, используем текущую модель как teacher
    # 0.01-0.1: медленно обновляемый teacher (более стабильно)
    ema_rate: float = 0.05
    
    # ============================================================
    # LOSS COMBINATION
    # ============================================================
    
    # Вес SDPO loss (для совместимости, не используется при loss_mode="sdpo")
    # При loss_mode="sdpo": только distillation loss (loss_weight игнорируется)
    # При loss_mode="grpo": только GRPO loss (loss_weight игнорируется)
    loss_weight: float = 1.0
    
    # Режим loss: "sdpo" (только distillation) или "grpo" (только GRPO)
    loss_mode: str = "sdpo"
    
    # ============================================================
    # TEMPLATES
    # ============================================================
    
    # Основной шаблон для reprompting
    # Placeholders: {prompt}, {solution}, {feedback}
    reprompt_template: str = """{prompt}{solution}{feedback}

Correctly solve the original question."""
    
    # Шаблон для solution секции
    # Placeholder: {successful_previous_attempt}
    solution_template: str = """

Correct solution:

{successful_previous_attempt}

"""
    
    # Шаблон для feedback секции
    # Placeholder: {feedback_raw}
    feedback_template: str = """

The following is feedback from your unsuccessful earlier attempt:

{feedback_raw}

"""
    
    # ============================================================
    # INHERITED OVERRIDES
    # ============================================================
    
    # SDPO работает хорошо с dynamic sampling
    dynamic_sampling: bool = True
    
    # Не используем Liger Fused Loss для SDPO (нужен доступ к logits)
    liger_fused_grpo: bool = False
    
    @property
    def algorithm(self) -> RLAlgorithm:
        return RLAlgorithm.SDPO
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфиг в словарь."""
        result = super().to_dict()
        result.update({
            # Core
            "success_threshold": self.success_threshold,
            "alpha": self.alpha,
            # Distillation
            "full_logit_distillation": self.full_logit_distillation,
            "distillation_topk": self.distillation_topk,
            "distillation_add_tail": self.distillation_add_tail,
            "is_clip": self.is_clip,
            # Reprompting
            "include_environment_feedback": self.include_environment_feedback,
            "environment_feedback_only_without_solution": self.environment_feedback_only_without_solution,
            "dont_reprompt_on_self_success": self.dont_reprompt_on_self_success,
            "remove_thinking_from_demonstration": self.remove_thinking_from_demonstration,
            "max_reprompt_len": self.max_reprompt_len,
            "reprompt_truncation": self.reprompt_truncation,
            # Teacher
            "ema_rate": self.ema_rate,
            "loss_weight": self.loss_weight,
            "loss_mode": self.loss_mode,
        })
        return result
