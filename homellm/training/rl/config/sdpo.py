"""
SDPO (Self-Distilled Policy Optimization) конфигурация.

SDPO — алгоритм с self-distillation из verl/VolcEngine:
- Self-distillation: успешные траектории используются как "учитель"
- Reprompting: модель получает успешное решение как контекст
- Environment feedback: ошибки компиляции/runtime используются для обучения

Особенно эффективен для математики и кода (верифицируемые домены).

Paper: "Self-Distilled Policy Optimization for Verifiable Domains"
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .base import RLAlgorithm
from .drgrpo import DrGRPOConfig


@dataclass
class SDPOConfig(DrGRPOConfig):
    """
    Конфигурация для SDPO алгоритма.
    
    SDPO расширяет Dr.GRPO с self-distillation:
    - Успешные траектории (reward >= threshold) становятся "учителем"
    - Модель получает reprompted контекст с успешным решением
    - KL divergence между student и teacher распределениями
    
    Ключевая идея: вместо только скалярного reward, модель учится
    имитировать успешные траектории через distillation.
    
    Attributes:
        success_threshold: Порог reward для "успешной" траектории
        alpha: Параметр для KL divergence (0=forward, 1=reverse, 0.5=JSD)
        full_logit_distillation: Использовать полное распределение или только selected tokens
        distillation_topk: Использовать только top-k логитов (экономия памяти)
        is_clip: Importance Sampling clip для стабильности
        include_feedback: Включать environment feedback в reprompt
        max_reprompt_len: Максимальная длина reprompted контекста
        ema_rate: EMA rate для teacher модели (0 = без EMA)
        loss_weight: Вес SDPO loss относительно GRPO loss
    """
    
    # ============================================================
    # SDPO CORE PARAMETERS
    # ============================================================
    
    # Порог успешности траектории
    # Траектории с reward >= threshold используются как демонстрации
    success_threshold: float = 0.5
    
    # Alpha для KL divergence:
    # - 0.0: Forward KL (student → teacher) — mode-seeking
    # - 1.0: Reverse KL (teacher → student) — mode-covering  
    # - 0.5: Jensen-Shannon Divergence (баланс, рекомендуется)
    alpha: float = 0.5
    
    # ============================================================
    # DISTILLATION PARAMETERS
    # ============================================================
    
    # Full-logit vs selected-token distillation
    # True: KL по всему распределению [vocab_size] — точнее, но дороже
    # False: KL только по selected tokens — экономит память
    full_logit_distillation: bool = False
    
    # Top-k логитов для distillation (если full_logit_distillation=True)
    # None = использовать все логиты
    # 100-500 = только top-k (экономия памяти)
    distillation_topk: Optional[int] = None
    
    # Importance Sampling clip для стабильности
    # Ограничивает влияние сильно отличающихся траекторий
    is_clip: float = 2.0
    
    # ============================================================
    # REPROMPTING PARAMETERS
    # ============================================================
    
    # Включать environment feedback в reprompt
    # True: ошибки компиляции, runtime errors добавляются в контекст
    include_feedback: bool = True
    
    # Максимальная длина reprompted контекста
    # Если превышает — обрезаем с конца
    max_reprompt_len: int = 4096
    
    # ============================================================
    # TEACHER MODEL PARAMETERS
    # ============================================================
    
    # EMA rate для teacher модели
    # 0.0: без EMA, используем текущую модель как teacher
    # 0.01-0.1: медленно обновляемый teacher (более стабильно)
    ema_rate: float = 0.0
    
    # ============================================================
    # LOSS COMBINATION
    # ============================================================
    
    # Вес SDPO loss относительно base GRPO loss
    # total_loss = grpo_loss + loss_weight * sdpo_loss
    loss_weight: float = 1.0
    
    # ============================================================
    # TEMPLATES
    # ============================================================
    
    # Шаблон для reprompting
    reprompt_template: str = """Here is the problem:
{question}

Here is a successful solution for reference:
{successful_solution}

Now solve this problem step by step."""
    
    # Шаблон для feedback
    feedback_template: str = """
Previous attempt feedback:
{feedback}
"""
    
    # ============================================================
    # INHERITED OVERRIDES
    # ============================================================
    
    # SDPO работает хорошо с dynamic sampling
    dynamic_sampling: bool = True
    
    # Не используем Liger Fused Loss для SDPO (нужен доступ к logits для distillation)
    liger_fused_grpo: bool = False
    
    @property
    def algorithm(self) -> RLAlgorithm:
        return RLAlgorithm.SDPO
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфиг в словарь."""
        result = super().to_dict()
        result.update({
            "success_threshold": self.success_threshold,
            "alpha": self.alpha,
            "full_logit_distillation": self.full_logit_distillation,
            "distillation_topk": self.distillation_topk,
            "is_clip": self.is_clip,
            "include_feedback": self.include_feedback,
            "max_reprompt_len": self.max_reprompt_len,
            "ema_rate": self.ema_rate,
            "loss_weight": self.loss_weight,
        })
        return result
