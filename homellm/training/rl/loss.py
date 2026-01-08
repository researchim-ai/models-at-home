"""
Loss функции для GRPO/RL.

Реализованы варианты:
- GRPO (стандартный)
- Dr.GRPO (без std нормализации, фиксированный делитель)
- DAPO (token-level loss, clip higher)
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .experience import Experience
from .config import GRPOConfig, RLAlgorithm


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Monte-Carlo аппроксимация KL дивергенции (k3 estimator).
    
    См.: http://joschu.net/blog/kl-approx.html
    
    KL ≈ exp(log_ref - log) - (log_ref - log) - 1
    
    Args:
        log_probs: Log-вероятности по текущей политике [batch, seq_len]
        log_probs_ref: Log-вероятности по референсной политике [batch, seq_len]
        action_mask: Маска токенов для учёта [batch, seq_len]
        
    Returns:
        KL дивергенция [batch, seq_len]
    """
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    
    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Среднее с учётом маски.
    
    Args:
        tensor: Входной тензор
        mask: Маска (1 = учитывать, 0 = игнорировать)
        dim: Размерность для редукции
        
    Returns:
        Среднее значение
    """
    if mask is None:
        return tensor.mean(dim=dim)
    
    masked = tensor * mask
    return masked.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)


def masked_sum(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: Optional[int] = None,
    constant_normalizer: Optional[float] = None,
) -> torch.Tensor:
    """
    Сумма с маской и опциональной фиксированной нормализацией.
    
    Используется в Dr.GRPO для устранения length bias:
    вместо деления на фактическую длину делим на фиксированную константу.
    
    Args:
        tensor: Входной тензор
        mask: Маска
        dim: Размерность для редукции
        constant_normalizer: Фиксированный делитель (для Dr.GRPO)
        
    Returns:
        Сумма / делитель
    """
    if mask is None:
        summed = tensor.sum(dim=dim)
    else:
        summed = (tensor * mask).sum(dim=dim)
    
    if constant_normalizer is not None and constant_normalizer > 0:
        return summed / constant_normalizer
    return summed


def compute_advantages(
    returns: torch.Tensor,
    use_std_normalization: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Вычисляет advantages (преимущества) для группы rollout'ов.
    
    Формула GRPO:
        A_i = (r_i - mean(r)) / std(r)
    
    Формула Dr.GRPO (без деления на std):
        A_i = r_i - mean(r)
    
    Args:
        returns: Rewards для группы [group_size] или [batch, group_size]
        use_std_normalization: Делить ли на std (True для GRPO, False для DrGRPO)
        eps: Эпсилон для стабильности
        
    Returns:
        Advantages той же размерности
    """
    mean_return = returns.mean(dim=-1, keepdim=True)
    advantages = returns - mean_return
    
    if use_std_normalization:
        std_return = returns.std(dim=-1, keepdim=True)
        advantages = advantages / (std_return + eps)
    
    return advantages


class GRPOLoss(nn.Module):
    """
    GRPO Loss с поддержкой разных вариантов алгоритма.
    
    Реализует:
    - PPO-style surrogate loss с клиппингом
    - KL регуляризацию (опционально)
    - Token-level vs sample-level агрегацию
    - Асимметричный клиппинг (clip higher для DAPO)
    
    Формула:
        L = -min(ratio * A, clip(ratio) * A) + β * KL
    
    где ratio = exp(log_π - log_π_old)
    """
    
    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        kl_weight: float = 0.0,
        token_level_loss: bool = False,
        fixed_length_normalizer: Optional[int] = None,
    ) -> None:
        """
        Args:
            config: GRPOConfig (если передан, остальные параметры игнорируются)
            clip_eps_low: Нижняя граница клиппинга
            clip_eps_high: Верхняя граница клиппинга (для DAPO: 0.28)
            kl_weight: Вес KL штрафа (0 для reasoning-RL)
            token_level_loss: Token-level агрегация (vs sample-level)
            fixed_length_normalizer: Фиксированный делитель для Dr.GRPO
        """
        super().__init__()
        
        if config is not None:
            self.clip_eps_low = config.clip_eps_low
            self.clip_eps_high = config.clip_eps_high
            self.kl_weight = config.kl_weight
            self.token_level_loss = config.token_level_loss
            self.fixed_length_normalizer = config.fixed_length_normalizer
            self.algorithm = config.algorithm
        else:
            self.clip_eps_low = clip_eps_low
            self.clip_eps_high = clip_eps_high
            self.kl_weight = kl_weight
            self.token_level_loss = token_level_loss
            self.fixed_length_normalizer = fixed_length_normalizer
            self.algorithm = RLAlgorithm.GRPO
        
        # Для логирования компонентов loss
        self.last_components: dict = {}
    
    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Вычисляет GRPO loss.
        
        Args:
            log_probs: Log-вероятности по текущей политике [batch, seq_len]
            experience: Experience с action_log_probs, advantages, masks
            
        Returns:
            Tuple[loss, metrics_dict]
        """
        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages
        
        # KL дивергенция
        if log_probs_ref is not None and self.kl_weight > 0:
            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                action_mask=action_mask,
            )
        else:
            kl = torch.zeros_like(log_probs)
        
        # Policy ratio: r(θ) = π(a|s) / π_old(a|s) = exp(log_π - log_π_old)
        ratio = (log_probs - old_log_probs).exp()
        
        # PPO-style clipped surrogate loss
        # Для DAPO используем асимметричный клиппинг [1-eps_low, 1+eps_high]
        ratio_clipped = ratio.clamp(
            1.0 - self.clip_eps_low, 
            1.0 + self.clip_eps_high
        )
        
        # Surrogate objectives
        surr1 = ratio * advantages
        surr2 = ratio_clipped * advantages
        
        # Loss = -min(surr1, surr2) + kl_weight * kl
        policy_loss = -torch.min(surr1, surr2)
        loss_per_token = policy_loss + self.kl_weight * kl
        
        # Агрегация loss
        if self.token_level_loss:
            # Token-level: каждый токен имеет равный вес (DAPO)
            if action_mask is not None:
                total_tokens = action_mask.sum()
                loss = (loss_per_token * action_mask).sum() / total_tokens.clamp(min=1)
            else:
                loss = loss_per_token.mean()
        else:
            # Sample-level с опциональной фиксированной нормализацией (Dr.GRPO)
            if self.fixed_length_normalizer is not None:
                # Dr.GRPO: делим на фиксированную константу
                loss = masked_sum(
                    loss_per_token, 
                    action_mask, 
                    dim=-1,
                    constant_normalizer=self.fixed_length_normalizer
                ).mean()
            else:
                # Стандартный GRPO: среднее по токенам в каждом сэмпле
                loss = masked_mean(loss_per_token, action_mask, dim=-1).mean()
        
        # Сохраняем компоненты для логирования
        metrics = {
            "loss": loss.item(),
            "kl_mean": masked_mean(kl, action_mask).mean().item() if self.kl_weight > 0 else 0,
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "ratio_max": ratio.max().item(),
            "ratio_min": ratio.min().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "clip_fraction": ((ratio < 1 - self.clip_eps_low) | (ratio > 1 + self.clip_eps_high)).float().mean().item(),
        }
        self.last_components = metrics
        
        return loss, metrics


class PolicyGradientLoss(nn.Module):
    """
    Простой Policy Gradient loss (REINFORCE).
    
    Может использоваться как baseline для сравнения с GRPO.
    
    L = -log_π(a|s) * A
    """
    
    def __init__(
        self,
        baseline_subtract: bool = True,
        normalize_advantages: bool = True,
    ) -> None:
        super().__init__()
        self.baseline_subtract = baseline_subtract
        self.normalize_advantages = normalize_advantages
    
    def forward(
        self,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            log_probs: Log-вероятности действий [batch, seq_len]
            returns: Rewards [batch, 1] или [batch]
            action_mask: Маска токенов [batch, seq_len]
            
        Returns:
            loss, metrics
        """
        if returns.dim() == 1:
            returns = returns.unsqueeze(-1)
        
        # Baseline: среднее по батчу
        if self.baseline_subtract:
            advantages = returns - returns.mean()
        else:
            advantages = returns
        
        # Нормализация
        if self.normalize_advantages and advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)
        
        # Policy gradient: L = -log_π * A
        # Расширяем advantages для умножения с log_probs
        if advantages.size(-1) == 1 and log_probs.dim() > 1:
            advantages = advantages.expand_as(log_probs)
        
        loss_per_token = -log_probs * advantages
        
        if action_mask is not None:
            loss = masked_mean(loss_per_token, action_mask, dim=-1).mean()
        else:
            loss = loss_per_token.mean()
        
        metrics = {
            "loss": loss.item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
        }
        
        return loss, metrics


def compute_entropy(log_probs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Вычисляет энтропию политики.
    
    H = -Σ p * log(p) = -Σ exp(log_p) * log_p
    
    Для мониторинга: энтропия должна медленно расти при здоровом обучении.
    Резкое падение = entropy collapse.
    
    Args:
        log_probs: Log-вероятности [batch, seq_len]
        action_mask: Маска токенов
        
    Returns:
        Средняя энтропия
    """
    probs = log_probs.exp()
    entropy = -probs * log_probs
    
    if action_mask is not None:
        return masked_mean(entropy, action_mask)
    return entropy.mean()


def compute_overlong_penalty(
    completion_lengths: torch.Tensor,
    max_length: int,
    buffer_length: int = 0,
    penalty_value: float = -1.0,
) -> torch.Tensor:
    """
    Мягкий штраф за слишком длинные ответы (из DAPO).
    
    - Длина < (max_length - buffer): штраф = 0
    - Длина > max_length: штраф = penalty_value
    - Промежуточная длина: линейная интерполяция
    
    Args:
        completion_lengths: Длины completions [batch]
        max_length: Максимальная допустимая длина
        buffer_length: Буферная зона
        penalty_value: Максимальный штраф (обычно -1)
        
    Returns:
        Штрафы [batch]
    """
    threshold_start = max_length - buffer_length
    
    penalties = torch.zeros_like(completion_lengths, dtype=torch.float)
    
    # Промежуточная зона: линейная интерполяция
    in_buffer = (completion_lengths >= threshold_start) & (completion_lengths <= max_length)
    if in_buffer.any():
        progress = (completion_lengths[in_buffer] - threshold_start).float() / max(buffer_length, 1)
        penalties[in_buffer] = progress * penalty_value
    
    # Превышение максимума
    over_max = completion_lengths > max_length
    penalties[over_max] = penalty_value
    
    return penalties
