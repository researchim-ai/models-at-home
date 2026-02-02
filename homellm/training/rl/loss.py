"""
Loss функции для GRPO/RL.

Реализованы варианты:
- GRPO (стандартный)
- Dr.GRPO (без std нормализации, фиксированный делитель)
- DAPO (token-level loss, clip higher)
- 🦁 LigerFusedLinearGRPOLoss — НЕ материализует logits (до 80% экономии памяти!)
"""
import logging
from typing import Optional, Tuple, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F

from .experience import Experience
from .legacy_config import GRPOConfig, RLAlgorithm

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


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
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Вычисляет advantages (преимущества) для группы rollout'ов.
    
    Формула GRPO:
        A_i = (r_i - mean(r)) / std(r)
    
    Формула Dr.GRPO (без деления на std):
        A_i = r_i - mean(r)
    
    🔥 Важно (из verl): для группы из 1 элемента:
        mean = 0, std = 1 (чтобы избежать деления на 0)
    
    Args:
        returns: Rewards для группы [group_size] или [batch, group_size]
        use_std_normalization: Делить ли на std (True для GRPO, False для DrGRPO)
        eps: Эпсилон для стабильности (verl использует 1e-6)
        
    Returns:
        Advantages той же размерности
    """
    # 🔥 Обработка группы из 1 элемента (из verl)
    # В этом случае mean = 0, std = 1 чтобы advantage = reward
    if returns.numel() == 1:
        return returns.clone()
    
    # Проверяем размер группы по последнему измерению
    group_size = returns.size(-1) if returns.dim() > 0 else 1
    if group_size == 1:
        # Для группы из 1: advantage = reward (mean=0, std=1)
        return returns.clone()
    
    mean_return = returns.mean(dim=-1, keepdim=True)
    advantages = returns - mean_return
    
    if use_std_normalization:
        std_return = returns.std(dim=-1, keepdim=True)
        # 🔥 Если std очень маленький (все rewards одинаковы), не делим
        # Это предотвращает очень большие advantages
        std_return = torch.where(
            std_return < eps,
            torch.ones_like(std_return),  # Используем 1 вместо деления на eps
            std_return
        )
        advantages = advantages / std_return
    
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
        # 🔥 Clamp для численной стабильности (из verl)
        log_ratio = log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = log_ratio.exp()
        
        # PPO-style clipped surrogate loss
        # Для DAPO используем асимметричный клиппинг [1-eps_low, 1+eps_high]
        ratio_clipped = ratio.clamp(
            1.0 - self.clip_eps_low, 
            1.0 + self.clip_eps_high
        )
        
        # Surrogate objectives
        # verl использует max(-r*A, -clip(r)*A), что эквивалентно -min(r*A, clip(r)*A)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * ratio_clipped
        clip_pg_losses = torch.maximum(pg_losses1, pg_losses2)
        
        # 🔥 Dual-clip PPO для negative advantages (из verl)
        # Нижний порог c=3.0 для стабильности при A < 0
        clip_ratio_c = 3.0
        pg_losses_lower = -advantages * clip_ratio_c
        dual_clip_losses = torch.minimum(pg_losses_lower, clip_pg_losses)
        
        # Применяем dual-clip только для negative advantages
        policy_loss = torch.where(advantages < 0, dual_clip_losses, clip_pg_losses)
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
            # 🔥 Dual-clip метрики (из verl)
            "dual_clip_fraction": ((advantages < 0) & (clip_pg_losses > pg_losses_lower)).float().mean().item(),
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


# ============================================================
# 🦁 LIGER FUSED GRPO LOSS
# ============================================================

class LigerFusedGRPOLoss(nn.Module):
    """
    🔥 Fused GRPO Loss с Liger Kernel — НЕ материализует logits!
    
    Это самая мощная оптимизация памяти для GRPO:
    - Вместо model() -> logits [B, T, V] -> log_probs -> loss
    - Делает: model(output_hidden_states=True) -> hidden [B, T, H] -> fused_loss
    
    Для vocab=150k и seq_len=512 экономит ~3GB на batch!
    
    Использование:
        loss_fn = LigerFusedGRPOLoss(model, config)
        
        # В training loop:
        outputs = model(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # последний layer
        loss, metrics = loss_fn(
            hidden_states=hidden_states,
            selected_token_ids=completion_ids,  # target tokens
            attention_mask=action_mask,
            advantages=advantages,
            old_per_token_logps=old_log_probs,
            ref_per_token_logps=ref_log_probs,
        )
    """
    
    def __init__(
        self,
        model: "PreTrainedModel",
        config: Optional[GRPOConfig] = None,
        beta: float = 0.0,
        loss_type: str = "grpo",
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        max_completion_length: Optional[int] = None,
        chunk_size: int = 1,
    ):
        super().__init__()
        
        # Импортируем Liger
        from .liger_utils import get_liger_fused_linear_grpo, is_liger_available
        
        if not is_liger_available():
            raise RuntimeError("LigerFusedGRPOLoss требует установленный liger-kernel!")
        
        LigerFusedLinearGRPOLoss = get_liger_fused_linear_grpo()
        if LigerFusedLinearGRPOLoss is None:
            raise RuntimeError("LigerFusedLinearGRPOLoss недоступен!")
        
        # Параметры из config или аргументов
        if config is not None:
            beta = config.kl_weight
            loss_type = getattr(config, 'liger_grpo_loss_type', 'dapo')
            epsilon_low = config.clip_eps_low
            epsilon_high = config.clip_eps_high
            max_completion_length = config.max_new_tokens
            chunk_size = getattr(config, 'liger_chunk_size', 1)
        
        self.liger_loss = LigerFusedLinearGRPOLoss(
            beta=beta,
            loss_type=loss_type,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            max_completion_length=max_completion_length,
            chunk_size=chunk_size,
            use_ref_model=beta > 0,
            compiled=False,  # torch.compile падает с динамическими размерами в GRPO
            importance_sampling_level="token",
            temperature=1.0,
        )
        
        # Сохраняем ссылку на lm_head
        self.lm_head_weight = model.lm_head.weight
        self.lm_head_bias = getattr(model.lm_head, 'bias', None)
        
        self.beta = beta
        self.loss_type = loss_type
        self.last_components: dict = {}
        
        logger.info(f"🦁 LigerFusedGRPOLoss инициализирован:")
        logger.info(f"   - loss_type: {loss_type}")
        logger.info(f"   - beta (KL weight): {beta}")
        logger.info(f"   - epsilon: [{epsilon_low}, {epsilon_high}]")
        logger.info(f"   - chunk_size: {chunk_size}")
        logger.info(f"   ⚡ Logits НЕ материализуются — экономия памяти!")
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden] — последний hidden state
        selected_token_ids: torch.Tensor,  # [batch, seq] — target token IDs
        attention_mask: torch.Tensor,  # [batch, seq] — action mask
        advantages: torch.Tensor,  # [batch] — advantages
        old_per_token_logps: Optional[torch.Tensor] = None,  # [batch, seq]
        ref_per_token_logps: Optional[torch.Tensor] = None,  # [batch, seq]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Вычисляет Fused GRPO Loss.
        
        Args:
            hidden_states: Последний hidden state модели [batch, seq, hidden]
            selected_token_ids: Target token IDs (completion tokens) [batch, seq]
            attention_mask: Маска completion токенов [batch, seq]
            advantages: Advantages для каждого sample [batch]
            old_per_token_logps: Log probs из rollout policy [batch, seq]
            ref_per_token_logps: Log probs из reference model [batch, seq]
        
        Returns:
            (loss, metrics_dict)
        """
        # Liger ожидает _input в формате [batch, seq, hidden] — передаём напрямую БЕЗ reshape
        # chunk_forward делает: logits = torch.matmul(input_chunk, weight.t())  # [B, T, H] @ [H, V]
        
        # Вызываем Liger Fused Loss
        result = self.liger_loss(
            hidden_states,         # [batch, seq, hidden] — НЕ делаем flatten!
            self.lm_head_weight,   # [vocab, hidden]
            selected_token_ids,    # [batch, seq]
            attention_mask,        # [batch, seq]
            advantages,            # [batch]
            bias=self.lm_head_bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
        )
        
        # Парсим результат
        if isinstance(result, tuple):
            loss = result[0]
            metrics_list = result[1] if len(result) > 1 else []
        else:
            loss = result
            metrics_list = []
        
        # Собираем метрики
        metrics = {
            "loss": loss.item() if hasattr(loss, 'item') else float(loss),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
        }
        
        # Метрики из Liger
        if len(metrics_list) >= 1 and self.beta > 0:
            kl_val = metrics_list[0]
            metrics["kl_mean"] = kl_val.item() if hasattr(kl_val, 'item') else float(kl_val)
        
        if len(metrics_list) >= 2:
            clip_idx = 1 if self.beta > 0 else 0
            if clip_idx < len(metrics_list):
                clip_val = metrics_list[clip_idx]
                metrics["clip_fraction"] = clip_val.item() if hasattr(clip_val, 'item') else float(clip_val)
        
        self.last_components = metrics
        return loss, metrics
    
    def forward_with_experience(
        self,
        hidden_states: torch.Tensor,
        experience: Experience,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Удобная обёртка для использования с Experience объектом.
        
        Args:
            hidden_states: Последний hidden state [batch, seq, hidden]
            experience: Experience объект с action_mask, advantages и т.д.
        
        Returns:
            (loss, metrics_dict)
        """
        # CAUSAL SHIFT для next-token prediction:
        # - hidden_states[i] предсказывает tokens[i+1]
        # - action_log_probs уже имеет размер [batch, T] (shifted при rollout)
        # - action_mask уже имеет размер [batch, T] (shifted при rollout)
        #
        # ВАЖНО: sequences и action_log_probs батчатся независимо,
        # поэтому их max_len может не совпадать на 1!
        # Нужно выровнять все тензоры до общего размера.
        
        # Получаем размеры
        seq_len = hidden_states.size(1)  # от модели
        target_len = experience.action_log_probs.size(1)  # shifted logprobs
        
        # Сдвигаем hidden states (убираем последний — он предсказывает несуществующий токен)
        # Результат: [batch, seq_len-1, hidden]
        shifted_hidden = hidden_states[:, :-1, :].contiguous()
        
        # Сдвигаем sequences (получаем labels)
        # Результат: [batch, seq_len-1]
        shifted_labels = experience.sequences[:, 1:].contiguous()
        
        # Теперь shifted_hidden и shifted_labels имеют размер seq_len-1
        # А action_log_probs и action_mask имеют размер target_len
        # Выравниваем до меньшего из двух
        final_len = min(shifted_hidden.size(1), target_len)
        
        # Обрезаем все тензоры до final_len (справа, т.к. left-padding)
        shifted_hidden = shifted_hidden[:, -final_len:, :].contiguous()
        shifted_labels = shifted_labels[:, -final_len:].contiguous()
        action_log_probs = experience.action_log_probs[:, -final_len:].contiguous()
        action_mask = experience.action_mask[:, -final_len:]
        ref_log_probs = experience.log_probs_ref
        if ref_log_probs is not None:
            ref_log_probs = ref_log_probs[:, -final_len:].contiguous()
        
        # Advantages должен быть [batch], не [batch, 1]
        advantages = experience.advantages
        if advantages.dim() > 1:
            advantages = advantages.squeeze(-1)
        
        # Action mask: Liger ожидает float/int, не bool
        if action_mask.dtype == torch.bool:
            action_mask = action_mask.to(shifted_hidden.dtype)
        
        return self.forward(
            hidden_states=shifted_hidden,             # [batch, final_len, hidden]
            selected_token_ids=shifted_labels,        # [batch, final_len]
            attention_mask=action_mask,               # [batch, final_len] — float
            advantages=advantages,                    # [batch]
            old_per_token_logps=action_log_probs,     # [batch, final_len]
            ref_per_token_logps=ref_log_probs,        # [batch, final_len] или None
        )


# ============================================================
# 🎓 SDPO LOSS (Self-Distilled Policy Optimization)
# ============================================================

class SDPOLoss(nn.Module):
    """
    🎓 SDPO Loss — Self-Distilled Policy Optimization.
    
    SDPO комбинирует GRPO loss с self-distillation:
    - GRPO loss на обычных rollouts (как в стандартном GRPO)
    - Self-distillation loss: KL между student и teacher (на reprompted контексте)
    
    Идея: успешные траектории становятся "учителем" для текущей политики,
    что даёт более плотный learning signal чем только скалярный reward.
    
    total_loss = grpo_loss + sdpo_loss_weight * distillation_loss
    
    Attributes:
        alpha: Параметр для KL (0=forward, 1=reverse, 0.5=JSD)
        success_threshold: Порог reward для "успешной" траектории
        is_clip: Importance sampling clip для стабильности
        loss_weight: Вес distillation loss относительно GRPO loss
    """
    
    def __init__(
        self,
        config: Optional[GRPOConfig] = None,
        # GRPO параметры
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        kl_weight: float = 0.0,
        token_level_loss: bool = False,
        fixed_length_normalizer: Optional[int] = None,
        # SDPO параметры
        alpha: float = 0.5,
        success_threshold: float = 0.5,
        full_logit_distillation: bool = False,
        distillation_topk: Optional[int] = None,
        is_clip: float = 2.0,
        loss_weight: float = 1.0,
    ) -> None:
        """
        Args:
            config: GRPOConfig с SDPO параметрами
            alpha: 0=forward KL, 1=reverse KL, 0.5=JSD (рекомендуется)
            success_threshold: Порог reward для учителя
            full_logit_distillation: Использовать полное распределение
            distillation_topk: Top-k логитов для distillation
            is_clip: IS clip для стабильности
            loss_weight: Вес SDPO loss
        """
        super().__init__()
        
        # GRPO параметры
        if config is not None:
            self.clip_eps_low = config.clip_eps_low
            self.clip_eps_high = config.clip_eps_high
            self.kl_weight = config.kl_weight
            self.token_level_loss = getattr(config, 'token_level_loss', False)
            self.fixed_length_normalizer = getattr(config, 'fixed_length_normalizer', None)
            # SDPO параметры
            self.alpha = getattr(config, 'sdpo_alpha', alpha)
            self.success_threshold = getattr(config, 'sdpo_success_threshold', success_threshold)
            self.full_logit_distillation = getattr(config, 'sdpo_full_logit_distillation', full_logit_distillation)
            self.distillation_topk = getattr(config, 'sdpo_distillation_topk', distillation_topk)
            self.is_clip = getattr(config, 'sdpo_is_clip', is_clip)
            self.loss_weight = getattr(config, 'sdpo_loss_weight', loss_weight)
        else:
            self.clip_eps_low = clip_eps_low
            self.clip_eps_high = clip_eps_high
            self.kl_weight = kl_weight
            self.token_level_loss = token_level_loss
            self.fixed_length_normalizer = fixed_length_normalizer
            self.alpha = alpha
            self.success_threshold = success_threshold
            self.full_logit_distillation = full_logit_distillation
            self.distillation_topk = distillation_topk
            self.is_clip = is_clip
            self.loss_weight = loss_weight
        
        # Базовый GRPO loss
        self.grpo_loss = GRPOLoss(
            clip_eps_low=self.clip_eps_low,
            clip_eps_high=self.clip_eps_high,
            kl_weight=self.kl_weight,
            token_level_loss=self.token_level_loss,
            fixed_length_normalizer=self.fixed_length_normalizer,
        )
        
        self.last_components: dict = {}
        
        logger.info(f"🎓 SDPOLoss инициализирован:")
        logger.info(f"   - alpha (KL type): {self.alpha} ({'JSD' if self.alpha == 0.5 else 'forward' if self.alpha == 0 else 'reverse'})")
        logger.info(f"   - success_threshold: {self.success_threshold}")
        logger.info(f"   - loss_weight: {self.loss_weight}")
        logger.info(f"   - full_logit_distillation: {self.full_logit_distillation}")
        logger.info(f"   - is_clip: {self.is_clip}")
    
    def compute_distillation_loss(
        self,
        student_log_probs: torch.Tensor,  # [batch, seq] или [batch, seq, vocab/topk]
        teacher_log_probs: torch.Tensor,  # [batch, seq] или [batch, seq, vocab/topk]
        action_mask: torch.Tensor,  # [batch, seq]
        old_log_probs: Optional[torch.Tensor] = None,  # [batch, seq]
        distillation_mask: Optional[torch.Tensor] = None,  # [batch] — какие сэмплы имеют teacher
        student_topk_log_probs: Optional[torch.Tensor] = None,  # [batch, seq, topk] — Top-K логиты
        teacher_topk_log_probs: Optional[torch.Tensor] = None,  # [batch, seq, topk] — Top-K логиты
    ) -> Tuple[torch.Tensor, dict]:
        """
        Вычисляет self-distillation loss между student и teacher.
        
        🔥 ОПТИМИЗАЦИЯ: Поддерживает Top-K Distillation из verl!
        Вместо KL по всему vocab (152k) используем только top-k токенов.
        Экономия памяти: 99.97% при k=50 vs vocab=152k
        
        Args:
            student_log_probs: Log-вероятности текущей политики [batch, seq]
            teacher_log_probs: Log-вероятности teacher [batch, seq]
            action_mask: Маска токенов [batch, seq]
            old_log_probs: Log-вероятности из rollout (для IS)
            distillation_mask: Маска сэмплов с teacher [batch]
            student_topk_log_probs: Top-K log_probs student [batch, seq, k] (для full_logit_distillation)
            teacher_topk_log_probs: Top-K log_probs teacher [batch, seq, k] (для full_logit_distillation)
            
        Returns:
            (loss, metrics)
        """
        metrics = {}
        
        # Комбинируем маски
        loss_mask = action_mask.float()
        if distillation_mask is not None:
            # distillation_mask [batch] -> [batch, 1] для broadcasting
            loss_mask = loss_mask * distillation_mask.unsqueeze(1).float()
        
        # Если нет сэмплов для distillation — возвращаем 0
        if loss_mask.sum() < 1:
            metrics["sdpo_empty_batch"] = True
            return torch.tensor(0.0, device=student_log_probs.device), metrics
        
        # ============================================================
        # 🔥 FULL-LOGIT / TOP-K DISTILLATION (из verl)
        # ============================================================
        if self.full_logit_distillation and student_topk_log_probs is not None and teacher_topk_log_probs is not None:
            # Top-K distillation: используем только top-k токенов
            # Это экономит ~99.97% памяти при k=50 vs vocab=152k
            
            # Ренормализация top-k log probs (чтобы сумма prob = 1)
            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                """Ренормализует top-k log_probs чтобы сумма prob = 1."""
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ
            
            student_distill = renorm_topk_log_probs(student_topk_log_probs)  # [batch, seq, k]
            teacher_distill = renorm_topk_log_probs(teacher_topk_log_probs)  # [batch, seq, k]
            
            # KL divergence по top-k распределению
            if self.alpha == 0.0:
                # Forward KL: KL(teacher || student)
                kl_loss = F.kl_div(
                    student_distill, teacher_distill, reduction="none", log_target=True
                )
            elif self.alpha == 1.0:
                # Reverse KL: KL(student || teacher)
                kl_loss = F.kl_div(
                    teacher_distill, student_distill, reduction="none", log_target=True
                )
            else:
                # Jensen-Shannon Divergence
                alpha_t = torch.tensor(
                    self.alpha, dtype=student_distill.dtype, device=student_distill.device
                )
                mixture_log_probs = torch.logsumexp(
                    torch.stack([
                        student_distill + torch.log(1 - alpha_t),
                        teacher_distill + torch.log(alpha_t)
                    ]),
                    dim=0,
                )
                kl_teacher = F.kl_div(mixture_log_probs, teacher_distill, reduction="none", log_target=True)
                kl_student = F.kl_div(mixture_log_probs, student_distill, reduction="none", log_target=True)
                kl_loss = torch.lerp(kl_student, kl_teacher, alpha_t)
            
            # Суммируем по k-dimension -> [batch, seq]
            per_token_loss = kl_loss.sum(dim=-1)
            metrics["sdpo_topk_distill"] = True
            metrics["sdpo_topk_k"] = student_topk_log_probs.shape[-1]
        else:
            # ============================================================
            # SIMPLE PER-TOKEN DISTILLATION (fallback)
            # ============================================================
            # Вычисляем KL divergence по per-token log_probs
            if self.alpha == 0.0:
                # Forward KL: student -> teacher (mode-seeking)
                per_token_loss = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs))
            elif self.alpha == 1.0:
                # Reverse KL: teacher -> student (mode-covering)
                log_ratio = student_log_probs - teacher_log_probs
                per_token_loss = log_ratio.detach() * student_log_probs
            else:
                # Jensen-Shannon Divergence (alpha = 0.5)
                alpha_t = torch.tensor(self.alpha, dtype=student_log_probs.dtype, device=student_log_probs.device)
                
                mixture_log_probs = torch.logsumexp(
                    torch.stack([
                        student_log_probs + torch.log(1 - alpha_t),
                        teacher_log_probs + torch.log(alpha_t)
                    ]),
                    dim=0,
                )
                
                kl_student = student_log_probs - mixture_log_probs
                kl_teacher = teacher_log_probs - mixture_log_probs
                
                per_token_loss = (1 - alpha_t) * (student_log_probs.exp() * kl_student) + \
                                 alpha_t * (teacher_log_probs.exp() * kl_teacher)
            
            metrics["sdpo_topk_distill"] = False
        
        # Importance Sampling clipping для стабильности
        if self.is_clip is not None and old_log_probs is not None:
            negative_approx_kl = (student_log_probs - old_log_probs).detach()
            negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
            ratio = torch.exp(negative_approx_kl).clamp(max=self.is_clip)
            per_token_loss = per_token_loss * ratio
        
        # Агрегация loss
        if self.token_level_loss:
            total_tokens = loss_mask.sum().clamp(min=1)
            loss = (per_token_loss * loss_mask).sum() / total_tokens
        else:
            if self.fixed_length_normalizer is not None:
                loss = masked_sum(
                    per_token_loss,
                    loss_mask,
                    dim=-1,
                    constant_normalizer=self.fixed_length_normalizer
                ).mean()
            else:
                loss = masked_mean(per_token_loss, loss_mask, dim=-1).mean()
        
        metrics["sdpo_distill_loss"] = loss.item()
        metrics["sdpo_empty_batch"] = False
        
        return loss, metrics
    
    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
        teacher_log_probs: Optional[torch.Tensor] = None,
        distillation_mask: Optional[torch.Tensor] = None,
        student_topk_log_probs: Optional[torch.Tensor] = None,  # 🔥 Top-K optimization
        teacher_topk_log_probs: Optional[torch.Tensor] = None,  # 🔥 Top-K optimization
    ) -> Tuple[torch.Tensor, dict]:
        """
        Вычисляет комбинированный SDPO loss.
        
        🔥 ОПТИМИЗАЦИЯ: Поддерживает Top-K Distillation!
        Если переданы student_topk_log_probs и teacher_topk_log_probs,
        используется KL по top-k токенам вместо всего vocab.
        
        Args:
            log_probs: Log-вероятности текущей политики [batch, seq]
            experience: Experience с rollout данными
            teacher_log_probs: Log-вероятности teacher [batch, seq] (опционально)
            distillation_mask: Маска сэмплов с teacher [batch]
            student_topk_log_probs: Top-K log_probs student [batch, seq, k] (опционально)
            teacher_topk_log_probs: Top-K log_probs teacher [batch, seq, k] (опционально)
            
        Returns:
            (total_loss, metrics)
        """
        # 1. Стандартный GRPO loss
        grpo_loss, grpo_metrics = self.grpo_loss(log_probs, experience)
        
        # 2. Self-distillation loss (если есть teacher данные)
        if teacher_log_probs is not None:
            distill_loss, distill_metrics = self.compute_distillation_loss(
                student_log_probs=log_probs,
                teacher_log_probs=teacher_log_probs,
                action_mask=experience.action_mask,
                old_log_probs=experience.action_log_probs,
                distillation_mask=distillation_mask,
                student_topk_log_probs=student_topk_log_probs,
                teacher_topk_log_probs=teacher_topk_log_probs,
            )
            
            # Комбинируем losses
            total_loss = grpo_loss + self.loss_weight * distill_loss
            
            # Собираем метрики
            metrics = {**grpo_metrics, **distill_metrics}
            metrics["grpo_loss"] = grpo_loss.item()
            metrics["sdpo_weight"] = self.loss_weight
            metrics["total_loss"] = total_loss.item()
        else:
            # Нет teacher данных — только GRPO loss
            total_loss = grpo_loss
            metrics = grpo_metrics
            metrics["sdpo_distill_loss"] = 0.0
            metrics["sdpo_empty_batch"] = True
        
        self.last_components = metrics
        return total_loss, metrics


def create_loss_function(
    model: "PreTrainedModel",
    config: GRPOConfig,
) -> nn.Module:
    """
    Создаёт loss функцию в зависимости от конфигурации.
    
    Выбор loss функции:
    - SDPO: SDPOLoss (GRPO + self-distillation)
    - Liger GRPO: LigerFusedGRPOLoss (оптимизация памяти)
    - Default: GRPOLoss
    
    Args:
        model: HuggingFace модель
        config: GRPOConfig
    
    Returns:
        Loss module (SDPOLoss, LigerFusedGRPOLoss или GRPOLoss)
    """
    # SDPO: используем SDPOLoss
    is_sdpo = (
        getattr(config, 'algorithm', None) == RLAlgorithm.SDPO or
        getattr(config, 'use_self_distillation', False)
    )
    
    if is_sdpo:
        logger.info("🎓 Используется SDPOLoss (GRPO + Self-Distillation)")
        return SDPOLoss(config=config)
    
    # Liger Fused GRPO: оптимизация памяти
    use_liger_fused = getattr(config, 'liger_fused_grpo', False) and getattr(config, 'use_liger', False)
    
    if use_liger_fused:
        try:
            from .liger_utils import is_liger_available
            
            if is_liger_available():
                loss_fn = LigerFusedGRPOLoss(model=model, config=config)
                logger.info("🦁 Используется LigerFusedGRPOLoss (память оптимизирована!)")
                return loss_fn
            else:
                logger.warning("⚠️ Liger недоступен, fallback на GRPOLoss")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось создать LigerFusedGRPOLoss: {e}")
            logger.warning("   Используем стандартный GRPOLoss")
    
    logger.info("📊 Используется стандартный GRPOLoss")
    return GRPOLoss(config=config)
