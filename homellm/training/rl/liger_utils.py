"""
Liger Kernel интеграция для GRPO тренировки.

Liger Kernel — это набор оптимизированных Triton kernels для LLM тренировки:

🔥 CHUNKED LOSS (экономия до 80% памяти):
- LigerFusedLinearGRPOLoss: GRPO без материализации logits
- LigerFusedLinearCrossEntropyLoss: CE без материализации logits (pretrain/SFT)
- LigerFusedLinearDPOLoss, LigerFusedLinearCPOLoss, etc.

⚡ LOW-LEVEL OPS:
- LigerCrossEntropyLoss: оптимизированный cross-entropy
- LigerRMSNorm: оптимизированный RMSNorm
- LigerKLDIVLoss: оптимизированный KL divergence
- Патчинг HF моделей (Qwen, Llama, Mistral и др.)

Документация: https://github.com/linkedin/Liger-Kernel
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, TYPE_CHECKING, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# ============================================================
# ГЛОБАЛЬНЫЕ КЭШИ
# ============================================================
_LIGER_AVAILABLE: Optional[bool] = None
_LIGER_CE_LOSS: Optional[Callable] = None
_LIGER_FUSED_LINEAR_CE: Optional[type] = None
_LIGER_FUSED_LINEAR_GRPO: Optional[type] = None
_LIGER_KL_DIV: Optional[type] = None

# Поддерживаемые архитектуры для автоматического патчинга
LIGER_SUPPORTED_MODELS = {"qwen2", "qwen", "llama", "mistral", "gemma", "gemma2", "phi", "phi3", "mixtral"}


def is_liger_available() -> bool:
    """Проверяет доступность Liger Kernel."""
    global _LIGER_AVAILABLE
    if _LIGER_AVAILABLE is not None:
        return _LIGER_AVAILABLE
    
    try:
        import liger_kernel
        _LIGER_AVAILABLE = True
        logger.info(f"✅ Liger Kernel доступен: v{getattr(liger_kernel, '__version__', 'unknown')}")
    except ImportError:
        _LIGER_AVAILABLE = False
        logger.warning("⚠️ Liger Kernel не установлен. Установите: pip install liger-kernel")
    
    return _LIGER_AVAILABLE


def get_liger_cross_entropy_loss() -> Optional[Callable]:
    """
    Возвращает LigerCrossEntropyLoss если доступен.
    
    Преимущества:
    - Более эффективное вычисление на GPU
    - Меньше памяти чем F.cross_entropy
    """
    global _LIGER_CE_LOSS
    
    if _LIGER_CE_LOSS is not None:
        return _LIGER_CE_LOSS
    
    if not is_liger_available():
        return None
    
    try:
        from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
        
        def liger_ce_loss(
            logits: torch.Tensor,
            targets: torch.Tensor,
            ignore_index: int = -100,
            reduction: str = "none",
        ) -> torch.Tensor:
            """
            Wrapper для LigerCrossEntropyFunction.
            
            Args:
                logits: [batch, vocab] или [batch*seq, vocab]
                targets: [batch] или [batch*seq]
                ignore_index: индекс для игнорирования
                reduction: "none", "mean", "sum"
            """
            # Liger CE ожидает 2D input
            if logits.dim() == 3:
                # [batch, seq, vocab] -> [batch*seq, vocab]
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)
            
            # LigerCrossEntropyFunction.apply signature:
            # (_input, target, weight, ignore_index, lse_square_scale, 
            #  label_smoothing, reduction, softcap, return_z_loss, return_token_accuracy)
            loss, z_loss, token_accuracy = LigerCrossEntropyFunction.apply(
                logits.contiguous(),  # _input: [BT, V]
                targets.contiguous(), # target: [BT]
                None,                 # weight: Optional[Tensor] — НЕ ignore_index!
                ignore_index,         # ignore_index: int
                0.0,                  # lse_square_scale: float
                0.0,                  # label_smoothing: float
                reduction,            # reduction: str ("none", "mean", "sum")
                None,                 # softcap: Optional[float]
                False,                # return_z_loss: bool
                False,                # return_token_accuracy: bool
            )
            
            return loss
        
        _LIGER_CE_LOSS = liger_ce_loss
        logger.info("✅ LigerCrossEntropyLoss загружен")
        return _LIGER_CE_LOSS
        
    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить LigerCrossEntropyLoss: {e}")
        return None


def get_liger_fused_linear_ce() -> Optional[type]:
    """
    Возвращает LigerFusedLinearCrossEntropyLoss если доступен.
    
    ГЛАВНАЯ ОПТИМИЗАЦИЯ для LLM (pretrain/SFT):
    - НЕ материализует полный logits тензор [batch, seq, vocab]
    - Для vocab=150k это экономит гигабайты памяти!
    
    Использование:
        loss_fn = LigerFusedLinearCrossEntropyLoss()
        loss = loss_fn(lm_head.weight, hidden_states, targets, lm_head.bias)
    """
    global _LIGER_FUSED_LINEAR_CE
    
    if _LIGER_FUSED_LINEAR_CE is not None:
        return _LIGER_FUSED_LINEAR_CE
    
    if not is_liger_available():
        return None
    
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        _LIGER_FUSED_LINEAR_CE = LigerFusedLinearCrossEntropyLoss
        logger.info("✅ LigerFusedLinearCrossEntropyLoss загружен")
        return _LIGER_FUSED_LINEAR_CE
    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить LigerFusedLinearCrossEntropyLoss: {e}")
        return None


def get_liger_fused_linear_grpo() -> Optional[type]:
    """
    Возвращает LigerFusedLinearGRPOLoss если доступен.
    
    🔥 ГЛАВНАЯ ОПТИМИЗАЦИЯ для GRPO:
    - Fused lm_head + GRPO loss computation
    - НЕ материализует полный logits тензор [batch, seq, vocab]
    - Встроенная поддержка: grpo, dapo, dr_grpo, bnpo loss types
    - Встроенный KL penalty
    - До 80% экономии памяти!
    
    Использование:
        loss_fn = LigerFusedLinearGRPOLoss(
            beta=0.04,  # KL penalty weight
            loss_type="grpo",  # или "dapo", "dr_grpo", "bnpo"
            epsilon_low=0.2,
            epsilon_high=0.2,
        )
        loss, metrics = loss_fn(
            hidden_states,  # [batch*seq, hidden]
            lm_head.weight,  # [vocab, hidden]
            selected_token_ids,  # [batch, seq]
            attention_mask,  # [batch, seq]
            advantages,  # [batch]
            bias=lm_head.bias,
            ref_per_token_logps=ref_logprobs,  # [batch, seq] (optional)
            old_per_token_logps=old_logprobs,  # [batch, seq] (optional)
        )
    """
    global _LIGER_FUSED_LINEAR_GRPO
    
    if _LIGER_FUSED_LINEAR_GRPO is not None:
        return _LIGER_FUSED_LINEAR_GRPO
    
    if not is_liger_available():
        return None
    
    try:
        from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
        _LIGER_FUSED_LINEAR_GRPO = LigerFusedLinearGRPOLoss
        logger.info("✅ LigerFusedLinearGRPOLoss загружен (fused GRPO без материализации logits)")
        return _LIGER_FUSED_LINEAR_GRPO
    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить LigerFusedLinearGRPOLoss: {e}")
        return None


def get_liger_kl_div() -> Optional[type]:
    """
    Возвращает LigerKLDIVLoss если доступен.
    
    Оптимизированный KL divergence для KL penalty в RL.
    """
    global _LIGER_KL_DIV
    
    if _LIGER_KL_DIV is not None:
        return _LIGER_KL_DIV
    
    if not is_liger_available():
        return None
    
    try:
        from liger_kernel.transformers import LigerKLDIVLoss
        _LIGER_KL_DIV = LigerKLDIVLoss
        logger.info("✅ LigerKLDIVLoss загружен")
        return _LIGER_KL_DIV
    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить LigerKLDIVLoss: {e}")
        return None


def apply_liger_patch_to_model(
    model: "PreTrainedModel",
    patch_rms_norm: bool = True,
    patch_rope: bool = True,
    patch_mlp: bool = True,
    patch_fused_linear_ce: bool = False,  # Для GRPO лучше отключить (мы используем свой loss)
) -> bool:
    """
    Применяет Liger патчи к HuggingFace модели.
    
    ВАЖНО для Liger 0.6.x:
    - cross_entropy и fused_linear_cross_entropy нельзя включать одновременно
    - Для GRPO мы используем свой loss, поэтому патчим только RMSNorm/RoPE/MLP
    
    Поддерживаемые архитектуры:
    - Qwen2
    - Llama / Llama2 / Llama3
    - Mistral
    - Gemma / Gemma2
    - Phi3
    
    Args:
        model: HuggingFace модель
        patch_rms_norm: патчить RMSNorm на LigerRMSNorm
        patch_rope: патчить RoPE embeddings
        patch_mlp: патчить MLP на fused SwiGLU/GeGLU
        patch_fused_linear_ce: патчить CrossEntropy на FusedLinearCrossEntropy
    
    Returns:
        True если патч применён, False если нет
    """
    if not is_liger_available():
        return False
    
    model_type = getattr(model.config, "model_type", "").lower()
    
    # Общие параметры для всех архитектур
    # ВАЖНО: cross_entropy=False чтобы не конфликтовать с fused_linear_cross_entropy
    common_kwargs = {
        "rms_norm": patch_rms_norm,
        "rope": patch_rope,
        "cross_entropy": False,  # Отключаем — используем свой chunked CE или Liger CE отдельно
        "fused_linear_cross_entropy": patch_fused_linear_ce,
    }
    
    try:
        # Qwen2
        if "qwen2" in model_type or "qwen" in model_type:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(
                **common_kwargs,
                swiglu=patch_mlp,
            )
            logger.info(f"✅ Liger патч применён к Qwen2 (rms={patch_rms_norm}, rope={patch_rope}, mlp={patch_mlp})")
            return True
        
        # Llama
        elif "llama" in model_type:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama(
                **common_kwargs,
                swiglu=patch_mlp,
            )
            logger.info(f"✅ Liger патч применён к Llama (rms={patch_rms_norm}, rope={patch_rope}, mlp={patch_mlp})")
            return True
        
        # Mistral
        elif "mistral" in model_type:
            from liger_kernel.transformers import apply_liger_kernel_to_mistral
            apply_liger_kernel_to_mistral(
                **common_kwargs,
                swiglu=patch_mlp,
            )
            logger.info(f"✅ Liger патч применён к Mistral (rms={patch_rms_norm}, rope={patch_rope}, mlp={patch_mlp})")
            return True
        
        # Gemma
        elif "gemma" in model_type:
            from liger_kernel.transformers import apply_liger_kernel_to_gemma
            apply_liger_kernel_to_gemma(
                rms_norm=patch_rms_norm,
                rope=patch_rope,
                cross_entropy=False,
                fused_linear_cross_entropy=patch_fused_linear_ce,
                geglu=patch_mlp,
            )
            logger.info(f"✅ Liger патч применён к Gemma (rms={patch_rms_norm}, rope={patch_rope}, mlp={patch_mlp})")
            return True
        
        # Phi3
        elif "phi" in model_type:
            from liger_kernel.transformers import apply_liger_kernel_to_phi3
            apply_liger_kernel_to_phi3(
                **common_kwargs,
                swiglu=patch_mlp,
            )
            logger.info(f"✅ Liger патч применён к Phi3 (rms={patch_rms_norm}, rope={patch_rope}, mlp={patch_mlp})")
            return True
        
        else:
            # INFO а не WARNING — для HomeModel и других кастомных моделей это нормально,
            # они используют Liger напрямую (LigerRMSNorm, LigerSiLUMul, LigerFusedCE)
            logger.info(f"ℹ️ Liger: автопатч HF-стиля не применён к '{model_type}' (используются прямые оптимизации)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка применения Liger патча: {e}")
        return False


def liger_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Cross-entropy с автоматическим fallback на Liger если доступен.
    
    Использует LigerCrossEntropyLoss если доступен, иначе F.cross_entropy.
    
    Args:
        logits: [batch, seq, vocab] или [batch*seq, vocab]
        targets: [batch, seq] или [batch*seq]
        ignore_index: индекс для игнорирования
        reduction: "none", "mean", "sum"
    
    Returns:
        Loss tensor
    """
    liger_ce = get_liger_cross_entropy_loss()
    
    if liger_ce is not None:
        try:
            return liger_ce(logits, targets, ignore_index=ignore_index, reduction=reduction)
        except Exception as e:
            logger.warning(f"⚠️ Liger CE failed, fallback to F.cross_entropy: {e}")
    
    # Fallback на стандартный cross_entropy
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
    
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction=reduction)


def chunked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Chunked cross-entropy для экономии памяти при больших batch/seq.
    
    Разбивает вычисление на части чтобы не материализовать весь logits сразу.
    Использует Liger если доступен.
    
    Args:
        logits: [batch, seq, vocab] или [batch*seq, vocab]
        targets: [batch, seq] или [batch*seq]
        chunk_size: размер чанка (tokens)
        ignore_index: индекс для игнорирования
    
    Returns:
        Per-token loss [batch*seq] или [batch, seq]
    """
    original_shape = logits.shape[:-1]  # [batch, seq] или [batch*seq]
    
    # Flatten если нужно
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
    
    total_tokens = logits.size(0)
    
    if total_tokens <= chunk_size:
        # Достаточно маленький — считаем сразу
        loss = liger_cross_entropy(logits, targets, ignore_index=ignore_index, reduction="none")
        return loss.reshape(original_shape) if len(original_shape) == 2 else loss
    
    # Chunked computation
    all_losses = []
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        chunk_logits = logits[start:end]
        chunk_targets = targets[start:end]
        
        chunk_loss = liger_cross_entropy(
            chunk_logits, 
            chunk_targets, 
            ignore_index=ignore_index, 
            reduction="none"
        )
        all_losses.append(chunk_loss)
    
    loss = torch.cat(all_losses, dim=0)
    return loss.reshape(original_shape) if len(original_shape) == 2 else loss


class LigerOptimizedLogProbs:
    """
    Оптимизированное вычисление log-probabilities с Liger.
    
    Основные оптимизации:
    1. Chunked forward pass для экономии памяти
    2. Liger CrossEntropy вместо F.cross_entropy
    3. Опциональный gradient checkpointing
    """
    
    def __init__(
        self,
        chunk_size: int = 2048,
        use_liger: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        self.chunk_size = chunk_size
        self.use_liger = use_liger and is_liger_available()
        self.use_gradient_checkpointing = use_gradient_checkpointing
    
    @torch.no_grad()
    def compute_log_probs(
        self,
        model: "PreTrainedModel",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисляет log-probabilities для последовательности.
        
        Args:
            model: Модель
            input_ids: [batch, seq]
            attention_mask: [batch, seq]
        
        Returns:
            log_probs: [batch, seq-1]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        # Если batch маленький — считаем сразу
        if batch_size * seq_len <= self.chunk_size:
            return self._forward_and_compute_logprobs(
                model, input_ids, attention_mask, position_ids
            )
        
        # Chunked по batch dimension
        all_log_probs = []
        for i in range(0, batch_size, max(1, self.chunk_size // seq_len)):
            end_i = min(i + max(1, self.chunk_size // seq_len), batch_size)
            
            chunk_log_probs = self._forward_and_compute_logprobs(
                model,
                input_ids[i:end_i],
                attention_mask[i:end_i],
                position_ids[i:end_i],
            )
            all_log_probs.append(chunk_log_probs)
            
            # Очищаем cache между чанками
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(all_log_probs, dim=0)
    
    def _forward_and_compute_logprobs(
        self,
        model: "PreTrainedModel",
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass + log probs computation."""
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        
        logits = output.logits[:, :-1]  # [batch, seq-1, vocab]
        targets = input_ids[:, 1:]  # [batch, seq-1]
        
        # Вычисляем log probs
        # Используем chunked CE для экономии памяти на vocab dimension
        nll = chunked_cross_entropy(
            logits,
            targets,
            chunk_size=self.chunk_size,
            ignore_index=-100,
        )
        
        return -nll  # NLL -> log probs


# Глобальный инстанс для использования в rollout.py
_liger_log_probs_computer: Optional[LigerOptimizedLogProbs] = None


def get_liger_log_probs_computer(
    chunk_size: int = 2048,
    use_liger: bool = True,
) -> LigerOptimizedLogProbs:
    """Возвращает глобальный инстанс LigerOptimizedLogProbs."""
    global _liger_log_probs_computer
    
    if _liger_log_probs_computer is None:
        _liger_log_probs_computer = LigerOptimizedLogProbs(
            chunk_size=chunk_size,
            use_liger=use_liger,
        )
    
    return _liger_log_probs_computer


# ============================================================
# FUSED GRPO LOSS MODULE (для trainer.py)
# ============================================================

class LigerGRPOLossModule(nn.Module):
    """
    Обёртка над LigerFusedLinearGRPOLoss для удобного использования в GRPOTrainer.
    
    🔥 Главная оптимизация для GRPO:
    - НЕ материализует logits [batch, seq, vocab] — экономия гигабайт!
    - Fused forward: hidden_states -> lm_head -> loss в одном kernel
    - Встроенный KL penalty (k3 estimator)
    - Поддержка всех loss types: grpo, dapo, dr_grpo, bnpo
    
    Использование:
        loss_module = LigerGRPOLossModule(model, config)
        loss, metrics = loss_module(
            hidden_states,  # outputs.hidden_states[-1]
            selected_token_ids,
            attention_mask,
            advantages,
            ref_per_token_logps=...,
            old_per_token_logps=...,
        )
    """
    
    def __init__(
        self,
        model: "PreTrainedModel",
        beta: float = 0.04,
        loss_type: str = "grpo",
        epsilon: float = 0.2,
        max_completion_length: Optional[int] = None,
        chunk_size: int = 1,
        use_ref_model: bool = False,
        compiled: bool = True,
    ):
        super().__init__()
        
        LigerFusedLinearGRPOLoss = get_liger_fused_linear_grpo()
        if LigerFusedLinearGRPOLoss is None:
            raise RuntimeError("LigerFusedLinearGRPOLoss недоступен!")
        
        self.loss_fn = LigerFusedLinearGRPOLoss(
            beta=beta,
            loss_type=loss_type,
            epsilon_low=epsilon,
            epsilon_high=epsilon,
            max_completion_length=max_completion_length,
            chunk_size=chunk_size,
            use_ref_model=use_ref_model,
            compiled=compiled,
            importance_sampling_level="token",
            temperature=1.0,
        )
        
        # Сохраняем ссылку на lm_head
        self.lm_head_weight = model.lm_head.weight
        self.lm_head_bias = getattr(model.lm_head, 'bias', None)
        
        self.beta = beta
        self.loss_type = loss_type
        logger.info(f"✅ LigerGRPOLossModule инициализирован (loss_type={loss_type}, beta={beta})")
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden] или [batch*seq, hidden]
        selected_token_ids: torch.Tensor,  # [batch, seq]
        attention_mask: torch.Tensor,  # [batch, seq]
        advantages: torch.Tensor,  # [batch]
        ref_per_token_logps: Optional[torch.Tensor] = None,  # [batch, seq]
        old_per_token_logps: Optional[torch.Tensor] = None,  # [batch, seq]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass для GRPO loss.
        
        Returns:
            loss: scalar loss
            metrics: dict с метриками (kl_div, clip_ratio)
        """
        batch_size = selected_token_ids.shape[0]
        seq_len = selected_token_ids.shape[1]
        
        # Reshape hidden_states если нужно: [batch, seq, hidden] -> [batch*seq, hidden]
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        
        # Вызываем fused loss
        result = self.loss_fn(
            hidden_states,           # [batch*seq, hidden]
            self.lm_head_weight,     # [vocab, hidden]
            selected_token_ids,      # [batch, seq]
            attention_mask,          # [batch, seq]
            advantages,              # [batch]
            bias=self.lm_head_bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
        )
        
        # LigerFusedLinearGRPOLoss возвращает (loss, [kl_div, clip_ratio]) или просто loss
        if isinstance(result, tuple):
            loss = result[0]
            metrics_list = result[1] if len(result) > 1 else []
        else:
            loss = result
            metrics_list = []
        
        # Парсим метрики
        metrics = {}
        if len(metrics_list) >= 1 and self.beta != 0.0:
            metrics["kl_div"] = metrics_list[0].item() if hasattr(metrics_list[0], 'item') else float(metrics_list[0])
        if len(metrics_list) >= 2:
            clip_idx = 1 if self.beta != 0.0 else 0
            if clip_idx < len(metrics_list):
                metrics["clip_ratio"] = metrics_list[clip_idx].item() if hasattr(metrics_list[clip_idx], 'item') else float(metrics_list[clip_idx])
        
        return loss, metrics


# ============================================================
# FUSED LINEAR CROSS-ENTROPY ДЛЯ PRETRAIN/SFT
# ============================================================

class LigerFusedCEModule(nn.Module):
    """
    Обёртка над LigerFusedLinearCrossEntropyLoss для pretrain/SFT.
    
    🔥 Главная оптимизация для language modeling:
    - НЕ материализует logits [batch, seq, vocab]
    - Fused forward: hidden_states -> lm_head -> CE loss в одном kernel
    - До 80% экономии памяти на vocab dimension!
    
    ⚠️ ВАЖНО: Для Causal LM мы делаем СДВИГ (shift):
    - hidden_states[:-1] предсказывают labels[1:]
    - Это стандартное поведение для language modeling (next-token prediction)
    
    ⚠️ DeepSpeed ZeRO-3: Используем GatheredParameters для сбора lm_head.weight
    перед вызовом Liger, так как Triton kernels не работают с шардированными параметрами.
    
    Использование:
        # Вместо:
        logits = model.lm_head(hidden_states)
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = F.cross_entropy(shift_logits.view(-1, vocab), shift_labels.view(-1))
        
        # Используйте:
        loss_module = LigerFusedCEModule(model, accelerator=accelerator)
        loss = loss_module(hidden_states, labels)  # Сдвиг делается внутри!
    """
    
    def __init__(
        self,
        model: "PreTrainedModel",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        accelerator: Optional[Any] = None,
    ):
        super().__init__()
        
        LigerFusedLinearCE = get_liger_fused_linear_ce()
        if LigerFusedLinearCE is None:
            raise RuntimeError("LigerFusedLinearCrossEntropyLoss недоступен!")
        
        self.loss_fn = LigerFusedLinearCE(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
        
        # Сохраняем ссылку на модель (не на weight!) для поддержки ZeRO-3
        self.model = model
        self.accelerator = accelerator
        
        # Определяем, используется ли ZeRO-3
        self.is_zero3 = False
        if accelerator is not None:
            try:
                ds_plugin = getattr(accelerator.state, 'deepspeed_plugin', None)
                if ds_plugin is not None:
                    zero_stage = getattr(ds_plugin, 'zero_stage', 0)
                    self.is_zero3 = zero_stage == 3
                    if self.is_zero3:
                        logger.info("✅ LigerFusedCEModule: обнаружен ZeRO-3, будем использовать GatheredParameters")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось определить ZeRO stage: {e}")
        
        logger.info(f"✅ LigerFusedCEModule инициализирован (ignore_index={ignore_index}, causal_shift=True)")
    
    def _get_lm_head_params(self):
        """Получает weight и bias из lm_head модели."""
        lm_head = self.model.lm_head
        weight = lm_head.weight
        bias = getattr(lm_head, 'bias', None)
        return weight, bias
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden]
        labels: torch.Tensor,  # [batch, seq]
    ) -> torch.Tensor:
        """
        Forward pass для Fused Linear CrossEntropy с CAUSAL SHIFT.
        
        ⚠️ ВАЖНО: Для Causal LM (next-token prediction) мы делаем сдвиг:
        - hidden_states[:, :-1] → предсказывают → labels[:, 1:]
        - Это эквивалентно shift_logits[:-1] vs shift_labels[1:] в стандартном CE
        
        Args:
            hidden_states: последний hidden state модели [batch, seq, hidden]
            labels: target token ids [batch, seq]
        
        Returns:
            loss: scalar (если reduction="mean") или [batch*(seq-1)] (если reduction="none")
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states должен быть 3D [batch, seq, hidden], получен {hidden_states.dim()}D")
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # ============================================================
        # CAUSAL SHIFT для next-token prediction
        # ============================================================
        # hidden_states[:, :-1] предсказывают labels[:, 1:]
        # Это стандартный подход для language modeling
        shift_hidden = hidden_states[:, :-1, :].contiguous()  # [batch, seq-1, hidden]
        shift_labels = labels[:, 1:].contiguous()              # [batch, seq-1]
        
        # Reshape: [batch, seq-1, hidden] -> [batch*(seq-1), hidden]
        shift_hidden = shift_hidden.reshape(-1, hidden_size)
        shift_labels = shift_labels.reshape(-1)
        
        # ============================================================
        # ZeRO-3: собираем lm_head.weight через GatheredParameters
        # ============================================================
        if self.is_zero3:
            return self._forward_with_gathered_params(shift_hidden, shift_labels)
        else:
            # Стандартный путь: прямой доступ к параметрам
            weight, bias = self._get_lm_head_params()
            return self.loss_fn(weight, shift_hidden, shift_labels, bias)
    
    def _forward_with_gathered_params(
        self,
        shift_hidden: torch.Tensor,
        shift_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass с GatheredParameters для ZeRO-3."""
        try:
            from deepspeed.runtime.zero.partition_parameters import GatheredParameters
            
            lm_head = self.model.lm_head
            params_to_gather = [lm_head.weight]
            if hasattr(lm_head, 'bias') and lm_head.bias is not None:
                params_to_gather.append(lm_head.bias)
            
            # modifier_rank=None: все ранки могут читать собранные параметры
            with GatheredParameters(params_to_gather, modifier_rank=None):
                weight = lm_head.weight
                bias = getattr(lm_head, 'bias', None)
                return self.loss_fn(weight, shift_hidden, shift_labels, bias)
                
        except ImportError:
            logger.warning("⚠️ DeepSpeed не найден, используем прямой доступ к параметрам")
            weight, bias = self._get_lm_head_params()
            return self.loss_fn(weight, shift_hidden, shift_labels, bias)


def create_liger_grpo_loss(
    model: "PreTrainedModel",
    config: Any,  # GRPOConfig
) -> Optional[LigerGRPOLossModule]:
    """
    Создаёт LigerGRPOLossModule если Liger доступен и включён.
    
    Args:
        model: HuggingFace модель
        config: GRPOConfig с параметрами
    
    Returns:
        LigerGRPOLossModule или None
    """
    if not getattr(config, 'use_liger', False):
        return None
    
    if not is_liger_available():
        logger.warning("⚠️ Liger недоступен, используем стандартный GRPO loss")
        return None
    
    if get_liger_fused_linear_grpo() is None:
        logger.warning("⚠️ LigerFusedLinearGRPOLoss недоступен, используем стандартный GRPO loss")
        return None
    
    try:
        loss_module = LigerGRPOLossModule(
            model=model,
            beta=getattr(config, 'kl_weight', 0.04),
            loss_type=getattr(config, 'liger_grpo_loss_type', 'dapo'),
            epsilon=getattr(config, 'epsilon', 0.2),
            max_completion_length=getattr(config, 'max_new_tokens', 512),
            chunk_size=getattr(config, 'liger_chunk_size', 1),
            use_ref_model=getattr(config, 'kl_weight', 0) > 0,
            compiled=True,
        )
        logger.info("🦁 LigerFusedLinearGRPOLoss активирован — logits НЕ материализуются!")
        return loss_module
    except Exception as e:
        logger.warning(f"⚠️ Не удалось создать LigerGRPOLossModule: {e}")
        return None


def create_liger_fused_ce(
    model: "PreTrainedModel",
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    accelerator: Optional[Any] = None,
) -> Optional[LigerFusedCEModule]:
    """
    Создаёт LigerFusedCEModule если Liger доступен.
    
    ⚠️ ВАЖНО: Для DeepSpeed ZeRO-3 необходимо передать accelerator!
    Без этого Liger не сможет корректно собрать шардированные веса lm_head.
    
    Использование для pretrain/SFT:
        loss_fn = create_liger_fused_ce(model, accelerator=accelerator)
        if loss_fn:
            # Используем fused loss (не материализует logits)
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            loss = loss_fn(hidden, labels)
        else:
            # Fallback на стандартный метод
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
    """
    if not is_liger_available():
        return None
    
    if get_liger_fused_linear_ce() is None:
        return None
    
    try:
        return LigerFusedCEModule(
            model=model,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            accelerator=accelerator,
        )
    except Exception as e:
        logger.warning(f"⚠️ Не удалось создать LigerFusedCEModule: {e}")
        return None
