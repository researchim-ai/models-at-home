"""
🎓 SDPO (Self-Distilled Policy Optimization) Utils

Ключевые компоненты:
1. build_teacher_batch - создаёт teacher inputs для ВСЕХ сэмплов
2. compute_sdpo_loss - вычисляет distillation loss с маской
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SDPOBatch:
    """Батч данных для SDPO teacher forward."""
    teacher_input_ids: torch.Tensor      # [batch, seq] - teacher inputs
    teacher_attention_mask: torch.Tensor  # [batch, seq]
    teacher_position_ids: torch.Tensor    # [batch, seq]
    self_distillation_mask: torch.Tensor  # [batch] - True если reprompted
    responses: torch.Tensor               # [batch, resp_len] - completion tokens


def compute_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет position_ids из attention_mask.
    """
    return torch.clamp(attention_mask.long().cumsum(dim=-1) - 1, min=0)


def build_teacher_batch(
    prompts: List[str],
    responses: torch.Tensor,  # [batch, resp_len]
    response_mask: torch.Tensor,  # [batch, resp_len]
    tokenizer,
    successful_trajectories: Dict[int, Dict],
    prompt_ids: List[int],
    completion_texts: Optional[List[str]] = None,
    config: Any = None,
    device: torch.device = None,
) -> Tuple[SDPOBatch, Dict[str, float]]:
    """
    Создаёт teacher batch для SDPO (как _maybe_build_self_distillation_batch в оригинале).
    
    🔥 ВАЖНО: Teacher input создаётся для ВСЕХ сэмплов!
    - С solution/feedback → reprompted input
    - Без ничего → оригинальный промпт
    
    Args:
        prompts: Оригинальные промпты [batch]
        responses: Completion tokens [batch, resp_len]
        response_mask: Маска completions [batch, resp_len]
        tokenizer: Токенизатор
        successful_trajectories: Успешные траектории по prompt_id
        prompt_ids: ID промптов для каждого сэмпла
        completion_texts: Тексты completions (для dont_reprompt_on_self_success)
        config: SDPOConfig
        device: Устройство
        
    Returns:
        (SDPOBatch, metrics)
    """
    batch_size = len(prompts)
    if device is None:
        device = responses.device
    
    # Параметры из конфига
    success_threshold = getattr(config, 'success_threshold', 0.9)
    include_feedback = getattr(config, 'include_environment_feedback', True)
    feedback_only_without_solution = getattr(config, 'environment_feedback_only_without_solution', False)
    dont_self_reprompt = getattr(config, 'dont_reprompt_on_self_success', True)
    remove_thinking = getattr(config, 'remove_thinking_from_demonstration', False)
    max_reprompt_len = getattr(config, 'max_reprompt_len', 4096)
    
    # Шаблоны
    reprompt_template = getattr(config, 'reprompt_template', 
        "{prompt}\n\n{solution}{feedback}")
    solution_template = getattr(config, 'solution_template',
        "Here is a successful solution for reference:\n{successful_previous_attempt}\n\n")
    feedback_template = getattr(config, 'feedback_template',
        "Previous attempt feedback:\n{feedback_raw}\n\n")
    
    # Собираем solution и feedback для каждого сэмпла
    solution_strs = []
    feedback_list = []
    
    for idx in range(batch_size):
        pid = prompt_ids[idx] if prompt_ids else None
        solution = None
        feedback = None
        
        if pid is not None and pid in successful_trajectories:
            traj_data = successful_trajectories[pid]
            
            # Поддерживаем обе структуры
            if isinstance(traj_data, dict):
                successful_list = traj_data.get('successful', [])
                current_list = traj_data.get('current', [])
            else:
                successful_list = traj_data if isinstance(traj_data, list) else []
                current_list = []
            
            # Выбираем solution
            if successful_list:
                current_completion = completion_texts[idx] if completion_texts else None
                
                if dont_self_reprompt and current_completion:
                    # Не используем текущий completion как solution
                    other_successful = [t for t in successful_list 
                                       if t.get('completion', t.get('response', '')) != current_completion]
                    if other_successful:
                        import random
                        traj = random.choice(other_successful)
                        solution = traj.get('completion', traj.get('response', ''))
                else:
                    import random
                    traj = random.choice(successful_list)
                    solution = traj.get('completion', traj.get('response', ''))
            
            # Получаем feedback
            if include_feedback and current_list:
                current_completion = completion_texts[idx] if completion_texts else None
                for traj in reversed(current_list):
                    if current_completion and traj.get('completion', traj.get('response', '')) == current_completion:
                        feedback = traj.get('feedback')
                        break
                if feedback is None and current_list:
                    feedback = current_list[-1].get('feedback')
        
        # Применяем feedback_only_without_solution
        if feedback_only_without_solution and solution is not None:
            feedback = None
        
        # Удаляем thinking если нужно
        if solution and remove_thinking:
            import re
            solution = re.sub(r'<think>.*?</think>', '', solution, flags=re.DOTALL).strip()
        
        solution_strs.append(solution)
        feedback_list.append(feedback)
    
    # Строим teacher messages для каждого сэмпла
    teacher_messages = []
    for i in range(batch_size):
        has_solution = solution_strs[i] is not None
        has_feedback = feedback_list[i] is not None
        
        # Определяем используется ли feedback
        use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)
        
        # Строим секции
        solution_section = ""
        if has_solution:
            solution_section = solution_template.format(
                successful_previous_attempt=solution_strs[i]
            )
        
        feedback_section = ""
        if use_feedback:
            feedback_section = feedback_template.format(
                feedback_raw=feedback_list[i]
            )
        
        # Комбинируем
        if use_feedback or has_solution:
            reprompt_text = reprompt_template.format(
                prompt=prompts[i],
                solution=solution_section,
                feedback=feedback_section,
            )
        else:
            reprompt_text = prompts[i]
        
        teacher_messages.append(reprompt_text)
    
    if hasattr(tokenizer, 'apply_chat_template'):
        chat_messages = [[{"role": "user", "content": msg}] for msg in teacher_messages]
        try:
            teacher_encoding = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                max_length=max_reprompt_len,
                padding=True,
                truncation=True,
            )
            teacher_encoding = {k: v.to(device) for k, v in teacher_encoding.items()}
        except Exception:
            # Fallback на простую токенизацию
            teacher_encoding = tokenizer(
                teacher_messages,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_reprompt_len,
            ).to(device)
    else:
        teacher_encoding = tokenizer(
            teacher_messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_reprompt_len,
        ).to(device)
    
    # Конкатенируем с responses
    teacher_input_ids = torch.cat([teacher_encoding['input_ids'], responses], dim=1)
    teacher_attention_mask = torch.cat([teacher_encoding['attention_mask'], response_mask], dim=1)
    teacher_position_ids = compute_position_ids(teacher_attention_mask)
    
    # self_distillation_mask = True если есть solution ИЛИ feedback
    feedback_used = [
        feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
        for i in range(batch_size)
    ]
    self_distillation_mask = torch.tensor(
        [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)],
        dtype=torch.float32,
        device=device
    )
    
    # Метрики
    num_with_solution = sum(1 for s in solution_strs if s is not None)
    num_with_feedback = sum(1 for f in feedback_list if f is not None)
    num_with_feedback_used = sum(1 for f in feedback_used if f)
    
    metrics = {
        "sdpo/success_sample_fraction": num_with_solution / batch_size,
        "sdpo/feedback_available_fraction": num_with_feedback / batch_size,
        "sdpo/feedback_used_fraction": num_with_feedback_used / batch_size,
        "sdpo/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
    }
    
    return SDPOBatch(
        teacher_input_ids=teacher_input_ids,
        teacher_attention_mask=teacher_attention_mask,
        teacher_position_ids=teacher_position_ids,
        self_distillation_mask=self_distillation_mask,
        responses=responses,
    ), metrics


def compute_sdpo_loss(
    student_log_probs: torch.Tensor,  # [batch, seq]
    teacher_log_probs: torch.Tensor,  # [batch, seq]
    response_mask: torch.Tensor,      # [batch, seq]
    self_distillation_mask: torch.Tensor,  # [batch]
    old_log_probs: Optional[torch.Tensor] = None,  # [batch, seq]
    student_topk_log_probs: Optional[torch.Tensor] = None,  # [batch, seq, k]
    teacher_topk_log_probs: Optional[torch.Tensor] = None,  # [batch, seq, k]
    student_all_log_probs: Optional[torch.Tensor] = None,   # [batch, seq, vocab]
    teacher_all_log_probs: Optional[torch.Tensor] = None,   # [batch, seq, vocab]
    alpha: float = 0.0,
    is_clip: Optional[float] = 2.0,
    full_logit_distillation: bool = True,
    distillation_topk: Optional[int] = None,
    distillation_add_tail: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Вычисляет SDPO loss точно как compute_self_distillation_loss в оригинале.
    
    Args:
        student_log_probs: Per-token log probs student [batch, seq]
        teacher_log_probs: Per-token log probs teacher [batch, seq]
        response_mask: Маска токенов [batch, seq]
        self_distillation_mask: Маска сэмплов с reprompt [batch]
        old_log_probs: Rollout log probs (для IS clipping)
        student_topk_log_probs: Top-K log probs student [batch, seq, k]
        teacher_topk_log_probs: Top-K log probs teacher [batch, seq, k]
        student_all_log_probs: Full vocab log probs student
        teacher_all_log_probs: Full vocab log probs teacher
        alpha: KL interpolation (0=forward, 1=reverse, 0.5=JSD)
        is_clip: Importance sampling clip value
        full_logit_distillation: Использовать full/topk logits
        distillation_topk: K для top-k distillation
        distillation_add_tail: Добавлять tail bucket
        
    Returns:
        (loss, metrics)
    """
    metrics = {}
    
    # 🔥 Loss mask = response_mask * self_distillation_mask
    loss_mask = response_mask.float()
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
    
    # ============================================================
    # DISTILLATION LOSS
    # ============================================================
    use_topk = distillation_topk is not None
    has_topk = student_topk_log_probs is not None and teacher_topk_log_probs is not None
    has_full = student_all_log_probs is not None and teacher_all_log_probs is not None
    
    if full_logit_distillation and use_topk and has_topk:
        # TOP-K DISTILLATION
        def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
            log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_s = torch.clamp(log_s, max=-1e-7)
            tail_log = torch.log(-torch.expm1(log_s))
            return torch.cat([log_probs, tail_log], dim=-1)
        
        def renorm_topk(logp: torch.Tensor) -> torch.Tensor:
            logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
            return logp - logZ
        
        if distillation_add_tail:
            student_distill = add_tail(student_topk_log_probs)
            teacher_distill = add_tail(teacher_topk_log_probs)
        else:
            student_distill = renorm_topk(student_topk_log_probs)
            teacher_distill = renorm_topk(teacher_topk_log_probs)
        
        metrics["sdpo_mode"] = "topk"
        metrics["sdpo_topk_k"] = student_topk_log_probs.shape[-1]
        
        # KL divergence
        if alpha == 0.0:
            kl_loss = F.kl_div(student_distill, teacher_distill, reduction="none", log_target=True)
        elif alpha == 1.0:
            kl_loss = F.kl_div(teacher_distill, student_distill, reduction="none", log_target=True)
        else:
            alpha_t = torch.tensor(alpha, dtype=student_distill.dtype, device=student_distill.device)
            mixture = torch.logsumexp(
                torch.stack([student_distill + torch.log(1 - alpha_t), 
                            teacher_distill + torch.log(alpha_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture, teacher_distill, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture, student_distill, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha_t)
        
        per_token_loss = kl_loss.sum(dim=-1)  # [batch, seq]
        
    elif full_logit_distillation and not use_topk and has_full:
        # FULL VOCAB DISTILLATION
        metrics["sdpo_mode"] = "full_vocab"
        
        if alpha == 0.0:
            kl_loss = F.kl_div(student_all_log_probs, teacher_all_log_probs, reduction="none", log_target=True)
        elif alpha == 1.0:
            kl_loss = F.kl_div(teacher_all_log_probs, student_all_log_probs, reduction="none", log_target=True)
        else:
            alpha_t = torch.tensor(alpha, dtype=student_all_log_probs.dtype, device=student_all_log_probs.device)
            mixture = torch.logsumexp(
                torch.stack([student_all_log_probs + torch.log(1 - alpha_t),
                            teacher_all_log_probs + torch.log(alpha_t)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture, teacher_all_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture, student_all_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha_t)
        
        per_token_loss = kl_loss.sum(dim=-1)
        
    else:
        # SIMPLE PER-TOKEN DISTILLATION (fallback)
        metrics["sdpo_mode"] = "simple"
        assert alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs
    
    # IS Clipping
    if is_clip is not None and old_log_probs is not None:
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio
    
    # 🔥 AGGREGATION (как в оригинале)
    batch_num_tokens = loss_mask.sum().clamp(min=1.0)
    loss = (per_token_loss * loss_mask).sum() / batch_num_tokens
    
    # 🔥 Защита grad_fn: если mask все нули, loss теряет grad
    # Восстанавливаем через student_log_probs
    if not loss.requires_grad and student_log_probs.requires_grad:
        loss = loss + (student_log_probs * 0).sum()
    
    metrics["sdpo_loss"] = loss.item()
    metrics["sdpo_empty_batch"] = self_distillation_mask.sum().item() == 0
    metrics["sdpo_num_reprompted"] = int(self_distillation_mask.sum().item())
    
    return loss, metrics
