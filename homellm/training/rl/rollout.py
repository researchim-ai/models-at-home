"""
Генерация rollout'ов (completions) для GRPO.

Rollout = генерация нескольких ответов на один промпт с вычислением rewards.
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, Any, Dict
import torch
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)

from .experience import Experience
from .config import GRPOConfig


@dataclass
class Rollout:
    """
    Результат rollout'а (генерации).
    
    Attributes:
        prompt: Исходный промпт
        prompt_ids: Token IDs промпта
        completions: Список сгенерированных ответов
        completion_ids: Token IDs ответов
        rewards: Rewards для каждого ответа
        is_truncated: Флаги обрезки (если достигнут max_length)
    """
    prompt: str
    prompt_ids: torch.Tensor
    completions: List[str]
    completion_ids: List[torch.Tensor]
    rewards: torch.Tensor
    is_truncated: List[bool]
    
    # Метаданные для вычисления reward
    metadata: Optional[Dict[str, Any]] = None


def build_reasoning_prompt(
    question: str,
    tokenizer: PreTrainedTokenizer,
    reasoning_format: str = "deepseek",
    system_prompt: Optional[str] = None,
) -> str:
    """
    Строит промпт для reasoning задачи.
    
    Args:
        question: Вопрос/задача
        tokenizer: Токенизатор для применения chat template
        reasoning_format: "deepseek" (<think>) или "simple" (<reasoning>)
        system_prompt: Системный промпт (опционально)
        
    Returns:
        Отформатированный промпт
    """
    if reasoning_format == "deepseek":
        default_system = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>"""
    else:  # simple
        default_system = """Отвечай строго в формате:
<reasoning>
(Шаги решения)
</reasoning>
<answer>
(Короткий итоговый ответ)
</answer>"""
    
    system = system_prompt or default_system
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback для токенизаторов без chat template
        prompt = f"{system}\n\nUser: {question}\n\nAssistant:"
    
    return prompt


def extract_answer_from_completion(
    completion: str,
    reasoning_format: str = "deepseek",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Извлекает reasoning и ответ из completion.
    
    Args:
        completion: Сгенерированный текст
        reasoning_format: "deepseek" или "simple"
        
    Returns:
        Tuple[reasoning, answer]
    """
    if reasoning_format == "deepseek":
        reasoning_pat = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        answer_pat = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    else:
        reasoning_pat = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
        answer_pat = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    
    reasoning_match = reasoning_pat.search(completion)
    answer_match = answer_pat.search(completion)
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    answer = answer_match.group(1).strip() if answer_match else None
    
    return reasoning, answer


def sequence_log_probs_from_logits(
    logits: torch.Tensor,
    output_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Вычисляет log-вероятности токенов из logits.
    
    Args:
        logits: Выходы модели [batch, seq_len, vocab_size]
        output_ids: ID токенов [batch, seq_len]
        
    Returns:
        Log-вероятности [batch, seq_len]
    """
    # ОПТИМИЗАЦИЯ ПАМЯТИ: Используем cross_entropy(reduction='none') вместо log_softmax + gather
    # Старый подход: log_probs = F.log_softmax(logits, dim=-1).gather(...)
    # Это создавало тензор [Batch, SeqLen, Vocab], который для Qwen (152k vocab) занимал ~5-6 GB
    # Новый подход: fused kernel cross_entropy вычисляет только нужные значения
    # log(p) = -cross_entropy
    
    # Reshape для cross_entropy: [N, C] и [N]
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    ids_flat = output_ids.reshape(-1)
    
    # Вычисляем negative log likelihood (loss) для каждого токена
    # reduction='none' возвращает тензор того же размера, что и input (batch * seq)
    nll = F.cross_entropy(logits_flat, ids_flat, reduction='none')
    
    # log_prob = -nll
    log_probs = -nll.view(batch_size, seq_len)
    
    return log_probs


def compute_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    accelerator=None,
) -> torch.Tensor:
    """
    Вычисляет log-вероятности для последовательности.
    
    ВАЖНО: Эта функция НЕ использует @torch.no_grad(), чтобы градиенты могли проходить
    при использовании в обучении. Используйте torch.no_grad() вручную там где нужно.
    
    Args:
        model: Языковая модель (может быть обернута в DDP)
        sequence_ids: Token IDs [batch, seq_len]
        attention_mask: Маска внимания [batch, seq_len]
        accelerator: Accelerator объект для unwrap модели (опционально)
        
    Returns:
        Log-вероятности [batch, seq_len-1]
    """
    # ВАЖНО (память): для обучения в distributed режиме НЕ делаем unwrap через Accelerator.
    # Иначе accelerate может оборачивать forward и конвертировать выходы в fp32 (convert_to_fp32),
    # что сильно увеличивает пик памяти на больших vocab (Qwen ~152k) и длинных seq.
    # Для DDP/FSDP forward() доступен напрямую на подготовленной модели.
    forward_model = model
    
    # Position IDs
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    
    # Forward pass
    output = forward_model(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    
    # Log probs для следующих токенов
    logits = output.logits[:, :-1]  # [batch, seq_len-1, vocab]
    target_ids = sequence_ids[:, 1:]  # [batch, seq_len-1]
    
    # ВАЖНО: НЕ конвертируем dtype - работаем с исходным dtype модели
    # Конвертация может разорвать граф градиентов
    # sequence_log_probs_from_logits работает с любым float dtype
    log_probs = sequence_log_probs_from_logits(
        logits,
        target_ids,
    )
    
    return log_probs


@torch.no_grad()
def generate_rollouts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    reference_answers: List[str],
    reward_fn: Callable,
    config: GRPOConfig,
    reference_model: Optional[PreTrainedModel] = None,
    device: Optional[torch.device] = None,
    accelerator=None,
    prompt_ids: Optional[List[int]] = None,
) -> List[Rollout]:
    """
    Генерирует rollout'ы для списка промптов.
    
    Args:
        model: Языковая модель (политика) - может быть обернута в DDP
        tokenizer: Токенизатор
        prompts: Список промптов (вопросов)
        reference_answers: Эталонные ответы для вычисления reward
        reward_fn: Функция вычисления reward(completion, reference) -> float
        config: Конфигурация GRPO
        reference_model: Референсная модель для KL (опционально)
        device: Устройство для вычислений
        accelerator: Accelerator объект для unwrap модели (опционально)
        
    Returns:
        Список Rollout для каждого промпта
    """
    # ВАЖНО: Если модель обернута в DDP, нужно использовать unwrapped модель для generate()
    # DDP не передает методы типа generate() напрямую
    if accelerator is not None:
        # Используем unwrapped модель для генерации
        unwrapped_model = accelerator.unwrap_model(model)
    elif hasattr(model, 'module'):
        # Если модель обернута в DDP напрямую (без accelerator)
        unwrapped_model = model.module
    else:
        # Модель не обернута, используем как есть
        unwrapped_model = model
    
    unwrapped_model.eval()
    if device is None:
        device = next(unwrapped_model.parameters()).device
    
    rollouts = []
    
    # Конфигурация генерации
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    for prompt_idx, (prompt, ref_answer) in enumerate(zip(prompts, reference_answers)):
        # Токенизация промпта.
        # ВАЖНО: build_reasoning_prompt(...) применяется на уровне датасета/тренера
        # (см. GRPOTrainer._train_epoch), поэтому здесь prompt уже может быть
        # "полным" (system+user). Не добавляем system второй раз.
        prompt_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
        ).to(device)
        
        prompt_length = prompt_inputs["input_ids"].size(1)
        
        # ВАЖНО: Параллельная генерация для группы (как в re-grpo)
        # Дублируем промпт group_size раз и генерируем все completions одним батчем
        # Это эффективнее чем последовательная генерация
        input_ids = prompt_inputs["input_ids"].repeat(config.group_size, 1)
        attention_mask = prompt_inputs["attention_mask"].repeat(config.group_size, 1)
        
        # Генерация всех group_size completions параллельно одним батчем
        # ВАЖНО: Используем unwrapped_model для generate() (DDP не поддерживает generate напрямую)
        # ВАЖНО: для FlashAttention generation должен идти под autocast fp16/bf16.
        # Иначе dtype может промоутиться в fp32 (особенно при LoRA) и flash-attn упадёт.
        mp = (getattr(config, "mixed_precision", None) or "bf16").lower()
        use_autocast = torch.cuda.is_available() and mp in ("bf16", "fp16")
        if use_autocast:
            amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
            autocast_ctx = torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with autocast_ctx:
            outputs = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        
        generated_ids = outputs.sequences
        
        # Декодирование completions
        completions = tokenizer.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
        )
        
        # Определяем truncated ответы
        is_truncated = []
        for i in range(config.group_size):
            completion_length = (generated_ids[i, prompt_length:] != tokenizer.pad_token_id).sum().item()
            is_truncated.append(completion_length >= config.max_new_tokens)
        
        # Вычисляем rewards
        rewards = torch.zeros(config.group_size, dtype=torch.float32, device=device)
        for i, completion in enumerate(completions):
            try:
                reward = reward_fn(
                    completion=completion,
                    reference_answer=ref_answer,
                    reasoning_format=config.reasoning_format,
                    is_truncated=is_truncated[i],
                )
                # Проверяем что reward - число
                if not isinstance(reward, (int, float)):
                    import logging
                    logging.warning(
                        f"Reward не число: {type(reward)} = {reward} для completion: {completion[:100]}..."
                    )
                    reward = 0.0
                rewards[i] = float(reward)
            except Exception as e:
                import logging
                logging.error(
                    f"Ошибка при вычислении reward для completion {i}: {e}\n"
                    f"Completion: {completion[:200]}...\n"
                    f"Reference: {ref_answer[:100]}..."
                )
                rewards[i] = 0.0
        
        # Создаём Rollout
        rollout = Rollout(
            prompt=prompt,
            prompt_ids=prompt_inputs["input_ids"][0],
            completions=completions,
            completion_ids=[generated_ids[i, prompt_length:] for i in range(config.group_size)],
            rewards=rewards,
            is_truncated=is_truncated,
            metadata={
                "reference_answer": ref_answer,
                "prompt_idx": prompt_idx,
                "prompt_id": (prompt_ids[prompt_idx] if prompt_ids is not None and prompt_idx < len(prompt_ids) else prompt_idx),
            }
        )
        rollouts.append(rollout)
    
    return rollouts


def rollout_to_experiences(
    rollout: Rollout,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: GRPOConfig,
    reference_model: Optional[PreTrainedModel] = None,
    device: Optional[torch.device] = None,
    accelerator=None,
) -> List[Experience]:
    """
    Конвертирует Rollout в список Experience для обучения.
    
    Args:
        rollout: Результат генерации
        model: Текущая модель (политика)
        tokenizer: Токенизатор
        config: Конфигурация
        reference_model: Референсная модель для KL
        device: Устройство
        
    Returns:
        Список Experience для каждого completion в группе
    """
    if device is None:
        device = next(model.parameters()).device
    
    experiences = []
    prompt_length = rollout.prompt_ids.size(0)
    
    # Вычисляем advantages для группы
    from .loss import compute_advantages
    
    advantages = compute_advantages(
        rollout.rewards,
        use_std_normalization=config.use_std_normalization,
    )
    
    for i in range(len(rollout.completions)):
        # Полная последовательность: prompt + completion
        completion_ids = rollout.completion_ids[i]
        
        # ВАЖНО: Убираем только реальный padding, НЕ EOS!
        # EOS - это действие, которому модель должна учиться
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        
        # Маскируем только реальный padding (после первого EOS или в конце)
        # Но сохраняем EOS токены, так как это действия модели
        non_pad_mask = completion_ids != pad_token_id
        # Если pad_token == eos_token, то не маскируем EOS (это нормально)
        if pad_token_id == tokenizer.eos_token_id:
            # В этом случае pad_token == eos_token, маскируем только padding после первого EOS
            # Находим первый EOS
            eos_positions = (completion_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                # Маскируем всё после первого EOS как padding
                non_pad_mask[first_eos + 1:] = False
                # Но сам EOS оставляем
                non_pad_mask[first_eos] = True
        
        completion_ids = completion_ids[non_pad_mask]
        
        sequence_ids = torch.cat([rollout.prompt_ids.to(device), completion_ids.to(device)])
        
        # Attention mask
        attention_mask = torch.ones_like(sequence_ids)
        
        # Action mask (только для completion токенов, включая EOS)
        # ВАЖНО: EOS токены НЕ маскируются - это действия модели
        action_mask = torch.zeros(sequence_ids.size(0) - 1, dtype=torch.bool, device=device)
        action_mask[prompt_length - 1:] = True  # Начинаем с позиции после prompt
        
        # Маскируем только реальный padding в action_mask
        # (padding уже убран из completion_ids, но на всякий случай)
        if pad_token_id is not None:
            # Если в sequence_ids есть pad_token_id, маскируем их
            pad_positions = (sequence_ids == pad_token_id).nonzero(as_tuple=True)[0]
            for pos in pad_positions:
                if pos > 0:  # Не маскируем первый токен
                    action_mask[pos - 1] = False
        
        # Log probs текущей политики (старые, для сохранения в Experience)
        # Используем no_grad т.к. это старые log_probs, не нужны градиенты
        with torch.no_grad():
            log_probs = compute_log_probs(
                model,
                sequence_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                accelerator=accelerator,
            ).squeeze(0)
        
        # Log probs референсной модели (для KL) - всегда без градиентов
        log_probs_ref = None
        if reference_model is not None and config.kl_weight > 0:
            with torch.no_grad():
                log_probs_ref = compute_log_probs(
                    reference_model,
                    sequence_ids.unsqueeze(0),
                    attention_mask.unsqueeze(0),
                    accelerator=accelerator,
                ).squeeze(0)
        
        exp = Experience(
            sequences=sequence_ids,
            prompt_length=prompt_length,
            action_log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            returns=rollout.rewards[i].unsqueeze(0),
            advantages=advantages[i].unsqueeze(0),
            attention_mask=attention_mask,
            action_mask=action_mask,
            completion_text=rollout.completions[i],
        )
        experiences.append(exp)
    
    return experiences
