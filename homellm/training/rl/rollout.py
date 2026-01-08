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
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def compute_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Вычисляет log-вероятности для последовательности.
    
    Args:
        model: Языковая модель
        sequence_ids: Token IDs [batch, seq_len]
        attention_mask: Маска внимания [batch, seq_len]
        
    Returns:
        Log-вероятности [batch, seq_len-1]
    """
    # Position IDs
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    
    # Forward pass
    output = model(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    
    # Log probs для следующих токенов
    logits = output.logits[:, :-1]  # [batch, seq_len-1, vocab]
    target_ids = sequence_ids[:, 1:]  # [batch, seq_len-1]
    
    log_probs = sequence_log_probs_from_logits(
        logits.to(torch.float32),
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
) -> List[Rollout]:
    """
    Генерирует rollout'ы для списка промптов.
    
    Args:
        model: Языковая модель (политика)
        tokenizer: Токенизатор
        prompts: Список промптов (вопросов)
        reference_answers: Эталонные ответы для вычисления reward
        reward_fn: Функция вычисления reward(completion, reference) -> float
        config: Конфигурация GRPO
        reference_model: Референсная модель для KL (опционально)
        device: Устройство для вычислений
        
    Returns:
        Список Rollout для каждого промпта
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
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
        # Токенизация промпта
        prompt_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
        ).to(device)
        
        prompt_length = prompt_inputs["input_ids"].size(1)
        
        # Дублируем для группы генераций
        input_ids = prompt_inputs["input_ids"].repeat(config.group_size, 1)
        attention_mask = prompt_inputs["attention_mask"].repeat(config.group_size, 1)
        
        # Генерация
        outputs = model.generate(
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
            reward = reward_fn(
                completion=completion,
                reference_answer=ref_answer,
                reasoning_format=config.reasoning_format,
                is_truncated=is_truncated[i],
            )
            rewards[i] = reward
        
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
        
        # Убираем padding из completion
        non_pad_mask = completion_ids != (tokenizer.pad_token_id or tokenizer.eos_token_id)
        completion_ids = completion_ids[non_pad_mask]
        
        sequence_ids = torch.cat([rollout.prompt_ids.to(device), completion_ids.to(device)])
        
        # Attention mask
        attention_mask = torch.ones_like(sequence_ids)
        
        # Action mask (только для completion токенов)
        action_mask = torch.zeros(sequence_ids.size(0) - 1, dtype=torch.bool, device=device)
        action_mask[prompt_length - 1:] = True  # Начинаем с позиции после prompt
        
        # Log probs текущей политики
        log_probs = compute_log_probs(
            model,
            sequence_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
        ).squeeze(0)
        
        # Log probs референсной модели (для KL)
        log_probs_ref = None
        if reference_model is not None and config.kl_weight > 0:
            log_probs_ref = compute_log_probs(
                reference_model,
                sequence_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
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
