"""
Генерация rollout'ов (completions) для GRPO.

Rollout = генерация нескольких ответов на один промпт с вычислением rewards.
"""
import logging
import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, Any, Dict
import torch
import torch.nn.functional as F

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)

# Liger Kernel интеграция для оптимизированного cross-entropy
from homellm.training.rl.liger_utils import (
    liger_cross_entropy,
    chunked_cross_entropy,
    is_liger_available,
)

logger = logging.getLogger(__name__)


@contextmanager
def ds3_gather_for_generation(model, accelerator):
    """
    Context manager для сбора параметров ZeRO-3 перед генерацией.
    
    КРИТИЧНО ДЛЯ ПРОИЗВОДИТЕЛЬНОСТИ:
    При ZeRO-3 параметры sharded между GPU. Без GatheredParameters 
    каждый forward (для каждого токена!) делает all-gather = ОЧЕНЬ медленно.
    
    GatheredParameters собирает все параметры ОДИН раз перед генерацией.
    Это работает и для ZeRO-3 с offload, и без offload.
    
    Источник: grpo_optimizations.md, TRL docs (ds3_gather_for_generation)
    """
    if accelerator is None:
        yield
        return
    
    # Проверяем что используется ZeRO-3
    ds_plugin = getattr(accelerator.state, 'deepspeed_plugin', None)
    if ds_plugin is None:
        yield
        return
    
    zero_stage = getattr(ds_plugin, 'zero_stage', 0)
    if zero_stage != 3:
        yield
        return
    
    # ZeRO-3: ВСЕГДА используем GatheredParameters для генерации
    # Без этого каждый токен = all-gather = зависание
    try:
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
        
        # model уже должен быть unwrapped (передаётся из generate_rollouts)
        params_to_gather = list(model.parameters())
        if not params_to_gather:
            logger.warning("  ⚠️ ds3_gather: нет параметров для сбора")
            yield
            return
        
        # Проверяем есть ли CPU offload (для логирования)
        has_cpu_offload = False
        try:
            ds_config = ds_plugin.deepspeed_config
            offload_param = ds_config.get('zero_optimization', {}).get('offload_param', {})
            param_device = offload_param.get('device', 'none') if isinstance(offload_param, dict) else 'none'
            has_cpu_offload = param_device == 'cpu'
        except Exception:
            pass
        
        offload_str = " (с CPU offload)" if has_cpu_offload else ""
        logger.info(f"  🔄 ds3_gather: собираем {len(params_to_gather)} параметров{offload_str}...")
        
        # modifier_rank=None означает что все ранки могут читать собранные параметры
        with GatheredParameters(params_to_gather, modifier_rank=None):
            logger.info("  ✅ Параметры собраны, начинаем генерацию")
            yield
            logger.info("  ✅ Генерация завершена, освобождаем параметры")
    
    except ImportError:
        logger.warning("  ⚠️ DeepSpeed не найден, продолжаем без ds3_gather (может быть медленно)")
        yield
    except Exception as e:
        logger.warning(f"  ⚠️ ds3_gather ошибка: {e}, продолжаем без оптимизации (может быть медленно)")
        yield

from .experience import Experience
from .legacy_config import GRPOConfig
from .rewards.base import RewardResult


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
        feedbacks: 🔥 SDPO: feedback от reward функций для каждого completion
    """
    prompt: str
    prompt_ids: torch.Tensor
    completions: List[str]
    completion_ids: List[torch.Tensor]
    rewards: torch.Tensor
    is_truncated: List[bool]
    
    # 🔥 SDPO: feedback от reward функций (ошибки, пояснения)
    # Используется для rich environment feedback в self-distillation
    feedbacks: Optional[List[Optional[str]]] = None
    
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
    # ОПТИМИЗАЦИЯ ПАМЯТИ: Используем Liger CrossEntropy если доступен
    # Liger CE более эффективен по памяти чем F.cross_entropy
    # Для больших vocab (Qwen ~152k) это критично
    
    batch_size, seq_len, vocab_size = logits.shape
    
    # Используем chunked cross-entropy для экономии памяти
    # Это разбивает вычисление на части если batch*seq слишком большой
    # chunk_size=4096 хорошо работает для большинства GPU
    nll = chunked_cross_entropy(
        logits,
        output_ids,
        chunk_size=4096,
        ignore_index=-100,
    )
    
    # log_prob = -nll
    log_probs = -nll
    
    return log_probs


def compute_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    accelerator=None,
    chunk_size: Optional[int] = None,  # 🔥 ОПТИМИЗАЦИЯ: auto-detect
) -> torch.Tensor:
    """
    Вычисляет log-вероятности для последовательности.
    
    🔥 ОПТИМИЗАЦИЯ ПАМЯТИ: Обрабатывает sequences по частям (chunked forward pass)
    чтобы не материализовать все logits [batch, seq, vocab] одновременно.
    
    Для batch=8, seq=1200, vocab=152k это экономит ~2.9 GB на logits!
    
    Автоматически определяет режим:
    - no_grad context: chunk_size=1 (максимальная экономия памяти для rollout)
    - with grad: chunk_size=batch_size (нужны все activations для backprop)
    
    ВАЖНО: Эта функция НЕ использует @torch.no_grad(), чтобы градиенты могли проходить
    при использовании в обучении. Используйте torch.no_grad() вручную там где нужно.
    
    Args:
        model: Языковая модель (может быть обернута в DDP)
        sequence_ids: Token IDs [batch, seq_len]
        attention_mask: Маска внимания [batch, seq_len]
        accelerator: Accelerator объект для unwrap модели (опционально)
        chunk_size: Сколько sequences обрабатывать за раз (None=auto-detect)
        
    Returns:
        Log-вероятности [batch, seq_len-1]
    """
    forward_model = model
    batch_size, seq_len = sequence_ids.shape
    device = sequence_ids.device
    
    # 🔥 AUTO-DETECT: если градиенты нужны, не делаем chunking (иначе backprop сломается)
    # Если no_grad context — chunk по 1 для максимальной экономии памяти
    if chunk_size is None:
        if torch.is_grad_enabled():
            # Training mode: нужны все activations для backprop
            chunk_size = batch_size
        else:
            # Inference mode (rollout): chunk по 1 для экономии памяти
            chunk_size = 1
    
    # Position IDs
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    
    # 🔥 CHUNKED FORWARD: обрабатываем по chunk_size sequences за раз
    if batch_size <= chunk_size:
        # Batch достаточно маленький — обрабатываем сразу
        output = forward_model(
            input_ids=sequence_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = output.logits[:, :-1]
        target_ids = sequence_ids[:, 1:]
        log_probs = sequence_log_probs_from_logits(logits, target_ids)
        del output, logits  # Освобождаем память
        return log_probs
    
    # Chunked processing (только для no_grad mode)
    all_log_probs = []
    for start_idx in range(0, batch_size, chunk_size):
        end_idx = min(start_idx + chunk_size, batch_size)
        
        # Forward pass для chunk
        chunk_output = forward_model(
            input_ids=sequence_ids[start_idx:end_idx],
            attention_mask=attention_mask[start_idx:end_idx],
            position_ids=position_ids[start_idx:end_idx],
            use_cache=False,
        )
        
        chunk_logits = chunk_output.logits[:, :-1]
        chunk_targets = sequence_ids[start_idx:end_idx, 1:]
        
        chunk_log_probs = sequence_log_probs_from_logits(chunk_logits, chunk_targets)
        all_log_probs.append(chunk_log_probs)
        
        # 🔥 Освобождаем память после каждого chunk
        del chunk_output, chunk_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return torch.cat(all_log_probs, dim=0)


def _batch_generate_multi_prompt(
    generate_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_batch: List[str],
    config: GRPOConfig,
    generation_config: GenerationConfig,
    device: torch.device,
    autocast_ctx,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    ОПТИМИЗАЦИЯ: Батчевая генерация для нескольких промптов одновременно.
    
    Вместо генерации по одному промпту, объединяем несколько промптов в батч,
    что даёт лучшую утилизацию GPU (особенно при коротких промптах).
    
    Args:
        generate_model: Модель для генерации
        tokenizer: Токенизатор
        prompt_batch: Список промптов для батча
        config: Конфигурация GRPO
        generation_config: Конфигурация генерации
        device: Устройство
        autocast_ctx: Контекст mixed precision
        
    Returns:
        Tuple[List[generated_ids], List[prompt_lengths]]
    """
    batch_size = len(prompt_batch)
    group_size = config.group_size
    
    # Токенизируем все промпты в батч с padding
    prompt_inputs = tokenizer(
        prompt_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_prompt_length,
    ).to(device)
    
    prompt_lengths = [
        (prompt_inputs["attention_mask"][i] == 1).sum().item()
        for i in range(batch_size)
    ]
    
    # Расширяем для group_size: каждый промпт дублируется G раз
    # [prompt0, prompt0, ..., prompt1, prompt1, ...]
    expanded_input_ids = prompt_inputs["input_ids"].repeat_interleave(group_size, dim=0)
    expanded_attention_mask = prompt_inputs["attention_mask"].repeat_interleave(group_size, dim=0)
    
    # Генерация всех completions одним батчем
    with autocast_ctx:
        outputs = generate_model.generate(
            input_ids=expanded_input_ids,
            attention_mask=expanded_attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
        )
    
    # Разделяем результаты обратно по промптам
    all_generated = outputs.sequences  # [batch_size * group_size, seq_len]
    generated_per_prompt = []
    for i in range(batch_size):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        generated_per_prompt.append(all_generated[start_idx:end_idx])
    
    return generated_per_prompt, prompt_lengths


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
    metadata_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Rollout]:
    """
    Генерирует rollout'ы для списка промптов.
    
    ОПТИМИЗАЦИИ:
    - Prefix Grouper: shared KV-cache для G completions (2-3x ускорение)
    - ds3_gather_for_generation: сбор параметров ZeRO-3 перед генерацией (10-100x)
    - Multi-prompt batching: несколько промптов генерируются одним батчем (1.5-2x)
    
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
        metadata_list: Список metadata для каждого промпта (для reward функций)
        
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
        # ВАЖНО: При ZeRO-3 с CPU offload параметры могут быть на CPU
        # Используем accelerator.device если доступен
        if accelerator is not None:
            device = accelerator.device
        else:
            try:
                device = next(unwrapped_model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"🎲 Начинаем генерацию {len(prompts)} rollouts на устройстве {device}")
    
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
    
    # Определяем используется ли ZeRO-3 (для DeepSpeed inference)
    is_zero3 = False
    if accelerator is not None:
        ds_plugin = getattr(accelerator.state, 'deepspeed_plugin', None)
        if ds_plugin is not None:
            zero_stage = getattr(ds_plugin, 'zero_stage', 0)
            is_zero3 = zero_stage == 3
    
    # Mixed precision настройка
    mp = (getattr(config, "mixed_precision", None) or "bf16").lower()
    use_autocast = torch.cuda.is_available() and mp in ("bf16", "fp16")
    if use_autocast:
        amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
        autocast_ctx = torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()
    
    # ВАЖНО: для генерации всегда используем unwrapped модель
    # - DDP: generate() не работает через DDP wrapper
    # - ZeRO-3 + GatheredParameters: параметры собраны, используем напрямую
    generate_model = unwrapped_model
    
    # Определяем использовать ли Prefix Grouper (shared KV-cache)
    # Включаем только для не-ZeRO-3 режимов (ZeRO-3 плохо работает с KV-cache)
    use_prefix_grouper = getattr(config, 'use_prefix_grouper', True) and not is_zero3
    
    # ОПТИМИЗАЦИЯ: Multi-prompt batching
    # Сколько промптов генерировать одним батчем (1 = отключено)
    # Не совместимо с Prefix Grouper (они взаимоисключающие)
    rollout_batch_size = getattr(config, 'rollout_batch_size', 1)
    use_multi_prompt_batch = rollout_batch_size > 1 and not use_prefix_grouper
    
    # ОПТИМИЗАЦИЯ: ds3_gather_for_generation
    # При ZeRO-3 собираем параметры один раз перед всеми генерациями
    # Передаём unwrapped модель — там берутся параметры для gather
    use_ds3_gather = getattr(config, 'ds3_gather_for_generation', True) and is_zero3
    ds3_gather_ctx = ds3_gather_for_generation(unwrapped_model, accelerator) if use_ds3_gather else nullcontext()
    
    with ds3_gather_ctx:
        # ============================================================
        # MULTI-PROMPT BATCHING: генерируем несколько промптов за раз
        # ============================================================
        if use_multi_prompt_batch:
            logger.info(f"  🚀 Multi-prompt batching: rollout_batch_size={rollout_batch_size}")
            
            for batch_start in range(0, len(prompts), rollout_batch_size):
                batch_end = min(batch_start + rollout_batch_size, len(prompts))
                prompt_batch = prompts[batch_start:batch_end]
                ref_batch = reference_answers[batch_start:batch_end]
                
                if batch_start == 0:
                    logger.info(f"  📊 First batch: {len(prompt_batch)} prompts, group_size={config.group_size}")
                    logger.info(f"  📊 Total generations per batch: {len(prompt_batch) * config.group_size}")
                
                # Генерируем батч
                generated_per_prompt, prompt_lengths = _batch_generate_multi_prompt(
                    generate_model=generate_model,
                    tokenizer=tokenizer,
                    prompt_batch=prompt_batch,
                    config=config,
                    generation_config=generation_config,
                    device=device,
                    autocast_ctx=autocast_ctx,
                )
                
                # Обрабатываем результаты для каждого промпта в батче
                for i, (prompt, ref_answer, generated_ids, prompt_length) in enumerate(
                    zip(prompt_batch, ref_batch, generated_per_prompt, prompt_lengths)
                ):
                    prompt_idx = batch_start + i
                    
                    # Декодирование completions
                    completions = tokenizer.batch_decode(
                        generated_ids[:, prompt_length:],
                        skip_special_tokens=True,
                    )
                    
                    # Определяем truncated ответы
                    is_truncated = []
                    for j in range(config.group_size):
                        completion_length = (generated_ids[j, prompt_length:] != tokenizer.pad_token_id).sum().item()
                        is_truncated.append(completion_length >= config.max_new_tokens)
                    
                    # 🔥 SDPO: Вычисляем rewards и собираем feedback
                    rewards = torch.zeros(config.group_size, dtype=torch.float32, device=device)
                    feedbacks: List[Optional[str]] = []
                    # Получаем metadata для текущего промпта
                    prompt_metadata = metadata_list[batch_idx * batch_size + i] if metadata_list else {}
                    for j, completion in enumerate(completions):
                        try:
                            result = reward_fn(
                                completion=completion,
                                reference_answer=ref_answer,
                                reasoning_format=config.reasoning_format,
                                is_truncated=is_truncated[j],
                                metadata=prompt_metadata,
                            )
                            # 🔥 SDPO: обрабатываем RewardResult с feedback
                            if isinstance(result, RewardResult):
                                rewards[j] = float(result.score)
                                feedbacks.append(result.feedback)
                            elif isinstance(result, (int, float)):
                                rewards[j] = float(result)
                                feedbacks.append(None)
                            else:
                                rewards[j] = 0.0
                                feedbacks.append(None)
                        except Exception as e:
                            logger.error(f"Ошибка reward для completion {j}: {e}")
                            rewards[j] = 0.0
                            feedbacks.append(f"Error computing reward: {str(e)}")
                    
                    # Создаём Rollout
                    # Нужно получить prompt_ids для этого промпта
                    prompt_inputs_single = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=config.max_prompt_length,
                    ).to(device)
                    
                    rollout = Rollout(
                        prompt=prompt,
                        prompt_ids=prompt_inputs_single["input_ids"][0],
                        completions=completions,
                        completion_ids=[generated_ids[j, prompt_length:] for j in range(config.group_size)],
                        rewards=rewards,
                        is_truncated=is_truncated,
                        feedbacks=feedbacks,  # 🔥 SDPO
                        metadata={
                            "reference_answer": ref_answer,
                            "prompt_idx": prompt_idx,
                            "prompt_id": (prompt_ids[prompt_idx] if prompt_ids is not None and prompt_idx < len(prompt_ids) else prompt_idx),
                        }
                    )
                    rollouts.append(rollout)
                
                if batch_start == 0:
                    logger.info(f"  ✅ First batch completed")
            
            return rollouts
        
        # ============================================================
        # SINGLE-PROMPT: генерируем по одному промпту (с Prefix Grouper)
        # ============================================================
        for prompt_idx, (prompt, ref_answer) in enumerate(zip(prompts, reference_answers)):
            # Токенизация промпта.
            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_prompt_length,
            ).to(device)
            
            prompt_length = prompt_inputs["input_ids"].size(1)
            
            if prompt_idx == 0:
                logger.info(f"  🔄 Первая генерация: is_zero3={is_zero3}, device={device}, group_size={config.group_size}")
                logger.info(f"  📊 Prompt length: {prompt_length}, max_new_tokens={config.max_new_tokens}")
                logger.info(f"  🚀 Prefix Grouper (shared KV-cache): {'ON' if use_prefix_grouper else 'OFF'}")
                logger.info(f"  🔧 ds3_gather_for_generation: {'ON' if use_ds3_gather else 'OFF'}")
                if is_zero3 and not use_ds3_gather:
                    logger.warning("  ⚠️ ZeRO-3 генерация может быть ОЧЕНЬ медленной! Включите ds3_gather_for_generation")
            
            with autocast_ctx:
                if use_prefix_grouper:
                    # ============================================================
                    # ОПТИМИЗАЦИЯ: Prefix Grouper - shared KV-cache
                    # ============================================================
                    # Идея: prompt прогоняем только ОДИН раз, получаем KV-cache,
                    # затем генерируем G completions с этим кэшем.
                    # Экономия: prompt_length * (G-1) forward passes
                    # ============================================================
                    
                    try:
                        from transformers.cache_utils import DynamicCache
                        
                        # Шаг 1: Прогнать prompt (кроме последнего токена) один раз, получить KV-cache
                        # Оставляем последний токен для начала генерации
                        with torch.no_grad():
                            past_key_values = DynamicCache()
                            
                            # Прогоняем prompt[:-1] чтобы получить кэш
                            # Последний токен будет передан в generate() как начальный
                            if prompt_length > 1:
                                prefix_ids = prompt_inputs["input_ids"][:, :-1]
                                prefix_mask = prompt_inputs["attention_mask"][:, :-1]
                                cached_seq_len = prefix_ids.size(1)
                                
                                # ВАЖНО: передаём cache_position для корректной работы с новым Cache API
                                cache_position = torch.arange(cached_seq_len, device=device)
                                
                                prefix_outputs = generate_model(
                                    input_ids=prefix_ids,
                                    attention_mask=prefix_mask,
                                    past_key_values=past_key_values,
                                    cache_position=cache_position,
                                    use_cache=True,
                                    return_dict=True,
                                )
                                # past_key_values теперь заполнен для prefix
                            else:
                                # Если prompt всего 1 токен, нет смысла в prefix grouper
                                raise ValueError("Prompt too short for prefix grouper")
                        
                        # Шаг 2: Расширить KV-cache для G генераций
                        legacy_cache = past_key_values.to_legacy_cache()
                        
                        expanded_legacy = []
                        for layer_kv in legacy_cache:
                            expanded_key = layer_kv[0].expand(config.group_size, -1, -1, -1).contiguous()
                            expanded_value = layer_kv[1].expand(config.group_size, -1, -1, -1).contiguous()
                            expanded_legacy.append((expanded_key, expanded_value))
                        expanded_legacy = tuple(expanded_legacy)
                        
                        expanded_cache = DynamicCache.from_legacy_cache(expanded_legacy)
                        
                        # Шаг 3: Генерация с shared KV-cache
                        # input_ids = только последний токен prompt'а (G раз)
                        last_token = prompt_inputs["input_ids"][:, -1:].repeat(config.group_size, 1)
                        
                        # attention_mask должен покрывать весь prefix + новые токены
                        gen_attention_mask = torch.ones(
                            config.group_size, cached_seq_len + 1,
                            dtype=prompt_inputs["attention_mask"].dtype,
                            device=device
                        )
                        
                        # ВАЖНО: cache_position для generate() должен начинаться с позиции после кэша
                        gen_cache_position = torch.tensor([cached_seq_len], device=device)
                        
                        outputs = generate_model.generate(
                            input_ids=last_token,
                            attention_mask=gen_attention_mask,
                            past_key_values=expanded_cache,
                            cache_position=gen_cache_position,
                            generation_config=generation_config,
                            return_dict_in_generate=True,
                            output_scores=False,
                        )
                        
                        # Результат: sequences начинается с last_token, затем сгенерированное
                        # Восстанавливаем полную последовательность: prefix + generated
                        prefix_expanded = prompt_inputs["input_ids"][:, :-1].repeat(config.group_size, 1)
                        generated_ids = torch.cat([prefix_expanded, outputs.sequences], dim=1)
                        
                        if prompt_idx == 0:
                            logger.info(f"  ✅ Prefix Grouper: cached {cached_seq_len} tokens, generated {outputs.sequences.size(1)} tokens")
                    
                    except Exception as e:
                        # Fallback к стандартной генерации при ошибке
                        if prompt_idx == 0:
                            logger.warning(f"  ⚠️ Prefix Grouper failed: {e}, using standard generation")
                        
                        input_ids = prompt_inputs["input_ids"].repeat(config.group_size, 1)
                        attention_mask = prompt_inputs["attention_mask"].repeat(config.group_size, 1)
                        
                        outputs = generate_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            generation_config=generation_config,
                            return_dict_in_generate=True,
                            output_scores=False,
                        )
                        generated_ids = outputs.sequences
                
                else:
                    # ============================================================
                    # Стандартная генерация (без Prefix Grouper)
                    # ============================================================
                    input_ids = prompt_inputs["input_ids"].repeat(config.group_size, 1)
                    attention_mask = prompt_inputs["attention_mask"].repeat(config.group_size, 1)
                    
                    outputs = generate_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
                    generated_ids = outputs.sequences
            
            if prompt_idx == 0:
                logger.info(f"  ✅ Первая генерация завершена, tokens: {generated_ids.shape}")
            
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
            
            # 🔥 SDPO: Вычисляем rewards и собираем feedback
            rewards = torch.zeros(config.group_size, dtype=torch.float32, device=device)
            feedbacks: List[Optional[str]] = []
            # Получаем metadata для текущего промпта
            prompt_metadata = metadata_list[prompt_idx] if metadata_list else {}
            for i, completion in enumerate(completions):
                try:
                    result = reward_fn(
                        completion=completion,
                        reference_answer=ref_answer,
                        reasoning_format=config.reasoning_format,
                        is_truncated=is_truncated[i],
                        metadata=prompt_metadata,
                    )
                    # 🔥 SDPO: обрабатываем RewardResult с feedback
                    if isinstance(result, RewardResult):
                        rewards[i] = float(result.score)
                        feedbacks.append(result.feedback)
                    elif isinstance(result, (int, float)):
                        rewards[i] = float(result)
                        feedbacks.append(None)
                    else:
                        logger.warning(
                            f"Reward не число: {type(result)} = {result} для completion: {completion[:100]}..."
                        )
                        rewards[i] = 0.0
                        feedbacks.append(None)
                except Exception as e:
                    logger.error(
                        f"Ошибка при вычислении reward для completion {i}: {e}\n"
                        f"Completion: {completion[:200]}...\n"
                        f"Reference: {ref_answer[:100]}..."
                    )
                    rewards[i] = 0.0
                    feedbacks.append(f"Error computing reward: {str(e)}")
            
            # Создаём Rollout
            rollout = Rollout(
                prompt=prompt,
                prompt_ids=prompt_inputs["input_ids"][0],
                completions=completions,
                completion_ids=[generated_ids[i, prompt_length:] for i in range(config.group_size)],
                rewards=rewards,
                is_truncated=is_truncated,
                feedbacks=feedbacks,  # 🔥 SDPO
                metadata={
                    "reference_answer": ref_answer,
                    "prompt_idx": prompt_idx,
                    "prompt_id": (prompt_ids[prompt_idx] if prompt_ids is not None and prompt_idx < len(prompt_ids) else prompt_idx),
                }
            )
            rollouts.append(rollout)
    
    return rollouts


@torch.no_grad()
def generate_rollouts_vllm(
    *,
    vllm_engine,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    reference_answers: List[str],
    reward_fn: Callable,
    config: GRPOConfig,
    prompt_ids: Optional[List[int]] = None,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
) -> List[Rollout]:
    """
    Генерация rollouts через vLLM.

    Мы возвращаем тот же формат Rollout, что и generate_rollouts (HF).
    old_logprobs/ref_logprobs считаются позже на training-модели (teacher-forcing),
    поэтому здесь нужны только токены и текст completions.
    """
    if len(prompts) != len(reference_answers):
        raise ValueError("prompts и reference_answers должны быть одинаковой длины")

    eos_id = tokenizer.eos_token_id
    stop_ids = [int(eos_id)] if eos_id is not None else None
    sampling_params = vllm_engine.make_sampling_params(
        n=int(config.group_size),
        temperature=float(config.temperature),
        top_p=float(config.top_p),
        max_tokens=int(config.max_new_tokens),
        stop_token_ids=stop_ids,
    )

    # vLLM batched generation: один вызов на batch prompts
    outputs = vllm_engine.generate(prompts, sampling_params)
    if len(outputs) != len(prompts):
        raise RuntimeError(f"vLLM вернул {len(outputs)} outputs на {len(prompts)} prompts")

    rollouts: List[Rollout] = []
    for prompt_idx, (prompt, ref_answer, out) in enumerate(zip(prompts, reference_answers, outputs)):
        # prompt token ids (для дальнейшего склеивания)
        prompt_tok = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids_tensor = prompt_tok["input_ids"][0]
        prompt_len = int(prompt_ids_tensor.size(0))

        completions: List[str] = []
        completion_ids: List[torch.Tensor] = []
        is_truncated: List[bool] = []

        # out.outputs: list of candidates (n == group_size)
        cand_list = getattr(out, "outputs", None)
        if cand_list is None:
            raise RuntimeError("vLLM output missing .outputs")

        for cand in cand_list:
            text = getattr(cand, "text", "")
            tok_ids = getattr(cand, "token_ids", None)
            if tok_ids is None:
                # fallback: tokenize text (менее точно при несовпадении токенизаторов)
                tok_ids = tokenizer(text, add_special_tokens=False).input_ids
            completions.append(text)
            completion_ids.append(torch.tensor(tok_ids, dtype=torch.long))
            finish_reason = getattr(cand, "finish_reason", None)
            is_truncated.append(finish_reason == "length")

        # 🔥 SDPO: Rewards и feedback
        rewards = torch.zeros(len(completions), dtype=torch.float)
        feedbacks: List[Optional[str]] = []
        # Получаем metadata для текущего промпта
        prompt_metadata = metadata_list[prompt_idx] if metadata_list else {}
        for i, comp in enumerate(completions):
            try:
                result = reward_fn(
                    completion=comp,
                    reference_answer=ref_answer,
                    reasoning_format=config.reasoning_format,
                    is_truncated=is_truncated[i],
                    metadata=prompt_metadata,
                )
                # 🔥 SDPO: обрабатываем RewardResult с feedback
                if isinstance(result, RewardResult):
                    rewards[i] = float(result.score)
                    feedbacks.append(result.feedback)
                elif isinstance(result, (int, float)):
                    rewards[i] = float(result)
                    feedbacks.append(None)
                else:
                    rewards[i] = 0.0
                    feedbacks.append(None)
            except Exception as e:
                logger.error(f"Ошибка reward_fn: {e}")
                rewards[i] = 0.0
                feedbacks.append(f"Error computing reward: {str(e)}")

        rollouts.append(
            Rollout(
                prompt=prompt,
                prompt_ids=prompt_ids_tensor,
                completions=completions,
                completion_ids=completion_ids,
                rewards=rewards,
                is_truncated=is_truncated,
                feedbacks=feedbacks,  # 🔥 SDPO
                metadata={
                    "reference_answer": ref_answer,
                    "prompt_idx": prompt_idx,
                    "prompt_id": (prompt_ids[prompt_idx] if prompt_ids is not None and prompt_idx < len(prompt_ids) else prompt_idx),
                },
            )
        )

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
        # ВАЖНО: при ZeRO-3 параметры могут быть sharded/offloaded, device параметров не надёжен
        if accelerator is not None:
            device = accelerator.device
        else:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    experiences = []
    prompt_length = rollout.prompt_ids.size(0)
    
    # Вычисляем advantages для группы
    from .loss import compute_advantages
    
    advantages = compute_advantages(
        rollout.rewards,
        use_std_normalization=config.use_std_normalization,
    )
    
    # ============================================================
    # ОПТИМИЗАЦИЯ: batched logprobs для всей группы (G completions)
    # Вместо G отдельных forward pass делаем 1 forward на батч.
    # ============================================================
    seq_tensors: List[torch.Tensor] = []
    seq_lens: List[int] = []
    cleaned_completion_ids: List[torch.Tensor] = []
    for i in range(len(rollout.completions)):
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
        cleaned_completion_ids.append(completion_ids)

        sequence_ids = torch.cat([rollout.prompt_ids.to(device), completion_ids.to(device)])
        seq_tensors.append(sequence_ids)
        seq_lens.append(int(sequence_ids.numel()))
    
    # padding справа: prompt всегда в начале, проще строить action_mask
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    max_len = max(seq_lens) if seq_lens else 0
    batch_size = len(seq_tensors)
    if batch_size == 0 or max_len < 2:
        return []

    batch_ids = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(seq_tensors):
        L = int(seq.numel())
        batch_ids[i, :L] = seq
        attention_mask[i, :L] = 1

    # Log probs текущей политики (old_logprobs) — без градиентов
    with torch.no_grad():
        batch_log_probs = compute_log_probs(
            model,
            batch_ids,
            attention_mask,
            accelerator=accelerator,
        )  # [B, max_len-1]

    # Log probs референсной модели (для KL) — тоже батчево и без градиентов
    batch_log_probs_ref = None
    if reference_model is not None and config.kl_weight > 0:
        with torch.no_grad():
            batch_log_probs_ref = compute_log_probs(
                reference_model,
                batch_ids,
                attention_mask,
                accelerator=accelerator,
            )

    # Теперь собираем Experience по каждому completion
    for i in range(batch_size):
        sequence_ids = seq_tensors[i]
        L = seq_lens[i]
        # attention_mask для конкретного семпла (без паддинга)
        attn = torch.ones(L, dtype=torch.long, device=device)
        
        # Action mask (только для completion токенов, включая EOS)
        # ВАЖНО: EOS токены НЕ маскируются - это действия модели
        action_mask = torch.zeros(L - 1, dtype=torch.bool, device=device)
        action_mask[prompt_length - 1 :] = True

        log_probs = batch_log_probs[i, : L - 1]
        log_probs_ref = batch_log_probs_ref[i, : L - 1] if batch_log_probs_ref is not None else None
        
        # 🔥 SDPO: передаём prompt для teacher reprompting
        prompt_id = None
        if rollout.metadata:
            prompt_id = rollout.metadata.get('prompt_id')
        
        exp = Experience(
            sequences=sequence_ids,
            prompt_length=prompt_length,
            action_log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            returns=rollout.rewards[i].unsqueeze(0),
            advantages=advantages[i].unsqueeze(0),
            attention_mask=attn,
            action_mask=action_mask,
            completion_text=rollout.completions[i],
            prompts=[rollout.prompt],  # 🔥 SDPO: оригинальный промпт
            prompt_ids=[prompt_id] if prompt_id is not None else None,  # 🔥 SDPO
        )
        experiences.append(exp)
    
    return experiences
