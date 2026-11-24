"""Multimodal adapters for homellm native RL algorithms; no TRL trainer."""
from __future__ import annotations

import copy
import re


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    return '\n'.join(str(item.get('content', '')) for item in completion)


def reward_function(kind, observer=None):
    def score(completions, answer, **kwargs):
        def normalize(text):
            return re.sub(r'\s+', ' ', str(text)).strip().casefold()
        results = []
        for completion, target in zip(completions, answer, strict=True):
            actual, expected = normalize(completion_text(completion)), normalize(target)
            if not expected:
                raise ValueError('GRPO requires a nonempty reference answer')
            results.append(float(actual == expected if kind == 'exact_match' else expected in actual))
        if observer:
            observer(completions, results)
        return results
    score.__name__ = kind
    return score


def prompt_and_answer(example):
    """Remove only the last assistant turn; never expose its answer in rollout input."""
    from homellm.training.vlm_data import _assistant_text_from_content
    messages = copy.deepcopy(example['messages'])
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('GRPO needs a final assistant reference answer')
    answer = _assistant_text_from_content(messages.pop()['content']).strip()
    if not answer or not messages or messages[-1]['role'] != 'user':
        raise ValueError('GRPO requires a user prompt followed by a nonempty assistant answer')
    # The processor receives images separately. Do not retain local URLs.
    for message in messages:
        if isinstance(message['content'], list):
            message['content'] = [({'type': 'image'} if item.get('type') == 'image' else item)
                                  for item in message['content']]
    return messages, answer


def run(config, metrics):
    try:
        _run(config, metrics)
    except Exception as exc:
        import traceback
        metrics.update(status='error', error=str(exc) + '\n' + traceback.format_exc())
        raise


def native_config(config):
    """Use the exact configuration contract consumed by homellm's RL losses."""
    from homellm.training.rl.legacy_config import GRPOConfig
    algorithm = {'dr_grpo': 'drgrpo'}.get(config['grpo_algorithm'], config['grpo_algorithm'])
    return GRPOConfig(algorithm=algorithm, group_size=config['grpo_group_size'],
        max_new_tokens=config['grpo_max_new_tokens'], temperature=config['grpo_temperature'],
        clip_eps_low=config['grpo_clip_eps_low'], clip_eps_high=config['grpo_clip_eps_high'],
        kl_weight=config['grpo_kl_weight'], max_refill_rounds=config.get('grpo_max_refill_rounds', 3),
        overlong_buffer=config.get('grpo_overlong_buffer', 0),
        overlong_penalty=config.get('grpo_overlong_penalty', -1.0),
        sdpo_success_threshold=config.get('sdpo_success_threshold', .5),
        sdpo_alpha=config.get('sdpo_alpha', .5), sdpo_loss_weight=config.get('sdpo_loss_weight', 1.0),
        liger_fused_grpo=False, use_liger=False)


def sequence_inputs(prompt_inputs, sequence):
    """Preserve processor vision tensors while appending generated text tokens."""
    import torch
    result = dict(prompt_inputs)
    extra = sequence.shape[-1] - prompt_inputs['input_ids'].shape[-1]
    if extra < 0:
        raise ValueError('Completion sequence is shorter than the image prompt')
    result['input_ids'] = sequence
    prompt_mask = prompt_inputs.get('attention_mask', torch.ones_like(prompt_inputs['input_ids']))
    result['attention_mask'] = torch.cat([prompt_mask, prompt_mask.new_ones((1, extra))], dim=1)
    for key in ('token_type_ids', 'mm_token_type_ids'):
        if key in result:
            result[key] = torch.cat([result[key], result[key].new_zeros((1, extra))], dim=1)
    if 'cross_attention_mask' in result:
        mask = result['cross_attention_mask']
        result['cross_attention_mask'] = torch.cat([mask, mask[:, -1:].expand(-1, extra, *mask.shape[2:])], dim=1)
    if 'position_ids' in result:
        raise ValueError('Explicit processor position_ids require a model-specific VLM adapter')
    return result


def selected_log_probs(model, inputs, completion_length):
    """Native-policy probabilities of the same sampled actions, with their pixels."""
    import torch
    output = model(**inputs, use_cache=False)
    logits = output.logits[:, -(completion_length + 1):-1].float()
    targets = inputs['input_ids'][:, -completion_length:]
    return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def teacher_messages(prompt, successful_solution):
    messages = copy.deepcopy(prompt)
    suffix = '\nHere is a successful solution for reference:\n' + successful_solution + '\nNow solve the original problem.'
    content = messages[-1]['content']
    if isinstance(content, list):
        content.append({'type': 'text', 'text': suffix})
    else:
        messages[-1]['content'] = str(content) + suffix
    return messages


def _run(config, metrics):
    from backend.services.vlm_native_trainer import NativeVLMTrainer
    NativeVLMTrainer(config, metrics).train()
