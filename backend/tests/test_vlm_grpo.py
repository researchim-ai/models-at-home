"""Policy dataset/reward contracts, independent of CUDA and optional TRL."""
import copy

import pytest

from backend.services.vlm_grpo import reward_function, prompt_and_answer


def test_reference_answer_never_enters_rollout_prompt():
    pytest.importorskip('homellm.training.vlm_data')
    example = {'messages': [
        {'role': 'user', 'content': [{'type': 'image', 'image': '/private/path.png'}, {'type': 'text', 'text': 'color?'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'secret answer'}]}]}
    original = copy.deepcopy(example)
    prompt, answer = prompt_and_answer(example)
    assert answer == 'secret answer'
    assert len(prompt) == 1 and prompt[0]['content'][0] == {'type': 'image'}
    assert 'secret answer' not in str(prompt)
    assert example == original


@pytest.mark.parametrize('messages', [[], [{'role': 'user', 'content': 'question'}],
    [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': ''}]])
def test_missing_reward_reference_is_rejected(messages):
    pytest.importorskip('homellm.training.vlm_data')
    with pytest.raises(ValueError, match='GRPO'):
        prompt_and_answer({'messages': messages})


def test_exact_reward_checks_whole_answer_and_normalizes_case_whitespace():
    reward = reward_function('exact_match')
    assert reward(['  RED \n square ', 'not red square', [{'content': 'red square'}]],
                  ['red square'] * 3) == [1., 0., 1.]


def test_contains_reward_is_explicit_and_empty_answers_are_rejected():
    reward = reward_function('contains')
    assert reward(['a red square', 'green'], ['red', 'red']) == [1., 0.]
    with pytest.raises(ValueError, match='nonempty'):
        reward(['anything'], [''])
    with pytest.raises(ValueError):
        reward(['one', 'two'], ['one'])


def test_rewards_can_be_observed_without_changing_their_values():
    seen = []
    score = reward_function('contains', lambda completions, rewards: seen.append((completions, rewards)))
    assert score(['red', 'blue'], ['red', 'red']) == [1., 0.]
    assert seen == [(['red', 'blue'], [1., 0.])]
