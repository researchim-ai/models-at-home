"""Optional ML contract tests; skipped in API-only CI environments."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('homellm.training.vlm_data')
from backend.services.vlm_runtime import runtime_types


class Tokenizer:
    pad_token_id = eos_token_id = 2
    unk_token_id = 0
    image_token = '<image>'
    padding_side = 'left'

    def convert_tokens_to_ids(self, text):
        return 7 if text == '<image>' else 0

    def convert_ids_to_tokens(self, value):
        return str(value)


class Processor:
    tokenizer = Tokenizer()
    words = {'user': 3, 'assistant': 4, 'question': 5, 'red': 8, 'square': 9, 'eos': 2, 'unknown': 0}

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        text = ' '.join(message['role'] + ' ' + ' '.join('<image>' if item['type'] == 'image' else item['text']
                         for item in message['content']) for message in messages)
        return text + (' assistant' if add_generation_prompt else '')

    def __call__(self, text, **kwargs):
        rows = []
        for value in text:
            ids = []
            for word in value.split():
                ids.extend([7, 7, 7] if word == '<image>' else [self.words.get(word, 0)])
            rows.append(ids)
        length = max(map(len, rows))
        return {'input_ids': torch.tensor([row + [2] * (length - len(row)) for row in rows]),
                'attention_mask': torch.tensor([[1] * len(row) + [0] * (length - len(row)) for row in rows]),
                'pixel_values': torch.ones(len(rows), 3, 4, 4), 'image_grid_thw': torch.ones(len(rows), 3)}


@pytest.fixture
def example(tmp_path):
    from PIL import Image
    image = tmp_path / 'tiny.png'
    Image.new('RGB', (4, 4)).save(image)
    return {'image_refs': [str(image)], 'messages': [
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'question'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'red eos'}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'question'}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'square eos'}]},
    ]}


def test_all_assistant_turns_are_supervised_and_eos_is_preserved(example):
    _, Collator = runtime_types()
    batch = Collator(Processor(), max_length=100)([example])
    labels = batch['labels'][0]
    assert labels[labels != -100].tolist() == [8, 2, 9, 2]
    assert 'pixel_values' in batch and 'image_grid_thw' in batch


def test_all_token_loss_preserves_unknown_tokens(example):
    _, Collator = runtime_types()
    example['messages'][0]['content'][-1]['text'] = 'unknown'
    labels = Collator(Processor(), assistant_only_loss=False)([example])['labels'][0]
    assert 0 in labels and 7 not in labels


def test_expanded_image_tokens_cannot_be_truncated_silently(example):
    _, Collator = runtime_types()
    with pytest.raises(ValueError, match='exceeds seq_len'):
        Collator(Processor(), max_length=3)([example])


def test_missing_image_is_an_error_instead_of_changing_batch(example):
    _, Collator = runtime_types()
    example['image_refs'].append('/nonexistent/second-image.png')
    with pytest.raises(ValueError, match='failed to load'):
        Collator(Processor())([example])


def test_no_answer_is_not_reported_as_successful_training(example):
    _, Collator = runtime_types()
    example['messages'] = example['messages'][:1]
    with pytest.raises(ValueError, match='no next-token'):
        Collator(Processor())([example])


@pytest.fixture
def preflight_setup(example, workspace, monkeypatch):
    import json
    from types import SimpleNamespace
    from transformers import AutoConfig, AutoModelForImageTextToText
    from homellm.training import vlm_common
    from backend.routes.vlm import VLMTrainingRequest, _normalize_and_validate_vlm_config
    data = workspace / 'datasets' / 'images.jsonl'
    record = {'images': example['image_refs'], 'messages': example['messages']}
    data.write_text(json.dumps(record) + '\n')
    monkeypatch.setattr(AutoConfig, 'from_pretrained', lambda *a, **k: SimpleNamespace(model_type='llava'))
    monkeypatch.setattr(AutoModelForImageTextToText, '_model_mapping', {SimpleNamespace})
    def forbid_weights(*args, **kwargs):
        raise AssertionError('Preflight must never load model weights')
    monkeypatch.setattr(AutoModelForImageTextToText, 'from_pretrained', forbid_weights)
    monkeypatch.setattr(vlm_common, 'load_vlm_processor', lambda *a, **k: Processor())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(name='Test GPU'))
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (8 * 2**30, 16 * 2**30))
    config = VLMTrainingRequest(model_path='local/test', dataset_path=str(data), use_lora=False).model_dump()
    _normalize_and_validate_vlm_config(config)
    return config, data, record


def test_preflight_uses_training_masks_without_loading_weights(preflight_setup):
    from backend.services.vlm_preflight import inspect
    config, _, _ = preflight_setup
    report = inspect(config)
    assert report['ready'], report
    assert report['sample']['supervised_tokens'] == 4
    assert sum(token['trainable'] for token in report['sample']['tokens']) == 4
    assert report['sample']['images'][0]['src'].startswith('data:image/png;base64,')


@pytest.mark.parametrize('problem', ['missing_image', 'invalid_json', 'no_answer', 'context'])
def test_preflight_rejects_broken_data(preflight_setup, problem):
    import json
    from backend.services.vlm_preflight import inspect
    config, path, record = preflight_setup
    if problem == 'missing_image':
        record['images'] = ['/nonexistent.png']
    elif problem == 'no_answer':
        record['messages'] = record['messages'][:1]
    elif problem == 'context':
        config['seq_len'] = 3
    path.write_text('not JSON' if problem == 'invalid_json' else json.dumps(record))
    report = inspect(config)
    assert report['ready'] is False
    assert any(item['status'] == 'error' and item['name'] == 'Данные train' for item in report['checks'])


def test_preflight_states_sampling_limit(preflight_setup):
    import json
    from backend.services.vlm_preflight import inspect
    config, path, record = preflight_setup
    path.write_text((json.dumps(record) + '\n') * 200 + 'uninspected row\n')
    report = inspect(config)
    assert report['ready']
    assert report['datasets'][0]['scanned_rows'] == 200
    assert report['datasets'][0]['complete'] is False
    assert any(item['status'] == 'warning' for item in report['checks'])


def test_preflight_cuda_error_still_shows_a_useful_data_preview(preflight_setup, monkeypatch):
    from backend.services.vlm_preflight import inspect
    config, _, _ = preflight_setup
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    report = inspect(config)
    assert not report['ready']
    assert report['sample']['supervised_tokens'] == 4
    assert any(item['name'] == 'GPU' and item['status'] == 'error' for item in report['checks'])


@pytest.mark.parametrize('sharded', [False, True])
def test_preflight_rejects_missing_local_weights(preflight_setup, tmp_path, sharded):
    from backend.services.vlm_preflight import inspect
    config, _, _ = preflight_setup
    model = tmp_path / 'empty-model'
    model.mkdir()
    config['base_model_path'] = str(model)
    if sharded:
        (model / 'model.safetensors.index.json').write_text('{"weight_map":{"layer":"missing-shard.safetensors"}}')
    report = inspect(config)
    assert not report['ready']
    assert any(item['name'] == 'Файлы весов' and item['status'] == 'error' for item in report['checks'])


def test_training_refuses_invalid_rows_beyond_the_preflight_sample(tmp_path):
    import json
    path = tmp_path / 'data.jsonl'
    path.write_text((json.dumps({'image': 'red.png', 'question': 'color?', 'answer': 'red'}) + '\n') * 201 + 'invalid json\n')
    Dataset, _ = runtime_types()
    with pytest.raises(ValueError, match='line 202'):
        Dataset(path)
