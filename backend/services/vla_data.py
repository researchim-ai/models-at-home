"""Strict JSONL demonstrations: one observation maps to one explicit action chunk."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'{name}: expected {size} numbers')
    if any(isinstance(item, bool) or not isinstance(item, (float, int)) for item in value):
        raise ValueError(f'{name}: expected numeric values')
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f'{name}: NaN/Infinity or float32 overflow')
    return array


class Demonstrations:
    def __init__(self, filename, config):
        self.rows, self.config = [], config
        base = Path(filename).parent
        with Path(filename).open(encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    image = row.get('image')
                    if not isinstance(image, str) or not image:
                        raise ValueError('image must be a local image path')
                    path = Path(image).expanduser()
                    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
                    if not path.is_file():
                        raise ValueError(f'Image not found: {path}')
                    instruction = row.get('instruction')
                    if not isinstance(instruction, str) or not instruction.strip():
                        raise ValueError('instruction must be nonempty text')
                    if len(instruction.encode('utf-8')) > config['max_instruction_bytes']:
                        raise ValueError('instruction exceeds max_instruction_bytes; no silent truncation')
                    state = _vector(row.get('state', []), config['state_dim'], 'state')
                    actions = row.get('actions')
                    if not isinstance(actions, list) or not 1 <= len(actions) <= config['action_horizon']:
                        raise ValueError('actions must contain 1..action_horizon explicit timesteps')
                    if config['action_type'] == 'continuous':
                        actions = np.stack([_vector(value, config['action_dim'], 'action') for value in actions])
                    elif config['action_type'] == 'hybrid':
                        combined = []
                        for action in actions:
                            if not isinstance(action, dict):
                                raise ValueError('Hybrid actions need continuous/discrete objects')
                            continuous = _vector(action.get('continuous'), config['action_dim'], 'continuous action')
                            discrete = action.get('discrete')
                            classes = config['discrete_classes']
                            if not isinstance(discrete, list) or len(discrete) != len(classes) or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n for v, n in zip(discrete, classes)):
                                raise ValueError('Hybrid discrete class IDs do not match discrete_classes')
                            combined.append(np.concatenate([continuous, np.asarray(discrete, dtype=np.float32)]))
                        actions = np.stack(combined)
                    else:
                        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < config['action_dim'] for value in actions):
                            raise ValueError('Discrete actions must be integer class IDs within action_dim')
                        actions = np.asarray(actions, dtype=np.int64)
                    self.rows.append({'image': str(path), 'instruction': instruction, 'state': state, 'actions': actions, 'raw_actions': row['actions'], 'episode_id': row.get('episode_id')})
                except Exception as exc:
                    raise ValueError(f'{filename}, line {number}: {exc}') from exc
        if not self.rows:
            raise ValueError('Dataset contains no demonstrations')

    def statistics(self):
        # Welford updates avoid cancellation for large physical offsets.
        def summary(values, dim):
            count, mean, m2 = 0, np.zeros(dim, dtype=np.float64), np.zeros(dim, dtype=np.float64)
            for array in values:
                for row in np.asarray(array).reshape(-1, dim) if dim else []:
                    count += 1
                    delta = row - mean
                    mean += delta / count
                    m2 += delta * (row - mean)
            std = np.sqrt(m2 / max(count, 1))
            std[std < 1e-6] = 1.0
            return {'mean': mean.tolist(), 'std': std.tolist()}
        return {'state': summary((row['state'][None] for row in self.rows), self.config['state_dim']),
                'action': summary((row['actions'][:, :self.config['action_dim']] for row in self.rows), self.config['action_dim'])
                if self.config['action_type'] != 'discrete' else None}

    def __len__(self):
        return len(self.rows)


def image_tensor(path, size):
    import torch
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB').resize((size, size))
        array = np.asarray(image, dtype=np.float32).copy() / 255.
    return torch.from_numpy(array).permute(2, 0, 1)


def batch(rows, config, stats, processor=None):
    import torch
    states = torch.tensor(np.stack([row['state'] for row in rows]))
    states = (states - states.new_tensor(stats['state']['mean'])) / states.new_tensor(stats['state']['std'])
    width = config['action_dim'] + (len(config.get('discrete_classes', [])) if config['action_type'] == 'hybrid' else 0)
    shape = (len(rows), config['action_horizon'], width) if config['action_type'] != 'discrete' else (len(rows), config['action_horizon'])
    targets = torch.zeros(shape, dtype=torch.float32 if config['action_type'] != 'discrete' else torch.long)
    mask = torch.zeros(len(rows), config['action_horizon'], dtype=torch.bool)
    for index, row in enumerate(rows):
        count = len(row['actions'])
        actions = torch.from_numpy(row['actions'])
        if stats['action'] is not None:
            actions = actions.clone()
            actions[:, :config['action_dim']] = (actions[:, :config['action_dim']] - actions.new_tensor(stats['action']['mean'])) / actions.new_tensor(stats['action']['std'])
        targets[index, :count] = actions
        mask[index, :count] = True
    if processor is None:
        inputs = {'images': torch.stack([image_tensor(row['image'], config['image_size']) for row in rows]),
                  'instructions': [row['instruction'] for row in rows]}
    else:
        from PIL import Image, ImageOps
        images, texts = [], []
        for row in rows:
            with Image.open(row['image']) as image:
                images.append(ImageOps.exif_transpose(image).convert('RGB'))
            texts.append(processor.apply_chat_template([{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': row['instruction']}]}], tokenize=False, add_generation_prompt=True))
        inputs = processor(text=texts, images=[[image] for image in images], return_tensors='pt', padding=True,
                           return_token_type_ids=False)
        if inputs['input_ids'].shape[1] > config['seq_len']:
            raise ValueError('Expanded VLA image/instruction exceeds seq_len; reduce pixels or increase context')
    return inputs, states, targets, mask
