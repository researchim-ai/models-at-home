"""Episode windows: frame[t], action[t] -> frame[t+1], never across terminals."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps


def vector(value, size, label):
    if not isinstance(value, list) or len(value) != size or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError(f'{label}: expected {size} numeric coordinates')
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f'{label}: values must be finite float32')
    return result


def action(value, c):
    def category(x, n):
        if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < n:
            raise ValueError(f'Action category must be an integer in [0, {n})')
        return np.eye(n, dtype=np.float32)[x]
    kind = c['action_type']
    if kind == 'discrete':
        return category(value, c['action_dim'])
    if kind == 'continuous':
        return vector(value, c['action_dim'], 'action')
    if not isinstance(value, dict) or not isinstance(value.get('discrete'), list) or len(value['discrete']) != len(c['discrete_classes']):
        raise ValueError('Hybrid action needs continuous vector and one discrete id per head')
    return np.concatenate([vector(value.get('continuous'), c['action_dim'], 'action'),
                           *[category(x, n) for x, n in zip(value['discrete'], c['discrete_classes'])]])


def action_width(c):
    return c['action_dim'] + (sum(c['discrete_classes']) if c['action_type'] == 'hybrid' else 0)


def image(path, size):
    with Image.open(path) as source:
        value = ImageOps.exif_transpose(source).convert('RGB').resize((size, size))
        return torch.from_numpy(np.array(value, dtype=np.float32).copy()).permute(2, 0, 1) / 255


class Episodes:
    def __init__(self, path, c):
        self.path, self.config = Path(path), c
        self.episodes, self.windows = [], []
        digest = hashlib.sha256(self.path.read_bytes())
        self.frame_paths, self.ids = set(), set()
        for line, text in enumerate(self.path.read_text().splitlines(), 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
                frames, actions = row['frames'], row['actions']
                if not isinstance(actions, list) or not actions or not isinstance(frames, list) or len(frames) != len(actions) + 1:
                    raise ValueError('frames must contain T+1 paths and actions T items')
                eid = row.get('episode_id')
                if not isinstance(eid, str) or not eid.strip() or eid in self.ids:
                    raise ValueError('Every episode needs a unique nonempty episode_id')
                self.ids.add(eid)
                paths = []
                for name in frames:
                    p = Path(name).expanduser()
                    p = (p if p.is_absolute() else self.path.parent / p).resolve()
                    if not p.is_file():
                        raise ValueError(f'Missing frame: {p}')
                    # Decode every frame, and fingerprint content for reliable resume.
                    image(p, c['image_size'])
                    digest.update(p.read_bytes())
                    paths.append(str(p)); self.frame_paths.add(str(p))
                a = np.stack([action(x, c) for x in actions])
                states = row.get('states')
                if c['state_dim']:
                    if not isinstance(states, list) or len(states) != len(frames):
                        raise ValueError('states must have T+1 vectors')
                    states = np.stack([vector(x, c['state_dim'], 'state') for x in states])
                elif states is not None:
                    raise ValueError('Set state_dim to match supplied states')
                rewards = row.get('rewards')
                if rewards is not None:
                    rewards = vector(rewards, len(actions), 'rewards')
                dones = row.get('dones')
                if dones is not None and (not isinstance(dones, list) or len(dones) != len(actions) or any(type(x) is not bool for x in dones)):
                    raise ValueError('dones must have T boolean values')
                ep = dict(frames=paths, actions=a, states=states, rewards=rewards, dones=dones)
                index = len(self.episodes)
                self.episodes.append(ep)
                for start in range(len(actions) - c['sequence_length'] + 1):
                    # A terminal transition may be the final target, never an input for a later step.
                    if dones is None or not any(dones[start:start + c['sequence_length'] - 1]):
                        self.windows.append((index, start))
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise ValueError(f'{self.path}:{line}: {exc}') from exc
        if not self.windows:
            raise ValueError('No valid sequence windows; supply longer episodes or reduce sequence_length')
        self.fingerprint = digest.hexdigest()

    def statistics(self):
        c = self.config
        def stats(values):
            a = np.concatenate(values).astype(np.float64)
            mean, std = a.mean(0), a.std(0)
            return {'mean': mean.tolist(), 'std': np.where(std < 1e-6, 1, std).tolist()}
        a = stats([ep['actions'] for ep in self.episodes])
        width = 0 if c['action_type'] == 'discrete' else c['action_dim']
        a['mean'][width:] = [0.] * (len(a['mean']) - width)
        a['std'][width:] = [1.] * (len(a['std']) - width)
        return {'actions': a, 'states': stats([ep['states'] for ep in self.episodes]) if c['state_dim'] else None}

    def batch(self, indices, stats, device):
        c, rows = self.config, []
        length = c['sequence_length']
        for index in indices:
            ep_id, start = self.windows[index]
            ep = self.episodes[ep_id]; end = start + length
            a = (ep['actions'][start:end] - np.array(stats['actions']['mean'])) / np.array(stats['actions']['std'])
            states = np.zeros((length + 1, 0), dtype=np.float32)
            if c['state_dim']:
                states = (ep['states'][start:end+1] - np.array(stats['states']['mean'])) / np.array(stats['states']['std'])
            rows.append(dict(images=torch.stack([image(p, c['image_size']) for p in ep['frames'][start:end+1]]),
                actions=torch.tensor(a, dtype=torch.float32), states=torch.tensor(states, dtype=torch.float32),
                rewards=torch.tensor(ep['rewards'][start:end] if ep['rewards'] is not None else np.zeros(length), dtype=torch.float32),
                dones=torch.tensor(ep['dones'][start:end] if ep['dones'] is not None else np.zeros(length), dtype=torch.float32),
                reward_mask=torch.full((length,), ep['rewards'] is not None), done_mask=torch.full((length,), ep['dones'] is not None)))
        return {key: torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}


def datasets(c):
    train = Episodes(c['dataset_path'], c)
    val = Episodes(c['val_data_path'], c) if c.get('val_data_path') else None
    if val and (train.ids & val.ids or train.frame_paths & val.frame_paths):
        raise ValueError('Training and validation must use independent episode IDs and frame paths')
    return train, val
