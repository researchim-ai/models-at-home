"""Read worker-owned metrics, including historical points after a UI restart."""
from __future__ import annotations

import json
import math
from pathlib import Path


def read(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        # Legacy workers may be in the middle of a non-atomic write.
        return None


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def snapshot(raw: dict) -> dict:
    data = {key: value for key, value in raw.items() if not key.endswith('_history')}
    aliases = {'step': 'current_step', 'loss': 'current_loss', 'learning_rate': 'current_lr',
               'reward': 'current_reward', 'kl_divergence': 'current_kl', 'val_loss': 'current_val_loss',
               'timestamp': 'elapsed_seconds'}
    for key, alias in aliases.items():
        value = raw.get(key, raw.get(alias))
        data[key] = _number(value)
    data['step'] = max(0, int(data['step'] or 0))
    data['timestamp'] = data['timestamp'] or 0
    gpus = []
    gpu_stats = raw.get('gpu_stats')
    for gpu in gpu_stats if isinstance(gpu_stats, list) else []:
        if not isinstance(gpu, dict):
            continue
        used, total = _number(gpu.get('memory_used_gb')), _number(gpu.get('memory_total_gb'))
        if used is None or total is None or used < 0 or total <= 0:
            continue
        gpus.append({**gpu, 'memory_percent': min(100, 100 * used / total),
                     'utilization': _number(gpu.get('utilization'))})
    data['gpu_stats'] = gpus
    # Strict JSON transport must never fail on NaN/Infinity in optional fields.
    return _finite(data)


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def history(raw: dict, limit: int = 500) -> list[dict]:
    steps = raw.get('steps_history')
    if not isinstance(steps, list):
        return []
    points = {}
    columns = {'loss': 'loss_history', 'learning_rate': 'lr_history', 'reward': 'reward_history',
               'kl_divergence': 'kl_history', 'grad_norm': 'grad_norm_history', 'val_loss': 'val_loss_history'}
    for index in range(max(0, len(steps) - limit), len(steps)):
        step = _number(steps[index])
        if step is None or step < 0:
            continue
        point = {'step': int(step), 'timestamp': 0, 'loss': None, 'learning_rate': None}
        for key, column in columns.items():
            values = raw.get(column)
            # Short eval histories cannot be aligned to training steps by index.
            if isinstance(values, list) and len(values) == len(steps):
                point[key] = _number(values[index])
        points[int(step)] = point
    eval_steps, eval_losses = raw.get('val_steps_history'), raw.get('val_loss_history')
    if isinstance(eval_steps, list) and isinstance(eval_losses, list) and len(eval_steps) == len(eval_losses):
        for step, loss in zip(eval_steps[-limit:], eval_losses[-limit:]):
            if _number(step) is None or step < 0:
                continue
            point = points.setdefault(int(step), {'step': int(step), 'timestamp': 0, 'loss': None, 'learning_rate': None})
            point['val_loss'] = _number(loss)
    return sorted(points.values(), key=lambda item: item['step'])[-limit:]
