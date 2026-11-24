"""Isolated native world model training, preflight and imagination worker."""
import argparse
import base64
import contextlib
from io import BytesIO
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def preflight(c):
    from backend.services.world_data import datasets
    from backend.services.world_training import device
    data, val = datasets(c)
    dev = device(c)
    checks = [dict(name='Episodes', status='pass', detail=f'{len(data.episodes)} episodes, {len(data.windows)} windows; all frames decoded and contracts checked'),
              dict(name='Device', status='pass', detail=str(dev)),
              dict(name='Holdout', status='pass' if val else 'warning', detail=f'{len(val.windows)} independent validation windows' if val else 'No holdout: training loss does not measure generalization')]
    for key in ('rewards', 'dones'):
        if not any(ep[key] is not None for ep in data.episodes):
            checks.append(dict(name=key, status='warning', detail=f'No {key} supervision; predictions will be omitted'))
    return dict(ready=True, checks=checks, data_summary={'train_windows':len(data.windows),'validation_windows':len(val.windows) if val else 0})


def rollout(p):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import load_file
    from backend.services.world_data import action, vector
    from backend.services.world_training import device
    from backend.services.world_model import WorldModel
    from backend.routes.world_models import WorldRequest
    path = Path(p['model_path'])
    meta = json.loads((path / 'world.json').read_text())
    if meta.get('format') != 'mah_world_v1':
        raise ValueError('Not a native world model export')
    c = WorldRequest(**meta['config']).model_dump()
    stats = meta['statistics']
    a = np.stack([action(x, c) for x in p['actions']])
    a = (a - np.array(stats['actions']['mean'])) / np.array(stats['actions']['std'])
    state = vector(p['state'], c['state_dim'], 'initial state')
    if c['state_dim']:
        state = (state - np.array(stats['states']['mean'])) / np.array(stats['states']['std'])
    raw = p['image_base64'].split(',')[-1]
    with Image.open(BytesIO(base64.b64decode(raw, validate=True))) as source:
        frame = np.array(ImageOps.exif_transpose(source).convert('RGB').resize((c['image_size'], c['image_size'])), dtype=np.float32).copy() / 255
    dev = device({**c, 'device': p['device']})
    model = WorldModel(c).to(dev)
    model.load_state_dict(load_file(str(path / 'model.safetensors'))); model.eval()
    with torch.no_grad():
        frames, states, rewards, dones = model.rollout(torch.tensor(frame).permute(2, 0, 1)[None].to(dev),
            torch.tensor(state, dtype=torch.float32)[None].to(dev), torch.tensor(a, dtype=torch.float32)[None].to(dev))
    if not all(torch.isfinite(x).all() for x in (frames, states, rewards, dones)):
        raise ValueError('Non-finite world model prediction')
    images = []
    for frame in frames[0]:
        buffer = BytesIO()
        Image.fromarray((frame.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')).save(buffer, format='PNG')
        images.append('data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())
    predicted_states = states[0].cpu().numpy()
    if c['state_dim']:
        predicted_states = predicted_states * np.array(stats['states']['std']) + np.array(stats['states']['mean'])
    return dict(frames=images, states=predicted_states.tolist() if c['state_dim'] else None,
        rewards=rewards[0].cpu().tolist() if meta['supervised']['rewards'] else None,
        done_probabilities=dones[0].sigmoid().cpu().tolist() if meta['supervised']['dones'] else None,
        warning='Imagined trajectory; quality and uncertainty require independent environment evaluation.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'preflight', 'rollout'], required=True)
    parser.add_argument('--config'); parser.add_argument('--metrics')
    args = parser.parse_args()
    try:
        if args.mode == 'train':
            from backend.services.world_training import train
            train(json.loads(Path(args.config).read_text()), args.metrics)
        else:
            payload = json.load(sys.stdin)
            with contextlib.redirect_stdout(sys.stderr):
                result = preflight(payload) if args.mode == 'preflight' else rollout(payload)
            print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        if args.mode == 'train' and args.metrics:
            from backend.paths import atomic_json
            try:
                metrics = json.loads(Path(args.metrics).read_text())
            except (ValueError, OSError):
                metrics = {}
            metrics.update(status='error', error=str(exc))
            atomic_json(Path(args.metrics), metrics)
        traceback.print_exc(); sys.exit(1)


if __name__ == '__main__':
    main()
