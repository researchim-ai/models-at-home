"""Native world dynamics optimization, exact same-plan resume, portable exports."""
import contextlib
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from backend.paths import atomic_json
from backend.services.world_data import datasets
from backend.services.world_model import WorldModel


def device(c):
    cuda = c['device'] != 'cpu' and torch.cuda.is_available()
    if c['device'] == 'cuda' and not cuda:
        raise ValueError('CUDA is unavailable')
    if cuda and c['mixed_precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bf16')
    return torch.device('cuda' if cuda else 'cpu')


def signature(c, train, val):
    fields = ('architecture', 'action_type', 'action_dim', 'discrete_classes', 'state_dim', 'image_size', 'sequence_length',
              'latent_dim', 'hidden_dim', 'batch_size', 'learning_rate', 'kl_weight', 'free_nats', 'mixed_precision', 'device')
    return hashlib.sha256(json.dumps({**{k: c[k] for k in fields}, 'train': train.fingerprint, 'val': val.fingerprint if val else None}, sort_keys=True).encode()).hexdigest()


def export(model, c, stats, path, supervised):
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path / 'model.safetensors'))
    atomic_json(path / 'world.json', {'format': 'mah_world_v1', 'config': c, 'statistics': stats, 'supervised': supervised})


def train(c, metrics_path):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    dev = device(c)
    data, val = datasets(c)
    stats = data.statistics()
    fingerprint = signature(c, data, val)
    model = WorldModel(c).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda' and c['mixed_precision'] == 'fp16')
    rng = torch.Generator().manual_seed(42)
    start = 0
    supervised = {'states': c['state_dim'] > 0, 'rewards': any(e['rewards'] is not None for e in data.episodes), 'dones': any(e['dones'] is not None for e in data.episodes)}
    if c.get('resume_from_checkpoint'):
        checkpoint = Path(c['resume_from_checkpoint'])
        state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
        if state['signature'] != fingerprint:
            raise ValueError('Checkpoint contract, dataset content or optimizer configuration changed')
        model.load_state_dict(load_file(str(checkpoint / 'model.safetensors')))
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng']); rng.set_state(state['sampler_rng'])
        if dev.type == 'cuda' and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start = state['step']
    if start >= c['max_steps']:
        raise ValueError('Checkpoint already reached max_steps')
    metrics = dict(status='running', current_step=start, steps_history=[], loss_history=[], lr_history=[], val_steps_history=[], val_loss_history=[])
    output = Path(c['output_dir'])
    def amp():
        return torch.autocast('cuda', dtype=torch.bfloat16 if c['mixed_precision'] == 'bf16' else torch.float16) if dev.type == 'cuda' and c['mixed_precision'] != 'no' else contextlib.nullcontext()
    started=time.perf_counter()
    metrics.update(train_step_seconds_history=[],validation_seconds_history=[],checkpoint_seconds_history=[])
    for step in range(start + 1, c['max_steps'] + 1):
        step_started=time.perf_counter()
        model.train()
        indices = torch.randint(len(data.windows), (c['batch_size'],), generator=rng).tolist()
        b = data.batch(indices, stats, dev)
        for attempt in range(16):
            optimizer.zero_grad(set_to_none=True)
            with amp():
                loss, parts = model.objective(b)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite world model loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if not finite:
                if not scaler.is_enabled():
                    raise ValueError('Non-finite world model gradients')
                # GradScaler skips this update and lowers its scale; retry without counting an optimizer step.
                scaler.step(optimizer); scaler.update()
                metrics['amp_overflows'] = metrics.get('amp_overflows', 0) + 1
                continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            break
        else:
            raise ValueError('FP16 gradients still overflow after 16 scale reductions')
        metrics.update(current_step=step, current_loss=float(loss.detach()), current_lr=c['learning_rate'], grad_norm=float(norm), **{k: float(v.detach()) for k, v in parts.items()})
        metrics['steps_history'].append(step); metrics['loss_history'].append(float(loss.detach())); metrics['lr_history'].append(c['learning_rate'])
        metrics['train_step_seconds_history'].append(time.perf_counter()-step_started)
        if val and (step % c['eval_every'] == 0 or step == c['max_steps']):
            eval_started=time.perf_counter()
            model.eval()
            # Validation scores strictly free-running future images, never posterior reconstructions.
            error, count = 0., 0
            with torch.no_grad(), amp():
                for offset in range(0, len(val.windows), c['batch_size']):
                    vb = val.batch(list(range(offset, min(offset+c['batch_size'], len(val.windows)))), stats, dev)
                    frames = model.rollout(vb['images'][:, 0], vb['states'][:, 0], vb['actions'])[0]
                    error += float((frames.float() - vb['images'][:, 1:]).square().sum())
                    count += frames.numel()
            metrics['current_val_loss'] = error / count
            metrics['val_steps_history'].append(step); metrics['val_loss_history'].append(error / count)
            metrics['validation_seconds_history'].append(time.perf_counter()-eval_started)
        if step % c['save_every'] == 0 or step == c['max_steps']:
            save_started=time.perf_counter()
            checkpoint = output / f'checkpoint-{step}'
            export(model, c, stats, checkpoint, supervised)
            torch.save(dict(step=step, signature=fingerprint, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                torch_rng=torch.get_rng_state(), sampler_rng=rng.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if dev.type == 'cuda' else None), checkpoint / 'training_state.pt')
            atomic_json(checkpoint / 'complete.json', {'format': 'mah_world_training_v1', 'step': step})
            metrics['checkpoint_seconds_history'].append(time.perf_counter()-save_started)
        elapsed=time.perf_counter()-started
        metrics.update(elapsed_seconds=elapsed,step_seconds=metrics['train_step_seconds_history'][-1],estimated_remaining_seconds=elapsed/(step-start)*(c['max_steps']-step))
        atomic_json(Path(metrics_path), metrics)
        print(f'step={step} loss={float(loss.detach()):.6f}', flush=True)
    final = output / 'final_model'
    export_started=time.perf_counter()
    export(model, c, stats, final, supervised)
    metrics['final_export_seconds']=time.perf_counter()-export_started
    metrics.update(status='completed', final_model_dir=str(final))
    atomic_json(Path(metrics_path), metrics)
    return metrics
