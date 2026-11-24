"""Behavior cloning for continuous action chunks or discrete action classes."""
import hashlib
import json
from pathlib import Path
import random


def signature(config):
    keys = ('architecture', 'training_backend', 'model_path', 'action_type', 'domain', 'action_names', 'discrete_names', 'discrete_classes', 'discrete_labels', 'action_dim', 'action_horizon', 'state_dim',
            'hidden_dim', 'image_size', 'vision_grid', 'max_instruction_bytes', 'tuning_method', 'mixed_precision',
            'batch_size', 'gradient_accumulation', 'learning_rate', 'warmup_steps', 'seq_len',
            'freeze_vision_tower', 'lora_r', 'lora_alpha', 'max_pixels')
    payload = {key: config.get(key) for key in keys}
    for key in ('data_path', 'val_data_path'):
        if config.get(key):
            digest = hashlib.sha256()
            with Path(config[key]).open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            payload[key] = digest.hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def action_loss(predictions, targets, mask, config):
    action_type = config['action_type']
    import torch.nn.functional as F
    if action_type == 'continuous':
        values = F.smooth_l1_loss(predictions.float(), targets, reduction='none').mean(-1)
    elif action_type == 'hybrid':
        dim = config['action_dim']
        values = F.smooth_l1_loss(predictions[..., :dim].float(), targets[..., :dim], reduction='none').mean(-1)
        offset = dim
        discrete = []
        for index, classes in enumerate(config['discrete_classes']):
            discrete.append(F.cross_entropy(predictions[..., offset:offset + classes].float().transpose(1, 2), targets[..., dim + index].long(), reduction='none'))
            offset += classes
        values = values + sum(discrete) / len(discrete)
    else:
        values = F.cross_entropy(predictions.float().transpose(1, 2), targets, reduction='none')
    return (values * mask).sum() / mask.sum().clamp(min=1)


def train(config, metrics):
    import torch
    import numpy as np
    from backend.services.vla_data import Demonstrations, batch
    from backend.services.vla_model import create, save, restore_weights
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cuda = config['device'] != 'cpu' and torch.cuda.is_available()
    if config['device'] == 'cuda' and not cuda:
        raise ValueError('CUDA is unavailable')
    if config['architecture'] == 'vlm_action' and not cuda:
        raise ValueError('VLM action fine-tuning requires CUDA')
    device = 'cuda' if cuda else 'cpu'
    if cuda and config['mixed_precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('Selected GPU does not support bf16')
    metrics.update(status='loading_dataset', stage='vla_sft', policy_format='mah_vla_v1', device=device,
                   effective_precision=config['mixed_precision'] if cuda else 'float32')
    data = Demonstrations(config['data_path'], config)
    validation = Demonstrations(config['val_data_path'], config) if config.get('val_data_path') else None
    if max(len(row['actions']) for row in data.rows) < config['action_horizon']:
        raise ValueError('No training targets for the full action horizon; reduce action_horizon or supply full chunks')
    stats, fingerprint = data.statistics(), signature(config)
    metrics.update(status='loading_model')
    model, processor = create(config)
    model.to(device)
    if config['mixed_precision'] == 'fp16':
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
    def schedule(step):
        if step < config['warmup_steps']:
            return step / max(1, config['warmup_steps'])
        ratio = (step - config['warmup_steps']) / max(1, config['max_steps'] - config['warmup_steps'])
        return .5 * (1 + np.cos(np.pi * min(1, ratio)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda and config['mixed_precision'] == 'fp16')
    step = cursor = 0
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(len(data), generator=generator).tolist()
    if config.get('resume_from_checkpoint'):
        directory = Path(config['resume_from_checkpoint'])
        state = torch.load(directory / 'training_state.pt', map_location='cpu', weights_only=False)
        if state['signature'] != fingerprint or state['format'] != 'mah_vla_training_v1':
            raise ValueError('VLA checkpoint data/model/optimizer configuration changed')
        restore_weights(model, directory)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        stats = state['statistics']
        step, cursor, order = state['step'], state['cursor'], state['order']
        generator.set_state(state['generator'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        if cuda:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        for group, base in zip(optimizer.param_groups, scheduler.base_lrs):
            group['lr'] = base * schedule(step)
        if step >= config['max_steps']:
            raise ValueError('Checkpoint reached max_steps; extend the training plan')
    from contextlib import nullcontext
    def amp():
        return torch.autocast('cuda', dtype=torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float16) if cuda and config['mixed_precision'] != 'no' else nullcontext()
    def prepare(rows):
        inputs, states, targets, mask = batch(rows, config, stats, processor)
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        return inputs, states.to(device), targets.to(device), mask.to(device)
    def checkpoint(directory, final=False):
        save(directory, model, processor, stats, config)
        if not final:
            torch.save({'format': 'mah_vla_training_v1', 'signature': fingerprint, 'step': step,
                'cursor': cursor, 'order': order, 'generator': generator.get_state(), 'statistics': stats,
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if cuda else [],
                'python_rng': random.getstate()}, directory / 'training_state.pt')
            from backend.paths import atomic_json
            atomic_json(directory / 'complete.json', {'format': 'mah_vla_training_v1', 'step': step})
        metrics.log_checkpoint(str(directory), step=step)
    metrics.update(status='training', total_steps=config['max_steps'], current_step=step,
                   trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    while step < config['max_steps']:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for _ in range(config['gradient_accumulation']):
            rows = []
            for _ in range(config['batch_size']):
                if cursor >= len(order):
                    order = torch.randperm(len(data), generator=generator).tolist()
                    cursor = 0
                rows.append(data.rows[order[cursor]])
                cursor += 1
            inputs, states, targets, mask = prepare(rows)
            with amp():
                predictions = model(inputs, states)
                loss = action_loss(predictions, targets, mask, config)
            if not torch.isfinite(loss):
                raise ValueError('Action loss is not finite')
            scaler.scale(loss / config['gradient_accumulation']).backward()
            total_loss += float(loss.detach()) / config['gradient_accumulation']
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('Action gradients are not finite')
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        metrics.log_step(step, total_loss, optimizer.param_groups[0]['lr'])
        metrics.metrics.setdefault('grad_norm_history', []).append(float(norm))
        metrics.update(action_loss=total_loss, optimizer_step=step)
        if validation and step % config['eval_every'] == 0:
            model.eval()
            losses, valid_count, total_error, class_correct = 0., 0., 0., 0.
            with torch.no_grad():
                for offset in range(0, len(validation), config['batch_size']):
                    inputs, states, targets, mask = prepare(validation.rows[offset:offset + config['batch_size']])
                    with amp():
                        predictions = model(inputs, states)
                        loss = action_loss(predictions, targets, mask, config)
                    count = float(mask.sum())
                    losses += float(loss) * count
                    valid_count += count
                    if config['action_type'] != 'discrete':
                        error = (predictions[..., :config['action_dim']].float() - targets[..., :config['action_dim']]).abs() * predictions.new_tensor(stats['action']['std'])
                        total_error += float((error.mean(-1) * mask).sum())
                    else:
                        total_error += float(((predictions.argmax(-1) == targets) * mask).sum())
                    if config['action_type'] == 'hybrid':
                        offset = config['action_dim']
                        for index, classes in enumerate(config['discrete_classes']):
                            class_correct += float(((predictions[..., offset:offset + classes].argmax(-1) == targets[..., config['action_dim'] + index].long()) * mask).sum())
                            offset += classes
            if config['action_type'] == 'hybrid':
                metrics.metrics['action_accuracy'] = class_correct / (valid_count * len(config['discrete_classes']))
            metrics.metrics.setdefault('val_steps_history', []).append(step)
            metrics.metrics.setdefault('val_loss_history', []).append(losses / valid_count)
            metrics.update(current_val_loss=losses / valid_count,
                **({'action_mae': total_error / valid_count} if config['action_type'] != 'discrete' else {'action_accuracy': total_error / valid_count}))
        if step % config['save_every'] == 0:
            checkpoint(Path(config['output_dir']) / f'checkpoint-{step}')
    final = Path(config['output_dir']) / 'final_model'
    checkpoint(final, final=True)
    metrics.update(status='completed', final_model_dir=str(final))
