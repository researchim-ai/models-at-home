"""VLA train/preflight/prediction entrypoint in the isolated ML Python."""
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


def preflight(config):
    import torch
    from PIL import Image, ImageOps
    from backend.services.vla_data import Demonstrations, batch
    report = {'ready': False, 'checks': [], 'sample': None}
    def check(name, status, detail):
        report['checks'].append({'name': name, 'status': status, 'detail': str(detail)[:2000]})
    try:
        data = Demonstrations(config['data_path'], config)
        if max(len(row['actions']) for row in data.rows) < config['action_horizon']:
            raise ValueError('No training targets for the full action horizon; reduce action_horizon or supply full chunks')
        stats = data.statistics()
        check('Demonstrations', 'pass', f'{len(data)} rows; all action/state dimensions and finite values checked')
        if config.get('val_data_path'):
            validation = Demonstrations(config['val_data_path'], config)
            check('Validation', 'pass', f'{len(validation)} rows; train-only normalization')
            overlap = {row['image'] for row in data.rows} & {row['image'] for row in validation.rows}
            train_episodes = {str(row['episode_id']) for row in data.rows if row['episode_id'] is not None}
            val_episodes = {str(row['episode_id']) for row in validation.rows if row['episode_id'] is not None}
            if overlap or train_episodes & val_episodes:
                check('Holdout overlap', 'warning', f'Train/validation share {len(overlap)} image paths or episode IDs; split by independent episodes/sessions before quality evaluation')
        else:
            validation = None
            check('Validation', 'warning', 'Holdout not selected; training loss alone does not measure policy quality')
        cuda = torch.cuda.is_available() and config['device'] != 'cpu'
        if (config['device'] == 'cuda' or config['architecture'] == 'vlm_action') and not cuda:
            raise ValueError('Selected action model requires CUDA')
        if cuda and config['mixed_precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('GPU does not support bf16')
        check('Device', 'pass', torch.cuda.get_device_name(0) if cuda else 'CPU')
        processor = None
        if config['architecture'] == 'vlm_action':
            from transformers import AutoConfig, AutoModelForImageTextToText
            from homellm.training.vlm_common import load_vlm_processor
            model_config = AutoConfig.from_pretrained(config['model_path'], trust_remote_code=True)
            if type(model_config) not in AutoModelForImageTextToText._model_mapping:
                raise ValueError('Installed Transformers does not support this VLM architecture')
            processor = load_vlm_processor(config['model_path'], config)
            if config['tuning_method'] in ('lora', 'qlora'):
                import peft
            if config['tuning_method'] == 'qlora':
                import bitsandbytes
            check('VLM backbone', 'pass', model_config.model_type + '; processor only, weights are not loaded')
        for dataset in (data, validation):
            if dataset is None:
                continue
            for row in dataset.rows[:200]:
                with Image.open(row['image']) as image:
                    image.verify()
                batch([row], config, stats, processor)
            if len(dataset) > 200:
                check('Image scope', 'warning', 'Image decoding/processor limited to first 200 rows per dataset')
        first = data.rows[0]
        with Image.open(first['image']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((256, 256))
            out = BytesIO()
            image.save(out, format='PNG')
        report['sample'] = {'image': 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode(),
            'instruction': first['instruction'], 'state': first['state'].tolist(), 'actions': first['raw_actions'],
            'valid_timesteps': len(first['actions']), 'horizon': config['action_horizon']}
        report['statistics'] = stats
        spec = f"{config['action_dim']} classes" if config['action_type'] == 'discrete' else f"{config['action_dim']} continuous coordinates"
        if config['action_type'] == 'hybrid':
            spec += f" + categorical heads {config['discrete_classes']}"
        check('Action policy', 'pass', f"{spec} × {config['action_horizon']} timesteps; short terminal chunks masked")
        report['ready'] = True
    except Exception as exc:
        check('VLA preflight', 'error', exc)
    return report


def predict(payload):
    import torch
    import numpy as np
    from backend.services.vla_model import create, restore_weights
    from backend.services.vla_data import batch, _vector
    from PIL import Image, ImageOps
    metadata = json.loads((Path(payload['model_path']) / 'policy.json').read_text())
    if metadata['format'] != 'mah_vla_v1':
        raise ValueError('Not a Studio VLA policy export')
    config, stats = metadata['config'], metadata['statistics']
    cuda = torch.cuda.is_available() and payload.get('device', 'auto') != 'cpu'
    if payload.get('device') == 'cuda' and not cuda:
        raise ValueError('CUDA is unavailable')
    if config['architecture'] == 'vlm_action' and not cuda:
        raise ValueError('VLM action policy requires CUDA')
    # Restore a local backbone snapshot; LoRA exports retain the original base reference.
    if config['architecture'] == 'vlm_action':
        from backend.services.vlm_runtime import adapter_source
        source = Path(payload['model_path']) / 'backbone'
        config = {**config, 'model_path': adapter_source(str(source)), 'base_model_path': adapter_source(str(source))}
    instruction = payload['instruction']
    if not instruction.strip() or len(instruction.encode()) > config['max_instruction_bytes']:
        raise ValueError('Invalid instruction length')
    state = _vector(payload.get('state', []), config['state_dim'], 'state')
    image = ImageOps.exif_transpose(Image.open(BytesIO(base64.b64decode(payload['image_base64'].split(',')[-1], validate=True)))).convert('RGB')
    model, processor = create(config)
    restore_weights(model, payload['model_path'])
    model.to('cuda' if cuda else 'cpu').eval()
    import tempfile
    with tempfile.TemporaryDirectory(prefix='mah-vla-predict-') as temporary:
        path = Path(temporary) / 'input.png'
        image.save(path)
        width = config['action_dim'] + (len(config.get('discrete_classes', [])) if config['action_type'] == 'hybrid' else 0)
        empty = np.zeros((1, width), dtype=np.float32) if config['action_type'] != 'discrete' else np.zeros(1, dtype=np.int64)
        inputs, states, _, _ = batch([{'image': str(path), 'instruction': instruction, 'state': state, 'actions': empty}], config, stats, processor)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        from contextlib import nullcontext
        amp = torch.autocast('cuda', dtype=torch.bfloat16 if config['mixed_precision'] == 'bf16' else torch.float16) if cuda and config['mixed_precision'] != 'no' else nullcontext()
        with torch.no_grad(), amp:
            output = model(inputs, states.to(device)).float()[0]
        if config['action_type'] == 'continuous':
            actions = output * output.new_tensor(stats['action']['std']) + output.new_tensor(stats['action']['mean'])
        elif config['action_type'] == 'hybrid':
            continuous = output[:, :config['action_dim']] * output.new_tensor(stats['action']['std']) + output.new_tensor(stats['action']['mean'])
            discrete, offset = [], config['action_dim']
            for classes in config['discrete_classes']:
                discrete.append(output[:, offset:offset + classes].argmax(-1).tolist())
                offset += classes
            if not torch.isfinite(continuous).all():
                raise ValueError('Nonfinite policy prediction')
            return {'actions': [{'continuous': row, 'discrete': [values[index] for values in discrete]} for index, row in enumerate(continuous.cpu().tolist())],
                    'action_type': 'hybrid', 'action_dim': config['action_dim'], 'action_horizon': config['action_horizon'],
                    'action_names': config['action_names'], 'discrete_names': config['discrete_names'], 'discrete_classes': config['discrete_classes'], 'discrete_labels': config['discrete_labels']}
        else:
            actions = output.argmax(-1)
        if not torch.isfinite(actions).all():
            raise ValueError('Nonfinite policy prediction')
        return {'actions': actions.cpu().tolist(), 'action_type': config['action_type'],
                'action_dim': config['action_dim'], 'action_horizon': config['action_horizon'], 'action_names': config['action_names']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'preflight', 'predict'], required=True)
    parser.add_argument('--config')
    parser.add_argument('--metrics')
    args = parser.parse_args()
    try:
        if args.mode == 'train':
            config = json.loads(Path(args.config).read_text())
            if config.get('training_backend') == 'unsloth':
                import unsloth
            from backend.services.vla_training import train
            from homellm.training.vlm_common import MetricsLogger
            train(config, MetricsLogger(Path(args.metrics)))
        else:
            payload = json.load(sys.stdin)
            with contextlib.redirect_stdout(sys.stderr):
                backend = payload.get('training_backend')
                if args.mode == 'predict':
                    backend = json.loads((Path(payload['model_path']) / 'policy.json').read_text())['config'].get('training_backend')
                if backend == 'unsloth':
                    import unsloth
                result = preflight(payload) if args.mode == 'preflight' else predict(payload)
            print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        if args.metrics:
            from backend.paths import atomic_json
            path = Path(args.metrics)
            try:
                previous = json.loads(path.read_text())
            except (OSError, ValueError):
                previous = {}
            atomic_json(path, {**previous, 'status': 'error', 'error': str(exc) + '\n' + traceback.format_exc()})
        raise
