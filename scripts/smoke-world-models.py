"""Run genuine CPU/CUDA world-model training, checkpoint extension and rollout."""
import argparse
import base64
import contextlib
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument('--architecture', choices=['latent_gru', 'rssm'], default='latent_gru')
parser.add_argument('--action-type', choices=['continuous', 'discrete', 'hybrid'], default='continuous')
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--precision', choices=['no', 'bf16', 'fp16'], default='no')
parser.add_argument('--gpu-id', type=int, default=0)
parser.add_argument('--steps', type=int, default=60)
parser.add_argument('--result', required=True)
args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
from backend.routes.world_models import WorldRequest
from backend.services.world_training import train
from backend.services.world_worker import rollout, preflight
import torch
from safetensors.torch import load_file

root = Path(tempfile.mkdtemp(prefix='mah-world-smoke-'))
demo = Path(__file__).resolve().parents[1] / 'examples' / 'world-models'
for split in ('train', 'validation'):
    rows = []
    for line in (demo / f'{split}.jsonl').read_text().splitlines():
        row = json.loads(line)
        row['frames'] = [str(demo / f) for f in row['frames']]
        original = row['actions']
        if args.action_type == 'discrete':
            row['actions'] = [(a[0] // 2 + 1) * 3 + a[1] // 2 + 1 for a in original]
        elif args.action_type == 'hybrid':
            row['actions'] = [{'continuous': a, 'discrete': [int(a[0] > 0)]} for a in original]
        rows.append(row)
    (root / f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
c = WorldRequest(architecture=args.architecture, dataset_path=str(root / 'train.jsonl'), val_data_path=str(root / 'validation.jsonl'),
    action_type=args.action_type, action_dim=9 if args.action_type == 'discrete' else 2,
    discrete_classes=[2] if args.action_type == 'hybrid' else [], state_dim=2,
    batch_size=8, max_steps=args.steps, save_every=args.steps, eval_every=args.steps, device=args.device,
    mixed_precision=args.precision, output_dir=str(root / 'out')).model_dump()
assert preflight(c)['ready']
with contextlib.redirect_stdout(sys.stderr):
    metrics = train(c, root / 'metrics.json')
assert metrics['loss_history'][-1] < metrics['loss_history'][0] * .8, metrics
before = load_file(str(Path(metrics['final_model_dir']) / 'model.safetensors'))
resumed = {**c, 'max_steps': args.steps+2, 'resume_from_checkpoint': str(root / 'out' / f'checkpoint-{args.steps}'), 'output_dir': str(root / 'resumed')}
with contextlib.redirect_stdout(sys.stderr):
    final = train(resumed, root / 'resume-metrics.json')
after = load_file(str(Path(final['final_model_dir']) / 'model.safetensors'))
assert any(not torch.equal(before[k], after[k]) for k in before)
row = json.loads((root / 'validation.jsonl').read_text().splitlines()[0])
prediction = rollout(dict(model_path=final['final_model_dir'], image_base64=base64.b64encode(Path(row['frames'][0]).read_bytes()).decode(),
    actions=row['actions'][:4], state=row['states'][0], device=args.device))
assert len(prediction['frames']) == 4
result = dict(architecture=args.architecture, action_type=args.action_type, device=args.device, precision=args.precision,
    gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None, steps=args.steps,
    first_loss=metrics['loss_history'][0], last_loss=metrics['loss_history'][-1], val_rollout_mse=metrics['current_val_loss'],
    amp_overflows=metrics.get('amp_overflows', 0), resume_step=final['current_step'], export_reload_frames=len(prediction['frames']), states=prediction['states'],
    rewards=prediction['rewards'], done_probabilities=prediction['done_probabilities'], artifact_dir=str(root))
Path(args.result).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
