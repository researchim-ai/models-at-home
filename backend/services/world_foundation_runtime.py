"""Local pretrained model adapters; embedded DIAMOND and optional Cosmos pipelines."""
import base64
from io import BytesIO
import json
from pathlib import Path
from backend.services.world_catalog import model_info


def seed_image(c):
    from PIL import Image, ImageOps
    raw = c.get('image_base64', '')
    if not raw and c.get('image_file'):
        raw = Path(c['image_file']).read_text()
    if not raw:
        return None
    data = base64.b64decode(raw.split(',')[-1], validate=True)
    with Image.open(BytesIO(data)) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError('Seed image exceeds 40 megapixels')
        return ImageOps.exif_transpose(source).convert('RGB')


def preflight(c):
    checks = []
    info = model_info(c['model_id'])
    def check(name, status, detail):
        checks.append(dict(name=name, status=status, detail=str(detail)))
    try:
        image = seed_image(c)
        if info['provider'] == 'cosmos3':
            import importlib.util
            if c.get('model_path'):
                root = Path(c['model_path'])
                index = json.loads((root / 'model_index.json').read_text())
                if index.get('_class_name') != 'Cosmos3OmniPipeline':
                    raise ValueError('Local snapshot is not a Cosmos3OmniPipeline')
                for component in ('transformer', 'vae', 'text_tokenizer', 'scheduler'):
                    if not (root / component).is_dir():
                        raise ValueError(f'Incomplete local snapshot: missing {component}')
                for component in ('transformer', 'vae'):
                    if not any((root / component).glob('*.safetensors')):
                        raise ValueError(f'Local snapshot is missing {component} weights; finish Download first')
                for index_path in root.glob('*/**/*.safetensors.index.json'):
                    shards = json.loads(index_path.read_text()).get('weight_map', {})
                    if any(not (index_path.parent / name).is_file() for name in shards.values()):
                        raise ValueError('Local snapshot is missing weight shards; finish Download first')
            import torch
            from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
            if not torch.cuda.is_available():
                raise ValueError('Pretrained Cosmos generation requires CUDA in this integration')
            if not torch.cuda.is_bf16_supported():
                raise ValueError('Cosmos requires BF16-capable CUDA GPU')
            if not importlib.util.find_spec('cosmos_guardrail'):
                raise ValueError('Install requirements-world-foundation.txt in the World Models Python, including cosmos_guardrail')
            if c['task'] == 'forward_dynamics':
                CosmosActionCondition(mode='forward_dynamics', chunk_size=len(c['actions']), domain_name=c['domain'],
                    resolution_tier=c['height'], raw_actions=torch.tensor(c['actions'], dtype=torch.float32), image=image)
            import psutil
            required = {'cosmos3_edge': 12, 'cosmos3_nano': 40, 'cosmos3_super': 140}[c['model_id']]
            if c['cpu_offload'] and psutil.virtual_memory().available / 2**30 < required:
                raise ValueError(f'CPU offload needs approximately {required}+ GiB available RAM for this model')
            if not c['cpu_offload'] and torch.cuda.get_device_properties(0).total_memory / 2**30 < required:
                raise ValueError(f'Full CUDA placement needs approximately {required}+ GiB VRAM; use CPU offload or a smaller model')
            from huggingface_hub import HfApi
            import os
            if not os.environ.get('HF_HUB_OFFLINE'):
                from huggingface_hub.errors import GatedRepoError
                try:
                    HfApi().auth_check('nvidia/Cosmos-1.0-Guardrail', repo_type='model')
                except GatedRepoError as exc:
                    raise ValueError('Нужен доступ к nvidia/Cosmos-1.0-Guardrail на Hugging Face. Разрешите доступ в этом репозитории, затем выполните hf auth login в ML-окружении или задайте HF_TOKEN для backend.') from exc
            check('Runtime', 'pass', f'Cosmos3OmniPipeline; {torch.cuda.get_device_name(0)}')
            check('Weights', 'warning', 'Preflight does not load multi-GB weights. Generation/download use the exact selected revision; Hub access and guardrail model access are required.')
            check('Model', 'pass', c.get('model_path') or f"{info['repo_id']}@{c.get('revision') or info['revision']}")
        else:
            import torch
            from backend.services.world_local import download
            if c.get('device') == 'cuda' and not torch.cuda.is_available():
                raise ValueError('CUDA unavailable')
            check('Architecture', 'pass', 'Embedded DIAMOND denoiser, UNet and EDM sampler; no external runtime or API')
            check('Weights', 'pass', c.get('model_path') or download(c['model_id'],c['game'],offline=True))
        return dict(ready=True, checks=checks)
    except Exception as exc:
        check('Preflight', 'error', str(exc))
        return dict(ready=False, checks=checks)


def download(c, progress):
    info = model_info(c['model_id'])
    if info['provider'] == 'cosmos3':
        from huggingface_hub import snapshot_download, HfApi
        revision = c.get('revision') or info['revision']
        # Resolve aliases once, record the immutable snapshot used.
        sha = HfApi().model_info(info['repo_id'], revision=revision).sha
        progress(status='downloading', resolved_revision=sha)
        path = snapshot_download(info['repo_id'], revision=sha, allow_patterns=[
            'model_index.json', 'transformer/*', 'vae/*', 'scheduler/*', 'text_tokenizer/*', 'sound_tokenizer/*'])
        return dict(model_path=path, resolved_revision=sha, repo_id=info['repo_id'], operation='download')
    if info['provider'] == 'diamond':
        from backend.services.world_local import download as local_download
        path=local_download(c['model_id'],c['game'])
        return dict(model_path=path,resolved_revision=info['revision'],repo_id=info['repo_id'],operation='download')
    raise ValueError('Unknown local world model')


def cosmos(c, progress):
    import torch
    from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    info = model_info(c['model_id'])
    image = seed_image(c)
    progress(status='loading')
    pipe = Cosmos3OmniPipeline.from_pretrained(c.get('model_path') or info['repo_id'],
        revision=c.get('revision') or info['revision'], torch_dtype=torch.bfloat16, enable_safety_checker=True)
    pipe.transformer.set_attention_backend('native')
    if c['cpu_offload']:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    if hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config,
        flow_shift=10. if c['task'] == 'forward_dynamics' else 12., use_karras_sigmas=False)
    def callback(pipeline, step, timestep, kwargs):
        progress(status='generating', current_step=step+1)
        return kwargs
    kwargs = dict(prompt=c['prompt'], fps=c['fps'], num_inference_steps=c['inference_steps'], guidance_scale=c['guidance_scale'],
        generator=torch.Generator(device='cuda').manual_seed(c['seed']), callback_on_step_end=callback)
    if c['task'] == 'forward_dynamics':
        kwargs.update(action=CosmosActionCondition(mode='forward_dynamics', chunk_size=len(c['actions']), domain_name=c['domain'],
            resolution_tier=c['height'], raw_actions=torch.tensor(c['actions'], dtype=torch.float32), image=image), use_system_prompt=False)
    else:
        kwargs.update(num_frames=c['num_frames'], height=c['height'], width=c['width'], negative_prompt=c['negative_prompt'])
        if c['task'] == 'image2video':
            kwargs['image'] = image
    progress(status='generating', current_step=0)
    with torch.inference_mode():
        result = pipe(**kwargs)
    return list(result.video)


def save_video(frames, output, fps):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError('Foundation model returned no frames')
    previews = []
    selected = set(np.linspace(0, len(frames)-1, min(12, len(frames)), dtype=int).tolist())
    with imageio.get_writer(str(output / 'generated.mp4'), fps=fps, codec='libx264', macro_block_size=1) as writer:
        for i, frame in enumerate(frames):
            if not isinstance(frame, Image.Image):
                a = np.asarray(frame)
                if a.dtype.kind == 'f':
                    if not np.isfinite(a).all():
                        raise ValueError('Non-finite pixels from foundation model')
                    a = (a.clip(0, 1) * 255).astype('uint8')
                frame = Image.fromarray(a)
            frame = frame.convert('RGB')
            writer.append_data(np.asarray(frame))
            if i in selected:
                frame = frame.copy(); frame.thumbnail((320, 320))
                buffer = BytesIO(); frame.save(buffer, format='JPEG', quality=85)
                previews.append('data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode())
    return dict(frame_count=len(frames), fps=fps, previews=previews, video_path=str(output / 'generated.mp4'))

