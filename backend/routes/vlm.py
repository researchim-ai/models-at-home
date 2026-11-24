"""VLM (Visual Language Model) training + inference endpoints."""
from __future__ import annotations

import base64
import io
import threading
import json
import os
import signal
import subprocess
import time
import uuid
import sys
from typing import Literal
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from backend.paths import get_project_root, safe_id, atomic_json
from backend.services import processes
from backend.services.vlm_environment import python_command as vlm_python

router = APIRouter()
_model_lock = threading.Lock()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
RUNS_DIR = PROJECT_ROOT / ".runs"
OUTPUT_DIR = PROJECT_ROOT / "out"

_vlm_state: dict = {"model": None, "processor": None, "path": None}


class VLMTrainingRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    stage: Literal["vlm_pretrain", "vlm_sft", "vlm_grpo"] = "vlm_sft"
    model_path: str = ""
    dataset_path: str = ""
    experiment_name: str = ""
    resume_from_checkpoint: str = ""
    resumed_from_run: str = ""
    batch_size: int = Field(default=2, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0)
    max_steps: int = Field(default=500, ge=0)
    num_epochs: int = Field(default=1, gt=0)
    seq_len: int = Field(default=2048, gt=0)
    warmup_steps: int = Field(default=50, ge=0)
    lr_schedule: str = "cosine"
    gradient_accumulation: int = Field(default=4, gt=0)
    mixed_precision: Literal['no', 'bf16', 'fp16'] = "bf16"
    output_dir: str = "out"
    use_lora: bool = True
    lora_rank: int = Field(default=16, gt=0)
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    tuning_method: Literal['', 'full', 'lora', 'qlora'] = ""
    use_flash_attention: bool = True
    gpu_ids: list[int] = []
    log_every: int = Field(default=10, gt=0)
    save_every: int = Field(default=100, gt=0)
    gradient_checkpointing: bool = True
    assistant_only_loss: bool | None = None
    freeze_vision_tower: bool | None = None
    freeze_projector: bool = False
    freeze_language_model: bool = False
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: list[str] = []
    lora_modules_to_save: list[str] = []
    data_base_dir: str = ''
    val_data_path: str = ''
    eval_steps: int = Field(default=0, ge=0)
    vlm_columns: dict = {}
    min_pixels: int | None = Field(default=None, gt=0)
    max_pixels: int | None = Field(default=None, gt=0)
    training_backend: Literal['transformers', 'unsloth'] = 'transformers'
    grpo_algorithm: Literal['grpo', 'dr_grpo', 'dapo', 'sdpo'] = 'grpo'
    grpo_group_size: int = Field(default=4, ge=2, le=64)
    grpo_max_new_tokens: int = Field(default=256, gt=0)
    grpo_temperature: float = Field(default=1.0, gt=0, le=2)
    grpo_kl_weight: float = Field(default=0.04, ge=0)
    grpo_clip_eps_low: float = Field(default=0.2, gt=0, lt=1)
    grpo_clip_eps_high: float = Field(default=0.2, gt=0)
    grpo_reward_fn: Literal['exact_match', 'contains'] = 'exact_match'
    grpo_max_refill_rounds: int = Field(default=3, ge=0, le=100)
    grpo_overlong_buffer: int = Field(default=0, ge=0)
    grpo_overlong_penalty: float = Field(default=-1.0, le=0)
    sdpo_success_threshold: float = Field(default=0.5, ge=0, le=1)
    sdpo_alpha: float = Field(default=0.5, ge=0, le=1)
    sdpo_loss_weight: float = Field(default=1.0, gt=0)
    sdpo_ema_rate: float = Field(default=0.0, ge=0, le=1)
    sdpo_full_logit_distillation: bool = False
    sdpo_distillation_topk: int | None = Field(default=None, gt=0)


    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, values):
        values = dict(values)
        if values.get('grpo_algorithm') == 'dapo':
            values.setdefault('grpo_kl_weight', 0.0)
            values.setdefault('grpo_clip_eps_high', 0.28)
        if "lora_r" in values:
            values["lora_rank"] = values["lora_r"]
        elif "lora_rank" in values:
            values["lora_r"] = values["lora_rank"]
        return values


@router.get('/recipes')
def vlm_recipes():
    from backend.services.vlm_recipes import catalog
    return catalog()


@router.post("/start")
def start_vlm_training(req: VLMTrainingRequest):
    run_id = "vlm_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = safe_id(RUNS_DIR, run_id)
    config = req.model_dump()
    try:
        _normalize_and_validate_vlm_config(config)
        worker_python = vlm_python()
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    config["run_id"] = run_id
    config['vlm_worker_version'] = 3
    config['rl_engine'] = 'homellm_native' if config['stage'] == 'vlm_grpo' else None
    config["started_at"] = datetime.now().isoformat()
    base_output = Path(req.output_dir).expanduser()
    if not base_output.is_absolute():
        base_output = PROJECT_ROOT / base_output
    config["output_dir"] = str((base_output / run_id).resolve())
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=False)
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "config.json", config)
    (run_dir / "metrics.json").write_text(json.dumps({"status": "initializing", "step": 0, "loss": 0}))

    env = os.environ.copy()
    env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if req.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in req.gpu_ids)

    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in req.gpu_ids) if req.gpu_ids else env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    cmd = [worker_python, str(Path(__file__).parent.parent / 'services' / 'vlm_worker.py'), "--config", str(run_dir / "config.json"), "--metrics", str(run_dir / "metrics.json")]

    proc = processes.launch(run_dir, cmd, cwd=PROJECT_ROOT, env=env)
    _raise_if_exited_early(proc, run_dir)

    (RUNS_DIR / "vlm_active_run.json").write_text(json.dumps({
        "run_id": run_id, "started_at": datetime.now().isoformat(), "config": config,
    }, indent=2))

    return {"run_id": run_id}


@router.post('/preflight')
def preflight_vlm(req: VLMTrainingRequest):
    config = req.model_dump()
    def failed(detail):
        return {'ready': False, 'checks': [{'name': 'Проверка эксперимента', 'status': 'error', 'detail': detail}],
                'sample': None, 'datasets': []}
    try:
        _normalize_and_validate_vlm_config(config)
    except ValueError as exc:
        return failed(str(exc))
    env = os.environ.copy()
    env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, req.gpu_ids)) if req.gpu_ids else env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    try:
        result = subprocess.run([vlm_python(), str(Path(__file__).parent.parent / 'services' / 'vlm_preflight_worker.py')],
            input=json.dumps(config), text=True, capture_output=True, cwd=PROJECT_ROOT, env=env, timeout=90)
        if result.returncode:
            return failed(result.stderr[-2000:] or f'Preflight worker exited: {result.returncode}')
        report = json.loads(result.stdout)
        if not isinstance(report, dict) or not isinstance(report.get('ready'), bool) or not isinstance(report.get('checks'), list):
            return failed('Invalid preflight worker response')
        return report
    except subprocess.TimeoutExpired:
        return failed('Проверка превысила 90 секунд. Проверьте доступ к модели и изображениям.')
    except (OSError, ValueError) as exc:
        return failed(str(exc))


@router.post("/stop/{run_id}")
def stop_vlm_training(run_id: str):
    run_dir = safe_id(RUNS_DIR, run_id)
    if not (run_dir / "config.json").is_file():
        raise HTTPException(404, "Run not found")
    processes.stop(run_dir)
    active = RUNS_DIR / "vlm_active_run.json"
    if active.exists():
        try:
            if json.loads(active.read_text()).get("run_id") == run_id:
                active.unlink(missing_ok=True)
        except (ValueError, OSError):
            pass
    return {"success": True}


@router.get("/runs")
def list_vlm_runs():
    runs = []
    if not RUNS_DIR.exists():
        return {"runs": []}

    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not d.name.startswith("vlm_"):
            continue
        config_file = d / "config.json"
        metrics_file = d / "metrics.json"

        config = {}
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
            except Exception:
                pass

        status = "unknown"
        metrics = {}
        current_step = 0
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
                status = metrics.get("status", "unknown")
                current_step = metrics.get("step", metrics.get("current_step", 0))
            except Exception:
                pass

        runs.append({
            "run_id": d.name,
            "process_running": processes.is_running(d),
            "stage": config.get("stage", "vlm_sft"),
            "status": status,
            "started_at": config.get("started_at", ""),
            "model_type": "vlm",
            "experiment_name": config.get("experiment_name", d.name),
            "config": config,
            "current_step": current_step,
            "total_steps": metrics.get("total_steps", config.get("max_steps", 0)),
            "checkpoints": [],
        })

    return {"runs": runs}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class VLMLoadRequest(BaseModel):
    model_path: str


class VLMGenerateRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    prompt: str
    image_base64: str = Field(max_length=20_000_000)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0, le=2)


@router.post("/load")
def load_vlm(req: VLMLoadRequest):
    """Load an image-text model supported by the installed Transformers."""
    with _model_lock:
        global _vlm_state
        try:
            python = vlm_python()
            if os.path.abspath(python) != os.path.abspath(sys.executable):
                from backend.services.vlm_inference import InferenceWorker
                previous = _vlm_state.get('worker')
                if previous:
                    previous.close()
                _vlm_state = {'model': None, 'processor': None, 'path': None}
                worker = InferenceWorker(python, PROJECT_ROOT)
                try:
                    result = worker.request('load', req.model_dump())
                except Exception:
                    worker.close()
                    raise
                _vlm_state = {'model': None, 'processor': None, 'path': req.model_path, 'worker': worker}
                return result
            import torch
            from transformers import AutoProcessor
            from backend.services.vlm_runtime import load_model, adapter_source

            source = str(Path(req.model_path).expanduser())
            if not Path(source).is_absolute() and (PROJECT_ROOT / source).exists():
                source = str((PROJECT_ROOT / source).resolve())
            processor_source = source
            if (Path(source) / 'adapter_config.json').is_file() and not any((Path(source) / name).is_file()
                for name in ('processor_config.json', 'preprocessor_config.json')):
                processor_source = adapter_source(source)
            processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
            precision = 'bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'no'
            model = load_model(source, {'mixed_precision': precision, 'tuning_method': 'full', 'use_flash_attention': False},
                               'cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            _vlm_state = {"model": model, "processor": processor, "path": req.model_path}
            return {"success": True}
        except Exception as e:
            raise HTTPException(500, f"Failed to load VLM: {e}")


@router.post("/unload")
def unload_vlm():
    with _model_lock:
        global _vlm_state
        if _vlm_state.get('worker'):
            _vlm_state['worker'].close()
        _vlm_state = {"model": None, "processor": None, "path": None}
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return {"success": True}


@router.get("/status")
def vlm_status():
    return {
        "loaded": _vlm_state["model"] is not None or bool(_vlm_state.get('worker') and _vlm_state['worker'].alive),
        "model_path": _vlm_state["path"],
    }


@router.post("/generate")
def generate_vlm(req: VLMGenerateRequest):
    """Generate a response for an image + text prompt."""
    with _model_lock:
        if _vlm_state.get('worker'):
            return _vlm_state['worker'].request('generate', req.model_dump())
        if _vlm_state["model"] is None:
            raise HTTPException(400, "No VLM loaded — call /api/vlm/load first")

        try:
            from PIL import Image
            import torch

            b64 = req.image_base64
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(b64, validate=True)
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as exc:
                raise HTTPException(400, 'Invalid base64 image') from exc

            model = _vlm_state["model"]
            processor = _vlm_state["processor"]

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": req.prompt},
                ],
            }]

            try:
                prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            except Exception:
                prompt_text = req.prompt

            inputs = processor(
                text=[prompt_text],
                images=[image],
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(model.device)

            with torch.inference_mode():
                settings = {'max_new_tokens': req.max_tokens, 'do_sample': req.temperature > 0}
                if req.temperature > 0:
                    settings['temperature'] = req.temperature
                output_ids = model.generate(**inputs, **settings)

            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            return {"text": text}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Inference failed: {e}")


def _resolve_existing_path(value: str, *, label: str) -> str:
    if not value:
        raise ValueError(f"{label} is required")

    p = Path(value).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()

    if not p.exists():
        raise ValueError(f"{label} does not exist: {p}")
    return str(p)


def _normalize_and_validate_vlm_config(config: dict) -> None:
    if len(config.get('gpu_ids', [])) > 1:
        raise ValueError('VLM currently supports a single selected GPU; distributed VLM training is not implemented')
    if config.get('min_pixels') and config.get('max_pixels') and config['min_pixels'] > config['max_pixels']:
        raise ValueError('min_pixels must not exceed max_pixels')
    if any(gpu < 0 for gpu in config.get('gpu_ids', [])):
        raise ValueError('gpu_ids must be non-negative')
    if config.get('max_steps', 0) <= 0:
        raise ValueError('max_steps must be positive')
    if config.get('stage') == 'vlm_grpo':
        if config.get('grpo_algorithm') == 'sdpo' and (config.get('sdpo_ema_rate') or
            config.get('sdpo_full_logit_distillation') or config.get('sdpo_distillation_topk')):
            raise ValueError('Native VLM SDPO currently supports selected-token distillation without EMA; full-logit/top-k/EMA transport is not implemented')
        if config.get('grpo_overlong_buffer', 0) > config['grpo_max_new_tokens']:
            raise ValueError('grpo_overlong_buffer must not exceed completion budget')
        if config.get('seq_len', 2048) <= config.get('grpo_max_new_tokens', 256):
            raise ValueError('seq_len must exceed grpo_max_new_tokens to leave room for image and prompt')
        if config.get('val_data_path'):
            raise ValueError('VLM GRPO validation is not implemented; use a separate evaluation run')
    if config.get("resume_from_checkpoint"):
        config["resume_from_checkpoint"] = _resolve_existing_path(config["resume_from_checkpoint"], label="checkpoint")
        if config.get('stage') == 'vlm_grpo':
            checkpoint = Path(config['resume_from_checkpoint'])
            try:
                manifest = json.loads((checkpoint / 'native_manifest.json').read_text())
                if not isinstance(manifest, dict) or manifest.get('engine') != 'homellm_native_v1' or not (checkpoint / 'native_state.pt').stat().st_size:
                    raise ValueError('Invalid native optimizer state')
            except (OSError, ValueError) as exc:
                raise ValueError('Resume requires a complete native VLM GRPO checkpoint; TRL/legacy checkpoints are incompatible') from exc
    dataset_path = config.get("data_path") or config.get("dataset_path")
    resolved_data = _resolve_existing_path(dataset_path, label="dataset_path")
    config["data_path"] = resolved_data
    config["dataset_path"] = resolved_data
    if not Path(resolved_data).is_file():
        raise ValueError('dataset_path must be a JSONL file')
    config['data_base_dir'] = _resolve_existing_path(config['data_base_dir'], label='image base directory') if config.get('data_base_dir') else str(Path(resolved_data).parent)
    if config.get('val_data_path'):
        config['val_data_path'] = _resolve_existing_path(config['val_data_path'], label='validation dataset')
    if config.get('assistant_only_loss') is None:
        config['assistant_only_loss'] = config.get('stage') != 'vlm_pretrain'
    if config.get('freeze_vision_tower') is None:
        config['freeze_vision_tower'] = config.get('stage') == 'vlm_pretrain'

    model_path = config.get("base_model_path") or config.get("model_path")
    if not model_path:
        raise ValueError("model_path/base_model_path is required")

    config["base_model_path"] = model_path
    config["model_name_or_path"] = model_path
    config["model_path"] = model_path
    local = Path(model_path).expanduser()
    if not local.is_absolute() and (PROJECT_ROOT / local).exists():
        local = PROJECT_ROOT / local
    if local.exists():
        config['base_model_path'] = config['model_name_or_path'] = config['model_path'] = str(local.resolve())
    config["lora_r"] = int(config.get("lora_r") or config.get("lora_rank") or 16)
    config["lora_rank"] = int(config.get("lora_rank") or config.get("lora_r") or 16)
    config["tuning_method"] = (
        config.get("tuning_method")
        or ("lora" if config.get("use_lora", True) else "full")
    )
    config['use_lora'] = config['tuning_method'] in ('lora', 'qlora')
    config["lr_schedule"] = config.get("lr_schedule") or config.get("lr_scheduler") or "cosine"
    config["seq_len"] = int(config.get("seq_len") or 2048)
    config["num_epochs"] = int(config.get("num_epochs") or config.get("epochs") or 1)


def _raise_if_exited_early(proc: subprocess.Popen, run_dir: Path) -> None:
    time.sleep(0.5)
    code = proc.poll()
    if code is None or code == 0:
        return

    stderr = run_dir / "stderr.log"
    detail = f"VLM process exited immediately with code {code}"
    if stderr.exists():
        tail = stderr.read_text(errors="ignore")[-2000:].strip()
        if tail:
            detail = f"{detail}: {tail}"

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            metrics["status"] = "error"
            metrics["error"] = detail
            metrics_path.write_text(json.dumps(metrics, indent=2))
        except Exception:
            pass

    raise HTTPException(500, detail)
