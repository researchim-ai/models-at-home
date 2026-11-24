"""Training endpoints: start/stop/list runs, continue training."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
import sys
from collections import deque
import re
from typing import Literal
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator, Field
from backend.paths import get_project_root, safe_id, atomic_json
from backend.services import processes, metrics as run_metrics

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
RUNS_DIR = PROJECT_ROOT / ".runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "out"


class TrainingRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    stage: Literal["pretrain", "continual_pretrain", "sft", "grpo"] = "pretrain"
    model_type: Literal['home', 'hf', 'home_moe', 'gpt2', 'blueprint'] = "home"
    model_blueprint: str = ''
    model_blueprint_spec: dict | None = None
    model_path: str = ""
    dataset_path: str = ""
    data_path: str = ""
    experiment_name: str = ""
    model_id: str = ""
    tokenizer_path: str = ""
    model_name_input: str = ""
    hidden_size: int = 512
    num_layers: int = 8
    n_heads: int = 8
    num_heads: int = 8
    intermediate_size: int = Field(default=0, ge=0)
    vocab_size: int = 50257
    dropout: float = Field(default=0.0, ge=0, lt=1)
    arch_preset: str = ""
    num_experts: int = 8
    num_experts_per_tok: int = 2
    expert_type: Literal['swiglu', 'mlp'] = "swiglu"
    optimizer: Literal['adamw', 'adamw_8bit', 'muon', 'magma_adamw'] = "adamw"
    batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = Field(default=0.1, ge=0)
    betas: str | tuple[float, float] = "0.9,0.95"
    resume_from_checkpoint: str = ""
    resumed_from_run: str = ""
    eps: float = Field(default=1e-8, gt=0)
    min_lr_ratio: float = Field(default=0.0, ge=0, le=1)
    training_mode: Literal["steps", "epochs"] = "steps"
    max_steps: int = 10000
    warmup_steps: int = 1000
    scheduler_resync_on_resume: bool = True
    lr_scheduler: str = "cosine"
    lr_schedule: str = ""
    mixed_precision: Literal['no', 'fp16', 'bf16'] = "bf16"
    fp16_pure: bool = False
    flash_attention: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    grad_checkpoint: bool = True
    liger_kernel: bool = True
    use_liger: bool = True
    liger_fused_ce: bool = True
    output_dir: str = "out"
    save_every: int = 100
    log_every: int = 10
    export_on_checkpoint: bool = True
    merge_lora: bool = True
    max_grad_norm: float = Field(default=1.0, ge=0)
    parallel_mode: Literal["single", "multi_gpu", "fsdp", "fsdp_offload", "deepspeed_zero2", "deepspeed_zero3", "deepspeed_zero3_offload"] = "single"
    gpu_ids: list[int] = []
    deepspeed_config: str = ""
    training_backend: Literal['models_at_home', 'models-at-home', 'unsloth'] = "models_at_home"
    # LoRA
    use_lora: bool = False
    tuning_method: Literal['full', 'lora', 'qlora'] = "full"
    lora_rank: int = 32
    lora_r: int = 32
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: list[str] = []
    # Pretrain / continual
    seq_len: int = 2048
    epochs: int = 1
    num_epochs: int = 1
    base_model_path: str = ""
    # SFT
    sft_chat_template: str = ""
    sft_data_format: Literal['chat', 'instruct'] = "chat"
    sft_max_seq_length: int = 2048
    sft_packing: bool = False
    sft_train_on_completions_only: bool = False
    # GRPO
    grpo_algorithm: str = "grpo"
    grpo_group_size: int = 8
    grpo_temperature: float = 0.7
    grpo_max_new_tokens: int = 512
    grpo_kl_weight: float = 0.04
    grpo_clip_eps_high: float = 0.2
    grpo_clip_eps_low: float = 0.2
    grpo_prompt_batch_size: int = 8
    grpo_train_batch_size: int = 2
    grpo_max_prompts: int = 1000
    grpo_learning_rate: float = 0.0
    grpo_max_steps: int = 0
    grpo_max_optim_steps: int = 0
    grpo_reward_fn: str = "exact_match"
    grpo_dataset_path: str = ""


    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, values):
        values = dict(values)
        if values.get('model_type') == 'checkpoint' and values.get('stage') in ('sft', 'continual_pretrain', 'grpo'):
            values['model_type'] = 'hf'
        for canonical, alias in (("n_heads", "num_heads"), ("lora_r", "lora_rank"),
                                 ("use_flash_attention", "flash_attention"),
                                 ("grad_checkpoint", "gradient_checkpointing"),
                                 ("use_liger", "liger_kernel"), ("num_epochs", "epochs"),
                                 ("lr_scheduler", "lr_schedule")):
            if canonical in values:
                values[alias] = values[canonical]
            elif alias in values:
                values[canonical] = values[alias]
        if values.get("stage") == "sft" and "sft_max_seq_length" in values:
            values["seq_len"] = values["sft_max_seq_length"]
        return values


@router.post("/start")
def start_training(req: TrainingRequest):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = safe_id(RUNS_DIR, run_id)
    config = req.model_dump()
    try:
        _normalize_and_validate_training_config(config)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    accel_config = None
    if req.parallel_mode != "single":
        if req.parallel_mode == 'multi_gpu' and req.gpu_ids and len(req.gpu_ids) < 2:
            raise HTTPException(400, 'Distributed training requires at least two selected GPUs')
        accel_config = _resolve_accel_config(req.parallel_mode)
        if not accel_config:
            raise HTTPException(400, f"Missing accelerate config for {req.parallel_mode}")
    config["run_id"] = run_id
    config["started_at"] = datetime.now().isoformat()
    config["ui_run_dir"] = str(run_dir)
    config["experiment_name"] = req.experiment_name.strip() or run_id
    base_output = Path(req.output_dir).expanduser()
    if not base_output.is_absolute():
        base_output = PROJECT_ROOT / base_output
    # Always allocate a fresh run directory, even when continuing an older run.
    config["output_dir"] = str((base_output / run_id).resolve())
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=False)
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "config.json", config)

    initial_metrics = {
        "status": "initializing",
        "step": 0,
        "loss": 0,
        "learning_rate": config["learning_rate"],
    }
    atomic_json(run_dir / "metrics.json", initial_metrics)

    env = os.environ.copy()

    if req.stage == "grpo":
        cmd = [sys.executable, str(Path(__file__).parent.parent / "services" / "grpo_worker.py"), "--config", str(run_dir / "config.json")]
    else:
        cmd = [
            sys.executable,
            str(Path(__file__).parent.parent / 'services' / 'text_worker.py'),
            "--config",
            str(run_dir / "config.json"),
            "--metrics",
            str(run_dir / "metrics.json"),
        ]

    if accel_config:
        launch = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file', accel_config,
                  '--mixed_precision', 'no' if req.fp16_pure and req.mixed_precision == 'fp16' else req.mixed_precision]
        if req.gpu_ids:
            launch += ['--num_processes', str(len(req.gpu_ids))]
        cmd = launch + cmd[1:]

    if req.gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in req.gpu_ids)

    proc = processes.launch(run_dir, cmd, cwd=PROJECT_ROOT, env=env)
    _raise_if_exited_early(proc, run_dir)

    active_run = {"run_id": run_id, "started_at": datetime.now().isoformat(), "config": config}
    atomic_json(RUNS_DIR / "active_run.json", active_run)

    return {"run_id": run_id}


@router.post("/stop/{run_id}")
def stop_training(run_id: str):
    run_dir = safe_id(RUNS_DIR, run_id)
    if not (run_dir / "config.json").is_file():
        raise HTTPException(404, "Run not found")
    processes.stop(run_dir)
    active_path = RUNS_DIR / "active_run.json"
    if active_path.exists():
        try:
            if json.loads(active_path.read_text()).get("run_id") == run_id:
                active_path.unlink(missing_ok=True)
        except (ValueError, OSError):
            pass

    return {"success": True}


@router.get("/runs")
def list_runs():
    runs = []
    if not RUNS_DIR.exists():
        return {"runs": []}

    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not (d / "config.json").is_file():
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
        current_step = 0
        total_steps = config.get("max_steps", 0)

        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
                status = metrics.get("status", "unknown")
                current_step = metrics.get("step", metrics.get("current_step", 0))
                total_steps = metrics.get("total_steps", total_steps)
            except Exception:
                pass

        checkpoints = []
        out_dir = _artifact_dir(config, d.name)
        if out_dir.exists():
            checkpoints = sorted([
                p.name for p in out_dir.iterdir()
                if p.is_dir() and p.name.startswith("checkpoint")
            ])

        runs.append({
            "run_id": d.name,
            "process_running": processes.is_running(d),
            "stage": config.get("stage", "unknown"),
            "status": status,
            "started_at": config.get("started_at") or datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
            "model_type": config.get("model_type", ""),
            "experiment_name": config.get("experiment_name", d.name),
            "config": config,
            "current_step": current_step,
            "total_steps": total_steps,
            "checkpoints": checkpoints,
            "final_model_path": str(out_dir / "final_model") if (out_dir / "final_model").is_dir() else None,
        })

    return {"runs": runs}


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    run_dir = safe_id(RUNS_DIR, run_id)
    if not run_dir.exists():
        raise HTTPException(404, "Run not found")

    config = {}
    config_file = run_dir / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text())

    metrics = {}
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text())
        except Exception:
            pass

    return {
        "run_id": run_id,
        "process_running": processes.is_running(run_dir),
        "stage": config.get("stage", "unknown"),
        "status": metrics.get("status", "unknown"),
        "config": config,
        "current_step": metrics.get("step", metrics.get("current_step", 0)),
        "total_steps": metrics.get("total_steps", config.get("max_steps", 0)),
    }


@router.get("/runs/{run_id}/metrics")
def get_run_metrics(run_id: str, limit: int = 500):
    if not 1 <= limit <= 5000:
        raise HTTPException(400, "limit must be between 1 and 5000")
    run_dir = safe_id(RUNS_DIR, run_id)
    if not (run_dir / "config.json").is_file():
        raise HTTPException(404, "Run not found")
    data = run_metrics.read(run_dir / "metrics.json")
    return {"metrics": run_metrics.snapshot(data) if data is not None else None,
            "history": run_metrics.history(data, limit) if data is not None else []}


@router.delete("/runs/{run_id}")
def delete_run(run_id: str):
    import shutil
    run_dir = safe_id(RUNS_DIR, run_id)
    if processes.is_running(run_dir):
        raise HTTPException(409, "Stop the run before deleting it")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return {"success": True}


@router.get("/runs/{run_id}/logs")
def run_logs(run_id: str, lines: int = 500):
    """Return the last N lines of stdout/stderr logs for a run."""
    run_dir = safe_id(RUNS_DIR, run_id)
    if not 1 <= lines <= 10000:
        raise HTTPException(400, "lines must be between 1 and 10000")
    if not run_dir.exists():
        raise HTTPException(404, "Run not found")

    def _tail(path: Path, n: int) -> str:
        if not path.exists():
            return ""
        try:
            with path.open(errors="ignore") as log:
                return "".join(deque(log, maxlen=n)).rstrip("\n")
        except Exception:
            return ""

    return {
        "run_id": run_id,
        "stdout": _tail(run_dir / "stdout.log", lines),
        "stderr": _tail(run_dir / "stderr.log", lines),
    }


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_steps: int | None = Field(default=None, gt=0)


@router.post("/runs/{run_id}/continue")
def continue_training(run_id: str, options: ResumeRequest | None = None):
    run_dir = safe_id(RUNS_DIR, run_id)
    config_file = run_dir / "config.json"
    if not config_file.exists():
        raise HTTPException(404, "Run config not found")

    if processes.is_running(run_dir):
        raise HTTPException(409, "Run is still running")
    config = json.loads(config_file.read_text())
    if config.get('stage') == 'sft' and config.get('training_backend') == 'unsloth':
        raise HTTPException(400, 'This Unsloth runtime does not support checkpoint resume')
    if config.get("stage") == "grpo":
        raise HTTPException(400, "This GRPO runtime does not support checkpoint resume")
    if config.get('stage') == 'vlm_grpo' and config.get('vlm_worker_version') != 3:
        raise HTTPException(400, 'Legacy reward-filtered SFT or TRL checkpoints cannot resume as native VLM GRPO; start a new policy run')
    if config.get('stage') == 'vla_sft' and config.get('vla_worker_version') != 1:
        raise HTTPException(400, 'Unsupported VLA checkpoint version')
    if config.get('stage') == 'world_model' and config.get('world_worker_version') != 1:
        raise HTTPException(400, 'Unsupported world model checkpoint version')
    if config.get('stage') == 'world_foundation':
        raise HTTPException(400, 'Foundation generation/download jobs are not training checkpoints')
    artifact_dir = _artifact_dir(config, run_id)
    checkpoints = [p for p in artifact_dir.glob("checkpoint*") if p.is_dir()]
    if config.get('stage') == 'vla_sft':
        from backend.routes.vla import complete_checkpoint
        checkpoints = [p for p in checkpoints if complete_checkpoint(p)]
    if config.get('stage') == 'world_model':
        from backend.routes.world_models import complete_checkpoint
        checkpoints = [p for p in checkpoints if complete_checkpoint(p)]
    if config.get('stage') == 'vlm_grpo':
        checkpoints = [p for p in checkpoints if (p / 'native_manifest.json').is_file() and (p / 'native_state.pt').is_file()]
    if not checkpoints:
        raise HTTPException(400, "No checkpoint available to resume")
    def checkpoint_step(path):
        match = re.search(r"(?:step|-)(\d+)$", path.name)
        return (int(match.group(1)) if match else -1, path.stat().st_mtime)
    checkpoint = max(checkpoints, key=checkpoint_step)
    if options and options.max_steps is not None:
        config["max_steps"] = options.max_steps
        config["training_mode"] = "steps"
    if config.get("training_mode", "steps") == "steps" and checkpoint_step(checkpoint)[0] >= config.get("max_steps", 0):
        raise HTTPException(400, "The checkpoint has reached max_steps; provide a larger max_steps to extend training")
    config["resume_from_checkpoint"] = str(checkpoint.resolve())
    config["resumed_from_run"] = run_id
    config["output_dir"] = str(artifact_dir.parent)
    if config.get('stage') == 'vla_sft':
        from backend.routes.vla import VLATrainingRequest, start
        return start(VLATrainingRequest(**config))
    if config.get('stage') == 'world_model':
        from backend.routes.world_models import WorldRequest, start
        return start(WorldRequest(**config))
    if str(config.get("stage", "")).startswith("vlm_"):
        from backend.routes.vlm import VLMTrainingRequest, start_vlm_training
        return start_vlm_training(VLMTrainingRequest(**config))
    req = TrainingRequest(**{k: v for k, v in config.items() if k in TrainingRequest.model_fields})
    return start_training(req)


@router.get("/runs/{run_id}/grpo-samples")
def grpo_samples(run_id: str, limit: int = 20):
    """Return recent GRPO sample generations from .runs/{id}/samples/ or samples.jsonl."""
    run_dir = safe_id(RUNS_DIR, run_id)
    if not run_dir.exists():
        raise HTTPException(404, "Run not found")

    if not 1 <= limit <= 1000:
        raise HTTPException(400, "limit must be between 1 and 1000")
    samples: list = []
    samples_file = run_dir / "samples.jsonl"
    if samples_file.exists():
        lines = samples_file.read_text().strip().split("\n")
        for line in lines[-limit:]:
            try:
                samples.append(json.loads(line))
            except Exception:
                continue

    samples_dir = run_dir / "samples"
    if samples_dir.exists() and not samples:
        for f in sorted(samples_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                samples.append(json.loads(f.read_text()))
            except Exception:
                continue

    return {"samples": samples}


def _artifact_dir(config: dict, run_id: str) -> Path:
    output = Path(config.get("output_dir") or "out").expanduser()
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    # Legacy workers wrote directly into output_dir. Prefer actual artifacts there.
    if (output / "final_model").exists() or any(output.glob("checkpoint*")):
        return output
    legacy = output / (config.get("experiment_name") or run_id)
    return legacy if legacy.exists() else output


def _resolve_accel_config(mode: str) -> str | None:
    configs_dir = PROJECT_ROOT / "configs"
    mapping = {
        "multi_gpu": "accelerate_multi_gpu.yaml",
        "fsdp": "accelerate_fsdp.yaml",
        "fsdp_offload": "accelerate_fsdp_offload.yaml",
        "deepspeed_zero2": "accelerate_deepspeed_zero2.yaml",
        "deepspeed_zero3": "accelerate_deepspeed_zero3.yaml",
        "deepspeed_zero3_offload": "accelerate_deepspeed_zero3_offload.yaml",
    }
    filename = mapping.get(mode)
    if filename:
        path = configs_dir / filename
        if path.exists():
            return str(path)
    return None


def _resolve_existing_path(value: str, *, label: str) -> str:
    """Resolve a user-provided path against the models-at-home root."""
    if not value:
        raise ValueError(f"{label} is required")

    p = Path(value).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()

    if not p.exists():
        raise ValueError(f"{label} does not exist: {p}")
    return str(p)


def _normalize_and_validate_training_config(config: dict) -> None:
    """Make UI/API config compatible with the legacy training workers."""
    stage = config.get("stage", "pretrain")
    if config.get('model_type') == 'blueprint' and stage == 'pretrain':
        from backend.services.blueprints import validate
        spec = config.get('model_blueprint_spec')
        if spec is None:
            path = _resolve_existing_path(config.get('model_blueprint', ''), label='blueprint')
            config['model_blueprint'] = path
            spec = json.loads(Path(path).read_text())
        spec = validate(spec)
        if config['seq_len'] > spec['max_position_embeddings']:
            raise ValueError('seq_len exceeds the blueprint context length')
        config['model_blueprint_spec'] = spec
        config['hidden_size'] = spec['hidden_size']
        config['vocab_size'] = spec['vocab_size']
    if len(set(config.get('gpu_ids', []))) != len(config.get('gpu_ids', [])):
        raise ValueError('gpu_ids must be unique')
    if config.get('training_backend') == 'unsloth':
        if stage not in ('sft', 'grpo'):
            raise ValueError('Unsloth is only supported for SFT and GRPO')
        if stage == 'sft' and config.get('resume_from_checkpoint'):
            raise ValueError('This Unsloth runtime does not support checkpoint resume')

    if stage == "pretrain" and config.get("model_type") == "hf":
        source = config.get("model_id") or config.get("base_model_path") or config.get("model_path")
        if not source:
            raise ValueError("HF pretrain requires a model ID or config directory")
        config["model_id"] = source
        config["tokenizer_path"] = config.get("tokenizer_path") or source
        # HFAdapter uses a non-empty base_model_path to load pretrained weights.
        config["base_model_path"] = ""
    elif stage == "pretrain":
        config["model_id"] = config.get("model_id") or "gpt2"
    if config.get("sft_chat_template"):
        config["chat_template"] = config["sft_chat_template"]
    if stage == 'sft':
        config['sft_columns'] = {'format': config['sft_data_format']}
        if config.get('training_backend') == 'unsloth' and (config.get('sft_packing') or config.get('sft_train_on_completions_only')):
            raise ValueError('This Unsloth runtime does not support the selected SFT packing/masking controls')
    model_source = config.get('base_model_path') or config.get('model_path')
    if stage != 'pretrain' and model_source:
        source = Path(model_source).expanduser()
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        saved_config = source / 'config.json'
        if saved_config.is_file() and json.loads(saved_config.read_text()).get('model_type') == 'homellm_blueprint':
            if stage == 'grpo' or config.get('training_backend') == 'unsloth' or config.get('tuning_method') == 'qlora':
                raise ValueError('Blueprint models support standard SFT/continual training; GRPO, Unsloth and QLoRA are not supported')
    if config.get('model_type') == 'home_moe' and not 1 <= config['num_experts_per_tok'] <= config['num_experts']:
        raise ValueError('num_experts_per_tok must be between 1 and num_experts')

    for key in ("batch_size", "gradient_accumulation", "hidden_size", "num_layers", "n_heads", "vocab_size", "seq_len", "num_epochs", "save_every", "log_every", "lora_r"):
        if config.get(key, 1) <= 0:
            raise ValueError(f"{key} must be positive")
    if config.get("learning_rate", 0) <= 0 or config.get("max_steps", 0) < 0 or config.get("warmup_steps", 0) < 0:
        raise ValueError("Invalid learning rate or step count")
    if config.get("training_mode") == "steps" and config.get("max_steps", 0) <= 0:
        raise ValueError("max_steps must be positive in steps mode")
    if stage == "pretrain" and config.get("model_type") in ("home", "gpt2", "home_moe"):
        if config["hidden_size"] % config["n_heads"]:
            raise ValueError("hidden_size must be divisible by n_heads")
        if config.get('model_type') != 'gpt2' and (config['hidden_size'] // config['n_heads']) % 2:
            raise ValueError('Rotary attention requires an even head dimension')
    if any(g < 0 for g in config.get("gpu_ids", [])):
        raise ValueError("gpu_ids must be non-negative")
    if stage == "grpo" and config.get("resume_from_checkpoint"):
        raise ValueError("This GRPO runtime does not support checkpoint resume")
    if config.get("resume_from_checkpoint"):
        config["resume_from_checkpoint"] = _resolve_existing_path(config["resume_from_checkpoint"], label="checkpoint")

    if stage == "grpo" and config.get("training_backend") != "unsloth" and config.get("grpo_group_size", 8) < 8:
        raise ValueError("This GRPO runtime requires grpo_group_size >= 8")

    # React uses dataset_path; legacy workers use data_path.
    dataset_path = config.get("data_path") or config.get("dataset_path")
    if dataset_path:
        resolved = _resolve_existing_path(dataset_path, label="dataset_path")
        config["data_path"] = resolved
        config["dataset_path"] = resolved

    # React often writes model_path; legacy fine-tune/RL workers expect base_model_path.
    model_path = config.get("base_model_path") or config.get("model_path")
    if model_path and stage in ("sft", "continual_pretrain", "grpo"):
        config["base_model_path"] = model_path
        config["model_path"] = model_path

    if stage in ("pretrain", "sft", "continual_pretrain") and not config.get("data_path"):
        raise ValueError(f"{stage} requires dataset_path")

    if stage in ("sft", "continual_pretrain", "grpo") and not config.get("base_model_path"):
        raise ValueError(f"{stage} requires base_model_path/model_path")

    if stage in ("sft", "continual_pretrain"):
        config["base_model_path"] = _resolve_existing_path(config["base_model_path"], label="base model (download it in Models first)")
        config["model_path"] = config["base_model_path"]
        model_config = Path(config['base_model_path']) / 'config.json'
        if model_config.is_file():
            saved = json.loads(model_config.read_text())
            capacity = saved.get('max_position_embeddings') or saved.get('n_positions')
            if isinstance(capacity, int) and config['seq_len'] > capacity:
                raise ValueError(f'seq_len exceeds the base model context length ({capacity})')

    if stage == "grpo":
        grpo_path = config.get("grpo_dataset_path") or config.get("data_path")
        if not grpo_path:
            raise ValueError("grpo requires dataset_path")
        if grpo_path:
            resolved = _resolve_existing_path(grpo_path, label="grpo_dataset_path")
            config["grpo_dataset_path"] = resolved
            config["data_path"] = resolved

    # Worker expects seq_len under this exact name.
    config["seq_len"] = int(config.get("seq_len") or config.get("sft_max_seq_length") or 2048)
    config["epochs"] = int(config.get("epochs") or config.get("num_epochs") or 1)
    config["num_epochs"] = int(config.get("num_epochs") or config.get("epochs") or 1)
    config["lr_schedule"] = config.get("lr_schedule") or config.get("lr_scheduler") or "cosine"
    config["lr_scheduler"] = config["lr_schedule"]
    if config['lr_scheduler'] not in ('cosine', 'linear', 'constant_with_warmup', 'cosine_with_restarts', 'constant'):
        raise ValueError('Unsupported lr_scheduler')
    config["hidden_size"] = int(config.get("hidden_size") or 512)
    config["num_layers"] = int(config.get("num_layers") or 8)
    config["n_heads"] = int(config.get("n_heads") or config.get("num_heads") or 8)
    config["num_heads"] = int(config.get("num_heads") or config.get("n_heads") or 8)
    config["use_flash_attention"] = bool(config.get("use_flash_attention", config.get("flash_attention", True)))
    config["use_liger"] = bool(config.get("use_liger", config.get("liger_kernel", False)))
    config["grad_checkpoint"] = bool(config.get("grad_checkpoint", config.get("gradient_checkpointing", False)))
    config["lora_r"] = int(config.get("lora_r") or config.get("lora_rank") or 32)
    config["lora_rank"] = int(config.get("lora_rank") or config.get("lora_r") or 32)
    if config.get("use_lora") and config.get("tuning_method") in ("", "full", None):
        config["tuning_method"] = "lora"
    if config.get("tuning_method") in ("lora", "qlora"):
        config["use_lora"] = True
    config["training_backend"] = str(config.get("training_backend") or "models-at-home").replace("_", "-")

    # Worker expects betas as a [beta1, beta2] list; the UI sends "0.9,0.95".
    betas = config.get("betas")
    try:
        parts = [float(x.strip()) for x in betas.split(",")] if isinstance(betas, str) else list(betas)
        if len(parts) != 2 or any(not 0 <= x < 1 for x in parts):
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("betas must contain two numbers in [0, 1)")
    config["betas"] = parts

    config["weight_decay"] = float(config.get("weight_decay", 0.1))
    config["min_lr_ratio"] = float(config.get("min_lr_ratio", 0.0) or 0.0)
    config["liger_fused_ce"] = bool(config.get("liger_fused_ce", True))
    config["fp16_pure"] = bool(config.get("fp16_pure", False))

    # Epochs vs steps: the worker uses max_steps only when it is truthy, and
    # otherwise derives the schedule from epochs. Honour the UI's explicit mode.
    if config.get("training_mode") == "epochs":
        config["max_steps"] = 0

    if stage == "grpo":
        config["grpo_learning_rate"] = float(config.get("grpo_learning_rate") or config.get("learning_rate") or 5e-6)
        config["learning_rate"] = config["grpo_learning_rate"]
        grpo_steps = int(
            config.get("grpo_max_optim_steps")
            or config.get("grpo_max_steps")
            or config.get("max_steps")
            or 0
        )
        if grpo_steps > 0:
            config["grpo_max_optim_steps"] = grpo_steps
            config["grpo_max_steps"] = grpo_steps
            config["max_steps"] = grpo_steps
        if config.get("grpo_reward_fn") and not config.get("grpo_reward_rules"):
            reward_fn = str(config.get("grpo_reward_fn"))
            config["grpo_reward_rules"] = [{"type": reward_fn}]


def _raise_if_exited_early(proc: subprocess.Popen, run_dir: Path) -> None:
    """Catch immediate subprocess failures before telling the UI the run started."""
    code = None
    for _ in range(10):
        time.sleep(0.2)
        code = proc.poll()
        if code is not None:
            break
    if code is None or code == 0:
        return

    stderr = run_dir / "stderr.log"
    detail = f"Training process exited immediately with code {code}"
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
