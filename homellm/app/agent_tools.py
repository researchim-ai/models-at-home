"""Controlled tools for Agent Studio."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / ".runs"
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "out"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RUNS_DIR.mkdir(exist_ok=True)


TEXT_TRAINING_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen35_9b_unsloth_sft": {
        "stage": "sft",
        "training_backend": "unsloth",
        "tuning_method": "qlora",
        "base_model_path": "Qwen/Qwen3.5-9B",
        "output_dir": "out/agent/text/qwen35_9b_unsloth_sft",
        "batch_size": 1,
        "gradient_accumulation": 16,
        "learning_rate": 2e-4,
        "num_epochs": 2,
        "seq_len": 4096,
        "warmup_steps": 50,
        "save_every": 200,
        "log_every": 10,
        "lora_r": 64,
        "lora_alpha": 128,
        "assistant_only_loss": True,
        "description": "Базовый QLoRA/Unsloth SFT под Qwen 3.5 9B для одной сильной GPU.",
    },
    "qwen35_9b_full_sft": {
        "stage": "sft",
        "training_backend": "models-at-home",
        "tuning_method": "full",
        "base_model_path": "Qwen/Qwen3.5-9B",
        "output_dir": "out/agent/text/qwen35_9b_full_sft",
        "batch_size": 1,
        "gradient_accumulation": 16,
        "learning_rate": 1e-5,
        "num_epochs": 1,
        "seq_len": 4096,
        "warmup_steps": 100,
        "save_every": 100,
        "log_every": 10,
        "gradient_checkpointing": True,
        "description": "Полный SFT для Qwen 3.5 9B, если хватает VRAM/распределенного режима.",
    },
    "small_pretrain_bootstrap": {
        "stage": "pretrain",
        "training_backend": "models-at-home",
        "output_dir": "out/agent/text/small_pretrain_bootstrap",
        "batch_size": 8,
        "gradient_accumulation": 8,
        "learning_rate": 3e-4,
        "num_epochs": 1,
        "seq_len": 2048,
        "warmup_steps": 100,
        "save_every": 250,
        "log_every": 10,
        "description": "Легкий стартовый претрейн/continued-pretrain по текстовому корпусу.",
    },
}

VLM_TRAINING_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen35_2b_vlm_sft": {
        "stage": "vlm_sft",
        "base_model_path": "Qwen/Qwen3.5-2B",
        "output_dir": "out/agent/vlm/qwen35_2b_vlm_sft",
        "tuning_method": "qlora",
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 2e-5,
        "num_epochs": 2,
        "seq_len": 3072,
        "warmup_steps": 100,
        "assistant_only_loss": True,
        "gradient_checkpointing": True,
        "freeze_vision_tower": True,
        "description": "Практичный VLM SFT preset под Qwen 3.5 2B.",
    },
    "qwen35_08b_vlm_pretrain": {
        "stage": "vlm_pretrain",
        "base_model_path": "Qwen/Qwen3.5-0.8B",
        "output_dir": "out/agent/vlm/qwen35_08b_vlm_pretrain",
        "tuning_method": "qlora",
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 1e-5,
        "num_epochs": 1,
        "seq_len": 1536,
        "warmup_steps": 50,
        "assistant_only_loss": False,
        "freeze_vision_tower": True,
        "description": "Небольшой multimodal alignment/pretrain preset.",
    },
}

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "list_training_capabilities",
        "description": "Показывает доступные типы обучения и краткие рекомендации по выбору пайплайна.",
        "arguments": {},
    },
    {
        "name": "list_local_models",
        "description": "Показывает локальные базовые модели, GGUF-файлы, адаптеры и результаты тренировок.",
        "arguments": {},
    },
    {
        "name": "list_datasets",
        "description": "Показывает доступные локальные датасеты из папки datasets/.",
        "arguments": {},
    },
    {
        "name": "preview_dataset",
        "description": "Читает несколько примеров из локального JSON/JSONL/TXT датасета.",
        "arguments": {
            "path": "str, относительный путь внутри datasets/ или абсолютный путь",
            "limit": "int, количество примеров",
        },
    },
    {
        "name": "get_training_presets",
        "description": "Возвращает готовые пресеты для text/VLM обучения.",
        "arguments": {
            "domain": "str: text | vlm | all",
        },
    },
    {
        "name": "start_text_training",
        "description": "Запускает text training через существующий trainer_worker.py.",
        "arguments": {
            "config": "dict с training config. Можно взять пресет и доопределить data_path/base_model_path/output_dir.",
        },
    },
    {
        "name": "start_vlm_training",
        "description": "Запускает VLM pretrain/SFT/GRPO через существующие воркеры.",
        "arguments": {
            "config": "dict с stage=vlm_pretrain|vlm_sft|vlm_grpo и остальными параметрами.",
        },
    },
    {
        "name": "get_run_status",
        "description": "Показывает состояние run по metrics.json и process PID.",
        "arguments": {
            "run_id": "str",
        },
    },
    {
        "name": "read_run_logs",
        "description": "Читает хвост stdout/stderr логов run.",
        "arguments": {
            "run_id": "str",
            "max_lines": "int",
        },
    },
    {
        "name": "stop_run",
        "description": "Останавливает запущенный train/job по PID.",
        "arguments": {
            "run_id": "str",
        },
    },
]


def get_tool_specs() -> List[Dict[str, Any]]:
    return TOOL_SPECS


def list_training_capabilities() -> Dict[str, Any]:
    return {
        "text_training": {
            "stages": ["pretrain", "continual_pretrain", "sft"],
            "entrypoint": "python -m homellm.app.trainer_worker",
            "supports": ["full fine-tune", "LoRA/QLoRA", "Unsloth SFT", "multi-GPU/DeepSpeed/FSDP"],
        },
        "vlm_training": {
            "stages": ["vlm_pretrain", "vlm_sft", "vlm_grpo"],
            "entrypoints": [
                "python -m homellm.training.vlm_pretrain",
                "python -m homellm.training.vlm_sft",
                "python -m homellm.training.vlm_grpo",
            ],
            "supports": ["image-text SFT", "continued multimodal pretrain", "GRPO"],
        },
        "notes": [
            "Для text SFT safest путь: trainer_worker с config.json.",
            "Для agentic workflows лучше запускать обучение отдельным процессом и мониторить через .runs/.",
            "Для локального агента inference идет через llama.cpp, а train через существующие Python workers.",
        ],
    }


def _resolve_local_path(path_value: str, default_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (default_root / candidate).resolve()


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"_error": str(exc)}


def _tail_lines(path: Path, max_lines: int = 80) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def list_local_models() -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []

    for gguf in MODELS_DIR.rglob("*.gguf"):
        items.append(
            {
                "kind": "gguf",
                "name": gguf.name,
                "path": _safe_relpath(gguf),
                "size_gb": round(gguf.stat().st_size / (1024**3), 2),
                "modified_at": datetime.fromtimestamp(gguf.stat().st_mtime).isoformat(),
            }
        )

    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        config_path = model_dir / "config.json"
        adapter_config = model_dir / "adapter_config.json"
        if not config_path.exists() and not adapter_config.exists():
            continue
        meta = _load_json_if_exists(config_path if config_path.exists() else adapter_config)
        items.append(
            {
                "kind": "transformers" if config_path.exists() else "adapter",
                "name": model_dir.name,
                "path": _safe_relpath(model_dir),
                "model_type": meta.get("model_type"),
                "base_model": meta.get("base_model_name_or_path"),
                "modified_at": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
            }
        )

    for output_model in OUTPUT_DIR.rglob("final_model"):
        meta = _load_json_if_exists(output_model / "config.json")
        items.append(
            {
                "kind": "trained_model",
                "name": output_model.parent.name,
                "path": _safe_relpath(output_model),
                "model_type": meta.get("model_type"),
                "modified_at": datetime.fromtimestamp(output_model.stat().st_mtime).isoformat(),
            }
        )

    items.sort(key=lambda item: item.get("modified_at", ""), reverse=True)
    return {"models": items[:200], "count": len(items)}


def list_datasets() -> Dict[str, Any]:
    datasets: List[Dict[str, Any]] = []
    if not DATASETS_DIR.exists():
        return {"datasets": [], "count": 0}

    patterns = ["*.jsonl", "*.json", "*.csv", "*.txt", "*.parquet"]
    for pattern in patterns:
        for path in DATASETS_DIR.rglob(pattern):
            datasets.append(
                {
                    "name": path.name,
                    "path": _safe_relpath(path),
                    "size_mb": round(path.stat().st_size / (1024**2), 2),
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                }
            )
    datasets.sort(key=lambda item: item.get("modified_at", ""), reverse=True)
    return {"datasets": datasets[:200], "count": len(datasets)}


def preview_dataset(path: str, limit: int = 3) -> Dict[str, Any]:
    dataset_path = _resolve_local_path(path, DATASETS_DIR)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    limit = max(1, min(int(limit), 10))

    if suffix == ".jsonl":
        examples: List[Any] = []
        with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    examples.append(line)
        return {"path": _safe_relpath(dataset_path), "format": "jsonl", "examples": examples}

    if suffix == ".json":
        with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {"path": _safe_relpath(dataset_path), "format": "json", "examples": data[:limit]}
        return {"path": _safe_relpath(dataset_path), "format": "json", "example": data}

    with open(dataset_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n") for _, line in zip(range(limit), f)]
    return {"path": _safe_relpath(dataset_path), "format": suffix.lstrip(".") or "text", "lines": lines}


def get_training_presets(domain: str = "all") -> Dict[str, Any]:
    domain = str(domain or "all").lower()
    result: Dict[str, Any] = {}
    if domain in {"text", "all"}:
        result["text"] = TEXT_TRAINING_PRESETS
    if domain in {"vlm", "all"}:
        result["vlm"] = VLM_TRAINING_PRESETS
    return result


def _write_initial_metrics(metrics_path: Path, stage: str) -> None:
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "starting",
                "stage": stage,
                "current_step": 0,
                "started_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def _spawn_run(run_id: str, cmd: List[str], env: Dict[str, str], stage: str, config: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command_path = run_dir / "command.txt"
    pid_path = run_dir / "pid"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    _write_initial_metrics(metrics_path, stage=stage)
    with open(command_path, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd))

    stdout_file = open(stdout_path, "w", encoding="utf-8")
    stderr_file = open(stderr_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
            env=env,
        )
    finally:
        stdout_file.close()
        stderr_file.close()

    with open(pid_path, "w", encoding="utf-8") as f:
        f.write(str(process.pid))

    return {
        "run_id": run_id,
        "pid": process.pid,
        "run_dir": _safe_relpath(run_dir),
        "config_path": _safe_relpath(config_path),
        "metrics_path": _safe_relpath(metrics_path),
        "stdout_path": _safe_relpath(stdout_path),
        "stderr_path": _safe_relpath(stderr_path),
        "command": cmd,
    }


def start_text_training(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config or {})
    stage = str(cfg.get("stage", "sft"))
    if stage not in {"pretrain", "continual_pretrain", "sft"}:
        raise ValueError(f"Unsupported text stage: {stage}")

    if not cfg.get("output_dir"):
        cfg["output_dir"] = f"out/agent/text/{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_id = f"agent_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve() / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(run_output_dir)

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)
    _write_initial_metrics(metrics_path, stage=stage)

    distributed_mode = cfg.get("distributed_mode", "default")
    config_file = cfg.get("config_file")
    num_gpus = int(cfg.get("num_gpus", 1) or 1)
    if distributed_mode != "default" and config_file:
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            str(config_file),
            "--num_processes",
            str(num_gpus),
            "-m",
            "homellm.app.trainer_worker",
            "--config",
            str(config_path),
            "--metrics",
            str(metrics_path),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "homellm.app.trainer_worker",
            "--config",
            str(config_path),
            "--metrics",
            str(metrics_path),
        ]

    env = os.environ.copy()
    gpu_ids = cfg.get("gpu_ids") or []
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    result = _spawn_run(run_id=run_id, cmd=cmd, env=env, stage=stage, config=cfg)
    result["output_dir"] = _safe_relpath(run_output_dir)
    return result


def start_vlm_training(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config or {})
    stage = str(cfg.get("stage", "vlm_sft"))
    stage_to_module = {
        "vlm_pretrain": "homellm.training.vlm_pretrain",
        "vlm_sft": "homellm.training.vlm_sft",
        "vlm_grpo": "homellm.training.vlm_grpo",
    }
    if stage not in stage_to_module:
        raise ValueError(f"Unsupported VLM stage: {stage}")

    if not cfg.get("output_dir"):
        cfg["output_dir"] = f"out/agent/vlm/{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_id = f"agent_vlm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve() / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(run_output_dir)

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)
    _write_initial_metrics(metrics_path, stage=stage)

    distributed_mode = cfg.get("distributed_mode", "default")
    config_file = cfg.get("config_file")
    num_gpus = int(cfg.get("num_gpus", 1) or 1)
    if distributed_mode != "default" and config_file:
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            str(config_file),
            "--num_processes",
            str(num_gpus),
            "-m",
            stage_to_module[stage],
            "--config",
            str(config_path),
            "--metrics",
            str(metrics_path),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            stage_to_module[stage],
            "--config",
            str(config_path),
            "--metrics",
            str(metrics_path),
        ]

    env = os.environ.copy()
    gpu_ids = cfg.get("gpu_ids") or []
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    result = _spawn_run(run_id=run_id, cmd=cmd, env=env, stage=stage, config=cfg)
    result["output_dir"] = _safe_relpath(run_output_dir)
    return result


def get_run_status(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")

    metrics = _load_json_if_exists(run_dir / "metrics.json")
    config = _load_json_if_exists(run_dir / "config.json")
    pid_path = run_dir / "pid"
    pid = None
    is_running = False
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            is_running = True
        except ProcessLookupError:
            is_running = False
        except Exception:
            is_running = False

    return {
        "run_id": run_id,
        "run_dir": _safe_relpath(run_dir),
        "pid": pid,
        "is_running": is_running,
        "metrics": metrics,
        "config_summary": {
            "stage": config.get("stage"),
            "base_model_path": config.get("base_model_path"),
            "output_dir": config.get("output_dir"),
            "training_backend": config.get("training_backend"),
            "tuning_method": config.get("tuning_method"),
        },
    }


def read_run_logs(run_id: str, max_lines: int = 80) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    max_lines = max(10, min(int(max_lines), 300))
    return {
        "run_id": run_id,
        "stdout": _tail_lines(run_dir / "stdout.log", max_lines=max_lines),
        "stderr": _tail_lines(run_dir / "stderr.log", max_lines=max_lines),
    }


def stop_run(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    pid_path = run_dir / "pid"
    if not pid_path.exists():
        return {"run_id": run_id, "stopped": False, "reason": "pid file missing"}
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return {"run_id": run_id, "stopped": True, "signal": "SIGTERM"}
    except Exception as exc:
        return {"run_id": run_id, "stopped": False, "reason": str(exc)}


def execute_tool(tool_name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
    args = arguments or {}
    registry = {
        "list_training_capabilities": lambda: list_training_capabilities(),
        "list_local_models": lambda: list_local_models(),
        "list_datasets": lambda: list_datasets(),
        "preview_dataset": lambda: preview_dataset(path=str(args.get("path", "")), limit=int(args.get("limit", 3))),
        "get_training_presets": lambda: get_training_presets(domain=str(args.get("domain", "all"))),
        "start_text_training": lambda: start_text_training(config=dict(args.get("config") or {})),
        "start_vlm_training": lambda: start_vlm_training(config=dict(args.get("config") or {})),
        "get_run_status": lambda: get_run_status(run_id=str(args.get("run_id", ""))),
        "read_run_logs": lambda: read_run_logs(run_id=str(args.get("run_id", "")), max_lines=int(args.get("max_lines", 80))),
        "stop_run": lambda: stop_run(run_id=str(args.get("run_id", ""))),
    }
    if tool_name not in registry:
        raise ValueError(f"Unknown tool: {tool_name}")
    return registry[tool_name]()
