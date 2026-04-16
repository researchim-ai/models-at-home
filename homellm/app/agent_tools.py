"""Controlled tools for Agent Studio."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
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
        "epochs": 2,
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
        "epochs": 1,
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
        "epochs": 1,
        "seq_len": 2048,
        "warmup_steps": 100,
        "save_every": 250,
        "log_every": 10,
        "description": "Легкий стартовый претрейн/continued-pretrain по текстовому корпусу.",
    },
    "pretrain_scratch_60m": {
        "stage": "pretrain",
        "training_backend": "models-at-home",
        "output_dir": "out/agent/text/pretrain_scratch_60m",
        "hidden_size": 512,
        "num_layers": 8,
        "num_heads": 8,
        "batch_size": 8,
        "gradient_accumulation": 8,
        "learning_rate": 5e-4,
        "epochs": 1,
        "seq_len": 1024,
        "warmup_steps": 500,
        "save_every": 500,
        "description": "Претрейн с нуля (from scratch) собственной маленькой модели ~60M параметров (homellm architecture).",
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
        "epochs": 2,
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
        "epochs": 1,
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
        "category": "discovery",
        "description": "Показывает доступные типы обучения и краткие рекомендации по выбору пайплайна.",
        "arguments": {},
    },
    {
        "name": "list_local_models",
        "category": "discovery",
        "description": "Показывает локальные trainable-модели, адаптеры и результаты тренировок. GGUF для inference сюда не входят.",
        "arguments": {},
    },
    {
        "name": "list_datasets",
        "category": "discovery",
        "description": "Показывает доступные локальные датасеты из папки datasets/.",
        "arguments": {},
    },
    {
        "name": "preview_dataset",
        "category": "discovery",
        "description": "Читает несколько примеров из локального JSON/JSONL/TXT датасета.",
        "arguments": {
            "path": "str, относительный путь внутри datasets/ или абсолютный путь",
            "limit": "int, количество примеров",
        },
    },
    {
        "name": "get_training_presets",
        "category": "planning",
        "description": "Возвращает готовые пресеты для text/VLM обучения.",
        "arguments": {
            "domain": "str: text | vlm | all",
        },
    },
    {
        "name": "start_text_training",
        "category": "execution",
        "description": "Запускает text training. У тебя есть ПОЛНЫЙ доступ к тонкой настройке: ты можешь передавать любые гиперпараметры.",
        "arguments": {
            "config": "dict с training config. Обязательные ключи: 'data_path', 'epochs'. Для SFT/continued_pretrain передай 'base_model_path'. Для pretrain своей модели с нуля НЕ ПЕРЕДАВАЙ 'base_model_path', а укажи параметры ('hidden_size', 'num_layers', 'num_heads' и т.д.). Другие ключи: 'learning_rate', 'batch_size', 'gradient_accumulation', 'seq_len' и т.д.",
        },
    },
    {
        "name": "start_vlm_training",
        "category": "execution",
        "description": "Запускает VLM training. У тебя есть ПОЛНЫЙ доступ к тонкой настройке: передавай в config любые нужные гиперпараметры.",
        "arguments": {
            "config": "dict с training config. Обязательные ключи: 'stage' (vlm_pretrain|vlm_sft|vlm_grpo), 'data_path' (путь к датасету), 'base_model_path', 'epochs' (строго ключ 'epochs'!), 'learning_rate', 'batch_size' и т.д.",
        },
    },
    {
        "name": "list_runs",
        "category": "monitoring",
        "description": "Показывает agent-run'ы и их текущее состояние.",
        "arguments": {
            "status": "str: running | finished | all",
        },
    },
    {
        "name": "get_run_status",
        "category": "monitoring",
        "description": "Показывает состояние run по metrics.json и process PID.",
        "arguments": {
            "run_id": "str",
        },
    },
    {
        "name": "get_run_config",
        "category": "monitoring",
        "description": "Возвращает полный config.json указанного run.",
        "arguments": {
            "run_id": "str",
        },
    },
    {
        "name": "get_run_metrics",
        "category": "monitoring",
        "description": "Возвращает metrics.json указанного run для графиков и диагностики.",
        "arguments": {
            "run_id": "str",
        },
    },
    {
        "name": "read_run_logs",
        "category": "monitoring",
        "description": "Читает хвост stdout/stderr логов run.",
        "arguments": {
            "run_id": "str",
            "max_lines": "int",
        },
    },
    {
        "name": "stop_run",
        "category": "control",
        "description": "Останавливает запущенный train/job по PID.",
        "arguments": {
            "run_id": "str",
        },
    },
    {
        "name": "run_system_command",
        "category": "execution",
        "description": "Выполняет системную bash-команду внутри контейнера для проверки хардвера (nvidia-smi, lscpu, free -h) или навигации.",
        "arguments": {
            "command": "str, bash команда"
        },
    },
    {
        "name": "start_grpo_training",
        "category": "execution",
        "description": "Запускает GRPO (RL) обучение. Подходит для улучшения рассуждений модели через reinforcement learning.",
        "arguments": {
            "config": "dict. Обязательные: 'base_model_path', 'data_path', 'epochs'. Дополнительные: 'learning_rate', 'batch_size', 'gradient_accumulation', 'seq_len', 'max_new_tokens', 'temperature', 'reward_rules' (list of dicts), 'training_backend' ('models-at-home' или 'unsloth'), 'grpo_use_rollout_engine' (bool), 'grpo_rollout_backend' ('hf' или 'vllm').",
        },
    },
    {
        "name": "download_hf_model",
        "category": "data",
        "description": "Скачивает модель с HuggingFace Hub в папку models/. Для обучения нужны transformers-модели (не GGUF).",
        "arguments": {
            "repo_id": "str, HuggingFace repo (например: 'Qwen/Qwen2.5-0.5B')",
            "save_name": "str, опционально — имя папки в models/"
        },
    },
    {
        "name": "download_hf_dataset",
        "category": "data",
        "description": "Скачивает датасет с HuggingFace Hub и сохраняет как JSONL в datasets/.",
        "arguments": {
            "repo_id": "str, HuggingFace dataset repo (например: 'gsm8k')",
            "save_name": "str, опционально — имя файла в datasets/",
            "split": "str, по умолчанию 'train'",
            "max_rows": "int, 0 = все строки"
        },
    },
    {
        "name": "delete_artifact",
        "category": "control",
        "description": "Удаляет артефакт: датасет, модель, run или эксперимент.",
        "arguments": {
            "artifact_type": "str: dataset | model | run | experiment",
            "path": "str, путь или имя (для run — run_id, для остального — относительный путь)"
        },
    },
    {
        "name": "list_checkpoints",
        "category": "monitoring",
        "description": "Показывает доступные чекпоинты для указанного run (для resume или анализа).",
        "arguments": {
            "run_id": "str"
        },
    },
]


def get_tool_specs() -> List[Dict[str, Any]]:
    return TOOL_SPECS


def get_tool_groups() -> List[Dict[str, Any]]:
    return [
        {
            "category": "discovery",
            "title": "Разведка и входные данные",
            "tools": [tool for tool in TOOL_SPECS if tool.get("category") == "discovery"],
        },
        {
            "category": "planning",
            "title": "Планирование и пресеты",
            "tools": [tool for tool in TOOL_SPECS if tool.get("category") == "planning"],
        },
        {
            "category": "data",
            "title": "Управление данными и моделями",
            "tools": [tool for tool in TOOL_SPECS if tool.get("category") == "data"],
        },
        {
            "category": "execution",
            "title": "Запуск обучения",
            "tools": [tool for tool in TOOL_SPECS if tool.get("category") == "execution"],
        },
        {
            "category": "monitoring",
            "title": "Мониторинг run",
            "tools": [tool for tool in TOOL_SPECS if tool.get("category") == "monitoring"],
        },
        {
            "category": "control",
            "title": "Управление процессами",
            "tools": [tool for tool in TOOL_SPECS if tool.get("category") == "control"],
        },
    ]


def list_training_capabilities() -> Dict[str, Any]:
    return {
        "text_training": {
            "stages": ["pretrain", "continual_pretrain", "sft"],
            "tool": "start_text_training",
            "entrypoint": "python -m homellm.app.trainer_worker",
            "supports": ["full fine-tune", "LoRA/QLoRA", "Unsloth SFT", "multi-GPU/DeepSpeed/FSDP"],
            "key_config_params": [
                "stage", "base_model_path", "data_path", "epochs", "learning_rate", "batch_size",
                "gradient_accumulation", "seq_len", "warmup_steps", "save_every", "log_every",
                "tuning_method (full/lora/qlora)", "training_backend (models-at-home/unsloth)",
                "lora_r", "lora_alpha", "gradient_checkpointing", "mixed_precision",
                "use_flash_attention", "val_ratio", "eval_every",
                "resume_from_checkpoint", "max_steps", "lr_schedule",
                "optimizer (adamw/adamw_8bit/muon/magma_adamw)",
                "sft_columns (dict)", "sft_template (str)", "chat_template (str)",
            ],
        },
        "grpo_training": {
            "stages": ["grpo"],
            "tool": "start_grpo_training",
            "entrypoint": "python -m homellm.training.rl.train_rl",
            "supports": ["GRPO/DAPO/SDPO", "vLLM rollout", "reward rules", "Unsloth backend"],
            "key_config_params": [
                "base_model_path", "data_path", "epochs", "learning_rate", "batch_size",
                "gradient_accumulation", "seq_len", "max_new_tokens", "temperature",
                "reward_rules (list of dicts)", "training_backend (models-at-home/unsloth)",
                "grpo_use_rollout_engine (bool)", "grpo_rollout_backend (hf/vllm)",
            ],
        },
        "vlm_training": {
            "stages": ["vlm_pretrain", "vlm_sft", "vlm_grpo"],
            "tool": "start_vlm_training",
            "entrypoints": [
                "python -m homellm.training.vlm_pretrain",
                "python -m homellm.training.vlm_sft",
                "python -m homellm.training.vlm_grpo",
            ],
            "supports": ["image-text SFT", "continued multimodal pretrain", "GRPO"],
            "key_config_params": [
                "stage", "base_model_path", "data_path", "epochs", "learning_rate", "batch_size",
                "gradient_accumulation", "seq_len", "warmup_steps", "save_every",
                "tuning_method (full/lora/qlora)", "freeze_vision_tower", "freeze_projector",
                "assistant_only_loss", "caption_prompt", "val_data_path",
            ],
        },
        "data_management": {
            "download_model": "download_hf_model — скачивает transformers-модель с HF",
            "download_dataset": "download_hf_dataset — скачивает датасет с HF в JSONL",
            "delete": "delete_artifact — удаляет датасет, модель, run или эксперимент",
        },
        "notes": [
            "Для text SFT safest путь: trainer_worker с config.json.",
            "Для GRPO используй start_grpo_training, НЕ start_text_training.",
            "Для agentic workflows лучше запускать обучение отдельным процессом и мониторить через .runs/.",
            "Для локального агента inference идет через llama.cpp, а train через существующие Python workers.",
            "GGUF-файлы используются только для локального inference агента и не должны предлагаться как базовые модели для обучения.",
            "list_runs показывает ВСЕ запуски — и от агента, и запущенные из UI студий.",
        ],
    }


def _resolve_local_path(path_value: str, default_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    
    # Check if path already starts with the root folder name to prevent doubling
    # e.g. path_value="datasets/foo.json", default_root=".../datasets" -> avoid ".../datasets/datasets/foo.json"
    if candidate.parts and candidate.parts[0] == default_root.name:
        candidate = Path(*candidate.parts[1:])
        
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
    return {
        "models": items[:200],
        "count": len(items),
        "notes": [
            "В выдачу включаются только trainable-модели и артефакты обучения.",
            "GGUF-файлы для llama.cpp намеренно скрыты, потому что они используются только для inference и не подходят как база для training.",
        ],
    }


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


def _write_initial_metrics(metrics_path: Path, stage: str, model_name_input: str = "agent_run") -> None:
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "starting",
                "stage": stage,
                "model_name_input": model_name_input,
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
    _write_initial_metrics(metrics_path, stage=stage, model_name_input=config.get("model_name_input", "agent_run"))
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

    # Guard against immediate startup failures so the agent does not report
    # "training launched" for runs that crash before the first metrics update.
    time.sleep(1.5)
    exit_code = process.poll()
    if exit_code is not None:
        stderr_tail = _tail_lines(stderr_path, max_lines=40)
        stdout_tail = _tail_lines(stdout_path, max_lines=40)
        error_message = "\n".join(stderr_tail[-12:] or stdout_tail[-12:] or [f"Process exited with code {exit_code}"])
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "error",
                    "stage": stage,
                    "current_step": 0,
                    "total_steps": 0,
                    "finished_at": datetime.now().isoformat(),
                    "error": error_message,
                    "exit_code": exit_code,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        raise RuntimeError(f"Run failed during startup: {error_message}")

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
        raise ValueError(f"Unsupported text stage: {stage}. Use start_grpo_training for GRPO.")

    if not cfg.get("output_dir"):
        cfg["output_dir"] = f"out/agent/text/{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not cfg.get("model_name_input"):
        cfg["model_name_input"] = Path(cfg["output_dir"]).name

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
    _write_initial_metrics(metrics_path, stage=stage, model_name_input=cfg.get("model_name_input", "agent_run"))

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
    result["monitoring_url"] = f"/?run_id={run_id}"
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
    if not cfg.get("model_name_input"):
        cfg["model_name_input"] = Path(cfg["output_dir"]).name

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
    _write_initial_metrics(metrics_path, stage=stage, model_name_input=cfg.get("model_name_input", "agent_vlm_run"))

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
    result["monitoring_url"] = f"/VLM_Studio?run_id={run_id}"
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


def get_run_config(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return {
        "run_id": run_id,
        "config": _load_json_if_exists(run_dir / "config.json"),
    }


def get_run_metrics(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return {
        "run_id": run_id,
        "metrics": _load_json_if_exists(run_dir / "metrics.json"),
    }


def list_runs(status: str = "all") -> Dict[str, Any]:
    wanted = str(status or "all").lower()
    runs: List[Dict[str, Any]] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        if run_dir.name in {"agent_llama_server", "agent_sessions", "ui_preferences.json"}:
            continue
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() and not config_path.exists():
            continue
        run_status = get_run_status(run_dir.name)
        is_running = bool(run_status.get("is_running"))
        if wanted == "running" and not is_running:
            continue
        if wanted == "finished" and is_running:
            continue
        runs.append(run_status)
    return {"status_filter": wanted, "runs": runs[:50], "count": len(runs)}


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


def run_system_command(command: str) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Timeout", "stdout": "", "stderr": "", "returncode": -1}
    except Exception as exc:
        return {"error": str(exc), "stdout": "", "stderr": "", "returncode": -1}


def start_grpo_training(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config or {})
    cfg["stage"] = "grpo"

    if not cfg.get("output_dir"):
        cfg["output_dir"] = f"out/agent/text/grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not cfg.get("model_name_input"):
        cfg["model_name_input"] = Path(cfg["output_dir"]).name

    run_id = f"agent_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve() / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(run_output_dir)
    cfg["ui_run_dir"] = str(RUNS_DIR / run_id)

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)
    _write_initial_metrics(metrics_path, stage="grpo", model_name_input=cfg.get("model_name_input", "agent_grpo"))

    config_json = json.dumps(cfg, default=str)

    distributed_mode = cfg.get("distributed_mode", "default")
    config_file = cfg.get("config_file")
    num_gpus = int(cfg.get("num_gpus", 1) or 1)
    if distributed_mode != "default" and config_file:
        cmd = [
            "accelerate", "launch",
            "--config_file", str(config_file),
            "--num_processes", str(num_gpus),
            "-m", "homellm.training.rl.train_rl",
            "--config_json", config_json,
        ]
    else:
        cmd = [
            sys.executable, "-m", "homellm.training.rl.train_rl",
            "--config_json", config_json,
        ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    gpu_ids = cfg.get("gpu_ids") or []
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    result = _spawn_run(run_id=run_id, cmd=cmd, env=env, stage="grpo", config=cfg)
    result["output_dir"] = _safe_relpath(run_output_dir)
    result["monitoring_url"] = f"/?run_id={run_id}"
    return result


def download_hf_model(repo_id: str, save_name: str = "") -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    if not save_name:
        save_name = repo_id.split("/")[-1]
    save_path = MODELS_DIR / save_name
    if save_path.exists():
        return {"status": "already_exists", "path": _safe_relpath(save_path), "name": save_name}

    save_path.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(save_path),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", "*.txt", "*.gitattributes", ".git*"],
        )
    except Exception as exc:
        import shutil
        shutil.rmtree(save_path, ignore_errors=True)
        raise RuntimeError(f"Failed to download {repo_id}: {exc}")

    config_file = save_path / "config.json"
    model_type = None
    if config_file.exists():
        meta = _load_json_if_exists(config_file)
        model_type = meta.get("model_type")

    return {
        "status": "downloaded",
        "path": _safe_relpath(save_path),
        "name": save_name,
        "model_type": model_type,
    }


def download_hf_dataset(repo_id: str, save_name: str = "", split: str = "train", max_rows: int = 0) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download
    import tempfile

    if not save_name:
        save_name = repo_id.split("/")[-1] + ".jsonl"
    if not save_name.endswith((".jsonl", ".json", ".csv", ".txt", ".parquet")):
        save_name += ".jsonl"

    save_path = DATASETS_DIR / save_name
    if save_path.exists():
        return {
            "status": "already_exists",
            "path": _safe_relpath(save_path),
            "size_mb": round(save_path.stat().st_size / (1024**2), 2),
        }

    try:
        from datasets import load_dataset
        ds = load_dataset(repo_id, split=split, streaming=True)
        
        count = 0
        with open(save_path, "w", encoding="utf-8") as f:
            for example in ds:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
                if max_rows > 0 and count >= max_rows:
                    break

        return {
            "status": "downloaded",
            "path": _safe_relpath(save_path),
            "rows": count,
            "size_mb": round(save_path.stat().st_size / (1024**2), 2),
        }
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download dataset {repo_id}: {exc}")


def delete_artifact(artifact_type: str, path: str) -> Dict[str, Any]:
    import shutil
    artifact_type = str(artifact_type).lower()
    
    if artifact_type == "dataset":
        target = _resolve_local_path(path, DATASETS_DIR)
        if not str(target).startswith(str(DATASETS_DIR)):
            raise ValueError("Path must be inside datasets/")
    elif artifact_type == "model":
        target = _resolve_local_path(path, MODELS_DIR)
        if not str(target).startswith(str(MODELS_DIR)):
            raise ValueError("Path must be inside models/")
    elif artifact_type == "run":
        target = RUNS_DIR / path
        if not str(target).startswith(str(RUNS_DIR)):
            raise ValueError("Path must be inside .runs/")
    elif artifact_type == "experiment":
        target = _resolve_local_path(path, OUTPUT_DIR)
        if not str(target).startswith(str(OUTPUT_DIR)):
            raise ValueError("Path must be inside out/")
    else:
        raise ValueError(f"Unknown artifact_type: {artifact_type}. Use: dataset, model, run, experiment")

    if not target.exists():
        return {"deleted": False, "reason": f"Not found: {target}"}

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"deleted": True, "path": str(target), "type": artifact_type}


def list_checkpoints(run_id: str) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_id
    config = _load_json_if_exists(run_dir / "config.json")
    output_dir = config.get("output_dir", "")
    
    checkpoints: List[Dict[str, Any]] = []
    if output_dir:
        out_path = Path(output_dir)
        if out_path.exists():
            for cp in sorted(out_path.glob("checkpoint-*"), reverse=True):
                if cp.is_dir():
                    checkpoints.append({
                        "name": cp.name,
                        "path": _safe_relpath(cp),
                        "step": cp.name.replace("checkpoint-", ""),
                        "size_mb": round(sum(f.stat().st_size for f in cp.rglob("*") if f.is_file()) / (1024**2), 1),
                    })
    return {"run_id": run_id, "checkpoints": checkpoints}


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
        "start_grpo_training": lambda: start_grpo_training(config=dict(args.get("config") or {})),
        "list_runs": lambda: list_runs(status=str(args.get("status", "all"))),
        "get_run_status": lambda: get_run_status(run_id=str(args.get("run_id", ""))),
        "get_run_config": lambda: get_run_config(run_id=str(args.get("run_id", ""))),
        "get_run_metrics": lambda: get_run_metrics(run_id=str(args.get("run_id", ""))),
        "read_run_logs": lambda: read_run_logs(run_id=str(args.get("run_id", "")), max_lines=int(args.get("max_lines", 80))),
        "stop_run": lambda: stop_run(run_id=str(args.get("run_id", ""))),
        "run_system_command": lambda: run_system_command(command=str(args.get("command", ""))),
        "download_hf_model": lambda: download_hf_model(
            repo_id=str(args.get("repo_id", "")),
            save_name=str(args.get("save_name", "")),
        ),
        "download_hf_dataset": lambda: download_hf_dataset(
            repo_id=str(args.get("repo_id", "")),
            save_name=str(args.get("save_name", "")),
            split=str(args.get("split", "train")),
            max_rows=int(args.get("max_rows", 0)),
        ),
        "delete_artifact": lambda: delete_artifact(
            artifact_type=str(args.get("artifact_type", "")),
            path=str(args.get("path", "")),
        ),
        "list_checkpoints": lambda: list_checkpoints(run_id=str(args.get("run_id", ""))),
    }
    if tool_name not in registry:
        raise ValueError(f"Unknown tool: {tool_name}")
    return registry[tool_name]()
