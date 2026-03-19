"""VLM Studio: modern vision-language training and inference."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

try:
    from homellm.i18n import get_current_language, load_translations, t
except ImportError:
    def t(key, **kwargs):
        return key

    def load_translations():
        pass

    def get_current_language():
        return "en"

try:
    from homellm.app.ui_preferences import DEFAULT_THEME, apply_theme_css, init_user_preferences
except ImportError:
    from ..ui_preferences import DEFAULT_THEME, apply_theme_css, init_user_preferences

load_translations()


def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parent


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent.parent.parent)
RUNS_DIR = PROJECT_ROOT / ".runs"
DATASET_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "out"
MODELS_DIR = PROJECT_ROOT / "models"
STUDY_MATERIALS = PROJECT_ROOT / "study_materials"
ACTIVE_RUN_FILE = RUNS_DIR / "vlm_active_run.json"
CONFIGS_DIR = PROJECT_ROOT / "configs"
ACCELERATE_MULTI_GPU_CONFIG = PROJECT_ROOT / "configs" / "accelerate_multi_gpu.yaml"
RUNS_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

VLM_STAGES = [
    ("vlm_pretrain", "vlm.stage.pretrain"),
    ("vlm_sft", "vlm.stage.sft"),
    ("vlm_grpo", "vlm.stage.grpo"),
]
STAGE_TO_WORKER = {
    "vlm_pretrain": "homellm.training.vlm_pretrain",
    "vlm_sft": "homellm.training.vlm_sft",
    "vlm_grpo": "homellm.training.vlm_grpo",
}

PARALLEL_TYPES = {
    "default": {"icon": "🖥️", "name": "Single GPU / CPU", "type": "default"},
    "multi_gpu": {"icon": "⚡", "name": "Multi-GPU (DDP)", "type": "multi_gpu"},
    "deepspeed_zero2": {"icon": "🧩", "name": "DeepSpeed ZeRO-2", "type": "deepspeed_zero2"},
    "deepspeed_zero3": {"icon": "🧠", "name": "DeepSpeed ZeRO-3", "type": "deepspeed_zero3"},
    "deepspeed_zero3_offload": {"icon": "💾", "name": "ZeRO-3 + Offload", "type": "deepspeed_zero3_offload"},
    "fsdp": {"icon": "📦", "name": "FSDP", "type": "fsdp"},
}

VLM_HF_PRESETS: List[Dict[str, str]] = [
    {
        "name": "Qwen 3.5 0.8B",
        "repo_id": "Qwen/Qwen3.5-0.8B",
        "save_name": "Qwen3.5-0.8B",
        "description": "Основной малый VLM для быстрых домашних экспериментов и small-task pretrain/SFT.",
    },
    {
        "name": "Qwen 3.5 2B",
        "repo_id": "Qwen/Qwen3.5-2B",
        "save_name": "Qwen3.5-2B",
        "description": "Основной VLM для 24GB и 2x3090, лучше для OCR/doc/ui задач.",
    },
    {
        "name": "Qwen2.5-VL 3B Instruct",
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "save_name": "Qwen2.5-VL-3B-Instruct",
        "description": "Хороший запасной instruct-бейзлайн, если нужен зрелый мультимодальный стек.",
    },
    {
        "name": "Qwen2.5-VL 7B Instruct",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "save_name": "Qwen2.5-VL-7B-Instruct",
        "description": "Сильный instruct-бейзлайн для 2x3090.",
    },
    {
        "name": "LLaVA-NeXT Mistral 7B",
        "repo_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "save_name": "llava-v1.6-mistral-7b",
        "description": "Альтернатива Qwen-линейке для классического image+instruction тюнинга.",
    },
]

SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "FastSFT_0.8B_24GB": {
        "stage": "vlm_sft",
        "base_model_path": "Qwen/Qwen3.5-0.8B",
        "tuning_method": "qlora",
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 2e-5,
        "num_epochs": 2,
        "seq_len": 2048,
        "warmup_steps": 50,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 896 * 28 * 28,
        "freeze_vision_tower": True,
        "assistant_only_loss": True,
        "gradient_checkpointing": True,
        "use_flash_attention": True,
        "description": "Быстрый SFT small VLM на одной 24GB карте.",
    },
    "FastSFT_2B_2x3090": {
        "stage": "vlm_sft",
        "base_model_path": "Qwen/Qwen3.5-2B",
        "tuning_method": "qlora",
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "seq_len": 3072,
        "warmup_steps": 100,
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
        "freeze_vision_tower": True,
        "assistant_only_loss": True,
        "gradient_checkpointing": True,
        "use_flash_attention": True,
        "num_gpus": 2,
        "description": "Основной сценарий под две 3090 для instruction tuning.",
    },
    "SmallPretrain_0.8B_caption_alignment": {
        "stage": "vlm_pretrain",
        "base_model_path": "Qwen/Qwen3.5-0.8B",
        "tuning_method": "qlora",
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 1e-5,
        "num_epochs": 1,
        "seq_len": 1536,
        "warmup_steps": 50,
        "freeze_vision_tower": True,
        "assistant_only_loss": False,
        "caption_prompt": "Describe this image in detail.",
        "description": "Небольшой continued multimodal pretrain на caption/OCR/VQA корпусах.",
    },
    "OCRTune_2B_doc_qa": {
        "stage": "vlm_sft",
        "base_model_path": "Qwen/Qwen3.5-2B",
        "tuning_method": "qlora",
        "batch_size": 1,
        "gradient_accumulation": 16,
        "learning_rate": 1e-5,
        "num_epochs": 2,
        "seq_len": 3072,
        "warmup_steps": 100,
        "min_pixels": 512 * 28 * 28,
        "max_pixels": 1536 * 28 * 28,
        "freeze_vision_tower": False,
        "assistant_only_loss": True,
        "description": "Пресет для OCR/doc QA с более высоким визуальным бюджетом.",
    },
    "ExperimentalVLM_GRPO": {
        "stage": "vlm_grpo",
        "base_model_path": "Qwen/Qwen3.5-0.8B",
        "tuning_method": "qlora",
        "batch_size": 1,
        "gradient_accumulation": 1,
        "learning_rate": 5e-6,
        "num_epochs": 1,
        "seq_len": 2048,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "freeze_vision_tower": True,
        "assistant_only_loss": True,
        "description": "Экспериментальный image-conditioned reward optimization.",
    },
}

VLM_HF_DATASETS: List[Dict[str, str]] = [
    {
        "id": "HuggingFaceH4/llava-instruct-mix-vsft",
        "name": "LLaVA Instruct Mix",
        "description": "Основной single-image instruction tuning корпус.",
        "split": "train",
        "format": "image_messages",
    },
    {
        "id": "Vision-Flan/vision-flan",
        "name": "Vision-Flan",
        "description": "Крупный human-labeled instruction tuning датасет.",
        "split": "train",
        "format": "image_messages",
    },
    {
        "id": "HuggingFaceM4/COCO",
        "name": "COCO Captions",
        "description": "Простой caption корпус для small-task pretrain/alignment.",
        "split": "train",
        "format": "image_caption",
    },
    {
        "id": "liuhaotian/LLaVA-Pretrain",
        "name": "LLaVA Pretrain",
        "description": "Caption-alignment корпус для continued multimodal pretrain.",
        "split": "train",
        "format": "image_caption",
    },
    {
        "id": "nielsr/caption-the-image",
        "name": "Caption the Image",
        "description": "Небольшой caption dataset для быстрых домашних прогонов.",
        "split": "train",
        "format": "image_caption",
    },
]


def save_active_run(run_id: str, config: Dict[str, Any]) -> None:
    with open(ACTIVE_RUN_FILE, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "config": config, "started_at": datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)


def load_active_run() -> Optional[Dict[str, Any]]:
    if not ACTIVE_RUN_FILE.exists():
        return None
    try:
        with open(ACTIVE_RUN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clear_active_run() -> None:
    ACTIVE_RUN_FILE.unlink(missing_ok=True)


def restore_session_state() -> None:
    active = load_active_run()
    if not active or not active.get("run_id"):
        return
    run_id = active["run_id"]
    if (RUNS_DIR / run_id).exists() and _is_process_running(run_id):
        st.session_state.vlm_current_run_id = run_id
        st.session_state.vlm_training_active = True
    else:
        clear_active_run()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_metrics(run_id: str) -> Optional[Dict[str, Any]]:
    return _load_json(RUNS_DIR / run_id / "metrics.json")


def _write_metrics_status(run_id: str, status: str) -> None:
    metrics_path = RUNS_DIR / run_id / "metrics.json"
    metrics = _load_json(metrics_path) or {}
    metrics["status"] = status
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def _is_process_running(run_id: str) -> bool:
    pid_path = RUNS_DIR / run_id / "pid"
    if not pid_path.exists():
        return False
    try:
        with open(pid_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except Exception:
        return False


def _list_vlm_runs() -> List[Path]:
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for entry in RUNS_DIR.iterdir():
        if not entry.is_dir():
            continue
        cfg = _load_json(entry / "config.json")
        if cfg and cfg.get("stage") in STAGE_TO_WORKER:
            runs.append(entry)
    return sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)


def _delete_checkpoint(run_id: str, checkpoint_path: str) -> tuple[bool, str]:
    try:
        path = Path(checkpoint_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            return False, "Чекпоинт уже отсутствует на диске"
        shutil.rmtree(path)
        metrics = _load_metrics(run_id) or {}
        metrics["checkpoints"] = [ckpt for ckpt in metrics.get("checkpoints", []) if ckpt.get("path") != checkpoint_path]
        with open(RUNS_DIR / run_id / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        return True, "Чекпоинт удалён"
    except Exception as exc:
        return False, str(exc)


def _delete_experiment(run_id: str) -> tuple[bool, str]:
    try:
        run_dir = RUNS_DIR / run_id
        cfg = _load_json(run_dir / "config.json") or {}
        output_dir = cfg.get("output_dir")
        if output_dir:
            out_path = Path(output_dir)
            if not out_path.is_absolute():
                out_path = PROJECT_ROOT / out_path
            if out_path.exists():
                shutil.rmtree(out_path, ignore_errors=True)
        shutil.rmtree(run_dir, ignore_errors=True)
        if st.session_state.get("vlm_current_run_id") == run_id:
            st.session_state.vlm_current_run_id = None
            clear_active_run()
        return True, "Эксперимент удалён"
    except Exception as exc:
        return False, str(exc)


def _download_hf_vlm_model(repo_id: str, save_name: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        st.error("huggingface_hub не установлен")
        return False
    save_path = MODELS_DIR / (save_name or repo_id.split("/")[-1])
    if save_path.exists():
        st.warning(f"Папка уже существует: {save_path}")
        return False
    try:
        with st.spinner(f"Скачиваем {repo_id}..."):
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(save_path),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt", "*.gitattributes", ".git*"],
            )
        return (save_path / "config.json").exists()
    except Exception as exc:
        st.error(f"Ошибка загрузки: {exc}")
        shutil.rmtree(save_path, ignore_errors=True)
        return False


def _dataset_output_record(row: Dict[str, Any], dataset_id: str, idx: int, format_type: str) -> Optional[Dict[str, Any]]:
    image = row.get("image")
    if image is None and row.get("images"):
        image = row["images"][0]
    if image is None:
        return None
    image_ref = None
    if hasattr(image, "save"):
        file_name = f"vlm_{dataset_id.replace('/', '_')}_{idx}.png"
        image_path = DATASET_DIR / file_name
        image.save(image_path)
        image_ref = file_name
    else:
        image_ref = str(image)

    if format_type == "image_messages":
        messages = row.get("messages") or row.get("conversations")
        if not messages and row.get("question") and (row.get("answer") or row.get("output")):
            answer = row.get("answer") or row.get("output")
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": str(row["question"])}]},
                {"role": "assistant", "content": [{"type": "text", "text": str(answer)}]},
            ]
        if not messages:
            return None
        return {"image": image_ref, "messages": messages, "schema": "single_image_messages"}

    caption = row.get("caption") or row.get("caption_text") or row.get("caption_gt") or row.get("text")
    if format_type == "image_caption" and caption:
        return {"image": image_ref, "caption": str(caption), "schema": "single_image_caption"}

    if format_type == "ocr_qa":
        question = row.get("question") or row.get("prompt")
        answer = row.get("answer") or row.get("response") or row.get("output")
        if question and answer:
            return {
                "image": image_ref,
                "question": str(question),
                "answer": str(answer),
                "schema": "ocr_or_doc_qa",
            }
    return None


def _download_hf_vlm_dataset(hf_id: str, split: str, out_name: str, format_type: str) -> str | None:
    try:
        from datasets import load_dataset
    except ImportError:
        return "datasets library not installed"
    out_path = DATASET_DIR / (out_name or "vlm_data.jsonl")
    try:
        dataset = load_dataset(hf_id, split=split, trust_remote_code=True)
    except Exception as exc:
        return str(exc)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            try:
                record = _dataset_output_record(row, hf_id, idx, format_type)
                if not record:
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception:
                continue
    return str(out_path) if written > 0 else None


def _get_vlm_datasets() -> List[Dict[str, Any]]:
    datasets = []
    for path in sorted(DATASET_DIR.glob("*.jsonl")):
        size_mb = path.stat().st_size / (1024 * 1024)
        preview = _read_dataset_preview(path, limit=1)
        datasets.append(
            {
                "name": path.name,
                "path": str(path),
                "size": f"{size_mb:.1f} MB",
                "schema": preview.get("schema", "unknown"),
                "count": preview.get("count", 0),
            }
        )
    return datasets


def _model_looks_like_vlm(cfg: Dict[str, Any]) -> bool:
    model_type = str(cfg.get("model_type", "")).lower()
    pipeline_tag = str(cfg.get("pipeline_tag", "")).lower()
    text = json.dumps(cfg, ensure_ascii=False).lower()
    known = {"qwen3_5", "qwen2_vl", "qwen2_5_vl", "llava", "llava_next", "internvl", "minicpmv"}
    return model_type in known or pipeline_tag == "image-text-to-text" or "vision" in text or "image-text-to-text" in text


def get_available_vlm_models() -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    for preset in VLM_HF_PRESETS:
        models.append({"name": f"🤗 {preset['name']}", "path": preset["repo_id"], "type": "hf", "family": preset["name"]})
    for final in OUTPUT_DIR.rglob("final_model"):
        cfg = _load_json(final / "config.json")
        if final.is_dir() and cfg:
            rel = final.relative_to(OUTPUT_DIR)
            models.append({"name": f"📁 {rel}", "path": str(final), "type": "local", "family": cfg.get("model_type", "local")})
    for model_dir in MODELS_DIR.iterdir() if MODELS_DIR.exists() else []:
        cfg = _load_json(model_dir / "config.json")
        if model_dir.is_dir() and cfg and _model_looks_like_vlm(cfg):
            models.append({"name": f"📦 {model_dir.name}", "path": str(model_dir), "type": "local", "family": cfg.get("model_type", "local")})
    return models


def _resolve_model_choice_label(path: str) -> str:
    if not path:
        return "❌ Не выбрано"
    return Path(path).name if "/" in str(path) else str(path)


def _default_gpu_ids(num_gpus: int) -> List[int]:
    return list(range(max(1, int(num_gpus))))


def _start_vlm_training(config: Dict[str, Any]) -> tuple[str, subprocess.Popen]:
    stage = config.get("stage", "vlm_sft")
    worker_module = STAGE_TO_WORKER[stage]
    prefix = stage.replace("vlm_", "")
    run_id = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")

    experiment_root = Path(config.get("output_dir", f"out/{stage}"))
    if not experiment_root.is_absolute():
        experiment_root = PROJECT_ROOT / experiment_root
    run_output_dir = experiment_root / run_id
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(run_output_dir)
    config["gpu_ids"] = config.get("gpu_ids") or _default_gpu_ids(int(config.get("num_gpus", 1)))

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"status": "starting", "current_step": 0, "stage": stage}, f, indent=2, ensure_ascii=False)

    env = os.environ.copy()
    if config.get("gpu_ids"):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in config["gpu_ids"])
    env["PYTHONUNBUFFERED"] = "1"

    num_gpus = int(config.get("num_gpus", 1))
    if num_gpus > 1 and ACCELERATE_MULTI_GPU_CONFIG.exists():
        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            str(ACCELERATE_MULTI_GPU_CONFIG),
            "--num_processes",
            str(num_gpus),
            "-m",
            worker_module,
            "--config",
            str(config_path),
            "--metrics",
            str(metrics_path),
        ]
    else:
        cmd = [sys.executable, "-m", worker_module, "--config", str(config_path), "--metrics", str(metrics_path)]

    with open(run_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(cmd))
    stdout_file = open(stdout_path, "w", encoding="utf-8")
    stderr_file = open(stderr_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_file,
        stderr=stderr_file,
        env=env,
        start_new_session=True,
    )
    with open(run_dir / "pid", "w", encoding="utf-8") as f:
        f.write(str(process.pid))
    st.session_state[f"vlm_stdout_file_{run_id}"] = stdout_file
    st.session_state[f"vlm_stderr_file_{run_id}"] = stderr_file
    save_active_run(run_id, config)
    return run_id, process


def _stop_vlm_training(run_id: str) -> bool:
    pid_path = RUNS_DIR / run_id / "pid"
    if not pid_path.exists():
        return False
    try:
        with open(pid_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _write_metrics_status(run_id, "stopped")
        if st.session_state.get("vlm_current_run_id") == run_id:
            clear_active_run()
        return True
    except Exception:
        return False


def _read_dataset_preview(path: Path, limit: int = 5) -> Dict[str, Any]:
    samples = []
    fields = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                samples.append(record)
                if isinstance(record, dict):
                    fields.update(record.keys())
            except json.JSONDecodeError:
                continue
    schema = "unknown"
    if samples:
        sample = samples[0]
        if sample.get("messages") or sample.get("conversations"):
            schema = sample.get("schema", "single_image_messages")
        elif sample.get("caption"):
            schema = sample.get("schema", "single_image_caption")
        elif sample.get("question") and sample.get("answer"):
            schema = sample.get("schema", "ocr_or_doc_qa")
    return {"samples": samples, "fields": sorted(fields), "schema": schema, "count": len(samples)}


def _render_quick_summary_vlm(config: Dict[str, Any]) -> bool:
    model_display = _resolve_model_choice_label(config.get("base_model_path", ""))
    data_path = config.get("data_path") or ""
    data_display = Path(data_path).name if data_path and Path(data_path).exists() else "❌ Не выбрано"
    stage_display = t(dict(VLM_STAGES).get(config.get("stage", "vlm_sft"), config.get("stage", "vlm_sft")))
    all_ready = bool(config.get("base_model_path")) and bool(data_path and Path(data_path).exists())
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%); border: 2px solid #444; border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:1rem;">
                <div style="background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:1rem;">
                    <div style="color:#888;">Модель</div>
                    <div style="color:white;">{model_display}</div>
                </div>
                <div style="background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:1rem;">
                    <div style="color:#888;">Данные</div>
                    <div style="color:white;">{data_display}</div>
                </div>
                <div style="background:#1a1a1a; border:1px solid #333; border-radius:8px; padding:1rem;">
                    <div style="color:#888;">Этап</div>
                    <div style="color:white;">{stage_display}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return all_ready


def _render_model_preview_vlm(config: Dict[str, Any]) -> None:
    stage = config.get("stage", "vlm_sft")
    freeze_vision = "да" if config.get("freeze_vision_tower") else "нет"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Тюнинг", config.get("tuning_method", "lora"))
    with c2:
        st.metric("Batch", config.get("batch_size", 1))
    with c3:
        st.metric("Seq", config.get("seq_len", 2048))
    with c4:
        st.metric("Warmup", config.get("warmup_steps", 0))
    st.caption(
        f"Этап: `{stage}` | Freeze vision tower: `{freeze_vision}` | "
        f"Min pixels: `{config.get('min_pixels', 'auto')}` | Max pixels: `{config.get('max_pixels', 'auto')}`"
    )


def _load_chat_model(model_path: str):
    cache = st.session_state.setdefault("vlm_chat_cache", {})
    if model_path not in cache:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            model_cls = AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = model_cls.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        cache[model_path] = {"processor": processor, "model": model}
    return cache[model_path]["processor"], cache[model_path]["model"]


def _generate_vlm_response(model_path: str, image, prompt: str, system_prompt: str, max_new_tokens: int, temperature: float) -> str:
    import torch

    processor, model = _load_chat_model(model_path)
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
    messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]})

    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    model_device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 0.1),
    )
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _safe_option_index(options: List[Any], value: Any, default: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default


def _closest_numeric_option(options: List[float], value: float) -> float:
    if not options:
        return value
    return min(options, key=lambda option: abs(float(option) - float(value)))


def _get_gpu_info() -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        {
                            "id": int(parts[0]),
                            "name": parts[1],
                            "memory_gb": round(float(parts[2]) / 1024, 1),
                            "compute_capability": parts[3],
                        }
                    )
    except Exception:
        pass
    return gpus


def _render_vlm_model_config(selected_stage: str) -> Dict[str, Any]:
    st.sidebar.header(f"🖼️ {t('vlm.sidebar.base_model')}")
    available_models = get_available_vlm_models()
    base_model_default = (
        st.session_state.get("vlm_selected_base_path")
        or st.session_state.get("vlm_cfg_base_model_path")
        or VLM_HF_PRESETS[0]["repo_id"]
    )
    model_names = [item["name"] for item in available_models]
    default_model_idx = next((idx for idx, item in enumerate(available_models) if item["path"] == base_model_default), 0)
    model_idx = (
        st.sidebar.selectbox(
            t("vlm.sidebar.base_model"),
            range(len(model_names)),
            index=default_model_idx if available_models else 0,
            format_func=lambda idx: model_names[idx] if available_models else "",
        )
        if available_models
        else None
    )
    base_model_path = available_models[model_idx]["path"] if available_models and model_idx is not None else base_model_default
    experiment_name = st.sidebar.text_input(
        t("vlm.sidebar.experiment_name"),
        value=st.session_state.get("vlm_cfg_experiment_name", selected_stage),
    )
    tuning_method = st.sidebar.selectbox(
        t("sidebar.tuning_method"),
        ["full", "lora", "qlora"],
        index=["full", "lora", "qlora"].index(st.session_state.get("vlm_cfg_tuning_method", "qlora"))
        if st.session_state.get("vlm_cfg_tuning_method", "qlora") in {"full", "lora", "qlora"}
        else 2,
    )
    lora_r = None
    lora_alpha = None
    lora_dropout = None
    lora_target_modules = None
    if tuning_method in {"lora", "qlora"}:
        st.sidebar.markdown("**LoRA параметры:**")
        lora_r = st.sidebar.slider("LoRA r (rank)", 8, 128, int(st.session_state.get("vlm_cfg_lora_r", 32)), step=8)
        lora_alpha = st.sidebar.slider("LoRA alpha", 8, 256, int(st.session_state.get("vlm_cfg_lora_alpha", 32)), step=8)
        lora_dropout = st.sidebar.slider("LoRA dropout", 0.0, 0.5, float(st.session_state.get("vlm_cfg_lora_dropout", 0.05)), step=0.05)
        default_modules = st.session_state.get(
            "vlm_cfg_lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        lora_target_modules = st.sidebar.multiselect(
            "Target modules",
            options=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "embed_tokens"],
            default=default_modules,
        )
        if not lora_target_modules:
            lora_target_modules = "all-linear"
    return {
        "stage": selected_stage,
        "base_model_path": base_model_path,
        "model_name_or_path": base_model_path,
        "experiment_name": experiment_name,
        "tuning_method": tuning_method,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
    }


def _render_vlm_training_config(selected_stage: str) -> Dict[str, Any]:
    st.sidebar.header(f"📈 {t('sidebar.hyperparams')}")
    lr_options = [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3]
    saved_lr = float(st.session_state.get("vlm_cfg_learning_rate", 2e-5))
    safe_lr = _closest_numeric_option(lr_options, saved_lr)
    seq_len_options = [512, 1024, 2048, 3072, 4096]
    safe_seq_len = int(st.session_state.get("vlm_cfg_seq_len", 2048))
    if safe_seq_len not in seq_len_options:
        seq_len_options = sorted(seq_len_options + [safe_seq_len])

    optimizer = st.sidebar.selectbox(
        "Optimizer",
        options=["adamw", "adamw_8bit", "muon", "magma_adamw"],
        index=0,
    )
    batch_size = st.sidebar.slider("Batch Size", 1, 256, int(st.session_state.get("vlm_cfg_batch_size", 2)))
    grad_accum = st.sidebar.slider("Gradient Accumulation", 1, 32, int(st.session_state.get("vlm_cfg_gradient_accumulation", 8)))
    st.sidebar.caption(f"Effective batch: {batch_size * grad_accum}")
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=lr_options,
        value=safe_lr,
        format_func=lambda x: f"{x:.0e}",
    )
    lr_schedule_label = st.sidebar.selectbox(
        "LR scheduler",
        options=[
            "Cosine (with warmup)",
            "Linear (with warmup)",
            "Constant (with warmup)",
            "Cosine with Restarts (with warmup)",
        ],
        index=0,
    )
    lr_schedule_map = {
        "Cosine (with warmup)": "cosine",
        "Linear (with warmup)": "linear",
        "Constant (with warmup)": "constant_with_warmup",
        "Cosine with Restarts (with warmup)": "cosine_with_restarts",
    }
    training_mode = st.sidebar.radio(t("training.mode"), [t("training.mode_epochs"), t("training.mode_steps")])
    if training_mode == t("training.mode_epochs"):
        num_epochs = st.sidebar.number_input("Epochs", 1, 10, int(st.session_state.get("vlm_cfg_num_epochs", 2)))
        max_steps = None
    else:
        num_epochs = 1
        max_steps = st.sidebar.number_input("Max Steps", 1, 1_000_000, int(st.session_state.get("vlm_cfg_max_steps", 2000)), step=100)
    seq_len = st.sidebar.selectbox(
        "Seq Length",
        seq_len_options,
        index=_safe_option_index(seq_len_options, safe_seq_len, default=_safe_option_index(seq_len_options, 2048, default=0)),
    )
    warmup_steps = st.sidebar.number_input("Warmup Steps", min_value=0, max_value=10000, value=int(st.session_state.get("vlm_cfg_warmup_steps", 100)))
    max_grad_norm = st.sidebar.number_input("Max Gradient Norm", min_value=0.0, max_value=10.0, value=float(st.session_state.get("vlm_cfg_max_grad_norm", 1.0)), step=0.1)
    weight_decay = st.sidebar.number_input("Weight Decay", min_value=0.0, max_value=1.0, value=float(st.session_state.get("vlm_cfg_weight_decay", 0.01)), step=0.01)
    min_lr_ratio = st.sidebar.slider("Min LR Ratio (Cosine floor)", 0.0, 0.2, float(st.session_state.get("vlm_cfg_min_lr_ratio", 0.0)), step=0.01)
    return {
        "optimizer": optimizer,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "learning_rate": float(learning_rate),
        "lr_schedule": lr_schedule_map[lr_schedule_label],
        "num_epochs": int(num_epochs),
        "max_steps": int(max_steps) if max_steps else None,
        "seq_len": int(seq_len),
        "warmup_steps": int(warmup_steps),
        "max_grad_norm": float(max_grad_norm),
        "weight_decay": float(weight_decay),
        "min_lr_ratio": float(min_lr_ratio),
    }


def _render_vlm_dataset_config(selected_stage: str) -> Dict[str, Any]:
    st.sidebar.header(f"📁 {t('sidebar.data')}")
    datasets = _get_vlm_datasets()
    dataset_options = [f"{item['name']} ({item['schema']}, {item['size']})" for item in datasets]
    if dataset_options:
        selected_dataset_label = st.sidebar.selectbox(t("data.select_dataset"), dataset_options, index=0 if dataset_options else None)
        selected_dataset = next((item for item in datasets if selected_dataset_label.startswith(item["name"])), None)
        data_path = selected_dataset["path"] if selected_dataset else ""
    else:
        st.sidebar.warning(t("data.no_datasets"))
        data_path = ""
    manual_data_path = st.sidebar.text_input("Путь к датасету", value=st.session_state.get("vlm_cfg_data_path", data_path or ""))
    if manual_data_path:
        data_path = manual_data_path
        if not Path(data_path).is_absolute() and (DATASET_DIR / data_path).exists():
            data_path = str(DATASET_DIR / data_path)
    st.sidebar.divider()
    st.sidebar.subheader("Validation / Eval")
    val_data_path = st.sidebar.text_input("Val dataset (опционально)", value=st.session_state.get("vlm_cfg_val_data_path", ""))
    eval_steps = st.sidebar.number_input("Eval Every N Steps", min_value=0, max_value=50000, value=int(st.session_state.get("vlm_cfg_eval_steps", 0)), step=10)
    eval_batch_size = st.sidebar.number_input("Eval Batch Size", min_value=1, max_value=64, value=int(st.session_state.get("vlm_cfg_eval_batch_size", 1)))
    assistant_only_loss = st.sidebar.checkbox("Assistant-only loss", value=bool(st.session_state.get("vlm_cfg_assistant_only_loss", selected_stage != "vlm_pretrain")))
    caption_prompt = st.sidebar.text_input("Caption prompt", value=st.session_state.get("vlm_cfg_caption_prompt", "Describe this image."))
    return {
        "data_path": data_path,
        "val_data_path": val_data_path,
        "eval_steps": int(eval_steps),
        "eval_batch_size": int(eval_batch_size),
        "assistant_only_loss": assistant_only_loss,
        "caption_prompt": caption_prompt,
        "data_base_dir": str(Path(data_path).parent) if data_path and Path(data_path).exists() else str(DATASET_DIR),
    }


def _render_vlm_output_config(model_name: str) -> Dict[str, Any]:
    st.sidebar.header(f"💾 {t('sidebar.save')}")
    output_dir = st.sidebar.text_input("Output Directory (Experiment Root)", value=str(OUTPUT_DIR / model_name))
    save_every = st.sidebar.number_input("Save Checkpoint Every N Steps", min_value=1, max_value=5000, value=int(st.session_state.get("vlm_cfg_save_every", 200)), step=10)
    log_every = st.sidebar.number_input("Log Every N Steps", min_value=1, max_value=1000, value=int(st.session_state.get("vlm_cfg_log_every", 10)), step=1)
    save_total_limit = st.sidebar.number_input("Keep last N checkpoints", min_value=1, max_value=20, value=int(st.session_state.get("vlm_cfg_save_total_limit", 2)))
    return {
        "output_dir": output_dir,
        "save_every": int(save_every),
        "log_every": int(log_every),
        "save_total_limit": int(save_total_limit),
    }


def _render_vlm_distributed_config(training_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    st.sidebar.header("🖥️ GPU / Memory")
    gpus = _get_gpu_info()
    if gpus:
        st.sidebar.success(f"✅ GPU найдено: {len(gpus)}")
        for gpu in gpus:
            st.sidebar.markdown(f"**GPU {gpu['id']}**: {gpu['name']}  \nVRAM: {gpu['memory_gb']} GB | CC: {gpu['compute_capability']}")
        gpu_options = [f"GPU {gpu['id']}: {gpu['name']}" for gpu in gpus]
        default_selected = gpu_options[: max(1, int(st.session_state.get('vlm_cfg_num_gpus', min(1, len(gpus)) or 1)))]
        selected_gpus = st.sidebar.multiselect("Выберите GPU", options=gpu_options, default=default_selected)
        gpu_ids = [gpu_options.index(item) for item in selected_gpus]
        num_gpus = len(gpu_ids)
    else:
        st.sidebar.warning("⚠️ GPU не найдены, будет использован CPU")
        gpu_ids = []
        num_gpus = 0
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Тип параллелизма")
    if num_gpus <= 1:
        available_modes = ["default", "deepspeed_zero3_offload"] if num_gpus == 1 else ["default"]
    else:
        available_modes = ["multi_gpu", "deepspeed_zero3", "deepspeed_zero3_offload", "deepspeed_zero2", "fsdp", "default"]
    labels = [f"{PARALLEL_TYPES[mode]['icon']} {PARALLEL_TYPES[mode]['name']}" for mode in available_modes]
    selected_label = st.sidebar.selectbox("Режим", labels, index=0)
    distributed_mode = available_modes[labels.index(selected_label)]
    config_file = None
    if distributed_mode != "default":
        cfg_path = CONFIGS_DIR / f"accelerate_{distributed_mode}.yaml"
        if cfg_path.exists():
            config_file = str(cfg_path)
            st.sidebar.caption(f"📄 Конфиг: `{cfg_path.name}`")
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Precision / Memory")
    mp_options = ["no", "fp16", "bf16"]
    saved_mp = st.session_state.get("vlm_cfg_mixed_precision", "bf16")
    mixed_precision = st.sidebar.selectbox(
        "Mixed Precision",
        mp_options,
        index=_safe_option_index(mp_options, saved_mp, default=2),
    )
    gradient_checkpointing = st.sidebar.checkbox("Gradient Checkpointing", value=bool(st.session_state.get("vlm_cfg_gradient_checkpointing", True)))
    flash_attention = st.sidebar.checkbox("FlashAttention", value=bool(st.session_state.get("vlm_cfg_use_flash_attention", True)))
    min_pixels = st.sidebar.number_input("min_pixels (0=auto)", value=int(st.session_state.get("vlm_cfg_min_pixels", 0)), step=784)
    max_pixels = st.sidebar.number_input("max_pixels (0=auto)", value=int(st.session_state.get("vlm_cfg_max_pixels", 0)), step=784)
    freeze_vision_tower = st.sidebar.checkbox("Freeze vision tower", value=bool(st.session_state.get("vlm_cfg_freeze_vision_tower", True)))
    freeze_projector = st.sidebar.checkbox("Freeze projector", value=bool(st.session_state.get("vlm_cfg_freeze_projector", False)))
    max_new_tokens = st.sidebar.slider("GRPO/chat max new tokens", 32, 512, int(st.session_state.get("vlm_cfg_max_new_tokens", 128)), step=32)
    temperature = st.sidebar.slider("GRPO/chat temperature", 0.0, 1.5, float(st.session_state.get("vlm_cfg_temperature", 0.7)), step=0.1)
    return {
        "distributed_mode": distributed_mode,
        "num_gpus": num_gpus,
        "gpu_ids": gpu_ids,
        "config_file": config_file,
        "mixed_precision": mixed_precision,
        "gradient_checkpointing": gradient_checkpointing,
        "use_flash_attention": flash_attention,
        "attn_implementation": "flash_attention_2" if flash_attention else None,
        "min_pixels": int(min_pixels) if min_pixels > 0 else None,
        "max_pixels": int(max_pixels) if max_pixels > 0 else None,
        "freeze_vision_tower": freeze_vision_tower,
        "freeze_projector": freeze_projector,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
    }


def main() -> None:
    st.set_page_config(page_title=t("vlm.title"), page_icon="🖼️", layout="wide", initial_sidebar_state="expanded")
    init_user_preferences(RUNS_DIR / "ui_preferences.json")
    apply_theme_css(st.session_state.get("ui_theme", DEFAULT_THEME))
    restore_session_state()

    st.title(f"🖼️ {t('vlm.title')}")
    st.caption(t("vlm.subtitle"))

    scenario_names = list(SCENARIO_PRESETS.keys())
    selected_scenario = st.sidebar.selectbox("Сценарий запуска", ["Без пресета"] + scenario_names, key="vlm_scenario_preset")
    if selected_scenario != "Без пресета":
        preset = SCENARIO_PRESETS[selected_scenario]
        st.sidebar.caption(preset["description"])
        if st.sidebar.button("Применить сценарий", key="apply_vlm_scenario"):
            for key, value in preset.items():
                if key != "description":
                    st.session_state[f"vlm_cfg_{key}"] = value
            st.rerun()

    stage_options = [(sid, t(label_key)) for sid, label_key in VLM_STAGES]
    default_stage = st.session_state.get("vlm_cfg_stage", "vlm_sft")
    default_stage_idx = [sid for sid, _ in stage_options].index(default_stage) if default_stage in [sid for sid, _ in stage_options] else 1
    stage_choice = st.sidebar.selectbox(
        t("vlm.sidebar.stage"),
        range(len(stage_options)),
        index=default_stage_idx,
        format_func=lambda idx: stage_options[idx][1],
    )
    selected_stage = stage_options[stage_choice][0]
    model_config = _render_vlm_model_config(selected_stage)
    training_config = _render_vlm_training_config(selected_stage)
    dataset_config = _render_vlm_dataset_config(selected_stage)
    distributed_config = _render_vlm_distributed_config(training_config)
    output_config = _render_vlm_output_config(model_config["experiment_name"])
    if st.sidebar.button(t("vlm.preset.2x3090")):
        st.session_state.vlm_cfg_tuning_method = "qlora"
        st.session_state.vlm_cfg_batch_size = 2
        st.session_state.vlm_cfg_gradient_accumulation = 8
        st.session_state.vlm_cfg_seq_len = 3072
        st.session_state.vlm_cfg_num_gpus = 2
        st.session_state.vlm_cfg_use_flash_attention = True
        st.rerun()
    full_config = {**model_config, **training_config, **dataset_config, **distributed_config, **output_config}

    tab_launch, tab_monitor, tab_chat, tab_history, tab_data, tab_models, tab_docs = st.tabs(
        [
            f"🚀 {t('vlm.tabs.launch')}",
            f"📊 {t('vlm.tabs.monitoring')}",
            f"💬 {t('vlm.tabs.chat')}",
            f"📜 {t('history.title')}",
            f"💾 {t('vlm.tabs.data')}",
            f"🤖 {t('vlm.tabs.models')}",
            f"📚 {t('vlm.tabs.docs')}",
        ]
    )

    with tab_launch:
        with st.expander(t("vlm.best_practices.title"), expanded=False):
            st.markdown(
                "\n".join(
                    [
                        "- Для домашних GPU используйте QLoRA, а не full finetune.",
                        "- Для Qwen 3.5 / Qwen2.5-VL держите включёнными gradient checkpointing и FlashAttention 2.",
                        "- Для small pretrain используйте caption/OCR/VQA корпуса и низкий LR.",
                        "- Для OCR/doc задач поднимайте `max_pixels`, но уменьшайте batch size.",
                    ]
                )
            )
        ready = _render_quick_summary_vlm(full_config)
        _render_model_preview_vlm(full_config)
        st.markdown("---")
        st.subheader("Сценарии и пояснения")
        if selected_scenario != "Без пресета":
            st.info(SCENARIO_PRESETS[selected_scenario]["description"])
        if selected_stage == "vlm_pretrain":
            st.caption("`vlm_pretrain` здесь — это continued multimodal pretraining/alignment, а не обучение с нуля.")
        elif selected_stage == "vlm_grpo":
            st.caption("`vlm_grpo` здесь — экспериментальный reward-guided режим для image-conditioned задач.")
        st.subheader("Полный конфиг")
        st.json(full_config)
        if not ready:
            st.warning("Выберите базовую модель и train dataset.")
        launch_label = {
            "vlm_pretrain": "Запустить VLM Pretrain",
            "vlm_sft": t("vlm.launch.button"),
            "vlm_grpo": "Запустить VLM GRPO",
        }[selected_stage]
        if st.button(launch_label, type="primary", disabled=not ready):
            run_id, process = _start_vlm_training(dict(full_config))
            st.session_state.vlm_current_run_id = run_id
            st.session_state.vlm_training_process = process
            st.session_state.vlm_training_active = True
            st.success(f"Запуск создан: {run_id}")
            st.rerun()

    with tab_monitor:
        runs = _list_vlm_runs()
        if not runs:
            st.info("Нет запусков VLM. Запустите обучение на вкладке «Запуск».")
        else:
            run_options = [run.name for run in runs]
            current_run = st.session_state.get("vlm_current_run_id")
            default_idx = run_options.index(current_run) if current_run in run_options else 0
            selected_run = st.selectbox("Run", run_options, index=default_idx)
            st.session_state.vlm_current_run_id = selected_run
            metrics = _load_metrics(selected_run) or {}
            status = metrics.get("status", "unknown")
            alive = _is_process_running(selected_run)
            c1, c2, c3 = st.columns(3)
            with c1:
                if alive and st.button("⏹ Остановить"):
                    _stop_vlm_training(selected_run)
                    st.rerun()
            with c2:
                if st.button("🔄 Обновить"):
                    st.rerun()
            with c3:
                st.caption(f"Статус: `{status}`")
            step = int(metrics.get("current_step", 0))
            total_steps = int(metrics.get("total_steps", 0) or 0)
            progress = step / total_steps if total_steps > 0 else 0.0
            st.progress(min(progress, 1.0))
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("Step", step)
            with m2:
                st.metric("Loss", f"{metrics.get('current_loss', 0.0):.4f}")
            with m3:
                st.metric("LR", f"{metrics.get('current_lr', 0.0):.2e}")
            with m4:
                st.metric("Reward", f"{metrics.get('current_reward', 0.0):.4f}" if "current_reward" in metrics else "—")
            with m5:
                st.metric("GPU MB", metrics.get("gpu_memory_used_mb", "—"))
            with m6:
                elapsed = int(metrics.get("elapsed_seconds", 0))
                eta = int(metrics.get("eta_seconds", 0))
                st.metric("Время", f"{elapsed // 60} мин", delta=f"ETA {eta // 60} мин" if eta else None)
            if metrics.get("loss_history"):
                try:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    steps = metrics.get("steps_history", list(range(len(metrics["loss_history"]))))
                    fig.add_trace(go.Scatter(x=steps, y=metrics["loss_history"], mode="lines", name="Loss"))
                    if metrics.get("reward_history"):
                        fig.add_trace(go.Scatter(x=steps[: len(metrics["reward_history"])], y=metrics["reward_history"], mode="lines", name="Reward"))
                    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True, key=f"vlm_metrics_{selected_run}")
                except Exception:
                    pass
            if metrics.get("sample_outputs"):
                with st.expander("Сэмплы ответов"):
                    for idx, sample in enumerate(metrics["sample_outputs"][-5:], start=1):
                        st.markdown(f"**Сэмпл {idx}**")
                        st.caption(f"Reward: {sample.get('reward', '—')}")
                        st.code(sample.get("prompt", ""), language="text")
                        st.write(sample.get("response", ""))
            checkpoints = metrics.get("checkpoints", [])
            if checkpoints:
                with st.expander("Чекпоинты"):
                    for idx, ckpt in enumerate(checkpoints):
                        col_ckpt1, col_ckpt2 = st.columns([5, 1])
                        with col_ckpt1:
                            st.caption(f"Step {ckpt.get('step')}: {ckpt.get('path')}")
                        with col_ckpt2:
                            if st.button("🗑️", key=f"del_ckpt_{selected_run}_{idx}"):
                                ok, message = _delete_checkpoint(selected_run, ckpt.get("path", ""))
                                st.toast(message, icon="✅" if ok else "❌")
                                if ok:
                                    st.rerun()
            for title, file_name in [("stdout", "stdout.log"), ("stderr", "stderr.log")]:
                with st.expander(title):
                    log_path = RUNS_DIR / selected_run / file_name
                    if log_path.exists():
                        st.text(log_path.read_text(encoding="utf-8", errors="replace")[-50000:])
                    else:
                        st.caption("Пусто")

    with tab_chat:
        models = get_available_vlm_models()
        if not models:
            st.info("Нет доступных VLM. Скачайте модель на вкладке «Модели» или обучите новую.")
        else:
            model_options = [model["name"] for model in models]
            preselected_model = st.session_state.get("vlm_selected_chat_model") or models[0]["path"]
            default_chat_idx = next((idx for idx, model in enumerate(models) if model["path"] == preselected_model), 0)
            chat_model_idx = st.selectbox("Модель", range(len(model_options)), index=default_chat_idx, format_func=lambda idx: model_options[idx])
            chat_model_path = models[chat_model_idx]["path"]
            system_prompt = st.text_input("System prompt", value="You are a helpful visual assistant.")
            max_tokens = st.slider("Max new tokens", 32, 1024, 256, step=32)
            chat_temp = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)
            image_file = st.file_uploader(t("vlm.chat.upload_image"), type=["png", "jpg", "jpeg", "webp"])
            prompt_text = st.text_input(t("vlm.chat.prompt"))
            if st.button(t("vlm.chat.generate"), type="primary") and image_file and prompt_text:
                from PIL import Image

                image = Image.open(image_file).convert("RGB")
                with st.spinner("Генерация ответа..."):
                    try:
                        answer = _generate_vlm_response(chat_model_path, image, prompt_text, system_prompt, max_tokens, chat_temp)
                        history = st.session_state.setdefault("vlm_chat_messages", [])
                        history.append({"role": "user", "content": prompt_text})
                        history.append({"role": "assistant", "content": answer})
                        st.write(answer)
                    except Exception as exc:
                        import traceback

                        st.error(str(exc))
                        st.code(traceback.format_exc())
            if st.session_state.get("vlm_chat_messages"):
                for idx in range(0, len(st.session_state["vlm_chat_messages"]), 2):
                    pair = st.session_state["vlm_chat_messages"][idx : idx + 2]
                    if pair:
                        st.caption(f"user: {pair[0].get('content', '')}")
                    if len(pair) > 1:
                        st.write(pair[1].get("content", ""))

    with tab_history:
        runs = _list_vlm_runs()
        if not runs:
            st.info("Нет истории VLM запусков.")
        else:
            for run_dir in runs[:30]:
                run_id = run_dir.name
                metrics = _load_metrics(run_id) or {}
                cfg = _load_json(run_dir / "config.json") or {}
                status = metrics.get("status", "unknown")
                status_emoji = {"training": "🟢", "running": "🟢", "completed": "✅", "error": "❌", "stopped": "⏹️"}.get(status, "⏳")
                title = f"{status_emoji} {run_id} | {Path(cfg.get('base_model_path', '')).name or cfg.get('base_model_path', 'model')}"
                with st.expander(title):
                    a1, a2, a3, a4, a5 = st.columns(5)
                    with a1:
                        st.metric("Stage", cfg.get("stage", "—"))
                    with a2:
                        st.metric("Steps", metrics.get("current_step", 0))
                    with a3:
                        st.metric("Loss", f"{metrics.get('current_loss', 0.0):.4f}")
                    with a4:
                        st.metric("Reward", f"{metrics.get('current_reward', 0.0):.4f}" if "current_reward" in metrics else "—")
                    with a5:
                        st.metric("Status", status)
                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        if st.button("📊 Мониторинг", key=f"history_monitor_{run_id}"):
                            st.session_state.vlm_current_run_id = run_id
                            st.rerun()
                    with b2:
                        final_model = Path(cfg.get("output_dir", "")) / "final_model"
                        if not final_model.is_absolute():
                            final_model = PROJECT_ROOT / final_model
                        if final_model.exists() and st.button("💬 Чат", key=f"history_chat_{run_id}"):
                            st.session_state.vlm_selected_chat_model = str(final_model)
                            st.rerun()
                    with b3:
                        checkpoints = [ckpt for ckpt in metrics.get("checkpoints", []) if Path(ckpt.get("path", "")).exists() or (PROJECT_ROOT / ckpt.get("path", "")).exists()]
                        if checkpoints and cfg.get("stage") in {"vlm_sft", "vlm_pretrain"} and st.button("▶️ Resume", key=f"resume_{run_id}"):
                            latest = checkpoints[-1]["path"]
                            resume_cfg = dict(cfg)
                            resume_cfg["resume_from_checkpoint"] = latest
                            new_run_id, _ = _start_vlm_training(resume_cfg)
                            st.session_state.vlm_current_run_id = new_run_id
                            st.rerun()
                    with b4:
                        if st.button("🗑️ Удалить", key=f"delete_run_{run_id}"):
                            ok, message = _delete_experiment(run_id)
                            st.toast(message, icon="✅" if ok else "❌")
                            if ok:
                                st.rerun()

    with tab_data:
        st.subheader(t("vlm.data.download_hf"))
        dataset_labels = [f"{item['name']} — {item['id']}" for item in VLM_HF_DATASETS]
        selected_dataset_idx = st.selectbox("HF dataset", range(len(dataset_labels)), format_func=lambda idx: dataset_labels[idx])
        dataset_out_name = st.text_input("Сохранить как", value="vlm_dataset.jsonl")
        if st.button(t("vlm.data.download_button")):
            dataset_info = VLM_HF_DATASETS[selected_dataset_idx]
            with st.spinner(f"Downloading {dataset_info['id']}..."):
                result = _download_hf_vlm_dataset(dataset_info["id"], dataset_info["split"], dataset_out_name, dataset_info["format"])
            if result and Path(result).exists():
                st.success(f"Сохранено: {result}")
                st.rerun()
            elif result:
                st.error(result)
            else:
                st.warning("Не удалось получить ни одной записи")
        st.markdown("---")
        st.subheader(t("vlm.data.preview"))
        local_datasets = _get_vlm_datasets()
        if not local_datasets:
            st.info("Локальные VLM-датасеты пока не найдены.")
        else:
            dataset_name = st.selectbox("Локальный датасет", [item["name"] for item in local_datasets])
            dataset_meta = next(item for item in local_datasets if item["name"] == dataset_name)
            preview = _read_dataset_preview(Path(dataset_meta["path"]), limit=5)
            st.caption(f"Schema: `{preview['schema']}` | Поля: {', '.join(preview['fields']) if preview['fields'] else '—'}")
            if st.button("Использовать этот датасет", key="use_dataset_for_launch"):
                st.session_state.vlm_cfg_data_path = dataset_meta["path"]
                st.rerun()
            for idx, sample in enumerate(preview["samples"], start=1):
                with st.expander(f"Пример {idx}"):
                    if sample.get("image") and not str(sample["image"]).startswith("http"):
                        image_path = Path(dataset_meta["path"]).parent / str(sample["image"])
                        if image_path.exists():
                            st.image(str(image_path), width=280)
                    st.json(sample)

    with tab_models:
        st.header(f"🤖 {t('vlm.models.available')}")
        col_download, col_list = st.columns([1, 2])
        with col_download:
            st.subheader("🤗 Скачать с HuggingFace")
            preset_names = [preset["name"] for preset in VLM_HF_PRESETS] + ["Ввести вручную"]
            preset_name = st.selectbox("Пресет", preset_names)
            preset = next((item for item in VLM_HF_PRESETS if item["name"] == preset_name), None)
            repo_id = st.text_input("Repo ID", value=preset["repo_id"] if preset else "")
            save_name = st.text_input("Имя папки в models/", value=preset["save_name"] if preset else "")
            if preset:
                st.caption(preset["description"])
            if st.button("Скачать модель", type="primary") and repo_id and save_name:
                if _download_hf_vlm_model(repo_id.strip(), save_name.strip()):
                    st.success("Модель скачана")
                    st.rerun()
        with col_list:
            st.subheader("Локальные и доступные VLM")
            models = get_available_vlm_models()
            for idx, model in enumerate(models):
                with st.expander(model["name"]):
                    st.caption(f"Путь: `{model['path']}`")
                    if st.button("🚀 Использовать в Launch", key=f"use_launch_model_{idx}"):
                        st.session_state.vlm_selected_base_path = model["path"]
                        st.rerun()
                    if st.button("💬 Использовать в Chat", key=f"use_chat_model_{idx}"):
                        st.session_state.vlm_selected_chat_model = model["path"]
                        st.rerun()

    with tab_docs:
        st.caption(t("vlm.docs.source"))
        lang = get_current_language()
        if lang not in ("ru", "en"):
            lang = "en"
        docs_path = STUDY_MATERIALS / lang / "VLM.md"
        if not docs_path.exists():
            docs_path = STUDY_MATERIALS / "en" / "VLM.md"
        if docs_path.exists():
            st.markdown(docs_path.read_text(encoding="utf-8"))
        else:
            st.info("Файл study_materials/.../VLM.md не найден.")


if __name__ == "__main__":
    main()
