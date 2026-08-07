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

VLM_HF_PRESETS: List[Dict[str, Any]] = [
    # --- Ультра-лёгкие (для слабых GPU / быстрых экспериментов) ---
    {
        "name": "SmolVLM-256M Instruct",
        "repo_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "save_name": "SmolVLM-256M-Instruct",
        "params": "256M",
        "vram": "~2 ГБ",
        "task": "Мультизадачный",
        "trust_remote_code": False,
        "description": "Самый маленький современный VLM. Идеален для первого запуска и слабых видеокарт (GTX 1060+).",
    },
    {
        "name": "SmolVLM-500M Instruct",
        "repo_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "save_name": "SmolVLM-500M-Instruct",
        "params": "500M",
        "vram": "~3 ГБ",
        "task": "Мультизадачный",
        "trust_remote_code": False,
        "description": "Компактный VLM с хорошим балансом скорость/качество для домашних задач.",
    },
    {
        "name": "moondream2",
        "repo_id": "vikhyatk/moondream2",
        "save_name": "moondream2",
        "params": "1.9B",
        "vram": "~4 ГБ",
        "task": "Caption / VQA",
        "trust_remote_code": True,
        "description": "Популярная лёгкая модель для описания изображений и ответов на вопросы. Требует trust_remote_code.",
    },
    # --- Основные рабочие лошадки (24GB) ---
    {
        "name": "SmolVLM Instruct (2.2B)",
        "repo_id": "HuggingFaceTB/SmolVLM-Instruct",
        "save_name": "SmolVLM-Instruct",
        "params": "2.2B",
        "vram": "~6 ГБ (QLoRA)",
        "task": "Instruction",
        "trust_remote_code": False,
        "description": "Отличный современный VLM для домашнего instruction tuning. Рекомендуется по умолчанию.",
    },
    {
        "name": "Qwen2-VL 2B Instruct",
        "repo_id": "Qwen/Qwen2-VL-2B-Instruct",
        "save_name": "Qwen2-VL-2B-Instruct",
        "params": "2B",
        "vram": "~8 ГБ (QLoRA)",
        "task": "Instruction / OCR",
        "trust_remote_code": False,
        "description": "Сильный компактный VLM с динамическим разрешением. Хорош для OCR и документов.",
    },
    {
        "name": "InternVL2 2B",
        "repo_id": "OpenGVLab/InternVL2-2B",
        "save_name": "InternVL2-2B",
        "params": "2B",
        "vram": "~8 ГБ (QLoRA)",
        "task": "Мультизадачный",
        "trust_remote_code": True,
        "description": "Мощный компактный VLM из линейки InternVL. Требует trust_remote_code.",
    },
    {
        "name": "PaliGemma2 3B (224)",
        "repo_id": "google/paligemma2-3b-pt-224",
        "save_name": "paligemma2-3b-pt-224",
        "params": "3B",
        "vram": "~10 ГБ (QLoRA)",
        "task": "Caption / VQA",
        "trust_remote_code": False,
        "description": "Модель Google для тонкой настройки под конкретные задачи (caption, VQA, detection).",
    },
    {
        "name": "Qwen2.5-VL 3B Instruct",
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "save_name": "Qwen2.5-VL-3B-Instruct",
        "params": "3B",
        "vram": "~10 ГБ (QLoRA)",
        "task": "Instruction / OCR",
        "trust_remote_code": False,
        "description": "Зрелый instruct-бейзлайн нового поколения. Лучший выбор для OCR/doc/UI задач на 24GB.",
    },
    # --- Для сильных сетапов (2x3090 / 48GB) ---
    {
        "name": "Qwen2.5-VL 7B Instruct",
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "save_name": "Qwen2.5-VL-7B-Instruct",
        "params": "7B",
        "vram": "~18 ГБ (QLoRA)",
        "task": "Instruction / OCR",
        "trust_remote_code": False,
        "description": "Сильный instruct-бейзлайн для 2x3090 и серьёзных задач.",
    },
    {
        "name": "LLaVA 1.5 7B",
        "repo_id": "llava-hf/llava-1.5-7b-hf",
        "save_name": "llava-1.5-7b",
        "params": "7B",
        "vram": "~18 ГБ (QLoRA)",
        "task": "Instruction",
        "trust_remote_code": False,
        "description": "Классический LLaVA — стабильный вариант для image+instruction тюнинга.",
    },
    {
        "name": "LLaVA-NeXT Mistral 7B",
        "repo_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "save_name": "llava-v1.6-mistral-7b",
        "params": "7B",
        "vram": "~20 ГБ (QLoRA)",
        "task": "Instruction",
        "trust_remote_code": False,
        "description": "Улучшенный LLaVA-NeXT с более высоким разрешением.",
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

VLM_DATASET_CATEGORIES = ["Все", "Instruction", "Caption", "VQA", "OCR / Документы"]

VLM_HF_DATASETS: List[Dict[str, Any]] = [
    # --- OCR / Документы (маленькие, идеальны для локального старта) ---
    {
        "id": "nielsr/docvqa_1200_examples",
        "name": "DocVQA (1200 примеров)",
        "category": "OCR / Документы",
        "description": "Крошечный doc-QA набор (1200 примеров). Идеален для первого локального прогона OCR/документов.",
        "split": "test",
        "subset": None,
        "format": "ocr_qa",
        "size": "~200 МБ",
        "recommended_rows": 1200,
    },
    {
        "id": "naver-clova-ix/cord-v2",
        "name": "CORD v2 (чеки)",
        "category": "OCR / Документы",
        "description": "Небольшой набор чеков для OCR/структурированного извлечения. ~1000 примеров.",
        "split": "train",
        "subset": None,
        "format": "image_caption",
        "size": "~200 МБ",
        "recommended_rows": 800,
    },
    # --- VQA (вопрос-ответ по картинке) ---
    {
        "id": "flaviagiammarino/vqa-rad",
        "name": "VQA-RAD (медицина)",
        "category": "VQA",
        "description": "Небольшой медицинский VQA (рентген/снимки). ~2000 примеров, хорош для узкой доменной настройки.",
        "split": "train",
        "subset": None,
        "format": "ocr_qa",
        "size": "~300 МБ",
        "recommended_rows": 2000,
    },
    {
        "id": "derek-thomas/ScienceQA",
        "name": "ScienceQA",
        "category": "VQA",
        "description": "Научный VQA с картинками и вариантами ответов. Хорош для reasoning по изображению.",
        "split": "train",
        "subset": None,
        "format": "ocr_qa",
        "size": "стриминг",
        "recommended_rows": 5000,
    },
    {
        "id": "HuggingFaceM4/VQAv2",
        "name": "VQAv2",
        "category": "VQA",
        "description": "Стандартный крупный VQA-бенчмарк. Качайте с лимитом строк для локального трена.",
        "split": "train",
        "subset": None,
        "format": "ocr_qa",
        "size": "большой (стриминг)",
        "recommended_rows": 10000,
    },
    # --- Caption (описание изображений, для alignment/pretrain) ---
    {
        "id": "nlphuji/flickr30k",
        "name": "Flickr30k",
        "category": "Caption",
        "description": "Классический caption-набор (~31k изображений, по 5 подписей). Отлично для alignment/pretrain.",
        "split": "test",
        "subset": None,
        "format": "image_caption",
        "size": "~4 ГБ",
        "recommended_rows": 5000,
    },
    {
        "id": "HuggingFaceM4/COCO",
        "name": "COCO Captions",
        "category": "Caption",
        "description": "Крупный caption корпус для small-task pretrain/alignment. Качайте с лимитом.",
        "split": "train",
        "subset": None,
        "format": "image_caption",
        "size": "большой (стриминг)",
        "recommended_rows": 10000,
    },
    # --- Instruction (мультимодальные диалоги) ---
    {
        "id": "HuggingFaceH4/llava-instruct-mix-vsft",
        "name": "LLaVA Instruct Mix",
        "category": "Instruction",
        "description": "Основной single-image instruction tuning корпус. Стандарт для VLM SFT.",
        "split": "train",
        "subset": None,
        "format": "image_messages",
        "size": "большой (стриминг)",
        "recommended_rows": 10000,
    },
    {
        "id": "TIGER-Lab/VisualWebInstruct",
        "name": "VisualWebInstruct",
        "category": "Instruction",
        "description": "Разнообразные мультимодальные инструкции из web-контента.",
        "split": "train",
        "subset": None,
        "format": "image_messages",
        "size": "большой (стриминг)",
        "recommended_rows": 8000,
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


def _coerce_text(value: Any) -> Optional[str]:
    """Reduce a field to a single readable string.

    Handles the common messy shapes found in HF datasets: lists (take first
    non-empty), multilingual dicts like ``{'en': ..., 'de': ...}`` (prefer
    English), and plain scalars.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("en", "english", "text", "value"):
            if value.get(key):
                return str(value[key]).strip() or None
        for v in value.values():
            if v:
                return str(v).strip() or None
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = _coerce_text(item)
            if coerced:
                return coerced
        return None
    text = str(value).strip()
    return text or None


def _normalize_caption(value: Any) -> Optional[str]:
    """Captions come as str or list[str] (e.g. Flickr30k has 5 captions)."""
    return _coerce_text(value)


def _extract_row_image(row: Dict[str, Any]) -> Any:
    for key in ("image", "images", "img", "picture"):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            return val[0] if val else None
        return val
    return None


def _dataset_output_record(
    row: Dict[str, Any],
    dataset_id: str,
    idx: int,
    format_type: str,
    images_dir: Path,
) -> Optional[Dict[str, Any]]:
    image = _extract_row_image(row)
    if image is None:
        return None

    if hasattr(image, "save"):
        file_name = f"{dataset_id.replace('/', '_')}_{idx}.png"
        try:
            image.convert("RGB").save(images_dir / file_name)
        except Exception:
            try:
                image.save(images_dir / file_name)
            except Exception:
                return None
        image_ref = f"{images_dir.name}/{file_name}"
    else:
        image_ref = str(image)

    if format_type == "image_messages":
        messages = row.get("messages") or row.get("conversations")
        if not messages:
            question = _coerce_text(row.get("question") or row.get("query") or row.get("prompt"))
            answer = _coerce_text(row.get("answer") or row.get("answers") or row.get("output") or row.get("response"))
            if question and answer:
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
        if not messages:
            return None
        return {"image": image_ref, "messages": messages, "schema": "single_image_messages"}

    if format_type == "image_caption":
        caption = _normalize_caption(
            row.get("caption")
            or row.get("caption_text")
            or row.get("caption_gt")
            or row.get("captions")
            or row.get("text")
            or row.get("label")
        )
        if caption:
            return {"image": image_ref, "caption": caption, "schema": "single_image_caption"}
        return None

    if format_type == "ocr_qa":
        question = _coerce_text(row.get("question") or row.get("query") or row.get("prompt"))
        answer = _coerce_text(row.get("answer") or row.get("answers") or row.get("response") or row.get("output"))
        if question and answer:
            return {
                "image": image_ref,
                "question": question,
                "answer": answer,
                "schema": "ocr_or_doc_qa",
            }
    return None


def _inspect_hf_dataset(repo_id: str) -> tuple[Optional[Dict[str, Any]], str]:
    """Fetch available configs/subsets and splits for a HF dataset repo.

    Returns (info_dict_or_None, message). Mirrors the "check repo" behaviour of
    the LLM studio so the user can pick a real subset/split before downloading.
    """
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names
    except ImportError:
        return None, "Библиотека 'datasets' не установлена"

    try:
        configs = get_dataset_config_names(repo_id)
    except Exception as exc:
        return None, f"Не удалось получить конфиги: {exc}"

    selected_config = configs[0] if configs else None
    splits: List[str] = []
    if selected_config is not None:
        try:
            splits = get_dataset_split_names(repo_id, selected_config)
        except Exception:
            splits = []
    else:
        try:
            splits = get_dataset_split_names(repo_id)
        except Exception:
            splits = []

    return (
        {"configs": configs, "splits": splits, "selected_config": selected_config},
        f"Найдено конфигов: {len(configs)}, splits: {splits or '—'}",
    )


def _download_hf_vlm_dataset(
    hf_id: str,
    split: str,
    out_name: str,
    format_type: str,
    subset: Optional[str] = None,
    limit_type: str = "rows",
    max_rows: int = 0,
    max_bytes: int = 0,
    progress_cb=None,
) -> tuple[Optional[str], str]:
    """Download and convert an HF VLM dataset to local JSONL + images.

    Streaming keeps memory/disk bounded. ``limit_type`` selects whether the
    download is capped by row count (``"rows"`` + ``max_rows``) or by on-disk
    size in bytes (``"gb"`` + ``max_bytes``, counting JSONL text plus images).
    Returns (path_or_None, message).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return None, "Библиотека 'datasets' не установлена"

    base_name = (out_name or hf_id.split("/")[-1]).replace(".jsonl", "")
    out_path = DATASET_DIR / f"{base_name}.jsonl"
    images_dir = DATASET_DIR / f"{base_name}_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    subset_arg = None if (not subset or subset.strip() == "" or subset.strip().lower() == "default") else subset.strip()
    if subset_arg:
        load_kwargs["name"] = subset_arg

    try:
        dataset = load_dataset(hf_id, **load_kwargs)
    except Exception as exc:
        return None, f"Не удалось открыть датасет: {exc}"

    written = 0
    scanned = 0
    total_bytes = 0
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, row in enumerate(dataset):
                scanned += 1
                try:
                    record = _dataset_output_record(row, hf_id, idx, format_type, images_dir)
                except Exception:
                    record = None
                if record:
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    line_bytes = len(line.encode("utf-8"))
                    img_bytes = 0
                    img_ref = record.get("image")
                    if img_ref and not str(img_ref).startswith(("http://", "https://")):
                        img_path = DATASET_DIR / str(img_ref)
                        if img_path.exists():
                            img_bytes = img_path.stat().st_size

                    if limit_type == "gb" and max_bytes > 0 and (total_bytes + line_bytes + img_bytes) > max_bytes:
                        # Would exceed the size cap — drop the just-saved image and stop.
                        if img_bytes:
                            (DATASET_DIR / str(img_ref)).unlink(missing_ok=True)
                        break

                    f.write(line)
                    written += 1
                    total_bytes += line_bytes + img_bytes
                    if progress_cb and written % 25 == 0:
                        progress_cb(written, total_bytes)
                    if limit_type == "rows" and max_rows > 0 and written >= max_rows:
                        break
                if scanned > 200 and written == 0:
                    # Format almost certainly mismatched — bail out early.
                    break
    except Exception as exc:
        return None, f"Ошибка при конвертации: {exc}"

    if written == 0:
        shutil.rmtree(images_dir, ignore_errors=True)
        out_path.unlink(missing_ok=True)
        return None, "Не удалось извлечь ни одного примера (проверьте формат/split)."

    size_mb = total_bytes / (1024 * 1024)
    return str(out_path), f"Сохранено {written} примеров ({size_mb:.1f} МБ вкл. картинки) → {out_path.name}"


def _estimate_jsonl_rows(path: Path, file_size: int | None = None) -> tuple[int, bool]:
    """Estimate the number of rows in a JSONL file without reading it whole.

    Returns ``(count, exact)``. For small files we count precisely; for large
    ones we sample the first lines and extrapolate from the average line size.
    This keeps the datasets listing fast even with multi-GB files in datasets/.
    """
    try:
        if file_size is None:
            file_size = path.stat().st_size
        if file_size == 0:
            return 0, True
        # Small files: exact count is cheap.
        if file_size <= 2 * 1024 * 1024:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for line in f if line.strip()), True
        # Large files: sample the first lines and extrapolate.
        sampled_bytes = 0
        sampled_lines = 0
        with open(path, "rb") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    sampled_bytes += len(line)
                    sampled_lines += 1
        if sampled_lines == 0 or sampled_bytes == 0:
            return 0, False
        avg = sampled_bytes / sampled_lines
        return max(1, int(file_size / avg)), False
    except Exception:
        return 0, False


def _record_has_image(record: Any) -> bool:
    """True if a JSONL record references at least one image (any common schema)."""
    if not isinstance(record, dict):
        return False
    for key in ("image", "images", "image_path", "input_image_path", "image_url"):
        value = record.get(key)
        if value:
            return True
    for conv_key in ("messages", "conversations"):
        messages = record.get(conv_key)
        if isinstance(messages, list):
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and (
                            item.get("type") in {"image", "image_url"} or item.get("image") or item.get("image_url")
                        ):
                            return True
    return False


def _dataset_has_images(path: Path, sample_size: int = 8) -> bool:
    """Detect whether a dataset is multimodal.

    A dataset counts as VLM if an accompanying ``{stem}_images`` directory exists
    or any of the first records reference an image. This is what actually decides
    whether ``VLMJsonlDataset`` will yield any training examples.
    """
    if (DATASET_DIR / f"{path.stem}_images").is_dir():
        return True
    preview = _read_dataset_preview(path, limit=sample_size)
    return any(_record_has_image(rec) for rec in preview.get("samples", []))


def _get_vlm_datasets() -> List[Dict[str, Any]]:
    datasets = []
    for path in sorted(DATASET_DIR.glob("*.jsonl")):
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        preview = _read_dataset_preview(path, limit=8)
        count, exact = _estimate_jsonl_rows(path, size_bytes)
        has_images = _dataset_has_images(path)
        datasets.append(
            {
                "name": path.name,
                "path": str(path),
                "size": f"{size_mb:.1f} MB",
                "schema": preview.get("schema", "unknown") if has_images else "⚠️ без изображений",
                "count": count,
                "count_exact": exact,
                "has_images": has_images,
            }
        )
    # VLM datasets (with images) first, so the default selection is a valid one.
    datasets.sort(key=lambda d: (not d["has_images"], d["name"]))
    return datasets


def _delete_vlm_dataset(path: str) -> tuple[bool, str]:
    """Delete a dataset JSONL and its associated images directory (if any)."""
    try:
        p = Path(path)
        if p.resolve().parent != DATASET_DIR.resolve():
            return False, "Можно удалять только файлы внутри datasets/"
        if p.exists():
            p.unlink()
        images_dir = DATASET_DIR / f"{p.stem}_images"
        if images_dir.exists() and images_dir.is_dir():
            shutil.rmtree(images_dir, ignore_errors=True)
        return True, "Датасет удалён"
    except Exception as exc:
        return False, str(exc)


def _delete_vlm_model(path: str) -> tuple[bool, str]:
    """Delete a locally downloaded model directory inside models/."""
    try:
        p = Path(path)
        if not str(p.resolve()).startswith(str(MODELS_DIR.resolve())):
            return False, "Можно удалять только модели внутри models/"
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            return True, "Модель удалена"
        return False, "Папка не найдена"
    except Exception as exc:
        return False, str(exc)


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


def _model_available_locally(model_ref: str) -> bool:
    """True if the model is a local dir or already in the HF hub cache.

    Used to decide whether we can safely run training with HF offline flags.
    """
    if not model_ref:
        return False
    p = Path(model_ref)
    if p.exists() and (p / "config.json").exists():
        return True
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    cache_dir = Path(hf_home) / "hub" / ("models--" + model_ref.replace("/", "--"))
    if cache_dir.exists() and any(cache_dir.glob("snapshots/*/config.json")):
        return True
    return False


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
    # If the base model is already local/cached, force HF offline so from_pretrained
    # never hangs contacting huggingface.co (model download is a separate step).
    base_model = config.get("base_model_path") or config.get("model_name_or_path") or ""
    if _model_available_locally(base_model):
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    data_exists = bool(data_path and Path(data_path).exists())
    has_images = data_exists and _dataset_has_images(Path(data_path))
    if not data_exists:
        data_display = "❌ Не выбрано"
    elif not has_images:
        data_display = f"⚠️ {Path(data_path).name} (нет изображений)"
    else:
        data_display = f"🖼️ {Path(data_path).name}"
    stage_display = t(dict(VLM_STAGES).get(config.get("stage", "vlm_sft"), config.get("stage", "vlm_sft")))
    all_ready = bool(config.get("base_model_path")) and data_exists and has_images
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


def render_vlm_sft_main_config(data_path: str) -> Dict[str, Any]:
    """Field-mapping configurator for VLM datasets (mirrors LLM SFT config).

    Lets the user pick which columns hold the image / question / answer / caption /
    messages and previews the resulting prompt + image. Returns ``{"vlm_columns": ...}``
    to be merged into the training config.
    """
    st.markdown("### 🛠️ Конфигурация полей датасета")
    if not data_path or not Path(data_path).exists():
        st.info("Выберите мультимодальный датасет в сайдбаре, чтобы настроить поля.")
        return {}

    preview = _read_dataset_preview(Path(data_path), limit=1)
    samples = preview.get("samples", [])
    if not samples:
        st.error("Не удалось прочитать пример из датасета.")
        return {}
    sample = samples[0]
    all_keys = list(sample.keys())
    list_fields = [k for k, v in sample.items() if isinstance(v, list)]
    image_candidates = [k for k in all_keys if k.lower() in ("image", "images", "image_path", "input_image_path", "image_url")]

    if sample.get("messages") or sample.get("conversations"):
        auto_format = "messages"
    elif sample.get("caption"):
        auto_format = "caption"
    else:
        auto_format = "vqa"

    col_json, col_cfg = st.columns([1, 1])
    with col_json:
        st.markdown("#### 📄 Пример записи:")
        with st.container(height=380):
            st.json(sample, expanded=True)

    with col_cfg:
        fmt_labels = {
            "vqa": "❓ VQA (вопрос/ответ)",
            "caption": "🖼️ Caption (описание)",
            "messages": "💬 Chat (сообщения)",
        }
        fmt_keys = list(fmt_labels.keys())
        labels = [fmt_labels[k] for k in fmt_keys]
        fmt_choice = st.radio(
            "Формат данных:", labels, index=fmt_keys.index(auto_format), horizontal=True, key="vlm_sft_format"
        )
        fmt = fmt_keys[labels.index(fmt_choice)]

        img_options = image_candidates or all_keys
        image_field = st.selectbox("🖼️ Поле с изображением:", img_options, index=0, key="vlm_sft_image_field")

        vlm_columns: Dict[str, Any] = {"format": fmt, "image_field": image_field}

        if fmt == "vqa":
            q_guess = next((k for k in all_keys if k.lower() in ("question", "prompt", "instruction", "query")), all_keys[0])
            a_guess = next((k for k in all_keys if k.lower() in ("answer", "response", "output", "completion", "label")), all_keys[-1])
            c1, c2 = st.columns(2)
            question_field = c1.selectbox("👤 Вопрос (user):", all_keys, index=all_keys.index(q_guess), key="vlm_sft_q")
            answer_field = c2.selectbox("🤖 Ответ (assistant):", all_keys, index=all_keys.index(a_guess), key="vlm_sft_a")
            vlm_columns.update(question_field=question_field, answer_field=answer_field)
        elif fmt == "caption":
            cap_guess = next((k for k in all_keys if k.lower() in ("caption", "text", "description", "answer")), all_keys[-1])
            caption_field = st.selectbox("🖼️ Поле описания:", all_keys, index=all_keys.index(cap_guess), key="vlm_sft_cap")
            caption_prompt = st.text_input(
                "Промпт для caption:", st.session_state.get("vlm_cfg_caption_prompt", "Describe this image."), key="vlm_sft_capprompt"
            )
            vlm_columns.update(caption_field=caption_field, caption_prompt=caption_prompt)
        else:
            if not list_fields:
                st.error("В записи нет поля-списка сообщений — выберите другой формат.")
                return {}
            msg_guess = next((k for k in list_fields if k.lower() in ("messages", "conversations")), list_fields[0])
            messages_field = st.selectbox("💬 Поле сообщений:", list_fields, index=list_fields.index(msg_guess), key="vlm_sft_msgs")
            vlm_columns.update(messages_field=messages_field)

        st.markdown("---")
        st.markdown("#### 👁️ Превью:")
        pc1, pc2 = st.columns([1, 2])
        with pc1:
            try:
                img_ref = sample.get(image_field)
                if isinstance(img_ref, list):
                    img_ref = img_ref[0] if img_ref else None
                if isinstance(img_ref, str) and img_ref:
                    img_path = img_ref
                    if not Path(img_path).is_absolute() and not img_path.startswith(("http://", "https://")):
                        img_path = str(Path(data_path).parent / img_ref)
                    st.image(img_path, use_container_width=True)
                else:
                    st.caption("Изображение недоступно для превью")
            except Exception as exc:
                st.caption(f"Изображение недоступно: {exc}")
        with pc2:
            try:
                if fmt == "vqa":
                    st.markdown(f"**👤 User:** {_coerce_text(sample.get(vlm_columns['question_field']))}")
                    st.markdown(f"**🤖 Assistant:** {_coerce_text(sample.get(vlm_columns['answer_field']))}")
                elif fmt == "caption":
                    st.markdown(f"**👤 User:** {vlm_columns['caption_prompt']}")
                    st.markdown(f"**🤖 Assistant:** {_coerce_text(sample.get(vlm_columns['caption_field']))}")
                else:
                    st.json(sample.get(vlm_columns["messages_field"]))
            except Exception as exc:
                st.caption(f"Превью текста недоступно: {exc}")

    return {"vlm_columns": vlm_columns}


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
    vlm_datasets = [d for d in datasets if d["has_images"]]
    dataset_options = [
        f"{'🖼️' if item['has_images'] else '⚠️'} {item['name']} ({item['schema']}, {item['size']})"
        for item in datasets
    ]
    if dataset_options:
        selected_dataset_label = st.sidebar.selectbox(t("data.select_dataset"), dataset_options, index=0)
        selected_dataset = next((item for item in datasets if selected_dataset_label.endswith(f"({item['schema']}, {item['size']})") and item["name"] in selected_dataset_label), None)
        data_path = selected_dataset["path"] if selected_dataset else ""
        if not vlm_datasets:
            st.sidebar.warning(
                "Нет мультимодальных датасетов (с изображениями). Скачайте VLM-датасет на вкладке «Данные» "
                "(например, DocVQA/VQA/caption) — VLM-обучению нужны пары изображение+текст."
            )
    else:
        st.sidebar.warning(t("data.no_datasets"))
        data_path = ""
    manual_data_path = st.sidebar.text_input("Путь к датасету", value=st.session_state.get("vlm_cfg_data_path", data_path or ""))
    if manual_data_path:
        data_path = manual_data_path
        if not Path(data_path).is_absolute() and (DATASET_DIR / data_path).exists():
            data_path = str(DATASET_DIR / data_path)
    # Guardrail: VLM training silently yields 0 examples on text-only data.
    if data_path and Path(data_path).exists() and not _dataset_has_images(Path(data_path)):
        st.sidebar.error(
            f"❌ В датасете `{Path(data_path).name}` нет изображений. "
            "VLM-обучение работает только с данными изображение+текст — выберите мультимодальный датасет."
        )
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
    flash_attention = st.sidebar.checkbox(
        "FlashAttention",
        value=bool(st.session_state.get("vlm_cfg_use_flash_attention", True)),
        help=(
            "Как в LLM Studio: включает flash-ядра. Для fp16/bf16 + LoRA/full пробует пакет "
            "flash_attention_2, для QLoRA и в остальных случаях использует SDPA flash-ядра PyTorch "
            "(это тоже FlashAttention-2). Выкл. — обычный (eager) attention."
        ),
    )
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


try:
    _st_fragment = st.fragment
except AttributeError:  # older Streamlit
    try:
        _st_fragment = st.experimental_fragment
    except AttributeError:
        _st_fragment = lambda *a, **k: (lambda fn: fn)  # no-op fallback


def _vlm_fmt_params(n: Any) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def _vlm_fmt_time(seconds: Any) -> str:
    try:
        seconds = int(float(seconds))
    except (TypeError, ValueError):
        return "—"
    if seconds <= 0:
        return "0с"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}ч {m}м"
    if m:
        return f"{m}м {s}с"
    return f"{s}с"


@_st_fragment(run_every=2)
def _vlm_live_monitor() -> None:
    """Auto-refreshing (every 2s) live dashboard for the selected VLM run.

    Mirrors the LLM Studio monitoring: status header, 6 metric cards, progress
    bar, Loss/LR charts (Reward/KL for GRPO), GPU load, checkpoints, samples, logs.
    """
    import plotly.graph_objects as go

    selected_run = st.session_state.get("vlm_current_run_id")
    if not selected_run:
        st.info("Выберите run выше.")
        return
    metrics = _load_metrics(selected_run) or {}
    alive = _is_process_running(selected_run)
    status = metrics.get("status") or ("training" if alive else "unknown")
    status = str(status)
    stage = str(metrics.get("stage", "vlm_sft"))
    is_grpo = "grpo" in stage

    status_emoji = {
        "training": "🟢",
        "completed": "✅",
        "error": "❌",
        "stopped": "⏹️",
        "starting": "⏳",
        "initializing": "⏳",
        "loading_model": "⏳",
        "loading_dataset": "⏳",
        "saving_model": "💾",
    }.get(status, "⏳")

    head_l, head_r = st.columns([0.8, 0.2])
    with head_l:
        st.subheader(f"{status_emoji} Статус: {status.upper()}")
    with head_r:
        if alive and st.button("⏹ Остановить", key=f"vlm_stop_{selected_run}", use_container_width=True):
            _stop_vlm_training(selected_run)
            st.rerun()

    if alive and status in ("starting", "initializing", "loading_model", "loading_dataset"):
        st.info(
            "⏳ Подготовка… Первый QLoRA-запуск после старта контейнера компилирует CUDA-ядра "
            "(bitsandbytes/Triton) — это может занять несколько минут. Дальше запуски идут быстро."
        )
    elif status == "completed":
        st.success(f"✅ Завершено — {metrics.get('training_duration', '')}")
    elif status == "error" and metrics.get("error"):
        with st.expander("❌ Текст ошибки", expanded=True):
            st.code(str(metrics["error"])[:4000])
    elif status == "stopped":
        st.warning("⏹️ Остановлено пользователем")

    current_step = int(metrics.get("current_step", 0) or 0)
    total_steps = int(metrics.get("total_steps", 0) or 0)
    progress = (current_step / total_steps * 100) if total_steps > 0 else 0.0
    progress = max(0.0, min(progress, 100.0))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        suffix = f"Step {current_step}/{total_steps}" if total_steps else f"Step {current_step}"
        st.metric("Прогресс", f"{progress:.1f}%", suffix)
    with c2:
        if is_grpo:
            st.metric("Reward", f"{metrics.get('current_reward', 0.0):.4f}")
        else:
            st.metric("Train Loss", f"{metrics.get('current_loss', 0.0):.4f}")
    with c3:
        if is_grpo:
            st.metric("KL", f"{metrics.get('current_kl', 0.0):.4f}" if "current_kl" in metrics else "—")
        else:
            vloss = metrics.get("current_val_loss")
            st.metric("Val Loss", "—" if vloss is None else f"{vloss:.4f}")
    with c4:
        st.metric("Learning Rate", f"{metrics.get('current_lr', 0.0):.2e}")
    with c5:
        tp = metrics.get("trainable_params") or metrics.get("num_parameters")
        st.metric("Параметры", _vlm_fmt_params(tp) if tp else "—")
    with c6:
        st.metric(
            "Время",
            _vlm_fmt_time(metrics.get("elapsed_seconds", 0)),
            delta=f"ETA {_vlm_fmt_time(metrics.get('eta_seconds', 0))}" if metrics.get("eta_seconds") else None,
        )

    st.progress(
        min(progress / 100, 1.0),
        text=f"Шаг {current_step} / {total_steps}" if total_steps else "Ожидание первого шага…",
    )
    st.caption(f"Attention: `{metrics.get('attn_implementation', '—')}` · Этап: `{stage}`")

    if metrics.get("loss_history"):
        steps_history = metrics.get("steps_history", list(range(1, len(metrics["loss_history"]) + 1)))
        cc1, cc2 = st.columns(2)
        with cc1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=steps_history, y=metrics["loss_history"], mode="lines", name="Train Loss", line=dict(color="#e94560", width=2)))
            if metrics.get("val_loss_history"):
                fig_loss.add_trace(go.Scatter(x=metrics.get("val_steps_history", []), y=metrics["val_loss_history"], mode="lines", name="Val Loss", line=dict(dash="dash", color="#60a5fa", width=2)))
            fig_loss.update_layout(title="Training Loss", xaxis_title="Step", yaxis_title="Loss", template="plotly_dark", height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_loss, use_container_width=True, key=f"vlm_loss_{selected_run}")
        with cc2:
            lr_history = metrics.get("lr_history", [])
            if lr_history:
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(x=steps_history[: len(lr_history)], y=lr_history, mode="lines", name="LR", line=dict(color="#60a5fa", width=2)))
                fig_lr.update_layout(title="Learning Rate Schedule", xaxis_title="Step", yaxis_title="LR", template="plotly_dark", height=300, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_lr, use_container_width=True, key=f"vlm_lr_{selected_run}")
        if is_grpo and (metrics.get("reward_history") or metrics.get("kl_history")):
            rc1, rc2 = st.columns(2)
            with rc1:
                rh = metrics.get("reward_history", [])
                if rh:
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Scatter(x=steps_history[: len(rh)], y=rh, mode="lines", name="Reward", line=dict(color="#10b981", width=2)))
                    fig_r.update_layout(title="🎯 Reward (GRPO)", xaxis_title="Step", yaxis_title="Reward", template="plotly_dark", height=300, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_r, use_container_width=True, key=f"vlm_reward_{selected_run}")
            with rc2:
                kh = metrics.get("kl_history", [])
                if kh:
                    fig_k = go.Figure()
                    fig_k.add_trace(go.Scatter(x=steps_history[: len(kh)], y=kh, mode="lines", name="KL", line=dict(color="#f59e0b", width=2)))
                    fig_k.update_layout(title="📊 KL Divergence (GRPO)", xaxis_title="Step", yaxis_title="KL", template="plotly_dark", height=300, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_k, use_container_width=True, key=f"vlm_kl_{selected_run}")
    else:
        st.info("📊 Графики появятся после первого залогированного шага.")

    gpu_stats = metrics.get("gpu_stats", [])
    if gpu_stats:
        st.subheader("🖥️ Загрузка GPU")
        gcols = st.columns(len(gpu_stats))
        for col, gpu in zip(gcols, gpu_stats):
            used = float(gpu.get("memory_used_gb", 0.0))
            total = float(gpu.get("memory_total_gb", 0.0))
            with col:
                st.metric(f"GPU {gpu.get('id', 0)}", f"{used:.1f} / {total:.1f} GB")
                if total > 0:
                    st.progress(min(used / total, 1.0))

    if metrics.get("sample_outputs"):
        with st.expander("🖼️ Сэмплы ответов"):
            for idx, sample in enumerate(metrics["sample_outputs"][-5:], start=1):
                st.markdown(f"**Сэмпл {idx}** · Reward: {sample.get('reward', '—')}")
                st.code(sample.get("prompt", ""), language="text")
                st.write(sample.get("response", ""))

    checkpoints = metrics.get("checkpoints", [])
    if checkpoints:
        with st.expander("📦 Чекпоинты"):
            for idx, ckpt in enumerate(checkpoints):
                col_ckpt1, col_ckpt2 = st.columns([5, 1])
                loss_val = ckpt.get("loss")
                label = f"Step {ckpt.get('step')}: "
                if loss_val is not None:
                    label += f"Loss {loss_val:.4f} | "
                label += str(ckpt.get("path"))
                col_ckpt1.caption(label)
                if col_ckpt2.button("🗑️", key=f"del_ckpt_{selected_run}_{idx}"):
                    ok, message = _delete_checkpoint(selected_run, ckpt.get("path", ""))
                    st.toast(message, icon="✅" if ok else "❌")
                    if ok:
                        st.rerun()

    for title, file_name in [("📄 stderr (лог)", "stderr.log"), ("📄 stdout", "stdout.log")]:
        with st.expander(title):
            log_path = RUNS_DIR / selected_run / file_name
            if log_path.exists():
                st.text(log_path.read_text(encoding="utf-8", errors="replace")[-50000:])
            else:
                st.caption("Лог пуст")


def main() -> None:
    q = st.query_params
    if "run_id" in q:
        target_run = q["run_id"]
        
        # Check if this is a TEXT run and redirect if necessary
        is_text = False
        config_path = RUNS_DIR / target_run / "config.json"
        if config_path.exists():
            import json
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if not str(cfg.get("stage", "")).startswith("vlm_"):
                        is_text = True
            except:
                pass
                
        if is_text:
            st.markdown(f'<meta http-equiv="refresh" content="0;url=/?run_id={target_run}">', unsafe_allow_html=True)
            return

        if st.session_state.get("vlm_last_processed_run_id") != target_run:
            st.session_state.vlm_current_run_id = target_run
            st.session_state.vlm_last_processed_run_id = target_run
            st.toast(f"📊 Выбран run: {target_run}. Перейдите на вкладку 'Мониторинг'!", icon="✅")

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
        col1, col2 = st.columns([2, 1])

        with col1:
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

            _dp = full_config.get("data_path") or ""
            if _dp and Path(_dp).exists() and _dataset_has_images(Path(_dp)):
                st.markdown("---")
                field_cfg = render_vlm_sft_main_config(_dp)
                if field_cfg:
                    full_config.update(field_cfg)

            if selected_scenario != "Без пресета":
                st.markdown("---")
                st.info(SCENARIO_PRESETS[selected_scenario]["description"])
            if selected_stage == "vlm_pretrain":
                st.caption("`vlm_pretrain` здесь — это continued multimodal pretraining/alignment, а не обучение с нуля.")
            elif selected_stage == "vlm_grpo":
                st.caption("`vlm_grpo` здесь — экспериментальный reward-guided режим для image-conditioned задач.")

            st.subheader(f"📋 {t('common.configuration')}")
            st.json(full_config)

        with col2:
            st.subheader(f"🎮 {t('common.control')}")

            if st.session_state.get("vlm_training_active"):
                st.info("🚀 Обучение запущено. Открой вкладку «📊 Мониторинг» — метрики обновляются live.")
                if st.button(f"⏹️ {t('button.stop_training')}", type="primary"):
                    run_id = st.session_state.get("vlm_current_run_id")
                    with st.spinner("Останавливаем тренировку..."):
                        stopped = _stop_vlm_training(run_id) if run_id else False
                    st.session_state.vlm_training_active = False
                    clear_active_run()
                    if stopped:
                        st.success(f"✅ {t('status.stopped')}")
                    else:
                        st.warning(f"⚠️ {t('warning.stop_failed')}")
                    time.sleep(1)
                    st.rerun()
            else:
                launch_label = {
                    "vlm_pretrain": "▶️ Запустить VLM Pretrain",
                    "vlm_sft": f"▶️ {t('vlm.launch.button')}",
                    "vlm_grpo": "🧠 Запустить VLM GRPO",
                }[selected_stage]
                button_disabled = not ready
                if st.button(launch_label, type="primary", disabled=button_disabled):
                    with st.spinner("Запуск..."):
                        run_id, process = _start_vlm_training(dict(full_config))
                        st.session_state.vlm_current_run_id = run_id
                        st.session_state.vlm_training_process = process
                        st.session_state.vlm_training_active = True
                        st.success(f"Обучение запущено! Run ID: {run_id}")
                        time.sleep(1)
                        st.rerun()
                if button_disabled:
                    _dp = full_config.get("data_path") or ""
                    if _dp and Path(_dp).exists() and not _dataset_has_images(Path(_dp)):
                        st.error(
                            f"❌ Датасет `{Path(_dp).name}` без изображений — VLM-обучение невозможно. "
                            "Выберите мультимодальный датасет (изображение+текст) на вкладке «Данные»."
                        )
                    else:
                        st.caption("⚠️ Выберите базовую модель и мультимодальный train dataset (с изображениями) для запуска")

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
            st.caption("🔄 Метрики обновляются автоматически каждые 2 секунды")
            _vlm_live_monitor()

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
                        st.markdown(f"[🔗 Ссылка на процесс](/VLM_Studio?run_id={run_id})")
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
        st.header(f"💾 {t('vlm.tabs.data')}")
        col_dl, col_list = st.columns([1, 2])

        with col_dl:
            with st.expander("📤 Загрузить свой файл (JSONL)", expanded=False):
                uploaded = st.file_uploader(
                    "JSONL (image + messages / caption / question+answer)",
                    type=["jsonl"],
                    key="vlm_upload_ds",
                    help="Строки JSON с полем 'image' (путь/URL) и 'messages'/'caption'/'question'+'answer'.",
                )
                if uploaded is not None:
                    up_name = st.text_input("Имя файла", value=uploaded.name, key="vlm_upload_name")
                    if st.button("📥 Сохранить файл", key="vlm_upload_save"):
                        target = DATASET_DIR / (up_name if up_name.endswith(".jsonl") else up_name + ".jsonl")
                        with open(target, "wb") as f:
                            f.write(uploaded.getbuffer())
                        st.toast(f"Файл {target.name} сохранён!", icon="✅")
                        time.sleep(0.5)
                        st.rerun()

            st.subheader("🤗 Скачать с HuggingFace")

            selected_cat = st.selectbox(
                "Категория задачи",
                VLM_DATASET_CATEGORIES,
                key="vlm_ds_category",
                help="Фильтр курируемого списка небольших VLM-датасетов по типу задачи.",
            )
            filtered_datasets = [
                item for item in VLM_HF_DATASETS
                if selected_cat == "Все" or item.get("category") == selected_cat
            ]

            preset_by_label: Dict[str, Dict[str, Any]] = {
                f"{d['name']} · {d.get('size', '?')}": d for d in filtered_datasets
            }
            preset_labels = list(preset_by_label.keys()) + ["✏️ Ввести вручную..."]

            format_labels = {
                "image_messages": "Инструкции (image + messages)",
                "image_caption": "Описания (image + caption)",
                "ocr_qa": "VQA / OCR (image + question + answer)",
            }

            def _apply_vlm_ds_preset() -> None:
                ds = preset_by_label.get(st.session_state.get("vlm_ds_preset_sel", ""))
                if not ds:
                    return
                st.session_state.vlm_dsdl_repo_id = ds["id"]
                st.session_state.vlm_dl_subset = ds.get("subset") or ""
                st.session_state.vlm_dl_split = ds.get("split", "train")
                st.session_state.vlm_dl_format = ds["format"]
                st.session_state.vlm_dl_max_rows = int(ds.get("recommended_rows", 2000))
                st.session_state.vlm_dl_out_name = ds["id"].split("/")[-1]
                st.session_state.pop("vlm_dl_subset_pick", None)
                st.session_state.pop("vlm_dl_split_pick", None)

            # Initialise defaults from the first preset once.
            if "vlm_dsdl_repo_id" not in st.session_state and filtered_datasets:
                first = filtered_datasets[0]
                st.session_state.vlm_dsdl_repo_id = first["id"]
                st.session_state.vlm_dl_subset = first.get("subset") or ""
                st.session_state.vlm_dl_split = first.get("split", "train")
                st.session_state.vlm_dl_format = first["format"]
                st.session_state.vlm_dl_max_rows = int(first.get("recommended_rows", 2000))
                st.session_state.vlm_dl_out_name = first["id"].split("/")[-1]

            st.selectbox(
                "📚 Готовый датасет",
                preset_labels,
                key="vlm_ds_preset_sel",
                on_change=_apply_vlm_ds_preset,
                help="Выберите курируемый датасет — поля заполнятся автоматически.",
            )
            if preset_by_label.get(st.session_state.get("vlm_ds_preset_sel", "")):
                st.caption(preset_by_label[st.session_state["vlm_ds_preset_sel"]]["description"])

            repo_id = st.text_input("Репозиторий (ID)", key="vlm_dsdl_repo_id")

            repo_info_store: Dict[str, Any] = st.session_state.setdefault("vlm_ds_repo_info", {})
            if st.button("🔍 Проверить репозиторий", key="vlm_ds_check"):
                with st.spinner(f"Анализируем {repo_id}..."):
                    info, msg = _inspect_hf_dataset(repo_id.strip())
                if info:
                    repo_info_store[repo_id.strip()] = info
                    st.session_state.pop("vlm_dl_subset_pick", None)
                    st.session_state.pop("vlm_dl_split_pick", None)
                    st.success(msg)
                else:
                    st.error(msg)

            info = repo_info_store.get(repo_id.strip(), {})
            configs = info.get("configs") or []
            splits = info.get("splits") or []

            if configs:
                cur = st.session_state.get("vlm_dl_subset", "")
                idx = configs.index(cur) if cur in configs else 0
                subset_val = st.selectbox("Subset (конфиг)", configs, index=idx, key="vlm_dl_subset_pick")
            else:
                subset_val = st.text_input(
                    "Subset (конфиг)",
                    key="vlm_dl_subset",
                    help="Оставьте пустым или 'default', если конфиг не нужен. Нажмите «Проверить репозиторий», чтобы увидеть список.",
                )
            if splits:
                cur = st.session_state.get("vlm_dl_split", "train")
                idx = splits.index(cur) if cur in splits else 0
                split_val = st.selectbox("Split", splits, index=idx, key="vlm_dl_split_pick")
            else:
                split_val = st.text_input("Split", key="vlm_dl_split")

            fmt_keys = list(format_labels.keys())
            # Значение приходит из session_state (в т.ч. из колбэков пресетов), поэтому
            # index не передаём — иначе Streamlit ругается на дублирование default + session_state.
            if st.session_state.get("vlm_dl_format") not in fmt_keys:
                st.session_state.vlm_dl_format = "image_messages" if "image_messages" in fmt_keys else fmt_keys[0]
            dl_format = st.selectbox(
                "Формат конвертации",
                fmt_keys,
                format_func=lambda k: format_labels[k],
                key="vlm_dl_format",
                help="Как интерпретировать поля датасета при сохранении в JSONL.",
            )

            with st.expander("🛠️ Лимиты скачивания", expanded=True):
                limit_type_label = st.radio(
                    "Ограничить по",
                    ["Кол-во строк", "Размеру (ГБ)"],
                    key="vlm_dl_limit_type",
                    horizontal=True,
                )
                if limit_type_label == "Кол-во строк":
                    st.number_input("Число строк (0 = всё)", min_value=0, step=500, key="vlm_dl_max_rows")
                else:
                    st.number_input(
                        "Размер, ГБ (вкл. картинки)",
                        min_value=0.1,
                        value=float(st.session_state.get("vlm_dl_max_gb", 1.0)),
                        step=0.5,
                        key="vlm_dl_max_gb",
                    )

            out_name = st.text_input("Сохранить как", key="vlm_dl_out_name")

            if limit_type_label == "Кол-во строк" and int(st.session_state.get("vlm_dl_max_rows", 0)) == 0:
                st.warning("Лимит = 0: датасет будет скачан целиком. Для больших наборов это займёт много места и времени.")

            if st.button("⬇️ Скачать и конвертировать", type="primary", key="vlm_dl_btn"):
                status_box = st.empty()

                def _progress(n: int, nbytes: int) -> None:
                    status_box.info(f"Обработано: {n} примеров, {nbytes / 1024**2:.1f} МБ...")

                is_rows = limit_type_label == "Кол-во строк"
                max_bytes = int(float(st.session_state.get("vlm_dl_max_gb", 1.0)) * 1024**3)
                with st.spinner(f"Скачиваем {repo_id} (streaming)..."):
                    path, message = _download_hf_vlm_dataset(
                        repo_id.strip(),
                        (split_val or "train").strip(),
                        out_name.strip(),
                        dl_format,
                        subset=(subset_val.strip() if isinstance(subset_val, str) else subset_val) or None,
                        limit_type="rows" if is_rows else "gb",
                        max_rows=int(st.session_state.get("vlm_dl_max_rows", 0)),
                        max_bytes=max_bytes,
                        progress_cb=_progress,
                    )
                if path:
                    status_box.success(message)
                    st.rerun()
                else:
                    status_box.error(message)

        with col_list:
            st.subheader("📁 Локальные датасеты")
            local_datasets = _get_vlm_datasets()
            if not local_datasets:
                st.info("Локальные VLM-датасеты пока не найдены. Скачайте набор слева или загрузите свой файл.")
            else:
                def _fmt_count(d: Dict[str, Any]) -> str:
                    prefix = "" if d.get("count_exact") else "~"
                    return f"{prefix}{d['count']} строк"

                dataset_name = st.selectbox(
                    "Датасет",
                    [item["name"] for item in local_datasets],
                    format_func=lambda n: next(
                        (f"{d['name']}  ·  {_fmt_count(d)}  ·  {d['size']}" for d in local_datasets if d["name"] == n),
                        n,
                    ),
                    key="vlm_local_ds_select",
                )
                dataset_meta = next(item for item in local_datasets if item["name"] == dataset_name)
                preview = _read_dataset_preview(Path(dataset_meta["path"]), limit=5)
                st.caption(
                    f"Schema: `{preview['schema']}`  ·  {_fmt_count(dataset_meta)}  ·  "
                    f"Поля: {', '.join(preview['fields']) if preview['fields'] else '—'}"
                )

                act1, act2 = st.columns(2)
                with act1:
                    if st.button("✅ Использовать для обучения", key="use_dataset_for_launch", use_container_width=True):
                        st.session_state.vlm_cfg_data_path = dataset_meta["path"]
                        st.success("Датасет выбран во вкладке «Запуск»")
                with act2:
                    if st.button("🗑️ Удалить датасет", key="delete_vlm_dataset", use_container_width=True):
                        ok, msg = _delete_vlm_dataset(dataset_meta["path"])
                        (st.success if ok else st.error)(msg)
                        if ok:
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

        st.subheader("🤗 Скачать VLM с HuggingFace")
        st.caption("Курируемые небольшие мультимодальные модели, подходящие для локального обучения (QLoRA/LoRA).")

        preset_names = [preset["name"] for preset in VLM_HF_PRESETS]
        selected_preset_name = st.selectbox(
            "Модель",
            preset_names + ["✏️ Ввести вручную"],
            format_func=lambda n: next(
                (f"{p['name']}  ·  {p['params']}  ·  {p['vram']}  ·  {p['task']}" for p in VLM_HF_PRESETS if p["name"] == n),
                n,
            ),
            key="vlm_model_preset_select",
        )
        preset = next((item for item in VLM_HF_PRESETS if item["name"] == selected_preset_name), None)

        if preset:
            st.info(f"**{preset['name']}** — {preset['description']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Параметры", preset["params"])
            m2.metric("VRAM (обучение)", preset["vram"])
            m3.metric("Задача", preset["task"])
            if preset.get("trust_remote_code"):
                st.caption("⚠️ Эта модель требует `trust_remote_code=True` при загрузке.")

        c1, c2 = st.columns(2)
        with c1:
            repo_id = st.text_input("Repo ID", value=preset["repo_id"] if preset else "", key="vlm_dl_repo_id")
        with c2:
            save_name = st.text_input("Имя папки в models/", value=preset["save_name"] if preset else "", key="vlm_dl_save_name")

        if st.button("⬇️ Скачать модель", type="primary", key="vlm_dl_model_btn") and repo_id and save_name:
            if _download_hf_vlm_model(repo_id.strip(), save_name.strip()):
                st.success("Модель скачана")
                st.rerun()

        st.markdown("---")
        st.subheader("📁 Доступные VLM")
        models = get_available_vlm_models()
        local_models = [m for m in models if m.get("type") == "local"]
        hf_models = [m for m in models if m.get("type") == "hf"]

        st.markdown("**Локальные модели** (скачанные / обученные)")
        if not local_models:
            st.info("Локальных VLM пока нет — скачайте модель выше или обучите свою.")
        for idx, model in enumerate(local_models):
            with st.expander(model["name"]):
                st.caption(f"Путь: `{model['path']}`  ·  Архитектура: `{model.get('family', '—')}`")
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("🚀 В Launch", key=f"use_launch_model_{idx}", use_container_width=True):
                        st.session_state.vlm_selected_base_path = model["path"]
                        st.success("Выбрано для обучения")
                with b2:
                    if st.button("💬 В Chat", key=f"use_chat_model_{idx}", use_container_width=True):
                        st.session_state.vlm_selected_chat_model = model["path"]
                        st.success("Выбрано для чата")
                with b3:
                    if str(model["path"]).startswith(str(MODELS_DIR)):
                        if st.button("🗑️ Удалить", key=f"del_model_{idx}", use_container_width=True):
                            ok, msg = _delete_vlm_model(model["path"])
                            (st.success if ok else st.error)(msg)
                            if ok:
                                st.rerun()

        with st.expander(f"🤗 Готовые модели HuggingFace ({len(hf_models)})"):
            st.caption("Их можно использовать напрямую по repo_id (скачаются при первом запуске) или скачать заранее выше.")
            for model in hf_models:
                st.markdown(f"- {model['name']} — `{model['path']}`")

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
