"""
Motels at Home Training Studio — Визуальное приложение для тренировки моделей
======================================================================

Запуск:
    streamlit run homellm/app/main.py
    
или:
    ./scripts/run_studio.sh
"""

import streamlit as st
import logging
import subprocess
import json
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import signal
from pathlib import Path
from datetime import datetime
from typing import Tuple
from contextlib import suppress
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import torch
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names, load_dataset_builder  # Добавляем импорт
try:
    from .docs import render_docs
except ImportError:
    from docs import render_docs

# Internationalization (i18n)
try:
    from homellm.i18n import t, load_translations
except ImportError:
    from ..i18n import t, load_translations

try:
    from .ui_preferences import (
        DEFAULT_THEME,
        init_user_preferences,
        persist_user_preferences_if_changed,
        apply_theme_css,
        render_first_run_setup,
    )
except ImportError:
    from ui_preferences import (
        DEFAULT_THEME,
        init_user_preferences,
        persist_user_preferences_if_changed,
        apply_theme_css,
        render_first_run_setup,
    )

# Пути
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_DIR = PROJECT_ROOT / "datasets"  # datasets с "s"!
MODELS_DIR = PROJECT_ROOT / "models"  # Скачанные HF модели
OUTPUT_DIR = PROJECT_ROOT / "out"
RUNS_DIR = PROJECT_ROOT / ".runs"
RUNS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Page Config
# ============================================================================

st.set_page_config(
    page_title="HomeLLM Training Studio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load translations for i18n
load_translations()

# ============================================================================
# Persistence — сохранение состояния между перезагрузками
# ============================================================================

ACTIVE_RUN_FILE = RUNS_DIR / "active_run.json"
USER_PREFS_FILE = RUNS_DIR / "ui_preferences.json"
init_user_preferences(USER_PREFS_FILE)
apply_theme_css(st.session_state.get("ui_theme", DEFAULT_THEME))


def save_active_run(run_id: str, config: dict = None):
    """Сохранить активный run в файл."""
    data = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "config": config or {}
    }
    with open(ACTIVE_RUN_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_active_run() -> dict | None:
    """Загрузить активный run из файла."""
    if not ACTIVE_RUN_FILE.exists():
        return None
    try:
        with open(ACTIVE_RUN_FILE) as f:
            return json.load(f)
    except:
        return None


def clear_active_run():
    """Очистить активный run."""
    ACTIVE_RUN_FILE.unlink(missing_ok=True)


def restore_session_state():
    """Восстановить состояние сессии после перезагрузки."""
    # Проверяем есть ли сохранённый активный run
    active = load_active_run()
    if active and active.get("run_id"):
        run_id = active["run_id"]
        
        # Проверяем существует ли run директория
        run_dir = RUNS_DIR / run_id
        if run_dir.exists():
            # Проверяем жив ли процесс (более надёжная проверка через metrics.json)
            pid_path = run_dir / "pid"
            metrics_path = run_dir / "metrics.json"
            process_alive = False
            metrics = None
            
            # Проверяем статус из metrics.json (более надёжно)
            if metrics_path.exists():
                try:
                    import time
                    metrics_mtime = metrics_path.stat().st_mtime
                    metrics_age_minutes = (time.time() - metrics_mtime) / 60
                    
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    
                    status = metrics.get("status", "")
                    # Если статус завершённый - точно не жив
                    if status in ("completed", "error", "stopped"):
                        process_alive = False
                    else:
                        # Проверяем процесс через PID
                        if pid_path.exists():
                            try:
                                with open(pid_path) as f:
                                    pid = int(f.read().strip())
                                os.kill(pid, 0)  # Проверка существования процесса
                                process_alive = True
                            except ProcessLookupError:
                                process_alive = False
                            except (ValueError, PermissionError):
                                # PermissionError может означать, что процесс существует, но у нас нет прав
                                # Но если метрики свежие (< 5 минут) - считаем живым
                                process_alive = metrics_age_minutes < 5
                    
                    # Если метрики не обновлялись давно — НЕ считаем процесс мёртвым автоматически.
                    # Сначала пытаемся проверить PID; PermissionError трактуем как "жив".
                    if status not in ("completed", "error", "stopped") and metrics_age_minutes > 5:
                        pid_alive = False
                        if pid_path.exists():
                            try:
                                with open(pid_path) as f:
                                    pid = int(f.read().strip())
                                os.kill(pid, 0)
                                pid_alive = True
                            except PermissionError:
                                pid_alive = True
                            except (ProcessLookupError, ValueError, FileNotFoundError):
                                pid_alive = False
                        if pid_alive:
                            process_alive = True
                            logger.warning(
                                f"Metrics not updated for {metrics_age_minutes:.1f} minutes, but PID looks alive. "
                                f"Treating process as running (metrics may be stalled)."
                            )
                        else:
                            process_alive = False
                            logger.warning(f"Metrics not updated for {metrics_age_minutes:.1f} minutes and PID not alive, assuming process dead")
                            clear_active_run()
                except Exception as e:
                    logger.warning(f"Failed to check metrics: {e}")
                    # Fallback на проверку PID
                    if pid_path.exists():
                        try:
                            with open(pid_path) as f:
                                pid = int(f.read().strip())
                            os.kill(pid, 0)
                            process_alive = True
                        except (ProcessLookupError, ValueError, PermissionError):
                            process_alive = False
            
            # Восстанавливаем состояние
            st.session_state.current_run_id = run_id
            st.session_state.training_active = process_alive
            
            # Если процесс завершён, очищаем active_run
            if not process_alive:
                if metrics and metrics.get("status") in ("completed", "error", "stopped"):
                    clear_active_run()


# ============================================================================
# Session State
# ============================================================================

if "training_process" not in st.session_state:
    st.session_state.training_process = None
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None
if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "selected_chat_model" not in st.session_state:
    st.session_state.selected_chat_model = None

# Восстанавливаем состояние при первой загрузке
if "session_restored" not in st.session_state:
    restore_session_state()
    st.session_state.session_restored = True


# ============================================================================
# Helper Functions
# ============================================================================

def get_available_datasets():
    """Получить список доступных датасетов."""
    datasets = []
    if DATASET_DIR.exists():
        for f in DATASET_DIR.glob("*.jsonl"):
            size_mb = f.stat().st_size / (1024 * 1024)
            datasets.append((f.name, f"{size_mb:.1f} MB"))
        # ВАЖНО: не показываем .json, т.к. это обычно массив, а не JSONL (построчный формат)
        # для f in DATASET_DIR.glob("*.json"):
        #     size_mb = f.stat().st_size / (1024 * 1024)
        #     datasets.append((f.name, f"{size_mb:.1f} MB"))
        for f in DATASET_DIR.glob("*.txt"):
            size_mb = f.stat().st_size / (1024 * 1024)
            datasets.append((f.name, f"{size_mb:.1f} MB"))
        for f in DATASET_DIR.glob("*.txt.gz"):
            size_mb = f.stat().st_size / (1024 * 1024)
            datasets.append((f.name, f"{size_mb:.1f} MB"))
        for f in DATASET_DIR.glob("*.jsonl.gz"):
            size_mb = f.stat().st_size / (1024 * 1024)
            datasets.append((f.name, f"{size_mb:.1f} MB"))
    return datasets


def get_gpu_info():
    """Получить информацию о доступных GPU."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            gpus.append({
                "id": i,
                "name": props.name,
                "memory_gb": round(memory_gb, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    return gpus


def get_available_configs():
    """Получить список доступных accelerate конфигов."""
    configs = []
    if CONFIGS_DIR.exists():
        for f in CONFIGS_DIR.glob("*.yaml"):
            name = f.stem.replace("accelerate_", "").replace("_", " ").title()
            configs.append({
                "file": f.name,
                "name": name,
                "path": str(f),
            })
    return configs


# Parallelism type descriptions (i18n keys)
PARALLEL_TYPES = {
    "default": {
        "name": "Single GPU / CPU",
        "type": "None",
        "description_key": "parallel.description.default",
        "icon": "🖥️",
    },
    "multi_gpu": {
        "name": "Multi-GPU (DDP)",
        "type": "Data Parallel",
        "description_key": "parallel.description.multi_gpu",
        "icon": "🔄",
    },
    "fsdp": {
        "name": "FSDP",
        "type": "Model Parallel",
        "description_key": "parallel.description.fsdp",
        "icon": "⚡",
    },
    "fsdp_offload": {
        "name": "FSDP + CPU Offload",
        "type": "Model Parallel + CPU Offload",
        "description_key": "parallel.description.fsdp_offload",
        "icon": "💾",
    },
    "fsdp2": {
        "name": "FSDP2 + CPU Offload",
        "type": "Model Parallel + CPU Offload",
        "description_key": "parallel.description.fsdp2",
        "icon": "🔥",
    },
    "deepspeed_zero2": {
        "name": "DeepSpeed ZeRO-2",
        "type": "Data Parallel + Optimizer Parallel",
        "description_key": "parallel.description.deepspeed_zero2",
        "icon": "🚀",
    },
    "deepspeed_zero3": {
        "name": "DeepSpeed ZeRO-3",
        "type": "Full Model Parallel",
        "description_key": "parallel.description.deepspeed_zero3",
        "icon": "💪",
    },
    "deepspeed_zero3_offload": {
        "name": "ZeRO-3 + CPU Offload",
        "type": "Model Parallel + CPU Offload",
        "description_key": "parallel.description.deepspeed_zero3_offload",
        "icon": "🧊",
    },
}


def estimate_parameters(
    hidden_size: int,
    num_layers: int,
    vocab_size: int = 50257,
    intermediate_size: int | None = None,
    arch_type: str = "home",  # "home" | "gpt2" | "home_moe"
    num_experts: int = 8,  # Для MoE
    max_position_embeddings: int = 4096,  # Для GPT-2 (learned pos embeddings)
) -> int:
    """
    Оценка количества параметров для разных архитектур.
    
    Архитектуры:
    - "home": HomeModel (LLaMA-style) с RMSNorm, RoPE, SwiGLU
    - "gpt2": GPT-2 с LayerNorm, Learned Pos, GELU
    - "home_moe": HomeModel с MoE (num_experts экспертов)
    
    Использует профили из homellm.models.memory_estimator для точного расчёта.
    """
    from homellm.models.memory_estimator import ARCHITECTURE_PROFILES
    
    h = int(hidden_size)
    l = int(num_layers)
    v = int(vocab_size)
    i = int(intermediate_size) if intermediate_size is not None else int(h * 4)
    s = int(max_position_embeddings)
    
    # Получаем профиль архитектуры
    profile_key = arch_type.lower()
    if profile_key not in ARCHITECTURE_PROFILES:
        profile_key = "home"  # fallback
    
    profile = ARCHITECTURE_PROFILES[profile_key]
    
    # Используем метод calculate_parameters из профиля
    return profile.calculate_parameters(
        vocab_size=v,
        hidden_size=h,
        num_layers=l,
        intermediate_size=i,
        max_position_embeddings=s,
        num_experts=num_experts if profile.is_moe else None,
    )


def format_params(n: int) -> str:
    """Форматирование количества параметров."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def get_preset_configs():
    """Базовые конфигурации пресетов (hidden, layers, heads)."""
    return {
        "Tiny": (512, 8, 8),
        "Small": (768, 12, 12),
        "Medium": (1024, 16, 16),
        "Large": (1280, 20, 20),
    }


def generate_preset_labels(
    arch_type: str,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    vocab_size: int = 50257,
) -> dict:
    """
    Генерирует названия пресетов с реальными размерами для выбранной архитектуры.
    
    Returns:
        dict: {"Label": (hidden, layers, heads), ...}
    """
    base_configs = get_preset_configs()
    result = {}
    
    for name, (h, l, n) in base_configs.items():
        intermediate = h * 4
        
        params = estimate_parameters(
            hidden_size=h,
            num_layers=l,
            vocab_size=vocab_size,
            intermediate_size=intermediate,
            arch_type=arch_type,
            num_experts=num_experts,
        )
        
        if arch_type == "home_moe":
            # Для MoE вычисляем активные параметры (сколько работает на токен)
            # Active = Total - (неактивные эксперты)
            # Неактивные = (num_experts - num_experts_per_tok) / num_experts * MLP_params
            
            # Считаем размер одного эксперта
            expert_params = 3 * h * intermediate  # SwiGLU: 3 * H * I
            total_expert_params = l * num_experts * expert_params
            active_expert_params = l * num_experts_per_tok * expert_params
            
            # Active params = Total - неактивные эксперты
            non_expert_params = params - total_expert_params - l * h * num_experts  # убираем gate тоже? нет, gate всегда активен
            active_params = non_expert_params + active_expert_params + l * h * num_experts  # gate активен
            
            label = f"{name} ({format_params(params)} total, {format_params(active_params)} active)"
        else:
            label = f"{name} ({format_params(params)})"
        
        result[label] = (h, l, n)
    
    return result


def format_time(seconds: float) -> str:
    """Форматирование времени."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def delete_checkpoint(run_id: str, checkpoint_path: str) -> tuple[bool, str]:
    """
    Удалить отдельный чекпоинт.
    
    Args:
        run_id: ID запуска
        checkpoint_path: Путь к чекпоинту
        
    Returns:
        (success, message) - успех операции и сообщение
    """
    import shutil
    
    try:
        ckpt_path = Path(checkpoint_path)
        
        # Если путь относительный, делаем абсолютным
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        
        if not ckpt_path.exists():
            return False, f"Чекпоинт не найден: {checkpoint_path}"
        
        # Удаляем папку чекпоинта
        shutil.rmtree(ckpt_path)
        
        # Обновляем metrics.json - удаляем запись о чекпоинте
        run_dir = RUNS_DIR / run_id
        metrics_path = run_dir / "metrics.json"
        
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                
                # Фильтруем чекпоинты - удаляем тот что удалили
                if "checkpoints" in metrics:
                    original_count = len(metrics["checkpoints"])
                    metrics["checkpoints"] = [
                        ckpt for ckpt in metrics["checkpoints"]
                        if ckpt.get("path") != checkpoint_path and not str(ckpt.get("path", "")).endswith(ckpt_path.name)
                    ]
                    
                    # Сохраняем обновленные метрики
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Удален чекпоинт {checkpoint_path}, обновлены метрики ({original_count} -> {len(metrics['checkpoints'])})")
            except Exception as e:
                logger.warning(f"Не удалось обновить metrics.json: {e}")
        
        return True, f"Чекпоинт удален: {ckpt_path.name}"
        
    except Exception as e:
        logger.error(f"Ошибка удаления чекпоинта {checkpoint_path}: {e}")
        return False, f"Ошибка удаления: {e}"


def delete_experiment(run_id: str) -> tuple[bool, str]:
    """
    Удалить весь эксперимент (run) вместе со всеми чекпоинтами и моделями.
    
    Args:
        run_id: ID запуска для удаления
        
    Returns:
        (success, message) - успех операции и сообщение
    """
    import shutil
    
    deleted_items = []
    errors = []
    
    try:
        run_dir = RUNS_DIR / run_id
        
        if not run_dir.exists():
            return False, f"Запуск не найден: {run_id}"
        
        # 1. Получаем output_dir из конфига (там хранятся чекпоинты и модели)
        output_dir = None
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    output_dir = config.get("output_dir")
            except Exception as e:
                logger.warning(f"Не удалось прочитать config.json: {e}")
        
        # 2. Удаляем output_dir (чекпоинты, final_model, etc.)
        if output_dir:
            output_path = Path(output_dir)
            if not output_path.is_absolute():
                output_path = PROJECT_ROOT / output_dir
            
            if output_path.exists():
                try:
                    shutil.rmtree(output_path)
                    deleted_items.append(f"📁 {output_path.name}")
                    logger.info(f"Удалена папка output: {output_path}")
                except Exception as e:
                    errors.append(f"output_dir: {e}")
                    logger.error(f"Не удалось удалить output_dir {output_path}: {e}")
        
        # 3. Удаляем директорию запуска в .runs/
        try:
            shutil.rmtree(run_dir)
            deleted_items.append(f"📊 .runs/{run_id}")
            logger.info(f"Удалена папка запуска: {run_dir}")
        except Exception as e:
            errors.append(f"run_dir: {e}")
            logger.error(f"Не удалось удалить run_dir {run_dir}: {e}")
        
        # Формируем сообщение
        if deleted_items and not errors:
            return True, f"Эксперимент удален: {', '.join(deleted_items)}"
        elif deleted_items and errors:
            return True, f"Частично удалено: {', '.join(deleted_items)}. Ошибки: {', '.join(errors)}"
        else:
            return False, f"Не удалось удалить: {', '.join(errors)}"
            
    except Exception as e:
        logger.error(f"Ошибка удаления эксперимента {run_id}: {e}")
        return False, f"Ошибка удаления: {e}"


def load_metrics(run_id: str) -> dict:
    """Загрузить метрики из файла."""
    run_dir = RUNS_DIR / run_id
    run_config = {}
    
    # Для GRPO читаем из metrics.jsonl (может быть в output_dir или run_dir)
    metrics_jsonl_paths = []
    
    # Пробуем найти metrics.jsonl в output_dir
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                run_config = json.load(f) or {}
                output_dir = run_config.get("output_dir", "")
                if output_dir:
                    metrics_jsonl_paths.append(Path(output_dir) / "metrics.jsonl")
        except:
            pass
    
    # Fallback: ищем в run_dir
    metrics_jsonl_paths.append(run_dir / "metrics.jsonl")
    
    for metrics_jsonl_path in metrics_jsonl_paths:
        if metrics_jsonl_path.exists():
            try:
                import pandas as pd
                # Читаем все строки из JSONL
                lines = []
                with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                lines.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                
                if lines:
                    df = pd.DataFrame(lines)
                    # Берем последнюю запись как текущие метрики
                    latest = df.iloc[-1].to_dict()
                    
                    # Добавляем историю для графиков
                    latest["reward_history"] = df["reward"].tolist() if "reward" in df.columns else []
                    latest["loss_history"] = df["loss"].tolist() if "loss" in df.columns else []
                    latest["kl_history"] = df["kl"].tolist() if "kl" in df.columns else []
                    latest["steps_history"] = df["step"].tolist() if "step" in df.columns else list(range(len(df)))
                    latest["lr_history"] = df["learning_rate"].tolist() if "learning_rate" in df.columns else (df["lr"].tolist() if "lr" in df.columns else [])
                    
                    # Добавляем текущие значения для метрик
                    # Для GRPO прогресс в UI считаем по rollout_step (покрытие датасета), а optim_step показываем отдельно.
                    latest["current_step"] = latest.get("current_step", latest.get("rollout_step", latest.get("step", len(lines) - 1)))
                    latest["current_loss"] = latest.get("loss", 0)
                    latest["current_lr"] = latest.get("learning_rate", latest.get("lr", 0))
                    latest["reward"] = latest.get("reward", latest.get("batch_reward_mean", 0))
                    
                    # Добавляем семплы если есть
                    if "samples" in df.columns:
                        latest["samples_history"] = df["samples"].tolist()
                    
                    # Статус может быть NaN из pandas, нужно привести к строке
                    raw_status = latest.get("status")
                    import math
                    if raw_status is None or (isinstance(raw_status, float) and math.isnan(raw_status)):
                        latest["status"] = "training"
                    else:
                        latest["status"] = str(raw_status)
                    latest["stage"] = "grpo"

                    # Фактические счётчики (если есть в jsonl)
                    try:
                        if "prompts_generated_total" in df.columns:
                            latest["prompts_generated_total"] = int(df["prompts_generated_total"].fillna(0).iloc[-1])
                        elif "prompts_generated" in df.columns:
                            latest["prompts_generated_total"] = int(df["prompts_generated"].fillna(0).sum())
                        if "prompts_used_total" in df.columns:
                            latest["prompts_used_total"] = int(df["prompts_used_total"].fillna(0).iloc[-1])
                        elif "prompts_used" in df.columns:
                            latest["prompts_used_total"] = int(df["prompts_used"].fillna(0).sum())
                        if "completions_generated_total" in df.columns:
                            latest["completions_generated_total"] = int(df["completions_generated_total"].fillna(0).iloc[-1])
                        elif "completions_generated" in df.columns:
                            latest["completions_generated_total"] = int(df["completions_generated"].fillna(0).sum())
                        if "experiences_tuned_total" in df.columns:
                            latest["experiences_tuned_total"] = int(df["experiences_tuned_total"].fillna(0).iloc[-1])
                        elif "experiences_tuned" in df.columns:
                            latest["experiences_tuned_total"] = int(df["experiences_tuned"].fillna(0).sum())
                    except Exception:
                        pass

                    # --- Нормализация прогресса/ETA для GRPO ---
                    # total_steps: берём из метрик (если логируется) или из run_config (config.json)
                    total_steps = latest.get("total_steps", None)
                    if total_steps in (None, "", 0):
                        # legacy: optim-step лимит
                        total_steps = run_config.get("grpo_max_optim_steps", run_config.get("grpo_max_steps", run_config.get("max_steps", None)))
                    try:
                        total_steps_int = int(total_steps) if total_steps is not None else None
                    except Exception:
                        total_steps_int = None
                    # Если лимита нет — оставляем None (UI покажет "без лимита"), а не "1".
                    latest["total_steps"] = total_steps_int if (total_steps_int is not None and total_steps_int > 0) else None

                    # elapsed/eta: по timestamp, если он есть
                    elapsed_seconds = 0.0
                    eta_seconds = 0.0
                    try:
                        if "timestamp" in df.columns:
                            t0 = pd.to_datetime(df["timestamp"].iloc[0], errors="coerce")
                            t1 = pd.to_datetime(df["timestamp"].iloc[-1], errors="coerce")
                            if pd.notna(t0) and pd.notna(t1):
                                # Если пока всего 1 запись, показываем elapsed как now - t0, чтобы не было "0s" в UI.
                                if len(df) == 1:
                                    elapsed_seconds = max(0.0, (pd.Timestamp.now(tz=t0.tz) - t0).total_seconds())
                                else:
                                    elapsed_seconds = max(0.0, (t1 - t0).total_seconds())

                        # ETA считаем по current_step (rollout_step), а не по optim_step,
                        # иначе прогресс/ETA будут "убегать" из-за multiple optimizer updates per rollout.
                        if "timestamp" in df.columns and "current_step" in df.columns and len(df) >= 2:
                            s0 = float(df["current_step"].iloc[0])
                            s1 = float(df["current_step"].iloc[-1])
                            # средняя скорость по наблюдаемым step (учитываем что лог может быть не на каждом шаге)
                            ds = max(0.0, s1 - s0)
                            if ds > 0 and elapsed_seconds > 0 and latest["total_steps"] is not None:
                                sec_per_step = elapsed_seconds / ds
                                remaining = max(0.0, float(latest["total_steps"]) - float(latest.get("current_step", s1)))
                                eta_seconds = sec_per_step * remaining
                    except Exception:
                        pass
                    latest["elapsed_seconds"] = float(elapsed_seconds)
                    latest["eta_seconds"] = float(eta_seconds)

                    # rollout_step тоже прокинем (если есть)
                    if "rollout_step" in latest:
                        latest["current_rollout_step"] = latest.get("rollout_step", 0)

                    return latest
            except Exception as e:
                # Не показываем ошибку если это не последний путь
                if metrics_jsonl_path == metrics_jsonl_paths[-1]:
                    pass  # Логируем только для последнего пути
    
    # Обычный путь для других стадий
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except:
            pass
    return None


def _close_run_log_files(run_id: str):
    """Закрыть файловые дескрипторы stdout/stderr, если они есть в session_state."""
    for k in (f"stdout_file_{run_id}", f"stderr_file_{run_id}"):
        f = st.session_state.get(k)
        if f:
            with suppress(Exception):
                f.close()
            with suppress(Exception):
                del st.session_state[k]


def start_training(config: dict) -> tuple[str, subprocess.Popen]:
    """Запустить тренировку в фоне."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Явно логируем ключевые UI-параметры, чтобы не было “тихих” перетираний пресетами
    try:
        logger.info(
            "Start training with: stage=%s mixed_precision=%s fp16_pure=%s grad_checkpoint=%s use_flash_attention=%s",
            config.get("stage"),
            config.get("mixed_precision"),
            config.get("fp16_pure"),
            config.get("grad_checkpoint"),
            config.get("use_flash_attention"),
        )
    except Exception:
        pass
    
    # ЛОГИКА ПУТЕЙ
    # config["output_dir"] - это корень эксперимента (например out/my_model)
    # Мы хотим сохранять чекпоинты в out/my_model/run_2023.../checkpoint_...
    experiment_root = Path(PROJECT_ROOT) / config.get("output_dir", "out/default")
    run_output_dir = experiment_root / run_id
    
    # Обновляем output_dir в конфиге, чтобы worker сохранял туда
    config["output_dir"] = str(run_output_dir)
    
    # Папка для метаданных запуска (логи, метрики)
    # Можно хранить их там же, где и чекпоинты, для удобства
    run_dir = RUNS_DIR / run_id # Оставляем .runs для внутренних логов стримлита
    run_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True) # Создаем папку для весов
    
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    
    # Сохраняем конфиг
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Создаём начальный metrics файл
    with open(metrics_path, "w") as f:
        json.dump({"status": "starting", "current_step": 0}, f)
    
    # Определяем команду в зависимости от режима distributed
    distributed_mode = config.get("distributed_mode", "default")
    config_file = config.get("config_file")
    num_gpus = config.get("num_gpus", 1)
    
    if distributed_mode != "default" and config_file:
        # Используем accelerate launch с конфигом
        # ВАЖНО: gradient_accumulation_steps передается через config.json, не через CLI флаги
        cmd = [
            "accelerate", "launch",
            "--config_file", config_file,
            "--num_processes", str(num_gpus),
            "-m", "homellm.app.trainer_worker",
            "--config", str(config_path),
            "--metrics", str(metrics_path)
        ]
    else:
        # Обычный запуск
        cmd = [
            "python", "-m", "homellm.app.trainer_worker",
            "--config", str(config_path),
            "--metrics", str(metrics_path)
        ]
    
    # Сохраняем команду для отладки
    cmd_path = run_dir / "command.txt"
    with open(cmd_path, "w") as f:
        f.write(" ".join(cmd))
    
    stdout_file = open(stdout_path, "w")
    stderr_file = open(stderr_path, "w")
    
    # ВАЖНО: применяем выбор GPU из UI
    env = os.environ.copy()
    gpu_ids = config.get("gpu_ids") or []
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,  # Отделяем от родительского процесса
        env=env,
    )
    
    # Сохраняем файловые дескрипторы для закрытия при остановке
    st.session_state[f"stdout_file_{run_id}"] = stdout_file
    st.session_state[f"stderr_file_{run_id}"] = stderr_file
    
    # Сохраняем PID для мониторинга
    pid_path = run_dir / "pid"
    with open(pid_path, "w") as f:
        f.write(str(process.pid))
    
    return run_id, process


def stop_training():
    """Остановить тренировку."""
    stopped = False
    
    # Пробуем остановить по PID из файла (более надёжно)
    if st.session_state.current_run_id:
        pid_path = RUNS_DIR / st.session_state.current_run_id / "pid"
        if pid_path.exists():
            try:
                with open(pid_path) as f:
                    pid = int(f.read().strip())
                # Убиваем process group (важно для accelerate/DDP)
                try:
                    # Пытаемся убить process group
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    stopped = True
                except (ProcessLookupError, OSError):
                    # Если не получилось (процесс не в группе или уже завершился), пробуем по PID
                    try:
                        os.kill(pid, signal.SIGTERM)
                        stopped = True
                    except ProcessLookupError:
                        pass
                
                # Ждём немного и проверяем
                time.sleep(0.5)
                try:
                    # Проверяем жив ли процесс
                    os.kill(pid, 0)
                    # Если жив, убиваем принудительно (process group)
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Процесс уже завершился
            except Exception as e:
                pass
        
        # Обновляем метрики
        metrics_path = RUNS_DIR / st.session_state.current_run_id / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                metrics["status"] = "stopped"
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
            except:
                pass
    
    # Также пробуем через subprocess
    if st.session_state.training_process:
        try:
            st.session_state.training_process.terminate()
            st.session_state.training_process.wait(timeout=2)
        except:
            try:
                st.session_state.training_process.kill()
            except:
                pass
        st.session_state.training_process = None
    
    st.session_state.training_active = False
    
    # Очищаем active_run
    clear_active_run()
    
    # Закрываем файловые дескрипторы логов
    if st.session_state.current_run_id:
        _close_run_log_files(st.session_state.current_run_id)
    
    return stopped


def start_grpo_training(config: dict) -> tuple[str, subprocess.Popen]:
    """Запустить GRPO обучение в фоне."""
    training_backend = config.get("training_backend", "models-at-home")
    run_id = datetime.now().strftime("grpo_%Y%m%d_%H%M%S")
    
    # Логируем backend для отладки
    logger.info(f"🧠 GRPO Training backend: {training_backend}")
    
    # Папки для сохранения
    experiment_root = Path(PROJECT_ROOT) / config.get("output_dir", "out/grpo")
    run_output_dir = experiment_root / run_id
    
    config["output_dir"] = str(run_output_dir)
    
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Для "железного" мониторинга прокидываем путь до run_dir в worker.
    # Worker будет дублировать metrics.jsonl/samples.jsonl в эту директорию.
    config["ui_run_dir"] = str(run_dir)
    
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    
    # Сохраняем конфиг
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    # Начальные метрики
    with open(metrics_path, "w") as f:
        json.dump({"status": "starting", "current_step": 0, "stage": "grpo"}, f)
    
    # Формируем команду для запуска GRPO
    # Используем --config_json для передачи всей конфигурации включая reward_rules
    import sys
    
    # Передаём конфиг как JSON строку
    config_json = json.dumps(config, default=str)
    
    # Определяем режим distributed (как в start_training)
    distributed_mode = config.get("distributed_mode", "default")
    config_file = config.get("config_file")
    num_gpus = config.get("num_gpus", 1)
    
    # Формируем команду в зависимости от режима distributed
    if distributed_mode != "default" and config_file:
        # Используем accelerate launch с конфигом (как в pretrain/SFT)
        cmd = [
            "accelerate", "launch",
            "--config_file", config_file,
            "--num_processes", str(num_gpus),
            "-m", "homellm.training.rl.train_rl",
            "--config_json", config_json,
        ]
    else:
        # Обычный запуск (single GPU или CPU)
        cmd = [
            sys.executable, "-m", "homellm.training.rl.train_rl",
            "--config_json", config_json,
        ]
    
    # Сохраняем команду для отладки (без длинного JSON)
    cmd_path = run_dir / "command.txt"
    with open(cmd_path, "w") as f:
        if distributed_mode != "default" and config_file:
            f.write(f"accelerate launch --config_file {config_file} --num_processes {num_gpus} -m homellm.training.rl.train_rl --config_json <config from {config_path}>")
        else:
            f.write(f"{sys.executable} -m homellm.training.rl.train_rl --config_json <config from {config_path}>")
    
    stdout_file = open(stdout_path, "w")
    stderr_file = open(stderr_path, "w")
    
    # ВАЖНО: применяем выбор GPU из UI (как в start_training)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    gpu_ids = list(config.get("gpu_ids") or [])
    
    # Если используется vLLM на другой GPU — добавляем её в CUDA_VISIBLE_DEVICES
    vllm_device = config.get("grpo_vllm_device", "")
    use_rollout_engine = config.get("grpo_use_rollout_engine", False)
    rollout_backend = config.get("grpo_rollout_backend", "hf")
    
    if use_rollout_engine and rollout_backend == "vllm" and vllm_device.startswith("cuda:"):
        vllm_gpu_id = int(vllm_device.split(":")[1])
        if vllm_gpu_id not in gpu_ids:
            gpu_ids.append(vllm_gpu_id)
            logger.info(f"🧩 Добавлена GPU {vllm_gpu_id} для vLLM rollout engine")
    
    if gpu_ids:
        # НЕ сортируем! Порядок важен для ремаппинга индексов внутри процесса
        # Первые GPU — для training, последняя (если добавлена) — для vLLM
        gpu_ids = list(dict.fromkeys(gpu_ids))  # Убираем дубликаты, сохраняя порядок
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logger.info(f"🎯 CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,
        env=env,
    )
    
    st.session_state[f"stdout_file_{run_id}"] = stdout_file
    st.session_state[f"stderr_file_{run_id}"] = stderr_file
    
    # Сохраняем PID
    pid_path = run_dir / "pid"
    with open(pid_path, "w") as f:
        f.write(str(process.pid))
    
    return run_id, process


def is_process_running(run_id: str) -> bool:
    """Проверить, запущен ли процесс."""
    pid_path = RUNS_DIR / run_id / "pid"
    if not pid_path.exists():
        return False
    
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        # Проверяем существует ли процесс
        os.kill(pid, 0)
        return True
    except PermissionError:
        # Процесс существует, но у нас нет прав (например, запущен от другого юзера)
        # Считаем, что он жив
        return True
    except (ProcessLookupError, ValueError, FileNotFoundError):
        return False


# ============================================================================
# UI Components
# ============================================================================

def render_header():
    st.markdown(f"# 🏠 {t('app.title')}")
    st.caption(t("app.subtitle"))


def get_nested_value(data: dict, path: str):
    """Получает значение по пути.
    
    Поддерживает:
    - 'key1.key2' - вложенные словари
    - 'messages [список из N эл.]' - возвращает весь список
    - 'messages[].content' - возвращает список значений поля из каждого элемента
    - 'messages[0]' - первый элемент списка
    """
    if not path: return None
    
    # Убираем суффиксы типа " [список]" или " [список из 3 эл.]"
    import re
    path = re.sub(r' \[список.*?\]$', '', path)
    
    # Обработка путей типа 'messages[].content' (все элементы списка)
    if "[]." in path:
        parts = path.split("[].", 1)
        list_path = parts[0]
        remaining_path = parts[1]
        
        list_val = get_nested_value(data, list_path)
        if isinstance(list_val, list):
            results = []
            for item in list_val:
                if isinstance(item, dict):
                    val = get_nested_value(item, remaining_path)
                    results.append(val)
            return results
        return None
    
    # Обработка путей типа 'messages[0]' (конкретный индекс)
    if "[" in path and "]" in path:
        # Парсим индекс
        match = re.search(r'\[(\d+)\]', path)
        if match:
            idx = int(match.group(1))
            base_path = path[:match.start()]
            after_path = path[match.end():]
            if after_path.startswith('.'):
                after_path = after_path[1:]
            
            base_val = get_nested_value(data, base_path)
            if isinstance(base_val, list) and 0 <= idx < len(base_val):
                if after_path:
                    return get_nested_value(base_val[idx], after_path)
                return base_val[idx]
            return None
    
    # Обычный путь через точки
    keys = path.split('.')
    curr = data
    try:
        for k in keys:
            if isinstance(curr, dict):
                curr = curr.get(k)
            elif isinstance(curr, list) and k.isdigit():
                curr = curr[int(k)]
            else:
                return None
            if curr is None: return None
        return curr
    except:
        return None


def get_dataset_columns(file_path: str):
    """Анализирует файл и возвращает список колонок (включая вложенные) и пример."""
    path = Path(file_path)
    if not path.exists():
        return [], {}
        
    try:
        sample_data = {}
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline()
                if line:
                    sample_data = json.loads(line)
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    sample_data = data[0]
                elif isinstance(data, dict):
                    # Если это словарь колонок, просто берем ключи
                    return list(data.keys()), {k: v[0] if isinstance(v, list) else v for k, v in data.items()}
        elif path.suffix == ".csv":
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row:
                    sample_data = row
        
        if sample_data:
            # Для get_dataset_columns нам не нужно возвращать плоские ключи, 
            # так как render_sft_main_config теперь сам строит дерево.
            # Но для совместимости оставим возврат sample_data
            return [], sample_data
            
    except Exception as e:
        st.error(t("error.file_read", error=str(e)))
        return [], {}
    
    return [], {}


def flatten_json_structure(d: dict, parent_path: str = '', depth: int = 0) -> list:
    """Преобразует JSON в плоский список с информацией о вложенности для отрисовки дерева."""
    items = []
    for k, v in d.items():
        current_path = f"{parent_path}.{k}" if parent_path else k
        
        if isinstance(v, dict):
            # Папка
            items.append({"type": "folder", "key": k, "path": current_path, "depth": depth, "val": ""})
            items.extend(flatten_json_structure(v, current_path, depth + 1))
        elif isinstance(v, list):
            # Список
            items.append({"type": "list", "key": k, "path": current_path, "depth": depth, "val": f"List[{len(v)}]"})
            # Показываем структуру первого элемента, если он есть
            if v:
                if isinstance(v[0], dict):
                    items.extend(flatten_json_structure(v[0], current_path, depth + 1))
                else:
                    # Список примитивов (строк и т.д.) - показываем как лист
                    # Но нам не нужно показывать каждый элемент, просто даем понять что внутри
                    # Для SFT мы обычно не выбираем элементы списка по отдельности, а весь список
                    pass
        else:
            # Лист (значение)
            items.append({"type": "leaf", "key": k, "path": current_path, "depth": depth, "val": str(v)})
    return items


def get_all_leaf_paths(data, parent_path: str = '', depth: int = 0, max_depth: int = 10) -> list:
    """Рекурсивно получает ВСЕ пути к значениям, включая глубокую вложенность."""
    if depth > max_depth:
        return []
    
    paths = []
    
    if isinstance(data, dict):
        for k, v in data.items():
            current_path = f"{parent_path}.{k}" if parent_path else k
            
            if isinstance(v, dict):
                # Вложенный словарь - рекурсивно
                paths.extend(get_all_leaf_paths(v, current_path, depth + 1, max_depth))
            elif isinstance(v, list):
                # Добавляем сам список как опцию (для chat-формата)
                paths.append(f"{current_path} [список из {len(v)} эл.]")
                # Раскрываем структуру первого элемента
                if v:
                    if isinstance(v[0], dict):
                        # Показываем поля внутри элементов списка
                        for inner_k, inner_v in v[0].items():
                            inner_path = f"{current_path}[].{inner_k}"
                            if isinstance(inner_v, dict):
                                paths.extend(get_all_leaf_paths(inner_v, inner_path, depth + 2, max_depth))
                            elif isinstance(inner_v, list):
                                paths.append(f"{inner_path} [список]")
                                if inner_v and isinstance(inner_v[0], dict):
                                    paths.extend(get_all_leaf_paths(inner_v[0], f"{inner_path}[]", depth + 3, max_depth))
                            else:
                                paths.append(inner_path)
                    else:
                        # Список примитивов
                        paths.append(f"{current_path}[0]")
            else:
                # Простое значение
                paths.append(current_path)
    
    return paths


def render_sft_main_config(data_path: str):
    """Универсальный конфигуратор SFT — автодетект + ручной выбор.
    
    Args:
        data_path: Путь к датасету
    """
    st.markdown(f"### 🛠️ {t('sft.data_config')}")
    
    columns, sample = get_dataset_columns(data_path)
    
    if not sample:
        st.error(t("error.file_empty"))
        return {}
    
    # ===== АВТОДЕТЕКТ ФОРМАТА =====
    def detect_chat_field(data: dict) -> tuple:
        """Ищет поле со списком сообщений (chat-формат)."""
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                first_item = value[0]
                # Проверяем есть ли поля похожие на role/content
                has_role = any(k.lower() in ['role', 'from', 'type'] for k in first_item.keys())
                has_content = any(k.lower() in ['content', 'value', 'text', 'message'] for k in first_item.keys())
                if has_role and has_content:
                    return key, value
        return None, None
    
    chat_field, chat_value = detect_chat_field(sample)
    detected_format = "chat" if chat_field else "instruct"
    
    # Получаем все пути для instruct режима
    all_paths = get_all_leaf_paths(sample)
    simple_fields = [k for k in sample.keys() if not isinstance(sample[k], (dict, list))]
    
    col_json, col_config = st.columns([1, 1])
    
    # ===== ЛЕВАЯ КОЛОНКА: JSON превью =====
    with col_json:
        st.markdown(f"#### 📄 {t('sft.example_record')}:")
        with st.container(height=500):
            st.json(sample, expanded=True)
    
    # ===== ПРАВАЯ КОЛОНКА: Конфигурация =====
    with col_config:
        # Показываем что автодетект нашел
        if detected_format == "chat":
            st.success(f"🔍 **{t('sft.detected_chat')}** `{chat_field}`")
        else:
            st.info(f"🔍 **{t('sft.detected_instruct')}**")
        
        # Переключатель режима
        format_choice = st.radio(
            "Формат данных:",
            ["💬 Chat (список сообщений)", "📝 Instruct (отдельные поля)"],
            index=0 if detected_format == "chat" else 1,
            key="sft_format_choice",
            horizontal=True
        )
        
        is_chat = "Chat" in format_choice
        
        st.markdown("---")
        
        sft_columns = {}
        
        if is_chat:
            # ===== CHAT РЕЖИМ =====
            st.markdown(f"#### 💬 {t('sft.chat_config')}")
            
            # Выбор поля со списком сообщений
            list_fields = [k for k, v in sample.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
            
            if not list_fields:
                st.error(f"❌ {t('warning.no_messages_field')}")
                return {}
            
            messages_field = st.selectbox(
                "📋 Поле с сообщениями:",
                list_fields,
                index=list_fields.index(chat_field) if chat_field in list_fields else 0,
                key="sft_messages_field"
            )
            
            messages = sample[messages_field]
            first_msg = messages[0]
            inner_fields = list(first_msg.keys())
            
            st.caption(f"Найдено {len(messages)} сообщений")
            
            # Маппинг полей
            c1, c2 = st.columns(2)
            
            role_guess = next((f for f in inner_fields if f.lower() in ['role', 'from', 'type']), inner_fields[0])
            content_guess = next((f for f in inner_fields if f.lower() in ['content', 'value', 'text', 'message']), inner_fields[-1])
            
            role_field = c1.selectbox(
                "Поле **роли**:",
                inner_fields,
                index=inner_fields.index(role_guess) if role_guess in inner_fields else 0,
                key="sft_chat_role"
            )
            content_field = c2.selectbox(
                "Поле **текста**:",
                inner_fields,
                index=inner_fields.index(content_guess) if content_guess in inner_fields else 0,
                key="sft_chat_content"
            )
            
            # Уникальные роли
            unique_roles = sorted(set(str(m.get(role_field, "")) for m in messages))
            st.caption(f"Роли в данных: `{', '.join(unique_roles)}`")
            
            # Маппинг ролей
            st.markdown("**Соответствие ролей:**")
            c1, c2, c3 = st.columns(3)
            
            sys_guess = next((r for r in unique_roles if 'system' in r.lower()), None)
            user_guess = next((r for r in unique_roles if r.lower() in ['user', 'human']), unique_roles[0] if unique_roles else "")
            asst_guess = next((r for r in unique_roles if r.lower() in ['assistant', 'gpt', 'bot']), unique_roles[-1] if len(unique_roles) > 1 else "")
            
            role_system = c1.selectbox("⚙️ System =", ["(нет)"] + unique_roles,
                index=(unique_roles.index(sys_guess) + 1) if sys_guess in unique_roles else 0, key="sft_map_sys")
            role_user = c2.selectbox("👤 User =", unique_roles,
                index=unique_roles.index(user_guess) if user_guess in unique_roles else 0, key="sft_map_user")
            role_assistant = c3.selectbox("🤖 Assistant =", unique_roles,
                index=unique_roles.index(asst_guess) if asst_guess in unique_roles else 0, key="sft_map_asst")
            
            sft_columns = {
                "format": "chat",
                "messages_path": messages_field,
                "role_field": role_field,
                "content_field": content_field,
                "role_system": role_system if role_system != "(нет)" else "",
                "role_user": role_user,
                "role_assistant": role_assistant
            }
            
        else:
            # ===== INSTRUCT РЕЖИМ =====
            st.markdown(f"#### 📝 {t('sft.instruct_config')}")
            st.caption(t("sft.select_role_fields"))
            
            # Все доступные пути
            field_options = ["(не выбрано)"] + all_paths
            
            system_path = st.selectbox(f"⚙️ **System** ({t('common.optional')}):", field_options, index=0, key="sft_inst_sys")
            user_path = st.selectbox(f"👤 **User** ({t('sft.input_instruction')}):", field_options, index=0, key="sft_inst_user")
            assistant_path = st.selectbox(f"🤖 **Assistant** ({t('sft.output_answer')}):", field_options, index=0, key="sft_inst_asst")
            
            if user_path == "(не выбрано)" or assistant_path == "(не выбрано)":
                st.warning(f"👆 {t('warning.select_fields')}")
                return {}
            
            sft_columns = {
                "format": "instruct",
                "instruction": user_path,
                "output": assistant_path,
                "system_field": system_path if system_path != "(не выбрано)" else ""
            }
        
        # ===== НАСТРОЙКИ ШАБЛОНА =====
        st.markdown("---")
        with st.expander(f"🏷️ {t('sft.tags_system')}", expanded=False):
            default_system = st.text_input(f"{t('sft.default_system')}:", "You are a helpful assistant.", key="sft_def_sys")
            
            # Qwen-style теги по умолчанию
            tc1, tc2, tc3 = st.columns(3)
            im_start = tc1.text_input("Start tag:", "<|im_start|>", key="sft_im_start")
            im_end = tc2.text_input("End tag:", "<|im_end|>", key="sft_im_end")
            separator = tc3.text_input("Separator:", "\n", key="sft_separator")
            
            st.caption("💡 Qwen-style формат: `<|im_start|>role\\ncontent<|im_end|>`")
        
        if 'default_system' not in dir():
            default_system = "You are a helpful assistant."
        if 'im_start' not in dir():
            im_start, im_end, separator = "<|im_start|>", "<|im_end|>", "\n"
        
        sft_template = {
            "system": default_system,
            "separator": separator,
            "im_start": im_start,
            "im_end": im_end,
            # Legacy fallback
            "user_tag": f"{im_start}user{separator}",
            "bot_tag": f"{im_start}assistant{separator}",
            "end_tag": im_end,
        }
        
        # ===== ПРЕВЬЮ =====
        st.markdown("---")
        st.markdown(f"#### 👁️ {t('sft.preview')}:")
        
        try:
            # Получаем токенизатор из session_state (если загружен для SFT)
            sft_tokenizer = st.session_state.get("sft_tokenizer")
            use_model_chat_template = False
            
            # Проверяем, есть ли у токенизатора chat_template
            if sft_tokenizer and hasattr(sft_tokenizer, 'chat_template') and sft_tokenizer.chat_template:
                use_model_chat_template = True
            
            # Также проверяем пользовательский chat_template
            user_chat_template = st.session_state.get("sft_user_chat_template", "").strip()
            if user_chat_template and sft_tokenizer:
                # Временно устанавливаем пользовательский шаблон
                sft_tokenizer.chat_template = user_chat_template
                use_model_chat_template = True
            
            preview = ""
            
            if sft_columns["format"] == "chat":
                # Получаем сообщения из sample
                messages = sample[sft_columns["messages_path"]]
                
                # Конвертируем в стандартный формат messages
                std_messages = []
                for msg in messages:
                    role_val = str(msg.get(sft_columns["role_field"], ""))
                    content_val = str(msg.get(sft_columns["content_field"], ""))[:300]
                    
                    # Маппинг ролей
                    if role_val == sft_columns.get("role_system"):
                        std_messages.append({"role": "system", "content": content_val})
                    elif role_val == sft_columns.get("role_user"):
                        std_messages.append({"role": "user", "content": content_val})
                    elif role_val == sft_columns.get("role_assistant"):
                        std_messages.append({"role": "assistant", "content": content_val})
                
                # Если нет системного сообщения, добавляем дефолтное
                if not std_messages or std_messages[0]["role"] != "system":
                    std_messages.insert(0, {"role": "system", "content": default_system})
                
                # Используем chat_template если есть
                if use_model_chat_template and sft_tokenizer:
                    try:
                        preview = sft_tokenizer.apply_chat_template(
                            std_messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        st.caption("✨ Превью сформировано через **chat_template модели**")
                    except Exception as e:
                        st.warning(f"Ошибка apply_chat_template: {e}. Используем fallback.")
                        use_model_chat_template = False
                
                # Fallback: Qwen-style формат с тегами
                if not use_model_chat_template or not preview:
                    sys_text = default_system
                    preview = ""
                    
                    for msg in messages:
                        role = str(msg.get(sft_columns["role_field"], ""))
                        content = str(msg.get(sft_columns["content_field"], ""))
                        
                        if role == sft_columns["role_system"]:
                            sys_text = content
                        elif role == sft_columns["role_user"]:
                            content_preview = content[:200] + ('...' if len(content) > 200 else '')
                            preview += f"{im_start}user{separator}{content_preview}{im_end}{separator}"
                        elif role == sft_columns["role_assistant"]:
                            content_preview = content[:200] + ('...' if len(content) > 200 else '')
                            preview += f"{im_start}assistant{separator}{content_preview}{im_end}{separator}"
                    
                    # System в начале
                    preview = f"{im_start}system{separator}{sys_text}{im_end}{separator}" + preview
                    st.caption("ℹ️ Превью сформировано через **теги** (chat_template не используется)")
            else:
                # Instruct формат
                user_val = str(get_nested_value(sample, sft_columns["instruction"]) or "")[:300]
                asst_val = str(get_nested_value(sample, sft_columns["output"]) or "")[:300]
                
                # System prompt
                sys_val = default_system
                system_field = sft_columns.get("system_field")
                if system_field and system_field != "(не выбрано)" and system_field.strip():
                    field_sys = get_nested_value(sample, system_field)
                    if field_sys is not None:
                        field_sys_str = str(field_sys).strip()
                        if field_sys_str:
                            sys_val = field_sys_str[:200]
                
                # Формируем messages для chat_template
                std_messages = [
                    {"role": "system", "content": sys_val},
                    {"role": "user", "content": user_val},
                    {"role": "assistant", "content": asst_val}
                ]
                
                # Используем chat_template если есть
                if use_model_chat_template and sft_tokenizer:
                    try:
                        preview = sft_tokenizer.apply_chat_template(
                            std_messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        st.caption("✨ Превью сформировано через **chat_template модели**")
                    except Exception as e:
                        st.warning(f"Ошибка apply_chat_template: {e}. Используем fallback.")
                        use_model_chat_template = False
                
                # Fallback: Qwen-style формат
                if not use_model_chat_template or not preview:
                    preview = (
                        f"{im_start}system{separator}{sys_val}{im_end}{separator}"
                        f"{im_start}user{separator}{user_val}{im_end}{separator}"
                        f"{im_start}assistant{separator}{asst_val}{im_end}"
                    )
                    st.caption("ℹ️ Превью сформировано через **теги** (chat_template не используется)")
            
            with st.container(height=400):
                st.code(preview, language=None)
            
            st.success(f"✅ {t('success.ready')}")
            
        except Exception as e:
            st.error(f"{t('error.generic')}: {e}")

    return {"sft_columns": sft_columns, "sft_template": sft_template}


# ============================================================================
# GRPO Configuration
# ============================================================================

def render_grpo_sidebar_config():
    """Конфигурация GRPO в сайдбаре.
    
    Training Backend теперь выбирается в render_model_config() (до выбора модели),
    вместе с методом тюнинга (lora/qlora/full).
    """
    st.sidebar.subheader(f"🧠 {t('grpo.params')}")
    
    # Алгоритм
    algorithm = st.sidebar.selectbox(
        "Алгоритм",
        ["grpo", "dapo", "dr_grpo", "sdpo"],
        format_func=lambda x: {
            "grpo": "⭐ GRPO (рекомендуется)",
            "dapo": "DAPO (Dynamic sAmpling)",
            "dr_grpo": "Dr.GRPO (улучшенный)",
            "sdpo": "🎓 SDPO (Self-Distillation)",
        }[x],
        help="""
        **GRPO** ⭐: Рекомендуется! Стандартный Group Relative Policy Optimization
        **DAPO**: Token-level loss + асимметричный клиппинг + dynamic sampling
        **Dr.GRPO**: Без деления на std, фиксированная нормализация
        **SDPO** 🎓: Self-Distilled Policy Optimization — успешные траектории как учитель
        """
    )
    
    # ============================================================
    # 🔥 ПРЕСЕТЫ ДЛЯ КАЖДОГО АЛГОРИТМА
    # ============================================================
    ALGORITHM_PRESETS = {
        "grpo": {
            "kl_weight": 0.0,  # GRPO работает без KL
            "clip_eps_high": 0.2,
            "group_size": 8,
            "learning_rate": 5e-6,  # Full fine-tuning
            "train_batch_size": 2,
            "max_new_tokens": 1024,
            "description": "Стандартный GRPO",
        },
        "dapo": {
            "kl_weight": 0.0,  # DAPO без KL
            "clip_eps_high": 0.28,  # Асимметричный клиппинг
            "group_size": 8,
            "learning_rate": 5e-6,
            "train_batch_size": 2,
            "max_new_tokens": 1024,
            "description": "DAPO с асимметричным клиппингом",
        },
        "dr_grpo": {
            "kl_weight": 0.001,  # Dr.GRPO рекомендует KL
            "clip_eps_high": 0.2,
            "group_size": 8,
            "learning_rate": 5e-6,
            "train_batch_size": 2,
            "max_new_tokens": 1024,
            "description": "Dr.GRPO с фиксированной нормализацией",
        },
        "sdpo": {
            # 🔥 SDPO ТРЕБУЕТ reference модель для teacher!
            "kl_weight": 0.01,  # Обязательно > 0 для teacher!
            "clip_eps_high": 0.2,
            "group_size": 8,
            "learning_rate": 5e-6,
            "train_batch_size": 2,
            "max_new_tokens": 1024,
            # SDPO-специфичные
            "sdpo_distillation_topk": 50,  # Top-K distillation
            "sdpo_ema_rate": 0.0,  # Teacher = Ref model
            "description": "🔥 SDPO с Top-K Distillation",
        },
    }
    
    preset = ALGORITHM_PRESETS[algorithm]
    
    # Показываем информацию о пресете
    if algorithm == "sdpo":
        st.sidebar.info(
            "🎓 **SDPO требует reference модель!**\n\n"
            "KL weight автоматически установлен > 0 для загрузки teacher модели.\n\n"
            "**Оптимизации из verl:**\n"
            "• Top-K Distillation (k=50)\n"
            "• Teacher = Reference Model"
        )
    elif algorithm == "dapo":
        st.sidebar.info(
            "🎯 **DAPO пресет:**\n\n"
            "• Асимметричный клиппинг (0.28)\n"
            "• Dynamic sampling\n"
            "• Token-level loss"
        )
    elif algorithm == "dr_grpo":
        st.sidebar.info(
            "🔬 **Dr.GRPO пресет:**\n\n"
            "• Без деления на std\n"
            "• Фиксированная нормализация\n"
            "• KL weight = 0.001"
        )
    
    # Генерация
    group_size = st.sidebar.slider(
        "Group size (G)",
        min_value=8,
        max_value=32,
        value=preset.get("group_size", 8),
        help="Количество генераций на один промпт. Важно: для GRPO обычно нужно G>=8 для стабильного обучения."
    )

    prompt_batch_size = st.sidebar.slider(
        "Prompt batch size (prompts/step)",
        min_value=1,
        max_value=64,
        value=8,
        step=1,
        help="Сколько разных промптов (задач) брать на один RL-шаг (rollouts_per_step в re-grpo)."
    )
    
    max_new_tokens = st.sidebar.slider(
        "Max new tokens",
        min_value=128,
        max_value=16384,
        value=preset.get("max_new_tokens", 1024),  # 🔥 Из пресета
        step=128,
        help="Максимальная длина генерируемого ответа"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Температура сэмплирования"
    )
    
    # Обучение — learning rate зависит от метода тюнинга
    # Если LoRA — используем 5e-5, если full fine-tuning — из пресета (5e-6)
    lr_options = [1e-7, 5e-7, 1e-6, 3e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4]
    # Берём ближайшее значение из пресета
    preset_lr = preset.get("learning_rate", 5e-6)
    default_lr = min(lr_options, key=lambda x: abs(x - preset_lr))
    
    grpo_learning_rate = st.sidebar.select_slider(
        "Learning Rate (GRPO)",
        options=lr_options,
        value=default_lr,  # 🔥 Из пресета
        format_func=lambda x: f"{x:.0e}",
        help="""**Для LoRA:** рекомендуется **5e-5** — сразу начинает учиться.
**Для full fine-tuning:** 1e-6 — 5e-6.
Слишком низкий LR с LoRA = медленная сходимость!"""
    )
    
    train_batch_size = st.sidebar.slider(
        "Train Batch Size",
        min_value=1,
        max_value=128,
        value=preset.get("train_batch_size", 2),  # 🔥 Из пресета
        step=1,
        help="Размер микро-батча при обучении на опыте. Уменьшите до 1-2, если возникает OOM"
    )

    grpo_grad_accum = st.sidebar.slider(
        "Gradient accumulation steps",
        min_value=1,
        max_value=32,
        value=4,
        step=1,
        help="Накопление градиентов (как в PPO/GRPO). Не меняет семантику данных, только эффективный batch на шаг оптимизации."
    )
    
    # Лимит по данным (понятнее пользователю, чем optim-steps).
    # По умолчанию: весь датасет (с учётом "Макс. примеров" в main-config).
    effective_ds = st.session_state.get("grpo_effective_dataset_size", None)
    if isinstance(effective_ds, int) and effective_ds > 0:
        grpo_max_prompts = st.sidebar.number_input(
            "Max prompts (по датасету)",
            min_value=1,
            max_value=int(effective_ds),
            value=int(effective_ds),
            step=max(1, int(effective_ds) // 50),
            help="Сколько задач (prompts) пройти всего. По умолчанию = весь датасет (с учётом max_samples).",
        )
    else:
        st.sidebar.info("Выберите датасет в GRPO (вкладка Запуск), чтобы лимит считался от его размера.")
        grpo_max_prompts = None
    
    epochs_per_step = st.sidebar.slider(
        "Epochs per step",
        min_value=1,
        max_value=5,
        value=1,
        help="Сколько раз обновлять политику на каждом батче rollout'ов"
    )
    
    # KL — используем пресет для default value!
    kl_weight_default = preset.get("kl_weight", 0.0)
    kl_weight_help = """**KL penalty** ограничивает отклонение от исходной политики.

- **0.0** — без KL (обычно для full fine-tuning)
- **0.001** — стандартное значение (если нестабильно)
- **0.01+** — сильный constraint"""
    
    if algorithm == "sdpo":
        kl_weight_help += "\n\n🔥 **SDPO:** KL > 0 обязателен для загрузки teacher модели!"
    
    kl_weight = st.sidebar.slider(
        "KL weight",
        min_value=0.0,
        max_value=0.1,
        value=kl_weight_default,  # 🔥 Используем пресет!
        step=0.001,
        format="%.3f",
        help=kl_weight_help
    )
    
    # 🔥 Предупреждение если SDPO с kl_weight=0
    if algorithm == "sdpo" and kl_weight == 0.0:
        st.sidebar.warning(
            "⚠️ **SDPO без reference модели!**\n\n"
            "Рекомендуется KL weight > 0 для загрузки teacher модели. "
            "Без неё teacher = student (менее эффективно)."
        )
    
    # Клиппинг и продвинутые параметры
    with st.sidebar.expander("⚙️ Продвинутые параметры"):
        clip_eps_low = st.slider("Clip ε (low)", 0.1, 0.3, 0.2, 0.01)
        clip_eps_high = st.slider(
            "Clip ε (high)", 
            0.1, 0.4, 
            preset.get("clip_eps_high", 0.2),  # 🔥 Из пресета!
            0.01,
            help="DAPO рекомендует 0.28 для верхней границы"
        )

        min_lr_ratio = st.slider(
            "Min LR ratio (floor)",
            0.0, 0.5,
            0.1,
            0.01,
            help="Нижний предел LR: lr = base_lr * ratio в конце cosine. 0.0 = до нуля."
        )
        
        # ============================================================
        # Настройки специфичные для каждого алгоритма
        # ============================================================
        
        if algorithm == "dapo":
            st.markdown("---")
            st.markdown("**🎯 DAPO-специфичные настройки**")
            st.success(
                "✅ **DAPO пресет применён!**\n\n"
                "• Clip ε high = 0.28 (асимметричный)\n"
                "• Dynamic sampling = ON\n"
                "• Token-level loss = ON"
            )
            
            dynamic_sampling = st.checkbox(
                "Dynamic sampling",
                value=True,
                help=(
                    "Фильтровать группы где все rewards одинаковы (zero-gradient).\n\n"
                    "**⚠️ Замедляет обучение** — делает дополнительные генерации!\n"
                    "Отключите если скорость важнее качества.\n\n"
                    "**⚠️ Multi-GPU:** может вызывать NCCL timeout если GPU получают разное кол-во данных!"
                )
            )
            
            # Предупреждение для multi-GPU
            if dynamic_sampling:
                st.warning(
                    "⚠️ **Multi-GPU + Dynamic sampling**\n\n"
                    "При 2+ GPU может вызывать рассинхронизацию (NCCL timeout). "
                    "Рекомендуется отключить dynamic sampling или уменьшить max_refill_rounds до 1."
                )
            
            # Максимум попыток добора (только если dynamic_sampling включён)
            if dynamic_sampling:
                max_refill_rounds = st.slider(
                    "Max refill rounds",
                    min_value=1,
                    max_value=8,
                    value=1,  # 🔥 Уменьшил default для стабильности
                    step=1,
                    help=(
                        "Сколько раз пытаться добирать группы.\n"
                        "**1** = рекомендуется для Multi-GPU (минимум рассинхронизации)\n"
                        "**2-3** = только для Single-GPU\n"
                        "**8** = максимум (медленно)"
                    )
                )
            else:
                max_refill_rounds = 0
            
            token_level_loss = st.checkbox(
                "Token-level loss",
                value=True,
                help="Агрегировать loss по токенам (DAPO), а не по сэмплам (GRPO)"
            )
        
        elif algorithm == "dr_grpo":
            st.markdown("---")
            st.markdown("**🔬 Dr.GRPO-специфичные настройки**")
            st.success(
                "✅ **Dr.GRPO пресет применён!**\n\n"
                "• KL weight = 0.001\n"
                "• Без деления на std\n"
                "• Фиксированная нормализация"
            )
            # Dr.GRPO не использует dynamic_sampling и token_level_loss
            dynamic_sampling = False
            max_refill_rounds = 0
            token_level_loss = False
            # SDPO параметры не нужны
            sdpo_success_threshold = 0.5
            sdpo_alpha = 0.5
            sdpo_loss_weight = 1.0
            sdpo_distillation_topk = None
            sdpo_full_logit_distillation = False
            sdpo_ema_rate = 0.0
        
        elif algorithm == "sdpo":
            st.markdown("---")
            st.markdown("**🎓 SDPO-специфичные настройки**")
            st.success(
                "✅ **SDPO пресет применён!**\n\n"
                "• KL weight = 0.01 (для загрузки teacher модели)\n"
                "• Top-K Distillation = 50 (экономия памяти)\n"
                "• Teacher = Reference Model"
            )
            
            sdpo_success_threshold = st.slider(
                "Success threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Порог reward для 'успешной' траектории. Траектории с reward >= threshold становятся учителем."
            )
            
            sdpo_alpha = st.slider(
                "Alpha (KL type)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0 = Forward KL (mode-seeking), 1 = Reverse KL (mode-covering), 0.5 = Jensen-Shannon (рекомендуется)"
            )
            
            sdpo_loss_weight = st.slider(
                "Distillation loss weight",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Вес distillation loss относительно GRPO loss. total_loss = grpo_loss + weight * distill_loss"
            )
            
            st.markdown("---")
            st.markdown("**🔥 Оптимизации памяти (из verl)**")
            
            # 🔥 Используем пресет для Top-K
            preset_topk = preset.get("sdpo_distillation_topk", 50)
            sdpo_distillation_topk = st.number_input(
                "Top-K Distillation (k)",
                min_value=0,
                max_value=500,
                value=preset_topk,  # 🔥 Из пресета!
                step=10,
                help="🔥 Вместо KL по всему vocab (152k) используем только top-k токенов. "
                     "k=50-100 экономит ~99% памяти! 0 = отключено"
            )
            sdpo_distillation_topk = sdpo_distillation_topk if sdpo_distillation_topk > 0 else None
            
            sdpo_full_logit_distillation = sdpo_distillation_topk is not None
            
            # 🔥 Используем пресет для EMA
            preset_ema = preset.get("sdpo_ema_rate", 0.0)
            sdpo_ema_rate = st.slider(
                "EMA Teacher rate",
                min_value=0.0,
                max_value=0.1,
                value=preset_ema,  # 🔥 Из пресета!
                step=0.01,
                format="%.2f",
                help="""**EMA Teacher** = медленно обновляемая копия student модели.

**При EMA = 0:**
• Full FT: Teacher = Reference Model
• LoRA: Teacher = Student (detached)

**При EMA > 0:**
• Full FT: EMA update reference model
• LoRA: EMA только LoRA весов (~50-200 MB) ✅

**Рекомендация:** 0.01-0.05 для более стабильного обучения."""
            )
            
            # Информация о LoRA + EMA
            if sdpo_ema_rate > 0:
                st.info(
                    "💡 **EMA активирован!**\n\n"
                    "При LoRA: хранится только EMA копия LoRA адаптеров (~50-200 MB), "
                    "не всей модели. Это позволяет использовать EMA Teacher без "
                    "значительного overhead по памяти."
                )
            
            # SDPO использует dynamic_sampling по умолчанию
            dynamic_sampling = st.checkbox(
                "Dynamic sampling",
                value=False,  # 🔥 Отключаем по умолчанию для стабильности multi-GPU
                help=(
                    "Фильтровать группы где все rewards одинаковы.\n\n"
                    "**⚠️ Multi-GPU:** может вызывать NCCL timeout! "
                    "Рекомендуется **отключить** при 2+ GPU."
                )
            )
            
            if dynamic_sampling:
                st.warning(
                    "⚠️ **Multi-GPU + Dynamic sampling**\n\n"
                    "При 2+ GPU может вызывать рассинхронизацию. "
                    "Отключите для стабильности!"
                )
                max_refill_rounds = st.slider(
                    "Max refill rounds",
                    min_value=1,
                    max_value=3,
                    value=1,
                    help="Сколько раз добирать группы. 1 = минимум для Multi-GPU"
                )
            else:
                max_refill_rounds = 0
            token_level_loss = False
        
        else:  # GRPO
            st.markdown("---")
            st.markdown("**📊 GRPO-специфичные настройки**")
            st.success(
                "✅ **GRPO пресет применён!**\n\n"
                "• KL weight = 0 (без penalty)\n"
                "• Нормализация: (r - mean) / std\n"
                "• Sample-level loss"
            )
            # GRPO не использует dynamic_sampling и token_level_loss
            dynamic_sampling = False
            max_refill_rounds = 0
            token_level_loss = False
            # SDPO параметры не нужны
            sdpo_success_threshold = 0.5
            sdpo_alpha = 0.5
            sdpo_loss_weight = 1.0
            sdpo_distillation_topk = None
            sdpo_full_logit_distillation = False
            sdpo_ema_rate = 0.0
        
        # DAPO тоже нужны default SDPO параметры
        if algorithm == "dapo":
            sdpo_success_threshold = 0.5
            sdpo_alpha = 0.5
            sdpo_loss_weight = 1.0
            sdpo_distillation_topk = None
            sdpo_full_logit_distillation = False
            sdpo_ema_rate = 0.0

        # Liger настройки берутся из общих настроек Precision & Memory (сайдбар)
        # Здесь только вычисляем loss_type на основе алгоритма
        algorithm_to_loss_type = {
            "grpo": "grpo",
            "dapo": "dapo", 
            "dr_grpo": "dr_grpo",
            "sdpo": "dr_grpo",  # SDPO использует Dr.GRPO как базу
        }
        grpo_liger_loss_type = algorithm_to_loss_type.get(algorithm, "grpo")
        
        st.markdown("---")
        st.markdown("**🚀 Rollout engine (отдельная модель для генерации)**")
        grpo_use_rollout_engine = st.checkbox(
            "Использовать отдельную модель для генерации",
            value=False,
            help=(
                "Если включено, генерация (rollout) будет выполняться отдельной моделью, "
                "а DDP/ZeRO-3/FSDP модель будет использоваться только для teacher-forcing logprobs и backprop. "
                "Это радикально ускоряет GRPO при ZeRO-3/FSDP."
            ),
        )
        grpo_rollout_backend = st.selectbox(
            "Rollout backend",
            options=["hf", "vllm"],
            index=0,
            help=(
                "**hf** = HuggingFace модель. Универсальный, работает везде.\n\n"
                "**vllm** = vLLM (PagedAttention, continuous batching). Максимальная скорость генерации!\n"
                "- Для LoRA: быстрая синхронизация (~секунды)\n"
                "- Для full fine-tuning: перезагрузка vLLM (~5-15 сек), увеличьте sync interval"
            ),
            disabled=not grpo_use_rollout_engine,
        )
        
        # vLLM: на какой GPU запускать?
        # Получаем количество GPU
        try:
            import torch
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except:
            num_gpus = 0
        
        # Предупреждение о multi-GPU DDP + vLLM
        selected_num_gpus = st.session_state.get("num_gpus", 1) or 1
        if grpo_rollout_backend == "vllm" and selected_num_gpus > 1:
            st.warning(
                "⚠️ **vLLM + Multi-GPU DDP не поддерживается!**\n\n"
                "При использовании нескольких GPU для DDP training, vLLM автоматически отключается.\n\n"
                "**Для использования vLLM**: выберите 1 GPU для training, "
                "а vLLM разместите на другой GPU."
            )
        
        # Выбор GPU для vLLM
        gpu_options = [f"cuda:{i}" for i in range(num_gpus)] if num_gpus > 0 else ["cuda:0"]
        gpu_labels = {}
        for i in range(num_gpus):
            try:
                name = torch.cuda.get_device_name(i)
                gpu_labels[f"cuda:{i}"] = f"🎮 GPU {i}: {name}"
            except:
                gpu_labels[f"cuda:{i}"] = f"🎮 GPU {i}"
        
        # Определяем GPU для training (первая выбранная в multi-select или 0)
        training_gpu_id = 0
        selected_gpus = st.session_state.get("selected_gpus", [0])
        if selected_gpus:
            training_gpu_id = selected_gpus[0] if isinstance(selected_gpus[0], int) else 0
        
        # По умолчанию vLLM на другой GPU если есть
        default_vllm_gpu_idx = 0
        if num_gpus > 1:
            # Выбираем GPU которая НЕ используется для training
            for i in range(num_gpus):
                if i != training_gpu_id:
                    default_vllm_gpu_idx = i
                    break
        
        grpo_vllm_device = st.selectbox(
            "vLLM GPU",
            options=gpu_options,
            index=min(default_vllm_gpu_idx, len(gpu_options) - 1),
            format_func=lambda x: gpu_labels.get(x, x),
            help=(
                "Выберите GPU для vLLM rollout модели.\n\n"
                "**Рекомендация**: Используйте ОТДЕЛЬНУЮ GPU для vLLM!\n"
                f"- Training на: cuda:{training_gpu_id}\n"
                f"- vLLM лучше на: cuda:{default_vllm_gpu_idx if num_gpus > 1 else 0}\n\n"
                "Если только 1 GPU — vLLM и training делят память (уменьшите % памяти для vLLM)."
            ),
            disabled=not (grpo_use_rollout_engine and grpo_rollout_backend == "vllm"),
        )
        
        # Предупреждение если только 1 GPU
        if grpo_rollout_backend == "vllm" and num_gpus <= 1:
            st.warning(
                "⚠️ **Доступна только 1 GPU!**\n\n"
                "vLLM и training будут на одной GPU.\n"
                "- Уменьшите **vLLM GPU Memory** до 30-40%\n"
                "- Или пробросьте больше GPU в Docker (`--gpus all`)"
            )
        
        # Определяем, на той же GPU или на отдельной
        vllm_gpu_id = int(grpo_vllm_device.split(":")[1]) if grpo_vllm_device.startswith("cuda:") else 0
        same_gpu = (vllm_gpu_id == training_gpu_id)
        
        if grpo_rollout_backend == "vllm" and same_gpu and num_gpus > 1:
            st.warning(
                f"⚠️ vLLM и training на одной GPU (cuda:{training_gpu_id})!\n"
                f"Рекомендуется выбрать другую GPU для vLLM."
            )
        
        # vLLM GPU memory utilization
        # Если на отдельной GPU — можно больше, если на той же — меньше
        default_memory = 40 if same_gpu else 85
        max_memory = 60 if same_gpu else 95
        
        grpo_vllm_gpu_memory = st.slider(
            "vLLM GPU Memory (%)",
            min_value=10,
            max_value=max_memory,
            value=default_memory,
            step=5,
            help=(
                "Сколько % GPU памяти выделить для vLLM.\n\n"
                "Если vLLM на отдельной GPU — ставьте 70-90%.\n"
                "Если на той же GPU что training — ставьте 30-50%."
            ),
            disabled=not (grpo_use_rollout_engine and grpo_rollout_backend == "vllm"),
        )
        grpo_rollout_sync_interval = st.slider(
            "Синхронизация весов (каждые N rollout-step)",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help=(
                "1 = максимально on-policy (чаще синхронизация = больше overhead). "
                "2-10 = быстрее, но rollout модель будет чуть 'stale'."
            ),
            disabled=not grpo_use_rollout_engine,
        )
        grpo_rollout_trainable_only = st.checkbox(
            "Синхронизировать только trainable параметры (LoRA)",
            value=True,
            help=(
                "**Рекомендуется для LoRA** — синхронизация быстрая (~MB адаптера).\n\n"
                "**Для full fine-tuning** — снимите галочку, будут синхронизироваться все веса (~GB). "
                "При ZeRO-3 это дорого, поэтому увеличьте sync_interval."
            ),
            disabled=not grpo_use_rollout_engine,
        )
        grpo_rollout_offload_to_cpu = st.checkbox(
            "Offload rollout модель на CPU между генерациями (экономит VRAM)",
            value=False,
            help="Полезно если VRAM не хватает (особенно при ZeRO-3 + full finetune). Может замедлить rollout.",
            disabled=not grpo_use_rollout_engine,
        )
    
    # ВАЖНО: LoRA параметры и квантизация берутся из render_model_config() (секция "🎯 Метод тюнинга")
    # Здесь мы НЕ дублируем их, чтобы избежать повторения в UI
    # Все LoRA параметры (use_lora, lora_r, lora_alpha, lora_dropout, lora_target_modules)
    # и квантизация (use_4bit, use_8bit) будут взяты из model_config
    
    # === Логирование сэмплов ===
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📝 Логирование сэмплов**")
    
    grpo_log_completions = st.sidebar.checkbox(
        "Показывать сгенерированные ответы",
        value=True,
        help="Выводить примеры сгенерированных ответов в консоль"
    )
    
    grpo_completion_log_interval = st.sidebar.slider(
        "Интервал логирования (шаги)",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Каждые N шагов показывать примеры генерации"
    )
    
    return {
        # Training Backend теперь в render_model_config()
        
        # GRPO параметры (обязательные)
        "grpo_algorithm": algorithm,
        "grpo_group_size": group_size,
        "grpo_prompt_batch_size": prompt_batch_size,
        "grpo_max_new_tokens": max_new_tokens,
        "grpo_temperature": temperature,
        "grpo_learning_rate": grpo_learning_rate,
        "grpo_train_batch_size": train_batch_size,
        "gradient_accumulation": grpo_grad_accum,
        "grpo_max_prompts": grpo_max_prompts,
        "grpo_epochs_per_step": epochs_per_step,
        "grpo_kl_weight": kl_weight,
        "grpo_clip_eps_low": clip_eps_low,
        "grpo_clip_eps_high": clip_eps_high,
        "grpo_dynamic_sampling": dynamic_sampling,
        "grpo_max_refill_rounds": max_refill_rounds,
        "grpo_token_level_loss": token_level_loss,
        "grpo_min_lr_ratio": min_lr_ratio,
        
        # SDPO параметры
        "sdpo_success_threshold": sdpo_success_threshold,
        "sdpo_alpha": sdpo_alpha,
        "sdpo_loss_weight": sdpo_loss_weight,
        "sdpo_distillation_topk": sdpo_distillation_topk,  # 🔥 Top-K Distillation
        "sdpo_full_logit_distillation": sdpo_full_logit_distillation,
        "sdpo_ema_rate": sdpo_ema_rate,

        # Liger loss_type — автоматически из алгоритма
        # Остальные Liger настройки берутся из distributed_config (Precision & Memory)
        "grpo_liger_loss_type": grpo_liger_loss_type,

        # Rollout engine
        "grpo_use_rollout_engine": grpo_use_rollout_engine,
        "grpo_rollout_backend": grpo_rollout_backend,
        "grpo_rollout_sync_interval": grpo_rollout_sync_interval,
        "grpo_rollout_trainable_only": grpo_rollout_trainable_only,
        "grpo_rollout_offload_to_cpu": grpo_rollout_offload_to_cpu,
        "grpo_vllm_gpu_memory": grpo_vllm_gpu_memory / 100.0,  # Конвертируем % в 0.0-1.0
        "grpo_vllm_device": grpo_vllm_device,  # "main_gpu" или "cpu"
        
        # Логирование
        "grpo_log_completions": grpo_log_completions,
        "grpo_completion_log_interval": grpo_completion_log_interval,
    }


def render_grpo_main_config(data_path: str = None):
    """Универсальный конструктор Reward функций с визуальным редактором правил."""
    import re
    import json as json_lib
    
    # =========================================================================
    # 1. ДАТАСЕТ ДЛЯ REASONING (расширенная настройка)
    # =========================================================================
    st.markdown(f"### 📚 {t('grpo.dataset_reasoning')}")
    
    grpo_dataset_path = None
    grpo_max_samples = None
    grpo_dataset_language = "en"
    dataset_source = "custom"
    dataset_key = "custom"
    
    # Инициализация session_state для настроек датасета
    if "grpo_field_mapping" not in st.session_state:
        st.session_state.grpo_field_mapping = {
            "prompt_field": "question",
            "reference_field": "answer",
            "metadata_fields": [],
        }
    if "grpo_prompt_template" not in st.session_state:
        st.session_state.grpo_prompt_template = "{{prompt}}"
    if "grpo_system_prompt" not in st.session_state:
        # Дефолтный reasoning промпт с тегами
        st.session_state.grpo_system_prompt = """You are a helpful assistant that solves problems step by step.
Think through the problem carefully inside <reasoning>...</reasoning> tags.
Then provide your final answer inside <answer>...</answer> tags.

Example format:
<reasoning>
Let me analyze this step by step...
Step 1: ...
Step 2: ...
Therefore, the answer is X.
</reasoning>
<answer>X</answer>"""
    
    # Получаем список локальных датасетов
    local_datasets = []
    if DATASET_DIR.exists():
        for f in sorted(DATASET_DIR.iterdir(), key=lambda x: x.name.lower()):
            if f.suffix in (".jsonl", ".json"):
                local_datasets.append(f)
    
    # Формируем список для выбора
    dataset_options = ["-- Выберите датасет --"]
    for f in local_datasets:
        dataset_options.append(str(f))
    
    # Определяем индекс по умолчанию
    default_idx = 0
    saved_selection = st.session_state.get("grpo_dataset_selectbox")
    if saved_selection and saved_selection in dataset_options:
        default_idx = dataset_options.index(saved_selection)
    elif data_path and data_path in dataset_options:
        default_idx = dataset_options.index(data_path)
    elif data_path:
        for i, opt in enumerate(dataset_options):
            if data_path in opt:
                default_idx = i
                break
    
    selected_dataset = st.selectbox(
        t("data.select_dataset"),
        options=dataset_options,
        index=default_idx,
        key="grpo_dataset_selectbox",
        help="Скачайте датасеты на вкладке 'Данные' → 🧠 Reasoning"
    )
    
    # Переменные для хранения данных датасета
    dataset_samples = []
    dataset_fields = []
    
    # Обрабатываем выбор датасета
    if selected_dataset and not selected_dataset.startswith("--"):
        grpo_dataset_path = selected_dataset
        st.session_state.grpo_dataset_path = grpo_dataset_path
        
        # Определяем язык по имени файла
        if "ru" in selected_dataset.lower() or "russian" in selected_dataset.lower():
            grpo_dataset_language = "ru"
        else:
            grpo_dataset_language = "en"
        
        # Загружаем семплы для превью и автодетекта полей
        try:
            p = Path(grpo_dataset_path)
            if p.exists():
                if p.suffix == ".jsonl":
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if i >= 100:  # Читаем до 100 для анализа
                                break
                            line = line.strip()
                            if line:
                                try:
                                    dataset_samples.append(json_lib.loads(line))
                                except:
                                    pass
                elif p.suffix == ".json":
                    with open(p, "r", encoding="utf-8") as f:
                        obj = json_lib.load(f)
                    if isinstance(obj, list):
                        dataset_samples = obj[:100]
                
                # Собираем все поля из датасета
                if dataset_samples:
                    all_fields = set()
                    for sample in dataset_samples[:20]:
                        if isinstance(sample, dict):
                            all_fields.update(sample.keys())
                    dataset_fields = sorted(list(all_fields))
        except Exception as e:
            st.error(f"Ошибка чтения датасета: {e}")
        
        st.success(f"✅ Датасет: `{Path(selected_dataset).name}` ({len(dataset_samples)} примеров загружено для превью)")
        
        # =====================================================================
        # ПРЕВЬЮ ДАННЫХ
        # =====================================================================
        with st.expander(f"👀 {t('data.preview')}", expanded=True):
            if dataset_samples:
                # Показываем первые 5 семплов
                preview_count = min(5, len(dataset_samples))
                
                # Табличный вид
                if dataset_fields:
                    st.caption(f"**Обнаруженные поля:** {', '.join(dataset_fields)}")
                
                # Показываем семплы
                for i, sample in enumerate(dataset_samples[:preview_count]):
                    with st.container():
                        st.markdown(f"**Пример {i+1}:**")
                        # Показываем каждое поле
                        cols = st.columns(2)
                        field_list = list(sample.items()) if isinstance(sample, dict) else []
                        for j, (key, value) in enumerate(field_list):
                            col_idx = j % 2
                            with cols[col_idx]:
                                val_str = str(value)[:300]
                                if len(str(value)) > 300:
                                    val_str += "..."
                                st.text_area(f"`{key}`", value=val_str, height=80, disabled=True, key=f"preview_{i}_{key}")
                        st.markdown("---")
            else:
                st.warning(t("warning.preview_load_failed"))
        
        # =====================================================================
        # МАППИНГ ПОЛЕЙ
        # =====================================================================
        st.markdown("#### 🔗 Маппинг полей датасета")
        st.caption(t("grpo.field_mapping_desc"))
        
        # Автодетект распространённых полей
        auto_prompt_field = None
        auto_reference_field = None
        
        prompt_candidates = ["question", "prompt", "input", "instruction", "problem", "query", "text"]
        reference_candidates = ["answer", "response", "output", "solution", "target", "completion", "label"]
        
        for candidate in prompt_candidates:
            if candidate in dataset_fields:
                auto_prompt_field = candidate
                break
        
        for candidate in reference_candidates:
            if candidate in dataset_fields:
                auto_reference_field = candidate
                break
        
        # Если в session_state есть сохранённые значения и они валидны - используем их
        saved_prompt_field = st.session_state.grpo_field_mapping.get("prompt_field")
        saved_reference_field = st.session_state.grpo_field_mapping.get("reference_field")
        
        if saved_prompt_field in dataset_fields:
            auto_prompt_field = saved_prompt_field
        if saved_reference_field in dataset_fields:
            auto_reference_field = saved_reference_field
        
        mapping_cols = st.columns(2)
        
        with mapping_cols[0]:
            # Поле промпта
            prompt_options = ["-- не выбрано --"] + dataset_fields
            prompt_idx = 0
            if auto_prompt_field and auto_prompt_field in prompt_options:
                prompt_idx = prompt_options.index(auto_prompt_field)
            
            prompt_field = st.selectbox(
                "📝 Поле промпта (вопрос/задача)",
                options=prompt_options,
                index=prompt_idx,
                key="grpo_prompt_field_select",
                help="Поле с вопросом/задачей для модели"
            )
            if prompt_field and prompt_field != "-- не выбрано --":
                st.session_state.grpo_field_mapping["prompt_field"] = prompt_field
        
        with mapping_cols[1]:
            # Поле референсного ответа
            reference_options = ["-- не выбрано --"] + dataset_fields
            reference_idx = 0
            if auto_reference_field and auto_reference_field in reference_options:
                reference_idx = reference_options.index(auto_reference_field)
            
            reference_field = st.selectbox(
                "✅ Поле ответа (эталон)",
                options=reference_options,
                index=reference_idx,
                key="grpo_reference_field_select",
                help="Поле с правильным ответом для reward функции"
            )
            if reference_field and reference_field != "-- не выбрано --":
                st.session_state.grpo_field_mapping["reference_field"] = reference_field
        
        # Дополнительные поля для metadata (могут использоваться в reward)
        other_fields = [f for f in dataset_fields if f not in [prompt_field, reference_field]]
        if other_fields:
            metadata_fields = st.multiselect(
                "📋 Дополнительные поля (для reward функций)",
                options=other_fields,
                default=st.session_state.grpo_field_mapping.get("metadata_fields", []),
                key="grpo_metadata_fields_select",
                help="Эти поля будут доступны как {{metadata.имя_поля}} в reward правилах"
            )
            st.session_state.grpo_field_mapping["metadata_fields"] = metadata_fields
        
        st.markdown("---")
        
        # =====================================================================
        # ШАБЛОН ПРОМПТА
        # =====================================================================
        st.markdown("#### 📝 Шаблон промпта")
        st.caption(t("grpo.prompt_template_desc"))
        
        # Опция: использовать system_prompt из датасета
        system_prompt_fields = ["-- использовать свой --"]
        if dataset_samples:
            sample = dataset_samples[0] if isinstance(dataset_samples[0], dict) else {}
            # Ищем поля которые могут содержать system prompt
            for key in sample.keys():
                if any(x in key.lower() for x in ["system", "instruction", "context"]):
                    system_prompt_fields.append(key)
        
        use_dataset_system = st.selectbox(
            "System prompt из датасета",
            options=system_prompt_fields,
            help="Если в датасете есть поле с system prompt, выберите его. Иначе используйте свой ниже.",
            key="grpo_system_from_dataset",
        )
        
        # Пресеты шаблонов (Custom по умолчанию!)
        template_presets = {
            "🎯 Custom (свой шаблон)": {
                "system": st.session_state.grpo_system_prompt,
                "template": st.session_state.grpo_prompt_template,
            },
            "🤔 Reasoning (теги <reasoning>/<answer>)": {
                "system": """You are a helpful assistant that solves problems step by step.
Think through the problem carefully inside <reasoning>...</reasoning> tags.
Then provide your final answer inside <answer>...</answer> tags.

Example format:
<reasoning>
Let me analyze this step by step...
Step 1: ...
Step 2: ...
Therefore, the answer is X.
</reasoning>
<answer>X</answer>""",
                "template": "{{prompt}}"
            },
            "🧮 Math RU (русский)": {
                "system": """Ты — полезный ассистент, который решает задачи пошагово.
Размышляй внутри тегов <reasoning>...</reasoning>.
Затем дай финальный ответ внутри тегов <answer>...</answer>.""",
                "template": "{{prompt}}"
            },
            "🤔 DeepSeek R1 стиль": {
                "system": """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags respectively.""",
                "template": "{{prompt}}"
            },
            "📚 QA (вопрос-ответ)": {
                "system": "Answer the question accurately and concisely.",
                "template": "Question: {{prompt}}\n\nAnswer:"
            },
            "🔢 Простой (без system prompt)": {
                "system": "",
                "template": "{{prompt}}"
            },
        }
        
        # По умолчанию Custom
        preset_options = list(template_presets.keys())
        default_idx = 0  # Custom первый
        
        selected_preset = st.selectbox(
            "Пресеты (для быстрой настройки)",
            options=preset_options,
            index=default_idx,
            key="grpo_template_preset",
            help="Выберите пресет чтобы заполнить поля. **Custom** сохраняет ваши настройки.",
        )
        
        preset_data = template_presets[selected_preset]
        
        # System prompt — ВСЕГДА показываем и сохраняем
        # Если выбран пресет (не Custom) и поле пустое — подставляем из пресета
        current_system = st.session_state.grpo_system_prompt
        if "Custom" not in selected_preset and not current_system:
            current_system = preset_data["system"]
        
        system_prompt = st.text_area(
            "System prompt",
            value=current_system,
            height=120,
            key="grpo_system_prompt_input",
            help="Системный промпт для модели. Этот текст будет использован при тренировке!"
        )
        # 🔥 ВАЖНО: Сохраняем в session_state сразу!
        st.session_state.grpo_system_prompt = system_prompt
        
        # Кнопка для загрузки пресета
        if "Custom" not in selected_preset:
            if st.button(f"📥 Загрузить пресет '{selected_preset}'", key="load_preset_btn"):
                st.session_state.grpo_system_prompt = preset_data["system"]
                st.session_state.grpo_prompt_template = preset_data["template"]
                st.rerun()
        
        # Шаблон промпта
        st.markdown("**Шаблон пользовательского промпта:**")
        st.caption("Доступные переменные: `{{prompt}}` (поле промпта), `{{reference}}` (поле ответа), `{{metadata.имя}}` (доп. поля)")
        
        current_template = st.session_state.grpo_prompt_template
        prompt_template = st.text_area(
            "Шаблон",
            value=current_template,
            height=80,
            key="grpo_prompt_template_input",
            label_visibility="collapsed",
        )
        # 🔥 ВАЖНО: Сохраняем в session_state сразу!
        st.session_state.grpo_prompt_template = prompt_template
        
        # Превью готового промпта
        if dataset_samples and prompt_field and prompt_field != "-- не выбрано --":
            st.markdown("**Превью готового промпта:**")
            sample = dataset_samples[0]
            
            # Подставляем значения
            preview_prompt = prompt_template
            if isinstance(sample, dict):
                # {{prompt}}
                if prompt_field in sample:
                    preview_prompt = preview_prompt.replace("{{prompt}}", str(sample[prompt_field]))
                # {{reference}}
                if reference_field and reference_field != "-- не выбрано --" and reference_field in sample:
                    preview_prompt = preview_prompt.replace("{{reference}}", str(sample[reference_field]))
                # {{metadata.xxx}}
                for key, value in sample.items():
                    preview_prompt = preview_prompt.replace(f"{{{{metadata.{key}}}}}", str(value))
            
            # Показываем превью
            preview_full = ""
            if system_prompt:
                preview_full = f"[System]: {system_prompt}\n\n[User]: {preview_prompt}"
            else:
                preview_full = f"[User]: {preview_prompt}"
            
            st.code(preview_full, language=None)
            
            # Показываем референсный ответ
            if reference_field and reference_field != "-- не выбрано --" and reference_field in sample:
                ref_val = str(sample[reference_field])[:500]
                if len(str(sample[reference_field])) > 500:
                    ref_val += "..."
                st.caption(f"**Эталонный ответ:** {ref_val}")
        
        st.markdown("---")
        
    else:
        # Очищаем session_state если датасет не выбран
        if "grpo_dataset_path" in st.session_state:
            del st.session_state.grpo_dataset_path
        st.warning(f"⚠️ {t('warning.select_dataset')}")
        prompt_field = None
        reference_field = None
    
    # Настройки датасета
    grpo_max_samples = st.number_input(
        "Макс. примеров (0 = все)",
        min_value=0,
        max_value=50000,
        value=0,
        step=100,
        help="Ограничить количество примеров для обучения"
    )

    # Оценка размера датасета для лимита "Max prompts" в sidebar
    effective_size = None
    try:
        if grpo_max_samples and int(grpo_max_samples) > 0:
            effective_size = int(grpo_max_samples)
        elif grpo_dataset_path:
            p = Path(grpo_dataset_path)
            if p.exists() and p.suffix == ".jsonl":
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    effective_size = sum(1 for _ in f if _.strip())
            elif p.exists() and p.suffix == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    obj = json_lib.load(f)
                if isinstance(obj, list):
                    effective_size = len(obj)
        if isinstance(effective_size, int) and effective_size > 0:
            st.session_state["grpo_effective_dataset_size"] = int(effective_size)
            st.caption(f"📊 Размер датасета: **{effective_size} примеров**")
    except Exception:
        pass
    
    # =========================================================================
    # 2. REWARD DESIGNER
    # =========================================================================
    st.markdown(f"### 🎯 {t('grpo.reward_designer')}")
    st.caption(t("grpo.reward_designer_desc"))
    
    # =========================================================================
    # Песочница — Тестовые данные (автозагрузка из датасета)
    # =========================================================================
    st.markdown(f"#### 🧪 {t('grpo.sandbox')}")
    
    # Получаем данные из датасета для песочницы
    # НЕ используем fallback значения - показываем реальные данные из датасета
    default_prompt = ""
    default_reference = ""
    reference_is_empty = False
    
    # Если датасет загружен - берём данные оттуда
    if dataset_samples and prompt_field and prompt_field != "-- не выбрано --":
        sample = dataset_samples[0]
        if isinstance(sample, dict) and prompt_field in sample:
            default_prompt = str(sample[prompt_field]) or ""
        if isinstance(sample, dict) and reference_field and reference_field != "-- не выбрано --" and reference_field in sample:
            ref_val = sample[reference_field]
            # Для GSM8K-стиля извлекаем число после ####
            if isinstance(ref_val, str) and "####" in ref_val:
                parts = ref_val.split("####")
                if len(parts) > 1:
                    default_reference = parts[-1].strip().replace(",", "").split()[0] if parts[-1].strip() else ""
                else:
                    default_reference = str(ref_val) if ref_val else ""
            else:
                default_reference = str(ref_val) if ref_val else ""
            
            # Проверяем пустой ли ответ
            reference_is_empty = not default_reference.strip()
    
    with st.expander(f"📝 {t('grpo.test_data')}", expanded=True):
        # Если датасет загружен, показываем селектор примера
        selected_sample_idx = 0
        if dataset_samples:
            sample_options = [f"Пример {i+1}" for i in range(min(10, len(dataset_samples)))]
            selected_sample_idx = st.selectbox(
                "Выберите пример из датасета",
                options=range(len(sample_options)),
                format_func=lambda x: sample_options[x],
                key="grpo_sandbox_sample_idx",
            )
            
            # Обновляем default_prompt/reference на основе выбранного примера
            if selected_sample_idx < len(dataset_samples):
                sample = dataset_samples[selected_sample_idx]
                if isinstance(sample, dict) and prompt_field and prompt_field in sample:
                    default_prompt = str(sample[prompt_field]) or ""
                if isinstance(sample, dict) and reference_field and reference_field != "-- не выбрано --" and reference_field in sample:
                    ref_val = sample[reference_field]
                    if isinstance(ref_val, str) and "####" in ref_val:
                        parts = ref_val.split("####")
                        if len(parts) > 1:
                            default_reference = parts[-1].strip().replace(",", "").split()[0] if parts[-1].strip() else ""
                        else:
                            default_reference = str(ref_val) if ref_val else ""
                    else:
                        default_reference = str(ref_val) if ref_val else ""
                    
                    reference_is_empty = not default_reference.strip()
            
            st.caption(t("grpo.data_loaded_from_dataset"))
        else:
            st.info(t("info.select_dataset_above"))
        
        # Создаём уникальный key на основе датасета и выбранного примера
        dataset_key_hash = hash(grpo_dataset_path or "none") % 10000
        
        sample_prompt = st.text_area(
            "**Промпт** (вопрос/задача)",
            value=default_prompt,
            height=100,
            key=f"sample_prompt_{dataset_key_hash}_{selected_sample_idx}",
        )
        
        # Показываем предупреждение если ответ пустой
        if reference_is_empty:
            st.warning(f"⚠️ {t('warning.field_empty')}")
        
        sample_reference = st.text_input(
            "**Эталонный ответ**",
            value=default_reference,
            key=f"sample_reference_{dataset_key_hash}_{selected_sample_idx}",
            help="Если пусто - значит в выбранном поле датасета нет ответа"
        )
        
        # Показываем дополнительные поля из датасета
        if dataset_samples and selected_sample_idx < len(dataset_samples):
            sample = dataset_samples[selected_sample_idx]
            metadata_fields_list = st.session_state.grpo_field_mapping.get("metadata_fields", [])
            if metadata_fields_list and isinstance(sample, dict):
                st.markdown("**Дополнительные поля:**")
                for mf in metadata_fields_list:
                    if mf in sample:
                        st.caption(f"`{{{{metadata.{mf}}}}}` = {str(sample[mf])[:100]}")
    
    st.markdown("---")
    
    # =========================================================================
    # Универсальный конструктор правил
    # =========================================================================
    st.markdown(f"#### 🏗️ {t('grpo.rule_constructor')}")
    
    # Справка по переменным
    with st.expander(f"📖 {t('grpo.help_vars')}", expanded=False):
        st.markdown("""
**Доступные переменные:**
- `{{response}}` — ответ модели (completion)
- `{{reference}}` — эталонный ответ (из поля ответа датасета)
- `{{prompt}}` — отформатированный промпт (после применения шаблона)
- `{{extracted.имя}}` — значение, извлечённое regex-группой
- `{{metadata.имя_поля}}` — значение из дополнительного поля датасета

**Доступ к данным датасета:**
Все поля из вашего датасета автоматически доступны через `{{metadata.имя_поля}}`.
Например, если датасет содержит поле `difficulty`, можно использовать `{{metadata.difficulty}}`.

**Операторы сравнения:**
- `contains` — содержит подстроку
- `not_contains` — не содержит
- `matches` — соответствует regex
- `not_matches` — не соответствует regex
- `equals` — точное совпадение
- `==`, `!=`, `>`, `<`, `>=`, `<=` — для чисел

**Пример regex с группами:**
```
<answer>(?P<model_answer>\\d+)</answer>
```
Извлечёт число в переменную `{{extracted.model_answer}}`

**Формула reward:**
```
1.0 if {{extracted.model_answer}} == {{reference}} else 0.0
```

**Пример с metadata:**
```
weight = float({{metadata.difficulty}}) / 10.0
```
        """)
    
    # Инициализация правил
    if "reward_rules" not in st.session_state:
        st.session_state.reward_rules = [
            {
                "id": 0,
                "name": "Проверка формата",
                "enabled": True,
                "weight": 1.0,
                "conditions": [
                    {"type": "contains", "target": "{{response}}", "value": "<reasoning>"},
                    {"type": "contains", "target": "{{response}}", "value": "</reasoning>"},
                    {"type": "contains", "target": "{{response}}", "value": "<answer>"},
                    {"type": "contains", "target": "{{response}}", "value": "</answer>"},
                ],
                "condition_logic": "all",  # all / any / custom
                "reward_formula": "1.0",
                "else_reward": "0.0",
            },
            {
                "id": 1,
                "name": "Правильный ответ",
                "enabled": True,
                "weight": 2.0,
                "extractors": [
                    {"name": "model_answer", "pattern": r"<answer>\s*(\d+)\s*</answer>", "source": "{{response}}"},
                ],
                "conditions": [
                    {"type": "equals_numeric", "left": "{{extracted.model_answer}}", "right": "{{reference}}", "tolerance": 0.01},
                ],
                "condition_logic": "all",
                "reward_formula": "1.0",
                "else_reward": "0.0",
            },
            {
                "id": 2,
                "name": "Качество reasoning",
                "enabled": True,
                "weight": 0.5,
                "extractors": [
                    {"name": "reasoning_text", "pattern": r"<reasoning>(.*?)</reasoning>", "source": "{{response}}", "flags": "DOTALL"},
                ],
                "conditions": [
                    {"type": "length_between", "target": "{{extracted.reasoning_text}}", "min": 50, "max": 2000},
                ],
                "condition_logic": "all",
                "reward_formula": "min(len({{extracted.reasoning_text}}) / 200.0, 1.0)",
                "else_reward": "0.0",
            },
        ]
        st.session_state.next_rule_id = 3
    
    # Функция для рендеринга одного правила
    def render_rule(rule, idx):
        with st.expander(
            f"{'✅' if rule['enabled'] else '⏸️'} **{rule['name']}** (вес: {rule['weight']})",
            expanded=False
        ):
            # Заголовок правила
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            with c1:
                rule["name"] = st.text_input(t("common.name"), value=rule["name"], key=f"rule_name_{rule['id']}")
            with c2:
                rule["weight"] = st.number_input(t("common.weight"), 0.0, 10.0, float(rule["weight"]), 0.1, key=f"rule_weight_{rule['id']}")
            with c3:
                rule["enabled"] = st.checkbox("Вкл", value=rule["enabled"], key=f"rule_enabled_{rule['id']}")
            with c4:
                if st.button("🗑️", key=f"rule_del_{rule['id']}"):
                    st.session_state.reward_rules.pop(idx)
                    st.rerun()
            
            # === EXTRACTORS (regex для извлечения значений) ===
            st.markdown("##### 🔍 Экстракторы (regex)")
            st.caption(t("grpo.extractors_desc"))
            
            if "extractors" not in rule:
                rule["extractors"] = []
            
            for ei, ext in enumerate(rule["extractors"]):
                ec1, ec2, ec3, ec4 = st.columns([2, 4, 2, 1])
                with ec1:
                    ext["name"] = st.text_input(t("common.name"), value=ext.get("name", f"var{ei}"), key=f"ext_name_{rule['id']}_{ei}")
                with ec2:
                    ext["pattern"] = st.text_input("Regex", value=ext.get("pattern", ""), key=f"ext_pattern_{rule['id']}_{ei}")
                with ec3:
                    ext["source"] = st.selectbox(
                        "Источник", 
                        ["{{response}}", "{{reference}}", "{{prompt}}"],
                        index=["{{response}}", "{{reference}}", "{{prompt}}"].index(ext.get("source", "{{response}}")),
                        key=f"ext_source_{rule['id']}_{ei}"
                    )
                with ec4:
                    if st.button("✖", key=f"ext_del_{rule['id']}_{ei}"):
                        rule["extractors"].pop(ei)
                        st.rerun()
            
            if st.button(f"➕ {t('button.add_extractor')}", key=f"add_ext_{rule['id']}"):
                rule["extractors"].append({"name": f"var{len(rule['extractors'])}", "pattern": r"(.*)", "source": "{{response}}"})
                st.rerun()
            
            st.markdown("---")
            
            # === CONDITIONS ===
            st.markdown("##### ⚡ Условия")
            
            condition_types = {
                "contains": "содержит",
                "not_contains": "не содержит",
                "matches": "соответствует regex",
                "not_matches": "не соответствует regex",
                "equals": "равно (строка)",
                "equals_numeric": "равно (число)",
                "greater": "> больше",
                "less": "< меньше",
                "length_between": "длина в диапазоне",
                "length_min": "длина >= мин",
                "length_max": "длина <= макс",
            }
            
            if "conditions" not in rule:
                rule["conditions"] = []
            
            for ci, cond in enumerate(rule["conditions"]):
                ctype = cond.get("type", "contains")
                
                cc1, cc2 = st.columns([1, 4])
                with cc1:
                    if ci > 0:
                        st.write("**AND**" if rule.get("condition_logic") == "all" else "**OR**")
                    else:
                        st.write("**IF**")
                
                with cc2:
                    ccc1, ccc2, ccc3, ccc4 = st.columns([3, 2, 3, 1])
                    
                    with ccc1:
                        # Левый операнд
                        target_options = ["{{response}}", "{{reference}}", "{{prompt}}"]
                        # Добавляем извлечённые переменные
                        for ext in rule.get("extractors", []):
                            target_options.append(f"{{{{extracted.{ext['name']}}}}}")
                        
                        left_val = cond.get("target") or cond.get("left", "{{response}}")
                        if left_val not in target_options:
                            target_options.append(left_val)
                        
                        new_left = st.selectbox(
                            "Что проверять",
                            target_options,
                            index=target_options.index(left_val) if left_val in target_options else 0,
                            key=f"cond_left_{rule['id']}_{ci}",
                            label_visibility="collapsed"
                        )
                        cond["target"] = new_left
                        cond["left"] = new_left
                    
                    with ccc2:
                        new_type = st.selectbox(
                            "Оператор",
                            list(condition_types.keys()),
                            format_func=lambda x: condition_types.get(x, x),
                            index=list(condition_types.keys()).index(ctype) if ctype in condition_types else 0,
                            key=f"cond_type_{rule['id']}_{ci}",
                            label_visibility="collapsed"
                        )
                        cond["type"] = new_type
                    
                    with ccc3:
                        # Правый операнд зависит от типа
                        if new_type in ["contains", "not_contains", "equals"]:
                            cond["value"] = st.text_input(t("common.value"), value=cond.get("value", ""), key=f"cond_val_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type in ["matches", "not_matches"]:
                            cond["pattern"] = st.text_input("Regex", value=cond.get("pattern", ""), key=f"cond_pat_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type == "equals_numeric":
                            right_opts = ["{{reference}}"] + [f"{{{{extracted.{e['name']}}}}}" for e in rule.get("extractors", [])]
                            right_val = cond.get("right", "{{reference}}")
                            if right_val not in right_opts:
                                right_opts.append(right_val)
                            cond["right"] = st.selectbox(t("common.compare_with"), right_opts, index=right_opts.index(right_val) if right_val in right_opts else 0, key=f"cond_right_{rule['id']}_{ci}", label_visibility="collapsed")
                            cond["tolerance"] = st.number_input("±", 0.0, 100.0, float(cond.get("tolerance", 0.01)), 0.01, key=f"cond_tol_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type in ["greater", "less"]:
                            cond["value"] = st.number_input(t("common.number"), value=float(cond.get("value", 0)), key=f"cond_num_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type == "length_between":
                            lc1, lc2 = st.columns(2)
                            cond["min"] = lc1.number_input("Мин", 0, 100000, int(cond.get("min", 10)), key=f"cond_min_{rule['id']}_{ci}")
                            cond["max"] = lc2.number_input("Макс", 0, 100000, int(cond.get("max", 5000)), key=f"cond_max_{rule['id']}_{ci}")
                        elif new_type == "length_min":
                            cond["min"] = st.number_input(t("common.min_length"), 0, 100000, int(cond.get("min", 10)), key=f"cond_minl_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type == "length_max":
                            cond["max"] = st.number_input(t("common.max_length"), 0, 100000, int(cond.get("max", 5000)), key=f"cond_maxl_{rule['id']}_{ci}", label_visibility="collapsed")
                    
                    with ccc4:
                        if st.button("✖", key=f"cond_del_{rule['id']}_{ci}"):
                            rule["conditions"].pop(ci)
                            st.rerun()
            
            # Логика объединения условий
            lc1, lc2 = st.columns([2, 3])
            with lc1:
                rule["condition_logic"] = st.radio(
                    "Логика",
                    ["all", "any"],
                    format_func=lambda x: "ВСЕ условия (AND)" if x == "all" else "ЛЮБОЕ условие (OR)",
                    index=0 if rule.get("condition_logic", "all") == "all" else 1,
                    key=f"cond_logic_{rule['id']}",
                    horizontal=True
                )
            with lc2:
                if st.button(f"➕ {t('button.add_condition')}", key=f"add_cond_{rule['id']}"):
                    rule["conditions"].append({"type": "contains", "target": "{{response}}", "value": ""})
                    st.rerun()
            
            st.markdown("---")
            
            # === REWARD FORMULA ===
            st.markdown("##### 🎯 Формула Reward")
            
            rc1, rc2 = st.columns(2)
            with rc1:
                rule["reward_formula"] = st.text_input(
                    "Если условия TRUE",
                    value=rule.get("reward_formula", "1.0"),
                    key=f"reward_form_{rule['id']}",
                    help="Можно использовать переменные и Python-выражения: `min(len({{extracted.text}}) / 100, 1.0)`"
                )
            with rc2:
                rule["else_reward"] = st.text_input(
                    "Если условия FALSE",
                    value=rule.get("else_reward", "0.0"),
                    key=f"else_form_{rule['id']}"
                )
    
    # Рендерим все правила
    for idx, rule in enumerate(st.session_state.reward_rules):
        render_rule(rule, idx)
    
    # Кнопки добавления
    st.markdown("---")
    
    col_add1, col_add2, col_add3 = st.columns(3)
    
    with col_add1:
        if st.button(f"➕ {t('button.empty_rule')}", type="secondary"):
            new_id = st.session_state.next_rule_id
            st.session_state.next_rule_id += 1
            st.session_state.reward_rules.append({
                "id": new_id,
                "name": f"Правило {new_id + 1}",
                "enabled": True,
                "weight": 1.0,
                "extractors": [],
                "conditions": [],
                "condition_logic": "all",
                "reward_formula": "1.0",
                "else_reward": "0.0",
            })
            st.rerun()
    
    with col_add2:
        preset_rules = st.selectbox(
            "Добавить шаблон",
            [
                "-- выберите --",
                "🔍 Regex + сравнение",
                "🏷️ Проверка тегов",
                "📏 Проверка длины",
                "🔤 Ключевые слова",
                "🔄 Штраф за повторы",
                "🐍 Python формула",
            ],
            key="preset_select"
        )
    
    with col_add3:
        if st.button(f"➕ {t('button.add_template')}", type="primary"):
            new_id = st.session_state.next_rule_id
            st.session_state.next_rule_id += 1
            
            if preset_rules == "🔍 Regex + сравнение":
                st.session_state.reward_rules.append({
                    "id": new_id, "name": "Regex извлечение", "enabled": True, "weight": 1.0,
                    "extractors": [{"name": "answer", "pattern": r"<answer>\s*(\d+)\s*</answer>", "source": "{{response}}"}],
                    "conditions": [{"type": "equals_numeric", "left": "{{extracted.answer}}", "right": "{{reference}}", "tolerance": 0.01}],
                    "condition_logic": "all", "reward_formula": "1.0", "else_reward": "0.0",
                })
            elif preset_rules == "🏷️ Проверка тегов":
                st.session_state.reward_rules.append({
                    "id": new_id, "name": "Формат тегов", "enabled": True, "weight": 1.0,
                    "extractors": [],
                    "conditions": [
                        {"type": "contains", "target": "{{response}}", "value": "<reasoning>"},
                        {"type": "contains", "target": "{{response}}", "value": "</reasoning>"},
                        {"type": "contains", "target": "{{response}}", "value": "<answer>"},
                        {"type": "contains", "target": "{{response}}", "value": "</answer>"},
                    ],
                    "condition_logic": "all", "reward_formula": "1.0", "else_reward": "0.0",
                })
            elif preset_rules == "📏 Проверка длины":
                st.session_state.reward_rules.append({
                    "id": new_id, "name": "Длина ответа", "enabled": True, "weight": 0.5,
                    "extractors": [],
                    "conditions": [{"type": "length_between", "target": "{{response}}", "min": 100, "max": 3000}],
                    "condition_logic": "all", "reward_formula": "1.0", "else_reward": "-0.5",
                })
            elif preset_rules == "🔤 Ключевые слова":
                st.session_state.reward_rules.append({
                    "id": new_id, "name": "Ключевые слова", "enabled": True, "weight": 0.3,
                    "extractors": [],
                    "conditions": [
                        {"type": "contains", "target": "{{response}}", "value": "therefore"},
                    ],
                    "condition_logic": "any", "reward_formula": "0.5", "else_reward": "0.0",
                })
            elif preset_rules == "🔄 Штраф за повторы":
                st.session_state.reward_rules.append({
                    "id": new_id, "name": "Без повторов", "enabled": True, "weight": 0.5,
                    "extractors": [],
                    "conditions": [{"type": "not_matches", "target": "{{response}}", "pattern": r"(.{20,})\1"}],
                    "condition_logic": "all", "reward_formula": "0.5", "else_reward": "-0.5",
                })
            elif preset_rules == "🐍 Python формула":
                st.session_state.reward_rules.append({
                    "id": new_id, "name": "Python формула", "enabled": True, "weight": 1.0,
                    "extractors": [{"name": "reasoning", "pattern": r"<reasoning>(.*?)</reasoning>", "source": "{{response}}", "flags": "DOTALL"}],
                    "conditions": [],
                    "condition_logic": "all", 
                    "reward_formula": "min(len({{extracted.reasoning}}) / 500.0, 1.0) if {{extracted.reasoning}} else 0.0",
                    "else_reward": "0.0",
                })
            else:
                st.warning(t("warning.select_template"))
            st.rerun()
    
    # =========================================================================
    # Формат reasoning
    # =========================================================================
    st.markdown("---")
    st.markdown(f"#### 📝 {t('grpo.reasoning_format')}")
    
    reasoning_format = st.selectbox(
        "Формат тегов",
        ["deepseek", "simple", "russian", "gsm8k"],
        format_func=lambda x: {
            "deepseek": "DeepSeek (<think>...</think>, <answer>...</answer>)",
            "simple": "Simple (<reasoning>...</reasoning>, <answer>...</answer>)",
            "russian": "Russian (на русском языке)",
            "gsm8k": "GSM8K (#### <number>)",
        }[x],
    )
    
    # Превью формата
    format_examples = {
        "deepseek": """<think>
Дано: ... 
Нужно найти: ...
Решение: ...
</think>
<answer>
42
</answer>""",
        "simple": """<reasoning>
Шаг 1: ...
Шаг 2: ...
</reasoning>
<answer>
42
</answer>""",
        "russian": """<reasoning>
Рассуждение: ...
Вычисления: ...
</reasoning>
<answer>
42
</answer>""",
        "gsm8k": """Let me solve this step by step.
Step 1: First, I need to...
Step 2: Then, I calculate...
Therefore, the answer is 42.
#### 42""",
    }
    
    with st.expander(f"📋 {t('grpo.response_format_example')}"):
        st.code(format_examples[reasoning_format], language=None)
    
    # Собираем конфигурацию reward правил (новый универсальный формат)
    reward_rules = [
        {
            "name": rule["name"],
            "weight": rule["weight"],
            "enabled": rule.get("enabled", True),
            "extractors": rule.get("extractors", []),
            "conditions": rule.get("conditions", []),
            "condition_logic": rule.get("condition_logic", "all"),
            "reward_formula": rule.get("reward_formula", "1.0"),
            "else_reward": rule.get("else_reward", "0.0"),
        }
        for rule in st.session_state.get("reward_rules", [])
        if rule.get("enabled", True)
    ]
    
    # Получаем настройки маппинга полей из session_state
    field_mapping = st.session_state.get("grpo_field_mapping", {
        "prompt_field": "question",
        "reference_field": "answer",
        "metadata_fields": [],
    })
    prompt_template_value = st.session_state.get("grpo_prompt_template", "{{prompt}}")
    system_prompt_value = st.session_state.get("grpo_system_prompt", "")
    
    return {
        "grpo_dataset_source": dataset_source,
        "grpo_dataset_key": dataset_key,
        "grpo_dataset_path": grpo_dataset_path,
        "grpo_dataset_language": grpo_dataset_language,
        "grpo_max_samples": grpo_max_samples if grpo_max_samples > 0 else None,
        "grpo_reward_rules": reward_rules,
        "grpo_reasoning_format": reasoning_format,
        "grpo_system_prompt": system_prompt_value,  # System prompt из настроек датасета
        # Настройки маппинга датасета
        "grpo_field_mapping": field_mapping,
        "grpo_prompt_template": prompt_template_value,
    }


def render_model_config():
    """Model configuration in sidebar."""
    st.sidebar.header(f"🧠 {t('sidebar.architecture')}")
    
    # Training mode - use i18n keys
    def get_stage_label(stage):
        return {
            "pretrain": t("stage.pretrain"),
            "continual_pretrain": t("stage.continual_pretrain"),
            "sft": t("stage.sft"),
            "grpo": f"🧠 {t('stage.grpo')}"
        }.get(stage, stage)
    
    selected_stage = st.sidebar.selectbox(
        t("sidebar.training_stage"),
        options=["pretrain", "continual_pretrain", "sft", "grpo"],
        format_func=get_stage_label,
        help=t("help.training_stage")
    )

    # === Training Backend (только для GRPO) ===
    # Ставим наверх, чтобы остальные настройки строились от выбранного backend.
    training_backend = "models-at-home"
    if selected_stage == "grpo":
        st.sidebar.subheader("⚙️ Training Backend")

        backend_options = ["🏠 models-at-home", "🦥 Unsloth"]
        selected_backend_display = st.sidebar.radio(
            "Backend",
            backend_options,
            index=0,  # По умолчанию models-at-home
            help=(
                "**🏠 models-at-home** (рекомендуется):\n"
                "• Multi-GPU поддержка (DDP)\n"
                "• Full fine-tuning + LoRA/QLoRA\n"
                "• FlashAttention + Liger Kernels\n\n"
                "**🦥 Unsloth** (быстрее на 1 GPU):\n"
                "• ⚡ 2x быстрее на **одной GPU**\n"
                "• 💾 До 70% меньше VRAM\n"
                "• ⚠️ **Только LoRA/QLoRA** (не full)\n"
                "• ⚠️ Multi-GPU: работает, но без fast_inference"
            ),
            key="training_backend_radio"
        )
        training_backend = "unsloth" if "Unsloth" in selected_backend_display else "models-at-home"

        # Показываем информацию о backend
        if training_backend == "unsloth":
            try:
                import unsloth
                st.sidebar.success("🦥 **Unsloth**: 2x быстрее, 70% меньше VRAM")
                st.sidebar.caption("⚠️ Только LoRA/QLoRA • 1 GPU для fast_inference")
            except ImportError:
                st.sidebar.error("🦥 **Unsloth не установлен!**")
        else:
            st.sidebar.info("🏠 **models-at-home**: Multi-GPU, Full/LoRA/QLoRA")

        st.sidebar.markdown("---")
    
    # Имя модели (для папки эксперимента)
    if selected_stage == "pretrain":
        model_name_default = "home_pretrain"
    elif selected_stage == "continual_pretrain":
        model_name_default = "home_continual_pretrain"
    elif selected_stage == "grpo":
        model_name_default = "home_grpo"
    else:
        model_name_default = "home_sft"
    model_name = st.sidebar.text_input(t("sidebar.experiment_name"), value=model_name_default, help=t("help.experiment_folder"))
    
    base_model_path = None
    
    if selected_stage in ("sft", "continual_pretrain", "grpo"):
        stage_label = {"sft": "SFT", "continual_pretrain": "Continual Pretraining", "grpo": "GRPO"}.get(selected_stage, selected_stage)
        st.sidebar.subheader(f"📦 {t('sidebar.base_model')}")
        available = get_available_models()
        
        if selected_stage == "continual_pretrain":
            # Для continual_pretrain фильтруем: предпочитаем final/export модели и HF модели
            # Checkpoint'ы тоже разрешены (для resume), но с предупреждением
            hf_models = [m for m in available if m["type"] == "hf"]
            final_models = [m for m in available if m["type"] == "final"]
            checkpoint_models = [m for m in available if m["type"] == "checkpoint"]
            
            if hf_models or final_models:
                st.sidebar.info("💡 Рекомендуется использовать 🤗 HF модель или final_model для continual pretraining")
                available_filtered = hf_models + final_models + checkpoint_models
            else:
                if checkpoint_models:
                    st.sidebar.warning(
                        "⚠️ Доступны только checkpoint'ы. Для resume это нормально, "
                        "но для начала continual pretraining лучше использовать final_model."
                    )
                available_filtered = checkpoint_models if checkpoint_models else available
        else:
            # Для SFT показываем все модели (HF модели первыми)
            hf_models = [m for m in available if m["type"] == "hf"]
            other_models = [m for m in available if m["type"] != "hf"]
            available_filtered = hf_models + other_models
        
        if not available_filtered:
            st.sidebar.warning(f"Нет доступных моделей для {stage_label}. Скачайте модель на вкладке 🤖 Модели или обучите Pretrain!")
            # Можно дать возможность ввести путь вручную
            base_model_path = st.sidebar.text_input("Путь к модели вручную", placeholder="/path/to/model")
        else:
            # Создаем список опций с пометками типов
            def get_model_label(m):
                if m["type"] == "hf":
                    return m["name"]  # Уже содержит 🤗
                elif m["type"] == "final":
                    return f"{m['name']} (✅ final)"
                else:
                    return f"{m['name']} (⚠️ checkpoint)"
            
            model_options = [get_model_label(m) for m in available_filtered]
            
            selected_base_name = st.sidebar.selectbox(
                t("model.select"), 
                options=model_options,
                help="🤗 — модели с HuggingFace, ✅ final — обученные модели, ⚠️ checkpoint — для resume"
            )
            
            # Находим модель по индексу (model_options и available_filtered соответствуют друг другу)
            selected_idx = model_options.index(selected_base_name)
            selected_model = available_filtered[selected_idx]
            base_model_path = selected_model["path"]
            
            # Показываем предупреждение для checkpoint в continual_pretrain
            if selected_stage == "continual_pretrain" and selected_model["type"] == "checkpoint":
                st.sidebar.info(
                    "ℹ️ Выбран checkpoint. Будет выполнен resume (восстановление оптимизатора и scheduler). "
                    "Для начала нового continual pretraining лучше использовать final_model или 🤗 HF модель."
                )
            elif selected_model["type"] == "hf":
                st.sidebar.success(f"✅ {t('info.hf_model')}")
            
            st.sidebar.caption(f"Путь: `{base_model_path}`")
    
    # Флаг, что параметры загружены из конфига
    loaded_config = None
    
    if selected_stage in ("sft", "continual_pretrain", "grpo") and base_model_path:
        # Пытаемся загрузить конфиг
        # ВАЖНО: различаем run_config.json (training params) и config.json (model params)
        try:
            base_path = Path(base_model_path)
            cfg_path = None
            cfg_type = None  # "run" или "model"
            
            # 1. Сначала ищем run_config.json (полный training config)
            # Для чекпоинтов: checkpoint_step6800 -> parent/run_config.json
            if (base_path.parent / "run_config.json").exists():
                cfg_path = base_path.parent / "run_config.json"
                cfg_type = "run"
            elif (base_path / "run_config.json").exists():
                cfg_path = base_path / "run_config.json"
                cfg_type = "run"
            # 2. Если run_config нет — используем config.json (только model params)
            elif (base_path / "config.json").exists():
                cfg_path = base_path / "config.json"
                cfg_type = "model"
            
            if cfg_path and cfg_path.exists():
                with open(cfg_path) as f:
                    loaded_config = json.load(f)
                if cfg_type == "run":
                    st.sidebar.success(f"✅ {t('info.params_loaded')} run_config.json")
                else:
                    # config.json содержит только параметры модели, нужно адаптировать ключи
                    # transformers config -> наш формат
                    if "num_hidden_layers" in loaded_config and "num_layers" not in loaded_config:
                        loaded_config["num_layers"] = loaded_config["num_hidden_layers"]
                    if "num_attention_heads" in loaded_config and "n_heads" not in loaded_config:
                        loaded_config["n_heads"] = loaded_config["num_attention_heads"]
                    if "max_position_embeddings" in loaded_config and "seq_len" not in loaded_config:
                        loaded_config["seq_len"] = loaded_config["max_position_embeddings"]
                    st.sidebar.info("ℹ️ Параметры модели загружены из config.json (training params не найдены)")
            else:
                st.sidebar.warning(f"⚠️ {t('warning.config_not_found')}")
        except Exception as e:
             st.sidebar.error(f"Ошибка чтения config: {e}")

    st.sidebar.subheader(f"⚙️ {t('sidebar.model_params')}")

    blueprint_path = ""
    model_type = "HomeModel (GPT-2 style)"  # default
    # Базовые значения для переопределения
    default_h, default_l, default_n, default_seq = 512, 8, 8, 2048

    # Выбор архитектуры ТОЛЬКО для pretrain (с нуля)
    # Для SFT/Continual Pretrain архитектура определяется базовой моделью
    if not loaded_config:
        # === ИНТЕГРАЦИЯ С VISUAL MODEL BUILDER ===
        blueprints_dir = PROJECT_ROOT / "blueprints"
        blueprints_dir.mkdir(exist_ok=True)
        blueprints = list(blueprints_dir.glob("*.json"))
            
        arch_options = [
            "HomeModel (LLaMA-style)",  # RMSNorm, RoPE, SwiGLU
            "GPT-2 (Classic)",          # LayerNorm, Learned Pos, GELU
            "HomeModel MoE",            # MoE с SwiGLU экспертами
            "Custom Blueprint (Visual Builder)"
        ]
        
        model_type = st.sidebar.selectbox(
            "Архитектура модели",
            options=arch_options,
            index=0,
            help="HomeModel (LLaMA-style): RMSNorm, RoPE, SwiGLU. GPT-2: LayerNorm, Learned Pos, GELU. MoE: Mixture of Experts."
        )
        
        # MoE параметры (показываем только при выборе MoE)
        num_experts = 8
        num_experts_per_tok = 2
        expert_type = "swiglu"
        
        if model_type == "HomeModel MoE":
            st.sidebar.markdown("**🔀 MoE параметры**")
            num_experts = st.sidebar.slider(
                "Num Experts",
                min_value=2,
                max_value=32,
                value=8,
                help="Количество экспертов в MoE слое"
            )
            num_experts_per_tok = st.sidebar.slider(
                "Top-K (активных экспертов)",
                min_value=1,
                max_value=min(8, num_experts),
                value=min(2, num_experts),
                help="Сколько экспертов активируется для каждого токена"
            )
            expert_type = st.sidebar.selectbox(
                "Expert Type",
                ["swiglu", "mlp"],
                index=0,
                help="SwiGLU: больше параметров, лучше качество. MLP: меньше параметров."
            )

    if not loaded_config and model_type == "Custom Blueprint (Visual Builder)":
        blueprints_dir = PROJECT_ROOT / "blueprints"
        blueprints = list(blueprints_dir.glob("*.json"))
        # Логика для Blueprint проектов
        if blueprints:
            bp_names = [b.name for b in blueprints]
            selected_bp = st.sidebar.selectbox("Проект (Visual Builder)", bp_names)
            blueprint_path = str(blueprints_dir / selected_bp)
        else:
            st.sidebar.warning("Blueprint проектов не найдено в папке `blueprints/`. Сохраните проект в Visual Model Builder или укажите путь вручную.")
            blueprint_path = st.sidebar.text_input("Путь к blueprint.json", value=str(blueprints_dir / "model.json"))
        
        # Загружаем инфо из blueprint
        try:
            with open(blueprint_path) as f:
                bp_data = json.load(f)
            st.sidebar.info(f"Blocks: {len(bp_data.get('blocks', []))} | Hidden: {bp_data.get('hidden_size')} | Vocab: {bp_data.get('vocab_size')}")
            
            # Обновляем дефолты для слайдеров ниже (чтобы они визуально соответствовали)
            default_h = bp_data.get("hidden_size", 512)
            default_seq = bp_data.get("max_position_embeddings", 2048)
            # Сложно посчитать слои из Repeater, но попробуем грубо
            default_l = len(bp_data.get("blocks", []))
            
            # Токенизатор
            st.sidebar.markdown("**Токенизатор**")
            tokenizer_mode = st.sidebar.radio("Tokenizer Source", ["Standard (GPT-2)", "HF Repo", "Local Path"], horizontal=True)
            if tokenizer_mode == "Standard (GPT-2)":
                tokenizer_path = "gpt2"
            elif tokenizer_mode == "HF Repo":
                tokenizer_path = st.sidebar.text_input("HF ID", "meta-llama/Llama-2-7b-hf")
            else:
                tokenizer_path = st.sidebar.text_input("Local Path", str(PROJECT_ROOT / "tokenizers/my_tok"))
                
        except Exception as e:
            st.sidebar.error(f"Invalid Blueprint: {e}")
            tokenizer_path = "gpt2"
    elif not loaded_config:
        # Стандартный токенизатор для pretrain
        tokenizer_path = None  # Будет определен стандартно
    else:
        # Для SFT/Continual Pretrain токенизатор берется из базовой модели
        tokenizer_path = None

    if loaded_config:
        # Режим для SFT/Continual Pretrain с загруженным конфигом
        # Слайдеры отключены - параметры архитектуры зафиксированы
        disabled_sliders = True
        
        # Используем значения из конфига (поддержка разных имен ключей)
        hidden_size = loaded_config.get("hidden_size", 512)
        # num_hidden_layers - HF, num_layers - наш конфиг
        num_layers = loaded_config.get("num_hidden_layers", loaded_config.get("num_layers", 8))
        num_attention_heads = loaded_config.get("num_attention_heads", loaded_config.get("n_heads", 8))
        max_position_embeddings = loaded_config.get("max_position_embeddings", loaded_config.get("seq_len", 2048))
        
        # Для совместимости возвращаем имена переменных как ожидается
        n_heads = num_attention_heads
        
        # Показываем тип модели из конфига
        base_model_type = loaded_config.get("model_type", loaded_config.get("architectures", ["Unknown"])[0] if "architectures" in loaded_config else "HomeModel")
        if isinstance(base_model_type, list):
            base_model_type = base_model_type[0] if base_model_type else "Unknown"
        st.sidebar.markdown(f"**Тип модели:** `{base_model_type}`")
        
        # Отображаем фиксированные параметры
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Hidden Size", hidden_size)
        c2.metric("Layers", num_layers)
        c1.metric("Heads", n_heads)
        c2.metric("Max Context", f"{max_position_embeddings:,}")
        
        st.sidebar.info(f"🔒 {t('info.architecture_fixed')}")
        
        # seq_len МОЖНО менять - тренировать на меньшем контексте
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚙️ Контекст для обучения**")
        
        # Опции seq_len: стандартные + максимум модели
        seq_len_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        if max_position_embeddings not in seq_len_opts:
            seq_len_opts.append(max_position_embeddings)
        seq_len_opts = sorted([s for s in seq_len_opts if s <= max_position_embeddings])
        
        # По умолчанию 2048 или максимум если меньше
        default_seq = min(2048, max_position_embeddings)
        default_idx = seq_len_opts.index(default_seq) if default_seq in seq_len_opts else len(seq_len_opts) - 1
        
        seq_len = st.sidebar.selectbox(
            "Seq Length (обучение)",
            seq_len_opts,
            index=default_idx,
            help=f"Длина контекста для обучения. Макс. модели: {max_position_embeddings:,}. Можно использовать меньше для экономии памяти."
        )
        
        if seq_len < max_position_embeddings:
            st.sidebar.caption(f"💡 Обучение на контексте {seq_len}, модель поддерживает до {max_position_embeddings:,}")
        
    else:
        # Дефолтные значения
        d_hid, d_layers = 512, 8
        
        if selected_stage == "sft":
            st.sidebar.caption(f"⚠️ {t('warning.check_params_match')}")

        # Определяем тип архитектуры для генерации пресетов
        arch_for_presets = {
            "HomeModel (LLaMA-style)": "home",
            "GPT-2 (Classic)": "gpt2",
            "HomeModel MoE": "home_moe",
            "Custom Blueprint (Visual Builder)": "home",
        }.get(model_type, "home")
        
        # Генерируем пресеты с реальными размерами для выбранной архитектуры
        presets = generate_preset_labels(
            arch_type=arch_for_presets,
            num_experts=num_experts if model_type == "HomeModel MoE" else 8,
            num_experts_per_tok=num_experts_per_tok if model_type == "HomeModel MoE" else 2,
        )
        
        preset_options = list(presets.keys()) + ["Custom"]
        preset = st.sidebar.selectbox(
            "Пресет",
            preset_options,
            index=0,
            help="Размеры рассчитаны для выбранной архитектуры" + (
                ". Для MoE: total = все параметры, active = работающие на токен" 
                if model_type == "HomeModel MoE" else ""
            )
        )
        
        if preset != "Custom" and preset in presets:
            default_h, default_l, default_n = presets[preset]
        else:
            # Если это blueprint mode, используем значения из blueprint как стартовые для слайдеров
            # Но не меняем default_h/l/n глобально, чтобы не сломать логику пресетов если переключат обратно
            if model_type != "Custom Blueprint (Visual Builder)":
                 default_h, default_l, default_n = 512, 8, 8
        
        # Слайдеры (теперь они показывают параметры блюпринта или позволяют менять параметры HomeModel)
        disabled_sliders = (model_type == "Custom Blueprint (Visual Builder)")
        
        hidden_size = st.sidebar.slider(
            "Hidden Size", 
            min_value=128, 
            max_value=2048, 
            value=default_h, 
            step=64,
            help="Размерность скрытого слоя",
            disabled=disabled_sliders
        )
        
        num_layers = st.sidebar.slider(
            "Num Layers", 
            min_value=2, 
            max_value=32, 
            value=default_l,
            help="Количество слоёв трансформера",
            disabled=disabled_sliders
        )
        
        n_heads = st.sidebar.slider(
            "Attention Heads", 
            min_value=2, 
            max_value=32, 
            value=default_n,
            help="Количество голов внимания",
            disabled=disabled_sliders
        )
        
        seq_len_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        if default_seq not in seq_len_opts: seq_len_opts.append(default_seq)
        seq_len_opts = sorted(seq_len_opts)
        
        seq_len = st.sidebar.selectbox(
            "Seq Length",
            seq_len_opts,
            index=seq_len_opts.index(default_seq) if default_seq in seq_len_opts else 0,
            help="Максимальная длина последовательности",
            disabled=disabled_sliders
        )
    
    # Проверка совместимости hidden_size и n_heads
    if not disabled_sliders and hidden_size % n_heads != 0:
        st.sidebar.error(f"⚠️ hidden_size ({hidden_size}) должен делиться на n_heads ({n_heads}) без остатка!")
        valid_heads = [str(i) for i in range(1, min(33, hidden_size+1)) if hidden_size % i == 0][:10]
        if valid_heads:
            st.sidebar.info(f"Рекомендуемые значения n_heads: {', '.join(valid_heads)}")
    
    # Оценка параметров для выбранной архитектуры
    # Используем loaded_config если есть, иначе дефолты
    vocab_size = int(loaded_config.get("vocab_size", 50257)) if loaded_config else 50257
    intermediate_size = int(loaded_config.get("intermediate_size") or (int(hidden_size) * 4)) if loaded_config else (int(hidden_size) * 4)
    
    # Определяем тип архитектуры для расчёта параметров
    if loaded_config:
        # Для загруженных моделей используем home профиль
        arch_for_estimate = "home"
        moe_experts = 8
    else:
        # Маппинг UI -> профиль
        arch_mapping = {
            "HomeModel (LLaMA-style)": "home",
            "GPT-2 (Classic)": "gpt2",
            "HomeModel MoE": "home_moe",
            "Custom Blueprint (Visual Builder)": "home",
        }
        arch_for_estimate = arch_mapping.get(model_type, "home")
        moe_experts = num_experts if model_type == "HomeModel MoE" else 8
    
    est_params = estimate_parameters(
        hidden_size,
        num_layers,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        arch_type=arch_for_estimate,
        num_experts=moe_experts,
        max_position_embeddings=seq_len if not loaded_config else int(loaded_config.get("max_position_embeddings", 4096)),
    )
    
    # Для MoE показываем два значения: Total и Active
    if not loaded_config and model_type == "HomeModel MoE":
        # Вычисляем активные параметры
        expert_params = 3 * hidden_size * intermediate_size  # SwiGLU эксперт
        total_expert_params = num_layers * num_experts * expert_params
        active_expert_params = num_layers * num_experts_per_tok * expert_params
        gate_params = num_layers * hidden_size * num_experts
        
        non_expert_params = est_params - total_expert_params - gate_params
        active_params = non_expert_params + active_expert_params + gate_params
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Total (≈)", format_params(est_params), help="Общее количество параметров модели")
        col2.metric("Active (≈)", format_params(active_params), help=f"Параметры, работающие на токен (Top-{num_experts_per_tok})")
        st.sidebar.caption(f"🔀 {num_experts} экспертов × {format_params(expert_params)} = {format_params(total_expert_params)} в MLP")
    else:
        st.sidebar.metric(t("model.params_approx"), format_params(est_params))
    
    # Model ID для pretrain from scratch (опционально, для HF моделей)
    model_id = None
    if selected_stage == "pretrain":
        st.sidebar.subheader("🔧 Инициализация модели")
        use_hf_model = st.sidebar.checkbox(
            "Использовать HuggingFace модель",
            value=False,
            help="Если включено, можно указать HF model_id для pretrain from scratch"
        )
        if use_hf_model:
            model_id = st.sidebar.text_input(
                "HF Model ID",
                placeholder="gpt2, microsoft/DialoGPT-small, etc.",
                help="HuggingFace model ID для инициализации с нуля"
            )
            if model_id:
                st.sidebar.info(f"Будет использована архитектура: {model_id}")
                # Предупреждение о безопасности
                st.sidebar.warning(
                    "⚠️ **Безопасность**: Загрузка моделей с `trust_remote_code=True` может выполнять "
                    "чужой код. Используйте только проверенные репозитории."
                )
    
    # Метод тюнинга (full/LoRA/QLoRA)
    st.sidebar.subheader(f"🎯 {t('sidebar.tuning_method')}")
    
    # При Unsloth (только GRPO) доступны только lora/qlora
    if training_backend == "unsloth":
        tuning_options = ["lora", "qlora"]
        tuning_index = 0  # lora по умолчанию
        tuning_help = "Unsloth поддерживает только LoRA и QLoRA (не full fine-tuning)"
    else:
        tuning_options = ["full", "lora", "qlora"]
        tuning_index = 0  # full по умолчанию
        tuning_help = "full: полный fine-tuning, lora: LoRA, qlora: QLoRA (4-bit + LoRA)"
    
    tuning_method = st.sidebar.selectbox(
        "Метод",
        tuning_options,
        index=tuning_index,
        help=tuning_help
    )
    
    lora_r = None
    lora_alpha = None
    lora_dropout = None
    lora_target_modules = None
    
    if tuning_method in ("lora", "qlora"):
        # Информация для QLoRA
        if tuning_method == "qlora":
            st.sidebar.info(
                "💡 **QLoRA режим:**\n\n"
                "• 4-bit квантизация базовой модели\n"
                "• LoRA адаптеры в fp16/bf16\n"
                "• ~60-70% экономии памяти!"
            )
        st.sidebar.markdown("**LoRA параметры:**")
        lora_r = st.sidebar.slider(
            "LoRA r (rank)", 
            min_value=8, max_value=128, value=32, step=8,
            help="Ранг LoRA матриц. **Рекомендуется ≥32** для хорошей сходимости. "
                 "При rank=32 для 0.5B-3B моделей сходимость почти как full fine-tuning."
        )
        lora_alpha = st.sidebar.slider(
            "LoRA alpha", 
            min_value=8, max_value=256, value=32, step=8,
            help="Scaling factor. Обычно = lora_r или 2×lora_r"
        )
        lora_dropout = st.sidebar.slider("LoRA dropout", min_value=0.0, max_value=0.5, value=0.05, step=0.05)
        
        # Стандартные target_modules для LLaMA/Qwen/Mistral-подобных архитектур
        LORA_TARGET_MODULES = [
            "q_proj",      # Query projection (Attention)
            "k_proj",      # Key projection (Attention)
            "v_proj",      # Value projection (Attention)
            "o_proj",      # Output projection (Attention)
            "gate_proj",   # Gate projection (MLP/SwiGLU)
            "up_proj",     # Up projection (MLP)
            "down_proj",   # Down projection (MLP)
            "lm_head",     # Output head (logits) — ОГРОМНЫЙ!
            "embed_tokens", # Input embeddings — ОГРОМНЫЙ!
        ]
        
        # Дефолт: all-linear (attention + MLP) — как рекомендует verl
        # Это даёт сходимость почти как full fine-tuning
        default_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_target_modules = st.sidebar.multiselect(
            "🎯 Target modules",
            options=LORA_TARGET_MODULES,
            default=default_modules,
            help="""**Какие слои модели обучать через LoRA:**

**Attention:**
• `q_proj` — Query projection  
• `k_proj` — Key projection  
• `v_proj` — Value projection  
• `o_proj` — Output projection  

**MLP/FFN:**
• `gate_proj` — Gate (SwiGLU)
• `up_proj` — Up projection
• `down_proj` — Down projection

**Output/Input (⚠️ осторожно!):**
• `lm_head` — Output head. **ОГРОМНЫЙ!**
• `embed_tokens` — Input embeddings. **ОГРОМНЫЙ!**

💡 **Рекомендации (из verl):**
- **all-linear** (все 7 слоёв) — сходимость ≈ full fine-tuning!
- Только attention (4 слоя) — быстрее, но хуже качество
- rank≥32 + all-linear = оптимальный баланс

⚠️ **lm_head/embed_tokens:**
- Только для адаптации к новому домену/языку
- Медленнее в Multi-GPU (find_unused_parameters)"""
        )
        
        # Если ничего не выбрано — автодетект
        if not lora_target_modules:
            lora_target_modules = None
    
    # Сборка конфига
    config = {
        "stage": selected_stage,
        "training_backend": training_backend,  # "models-at-home" или "unsloth"
        "base_model_path": base_model_path,
        "model_name_input": model_name,
        "model_id": model_id if model_id else None,
        "tuning_method": tuning_method,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": lora_target_modules,
        # Сохраняем значения слайдеров (для справки)
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
    }

    if model_type == "Custom Blueprint (Visual Builder)":
        config["model_type"] = "blueprint"
        config["blueprint_path"] = blueprint_path
        if tokenizer_path:
             config["tokenizer_path"] = tokenizer_path
    elif model_type == "GPT-2 (Classic)":
        config["model_type"] = "gpt2"
        config["arch_preset"] = "gpt2"
    elif model_type == "HomeModel MoE":
        config["model_type"] = "home_moe"
        config["arch_preset"] = "home_moe"
        config["num_experts"] = num_experts
        config["num_experts_per_tok"] = num_experts_per_tok
        config["expert_type"] = expert_type
    else:
        # HomeModel (LLaMA-style) или другие
        config["model_type"] = "home"
        
    return config


def render_training_config():
    """Training hyperparameters configuration."""
    st.sidebar.header(f"📈 {t('sidebar.hyperparams')}")

    optimizer = st.sidebar.selectbox(
        "Optimizer",
        options=[
            "adamw",
            "adamw_8bit",
            "muon",
            "magma_adamw",
        ],
        index=0,
        help=(
            "adamw — стандартный optimizer.\n"
            "adamw_8bit — экономия памяти (требует bitsandbytes, без DeepSpeed).\n"
            "muon — MuonWithAuxAdam: Muon для hidden 2D параметров, AdamW для остальных. "
            "Поддерживает DeepSpeed (включая ZeRO-3/offload), но может быть медленнее AdamW из-за gather/scatter.\n"
            "magma_adamw — AdamW + стохастическое masking обновлений."
        ),
    )
    
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=1,
        max_value=256,
        value=4,
        help="Размер батча"
    )
    
    grad_accum = st.sidebar.slider(
        "Gradient Accumulation",
        min_value=1,
        max_value=32,
        value=8,
        help="Шаги накопления градиента"
    )
    
    st.sidebar.caption(f"Effective batch: {batch_size * grad_accum}")
    
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
        value=5e-4,
        format_func=lambda x: f"{x:.0e}"
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
        help=(
            "Выберите как менять learning rate по мере обучения. "
            "Важно: scheduler шагает на update-step (после grad_accum), не на каждом micro-batch."
        ),
    )
    lr_schedule_map = {
        "Cosine (with warmup)": "cosine",
        "Linear (with warmup)": "linear",
        "Constant (with warmup)": "constant_with_warmup",
        "Cosine with Restarts (with warmup)": "cosine_with_restarts",
    }
    lr_schedule = lr_schedule_map[lr_schedule_label]

    min_lr_ratio = st.sidebar.slider(
        "Min LR Ratio (Cosine floor)",
        min_value=0.0,
        max_value=0.2,
        value=0.0,
        step=0.01,
        help="0.0 = cosine может уйти почти в 0 к концу. Например 0.05 = не ниже 5% от base LR."
    )
    
    warmup_default = 200 if optimizer == "muon" else 1000
    warmup_steps = st.sidebar.number_input(
        "Warmup Steps",
        min_value=0,
        max_value=10000,
        value=warmup_default
    )

    scheduler_resync_on_resume = st.sidebar.checkbox(
        "Resync LR scheduler при resume (фикс для старых чекпоинтов)",
        value=True,
        help=(
            "Если чекпоинт был сохранён со scheduler, который шагал по micro-batch (а не по update-step), "
            "LR при resume может стать почти 0. Этот флаг принудительно выставляет scheduler на global_step."
        ),
    )
    
    # Choice: epochs or max_steps
    training_mode = st.sidebar.radio(
        t("training.mode"),
        [t("training.mode_epochs"), t("training.mode_steps")],
        help=t("help.training_mode")
    )
    
    if training_mode == t("training.mode_epochs"):
        epochs = st.sidebar.number_input(
            "Epochs",
            min_value=1,
            max_value=10,
            value=1
        )
        max_steps = None
    else:
        epochs = 1
        max_steps = st.sidebar.number_input(
            "Max Steps",
            min_value=1,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Максимальное количество шагов обучения"
        )
    
    max_grad_norm_default = 0.0 if optimizer == "muon" else 1.0
    max_grad_norm = st.sidebar.number_input(
        "Max Gradient Norm",
        min_value=0.0,
        max_value=10.0,
        value=max_grad_norm_default,
        step=0.1,
        help="Gradient clipping для стабильности (0 = отключить)"
    )

    weight_decay = st.sidebar.number_input(
        "Weight Decay",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Коэффициент decoupled weight decay для AdamW-подобных оптимизаторов.",
    )

    betas = st.sidebar.text_input(
        "Adam Betas",
        value="0.9,0.95",
        help="Формат: beta1,beta2",
    )
    try:
        b1_str, b2_str = [x.strip() for x in betas.split(",", 1)]
        beta1 = float(b1_str)
        beta2 = float(b2_str)
    except Exception:
        beta1, beta2 = 0.9, 0.95
        st.sidebar.warning("Некорректный формат betas, использую 0.9, 0.95")

    eps_default = 1e-7 if optimizer == "muon" else 1e-8
    eps = st.sidebar.number_input(
        "Optimizer Epsilon",
        min_value=1e-12,
        max_value=1e-4,
        value=eps_default,
        format="%.1e",
        help="Epsilon для численной стабильности оптимизатора.",
    )

    muon_momentum = 0.95
    muon_nesterov = True
    muon_ns_steps = 5
    muon_ns_coefficients = [3.4445, -4.775, 2.0315]
    muon_adjust_lr_fn = "original"
    muon_lr = float(learning_rate)
    muon_weight_decay = float(weight_decay)
    muon_aux_adamw_lr = float(learning_rate)
    muon_aux_adamw_weight_decay = float(weight_decay)
    muon_aux_adamw_eps = 1e-10
    muon_ds_strict_mode = True
    muon_ds_profile_optimizer_step = False
    muon_ds_fast_aux_adamw = True
    muon_ds_gather_bucket_numel = 50_000_000
    muon_exclude_patterns = [
        r"(^|\.)(embed|embeddings|embed_tokens|tok_embeddings|token_embeddings|wte|wpe|word_embeddings|position_embeddings)(\.|$)",
        r"(^|\.)(lm_head|output|classifier|head)(\.|$)",
        r"(^|\.)(lora(_[AaBb])?|lora[AaBb]?|adapter|adapters|ia3|prompt|prefix)(\.|$)",
    ]
    muon_hidden_patterns = [
        r"(^|\.)(layers|h|blocks)(\.|$)",
    ]
    if optimizer == "muon":
        st.sidebar.markdown("**Muon параметры**")
        muon_momentum = st.sidebar.slider(
            "Muon momentum",
            min_value=0.0,
            max_value=0.999,
            value=0.95,
            step=0.001,
        )
        muon_nesterov = st.sidebar.checkbox(
            "Muon nesterov",
            value=True,
        )
        muon_ns_steps = st.sidebar.slider(
            "Muon ns_steps",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )
        muon_ns_coeff_input = st.sidebar.text_input(
            "Muon ns_coefficients (a,b,c)",
            value="3.4445,-4.775,2.0315",
            help="Формат: a,b,c",
        )
        try:
            ca, cb, cc = [x.strip() for x in muon_ns_coeff_input.split(",", 2)]
            muon_ns_coefficients = [float(ca), float(cb), float(cc)]
        except Exception:
            muon_ns_coefficients = [3.4445, -4.775, 2.0315]
            st.sidebar.warning("Некорректные ns_coefficients, использую дефолтные.")

        muon_adjust_lr_mode = st.sidebar.selectbox(
            "Muon adjust_lr_fn",
            options=["match_rms_adamw", "original", "default(None)"],
            index=1,
            help=(
                "Рекомендуется match_rms_adamw для LLM: позволяет использовать LR/WD ближе к AdamW.\n"
                "original — классический вариант Keller."
            ),
        )
        muon_adjust_lr_fn = None if muon_adjust_lr_mode == "default(None)" else muon_adjust_lr_mode

        st.sidebar.markdown("**Muon/Aux AdamW LR и WD**")
        muon_lr = st.sidebar.select_slider(
            "Muon LR (2D hidden params)",
            options=[1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2],
            value=2e-2,
            format_func=lambda x: f"{x:.0e}",
            help=(
                "Для match_rms_adamw обычно стартуют с AdamW-подобного LR (например 3e-4). "
                "Для original часто нужен существенно выше."
            ),
        )
        muon_aux_adamw_lr = st.sidebar.select_slider(
            "Muon Aux AdamW LR (non-2D/excluded params)",
            options=[1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
            value=3e-4,
            format_func=lambda x: f"{x:.0e}",
            help="LR для AdamW-части гибридного Muon (bias/norm/embeddings/lm_head).",
        )
        muon_weight_decay = st.sidebar.number_input(
            "Muon Weight Decay",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
        )
        muon_aux_adamw_weight_decay = st.sidebar.number_input(
            "Muon Aux AdamW Weight Decay",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
        )
        muon_aux_adamw_eps = st.sidebar.number_input(
            "Muon Aux AdamW Epsilon",
            min_value=1e-12,
            max_value=1e-6,
            value=1e-10,
            format="%.1e",
            help="В README Muon в примере для aux AdamW используют eps=1e-10.",
        )
        st.sidebar.markdown("**Muon + DeepSpeed (advanced)**")
        muon_ds_strict_mode = st.sidebar.checkbox(
            "Muon DS strict mode",
            value=True,
            help=(
                "Если включено, Muon остановит обучение при нарушении distributed-инвариантов "
                "(например, когда full grad недоступен для Muon-параметра)."
            ),
        )
        muon_ds_profile_optimizer_step = st.sidebar.checkbox(
            "Muon DS profile optimizer step",
            value=False,
            help="Логировать время gather/NS/scatter и оценку объема собранных параметров.",
        )
        muon_ds_fast_aux_adamw = st.sidebar.checkbox(
            "Muon DS fast aux AdamW",
            value=True,
            help=(
                "Ускоренный путь для aux AdamW в distributed режиме: обновление локальных shard-параметров "
                "без full gather. Обычно заметно быстрее на ZeRO-3/CPU offload."
            ),
        )
        muon_ds_gather_bucket_numel = int(
            st.sidebar.number_input(
                "Muon DS gather bucket size (numel)",
                min_value=1_000_000,
                max_value=500_000_000,
                value=50_000_000,
                step=1_000_000,
                help=(
                    "Размер bucket для gather Muon-параметров. Меньше — ниже пик памяти, "
                    "больше — меньше накладных расходов на gather/scatter."
                ),
            )
        )

        muon_exclude_text = st.sidebar.text_input(
            "Muon exclude patterns (regex, comma-separated)",
            value=", ".join(muon_exclude_patterns),
            help=(
                "Параметры, которые НЕ должны идти в Muon (обычно embeddings/lm_head и LoRA/adapters). "
                "Они будут оптимизироваться AdamW в гибридном режиме."
            ),
        )
        parsed_patterns = [x.strip() for x in muon_exclude_text.split(",") if x.strip()]
        if parsed_patterns:
            muon_exclude_patterns = parsed_patterns

        muon_hidden_text = st.sidebar.text_input(
            "Muon hidden patterns (regex, comma-separated)",
            value=", ".join(muon_hidden_patterns),
            help=(
                "Паттерны для hidden transformer-блоков, куда можно применять Muon "
                "(обычно слои типа layers/h/blocks)."
            ),
        )
        parsed_hidden_patterns = [x.strip() for x in muon_hidden_text.split(",") if x.strip()]
        if parsed_hidden_patterns:
            muon_hidden_patterns = parsed_hidden_patterns

    magma_prob = 0.5
    magma_tau = 2.0
    magma_ema_beta = 0.9
    magma_cosine_eps = 1e-12
    if optimizer == "magma_adamw":
        st.sidebar.markdown("**Magma параметры**")
        magma_prob = st.sidebar.slider(
            "Magma sampling ratio (p)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Вероятность оставить update блока на шаге.",
        )
        magma_tau = st.sidebar.select_slider(
            "Magma temperature (tau)",
            options=[0.5, 1.0, 2.0, 4.0],
            value=2.0,
            help="Температура для sigmoid(cos/tau).",
        )
        magma_ema_beta = st.sidebar.slider(
            "Magma EMA beta",
            min_value=0.0,
            max_value=0.99,
            value=0.9,
            step=0.01,
            help="Сглаживание коэффициента масштаба s_t.",
        )
        magma_cosine_eps = st.sidebar.number_input(
            "Magma cosine epsilon",
            min_value=1e-16,
            max_value=1e-6,
            value=1e-12,
            format="%.1e",
            help="Стабилизатор в знаменателе cosine similarity.",
        )
    
    return {
        "optimizer": optimizer,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "betas": [beta1, beta2],
        "eps": eps,
        "lr_schedule": lr_schedule,
        "min_lr_ratio": min_lr_ratio,
        "warmup_steps": warmup_steps,
        "scheduler_resync_on_resume": scheduler_resync_on_resume,
        "epochs": epochs,
        "max_steps": max_steps,
        "max_grad_norm": max_grad_norm,
        "muon_momentum": muon_momentum,
        "muon_nesterov": muon_nesterov,
        "muon_ns_steps": muon_ns_steps,
        "muon_ns_coefficients": muon_ns_coefficients,
        "muon_adjust_lr_fn": muon_adjust_lr_fn,
        "muon_lr": muon_lr,
        "muon_weight_decay": muon_weight_decay,
        "muon_aux_adamw_lr": muon_aux_adamw_lr,
        "muon_aux_adamw_weight_decay": muon_aux_adamw_weight_decay,
        "muon_aux_adamw_eps": muon_aux_adamw_eps,
        "muon_ds_strict_mode": muon_ds_strict_mode,
        "muon_ds_profile_optimizer_step": muon_ds_profile_optimizer_step,
        "muon_ds_fast_aux_adamw": muon_ds_fast_aux_adamw,
        "muon_ds_gather_bucket_numel": muon_ds_gather_bucket_numel,
        "muon_hidden_patterns": muon_hidden_patterns,
        "muon_exclude_patterns": muon_exclude_patterns,
        "magma_prob": magma_prob,
        "magma_tau": magma_tau,
        "magma_ema_beta": magma_ema_beta,
        "magma_cosine_eps": magma_cosine_eps,
    }


def render_dataset_config(stage="pretrain"):
    """Dataset selection and validation settings."""
    st.sidebar.header(f"📁 {t('sidebar.data')}")
    
    datasets = get_available_datasets()
    
    if datasets:
        dataset_options = [f"{name} ({size})" for name, size in datasets]
        selected = st.sidebar.selectbox(t("data.select_dataset"), dataset_options)
        selected_name = selected.split(" (")[0]
        data_path = str(DATASET_DIR / selected_name)
    else:
        st.sidebar.warning(t("data.no_datasets"))
        data_path = st.sidebar.text_input("Путь к датасету", "datasets/data.jsonl")

    # Sharding mode: гарантирует отсутствие двойного шардинга и корректную семантику resume
    st.sidebar.divider()
    st.sidebar.subheader(f"🧩 {t('data.sharding')}")
    sharding_mode = st.sidebar.selectbox(
        "Sharding mode",
        options=["auto", "dataset", "accelerate"],
        index=0,
        help=(
            "auto: для streaming (IterableDataset) выбираем dataset-level шардинг (строго и совместимо с strict resume).\n"
            "dataset: шардинг делает сам датасет (shard=True), DataLoader НЕ готовим через accelerate.\n"
            "accelerate: шардинг делает accelerate.prepare(DataLoader); строгий resume для streaming отключается."
        ),
    )
    
    # Validation / Eval
    st.sidebar.divider()
    st.sidebar.subheader(f"📊 {t('data.validation')}")
    
    val_ratio = st.sidebar.slider(
        "Validation fraction",
        min_value=0.0,
        max_value=0.2,
        value=0.01,
        step=0.005,
        help="Доля данных под validation, если отдельный val-файл не задан"
    )
    
    eval_every = st.sidebar.number_input(
        "Eval Every N Steps",
        min_value=0,
        max_value=50000,
        value=200,
        step=10,
        help="Как часто запускать валидацию (0 = отключить)"
    )
    
    eval_batches = st.sidebar.number_input(
        "Eval Batches",
        min_value=1,
        max_value=500,
        value=20,
        step=1,
        help="Сколько батчей прогонять на валидации (чтобы не было слишком долго)"
    )
    
    return {
        "data_path": data_path,
        "sharding_mode": sharding_mode,
        "val_ratio": val_ratio,
        "eval_every": eval_every,
        "eval_batches": eval_batches,
    }


def render_output_config(model_name="training_run"):
    """Output configuration."""
    st.sidebar.header(f"💾 {t('sidebar.save')}")
    
    # Автоматический путь: out/{model_name}
    default_dir = f"out/{model_name}"
    
    output_dir = st.sidebar.text_input(
        "Output Directory (Experiment Root)",
        value=default_dir,
        help="Корневая папка для всех запусков этого эксперимента"
    )
    
    save_every = st.sidebar.number_input(
        "Save Checkpoint Every N Steps",
        min_value=1,
        max_value=5000,
        value=100,
        step=1,
        help="Как часто сохранять чекпоинты"
    )

    export_on_checkpoint = st.sidebar.checkbox(
        "Экспортировать final_model при каждом сохранении чекпоинта",
        value=True,
        help=(
            "Будет обновлять `final_model/` на каждом checkpoint, чтобы модель можно было сразу грузить в чат. "
            "Минус: дополнительное время и место на диске."
        ),
    )
    
    merge_lora = st.sidebar.checkbox(
        "Merge LoRA в final_model",
        value=True,
        help=(
            "Если включено, LoRA адаптеры будут объединены с базовой моделью при сохранении final_model. "
            "Это упрощает inference (модель загружается как обычная HF модель), но увеличивает размер файла. "
            "Если выключено, сохраняются только LoRA веса (требует PEFT для загрузки)."
        ),
    )
    
    log_every = st.sidebar.number_input(
        "Log Every N Steps",
        min_value=1,
        max_value=1000,
        value=10,
        help="Как часто обновлять метрики"
    )
    
    # Показываем информацию о чекпоинтах
    output_path = PROJECT_ROOT / output_dir
    if output_path.exists():
        checkpoints = list(output_path.rglob("checkpoint_step*"))
        final_models = list(output_path.rglob("final_model"))
        final_model = final_models[0] if final_models else None
        
        if checkpoints or final_model:
            st.sidebar.caption(f"📦 Найдено чекпоинтов: {len(checkpoints)}")
            if final_model and final_model.exists():
                st.sidebar.caption(f"✅ {t('chat.final_model_saved')}")
    
    return {
        "output_dir": output_dir,
        "save_every": save_every,
        "export_on_checkpoint": export_on_checkpoint,
        "merge_lora": merge_lora,
        "log_every": log_every,
        "tokenizer_path": "gpt2"
    }


def get_available_models():
    """Получить список доступных обученных моделей (рекурсивный поиск).
    
    Поддерживаемые типы:
    - Pretrain модели (home_pretrain/)
    - SFT модели (home_sft/)  
    - GRPO/RL модели (home_grpo/, home_rl/)
    - LoRA адаптеры (adapter_config.json)
    - HuggingFace модели (models/)
    """
    models = []
    
    def detect_training_type(model_dir: Path) -> str:
        """Определяет тип тренировки по пути."""
        path_str = str(model_dir).lower()
        if "grpo" in path_str or "_rl" in path_str:
            return "grpo"
        elif "sft" in path_str:
            return "sft"
        elif "pretrain" in path_str:
            return "pretrain"
        return "unknown"
    
    def is_lora_model(model_dir: Path) -> bool:
        """Проверяет, является ли модель LoRA адаптером."""
        return (model_dir / "adapter_config.json").exists()
    
    def get_model_info(model_dir: Path) -> dict:
        """Читает информацию о модели из конфигов."""
        info = {"max_context": None, "vocab_size": None, "hidden_size": None, "num_params": None}
        
        # Читаем config.json
        config_path = model_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                info["max_context"] = cfg.get("max_position_embeddings") or cfg.get("n_positions") or cfg.get("max_seq_len")
                info["vocab_size"] = cfg.get("vocab_size")
                info["hidden_size"] = cfg.get("hidden_size") or cfg.get("n_embd")
                info["model_type"] = cfg.get("model_type", "unknown")
            except:
                pass
        
        # Для LoRA читаем adapter_config.json
        adapter_config_path = model_dir / "adapter_config.json"
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path) as f:
                    adapter_cfg = json.load(f)
                info["base_model"] = adapter_cfg.get("base_model_name_or_path")
                info["lora_r"] = adapter_cfg.get("r")
                info["lora_alpha"] = adapter_cfg.get("lora_alpha")
            except:
                pass
        
        return info
    
    def has_model_weights(model_dir: Path) -> bool:
        """Проверяет наличие весов модели."""
        return (
            (model_dir / "pytorch_model.bin").exists() or 
            (model_dir / "model.safetensors").exists() or
            (model_dir / "adapter_model.bin").exists() or
            (model_dir / "adapter_model.safetensors").exists() or
            any(model_dir.glob("model-*.safetensors")) or  # Sharded модели
            any(model_dir.glob("pytorch_model-*.bin"))
        )
    
    # 1. Ищем рекурсивно в out/ (обученные модели)
    if OUTPUT_DIR.exists():
        # Ищем config.json и adapter_config.json
        config_files = list(OUTPUT_DIR.rglob("config.json")) + list(OUTPUT_DIR.rglob("adapter_config.json"))
        seen_dirs = set()
        
        for config_file in config_files:
            model_dir = config_file.parent
            
            # Пропускаем дубликаты
            if str(model_dir) in seen_dirs:
                continue
            seen_dirs.add(str(model_dir))
            
            # Проверяем наличие весов
            if not has_model_weights(model_dir):
                continue
            
            # Определяем тип
            is_lora = is_lora_model(model_dir)
            training_type = detect_training_type(model_dir)
            
            # Определяем тип модели (final/checkpoint)
            m_type = "checkpoint" if "checkpoint" in model_dir.name.lower() else "final"
            if model_dir.name == "final_model":
                m_type = "final"
            elif model_dir.name == "lora_adapters":
                m_type = "lora"
            
            # Получаем информацию о модели
            model_info = get_model_info(model_dir)
            
            # Формируем красивое имя
            rel_path = model_dir.relative_to(OUTPUT_DIR)
            
            # Эмодзи по типу тренировки
            type_emoji = {
                "pretrain": "📚",
                "sft": "💬", 
                "grpo": "🧠",
                "unknown": "📦"
            }.get(training_type, "📦")
            
            lora_badge = " [LoRA]" if is_lora else ""
            
            models.append({
                "name": f"{type_emoji} {rel_path}{lora_badge}",
                "path": str(model_dir),
                "type": m_type,
                "training_type": training_type,
                "is_lora": is_lora,
                "model_info": model_info,
                "time": model_dir.stat().st_mtime
            })
    
    # 2. Ищем в models/ (скачанные с HuggingFace)
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                if not has_model_weights(model_dir):
                    continue
                    
                # Должен быть config.json
                if not (model_dir / "config.json").exists():
                    continue
                
                model_info = get_model_info(model_dir)
                
                models.append({
                    "name": f"🤗 {model_dir.name}",
                    "path": str(model_dir),
                    "type": "hf",
                    "training_type": "base",
                    "is_lora": False,
                    "model_info": model_info,
                    "time": model_dir.stat().st_mtime
                })
    
    # Сортируем по времени (новые сверху)
    models.sort(key=lambda x: x["time"], reverse=True)
    return models


def render_distributed_config(training_config: dict | None = None, is_grpo: bool = False, grpo_backend: str | None = None):
    """GPU, parallelism and memory configuration."""
    st.sidebar.header(f"🖥️ {t('sidebar.gpu_memory')}")
    
    # Информация о GPU
    gpus = get_gpu_info()
    
    if gpus:
        st.sidebar.success(f"✅ {t('gpu.found')}: {len(gpus)}")
        
        # Показываем карточки GPU
        for gpu in gpus:
            st.sidebar.markdown(f"""
            **GPU {gpu['id']}**: {gpu['name']}  
            📊 VRAM: {gpu['memory_gb']} GB | CC: {gpu['compute_capability']}
            """)
        
        # Выбор GPU для обучения
        gpu_options = [f"GPU {g['id']}: {g['name']}" for g in gpus]
        if len(gpus) > 1:
            selected_gpus = st.sidebar.multiselect(
                "Выберите GPU",
                options=gpu_options,
                default=gpu_options,
                help="Выберите GPU для обучения",
                key="gpu_select_multiselect"
            )
            num_gpus = len(selected_gpus)
            gpu_ids = [gpu_options.index(g) for g in selected_gpus]
        else:
            num_gpus = 1
            gpu_ids = [0]
            st.sidebar.info(t("gpu.single_gpu"))
    else:
        st.sidebar.warning("⚠️ GPU не найдены, будет использован CPU")
        num_gpus = 0
        gpu_ids = []
    
    st.sidebar.markdown("---")
    
    # Выбор типа параллелизма
    st.sidebar.subheader(f"⚡ {t('parallel.type')}")
    
    # Определяем доступные опции и рекомендуемый режим
    if num_gpus == 0:
        available_modes = ["default"]
        recommended_idx = 0
    elif num_gpus == 1:
        available_modes = ["default", "deepspeed_zero3_offload"]
        recommended_idx = 0
    else:
        # При нескольких GPU рекомендуем multi_gpu или fsdp
        available_modes = ["multi_gpu", "deepspeed_zero3", "deepspeed_zero3_offload", "deepspeed_zero2", "fsdp", "fsdp_offload", "fsdp2", "default"]
        recommended_idx = 0  # multi_gpu по умолчанию
    
    # Форматируем опции для selectbox
    mode_options = []
    for i, mode in enumerate(available_modes):
        info = PARALLEL_TYPES[mode]
        label = f"{info['icon']} {info['name']}"
        if i == recommended_idx and num_gpus > 1:
            label += " ⭐"  # Отмечаем рекомендуемый
        mode_options.append(label)
    
    selected_mode_display = st.sidebar.selectbox(
        "Режим",
        options=mode_options,
        index=recommended_idx,
        help="Выберите стратегию распределённого обучения"
    )
    
    # Находим выбранный режим
    selected_idx = mode_options.index(selected_mode_display)
    selected_mode = available_modes[selected_idx]
    mode_info = PARALLEL_TYPES[selected_mode]
    
    # Показываем информацию о выбранном режиме
    st.sidebar.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin: 10px 0;">
    <b>{t('parallel.type')}:</b> {mode_info['type']}<br>
    <small>{t(mode_info['description_key'])}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Предупреждение если выбран single GPU при нескольких
    if num_gpus > 1 and selected_mode == "default":
        st.sidebar.warning(f"⚠️ Выбран Single GPU, но доступно {num_gpus} GPU. Рекомендуем Multi-GPU!")
    
    # Конфиг файл
    config_file = None
    if selected_mode != "default":
        config_path = CONFIGS_DIR / f"accelerate_{selected_mode}.yaml"
        if config_path.exists():
            config_file = str(config_path)
            st.sidebar.caption(f"📄 Конфиг: `{config_path.name}`")
    
    # Показываем итоговую конфигурацию запуска
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"🚀 {t('parallel.mode')}")
    
    if num_gpus == 0:
        launch_info = "**Устройство:** CPU"
    elif selected_mode == "default":
        launch_info = f"**Устройство:** GPU {gpu_ids[0] if gpu_ids else 0}"
    else:
        launch_info = f"**Устройства:** {num_gpus} × GPU\n**Режим:** {mode_info['type']}"
    
    st.sidebar.info(launch_info)

    # Compute / precision (нужно и для GRPO, потому что training_config для GRPO пустой)
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"🧠 {t('sidebar.precision_memory')}")
    
    # Backend выбирается только в render_model_config() для GRPO.
    # Здесь используем уже выбранное значение (или дефолт для остальных стадий).
    training_backend = grpo_backend if is_grpo else "models-at-home"

    # Если training_config передан (SFT/Pretrain) — берём дефолт из него, иначе bf16 (GRPO дефолт)
    default_mp = (training_config.get("mixed_precision") if training_config else None) or "bf16"
    mixed_precision = st.sidebar.selectbox(
        "Mixed Precision",
        ["no", "fp16", "bf16"],
        index=["no", "fp16", "bf16"].index(default_mp) if default_mp in ("no", "fp16", "bf16") else 2,
        help=(
            "bf16 рекомендуется для Ampere+ GPU (обычно меньше VRAM и стабильнее). "
            "fp16 в AMP режиме часто держит fp32-\"master\" веса + GradScaler и может занимать больше VRAM, чем bf16. "
            "Для FlashAttention нужен fp16/bf16."
        ),
    )

    # FP16: два режима
    # - AMP fp16 (дефолт): fp32 master-веса + GradScaler => может стартовать с большим VRAM, чем bf16
    # - Pure fp16: веса fp16, без GradScaler => VRAM ближе к bf16, но может быть менее стабильным
    fp16_pure = False
    if mixed_precision == "fp16":
        fp16_pure = st.sidebar.checkbox(
            "FP16 Pure (веса fp16, без GradScaler)",
            value=False,
            help=(
                "Снижает VRAM на старте (веса fp16, как у bf16), но может быть менее стабильным, чем AMP fp16. "
                "Используйте, если fp16 OOM'ится из-за fp32 master-весов."
            ),
        )

    default_gc = bool(training_config.get("grad_checkpoint", True)) if training_config else True
    grad_checkpoint = st.sidebar.checkbox(
        "Gradient Checkpointing",
        value=default_gc,
        help="Экономит VRAM, но медленнее. Для GRPO (особенно full+длинные ответы) часто must-have.",
    )

    # === Backend-specific optimizations ===
    # Показываем только для models-at-home backend
    if training_backend == "models-at-home":
        # FlashAttention toggle (для всех стадий).
        # - HF модели: attn_implementation=flash_attention_2 (требует flash_attn + fp16/bf16)
        # - Home модели: SDPA (scaled_dot_product_attention) может использовать flash kernel автоматически при fp16/bf16
        default_fa = bool(training_config.get("use_flash_attention", True)) if training_config else True
        flash_attention = st.sidebar.checkbox(
            "FlashAttention (ускорение attention)",
            value=default_fa,
            help=(
                "Включает быстрый attention где возможно. "
                "Для HF-моделей использует FlashAttention2 (если установлен flash-attn и включен fp16/bf16). "
                "Для Home-моделей управляет использованием SDPA."
            ),
        )
        
        # Liger Kernel — оптимизированные Triton kernels для ускорения и экономии памяти
        # Применяется ко ВСЕМ режимам: Pretrain, SFT, GRPO
        default_liger = bool(training_config.get("use_liger", True)) if training_config else True
        use_liger = st.sidebar.checkbox(
            "🦁 Liger Kernel оптимизации",
            value=default_liger,
            help=(
                "**Применяется ко ВСЕМ режимам** (Pretrain, SFT, GRPO).\n\n"
                "Включает оптимизированные Triton-кернелы:\n"
                "• RMSNorm, RoPE, MLP — патчинг модели\n"
                "• 🔥 Fused Loss — НЕ материализует logits (до 80% экономии памяти!)\n\n"
                "**Поддерживаемые модели**: Qwen, Llama, Mistral, Gemma, Phi."
            ),
        )
        
        # Fused Loss — НЕ материализует logits!
        # Для Pretrain/SFT это Fused CE, для GRPO — Fused GRPO Loss (включается автоматически)
        default_liger_fused_ce = bool(training_config.get("liger_fused_ce", True)) if training_config else True
        liger_fused_ce = st.sidebar.checkbox(
            "🔥 Fused Loss (экономия памяти)",
            value=default_liger_fused_ce,
            disabled=not use_liger,
            help=(
                "**Pretrain/SFT**: LigerFusedLinearCrossEntropyLoss\n"
                "**GRPO**: LigerFusedLinearGRPOLoss (автоматически)\n\n"
                "НЕ материализует полный logits тензор [batch, seq, vocab] — "
                "экономия памяти до 80%! **Рекомендуется всегда включать.**"
            ),
        )
    else:
        # Unsloth backend — эти настройки управляются Unsloth автоматически
        flash_attention = True  # Unsloth включает свои оптимизации
        use_liger = False  # Unsloth имеет свои Triton kernels
        liger_fused_ce = False
        
        st.sidebar.caption(
            "🦥 **Unsloth оптимизации включены автоматически:**\n"
            "• Triton RMSNorm, RoPE, SwiGLU\n"
            "• Smart Gradient Checkpointing\n"
            "• Fused Cross-Entropy Loss"
        )

    # Пояснение про batch semantics (частая причина "почему так много VRAM в DDP")
    if training_config:
        try:
            micro_bsz = int(training_config.get("batch_size", 1))
            grad_accum = int(training_config.get("gradient_accumulation", 1))
            eff_per_gpu = micro_bsz * grad_accum
            global_batch = eff_per_gpu * max(1, int(num_gpus or 1))
            st.sidebar.caption(
                f"Batch semantics: **per‑GPU microbatch** = {micro_bsz}, "
                f"accum = {grad_accum} → **effective per‑GPU** = {eff_per_gpu}, "
                f"**global** = {global_batch} (×{max(1, int(num_gpus or 1))} GPU)"
            )
            st.sidebar.caption("Важно: в DDP `batch_size` применяется на каждом процессе, т.е. это именно per‑GPU.")
        except Exception:
            pass
    
    return {
        "distributed_mode": selected_mode,
        "num_gpus": num_gpus,
        "gpu_ids": gpu_ids,
        "config_file": config_file,
        "parallel_type": mode_info['type'],
        "mixed_precision": mixed_precision,
        "fp16_pure": fp16_pure,
        "grad_checkpoint": grad_checkpoint,
        "use_flash_attention": flash_attention,
        "use_liger": use_liger,
        "liger_fused_ce": liger_fused_ce,
        "training_backend": training_backend,  # "models-at-home" или "unsloth"
    }


# Graceful fallback для @st.fragment (работает только в новых версиях Streamlit)
try:
    fragment = st.fragment
except (AttributeError, Exception):
    # Fallback для старых версий Streamlit
    fragment = lambda *args, **kwargs: lambda fn: fn

@fragment(run_every=3)  # Автообновление каждые 3 секунды
def live_metrics_fragment():
    """Fragment для живого обновления метрик без перезагрузки всей страницы."""
    if not st.session_state.current_run_id:
        st.info(t("status.select_run"))
        return
    
    run_id = st.session_state.current_run_id
    metrics = load_metrics(run_id)
    process_alive = is_process_running(run_id)
    
    # Статус
    if process_alive:
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.success(f"🟢 {t('status.running')} (Run: {run_id})")
        with col2:
            if st.button("⏹️ Стоп", key=f"stop_{run_id}", use_container_width=True):
                with st.spinner("Останавливаем..."):
                    stop_training()
                st.rerun()
    else:
        if metrics and metrics.get("status") == "completed":
            duration = metrics.get("training_duration", "unknown")
            st.success(f"✅ {t('status.completed')} {duration} (Run: {run_id})")
            clear_active_run()  # Очищаем active_run.json при завершении
            _close_run_log_files(run_id)  # Закрываем файловые дескрипторы
        elif metrics and metrics.get("status") == "error":
            st.error(f"❌ {t('status.error')} (Run: {run_id})")
            clear_active_run()  # Очищаем active_run.json при ошибке
            _close_run_log_files(run_id)  # Закрываем файловые дескрипторы
        elif metrics and metrics.get("status") == "stopped":
            st.warning(f"⏹️ {t('status.stopped')} (Run: {run_id})")
            clear_active_run()  # Очищаем active_run.json при остановке
            _close_run_log_files(run_id)  # Закрываем файловые дескрипторы
        else:
            st.info(f"📋 {t('status.viewing_metrics')} (Run: {run_id})")
    
    if metrics:
        render_metrics_dashboard(metrics)
    else:
        st.info(t("status.no_metrics"))


def render_metrics_dashboard(metrics: dict):
    """Дашборд с метриками обучения."""
    
    status = metrics.get("status", "unknown")
    # Преобразуем в строку (может быть NaN/float из pandas)
    import math
    if status is None or (isinstance(status, float) and math.isnan(status)):
        status = "training"
    status = str(status)
    
    # Status indicator
    status_emoji = {
        "training": "🟢", 
        "completed": "✅", 
        "error": "❌",
        "initializing": "⏳",
        "loading_tokenizer": "⏳",
        "loading_dataset": "⏳",
        "building_model": "⏳",
        "saving_model": "💾",
    }.get(status, "⏳")
    
    st.subheader(f"{status_emoji} {t('common.status')}: {status.upper()}")
    
    # Параметры модели из metrics.json (если есть) или из config
    model_params = None
    try:
        # Сначала пробуем взять из metrics.json (более точно)
        if "num_parameters" in metrics:
            model_params = metrics["num_parameters"]
        else:
            # Fallback: рассчитываем из config
            run_id = st.session_state.get("current_run_id", "active")
            if run_id and run_id != "active":
                run_dir = RUNS_DIR / run_id
                config_path = run_dir / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        rc = json.load(f)
                        if "hidden_size" in rc and "num_layers" in rc:
                            vocab_size = rc.get("vocab_size", 50257)
                            intermediate_size = rc.get("intermediate_size")
                            model_params = estimate_parameters(
                                rc["hidden_size"],
                                rc["num_layers"],
                                vocab_size=vocab_size,
                                intermediate_size=intermediate_size,
                            )
    except Exception:
        pass
    
    # Metrics cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        current_step = metrics.get("current_step", 0)
        total_steps = metrics.get("total_steps", None)
        progress = (current_step / total_steps * 100) if isinstance(total_steps, (int, float)) and total_steps > 0 else 0
        # защита от "48000%" и прочих артефактов
        progress = max(0.0, min(float(progress), 100.0))
        planned_total = metrics.get("planned_total_steps", None)
        suffix = f"Step {current_step}/{total_steps}" if isinstance(total_steps, (int, float)) and total_steps else f"Step {current_step} (без лимита)"
        if planned_total is not None and int(planned_total) != int(total_steps):
            suffix = f"{suffix} (план: {planned_total})"
        st.metric(t("status.progress"), f"{progress:.1f}%", suffix)

    # Дополнительный прогресс для GRPO: по датасету (сколько промптов прошло)
    if metrics.get("stage") == "grpo":
        run_id = st.session_state.get("current_run_id", None)
        dataset_total = None
        prompt_bsz = metrics.get("prompt_batch_size", None)
        group_size = metrics.get("group_size", None)
        rollout_step = metrics.get("rollout_step", metrics.get("current_rollout_step", 0))
        num_gpus = 1
        try:
            if run_id:
                run_dir = RUNS_DIR / run_id
                cfg_path = run_dir / "config.json"
                if cfg_path.exists():
                    with open(cfg_path) as f:
                        rc = json.load(f) or {}
                    # num_gpus сохраняем в config.json при запуске GRPO
                    num_gpus = int(rc.get("num_gpus", 1) or 1)
                    # dataset_size: берём из run-config если было сохранено, иначе попробуем dataset_info.json из output_dir
                    dataset_total = rc.get("dataset_size", None)
                    if dataset_total is None:
                        out_dir = rc.get("output_dir")
                        if out_dir:
                            info_path = Path(out_dir) / "dataset_info.json"
                            if info_path.exists():
                                with open(info_path) as inf:
                                    dataset_total = (json.load(inf) or {}).get("dataset_size", None)
        except Exception:
            pass
        try:
            dataset_total = int(dataset_total) if dataset_total is not None else None
        except Exception:
            dataset_total = None
        try:
            prompt_bsz = int(prompt_bsz) if prompt_bsz is not None else None
        except Exception:
            prompt_bsz = None
        try:
            group_size = int(group_size) if group_size is not None else None
        except Exception:
            group_size = None
        try:
            rollout_step = int(rollout_step)
        except Exception:
            rollout_step = 0

        # prompts/completions: предпочитаем ФАКТ из метрик, иначе fallback на оценку
        prompts_seen = metrics.get("prompts_generated_total", None)
        prompts_used = metrics.get("prompts_used_total", None)
        completions_seen = metrics.get("completions_generated_total", None)
        experiences_tuned = metrics.get("experiences_tuned_total", None)

        # Оценка "сколько промптов прошло по датасету" из rollout_step (глобально, с учётом num_gpus).
        # Это соответствует пользовательской семантике "prompts/step".
        prompts_seen_est = None
        if prompt_bsz is not None:
            prompts_seen_est = rollout_step * prompt_bsz * max(1, num_gpus)
            if prompts_seen is None or (isinstance(prompts_seen, (int, float)) and float(prompts_seen) < float(prompts_seen_est)):
                prompts_seen = prompts_seen_est
        if completions_seen is None and prompts_seen is not None and group_size is not None:
            completions_seen = prompts_seen * group_size

        if prompts_seen is not None:
            if dataset_total is not None and dataset_total > 0:
                ds_progress = max(0.0, min(100.0, prompts_seen / dataset_total * 100.0))
                st.caption(f"Прогресс по датасету (генерация): **{int(prompts_seen):,}/{dataset_total:,} промптов** ({ds_progress:.1f}%)")
                st.progress(ds_progress / 100.0)
            else:
                st.caption(f"Промптов обработано (генерация): **{int(prompts_seen):,}**")

            if prompts_used is not None:
                st.caption(f"Промптов использовано в обучении (после фильтрации): **{int(prompts_used):,}**")
            if completions_seen is not None:
                st.caption(f"Completion'ов сгенерировано: **{int(completions_seen):,}**")
            if experiences_tuned is not None:
                st.caption(f"Experience'ов протюнено (батчей в train): **{int(experiences_tuned):,}**")

        # Скорости (prompts/s, completions/s) по истории steps/timestamps
        try:
            elapsed = float(metrics.get("elapsed_seconds", 0.0))
            if elapsed > 0 and prompts_seen is not None:
                st.caption(f"Скорость: **{(float(prompts_seen)/elapsed):.2f} промптов/с**")
            if elapsed > 0 and completions_seen is not None:
                st.caption(f"Скорость: **{(float(completions_seen)/elapsed):.2f} completion/с**")
            if elapsed > 0 and experiences_tuned is not None:
                st.caption(f"Скорость: **{(float(experiences_tuned)/elapsed):.2f} tuned exp/с**")
        except Exception:
            pass
    
    with col2:
        if metrics.get("stage") == "grpo":
            reward = metrics.get("reward", metrics.get("batch_reward_mean", 0))
            st.metric("Reward", f"{reward:.4f}")
        else:
            loss = metrics.get("current_loss", 0)
            st.metric("Train Loss", f"{loss:.4f}")
    
    with col3:
        if metrics.get("stage") == "grpo":
            kl = metrics.get("kl", 0)
            st.metric("KL Divergence", f"{kl:.4f}")
        else:
            vloss = metrics.get("current_val_loss", None)
            if vloss is None:
                st.metric("Val Loss", "—")
            else:
                st.metric("Val Loss", f"{vloss:.4f}")
    
    with col4:
        lr = metrics.get("current_lr", 0)
        st.metric("Learning Rate", f"{lr:.2e}")
    
    with col5:
        if model_params:
            st.metric(t("model.params"), format_params(model_params))
        else:
            st.metric(t("model.params"), "—")

    # Доп. пояснения: план vs факт, причина остановки, LR floor
    planned_total = metrics.get("planned_total_steps", None)
    total_steps = metrics.get("total_steps", None)
    stop_reason = metrics.get("stop_reason", None)
    min_lr_ratio = metrics.get("min_lr_ratio", None)
    if planned_total is not None and total_steps is not None and int(planned_total) != int(total_steps):
        st.caption(f"План шагов: {planned_total} • Факт (для прогресса/ETA): {total_steps}")
    if stop_reason:
        st.caption(f"Причина остановки: `{stop_reason}`")
    if min_lr_ratio is not None and float(min_lr_ratio) > 0:
        st.caption(f"Cosine LR floor включён: min_lr_ratio={float(min_lr_ratio):.2f}")
    
    with col6:
        eta = metrics.get("eta_seconds", 0)
        elapsed = metrics.get("elapsed_seconds", 0)
        st.metric(t("status.time"), f"{format_time(elapsed)}", delta=f"{t('status.remaining')}: {format_time(eta)}", delta_color="normal")
    
    # Progress bar
    st.progress(min(progress / 100, 1.0))
    
    # Charts
    # ВАЖНО: используем стабильный ключ по run_id, чтобы избежать утечки памяти
    rid = st.session_state.get("current_run_id", "active") or "active"
    
    # Специальная секция для GRPO
    if metrics.get("stage") == "grpo":
        st.markdown("---")
        st.subheader(f"🧠 {t('grpo.monitoring')}")
        
        # Метрики GRPO
        col_grpo1, col_grpo2, col_grpo3, col_grpo4 = st.columns(4)
        with col_grpo1:
            reward = metrics.get("reward", metrics.get("batch_reward_mean", 0))
            st.metric("Reward", f"{reward:.4f}")
        with col_grpo2:
            kl = metrics.get("kl", 0)
            st.metric("KL Divergence", f"{kl:.4f}")
        with col_grpo3:
            grad_norm = metrics.get("grad_norm", 0)
            st.metric("Grad Norm", f"{grad_norm:.4f}")
        with col_grpo4:
            buffer_size = metrics.get("buffer_size", 0)
            st.metric("Buffer Size", f"{buffer_size}")
        
        # Графики для GRPO
        if metrics.get("reward_history") and len(metrics["reward_history"]) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Reward chart
                fig_reward = go.Figure()
                steps = metrics.get("steps_history", list(range(len(metrics["reward_history"]))))
                fig_reward.add_trace(go.Scatter(
                    x=steps,
                    y=metrics["reward_history"],
                    mode='lines',
                    name='Reward',
                    line=dict(color='#10b981', width=2)
                ))
                # Добавляем скользящее среднее
                if len(metrics["reward_history"]) > 10:
                    import pandas as pd
                    df_reward = pd.DataFrame({"reward": metrics["reward_history"]})
                    df_reward["reward_smooth"] = df_reward["reward"].rolling(window=min(10, len(df_reward)//4), min_periods=1).mean()
                    fig_reward.add_trace(go.Scatter(
                        x=steps,
                        y=df_reward["reward_smooth"].tolist(),
                        mode='lines',
                        name='Reward (smooth)',
                        line=dict(color='#34d399', width=2, dash='dash')
                    ))
                fig_reward.update_layout(
                    title="Reward Curve",
                    xaxis_title="Step",
                    yaxis_title="Reward",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_reward, key=f"reward_chart_{rid}")
            
            with col2:
                # Loss chart для GRPO
                if metrics.get("loss_history") and len(metrics["loss_history"]) > 0:
                    fig_loss = go.Figure()
                    steps = metrics.get("steps_history", list(range(len(metrics["loss_history"]))))
                    fig_loss.add_trace(go.Scatter(
                        x=steps,
                        y=metrics["loss_history"],
                        mode='lines',
                        name='GRPO Loss',
                        line=dict(color='#e94560', width=2)
                    ))
                    if metrics.get("kl_history") and len(metrics["kl_history"]) > 0:
                        fig_loss.add_trace(go.Scatter(
                            x=steps,
                            y=metrics["kl_history"],
                            mode='lines',
                            name='KL Divergence',
                            line=dict(color='#f59e0b', width=2, dash='dash')
                        ))
                    fig_loss.update_layout(
                        title="Loss & KL Divergence",
                        xaxis_title="Step",
                        yaxis_title="Loss / KL",
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig_loss, key=f"grpo_loss_chart_{rid}")
                else:
                    st.info(t("info.waiting_loss"))
        
        # Окошко с семплами
        st.markdown("---")
        st.subheader(f"📝 {t('grpo.samples')}")
        
        # Загружаем семплы из файла если есть
        run_id = st.session_state.get("current_run_id")
        if run_id and run_id != "active":
            run_dir = RUNS_DIR / run_id
            
            # Пробуем найти samples.jsonl в run_dir или в output_dir
            samples_file = None
            config_path = run_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                        output_dir = config.get("output_dir", "")
                        if output_dir:
                            samples_file = Path(output_dir) / "samples.jsonl"
                except:
                    pass
            
            # Fallback: ищем в run_dir
            if not samples_file or not samples_file.exists():
                samples_file = run_dir / "samples.jsonl"
            
            samples_data = []
            if samples_file.exists():
                try:
                    with open(samples_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                samples_data.append(json.loads(line))
                except Exception as e:
                    st.warning(f"Не удалось загрузить семплы: {e}")
            
            if samples_data:
                # Показываем последние N семплов
                num_samples = st.slider("Количество семплов", 1, min(10, len(samples_data)), 3)
                recent_samples = samples_data[-num_samples:]
                
                for idx, sample in enumerate(reversed(recent_samples)):
                    with st.expander(f"Семпл {len(samples_data) - idx} (Step {sample.get('step', '?')})", expanded=(idx == 0)):
                        prompt = sample.get("prompt", "")
                        reference = sample.get("reference_answer", "")
                        completions = sample.get("completions", [])
                        rewards = sample.get("rewards", [])
                        
                        # Показываем полный промпт+ответ для первого семпла (чтобы видеть что модель видит)
                        if idx == 0 and completions:
                            st.markdown("**🔍 Полный промпт + ответ модели (что видит модель):**")
                            st.caption("Это полный текст который модель видит при генерации, включая системный промпт с инструкциями о тегах")
                            
                            # Используем full_texts если есть, иначе конкатенируем
                            full_texts = sample.get("full_texts", [])
                            if not full_texts:
                                full_texts = [prompt + comp for comp in completions]
                            
                            # Показываем лучший ответ (с максимальным reward)
                            if rewards:
                                best_completion_idx = max(range(len(rewards)), key=lambda i: rewards[i])
                                best_reward = rewards[best_completion_idx]
                                best_full_text = full_texts[best_completion_idx] if best_completion_idx < len(full_texts) else prompt + completions[best_completion_idx]
                                
                                # Выделяем системный промпт если он есть
                                system_prompt_start = best_full_text.find("system") if "system" in best_full_text.lower() else -1
                                if system_prompt_start == -1:
                                    # Ищем начало инструкций
                                    for marker in ["<|im_start|>", "A conversation", "Отвечай строго"]:
                                        if marker in best_full_text:
                                            system_prompt_start = best_full_text.find(marker)
                                            break
                                
                                st.code(best_full_text, language=None)
                                st.caption(f"✅ Лучший ответ (reward={best_reward:.4f})")
                                
                                # Также показываем худший для сравнения
                                worst_completion_idx = min(range(len(rewards)), key=lambda i: rewards[i])
                                worst_reward = rewards[worst_completion_idx]
                                if worst_reward < best_reward:
                                    with st.expander(f"📉 Худший ответ (reward={worst_reward:.4f})", expanded=False):
                                        worst_full_text = full_texts[worst_completion_idx] if worst_completion_idx < len(full_texts) else prompt + completions[worst_completion_idx]
                                        st.code(worst_full_text, language=None)
                                        st.caption("Сравните с лучшим ответом выше - видно ли инструкции о тегах в промпте?")
                            else:
                                # Если нет rewards, показываем первый
                                st.code(full_texts[0] if full_texts else prompt + completions[0], language=None)
                            
                            st.markdown("---")
                        
                        # Текущее отображение: промпт и ответы отдельно
                        col_s1, col_s2 = st.columns([1, 1])
                        
                        with col_s1:
                            st.markdown("**📥 Промпт:**")
                            st.code(prompt, language=None)
                        
                        with col_s2:
                            st.markdown("**✅ Эталонный ответ:**")
                            st.code(reference, language=None)
                        
                        st.markdown("**🤖 Ответы модели:**")
                        
                        if completions:
                            for i, (completion, reward) in enumerate(zip(completions, rewards)):
                                reward_color = "🟢" if reward > 0.5 else "🟡" if reward > 0 else "🔴"
                                with st.container():
                                    st.markdown(f"{reward_color} **Ответ {i+1}** (Reward: {reward:.4f})")
                                    st.code(completion[:500] + ("..." if len(completion) > 500 else ""), language=None)
                        else:
                            st.info(t("info.waiting_generations"))
            else:
                st.info(t("info.samples_will_appear"))
        
        st.markdown("---")
    
    # Обычные графики для других стадий
    elif metrics.get("loss_history"):
        # Генерируем steps_history если его нет
        steps_history = metrics.get("steps_history", list(range(1, len(metrics["loss_history"]) + 1)))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss chart
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=steps_history,
                y=metrics["loss_history"],
                mode='lines',
                name='Train Loss',
                line=dict(color='#e94560', width=2)
            ))
            if metrics.get("val_loss_history"):
                fig_loss.add_trace(go.Scatter(
                    x=metrics["val_steps_history"],
                    y=metrics["val_loss_history"],
                    mode='lines',
                    name='Val Loss',
                    line=dict(width=2, dash="dash", color='#60a5fa')
                ))
            fig_loss.update_layout(
                title="Training & Validation Loss",
                xaxis_title="Step",
                yaxis_title="Loss",
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_loss, key=f"loss_chart_{rid}")
        
        with col2:
            # LR chart
            lr_history = metrics.get("lr_history", [])
            if lr_history:
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    x=steps_history[:len(lr_history)],
                    y=lr_history,
                    mode='lines',
                    name='LR',
                    line=dict(color='#60a5fa', width=2)
                ))
                fig_lr.update_layout(
                    title="Learning Rate Schedule",
                    xaxis_title="Step",
                    yaxis_title="LR",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_lr, key=f"lr_chart_{rid}")
        
        # GRPO специфичные графики: Reward и KL
        if metrics.get("reward_history") or metrics.get("kl_history"):
            col3, col4 = st.columns(2)
            
            with col3:
                reward_history = metrics.get("reward_history", [])
                if reward_history:
                    fig_reward = go.Figure()
                    fig_reward.add_trace(go.Scatter(
                        x=steps_history[:len(reward_history)],
                        y=reward_history,
                        mode='lines',
                        name='Reward',
                        line=dict(color='#10b981', width=2)
                    ))
                    fig_reward.update_layout(
                        title="🎯 Reward (GRPO)",
                        xaxis_title="Step",
                        yaxis_title="Reward",
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig_reward, key=f"reward_chart_{rid}")
            
            with col4:
                kl_history = metrics.get("kl_history", [])
                if kl_history:
                    fig_kl = go.Figure()
                    fig_kl.add_trace(go.Scatter(
                        x=steps_history[:len(kl_history)],
                        y=kl_history,
                        mode='lines',
                        name='KL Divergence',
                        line=dict(color='#f59e0b', width=2)
                    ))
                    fig_kl.update_layout(
                        title="📊 KL Divergence (GRPO)",
                        xaxis_title="Step",
                        yaxis_title="KL",
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig_kl, key=f"kl_chart_{rid}")
    
    # Checkpoints
    if metrics.get("checkpoints"):
        with st.expander("📦 Checkpoints"):
            for ckpt in metrics["checkpoints"]:
                ckpt_loss = ckpt.get("loss")
                if ckpt_loss is not None:
                    st.text(f"Step {ckpt['step']}: Loss {ckpt_loss:.4f} | {ckpt['path']}")
                else:
                    st.text(f"Step {ckpt['step']}: {ckpt['path']}")
    
    # Пример сформированного промпта (для SFT)
    sample_prompt = metrics.get("sample_prompt")
    if sample_prompt:
        # Определяем тип датасета по наличию stage в метриках
        stage = metrics.get("stage", "pretrain")
        if stage == "sft":
            title = "📝 Пример сформированного промпта (SFT)"
            caption = "Это пример того, как выглядит промпт, который видит модель во время обучения:"
            tip = "💡 Модель учится генерировать текст после тега ассистента в том же формате"
        else:
            title = "📝 Пример текста из датасета (Pretrain)"
            caption = "Это пример текста, который видит модель во время обучения:"
            tip = "💡 Модель учится предсказывать следующий токен в тексте"
        
        with st.expander(title, expanded=True):
            st.caption(caption)
            st.code(sample_prompt, language=None)
            st.caption(tip)
    
    # GPU статистика
    gpu_stats = metrics.get("gpu_stats", [])
    if gpu_stats:
        st.subheader(f"🖥️ {t('metrics.gpu_load')}")
        
        cols = st.columns(len(gpu_stats))
        for i, (col, gpu) in enumerate(zip(cols, gpu_stats)):
            with col:
                st.markdown(f"**GPU {gpu['id']}**")
                
                # Memory bar
                mem_percent = gpu.get('memory_percent', 0)
                st.progress(min(mem_percent / 100, 1.0), text=f"VRAM: {gpu['memory_used_gb']:.1f} / {gpu['memory_total_gb']:.1f} GB ({mem_percent:.0f}%)")
                
                # Utilization
                util = gpu.get('utilization')
                if util is not None:
                    st.progress(min(util / 100, 1.0), text=f"Загрузка: {util}%")
                else:
                    st.caption("Загрузка: N/A")
    
    # Error
    if metrics.get("error"):
        st.error(f"❌ {t('error.training')}")
        with st.expander(f"{t('error.traceback')}", expanded=True):
            st.code(metrics['error'], language="python")
    
    # Логи процесса
    if st.session_state.current_run_id:
        run_dir = RUNS_DIR / st.session_state.current_run_id
        stderr_path = run_dir / "stderr.log"
        stdout_path = run_dir / "stdout.log"
        
        with st.expander(f"📋 {t('training.logs')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("stdout (последние 500 строк)")
                if stdout_path.exists():
                    with open(stdout_path) as f:
                        lines = f.readlines()
                        content = "".join(lines[-500:])
                        st.code(content if content else "(пусто)", language=None)
            
            with col2:
                st.caption("stderr (последние 500 строк)")
                if stderr_path.exists():
                    with open(stderr_path) as f:
                        lines = f.readlines()
                        content = "".join(lines[-500:])
                        st.code(content if content else "(пусто)", language=None)



def download_hf_dataset(repo_id, subset, split, limit_type, limit_val, limit_bytes, save_path, filters=None):
    """Функция скачивания и сохранения датасета."""
    try:
        status_text = f"Начинаем скачивание: {repo_id}..."
        st.toast(status_text)
        print(status_text)
        
        # Параметры stream=True чтобы не качать все в память
        # Для многих датасетов "default" может ломать load_dataset - передаём None
        subset_arg = None if (not subset or subset.strip() == "" or subset.lower() == "default") else subset
        
        ds = load_dataset(repo_id, subset_arg, split=split, streaming=True)
        
        # Создаем файл
        save_path = DATASET_DIR / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        current_bytes = 0
        
        with open(save_path, "w", encoding="utf-8") as f:
            for item in ds:
                # Применение фильтров
                if filters:
                    # Фильтр по score
                    if "score_col" in filters and "min_score" in filters:
                         col = filters["score_col"]
                         min_s = filters["min_score"]
                         # Проверяем наличие колонки и типа
                         if col in item and item[col] is not None:
                             try:
                                 val = float(item[col])
                                 if val < min_s:
                                     continue
                             except ValueError:
                                 pass
                    
                    # Фильтр по языку
                    if "lang_col" in filters and "target_lang" in filters:
                         col = filters["lang_col"]
                         target = filters["target_lang"]
                         if col in item and item[col] is not None:
                             val = str(item[col])
                             if target.lower() not in val.lower():
                                 continue
                
                # Сохраняем как JSONL
                line = json.dumps(item, ensure_ascii=False) + "\n"
                line_bytes = len(line.encode('utf-8'))
                
                # Check limits
                if limit_type == t("data.limit_rows") and count >= limit_val:
                    break
                if limit_type == t("data.limit_gb") and (current_bytes + line_bytes) > limit_bytes:
                    break
                    
                f.write(line)
                count += 1
                current_bytes += line_bytes
                
                if count % 1000 == 0:
                    print(f"Downloaded {count} lines, {current_bytes / 1024**2:.2f} MB")

        st.success(f"{t('success.saved')} {count} rows ({current_bytes / 1024**2:.2f} MB) → {save_path}")
        return True

    except Exception as e:
        st.error(f"{t('error.generic')}: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def render_data_manager(stage: str = "pretrain"):
    """Вкладка управления данными.
    
    Args:
        stage: Текущий режим ('pretrain', 'sft', 'grpo', 'continual_pretrain')
    """
    st.header(f"💾 {t('data.manager')}")
    
    col_upload, col_list = st.columns([1, 2])
    
    with col_upload:
        # Секция 1: Загрузка локальных файлов
        with st.expander(f"📤 {t('data.upload_local')}", expanded=False):
            uploaded_files = st.file_uploader(
                "Перетащите файлы сюда", 
                type=["jsonl", "txt"],  # ВАЖНО: не включаем .json, т.к. это обычно массив, а не JSONL 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button(f"📥 {t('button.save_files')}"):
                    for uploaded_file in uploaded_files:
                        save_path = DATASET_DIR / uploaded_file.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.toast(f"Файл {uploaded_file.name} сохранён!", icon="✅")
                    time.sleep(1)
                    st.rerun()

        # Секция 2: Загрузка с HuggingFace
        st.subheader(f"🤗 {t('data.download_hf')}")
        
        # Разные пресеты для разных режимов
        if stage == "grpo":
            # GRPO — Reasoning датасеты
            presets = {
                # English
                "🧠 GSM8K (math, EN)": ("gsm8k", "main", "train"),
                "🧠 OpenR1-Math-220k": ("open-r1/OpenR1-Math-220k", "default", "train"),
                "🧠 ARC-Challenge (EN)": ("allenai/ai2_arc", "ARC-Challenge", "train"),
                "🧠 CommonsenseQA (EN)": ("tau/commonsense_qa", "default", "train"),
                "🧠 HellaSwag (EN)": ("Rowan/hellaswag", "default", "train"),
                "🧠 TriviaQA (EN)": ("trivia_qa", "rc", "train"),
                "🧠 PIQA (EN)": ("piqa", "default", "train"),
                "🧠 WinoGrande (EN)": ("winogrande", "winogrande_xl", "train"),
                # Russian
                "🧠 GSM8K-RU (math, RU)": ("d0rj/gsm8k-ru", "default", "train"),
                "🧠 Gromov-MAX (math, RU)": ("attn-signs/gromov-max", "default", "train"),
                "🧠 MGSM-RU (multilingual)": ("juletxara/mgsm", "ru", "train"),
                # Ручной ввод
                "📝 Ввести вручную...": (None, None, None),
            }
            st.caption("🧠 **Reasoning датасеты** для GRPO тренировки")
        elif stage == "sft":
            # SFT — Instruction-following датасеты
            presets = {
                # Russian
                "🔵 OpenOrca-ru": ("d0rj/OpenOrca-ru", "default", "train"),
                "🔵 ru-instruct": ("d0rj/ru-instruct", "default", "train"),
                "🔵 Gromov-MAX": ("attn-signs/gromov-max", "default", "train"),
                "🔵 GrandMaster-PRO-MAX": ("Vikhrmodels/GrandMaster-PRO-MAX", "default", "train"),
                # English
                "🔵 OpenOrca (EN)": ("Open-Orca/OpenOrca", "default", "train"),
                "🔵 Alpaca-GPT4 (EN)": ("vicgalle/alpaca-gpt4", "default", "train"),
                "🔵 ShareGPT (EN)": ("anon8231489123/ShareGPT_Vicuna_unfiltered", "default", "train"),
                "🔵 Dolly-15k (EN)": ("databricks/databricks-dolly-15k", "default", "train"),
                "🔵 FLAN (EN)": ("Muennighoff/flan", "default", "train"),
                # Ручной ввод
                "📝 Ввести вручную...": (None, None, None),
            }
            st.caption("🔵 **Instruction-following датасеты** для SFT тренировки")
        else:
            # Pretrain / Continual Pretrain — Large text corpora
            presets = {
                # Russian
                "🟢 FineWeb-2 (Russian)": ("HuggingFaceFW/fineweb-2", "rus_Cyrl", "train"),
                "🟢 MC4-ru (Russian web)": ("mc4", "ru", "train"),
                "🟢 Wikipedia-ru": ("graelo/wikipedia", "20230601.ru", "train"),
                "🟢 Taiga Corpus (RU)": ("IlyaGusev/taiga_ru", "default", "train"),
                # English
                "🟢 FineWeb-Edu (Educational)": ("HuggingFaceFW/fineweb-edu", "default", "train"),
                "🟢 Wikitext-103": ("wikitext", "wikitext-103-v1", "train"),
                "🟢 The Pile (subset)": ("EleutherAI/pile", "default", "train"),
                "🟢 C4 (EN)": ("allenai/c4", "en", "train"),
                "🟢 RedPajama-v2": ("togethercomputer/RedPajama-Data-V2", "default", "train"),
                "🟢 SlimPajama": ("cerebras/SlimPajama-627B", "default", "train"),
                # Ручной ввод
                "📝 Ввести вручную...": (None, None, None),
            }
            st.caption(f"🟢 **Text corpora** {t('data.for_pretrain')}")
        
        # Инициализация дефолтных значений (FineWeb-2 Russian)
        if "hf_repo_id_input" not in st.session_state:
            st.session_state.hf_repo_id_input = "HuggingFaceFW/fineweb-2"
        if "hf_subset_default" not in st.session_state:
            st.session_state.hf_subset_default = "rus_Cyrl"
        if "hf_split_default" not in st.session_state:
            st.session_state.hf_split_default = "train"
        
        # Предзаполняем кэш для дефолтного пресета (FineWeb-2)
        # чтобы пользователь мог сразу скачивать без нажатия "Проверить"
        if "ds_repo_info" not in st.session_state:
            st.session_state.ds_repo_info = {}
        
        default_repo = "HuggingFaceFW/fineweb-2"
        if default_repo not in st.session_state.ds_repo_info:
            st.session_state.ds_repo_info[default_repo] = {
                "configs": ["rus_Cyrl"],  # Дефолтный язык
                "splits": ["train", "test"],  # Известные splits
                "features": {},  # Заполнится при проверке
                "selected_config": "rus_Cyrl"
            }
        
        def on_preset_change():
            """Callback для обновления всех полей при выборе пресета."""
            sel = st.session_state.dataset_preset_selector
            preset_data = presets.get(sel)
            if preset_data and preset_data[0]:
                st.session_state.hf_repo_id_input = preset_data[0]
                st.session_state.hf_subset_default = preset_data[1]
                st.session_state.hf_split_default = preset_data[2]
                
                # Сбрасываем selectbox чтобы применились новые дефолты
                if "hf_split_select" in st.session_state:
                    del st.session_state.hf_split_select
                if "hf_subset_select" in st.session_state:
                    del st.session_state.hf_subset_select

        # Селектор пресетов (по умолчанию FineWeb-2 Russian)
        st.selectbox(
            f"📚 {t('data.popular_datasets')}",
            options=list(presets.keys()),
            index=0,  # FineWeb-2 Russian первый
            key="dataset_preset_selector",
            on_change=on_preset_change,
            help="Выберите готовый датасет — все поля заполнятся автоматически"
        )

        repo_id = st.text_input("Репозиторий (ID)", key="hf_repo_id_input")
        
        # Кнопка проверки репозитория
        if st.button(f"🔍 {t('button.check_repo')}"):
            try:
                with st.spinner(f"Анализируем {repo_id}..."):
                    # 1. Получаем конфиги
                    configs = get_dataset_config_names(repo_id)
                    
                    # 2. Выбираем конфиг для получения splits/features
                    # Приоритет: дефолтный (rus_Cyrl) > первый в списке
                    default_subset = st.session_state.hf_subset_default
                    if default_subset in configs:
                        selected_config = default_subset
                    else:
                        selected_config = configs[0] if configs else None
                    
                    splits = []
                    features_info = {}
                    
                    if selected_config:
                        splits = get_dataset_split_names(repo_id, selected_config)
                        # 3. Пытаемся получить информацию о структуре (features)
                        try:
                            ds_builder = load_dataset_builder(repo_id, selected_config)
                            if ds_builder.info.features:
                                features_info = ds_builder.info.features
                        except Exception as e:
                            print(f"Could not load features: {e}")

                    st.session_state.ds_repo_info[repo_id] = {
                        "configs": configs,
                        "splits": splits,
                        "features": features_info,
                        "selected_config": selected_config  # Запоминаем для какого конфига splits
                    }
                    
                    # ВАЖНО: Сбрасываем выбор виджетов чтобы применились новые данные
                    # Устанавливаем дефолтные значения
                    if "hf_split_select" in st.session_state:
                        del st.session_state.hf_split_select
                    if "hf_subset_select" in st.session_state:
                        del st.session_state.hf_subset_select
                    
                    st.success(f"✅ {t('success.found_configs')} {len(configs)}, splits: {splits}")
            except Exception as e:
                st.error(f"{t('error.fetch_info')}: {e}")

        # Работаем с кэшированной информацией
        repo_info = st.session_state.ds_repo_info.get(repo_id, {})
        available_configs = repo_info.get("configs", [])
        available_splits = repo_info.get("splits", [])
        features = repo_info.get("features", {})
        
        if available_configs:
            # Если есть дефолтный subset, пытаемся найти его в списке
            default_idx = 0
            if st.session_state.hf_subset_default in available_configs:
                default_idx = available_configs.index(st.session_state.hf_subset_default)
            subset = st.selectbox("Subset (конфиг)", available_configs, index=default_idx, key="hf_subset_select")
        else:
            subset = st.text_input("Subset (конфиг)", st.session_state.hf_subset_default, key="hf_subset_input")
        
        if available_splits:
            default_idx = 0
            if st.session_state.hf_split_default in available_splits:
                default_idx = available_splits.index(st.session_state.hf_split_default)
            split = st.selectbox("Split", available_splits, index=default_idx, key="hf_split_select")
        else:
            split = st.text_input("Split", st.session_state.hf_split_default, key="hf_split_input")

        # --- УМНЫЕ ФИЛЬТРЫ ---
        with st.expander(f"🛠️ {t('data.filters_limits')}", expanded=True):
            # Лимиты (всегда доступны)
            col_lim1, col_lim2 = st.columns(2)
            with col_lim1:
                limit_type = st.radio(t("data.limit_type"), [t("data.limit_gb"), t("data.limit_rows")], key="limit_type")
            
            with col_lim2:
                limit_val = 0
                limit_bytes = 0
                
                if limit_type == t("data.limit_rows"):
                    limit_val = st.number_input(t("data.row_count"), value=100000, step=10000, key="limit_val")
                else:
                    limit_gb = st.number_input(t("data.size_gb"), value=2.0, step=0.5, min_value=0.1, key="limit_gb")
                    limit_bytes = int(limit_gb * 1024**3)
            
            st.divider()
            
            # Фильтры на основе структуры (features)
            active_filters = {}
            
            if features:
                st.caption(f"🔍 {t('data.column_filters')}:")
                
                # Получаем список колонок и их типов
                # features это словарь {col_name: feature_info}
                # feature_info может быть строкой 'Value(dtype='string')' или объектом
                
                # 1. Фильтр числовой (Score/Quality)
                float_cols = []
                string_cols = []
                
                for col_name, feature_def in features.items():
                    # Пытаемся определить тип
                    dtype = getattr(feature_def, 'dtype', str(feature_def))
                    if 'float' in str(dtype):
                        float_cols.append(col_name)
                    elif 'string' in str(dtype):
                        string_cols.append(col_name)
                
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    if float_cols:
                        st.markdown("**Фильтр по значению (float)**")
                        selected_float_col = st.selectbox(t("common.select_column"), [f"({t('common.none')})"] + float_cols, key="sel_float_col")
                        if selected_float_col != "(нет)":
                            min_val = st.slider(f"Мин. значение {selected_float_col}", 0.0, 1.0, 0.0, key="val_float_col")
                            active_filters["score_col"] = selected_float_col
                            active_filters["min_score"] = min_val
                    else:
                        st.caption("Числовые колонки не найдены")

                with col_f2:
                    if string_cols:
                        st.markdown("**Фильтр по тексту (contains)**")
                        selected_str_col = st.selectbox(t("common.select_column"), [f"({t('common.none')})"] + string_cols, key="sel_str_col")
                        if selected_str_col != "(нет)":
                            target_str = st.text_input(f"Текст должен содержать:", key="val_str_col")
                            if target_str:
                                active_filters["lang_col"] = selected_str_col
                                active_filters["target_lang"] = target_str
                    else:
                        st.caption("Текстовые колонки не найдены")

            else:
                st.info(t("info.dataset_structure_not_loaded"))


        # Формируем дефолтное имя файла из repo_id
        # Например: "HuggingFaceFW/fineweb-2" → "fineweb-2.jsonl"
        # "d0rj/gsm8k-ru" → "gsm8k-ru.jsonl"
        def compute_default_filename():
            """Вычисляет имя файла на основе текущих значений repo_id и subset."""
            computed_name = "dataset.jsonl"
            if repo_id:
                # Берём часть после "/" (или всё если нет "/")
                name_part = repo_id.split("/")[-1] if "/" in repo_id else repo_id
                # Добавляем subset если он не default
                current_subset = st.session_state.get('hf_subset_select') or st.session_state.get('hf_subset_input') or st.session_state.get('hf_subset_default', '')
                if current_subset and current_subset not in ('default', 'main', ''):
                    name_part = f"{name_part}-{current_subset}"
                computed_name = f"{name_part}.jsonl"
            return computed_name
        
        computed_filename = compute_default_filename()
        
        # Создаём уникальный ключ на основе repo_id и subset
        # Используем этот key для виджета, чтобы он пересоздавался при изменении repo_id/subset
        current_subset = st.session_state.get('hf_subset_select') or st.session_state.get('hf_subset_input') or st.session_state.get('hf_subset_default', '')
        repo_subset_key = f"{repo_id}::{current_subset}"
        
        # Нормализуем ключ для использования в session_state (убираем спецсимволы)
        normalized_key = repo_subset_key.replace('/', '_').replace(':', '_').replace('-', '_')
        widget_key = f"save_filename_{normalized_key}"
        
        # Если это новый ключ (repo_id/subset изменились), используем вычисленное имя
        if widget_key not in st.session_state:
            st.session_state[widget_key] = computed_filename
        
        save_filename = st.text_input(
            t("data.filename_save"), 
            value=st.session_state.get(widget_key, computed_filename), 
            key=widget_key,
            help="Автоматически формируется из названия репозитория и subset. Можно изменить вручную."
        )
        
        # Кнопка скачивания с использованием callback
        def on_download_click(active_filters_map):
            # Получаем значения из session_state явно
            r_id = st.session_state.get('hf_repo_id_input')
            
            # Определяем subset и split
            if st.session_state.get('hf_subset_select'):
                sub = st.session_state.get('hf_subset_select')
            else:
                sub = st.session_state.get('hf_subset_input')
                
            if st.session_state.get('hf_split_select'):
                spl = st.session_state.get('hf_split_select')
            else:
                spl = st.session_state.get('hf_split_input')
                
            l_type = st.session_state.get('limit_type')
            l_val = st.session_state.get('limit_val', 0) or 0
            
            l_gb = st.session_state.get('limit_gb', 2.0)
            l_bytes = int(l_gb * 1024**3)

            # Получаем имя файла используя тот же динамический key
            current_subset_for_key = sub or st.session_state.get('hf_subset_default', '')
            repo_subset_key_for_download = f"{r_id}::{current_subset_for_key}"
            normalized_key_for_download = repo_subset_key_for_download.replace('/', '_').replace(':', '_').replace('-', '_')
            widget_key_for_download = f"save_filename_{normalized_key_for_download}"
            s_path = st.session_state.get(widget_key_for_download, "dataset.jsonl")
            
            # Передаём фильтры как есть (active_filters_map уже содержит нужные ключи)
            filters_to_pass = active_filters_map or None
            
            download_hf_dataset(r_id, sub, spl, l_type, l_val, l_bytes, s_path, filters=filters_to_pass)

        st.button(t("button.download_process"), on_click=on_download_click, args=(active_filters,))
    
    with col_list:
        st.subheader(t("data.available_datasets"))
        
        datasets = []
        if DATASET_DIR.exists():
            # JSONL / JSON
            for f in list(DATASET_DIR.glob("*.jsonl")) + list(DATASET_DIR.glob("*.json")):
                size_mb = f.stat().st_size / (1024 * 1024)
                datasets.append({
                    "name": f.name,
                    "type": "JSONL/JSON",
                    "size_mb": size_mb,
                    "path": f
                })
            # TXT
            for f in DATASET_DIR.glob("*.txt"):
                size_mb = f.stat().st_size / (1024 * 1024)
                datasets.append({
                    "name": f.name,
                    "type": "Text",
                    "size_mb": size_mb,
                    "path": f
                })
        
        if not datasets:
            st.info(t("info.no_datasets"))
        else:
            # Отображаем список
            for ds in datasets:
                with st.expander(f"📄 {ds['name']} ({ds['size_mb']:.1f} MB)"):
                    st.caption(f"Тип: {ds['type']}")
                    
                    # Preview
                    try:
                        with open(ds['path'], "r", encoding="utf-8") as f:
                            head = []
                            for i, line in enumerate(f):
                                if i >= 5:
                                    break
                                head.append(line.strip())
                        st.markdown("**Preview (первые 5 строк):**")
                        if head:
                            st.code("\n".join(head), language="json" if "JSON" in ds['type'] else "text")
                        else:
                                    st.info(t("error.file_empty"))
                        
                        col_del, col_info = st.columns([1, 4])
                        with col_del:
                            if st.button(f"🗑️ {t('button.delete')}", key=f"del_{ds['name']}"):
                                ds['path'].unlink()
                                st.toast(f"Файл {ds['name']} удалён", icon="🗑️")
                                time.sleep(1)
                                st.rerun()
                    except Exception as e:
                        st.error(t("error.file_read", error=str(e)))


def download_hf_model(repo_id: str, save_name: str, revision: str = "main"):
    """
    Скачивает модель с HuggingFace и сохраняет локально в MODELS_DIR.
    """
    from huggingface_hub import snapshot_download
    
    save_path = MODELS_DIR / save_name
    
    # Проверяем, не существует ли уже
    if save_path.exists():
        st.warning(f"⚠️ {t('warning.model_exists', name=save_name)}")
        return False
    
    try:
        with st.spinner(f"⏳ Скачиваем {repo_id}..."):
            # Создаем прогресс-бар
            progress_bar = st.progress(0, text="Инициализация...")
            status_text = st.empty()
            
            # Скачиваем модель
            status_text.text(f"Скачиваем файлы модели из {repo_id}...")
            progress_bar.progress(10, text="Загрузка файлов...")
            
            # snapshot_download скачивает всю модель
            local_path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(save_path),
                local_dir_use_symlinks=False,  # Копируем файлы, не симлинки
                ignore_patterns=["*.md", "*.txt", "*.gitattributes", ".git*"],  # Пропускаем ненужное
            )
            
            progress_bar.progress(90, text="Проверка...")
            
            # Проверяем, что скачалось
            config_file = save_path / "config.json"
            if not config_file.exists():
                st.error(f"❌ {t('error.no_config')}")
                return False
            
            # Читаем конфиг для отображения информации
            import json
            with open(config_file) as f:
                model_config = json.load(f)
            
            # Получаем информацию о модели
            model_type = model_config.get("model_type", "unknown")
            hidden_size = model_config.get("hidden_size", "?")
            num_layers = model_config.get("num_hidden_layers", model_config.get("n_layer", "?"))
            vocab_size = model_config.get("vocab_size", "?")
            
            progress_bar.progress(100, text="Готово!")
            status_text.empty()
            
            st.success(f"""✅ Модель скачана!
- **Путь:** `{save_path}`
- **Тип:** {model_type}
- **Hidden:** {hidden_size}, **Layers:** {num_layers}, **Vocab:** {vocab_size}
""")
            return True
            
    except Exception as e:
        st.error(f"❌ {t('error.download')}: {e}")
        import traceback
        print(traceback.format_exc())
        # Удаляем частично скачанное
        if save_path.exists():
            import shutil
            shutil.rmtree(save_path, ignore_errors=True)
        return False


def render_model_manager():
    """Вкладка управления моделями (скачивание с HuggingFace)."""
    st.header(f"🤖 {t('models.manager')}")
    
    col_download, col_list = st.columns([1, 2])
    
    with col_download:
        st.subheader(f"🤗 {t('data.download_hf')}")
        
        # Пресеты популярных небольших моделей для continual pretraining / SFT
        model_presets = {
            "🔥 SmolLM2-135M (135M params)": ("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M"),
            "🔥 SmolLM2-360M (360M params)": ("HuggingFaceTB/SmolLM2-360M", "SmolLM2-360M"),
            "🔥 SmolLM2-1.7B (1.7B params)": ("HuggingFaceTB/SmolLM2-1.7B", "SmolLM2-1.7B"),
            "🦙 TinyLlama-1.1B (1.1B params)": ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "TinyLlama-1.1B"),
            "🐍 Pythia-70M (70M params)": ("EleutherAI/pythia-70m", "Pythia-70M"),
            "🐍 Pythia-160M (160M params)": ("EleutherAI/pythia-160m", "Pythia-160M"),
            "🐍 Pythia-410M (410M params)": ("EleutherAI/pythia-410m", "Pythia-410M"),
            "🐍 Pythia-1B (1B params)": ("EleutherAI/pythia-1b", "Pythia-1B"),
            "🤖 GPT-2 Small (124M params)": ("openai-community/gpt2", "GPT2-Small"),
            "🤖 GPT-2 Medium (355M params)": ("openai-community/gpt2-medium", "GPT2-Medium"),
            "🦊 Qwen2.5-0.5B (0.5B params)": ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
            "🦊 Qwen2.5-1.5B (1.5B params)": ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
            "🇷🇺 ruGPT3-Small (125M, Russian)": ("ai-forever/rugpt3small_based_on_gpt2", "ruGPT3-Small"),
            "📝 Ввести вручную...": (None, None),
        }
        
        # Инициализация
        if "model_repo_id" not in st.session_state:
            st.session_state.model_repo_id = "HuggingFaceTB/SmolLM2-135M"
        if "model_save_name" not in st.session_state:
            st.session_state.model_save_name = "SmolLM2-135M"
        
        def on_model_preset_change():
            sel = st.session_state.model_preset_selector
            preset_data = model_presets.get(sel)
            if preset_data and preset_data[0]:
                st.session_state.model_repo_id = preset_data[0]
                st.session_state.model_save_name = preset_data[1]
        
        st.selectbox(
            f"📚 {t('models.popular')}",
            options=list(model_presets.keys()),
            index=0,
            key="model_preset_selector",
            on_change=on_model_preset_change,
            help="Выберите модель — repo_id заполнится автоматически"
        )
        
        repo_id = st.text_input(t("models.repo_id"), key="model_repo_id")
        save_name = st.text_input(t("models.save_name"), key="model_save_name", 
                                   help="Папка в models/, куда будет сохранена модель")
        
        # Model info
        st.markdown(f"""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 12px; border-radius: 8px; margin: 10px 0;
            border: 1px solid #0f3460; color: #e8e8e8;">
<b style="color: #4fc3f7;">💡 {t('models.recommendations')}:</b><br>
• <b style="color: #81d4fa;">SmolLM2</b> — {t('models.rec_smollm2')}<br>
• <b style="color: #81d4fa;">Pythia</b> — {t('models.rec_pythia')}<br>
• <b style="color: #81d4fa;">TinyLlama</b> — {t('models.rec_tinyllama')}<br>
• <b style="color: #81d4fa;">Qwen2.5</b> — {t('models.rec_qwen')}
</div>
""", unsafe_allow_html=True)
        
        # Оценка размера
        size_estimates = {
            "70m": "~150 MB", "135m": "~300 MB", "160m": "~350 MB",
            "360m": "~800 MB", "410m": "~900 MB", "0.5b": "~1 GB",
            "1b": "~2 GB", "1.1b": "~2.5 GB", "1.5b": "~3 GB", "1.7b": "~3.5 GB",
            "124m": "~500 MB", "355m": "~1.5 GB", "125m": "~500 MB",
        }
        
        estimated_size = t("common.unknown")
        repo_lower = repo_id.lower()
        for size_key, size_val in size_estimates.items():
            if size_key in repo_lower:
                estimated_size = size_val
                break
        
        st.caption(f"📦 {t('models.estimated_size')}: **{estimated_size}**")
        
        if st.button(f"⬇️ {t('button.download_model')}", type="primary"):
            if not repo_id or not save_name:
                st.error(t("error.missing_repo_id"))
            else:
                success = download_hf_model(repo_id, save_name)
                if success:
                    time.sleep(1)
                    st.rerun()
    
    with col_list:
        st.subheader(f"📁 {t('model.downloaded')}")
        
        models = []
        if MODELS_DIR.exists():
            for model_dir in MODELS_DIR.iterdir():
                if model_dir.is_dir():
                    config_path = model_dir / "config.json"
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path) as f:
                                cfg = json.load(f)
                            
                            # Размер папки
                            total_size = sum(
                                f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                            )
                            size_gb = total_size / (1024**3)
                            
                            models.append({
                                "name": model_dir.name,
                                "path": model_dir,
                                "model_type": cfg.get("model_type", "unknown"),
                                "hidden_size": cfg.get("hidden_size", "?"),
                                "num_layers": cfg.get("num_hidden_layers", cfg.get("n_layer", "?")),
                                "vocab_size": cfg.get("vocab_size", "?"),
                                "size_gb": size_gb,
                            })
                        except Exception:
                            models.append({
                                "name": model_dir.name,
                                "path": model_dir,
                                "model_type": "?",
                                "hidden_size": "?",
                                "num_layers": "?",
                                "vocab_size": "?",
                                "size_gb": 0,
                            })
        
        if not models:
            st.info(t("info.no_downloaded_models"))
        else:
            for m in sorted(models, key=lambda x: x["name"]):
                with st.expander(f"🤖 {m['name']} ({m['size_gb']:.2f} GB)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
- **Тип:** `{m['model_type']}`
- **Hidden Size:** {m['hidden_size']}
- **Layers:** {m['num_layers']}
- **Vocab:** {m['vocab_size']}
""")
                    with col2:
                        st.caption(f"📂 Путь: `{m['path']}`")
                        
                        # Кнопка использования
                        if st.button(f"🚀 {t('button.use')}", key=f"use_{m['name']}", 
                                     help="Выбрать эту модель для Continual Pretrain / SFT"):
                            st.session_state.selected_base_model = str(m['path'])
                            st.toast(f"Модель {m['name']} выбрана! Перейдите в Запуск и выберите Continual Pretrain или SFT.", icon="✅")
                        
                        # Кнопка удаления
                        if st.button(f"🗑️ {t('button.delete')}", key=f"del_model_{m['name']}"):
                            import shutil
                            shutil.rmtree(m['path'])
                            st.toast(f"{t('models.deleted')}: {m['name']}", icon="🗑️")
                            time.sleep(1)
                            st.rerun()
        
        # Tip
        st.markdown("---")
        st.info(t("models.usage_tip"))


def _bytes_to_gb(x: int) -> float:
    return float(x) / (1024**3)


def _sum_tensor_bytes(obj) -> int:
    """
    Считает байты тензоров на CUDA внутри структуры (тензор/список/кортеж/словарь).
    """
    import torch

    total = 0
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        if obj.is_cuda:
            return int(obj.numel() * obj.element_size())
        return 0
    if isinstance(obj, dict):
        for v in obj.values():
            total += _sum_tensor_bytes(v)
        return total
    if isinstance(obj, (list, tuple)):
        for v in obj:
            total += _sum_tensor_bytes(v)
        return total
    return 0


def _estimate_memory_footprint(config, batch_size, distributed_mode="default", num_gpus=1):
    """
    Универсальная оценка VRAM для любой архитектуры модели.
    Использует архитектурные профили для точного расчета параметров и активаций.
    """
    from homellm.models.memory_estimator import estimate_memory_footprint
    return estimate_memory_footprint(config, batch_size, distributed_mode, num_gpus)


def _profile_memory_footprint_cuda(config, batch_size: int):
    """
    Точный (насколько возможно) замер по фактическим CUDA аллокациям:
    - делаем warmup шаг, чтобы AdamW проинициализировал state
    - затем меряем peak allocated/reserved на следующем шаге
    Возвращаем breakdown: Model+Optim (steady), Act (peak - steady), Buf (reserved - peak).
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")

    # Собираем HomeForCausalLM (для Blueprint режима лучше делать отдельно; пока профилируем базовую Home модель)
    from homellm.models.home_model import HomeConfig, HomeForCausalLM

    vocab_size = int(config.get("vocab_size", 50257))
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_layers"])
    n_heads = int(config["n_heads"])
    seq_len = int(config["seq_len"])

    mp = (config.get("mixed_precision") or "no").lower()
    if mp == "bf16":
        dtype = torch.bfloat16
    elif mp == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model_cfg = HomeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=n_heads,
        max_position_embeddings=seq_len,
        dropout=float(config.get("dropout", 0.0)),
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = HomeForCausalLM(model_cfg).to(device=device, dtype=dtype)
    model.train()
    if config.get("grad_checkpoint", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    opt_name = (config.get("optimizer") or "adamw").lower()
    lr = float(config.get("lr", 1e-3))
    wd = float(config.get("weight_decay", 0.01))
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def run_step():
        input_ids = torch.randint(0, vocab_size, (int(batch_size), seq_len), device=device)
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Warmup: инициализация optimizer state + прогрев allocator'а
    torch.cuda.synchronize()
    run_step()
    torch.cuda.synchronize()

    # Измерение
    torch.cuda.reset_peak_memory_stats(device)
    alloc_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)

    run_step()
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    alloc_after = torch.cuda.memory_allocated(device)

    # Tensor-based breakdown (приближает "что именно живёт", независимо от caching allocator)
    weights_bytes = 0
    for p in model.parameters():
        if p.is_cuda:
            weights_bytes += int(p.numel() * p.element_size())
    for b in model.buffers():
        if torch.is_tensor(b) and b.is_cuda:
            weights_bytes += int(b.numel() * b.element_size())

    opt_state_bytes = 0
    for st in optimizer.state.values():
        opt_state_bytes += _sum_tensor_bytes(st)

    # После zero_grad(set_to_none=True) градиенты не должны держать память
    grads_bytes = 0
    for p in model.parameters():
        if p.grad is not None and p.grad.is_cuda:
            grads_bytes += int(p.grad.numel() * p.grad.element_size())

    steady_alloc = alloc_after
    act_alloc = max(0, int(peak_alloc - steady_alloc))
    buf_alloc = max(0, int(peak_reserved - peak_alloc))

    total_peak = peak_reserved  # самый честный "сколько попросил у драйвера" на пике шага

    return {
        "method": "profile_cuda",
        "total_gb": round(_bytes_to_gb(total_peak), 2),
        "model_gb": round(_bytes_to_gb(steady_alloc), 2),
        "act_gb": round(_bytes_to_gb(act_alloc), 2),
        "buf_gb": round(_bytes_to_gb(buf_alloc), 2),
        "params": int(sum(p.numel() for p in model.parameters())),
        "detail": {
            "alloc_before_gb": round(_bytes_to_gb(alloc_before), 3),
            "reserved_before_gb": round(_bytes_to_gb(reserved_before), 3),
            "peak_alloc_gb": round(_bytes_to_gb(peak_alloc), 3),
            "peak_reserved_gb": round(_bytes_to_gb(peak_reserved), 3),
            "alloc_after_gb": round(_bytes_to_gb(alloc_after), 3),
            "tensor_weights_gb": round(_bytes_to_gb(weights_bytes), 3),
            "tensor_opt_state_gb": round(_bytes_to_gb(opt_state_bytes), 3),
            "tensor_grads_gb": round(_bytes_to_gb(grads_bytes), 3),
        },
        "notes": "Профилирование CUDA: warmup + измерение peak. 'Buf' = caching allocator (reserved - peak allocated).",
    }


def calculate_memory_footprint(config, batch_size, distributed_mode="default", num_gpus=1, *, method: str = "estimate"):
    """
    Возвращает оценку/замер VRAM для превью.
    method:
      - 'estimate': быстрая формула (fallback)
      - 'profile_cuda': реальный замер на текущей GPU (может быть медленно/может OOM)
    """
    try:
        if method == "profile_cuda":
            return _profile_memory_footprint_cuda(config, batch_size=int(batch_size))
        return _estimate_memory_footprint(config, batch_size=int(batch_size), distributed_mode=distributed_mode, num_gpus=num_gpus)
    except Exception as e:
        print(f"Error calculating VRAM ({method}): {e}")
        # Фолбэк на оценку
        out = _estimate_memory_footprint(config, batch_size=int(batch_size), distributed_mode=distributed_mode, num_gpus=num_gpus)
        out["notes"] = f"{out.get('notes','')} | profile error: {e}"
        return out


def render_quick_summary(model_config: dict, dataset_config: dict, distributed_config: dict, full_config: dict = None) -> bool:
    """
    Отображает быструю табличку с выбранными параметрами (модель, данные, режим).
    Возвращает True если все 3 параметра выбраны, иначе False.
    """
    # Получаем информацию о модели
    model_name = model_config.get("model_name_input", "Не выбрано")
    base_model_path = model_config.get("base_model_path")
    stage = model_config.get("stage", "pretrain")
    
    if base_model_path:
        model_display = f"{Path(base_model_path).name}"
    elif model_name and model_name != "Не выбрано":
        model_display = model_name
    else:
        model_display = "❌ Не выбрано"
    
    # Получаем информацию о данных
    # Для GRPO датасет может быть в full_config (grpo_dataset_path), session_state или выбирается в main area
    data_path = dataset_config.get("data_path")
    if not data_path and stage == "grpo":
        # Сначала проверяем selectbox напрямую через его key (самый надежный способ)
        selectbox_value = st.session_state.get("grpo_dataset_selectbox")
        if selectbox_value and not selectbox_value.startswith("--"):
            data_path = selectbox_value
        # Затем проверяем session_state (обновляется в render_grpo_main_config)
        if not data_path:
            data_path = st.session_state.get("grpo_dataset_path")
        # Затем проверяем full_config (обновляется после render_grpo_main_config)
        if not data_path and full_config:
            data_path = full_config.get("grpo_dataset_path")
            # Инициализируем session_state из full_config если там есть датасет
            if data_path and "grpo_dataset_path" not in st.session_state:
                st.session_state.grpo_dataset_path = data_path
    
    if data_path:
        data_display = Path(data_path).name
    elif stage == "grpo":
        data_display = "📝 Выберите в настройках ниже"
    else:
        data_display = "❌ Не выбрано"
    
    # Получаем информацию о режиме тренировки
    distributed_mode = distributed_config.get("distributed_mode", "default")
    mode_info = PARALLEL_TYPES.get(distributed_mode, PARALLEL_TYPES["default"])
    training_mode_display = mode_info.get("name", "Не выбрано")
    
    # Проверяем что все выбрано
    has_model = model_display != "❌ Не выбрано"
    # Для GRPO датасет может быть выбран позже в main area
    has_data = data_display != "❌ Не выбрано" and "Выберите в настройках ниже" not in data_display
    has_mode = distributed_mode != "default" or distributed_config.get("num_gpus", 1) > 0
    
    # Для GRPO датасет обязателен, но выбирается в main area (после render_quick_summary)
    # Проверяем через session_state или full_config
    if stage == "grpo":
        # Для GRPO датасет обязателен для запуска
        all_selected = has_model and has_mode and has_data
    else:
        all_selected = has_model and has_data and has_mode
    
    # Отображаем табличку
    st.markdown("""
    <style>
    .quick-summary {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        border: 2px solid #444;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .quick-summary-header {
        color: #ff6b6b;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .quick-summary-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    .quick-summary-item {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .quick-summary-number {
        font-size: 2rem;
        font-weight: 800;
        color: #ff6b6b;
        margin-bottom: 0.5rem;
    }
    .quick-summary-label {
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .quick-summary-value {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 600;
        word-break: break-word;
    }
    .quick-summary-check {
        color: #22c55e;
        font-size: 1.5rem;
        margin-top: 0.5rem;
    }
    .quick-summary-warning {
        color: #f59e0b;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Определяем иконки и цвета
    check_icon = "✅" if all_selected else "⚠️"
    status_color = "#22c55e" if all_selected else "#f59e0b"
    
    # Warning messages for missing items
    warn_model = f"⚠️ {t('warning.select_model')}"
    warn_data = f"⚠️ {t('warning.select_data')}"
    warn_mode = f"⚠️ {t('warning.select_mode')}"
    
    st.markdown(f"""
    <div class="quick-summary">
        <div class="quick-summary-header">
            {t('launch.config_title')}
        </div>
        <div class="quick-summary-grid">
            <div class="quick-summary-item">
                <div class="quick-summary-label">{t('launch.model')}</div>
                <div class="quick-summary-value">{model_display}</div>
                {"<div class='quick-summary-check'>✅</div>" if has_model else f"<div class='quick-summary-warning'>{warn_model}</div>"}
            </div>
            <div class="quick-summary-item">
                <div class="quick-summary-label">{t('launch.data')}</div>
                <div class="quick-summary-value">{data_display}</div>
                {"<div class='quick-summary-check'>✅</div>" if has_data else f"<div class='quick-summary-warning'>{warn_data}</div>"}
            </div>
            <div class="quick-summary-item">
                <div class="quick-summary-label">{t('launch.training_mode')}</div>
                <div class="quick-summary-value">{training_mode_display}</div>
                {"<div class='quick-summary-check'>✅</div>" if has_mode else f"<div class='quick-summary-warning'>{warn_mode}</div>"}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return all_selected


def render_model_preview(config: dict, distributed_config: dict = None):
    """Превью архитектуры модели и настроек параллелизма."""
    st.subheader(f"📐 {t('model.architecture')}")
    
    stage = config.get("stage", "pretrain")
    if stage == "sft":
        st.info(f"🔄 **{t('mode.sft')}**\n{t('model.base')}: `{Path(config.get('base_model_path') or 'Unknown').name}`")
    elif stage == "continual_pretrain":
        st.info(f"🔄 **{t('mode.continual_pretrain')}**\n{t('model.base')}: `{Path(config.get('base_model_path') or 'Unknown').name}`")
    elif stage == "grpo":
        st.info(f"🧠 **{t('mode.grpo')}**\n{t('model.base')}: `{Path(config.get('base_model_path') or 'Unknown').name}`")
    else:
        st.success(f"🏗️ **{t('mode.pretrain')}**")

    # Рассчитываем память
    # Нам нужен batch_size из конфига (это батч на девайс)
    batch_size = config.get("batch_size", 1)
    # Проверяем num_gpus и distributed_mode из обоих источников
    dist_mode = "default"
    n_gpus = 1
    if distributed_config:
        dist_mode = distributed_config.get("distributed_mode", "default")
        n_gpus = distributed_config.get("num_gpus", 1)
    # Также проверяем config напрямую (на случай если distributed_config не передан)
    if config.get("num_gpus"):
        n_gpus = int(config.get("num_gpus", 1))
    if config.get("distributed_mode"):
        dist_mode = config.get("distributed_mode", "default")

    mem_method = "estimate"
    # if torch.cuda.is_available():
    #     with st.expander("🧠 Память GPU: оценка vs точный замер", expanded=False):
    #         st.caption("Оценка — мгновенно, но приблизительно. Точный замер — запускает 2 train-step на GPU (warmup + измерение) и может быть медленным/может упасть по OOM.")
    #         do_profile = st.checkbox("Сделать точный замер на CUDA (2 шага)", value=False, key="profile_vram_cuda")
    #         if do_profile:
    #             mem_method = "profile_cuda"

    mem_info = calculate_memory_footprint(config, batch_size, dist_mode, n_gpus, method=mem_method)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hidden Size", config["hidden_size"])
        st.metric("Layers", config["num_layers"])
    
    with col2:
        st.metric("Attention Heads", config["n_heads"])
        st.metric("Head Dim", config["hidden_size"] // config["n_heads"])
    
    with col3:
        st.metric(t("model.params"), format_params(mem_info["params"]))
        
        # Цвет метрики в зависимости от размера (примерно для 24GB карты)
        val = mem_info["total_gb"]
        color = "normal"
        if val > 24: color = "off" # красный оттенок в дельте обычно
        
        title = "VRAM (Profile)" if mem_info.get("method") == "profile_cuda" else "VRAM (Estimate)"
        st.metric(
            title,
            f"{val:.1f} GB",
            delta=f"M: {mem_info['model_gb']} + A: {mem_info['act_gb']} GB",
            delta_color=color,
            help=(mem_info.get("notes") or "M: Model+Optim (steady)\nA: Activations/temporaries (peak - steady)\nBuf: caching allocator")
        )
    
    # Визуализация использования памяти
    if mem_info["total_gb"] > 0:
        st.caption("📊 Распределение памяти GPU:")
        
        # Создаем простой бар чарт через HTML/CSS для наглядности
        total = mem_info["total_gb"]
        p_model = (mem_info["model_gb"] / total) * 100
        p_act = (mem_info["act_gb"] / total) * 100
        buf_gb = float(mem_info.get("buf_gb", 0.0))
        p_buff = (buf_gb / total) * 100 if total > 0 else 0
        
        st.markdown(f"""
        <div style="display: flex; height: 20px; width: 100%; background: #333; border-radius: 4px; overflow: hidden; margin-top: 5px;">
            <div style="width: {p_model}%; background: #3b82f6; text-align: center; color: white; font-size: 10px; line-height: 20px;" title="Model & Optim">Model</div>
            <div style="width: {p_act}%; background: #e94560; text-align: center; color: white; font-size: 10px; line-height: 20px;" title="Activations">Act</div>
            <div style="width: {p_buff}%; background: #777; text-align: center; color: white; font-size: 10px; line-height: 20px;" title="Buffer">Buf</div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #888; margin-top: 2px;">
            <span>Model + Optim: {mem_info['model_gb']} GB</span>
            <span>Activations: {mem_info['act_gb']} GB</span>
        </div>
        """, unsafe_allow_html=True)

        if mem_info.get("notes"):
            st.caption(mem_info["notes"])

        if mem_info.get("method") == "profile_cuda" and isinstance(mem_info.get("detail"), dict):
            with st.expander("🔍 Детали замера (CUDA allocator / tensor sums)", expanded=False):
                st.json(mem_info["detail"])

        if mem_info["act_gb"] > mem_info["model_gb"] * 2:
            st.warning(f"⚠️ {t('warning.high_memory')}")

    
    # Визуальная схема архитектуры
    st.markdown(f"""
    <div class="model-ascii">
    ┌─────────────────────────────────────┐
    │         HomeForCausalLM             │
    ├─────────────────────────────────────┤
    │  Embedding: 50257 → {config['hidden_size']:4d}           │
    │  ┌─────────────────────────────┐    │
    │  │ HomeBlock × {config['num_layers']:2d}              │    │
    │  │  • RMSNorm → Attention      │    │
    │  │  • {config['n_heads']:2d} heads × {config['hidden_size']//config['n_heads']:3d} dim      │    │
    │  │  • RMSNorm → FFN (SwiGLU)   │    │
    │  └─────────────────────────────┘    │
    │  RMSNorm → LM Head                  │
    └─────────────────────────────────────┘
    </div>
        """, unsafe_allow_html=True)
    
    # Информация о параллелизме

    if distributed_config:
        st.subheader(f"⚡ {t('parallel.title')}")
        
        mode = distributed_config.get("distributed_mode", "default")
        mode_info = PARALLEL_TYPES.get(mode, PARALLEL_TYPES["default"])
        num_gpus = distributed_config.get("num_gpus", 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(t("parallel.mode"), mode_info["name"])
        
        with col2:
            st.metric(t("parallel.type"), mode_info["type"])
        
        with col3:
            if num_gpus > 0:
                st.metric("GPU", f"{num_gpus}")
            else:
                st.metric(t("parallel.device"), "CPU")
        
        # Схема параллелизма
        if mode == "default":
            parallel_diagram = """
┌─────────────────────────┐
│      Single Device      │
│  ┌───────────────────┐  │
│  │   Full Model      │  │
│  │   Full Optimizer  │  │
│  │   Full Gradients  │  │
│  └───────────────────┘  │
└─────────────────────────┘
"""
        elif mode == "multi_gpu":
            parallel_diagram = f"""
┌─────────────── Data Parallel ───────────────┐
│                                              │
│  ┌─────────┐  ┌─────────┐       ┌─────────┐ │
│  │  GPU 0  │  │  GPU 1  │  ...  │  GPU N  │ │
│  │ Model   │  │ Model   │       │ Model   │ │
│  │ (copy)  │  │ (copy)  │       │ (copy)  │ │
│  └────┬────┘  └────┬────┘       └────┬────┘ │
│       │            │                 │      │
│       └──────── Sync Gradients ──────┘      │
│                                              │
└──────────────────────────────────────────────┘
Каждая GPU: полная копия модели, часть батча
"""
        elif mode == "fsdp":
            parallel_diagram = f"""
┌────────────────── FSDP ─────────────────────┐
│                                              │
│  ┌─────────┐  ┌─────────┐       ┌─────────┐ │
│  │  GPU 0  │  │  GPU 1  │  ...  │  GPU N  │ │
│  │ Shard 0 │  │ Shard 1 │       │ Shard N │ │
│  └────┬────┘  └────┬────┘       └────┬────┘ │
│       │            │                 │      │
│       └─── All-Gather for Forward ──┘       │
│       └─── Reduce-Scatter Backward ─┘       │
│                                              │
│  🦁 Liger fused CE: ДА (экономия памяти)    │
└──────────────────────────────────────────────┘
Модель шардирована между GPU
"""
        elif mode == "fsdp_offload":
            parallel_diagram = f"""
┌────────── FSDP + CPU Offload ───────────────┐
│                                              │
│  ┌─────────┐  ┌─────────┐       ┌─────────┐ │
│  │  GPU 0  │  │  GPU 1  │  ...  │  GPU N  │ │
│  │ Shard 0 │  │ Shard 1 │       │ Shard N │ │
│  └────┬────┘  └────┬────┘       └────┬────┘ │
│       │            │                 │      │
│       └─── All-Gather for Forward ──┘       │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │           💾 CPU RAM                 │   │
│  │        Offloaded Parameters          │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ⚠️ Liger fused CE: НЕТ (уменьшите batch)   │
└──────────────────────────────────────────────┘
Параметры на CPU — экономия VRAM
"""
        elif mode == "fsdp2":
            parallel_diagram = f"""
┌────────── FSDP2 + CPU Offload ──────────────┐
│                                              │
│  ┌─────────┐  ┌─────────┐       ┌─────────┐ │
│  │  GPU 0  │  │  GPU 1  │  ...  │  GPU N  │ │
│  │ DTensor │  │ DTensor │       │ DTensor │ │
│  │ Shard 0 │  │ Shard 1 │       │ Shard N │ │
│  └────┬────┘  └────┬────┘       └────┬────┘ │
│       │            │                 │      │
│       └─── Per-Parameter Sharding ──┘       │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │           💾 CPU RAM                 │   │
│  │        Offloaded Parameters          │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ⚠️ Liger fused CE: НЕТ (уменьшите batch)   │
└──────────────────────────────────────────────┘
FSDP v2: DTensor + CPU Offload
"""
        elif "deepspeed" in mode:
            if "zero3" in mode:
                parallel_diagram = f"""
┌────────── DeepSpeed ZeRO-3 ─────────────────┐
│                                              │
│  ┌─────────┐  ┌─────────┐       ┌─────────┐ │
│  │  GPU 0  │  │  GPU 1  │  ...  │  GPU N  │ │
│  │ Params  │  │ Params  │       │ Params  │ │
│  │  1/N    │  │  1/N    │       │  1/N    │ │
│  │ Optim   │  │ Optim   │       │ Optim   │ │
│  │  1/N    │  │  1/N    │       │  1/N    │ │
│  └─────────┘  └─────────┘       └─────────┘ │
│                                              │
│  {'+ CPU Offload (параметры на CPU)' if 'offload' in mode else ''}          │
└──────────────────────────────────────────────┘
Всё шардировано: максимальная экономия VRAM
"""
            else:
                parallel_diagram = f"""
┌────────── DeepSpeed ZeRO-2 ─────────────────┐
│                                              │
│  ┌─────────┐  ┌─────────┐       ┌─────────┐ │
│  │  GPU 0  │  │  GPU 1  │  ...  │  GPU N  │ │
│  │ Full    │  │ Full    │       │ Full    │ │
│  │ Model   │  │ Model   │       │ Model   │ │
│  │ Optim/N │  │ Optim/N │       │ Optim/N │ │
│  └─────────┘  └─────────┘       └─────────┘ │
│                                              │
└──────────────────────────────────────────────┘
Оптимизатор и градиенты шардированы
"""
        else:
            parallel_diagram = ""
        
        if parallel_diagram:
            st.markdown(f"""
<div class="model-ascii">
{parallel_diagram}
</div>
            """, unsafe_allow_html=True)


def export_model_to_hf(model, tokenizer, source_path: str):
    """Экспортирует модель и токенизатор в стандартный HF формат.
    
    ВАЖНО: Если модель использует LoRA/QLoRA, мерджит адаптер в базу,
    чтобы экспортированная модель была "готовой" и загружалась как обычная.
    """
    try:
        from peft import PeftModel
        
        source = Path(source_path)
        # Создаем имя для экспорта: export_TIMESTAMP
        export_name = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Если это чекпоинт, сохраняем рядом с ним, но в отдельную папку
        # out/model/run/checkpoint_X -> out/model/run/export_X
        if "checkpoint" in source.name:
            export_dir = source.parent / f"export_{source.name}"
        else:
            export_dir = source.parent / export_name
            
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Если это PEFT-модель (LoRA/QLoRA) — мерджим адаптер в базу
        export_model = model
        try:
            if isinstance(model, PeftModel):
                logger.info("Merging LoRA adapter into base model for export...")
                export_model = model.merge_and_unload()
                logger.info("LoRA adapter merged successfully")
        except Exception as e:
            logger.warning(f"LoRA merge failed during export, saving as-is: {e}")
            # Продолжаем с исходной моделью
        
        # Сохраняем
        export_model.save_pretrained(export_dir, safe_serialization=True)
        tokenizer.save_pretrained(export_dir)
        
        return str(export_dir)
    except Exception as e:
        st.error(f"{t('error.export')}: {e}")
        return None


# ============================================================================
# Main App
# ============================================================================

def main():
    q = st.query_params
    if "run_id" in q:
        target_run = q["run_id"]
        
        # Check if this is a VLM run and redirect if necessary
        is_vlm = False
        config_path = RUNS_DIR / target_run / "config.json"
        if config_path.exists():
            import json
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if str(cfg.get("stage", "")).startswith("vlm_"):
                        is_vlm = True
            except:
                pass
                
        if is_vlm:
            st.markdown(f'<meta http-equiv="refresh" content="0;url=/VLM_Studio?run_id={target_run}">', unsafe_allow_html=True)
            return

        if st.session_state.get("last_processed_run_id") != target_run:
            st.session_state.current_run_id = target_run
            st.session_state.last_processed_run_id = target_run
            st.toast(f"📊 Выбран run: {target_run}. Перейдите на вкладку 'Мониторинг'!", icon="✅")

    if not st.session_state.get("onboarding_done", False):
        render_first_run_setup(USER_PREFS_FILE)
        return

    render_header()
    
    # Sidebar configs
    model_config = render_model_config()
    st.session_state.current_model_name = model_config.get("model_name_input", "home_model")
    
    current_stage = model_config.get("stage", "pretrain")
    
    # GRPO имеет свою конфигурацию обучения
    if current_stage == "grpo":
        grpo_sidebar_config = render_grpo_sidebar_config()
        # Для GRPO не нужна стандартная конфигурация обучения
        training_config = {}
        # ВАЖНО: Для GRPO тоже используем distributed config для поддержки multi-GPU
        # Создаем минимальный training_config для render_distributed_config
        dummy_training_config = {
            # показываем честную семантику в GPU табе: microbatch = train_batch_size
            "batch_size": grpo_sidebar_config.get("grpo_train_batch_size", 2),
            "gradient_accumulation": grpo_sidebar_config.get("gradient_accumulation", 4),
        }
        # Передаём training_backend из model_config (выбран до метода тюнинга)
        grpo_backend = model_config.get("training_backend", "models-at-home")
        distributed_config = render_distributed_config(training_config=dummy_training_config, is_grpo=True, grpo_backend=grpo_backend)
    else:
        grpo_sidebar_config = {}
        training_config = render_training_config()
        distributed_config = render_distributed_config(training_config=training_config, is_grpo=False)
    
    # Передаем stage в dataset_config
    # Для GRPO датасет настраивается в main area
    if current_stage != "grpo":
        dataset_config = render_dataset_config(stage=current_stage)
    else:
        dataset_config = {}
    
    output_config = render_output_config(st.session_state.current_model_name)
    
    # Merge configs
    # ВАЖНО: grpo_sidebar_config должен содержать ВСЕ необходимые параметры
    # Приоритет: grpo_sidebar_config > model_config (для GRPO)
    full_config = {**model_config, **training_config, **dataset_config, **output_config, **grpo_sidebar_config}
    
    # Для GRPO: проверяем что все обязательные параметры установлены
    if current_stage == "grpo":
        # Проверяем что LoRA параметры установлены (если use_lora=True)
        # LoRA параметры должны быть в model_config (из render_model_config())
        # Определяем use_lora из tuning_method (lora/qlora = use_lora=True)
        tuning_method = model_config.get("tuning_method", "full")
        use_lora_from_model = tuning_method in ("lora", "qlora")
        
        # Если выбран метод lora/qlora, проверяем что параметры установлены
        if use_lora_from_model:
            if "lora_r" not in model_config or model_config.get("lora_r") is None:
                raise ValueError(
                    "❌ Выбран метод 'lora' или 'qlora', но lora_r не установлен! "
                    "Убедитесь что в секции '🎯 Метод тюнинга' указан параметр 'LoRA r'."
                )
            if "lora_alpha" not in model_config or model_config.get("lora_alpha") is None:
                raise ValueError(
                    "❌ Выбран метод 'lora' или 'qlora', но lora_alpha не установлен! "
                    "Убедитесь что в секции '🎯 Метод тюнинга' указан параметр 'LoRA alpha'."
                )
            # Копируем LoRA параметры из model_config в full_config для передачи в train_rl.py
            full_config["use_lora"] = True
            full_config["lora_r"] = model_config["lora_r"]
            full_config["lora_alpha"] = model_config["lora_alpha"]
            full_config["lora_dropout"] = model_config.get("lora_dropout")
            full_config["lora_target_modules"] = model_config.get("lora_target_modules")
            # Квантизация только для qlora
            if tuning_method == "qlora":
                full_config["use_4bit"] = True  # QLoRA всегда использует 4-bit
                full_config["use_8bit"] = model_config.get("use_8bit", False)
            else:
                full_config["use_4bit"] = False
                full_config["use_8bit"] = False
        else:
            # Если метод full, use_lora=False
            full_config["use_lora"] = False
            full_config["use_4bit"] = False
            full_config["use_8bit"] = False
        
        # Проверяем что обязательные GRPO параметры установлены
        required_grpo_params = [
            "grpo_algorithm", "grpo_group_size", "grpo_max_new_tokens",
            "grpo_temperature", "grpo_learning_rate", "grpo_kl_weight",
            "grpo_clip_eps_low"
        ]
        missing_params = [p for p in required_grpo_params if p not in full_config]
        if missing_params:
            raise ValueError(
                f"❌ Отсутствуют обязательные GRPO параметры: {missing_params}. "
                f"Убедитесь что render_grpo_sidebar_config() возвращает все параметры."
            )
    full_config["distributed_mode"] = distributed_config["distributed_mode"]
    full_config["num_gpus"] = distributed_config["num_gpus"]
    full_config["config_file"] = distributed_config["config_file"]
    full_config["gpu_ids"] = distributed_config.get("gpu_ids", [])
    # ВАЖНО: значения из UI (sidebar) должны иметь приоритет над пресетами training_config.
    # Иначе пользователь выбирает одно, а в ран уезжает другое (как было с mixed_precision=no -> bf16).
    full_config["mixed_precision"] = distributed_config.get("mixed_precision", "bf16")
    full_config["fp16_pure"] = distributed_config.get("fp16_pure", False)
    full_config["grad_checkpoint"] = distributed_config.get("grad_checkpoint", True)
    full_config["use_flash_attention"] = distributed_config.get("use_flash_attention", True)
    full_config["use_liger"] = distributed_config.get("use_liger", True)
    full_config["liger_fused_ce"] = distributed_config.get("liger_fused_ce", False)  # Fused CE для pretrain/SFT
    # Для GRPO training_backend берётся из grpo_sidebar_config (уже в full_config)
    # Для остальных режимов — из distributed_config
    if current_stage != "grpo":
        full_config["training_backend"] = distributed_config.get("training_backend", "models-at-home")
    # Для GRPO training_backend уже есть в full_config из grpo_sidebar_config
    
    # Для SFT, Continual Pretrain и GRPO используем токенизатор базовой модели
    if model_config.get("stage") in ("sft", "continual_pretrain", "grpo") and model_config.get("base_model_path"):
        full_config["tokenizer_path"] = model_config["base_model_path"]
    
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        f"🚀 {t('tabs.launch')}", 
        f"📊 {t('tabs.monitoring')}", 
        f"💬 {t('tabs.chat')}", 
        f"📜 {t('tabs.history')}",
        f"💾 {t('tabs.data')}", 
        f"🤖 {t('tabs.models')}", 
        f"📚 {t('tabs.docs')}"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Конфигурация тренировки
            all_ready = render_quick_summary(model_config, dataset_config, distributed_config, full_config)
            
            # Передаем full_config, чтобы калькулятор памяти видел batch_size и grad_checkpoint
            render_model_preview(full_config, distributed_config)
            
            # SFT: Chat Template модели (отдельный блок ПЕРЕД настройкой данных)
            if model_config.get("stage") == "sft" and model_config.get("base_model_path"):
                st.markdown("---")
                st.markdown(f"### 📝 {t('sft.chat_template')}")
                
                # Загружаем chat_template из базовой модели
                base_model_path = model_config.get("base_model_path")
                model_chat_template = None
                
                try:
                    from transformers import AutoTokenizer
                    with st.spinner("Загрузка chat_template из модели..."):
                        tok = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
                        # Сохраняем токенизатор в session_state для использования в превью
                        st.session_state.sft_tokenizer = tok
                        if hasattr(tok, 'chat_template') and tok.chat_template:
                            model_chat_template = tok.chat_template
                except Exception as e:
                    st.warning(f"{t('warning.tokenizer_load_failed')}: {e}")
                    st.session_state.sft_tokenizer = None
                
                # Показываем статус
                if model_chat_template:
                    st.success(f"✅ {t('success.chat_template_loaded')} `{Path(base_model_path).name}`")
                else:
                    st.info(t("info.no_chat_template"))
                
                # Инициализируем session_state если нужно
                if "sft_user_chat_template" not in st.session_state:
                    st.session_state.sft_user_chat_template = model_chat_template or ""
                
                # Редактируемое поле для chat_template
                user_chat_template = st.text_area(
                    "Chat Template (Jinja2):",
                    value=st.session_state.sft_user_chat_template,
                    height=200,
                    key="sft_chat_template_field",
                    help="Jinja2 шаблон для форматирования диалогов. Оставьте пустым для автогенерации.",
                    placeholder="Оставьте пустым для автогенерации на основе тегов..."
                )
                
                # Синхронизируем с session_state
                st.session_state.sft_user_chat_template = user_chat_template
                
                # Кнопки управления
                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
                with col_btn1:
                    if st.button("✨ Qwen-style", key="sft_qwen_template", help=t("help.qwen_template")):
                        # Qwen-style chat template (упрощённый, без tools)
                        qwen_template = """{%- if messages[0]['role'] == 'system' -%}
{{ '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
{%- else -%}
{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}
{%- endif -%}
{%- for message in messages -%}
{%- if message.role == 'user' or (message.role == 'system' and not loop.first) -%}
{{ '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- elif message.role == 'assistant' -%}
{{ '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<|im_start|>assistant\\n' }}
{%- endif -%}"""
                        st.session_state.sft_user_chat_template = qwen_template
                        st.rerun()
                with col_btn2:
                    if model_chat_template and st.button(f"↩️ {t('button.from_model')}", key="sft_restore_template"):
                        st.session_state.sft_user_chat_template = model_chat_template
                        st.rerun()
                with col_btn3:
                    if st.button(f"🗑️ {t('button.clear')}", key="sft_clear_template"):
                        st.session_state.sft_user_chat_template = ""
                        st.rerun()
                with col_btn4:
                    if user_chat_template.strip():
                        st.caption(f"📏 {len(user_chat_template)} симв.")
                
                # Добавляем chat_template в конфиг
                full_config["chat_template"] = user_chat_template.strip() if user_chat_template.strip() else None
            
            # SFT Config (Main Area) - настройка данных
            if model_config.get("stage") == "sft" and dataset_config.get("data_path"):
                st.markdown("---")
                sft_cfg = render_sft_main_config(dataset_config["data_path"])
                full_config.update(sft_cfg)
            
            # GRPO Config (Main Area) - настройка reward функций и датасета
            if model_config.get("stage") == "grpo":
                st.markdown("---")
                grpo_main_cfg = render_grpo_main_config(dataset_config.get("data_path"))
                full_config.update(grpo_main_cfg)
            
            st.subheader(f"📋 {t('common.configuration')}")
            st.json(full_config)
        
        with col2:
            st.subheader(f"🎮 {t('common.control')}")
            
            if st.session_state.training_active:
                if st.button(f"⏹️ {t('button.stop_training')}", type="primary"):
                    with st.spinner("Останавливаем тренировку..."):
                        stopped = stop_training()
                    if stopped:
                        st.success(f"✅ {t('status.stopped')}")
                    else:
                        st.warning(f"⚠️ {t('warning.stop_failed')}")
                    time.sleep(1)
                    st.rerun()
            else:
                # Для GRPO отдельная кнопка и запуск
                if model_config.get("stage") == "grpo":
                    button_disabled = not all_ready
                    if st.button(f"🧠 {t('button.start_grpo')}", type="primary", disabled=button_disabled):
                        with st.spinner("Запуск GRPO..."):
                            run_id, process = start_grpo_training(full_config)
                            st.session_state.current_run_id = run_id
                            st.session_state.training_process = process
                            st.session_state.training_active = True
                            save_active_run(run_id, full_config)
                            st.success(f"GRPO обучение запущено! Run ID: {run_id}")
                            time.sleep(1)
                            st.rerun()
                    if button_disabled:
                        st.caption("⚠️ Выберите модель, данные и режим тренировки для запуска")
                else:
                    button_disabled = not all_ready
                    if st.button(f"▶️ {t('button.start_training')}", type="primary", disabled=button_disabled):
                        with st.spinner("Запуск..."):
                            run_id, process = start_training(full_config)
                            st.session_state.current_run_id = run_id
                            st.session_state.training_process = process
                            st.session_state.training_active = True
                            # Сохраняем активный run для persistence
                            save_active_run(run_id, full_config)
                            st.success(f"Тренировка запущена! Run ID: {run_id}")
                            time.sleep(1)
                            st.rerun()
                    if button_disabled:
                        st.caption("⚠️ Выберите модель, данные и режим тренировки для запуска")
    
    with tab2:
        # Используем fragment для автоматического обновления без перезагрузки страницы
        live_metrics_fragment()
    
    with tab4:
        st.header(f"📜 {t('history.title')}")
        st.markdown("---")
        
        # Фильтруем только директории (игнорируем файлы типа active_run.json)
        # ВАЖНО: Сортируем по времени модификации (не алфавитно!), чтобы grpo_ и обычные запуски
        # отображались вместе в правильном хронологическом порядке
        runs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        
        if runs:
            # Показываем последние 30 запусков
            for run_dir in runs[:30]: 
                run_id = run_dir.name
                metrics = load_metrics(run_id)
                
                if metrics:
                    status = metrics.get("status", "unknown")
                    # Пропускаем пустые запуски (если они старые и ничего не сделали)
                    is_empty = metrics.get("current_step", 0) == 0 and not metrics.get("checkpoints")
                    if is_empty and status not in ("training", "running"):
                         continue

                    status_emoji = {"training": "🟢", "completed": "✅", "error": "❌", "stopped": "⏹️", "resumed": "▶️"}.get(status, "⏳")
                    
                    # Пытаемся получить имя модели из конфига для заголовка
                    model_name_display = run_id
                    try:
                        config_path = run_dir / "config.json"
                        if config_path.exists():
                            with open(config_path) as f:
                                rc = json.load(f)
                                # Ищем имя модели или output_dir
                                if "model_name_input" in rc:
                                    model_name_display = f"{run_id} | {rc['model_name_input']}"
                                elif "output_dir" in rc:
                                    out_d = Path(rc["output_dir"])
                                    # out/home_pretrain/run_id -> home_pretrain
                                    if out_d.name == run_id:
                                        model_name_display = f"{run_id} | {out_d.parent.name}"
                                    else:
                                        model_name_display = f"{run_id} | {out_d.name}"
                    except:
                        pass
                    
                    with st.expander(f"{status_emoji} {model_name_display}"):
                        # Параметры модели из metrics.json (если есть) или из config
                        model_params = None
                        try:
                            # Сначала пробуем взять из metrics.json (более точно)
                            if "num_parameters" in metrics:
                                model_params = metrics["num_parameters"]
                            else:
                                # Fallback: рассчитываем из config
                                config_path = run_dir / "config.json"
                                if config_path.exists():
                                    with open(config_path) as f:
                                        rc = json.load(f)
                                        if "hidden_size" in rc and "num_layers" in rc:
                                            vocab_size = rc.get("vocab_size", 50257)
                                            intermediate_size = rc.get("intermediate_size")
                                            model_params = estimate_parameters(
                                                rc["hidden_size"],
                                                rc["num_layers"],
                                                vocab_size=vocab_size,
                                                intermediate_size=intermediate_size,
                                            )
                        except Exception:
                            pass
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Steps", metrics.get("current_step", 0))
                        with col2:
                            st.metric("Final Loss", f"{metrics.get('current_loss', 0):.4f}")
                        with col3:
                            if model_params:
                                st.metric(t("model.params"), format_params(model_params))
                            else:
                                st.metric(t("model.params"), "—")
                        with col4:
                            st.metric("Status", status)
                        with col5:
                            st.metric("Duration", metrics.get("training_duration", "-"))
                        
                        # Чекпоинты этого запуска
                        checkpoints = metrics.get("checkpoints", [])
                        if checkpoints:
                            st.markdown("**📦 Чекпоинты:**")
                            # Для получения loss из history, если его нет в чекпоинте
                            loss_history = metrics.get("loss_history", [])
                            steps_history = metrics.get("steps_history", [])
                            
                            for ckpt_idx, ckpt in enumerate(checkpoints):
                                ckpt_loss = ckpt.get("loss")
                                # Если loss нет в чекпоинте, пытаемся найти в loss_history
                                if ckpt_loss is None and steps_history and loss_history:
                                    ckpt_step = ckpt.get("step")
                                    if ckpt_step in steps_history:
                                        idx = steps_history.index(ckpt_step)
                                        if idx < len(loss_history):
                                            ckpt_loss = loss_history[idx]
                                
                                # Колонки: инфо о чекпоинте + кнопка удаления
                                ckpt_col1, ckpt_col2 = st.columns([5, 1])
                                
                                with ckpt_col1:
                                    # Проверяем существует ли чекпоинт на диске
                                    ckpt_path = Path(ckpt['path'])
                                    if not ckpt_path.is_absolute():
                                        ckpt_path = PROJECT_ROOT / ckpt['path']
                                    ckpt_exists = ckpt_path.exists()
                                    
                                    if ckpt_loss is not None:
                                        info_text = f"Step {ckpt['step']}: Loss {ckpt_loss:.4f} | {ckpt['path']}"
                                    else:
                                        info_text = f"Step {ckpt['step']}: {ckpt['path']}"
                                    
                                    if not ckpt_exists:
                                        st.caption(f"~~{info_text}~~ *(удален с диска)*")
                                    else:
                                        st.caption(info_text)
                                
                                with ckpt_col2:
                                    if ckpt_exists:
                                        # Уникальный ключ для каждой кнопки удаления чекпоинта
                                        del_key = f"del_ckpt_{run_id}_{ckpt['step']}_{ckpt_idx}"
                                        if st.button("🗑️", key=del_key, help=f"Удалить чекпоинт step {ckpt['step']}"):
                                            success, msg = delete_checkpoint(run_id, ckpt['path'])
                                            if success:
                                                st.toast(msg, icon="✅")
                                                time.sleep(0.5)
                                                st.rerun()
                                            else:
                                                st.toast(msg, icon="❌")
                        
                        # Кнопки действий
                        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                        with btn_col1:
                            if st.button(f"📊 {t('metrics.title')}", key=f"metrics_{run_id}"):
                                st.session_state.current_run_id = run_id
                                st.toast(f"✅ Выбран run: {run_id}. Перейдите на вкладку 📊 Мониторинг", icon="📊")
                            st.markdown(f"[🔗 Ссылка на процесс](/?run_id={run_id})")
                        with btn_col2:
                            # Проверяем есть ли модель для чата
                            config_path = run_dir / "config.json"
                            if config_path.exists():
                                try:
                                    with open(config_path) as f:
                                        run_config = json.load(f)
                                    model_dir = PROJECT_ROOT / run_config.get("output_dir", "")
                                    final_model = model_dir / "final_model"
                                    if final_model.exists():
                                        if st.button(f"💬 {t('tabs.chat')}", key=f"chat_run_{run_id}"):
                                            st.session_state.selected_chat_model = str(final_model)
                                            st.toast("✅ Модель выбрана! Перейдите на вкладку 💬 Чат", icon="💬")
                                except:
                                    pass
                        with btn_col3:
                            # Кнопка продолжения тренировки (если были чекпоинты)
                            # Проверяем что чекпоинт РЕАЛЬНО существует на диске
                            valid_ckpt = None
                            if checkpoints:
                                latest_ckpt_path = checkpoints[-1]['path']
                                # Превращаем в абсолютный путь если он относительный
                                abs_ckpt_path = Path(latest_ckpt_path)
                                if not abs_ckpt_path.is_absolute():
                                    abs_ckpt_path = PROJECT_ROOT / latest_ckpt_path
                                
                                if abs_ckpt_path.exists():
                                    valid_ckpt = str(abs_ckpt_path)

                            if valid_ckpt:
                                if st.button(f"▶️ {t('button.continue_training')}", key=f"continue_{run_id}", help=t("help.continue_training")):
                                    try:
                                        # 2. Загружаем конфиг старого запуска
                                        config_path = run_dir / "config.json"
                                        with open(config_path) as f:
                                            old_config = json.load(f)
                                        
                                        # 3. Корректируем output_dir чтобы не создавать вложенность
                                        # Старый output_dir указывал на папку конкретного запуска (run_ID)
                                        # Мы хотим чтобы новый запуск был на уровне с старым (в родительской папке)
                                        old_output_dir = Path(old_config.get("output_dir", ""))
                                        # Если путь абсолютный - берем родителя. Если нет - тоже (надеемся)
                                        # start_training ожидает путь к корню эксперимента
                                        old_config["output_dir"] = str(old_output_dir.parent)
                                        
                                        # 4. Устанавливаем флаг resume (АБСОЛЮТНЫЙ ПУТЬ)
                                        old_config["resume_from_checkpoint"] = valid_ckpt
                                        
                                        # 5. Запускаем
                                        with st.spinner(f"Возобновляем обучение с {valid_ckpt}..."):
                                            new_run_id, process = start_training(old_config)
                                            st.session_state.current_run_id = new_run_id
                                            st.session_state.training_process = process
                                            st.session_state.training_active = True
                                            
                                            save_active_run(new_run_id, old_config)
                                            
                                            st.success(f"Тренировка возобновлена! Run ID: {new_run_id}")
                                            time.sleep(1)
                                            st.rerun()
                                            
                                    except Exception as e:
                                        st.error(f"Не удалось возобновить: {e}")
                            elif checkpoints:
                                # Чекпоинты были в метриках, но удалены с диска
                                st.button(f"⚠️ {t('status.files_deleted')}", key=f"gone_{run_id}", disabled=True, help=t("help.checkpoint_not_found"))
                        
                        with btn_col4:
                            # Кнопка удаления эксперимента с подтверждением
                            delete_confirm_key = f"delete_confirm_{run_id}"
                            
                            # Инициализируем состояние подтверждения если его нет
                            if delete_confirm_key not in st.session_state:
                                st.session_state[delete_confirm_key] = False
                            
                            if not st.session_state[delete_confirm_key]:
                                # Первое нажатие - показываем кнопку удаления
                                if st.button("🗑️ Удалить", key=f"del_exp_{run_id}", help="Удалить эксперимент со всеми чекпоинтами"):
                                    st.session_state[delete_confirm_key] = True
                                    st.rerun()
                            else:
                                # Показываем кнопки подтверждения
                                confirm_col1, confirm_col2 = st.columns(2)
                                with confirm_col1:
                                    if st.button("✅ Да", key=f"confirm_del_{run_id}", type="primary"):
                                        # Выполняем удаление
                                        success, msg = delete_experiment(run_id)
                                        st.session_state[delete_confirm_key] = False
                                        if success:
                                            st.toast(msg, icon="✅")
                                            # Если удалили текущий выбранный run, сбрасываем
                                            if st.session_state.current_run_id == run_id:
                                                st.session_state.current_run_id = None
                                            time.sleep(0.5)
                                            st.rerun()
                                        else:
                                            st.toast(msg, icon="❌")
                                with confirm_col2:
                                    if st.button("❌ Нет", key=f"cancel_del_{run_id}"):
                                        st.session_state[delete_confirm_key] = False
                                        st.rerun()
                        
                        # Показываем что выбрано
                        if st.session_state.current_run_id == run_id:
                            st.info("👆 Перейдите на вкладку **📊 Мониторинг**")
        else:
            st.info("Нет предыдущих запусков")
            
    with tab5:
        render_data_manager(stage=current_stage)
        
        # Подсказка про чат
        st.markdown("---")
        st.info("💡 Чтобы пообщаться с моделью, перейдите на вкладку **💬 Чат** (в верхней части страницы)")

    with tab6:
        render_model_manager()
    
    with tab7:
        render_docs()
    
    with tab3:
        st.header(f"💬 {t('chat.title')}")
        st.markdown("---")
        
        # Список моделей обновляется при каждом заходе на вкладку Чат
        available_models = get_available_models()
        
        if available_models:
            # Выбор модели
            col1, col2 = st.columns([3, 1])
            
            with col1:
                model_options = [m["name"] for m in available_models]
                
                # Если модель выбрана из истории - находим её индекс
                default_idx = 0
                if st.session_state.selected_chat_model:
                    for i, m in enumerate(available_models):
                        if m["path"] == st.session_state.selected_chat_model:
                            default_idx = i
                            break
                
                selected_model_name = st.selectbox(
                    t("chat.select_model"),
                    options=model_options,
                    index=default_idx,
                    help="Выберите обученную модель для чата"
                )
                selected_model = next(m for m in available_models if m["name"] == selected_model_name)
                
                # Сбрасываем selected_chat_model после использования
                if st.session_state.selected_chat_model:
                    st.session_state.selected_chat_model = None
            
            with col2:
                # Информация о модели
                model_type = selected_model.get("type", "unknown")
                training_type = selected_model.get("training_type", "unknown")
                is_lora = selected_model.get("is_lora", False)
                model_info = selected_model.get("model_info", {})
                
                # Карточка информации о модели
                info_cols = st.columns([2, 1])
                with info_cols[0]:
                    # Тип модели
                    type_labels = {
                        "final": f"✅ {t('chat.final_model')}",
                        "checkpoint": "📦 Чекпоинт",
                        "lora": "🔧 LoRA адаптер",
                        "hf": "🤗 HuggingFace"
                    }
                    st.markdown(f"**{type_labels.get(model_type, '📦 Модель')}**")
                    
                    # Тип тренировки
                    training_labels = {
                        "pretrain": "Pre-training",
                        "sft": "SFT (Supervised Fine-Tuning)",
                        "grpo": "GRPO (Reasoning)",
                        "base": "Base Model"
                    }
                    if training_type != "unknown":
                        st.caption(f"{t('chat.training_type')}: {training_labels.get(training_type, training_type)}")
                
                with info_cols[1]:
                    # Технические характеристики
                    if model_info.get("max_context"):
                        st.metric(t("model.context"), f"{model_info['max_context']:,}")
                    if is_lora and model_info.get("lora_r"):
                        st.caption(f"LoRA r={model_info['lora_r']}")
                
                # Показываем базовую модель для LoRA
                if is_lora and model_info.get("base_model"):
                    st.info(f"🔗 Базовая модель: {model_info['base_model']}")
                
                st.caption(f"📁 {selected_model['path']}")
            
            # Параметры генерации
            with st.expander(f"⚙️ {t('chat.generation_params')}", expanded=True):
                # Определяем максимальный контекст из конфига модели
                max_context = model_info.get("max_context") or 32168
                default_max_tokens = min(256, max_context // 4)
                max_tokens_limit = min(max_context, 32168)  # Ограничиваем разумным максимумом
                
                gen_col1, gen_col2 = st.columns(2)
                with gen_col1:
                    max_tokens = st.slider(
                        t("chat.max_tokens"), 
                        min_value=16, 
                        max_value=max_tokens_limit, 
                        value=default_max_tokens,
                        step=16,
                        help=f"Максимум новых токенов. Контекст модели: {max_context:,}"
                    )
                    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05, help="0 = greedy decoding")
                
                with gen_col2:
                    # По умолчанию top_p=1.0 и top_k=0 (отключены) — только temperature влияет
                    top_p = st.slider("Top-p (nucleus)", 0.1, 1.0, 1.0, 0.05, help="1.0 = отключено")
                    top_k = st.slider("Top-k", 0, 100, 0, help="0 = отключено")

                # Inference Backend
                from homellm.app.vllm_chat import is_vllm_available
                vllm_available = is_vllm_available()
                
                st.markdown("---")
                backend_col1, backend_col2 = st.columns(2)
                
                with backend_col1:
                    # vLLM первым если доступен (быстрее)
                    backend_options = []
                    if vllm_available:
                        backend_options.append(f"vLLM ({t('chat.faster')})")
                    backend_options.append("Transformers")
                    
                    if "chat_inference_backend" not in st.session_state:
                        # По умолчанию vLLM если доступен
                        st.session_state.chat_inference_backend = f"vLLM ({t('chat.faster')})" if vllm_available else "Transformers"
                    
                    inference_backend = st.selectbox(
                        t("chat.inference_backend"),
                        options=backend_options,
                        index=backend_options.index(st.session_state.chat_inference_backend) if st.session_state.chat_inference_backend in backend_options else 0,
                        help="vLLM: быстрее (PagedAttention), но требует больше VRAM",
                        key="chat_backend_select",
                    )
                    st.session_state.chat_inference_backend = inference_backend
                    
                    if not vllm_available:
                        st.caption("ℹ️ vLLM: `pip install vllm`")
                
                with backend_col2:
                    # Режим промпта - зависит от наличия chat_template
                    if "chat_prompt_mode" not in st.session_state:
                        st.session_state.chat_prompt_mode = "completion"
                    
                    # Проверяем наличие chat_template у загруженной модели
                    has_template = st.session_state.get("chat_has_template", False)
                    model_loaded = st.session_state.get("chat_backend") is not None
                    
                    if model_loaded and has_template:
                        # Both modes available
                        prompt_mode_label = st.selectbox(
                            t("chat.mode"),
                            options=["Chat (template)", "Completion"],
                            index=0 if st.session_state.chat_prompt_mode == "chat" else 1,
                            help=t("help.chat_mode"),
                            key="chat_prompt_mode_select",
                        )
                        prompt_mode = "chat" if "Chat" in prompt_mode_label else "completion"
                    elif model_loaded and not has_template:
                        # Only Completion, Chat unavailable
                        st.selectbox(
                            t("chat.mode"),
                            options=[f"Completion ({t('chat.no_template')})"],
                            index=0,
                            disabled=True,
                            help=t("help.no_chat_template"),
                            key="chat_prompt_mode_select",
                        )
                        prompt_mode = "completion"
                    else:
                        # Model not loaded - show placeholder
                        st.selectbox(
                            t("chat.mode"),
                            options=[t("chat.load_model_prompt")],
                            index=0,
                            disabled=True,
                            key="chat_prompt_mode_select",
                        )
                        prompt_mode = st.session_state.chat_prompt_mode
                    
                    st.session_state.chat_prompt_mode = prompt_mode
            
            # Инициализация чата
            if "chat_model" not in st.session_state:
                st.session_state.chat_model = None
                st.session_state.chat_tokenizer = None
                st.session_state.chat_model_path = None
                st.session_state.chat_has_template = False
                st.session_state.chat_prompt_mode = "completion"
                st.session_state.chat_backend = None  # VLLMChatBackend или TransformersChatBackend
                st.session_state.chat_backend_type = "transformers"  # "transformers" или "vllm"
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Кнопка загрузки модели
            if st.session_state.chat_model_path != selected_model["path"]:
                if st.button(f"🔄 {t('button.load_model')}", type="primary"):
                    with st.spinner("Загружаем модель..."):
                        try:
                            from transformers import AutoTokenizer, AutoModelForCausalLM
                            from homellm.models.adapters import detect_model_type
                            from homellm.models.home_model import HomeForCausalLM
                            from homellm.app.vllm_chat import VLLMChatBackend, TransformersChatBackend, is_vllm_available
                            
                            model_path = Path(selected_model["path"])
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            dtype = torch.float16 if device == "cuda" else torch.float32
                            dtype_str = "float16" if device == "cuda" else "float32"
                            
                            # Проверяем наличие config.json или adapter_config.json
                            config_json = model_path / "config.json"
                            adapter_config_path = model_path / "adapter_config.json"
                            is_lora_adapter = adapter_config_path.exists()
                            
                            if not config_json.exists() and not is_lora_adapter:
                                raise ValueError(f"config.json или adapter_config.json не найден в {model_path}")
                            
                            # Определяем тип модели
                            model_type = detect_model_type(model_path) if config_json.exists() else "hf"
                            
                            # Выбранный backend
                            use_vllm = inference_backend.startswith("vLLM") and is_vllm_available()
                            
                            st.info(f"Загружаем модель через {'vLLM' if use_vllm else 'Transformers'}...")
                            
                            # === vLLM Backend ===
                            if use_vllm:
                                # Для кастомных моделей: подготавливаем для vLLM (auto_map + файлы)
                                from homellm.models.adapters import is_custom_architecture, prepare_model_for_vllm
                                if is_custom_architecture(model_type):
                                    st.info("🔧 Подготовка кастомной модели для vLLM...")
                                    prepare_model_for_vllm(model_path)
                                
                                # Для LoRA адаптеров с vLLM: загружаем базовую модель + hot-swap LoRA
                                if is_lora_adapter:
                                    with open(adapter_config_path) as f:
                                        adapter_cfg = json.load(f)
                                    base_model_id = adapter_cfg.get("base_model_name_or_path")
                                    
                                    if not base_model_id:
                                        raise ValueError("base_model_name_or_path не найден в adapter_config.json")
                                    
                                    st.info(f"Загружаем базовую модель в vLLM: {base_model_id}")
                                    
                                    chat_backend = VLLMChatBackend(
                                        model_path=base_model_id,
                                        dtype=dtype_str,
                                        gpu_memory_utilization=0.9,
                                        enable_lora=True,
                                        max_lora_rank=adapter_cfg.get("r", 64),
                                    )
                                    chat_backend.set_lora(str(model_path))
                                    st.success("✅ vLLM загружен с LoRA адаптером")
                                else:
                                    chat_backend = VLLMChatBackend(
                                        model_path=str(model_path),
                                        dtype=dtype_str,
                                        gpu_memory_utilization=0.9,
                                    )
                                    st.success("✅ vLLM загружен")
                                
                                st.session_state.chat_model = None  # vLLM управляет моделью внутри
                                st.session_state.chat_tokenizer = chat_backend.tokenizer
                                st.session_state.chat_backend = chat_backend
                                st.session_state.chat_backend_type = "vllm"
                                st.session_state.chat_has_template = chat_backend.has_chat_template
                            
                            # === Transformers Backend ===
                            else:
                                # Загружаем токенизатор
                                tokenizer = None
                                tokenizer_source = None
                                
                                # Для LoRA адаптеров - сначала пробуем базовую модель
                                if is_lora_adapter:
                                    try:
                                        with open(adapter_config_path) as f:
                                            adapter_cfg = json.load(f)
                                        base_model_id = adapter_cfg.get("base_model_name_or_path")
                                        if base_model_id:
                                            tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
                                            tokenizer_source = base_model_id
                                    except Exception:
                                        pass
                                
                                # Пробуем загрузить из папки модели
                                if tokenizer is None:
                                    try:
                                        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
                                        tokenizer_source = str(model_path)
                                    except Exception:
                                        pass
                                
                                # Fallback: ищем в run config
                                if tokenizer is None:
                                    try:
                                        run_root = model_path.parent if "checkpoint" in model_path.name else model_path
                                        run_id = run_root.name
                                        run_cfg_path = RUNS_DIR / run_id / "config.json"
                                        if run_cfg_path.exists():
                                            with open(run_cfg_path, "r", encoding="utf-8") as f:
                                                run_cfg = json.load(f)
                                            tok_src = run_cfg.get("tokenizer_path") or run_cfg.get("base_model_path")
                                            if tok_src:
                                                tokenizer = AutoTokenizer.from_pretrained(str(tok_src), trust_remote_code=True)
                                                tokenizer_source = tok_src
                                    except Exception:
                                        pass
                                
                                # Последний fallback - GPT2 (но предупреждаем)
                                if tokenizer is None:
                                    st.warning("⚠️ Не удалось загрузить токенизатор модели, используется GPT2 (без chat_template)")
                                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                                    tokenizer_source = "gpt2"
                                else:
                                    if tokenizer_source:
                                        st.caption(f"Токенизатор: {tokenizer_source}")
                                
                                # Загрузка модели
                                if is_lora_adapter:
                                    st.info("🔄 Обнаружен LoRA адаптер, загружаем с merge...")
                                    try:
                                        from peft import PeftModel
                                        
                                        with open(adapter_config_path) as f:
                                            adapter_cfg = json.load(f)
                                        base_model_id = adapter_cfg.get("base_model_name_or_path")
                                        
                                        if not base_model_id:
                                            raise ValueError("base_model_name_or_path не найден в adapter_config.json")
                                        
                                        st.info(f"Загружаем базовую модель: {base_model_id}")
                                        
                                        base_model = AutoModelForCausalLM.from_pretrained(
                                            base_model_id, torch_dtype=dtype, trust_remote_code=True
                                        )
                                        model = PeftModel.from_pretrained(base_model, str(model_path))
                                        
                                        st.info("Merging LoRA адаптеры...")
                                        model = model.merge_and_unload()
                                        st.success("✅ LoRA адаптеры успешно объединены")
                                        
                                    except ImportError:
                                        st.error("❌ Для загрузки LoRA адаптеров требуется peft. Установите: pip install peft")
                                        raise
                                else:
                                    if model_type == "home":
                                        model = HomeForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype)
                                    else:
                                        model = AutoModelForCausalLM.from_pretrained(
                                            str(model_path), trust_remote_code=True, torch_dtype=dtype
                                        )
                                
                                model = model.to(device)
                                model.eval()
                                
                                if tokenizer.pad_token is None:
                                    if tokenizer.eos_token:
                                        tokenizer.pad_token = tokenizer.eos_token
                                
                                chat_backend = TransformersChatBackend(model, tokenizer, device)
                                
                                st.session_state.chat_model = model
                                st.session_state.chat_tokenizer = tokenizer
                                st.session_state.chat_backend = chat_backend
                                st.session_state.chat_backend_type = "transformers"
                                st.session_state.chat_has_template = bool(getattr(tokenizer, "chat_template", None))
                                st.success("✅ Transformers модель загружена!")
                            
                            st.session_state.chat_model_path = str(model_path)
                            st.session_state.messages = []
                            st.session_state.chat_prompt_mode = "chat" if st.session_state.chat_has_template else "completion"
                            st.rerun()
                        except Exception as e:
                            import traceback
                            st.error(f"Ошибка загрузки: {e}")
                            st.code(traceback.format_exc())
                            
                            # Fallback для старых чекпоинтов (если AutoModelForCausalLM не сработал)
                            st.warning("Пробуем fallback загрузку для старых чекпоинтов...")
                            try:
                                from homellm.models.home_model import HomeForCausalLM, HomeConfig
                                from safetensors.torch import load_file
                                
                                model_safetensors = model_path / "model.safetensors"
                                model_bin = model_path / "pytorch_model.bin"
                                
                                if not (model_safetensors.exists() or model_bin.exists()):
                                    raise ValueError("Не найдены веса модели")
                                
                                # Загружаем токенизатор
                                if not st.session_state.get("chat_tokenizer"):
                                    st.session_state.chat_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                                    if st.session_state.chat_tokenizer.pad_token is None:
                                        st.session_state.chat_tokenizer.pad_token = st.session_state.chat_tokenizer.eos_token
                                
                                # Загружаем конфиг
                                if config_json.exists():
                                    config = HomeConfig.from_pretrained(str(model_path))
                                else:
                                    # Ищем в родительской директории
                                    run_config_path = model_path.parent / "run_config.json"
                                    if run_config_path.exists():
                                        import json as json_module
                                        with open(run_config_path) as f:
                                            run_cfg = json_module.load(f)
                                        config = HomeConfig(
                                            vocab_size=len(st.session_state.chat_tokenizer),
                                            hidden_size=run_cfg.get("hidden_size", 512),
                                            num_hidden_layers=run_cfg.get("num_layers", 8),
                                            num_attention_heads=run_cfg.get("n_heads", 8),
                                            max_position_embeddings=run_cfg.get("seq_len", 512),
                                        )
                                    else:
                                        raise ValueError("Не найден config.json")
                                
                                # Создаём и загружаем модель
                                st.session_state.chat_model = HomeForCausalLM(config)
                                
                                # Загружаем веса
                                if model_safetensors.exists():
                                    state_dict = load_file(str(model_safetensors))
                                else:
                                    state_dict = torch.load(str(model_bin), map_location="cpu")
                                
                                missing, unexpected = st.session_state.chat_model.load_state_dict(state_dict, strict=False)
                                
                                if missing:
                                    real_missing = [k for k in missing if k != "lm_head.weight"]
                                    if real_missing:
                                        st.warning(f"⚠️ Отсутствуют веса: {real_missing[:5]}...")
                                
                                if hasattr(st.session_state.chat_model, "tie_weights"):
                                    st.session_state.chat_model.tie_weights()
                                
                                st.session_state.chat_model = st.session_state.chat_model.to(device)
                                st.session_state.chat_model.eval()
                                st.session_state.chat_model_path = str(model_path)
                                st.session_state.messages = []
                                st.success("✅ Модель загружена (fallback метод)!")
                                st.rerun()
                            except Exception as e2:
                                st.error(f"Fallback загрузка тоже не удалась: {e2}")
                                import traceback
                                st.code(traceback.format_exc())
            else:
                # === Информационная панель загруженной модели ===
                backend_type = st.session_state.get("chat_backend_type", "transformers")
                backend_emoji = "⚡" if backend_type == "vllm" else "🔧"
                
                st.success(f"{backend_emoji} Модель загружена: **{selected_model_name}**")
                
                # Компактная информация о модели
                info_cols = st.columns([2, 1, 1])
                with info_cols[0]:
                    st.caption(f"Backend: {backend_type.upper()}")
                with info_cols[1]:
                    if st.session_state.chat_has_template:
                        st.caption("✅ Chat template")
                        # Показываем подсказку о шаблоне
                        tokenizer = st.session_state.chat_tokenizer
                        if tokenizer and hasattr(tokenizer, 'chat_template'):
                            template_preview = str(tokenizer.chat_template)[:100]
                            if len(template_preview) == 100:
                                template_preview += "..."
                            st.caption(f"```{template_preview[:50]}...```")
                    else:
                        st.caption("⚠️ Нет chat template")
                        st.caption("Только режим Completion")
                with info_cols[2]:
                    if selected_model.get("is_lora"):
                        st.caption("🔧 LoRA")
                
                # Кнопки действий
                action_cols = st.columns([1, 1, 1])
                with action_cols[0]:
                    if st.button("🗑️ Выгрузить", help="Освободить память"):
                        st.session_state.chat_model = None
                        st.session_state.chat_backend = None
                        st.session_state.chat_tokenizer = None
                        st.session_state.chat_model_path = None
                        st.session_state.messages = []
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        st.rerun()
                
                with action_cols[1]:
                    if st.button("🔄 Очистить чат"):
                        st.session_state.messages = []
                        st.rerun()
                
                with action_cols[2]:
                    # Кнопка экспорта (для чекпоинтов)
                    if st.session_state.chat_model is not None:
                        if st.button("💾 Экспорт HF"):
                            with st.spinner("Экспорт модели..."):
                                export_path = export_model_to_hf(
                                    st.session_state.chat_model, 
                                    st.session_state.chat_tokenizer, 
                                    st.session_state.chat_model_path
                                )
                                if export_path:
                                    st.success(f"Экспортировано: `{export_path}`")
                                    time.sleep(2)
                                    st.rerun()

                # --- НАСТРОЙКИ СИСТЕМНОГО ПРОМПТА ---
                with st.expander(f"💬 {t('chat.system_prompt')}", expanded=False):
                    # Предустановленные промпты
                    preset_prompts = {
                        "Нет": "",
                        "Ассистент": "Ты — полезный ИИ-ассистент. Отвечай точно и по делу.",
                        "Reasoning": "Ты — ИИ для решения задач. Сначала рассуждай пошагово в теге <think>, затем дай ответ.",
                        "Программист": "Ты — опытный программист. Пиши чистый, читаемый код с комментариями.",
                        "Переводчик": "Ты — профессиональный переводчик. Переводи текст точно, сохраняя стиль.",
                        "Кастомный": None
                    }
                    preset_keys = list(preset_prompts.keys())
                    
                    # Инициализация системного промпта если не существует
                    if "system_prompt" not in st.session_state:
                        st.session_state.system_prompt = ""
                    if "chat_system_prompt_preset" not in st.session_state:
                        st.session_state.chat_system_prompt_preset = "Нет"
                    
                    # Определяем текущий индекс из session_state (сохраняется при очистке чата)
                    current_preset = st.session_state.get("chat_system_prompt_preset", "Нет")
                    current_index = preset_keys.index(current_preset) if current_preset in preset_keys else 0
                    
                    preset = st.selectbox(
                        "Шаблон",
                        options=preset_keys,
                        index=current_index,
                        key="system_prompt_preset_widget"
                    )
                    # Сохраняем выбор в отдельный ключ (не виджетный)
                    st.session_state.chat_system_prompt_preset = preset
                    
                    if preset != "Кастомный" and preset != "Нет":
                        st.session_state.system_prompt = preset_prompts[preset]
                        st.code(preset_prompts[preset], language=None)
                    elif preset == "Кастомный":
                        system_prompt_input = st.text_area(
                            "Свой промпт:",
                            value=st.session_state.get("system_prompt", ""),
                            height=100,
                            key="system_prompt_input"
                        )
                        st.session_state.system_prompt = system_prompt_input.strip()
                    else:
                        # "Нет" - НЕ сбрасываем если был Кастомный с текстом
                        if st.session_state.get("chat_system_prompt_preset") != "Кастомный":
                            st.session_state.system_prompt = ""
                        st.caption("Используется дефолтный промпт модели")
                
                # --- ИНТЕРФЕЙС ЧАТА С ФИКСИРОВАННЫМ СКРОЛЛОМ ---
                chat_container = st.container(height=500) # Прокручиваемый контейнер
                
                with chat_container:
                    # Показываем историю чата
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                
                # Ввод пользователя (всегда внизу)
                if prompt := st.chat_input("Введите сообщение..."):
                    # Добавляем сообщение пользователя
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Обновляем контейнер (показываем сообщение юзера сразу)
                    with chat_container:
                        with st.chat_message("user"):
                            st.write(prompt)
                    
                    # Генерируем ответ
                    with chat_container: # Ответ тоже пишем в контейнер
                        with st.chat_message("assistant"):
                            with st.spinner("Генерация..."):
                                try:
                                    chat_backend = st.session_state.chat_backend
                                    tokenizer = st.session_state.chat_tokenizer
                                    model = st.session_state.chat_model
                                    backend_type = st.session_state.get("chat_backend_type", "transformers")
                                    
                                    # Берем историю + новое сообщение
                                    conversation = st.session_state.messages.copy()
                                    
                                    has_template = st.session_state.chat_has_template
                                    use_chat_template = (prompt_mode == "chat") and has_template
                                    # Если каким-то образом выбран chat режим без template - fallback
                                    if prompt_mode == "chat" and not has_template:
                                        use_chat_template = False

                                    # Обработка системного промпта (только для режима chat_template)
                                    if use_chat_template:
                                        system_prompt = st.session_state.get("system_prompt", "").strip()
                                        
                                        if conversation and conversation[0].get("role") == "system":
                                            conversation.pop(0)
                                        
                                        if system_prompt:
                                            conversation.insert(0, {"role": "system", "content": system_prompt})
                                    
                                    # Формируем prompt_text
                                    if use_chat_template:
                                        prompt_text = chat_backend.apply_chat_template(
                                            conversation, 
                                            add_generation_prompt=True
                                        )
                                    else:
                                        prompt_text = ""
                                        for m in conversation:
                                            prompt_text += f"{m['role']}: {m['content']}\n"
                                        prompt_text += "assistant: "
                                    
                                    # === Генерация через backend ===
                                    response = chat_backend.generate(
                                        prompt=prompt_text,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        top_p=top_p if top_p < 1.0 else None,  # 1.0 = отключено
                                        top_k=top_k if top_k > 0 else None,    # 0 = отключено
                                    )
                                    
                                    st.write(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                except Exception as e:
                                    import traceback
                                    st.session_state.last_chat_error = traceback.format_exc()
                                    st.error(f"Ошибка генерации: {e}")
                                    st.code(st.session_state.last_chat_error)
        else:
            st.info("Нет обученных моделей. Запустите тренировку во вкладке 'Запуск'!")
            
            # Показываем где искать модели
            st.markdown("""
            **Модели будут доступны после тренировки:**
            - `out/*/final_model/` — финальные модели
            - `out/*/checkpoint_*/` — чекпоинты
            """)

    persist_user_preferences_if_changed(USER_PREFS_FILE)


if __name__ == "__main__":
    main()


