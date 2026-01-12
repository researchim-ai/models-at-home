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

# Custom CSS — чистая тёмная тема с хорошим контрастом
st.markdown("""
<style>
    /* Заголовок */
    .main-header {
        color: #ff6b6b;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #888888;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Статус */
    .status-running {
        color: #22c55e !important;
        font-weight: bold;
    }
    
    .status-completed {
        color: #3b82f6 !important;
        font-weight: bold;
    }
    
    .status-error {
        color: #ef4444 !important;
        font-weight: bold;
    }
    
    /* ASCII art блок */
    .model-ascii {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        color: #00ff88;
        white-space: pre;
        overflow-x: auto;
    }
    
    /* Карточки метрик */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Кнопки запуска */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        color: white !important;
        border: none;
        font-weight: 600;
    }
    
    /* Code блоки */
    pre {
        background: #0d1117 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }
    
    code {
        color: #79c0ff !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Persistence — сохранение состояния между перезагрузками
# ============================================================================

ACTIVE_RUN_FILE = RUNS_DIR / "active_run.json"


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
    if ACTIVE_RUN_FILE.exists():
        ACTIVE_RUN_FILE.unlink()


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


# Описания типов параллелизма
PARALLEL_TYPES = {
    "default": {
        "name": "Single GPU / CPU",
        "type": "None",
        "description": "Обучение на одном устройстве без параллелизма",
        "icon": "🖥️",
    },
    "multi_gpu": {
        "name": "Multi-GPU (DDP)",
        "type": "Data Parallel",
        "description": "Distributed Data Parallel — каждая GPU получает копию модели и часть батча",
        "icon": "🔄",
    },
    "fsdp": {
        "name": "FSDP",
        "type": "Data Parallel + Model Parallel",
        "description": "Fully Sharded Data Parallel — модель шардируется между GPU (PyTorch native)",
        "icon": "⚡",
    },
    "deepspeed_zero2": {
        "name": "DeepSpeed ZeRO-2",
        "type": "Data Parallel + Optimizer Parallel",
        "description": "Шардирование оптимизатора и градиентов между GPU",
        "icon": "🚀",
    },
    "deepspeed_zero3": {
        "name": "DeepSpeed ZeRO-3",
        "type": "Full Model Parallel",
        "description": "Полное шардирование: модель + оптимизатор + градиенты",
        "icon": "💪",
    },
    "deepspeed_zero3_offload": {
        "name": "ZeRO-3 + CPU Offload",
        "type": "Model Parallel + CPU Offload",
        "description": "Полное шардирование + выгрузка на CPU для экономии VRAM",
        "icon": "🧊",
    },
}


def estimate_parameters(
    hidden_size: int,
    num_layers: int,
    vocab_size: int = 50257,
    intermediate_size: int | None = None,
) -> int:
    """
    Оценка количества параметров для нашей `HomeModel` (см. `homellm/models/home_model.py`).

    ВАЖНО:
    - У нас RoPE (позиционные эмбеддинги НЕ обучаемые) => `seq_len` на число параметров не влияет.
    - lm_head weight tied к embed_tokens => не удваиваем vocab*hidden.
    """
    h = int(hidden_size)
    l = int(num_layers)
    v = int(vocab_size)
    i = int(intermediate_size) if intermediate_size is not None else int(h * 4)

    # Embedding
    embed = v * h

    # Per-layer:
    # - Attention: q/k/v/out, bias=False => 4 * (H*H)
    attn = 4 * h * h
    # - SwiGLU MLP: w1(H->I), w2(I->H), w3(H->I), bias=False => 3 * (H*I)
    mlp = 3 * h * i
    # - RMSNorm weights: 2 * H
    norms = 2 * h

    # Final norm
    final_norm = h

    return int(embed + l * (attn + mlp + norms) + final_norm)


def format_params(n: int) -> str:
    """Форматирование количества параметров."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def format_time(seconds: float) -> str:
    """Форматирование времени."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


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
                    
                    latest["status"] = latest.get("status", "training")
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
    run_id = datetime.now().strftime("grpo_%Y%m%d_%H%M%S")
    
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
            "-m", "homellm.training.rl.train_gsm8k",
            "--config_json", config_json,
        ]
    else:
        # Обычный запуск (single GPU или CPU)
        cmd = [
            sys.executable, "-m", "homellm.training.rl.train_gsm8k",
            "--config_json", config_json,
        ]
    
    # Сохраняем команду для отладки (без длинного JSON)
    cmd_path = run_dir / "command.txt"
    with open(cmd_path, "w") as f:
        if distributed_mode != "default" and config_file:
            f.write(f"accelerate launch --config_file {config_file} --num_processes {num_gpus} -m homellm.training.rl.train_gsm8k --config_json <config from {config_path}>")
        else:
            f.write(f"{sys.executable} -m homellm.training.rl.train_gsm8k --config_json <config from {config_path}>")
    
    stdout_file = open(stdout_path, "w")
    stderr_file = open(stderr_path, "w")
    
    # ВАЖНО: применяем выбор GPU из UI (как в start_training)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    gpu_ids = config.get("gpu_ids") or []
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logger.info(f"🎯 Используются GPU: {gpu_ids}")
    
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
    st.markdown("# 🏠 Models at Home Training Studio")
    st.caption("Визуальный интерфейс для тренировки языковых моделей дома")


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
        st.error(f"Ошибка чтения файла: {e}")
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
    """Универсальный конфигуратор SFT — автодетект + ручной выбор."""
    st.markdown("### 🛠️ Настройка данных для SFT")
    
    columns, sample = get_dataset_columns(data_path)
    
    if not sample:
        st.error("Не удалось прочитать файл или он пуст.")
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
        st.markdown("#### 📄 Пример записи:")
        with st.container(height=500):
            st.json(sample, expanded=True)
    
    # ===== ПРАВАЯ КОЛОНКА: Конфигурация =====
    with col_config:
        # Показываем что автодетект нашел
        if detected_format == "chat":
            st.success(f"🔍 **Автодетект:** найден Chat-формат в поле `{chat_field}`")
        else:
            st.info("🔍 **Автодетект:** Instruct-формат (отдельные поля)")
        
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
            st.markdown("#### 💬 Настройка Chat-формата")
            
            # Выбор поля со списком сообщений
            list_fields = [k for k, v in sample.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
            
            if not list_fields:
                st.error("❌ Не найдено полей со списком сообщений!")
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
            st.markdown("#### 📝 Настройка Instruct-формата")
            st.caption("Выберите поля для каждой роли:")
            
            # Все доступные пути
            field_options = ["(не выбрано)"] + all_paths
            
            system_path = st.selectbox("⚙️ **System** (опционально):", field_options, index=0, key="sft_inst_sys")
            user_path = st.selectbox("👤 **User** (вопрос/инструкция):", field_options, index=0, key="sft_inst_user")
            assistant_path = st.selectbox("🤖 **Assistant** (ответ):", field_options, index=0, key="sft_inst_asst")
            
            if user_path == "(не выбрано)" or assistant_path == "(не выбрано)":
                st.warning("👆 Выберите поля **User** и **Assistant**")
                return {}
            
            sft_columns = {
                "format": "instruct",
                "instruction": user_path,
                "output": assistant_path,
                "system_field": system_path if system_path != "(не выбрано)" else ""
            }
        
        # ===== НАСТРОЙКИ ШАБЛОНА =====
        st.markdown("---")
        with st.expander("🏷️ Теги и системный промпт", expanded=False):
            default_system = st.text_input("System prompt (по умолч.):", "You are a helpful assistant.", key="sft_def_sys")
            tc1, tc2 = st.columns(2)
            user_tag = tc1.text_input("User tag:", "### User:", key="sft_tag_user")
            assistant_tag = tc2.text_input("Assistant tag:", "### Assistant:", key="sft_tag_asst")
        
        if 'default_system' not in dir():
            default_system, user_tag, assistant_tag = "You are a helpful assistant.", "### User:", "### Assistant:"
        
        sft_template = {
            "system": default_system,
            "separator": "\n\n",
            "user_tag": user_tag,
            "bot_tag": assistant_tag
        }
        
        # ===== ПРЕВЬЮ =====
        st.markdown("---")
        st.markdown("#### 👁️ Превью:")
        
        try:
            sep = "\n\n"
            preview = ""
            
            if sft_columns["format"] == "chat":
                messages = sample[sft_columns["messages_path"]]
                sys_text = default_system
                
                for msg in messages:
                    role = str(msg.get(sft_columns["role_field"], ""))
                    content = str(msg.get(sft_columns["content_field"], ""))
                    
                    if role == sft_columns["role_system"]:
                        sys_text = content
                    elif role == sft_columns["role_user"]:
                        preview += f"{user_tag}\n{content[:200]}{'...' if len(content) > 200 else ''}{sep}"
                    elif role == sft_columns["role_assistant"]:
                        preview += f"{assistant_tag}\n{content[:200]}{'...' if len(content) > 200 else ''}{sep}"
                
                preview = f"{sys_text}{sep}" + preview + "<|endoftext|>"
            else:
                user_val = str(get_nested_value(sample, sft_columns["instruction"]) or "")[:300]
                asst_val = str(get_nested_value(sample, sft_columns["output"]) or "")[:300]
                
                # System prompt: сначала пытаемся получить из семпла, если указано поле
                sys_val = default_system
                system_field = sft_columns.get("system_field")
                
                # Если указано поле system_field, пытаемся получить значение из семпла
                if system_field and system_field != "(не выбрано)" and system_field.strip():
                    field_sys = get_nested_value(sample, system_field)
                    # Если значение найдено и не пустое, используем его вместо дефолтного
                    if field_sys is not None:
                        field_sys_str = str(field_sys).strip()
                        if field_sys_str:
                            sys_val = field_sys_str[:200]
                
                # Формируем превью: системный промпт всегда показываем в начале
                # ВАЖНО: системный промпт должен быть виден, даже если он дефолтный
                preview = f"{sys_val}{sep}{user_tag}\n{user_val}{sep}{assistant_tag}\n{asst_val}<|endoftext|>"
            
            with st.container(height=400):
                st.code(preview, language=None)
            
            st.success("✅ Готово!")
            
        except Exception as e:
            st.error(f"Ошибка: {e}")

    return {"sft_columns": sft_columns, "sft_template": sft_template}


# ============================================================================
# GRPO Configuration
# ============================================================================

def render_grpo_sidebar_config():
    """Конфигурация GRPO в сайдбаре."""
    st.sidebar.subheader("🧠 Параметры GRPO")
    
    # Алгоритм
    algorithm = st.sidebar.selectbox(
        "Алгоритм",
        ["grpo", "drgrpo", "dapo"],
        format_func=lambda x: {
            "grpo": "GRPO (стандартный)",
            "drgrpo": "Dr.GRPO (улучшенный)",
            "dapo": "DAPO (полный)",
        }[x],
        help="""
        **GRPO**: Стандартный Group Relative Policy Optimization
        **Dr.GRPO**: Без деления на std, фиксированная нормализация
        **DAPO**: + асимметричный клиппинг, + dynamic sampling
        """
    )
    
    # Генерация
    group_size = st.sidebar.slider(
        "Group size (G)",
        min_value=8,
        max_value=32,
        value=8,
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
        max_value=4096,
        value=1024,
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
    
    # Обучение
    grpo_learning_rate = st.sidebar.select_slider(
        "Learning Rate (GRPO)",
        options=[1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5],
        value=5e-6,
        format_func=lambda x: f"{x:.0e}",
        help="Для RL обычно требуется меньший LR чем для SFT"
    )
    
    train_batch_size = st.sidebar.slider(
        "Train Batch Size",
        min_value=1,
        max_value=16,
        value=2,
        step=1,
        help="Размер микро-батча при обучении на опыте. Уменьшите до 1-2, если возникает OOM (как в re-grpo)."
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
    
    # KL
    kl_weight = st.sidebar.slider(
        "KL weight",
        min_value=0.0,
        max_value=0.1,
        value=0.0,
        step=0.01,
        help="Вес KL-штрафа. Для reasoning обычно 0"
    )
    
    # Клиппинг
    with st.sidebar.expander("⚙️ Продвинутые параметры"):
        clip_eps_low = st.slider("Clip ε (low)", 0.1, 0.3, 0.2, 0.01)
        clip_eps_high = st.slider(
            "Clip ε (high)", 
            0.1, 0.4, 
            0.28 if algorithm == "dapo" else 0.2, 
            0.01,
            help="DAPO рекомендует 0.28 для верхней границы"
        )
        
        dynamic_sampling = st.checkbox(
            "Dynamic sampling",
            value=algorithm == "dapo",
            help="Фильтровать группы с нулевым градиентом"
        )
        
        token_level_loss = st.checkbox(
            "Token-level loss",
            value=algorithm == "dapo",
            help="Агрегировать loss по токенам, а не по сэмплам"
        )

        min_lr_ratio = st.slider(
            "Min LR ratio (floor)",
            0.0, 0.5,
            0.1,
            0.01,
            help="Нижний предел LR: lr = base_lr * ratio в конце cosine. 0.0 = до нуля."
        )
    
    # ВАЖНО: LoRA параметры и квантизация берутся из render_model_config() (секция "🎯 Метод тюнинга")
    # Здесь мы НЕ дублируем их, чтобы избежать повторения в UI
    # Все LoRA параметры (use_lora, lora_r, lora_alpha, lora_dropout, lora_target_modules)
    # и квантизация (use_4bit, use_8bit) будут взяты из model_config
    
    return {
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
        "grpo_token_level_loss": token_level_loss,
        "grpo_min_lr_ratio": min_lr_ratio,
    }


def render_grpo_main_config(data_path: str = None):
    """Универсальный конструктор Reward функций с визуальным редактором правил."""
    import re
    import json as json_lib
    
    # =========================================================================
    # 1. ДАТАСЕТ (первая секция!)
    # =========================================================================
    st.markdown("### 📚 Датасет для Reasoning")
    
    grpo_dataset_path = None
    grpo_max_samples = None
    grpo_dataset_language = "en"
    dataset_source = "custom"
    dataset_key = "custom"
    
    # Получаем список локальных датасетов
    local_datasets = []
    if DATASET_DIR.exists():
        for f in sorted(DATASET_DIR.iterdir(), key=lambda x: x.name.lower()):
            if f.suffix in (".jsonl", ".json"):
                local_datasets.append(f)
    
    # Формируем список для выбора (все датасеты)
    dataset_options = ["-- Выберите датасет --"]
    for f in local_datasets:
        dataset_options.append(str(f))
    
    # Если data_path передан, используем его
    default_idx = 0
    if data_path and data_path in dataset_options:
        default_idx = dataset_options.index(data_path)
    elif data_path:
        # Проверяем по имени файла
        for i, opt in enumerate(dataset_options):
            if data_path in opt:
                default_idx = i
                break
    
    selected_dataset = st.selectbox(
        "Выберите датасет",
        options=dataset_options,
        index=default_idx,
        help="Скачайте датасеты на вкладке 'Данные' → 🧠 Reasoning: GSM8K, MATH-RU и др."
    )
    
    # Обрабатываем выбор
    if selected_dataset and not selected_dataset.startswith("--"):
        grpo_dataset_path = selected_dataset
        st.success(f"✅ Выбран: `{Path(selected_dataset).name}`")
        
        # Определяем язык по имени файла
        if "ru" in selected_dataset.lower() or "russian" in selected_dataset.lower():
            grpo_dataset_language = "ru"
        else:
            grpo_dataset_language = "en"
    else:
        st.warning("⚠️ Выберите датасет или скачайте его на вкладке **💾 Данные** → 🧠 Reasoning")
    
    # Настройки датасета
    col1, col2 = st.columns(2)
    with col1:
        grpo_max_samples = st.number_input(
            "Макс. примеров (0 = все)",
            min_value=0,
            max_value=50000,
            value=0,
            step=100,
            help="Ограничить количество примеров для обучения"
        )
    with col2:
        grpo_dataset_language = st.selectbox(
            "Язык датасета",
            ["en", "ru"],
            index=1 if grpo_dataset_language == "ru" else 0,
            format_func=lambda x: "🇬🇧 English" if x == "en" else "🇷🇺 Русский",
        )
    
    st.markdown("---")

    # Оценка размера датасета для лимита "Max prompts" в sidebar
    try:
        effective_size = None
        if grpo_max_samples and int(grpo_max_samples) > 0:
            # Если пользователь уже ограничил max_samples — используем это как "эффективный размер"
            effective_size = int(grpo_max_samples)
        elif grpo_dataset_path:
            p = Path(grpo_dataset_path)
            if p.exists() and p.suffix == ".jsonl":
                # Для reasoning датасетов это обычно не гигабайты — считаем честно
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    effective_size = sum(1 for _ in f if _.strip())
            elif p.exists() and p.suffix == ".json":
                import json as _json
                with open(p, "r", encoding="utf-8") as f:
                    obj = _json.load(f)
                if isinstance(obj, list):
                    effective_size = len(obj)
        if isinstance(effective_size, int) and effective_size > 0:
            st.session_state["grpo_effective_dataset_size"] = int(effective_size)
            st.caption(f"Размер датасета для лимита: **{effective_size} prompts**")
    except Exception:
        # не мешаем запуску UI, даже если не смогли посчитать
        pass
    
    # =========================================================================
    # 2. REWARD DESIGNER
    # =========================================================================
    st.markdown("### 🎯 Reward Designer")
    st.caption("Создавайте гибкие правила вознаграждения с условиями, паттернами и формулами")
    
    # =========================================================================
    # Песочница — Тестовые данные
    # =========================================================================
    st.markdown("#### 🧪 Песочница для проектирования")
    
    with st.expander("📝 Тестовые данные", expanded=True):
        col_left, col_right = st.columns(2)
        
        with col_left:
            sample_prompt = st.text_area(
                "**Промпт** (вопрос/задача)",
                value="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                height=100,
                key="sample_prompt",
            )
            sample_reference = st.text_input(
                "**Эталонный ответ**",
                value="72",
                key="sample_reference",
            )
        
        with col_right:
            st.markdown("**Ответ модели** (редактируйте для тестов):")
            # Пресеты ответов
            response_presets = {
                "✅ Правильный с тегами": """<reasoning>
In April, Natalia sold 48 clips.
In May, she sold half as many: 48 / 2 = 24 clips.
Total: 48 + 24 = 72 clips.
</reasoning>
<answer>72</answer>""",
                "⚠️ Неправильный ответ": """<reasoning>
April: 48 clips
May: 48 / 2 = 24 clips  
Total: 48 + 24 = 70 clips
</reasoning>
<answer>70</answer>""",
                "❌ Без тегов": "Natalia sold 48 clips in April and 24 in May, so 72 total.",
                "🔁 С повторами": """<reasoning>
Let me solve this step by step.
Step by step I will solve this.
In April she sold 48 clips.
In April she sold 48 clips.
Total is 72.
</reasoning>
<answer>72</answer>""",
            }
            preset = st.selectbox("Быстрый выбор:", list(response_presets.keys()))
            current_response = st.text_area(
                "Ответ модели",
                value=response_presets[preset],
                height=180,
                key="current_test_response",
                label_visibility="collapsed",
            )
    
    st.markdown("---")
    
    # =========================================================================
    # Универсальный конструктор правил
    # =========================================================================
    st.markdown("#### 🏗️ Конструктор Reward-правил")
    
    # Справка по переменным
    with st.expander("📖 Справка: переменные и синтаксис", expanded=False):
        st.markdown("""
**Доступные переменные:**
- `{{response}}` — ответ модели
- `{{reference}}` — эталонный ответ  
- `{{prompt}}` — исходный промпт
- `{{extracted.имя}}` — значение, извлечённое regex-группой

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
                rule["name"] = st.text_input("Название", value=rule["name"], key=f"rule_name_{rule['id']}")
            with c2:
                rule["weight"] = st.number_input("Вес", 0.0, 10.0, float(rule["weight"]), 0.1, key=f"rule_weight_{rule['id']}")
            with c3:
                rule["enabled"] = st.checkbox("Вкл", value=rule["enabled"], key=f"rule_enabled_{rule['id']}")
            with c4:
                if st.button("🗑️", key=f"rule_del_{rule['id']}"):
                    st.session_state.reward_rules.pop(idx)
                    st.rerun()
            
            # === EXTRACTORS (regex для извлечения значений) ===
            st.markdown("##### 🔍 Экстракторы (regex)")
            st.caption("Извлекают значения из текста в переменные `{{extracted.имя}}`")
            
            if "extractors" not in rule:
                rule["extractors"] = []
            
            for ei, ext in enumerate(rule["extractors"]):
                ec1, ec2, ec3, ec4 = st.columns([2, 4, 2, 1])
                with ec1:
                    ext["name"] = st.text_input("Имя", value=ext.get("name", f"var{ei}"), key=f"ext_name_{rule['id']}_{ei}")
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
            
            if st.button("➕ Добавить экстрактор", key=f"add_ext_{rule['id']}"):
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
                            cond["value"] = st.text_input("Значение", value=cond.get("value", ""), key=f"cond_val_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type in ["matches", "not_matches"]:
                            cond["pattern"] = st.text_input("Regex", value=cond.get("pattern", ""), key=f"cond_pat_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type == "equals_numeric":
                            right_opts = ["{{reference}}"] + [f"{{{{extracted.{e['name']}}}}}" for e in rule.get("extractors", [])]
                            right_val = cond.get("right", "{{reference}}")
                            if right_val not in right_opts:
                                right_opts.append(right_val)
                            cond["right"] = st.selectbox("Сравнить с", right_opts, index=right_opts.index(right_val) if right_val in right_opts else 0, key=f"cond_right_{rule['id']}_{ci}", label_visibility="collapsed")
                            cond["tolerance"] = st.number_input("±", 0.0, 100.0, float(cond.get("tolerance", 0.01)), 0.01, key=f"cond_tol_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type in ["greater", "less"]:
                            cond["value"] = st.number_input("Число", value=float(cond.get("value", 0)), key=f"cond_num_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type == "length_between":
                            lc1, lc2 = st.columns(2)
                            cond["min"] = lc1.number_input("Мин", 0, 100000, int(cond.get("min", 10)), key=f"cond_min_{rule['id']}_{ci}")
                            cond["max"] = lc2.number_input("Макс", 0, 100000, int(cond.get("max", 5000)), key=f"cond_max_{rule['id']}_{ci}")
                        elif new_type == "length_min":
                            cond["min"] = st.number_input("Мин длина", 0, 100000, int(cond.get("min", 10)), key=f"cond_minl_{rule['id']}_{ci}", label_visibility="collapsed")
                        elif new_type == "length_max":
                            cond["max"] = st.number_input("Макс длина", 0, 100000, int(cond.get("max", 5000)), key=f"cond_maxl_{rule['id']}_{ci}", label_visibility="collapsed")
                    
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
                if st.button("➕ Добавить условие", key=f"add_cond_{rule['id']}"):
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
        if st.button("➕ Пустое правило", type="secondary"):
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
        if st.button("➕ Добавить шаблон", type="primary"):
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
                st.warning("Выберите шаблон")
            st.rerun()
    
    # =========================================================================
    # ТЕСТИРОВАНИЕ
    # =========================================================================
    st.markdown("---")
    st.markdown("#### ▶️ Тестирование правил")
    
    def evaluate_rules(response: str, reference: str, prompt: str, rules: list) -> list:
        """Вычисляет reward для каждого правила."""
        results = []
        
        for rule in rules:
            if not rule.get("enabled", True):
                continue
            
            result = {
                "name": rule["name"],
                "weight": rule["weight"],
                "extracted": {},
                "conditions_met": [],
                "all_conditions_true": False,
                "raw_reward": 0.0,
                "weighted_reward": 0.0,
                "details": "",
            }
            
            # 1. Экстракторы
            for ext in rule.get("extractors", []):
                source = ext.get("source", "{{response}}")
                source_text = source.replace("{{response}}", response).replace("{{reference}}", reference).replace("{{prompt}}", prompt)
                
                pattern = ext.get("pattern", "")
                flags = 0
                if "DOTALL" in ext.get("flags", ""):
                    flags |= re.DOTALL
                if "IGNORECASE" in ext.get("flags", ""):
                    flags |= re.IGNORECASE
                
                try:
                    match = re.search(pattern, source_text, flags)
                    if match:
                        # Поддержка именованных групп и обычных
                        if match.groups():
                            result["extracted"][ext["name"]] = match.group(1)
                        elif match.groupdict():
                            result["extracted"].update(match.groupdict())
                        else:
                            result["extracted"][ext["name"]] = match.group(0)
                    else:
                        result["extracted"][ext["name"]] = ""
                except re.error as e:
                    result["extracted"][ext["name"]] = f"REGEX_ERROR: {e}"
            
            # 2. Условия
            conditions_results = []
            for cond in rule.get("conditions", []):
                cond_type = cond.get("type", "contains")
                
                # Подставляем переменные
                def substitute_vars(text):
                    if not isinstance(text, str):
                        return text
                    text = text.replace("{{response}}", response)
                    text = text.replace("{{reference}}", reference)
                    text = text.replace("{{prompt}}", prompt)
                    for k, v in result["extracted"].items():
                        text = text.replace(f"{{{{extracted.{k}}}}}", str(v) if v else "")
                    return text
                
                target = substitute_vars(cond.get("target") or cond.get("left", ""))
                
                cond_met = False
                cond_detail = ""
                
                try:
                    if cond_type == "contains":
                        value = substitute_vars(cond.get("value", ""))
                        cond_met = value in target
                        cond_detail = f"'{value[:30]}...' {'✓ найдено' if cond_met else '✗ не найдено'}"
                    
                    elif cond_type == "not_contains":
                        value = substitute_vars(cond.get("value", ""))
                        cond_met = value not in target
                        cond_detail = f"'{value[:30]}' {'✓ не содержится' if cond_met else '✗ содержится'}"
                    
                    elif cond_type == "matches":
                        pattern = cond.get("pattern", "")
                        cond_met = bool(re.search(pattern, target))
                        cond_detail = f"regex {'✓' if cond_met else '✗'}"
                    
                    elif cond_type == "not_matches":
                        pattern = cond.get("pattern", "")
                        cond_met = not bool(re.search(pattern, target))
                        cond_detail = f"not regex {'✓' if cond_met else '✗'}"
                    
                    elif cond_type == "equals":
                        value = substitute_vars(cond.get("value", ""))
                        cond_met = target.strip() == value.strip()
                        cond_detail = f"{'✓' if cond_met else '✗'} '{target[:20]}' == '{value[:20]}'"
                    
                    elif cond_type == "equals_numeric":
                        right = substitute_vars(cond.get("right", ""))
                        tolerance = float(cond.get("tolerance", 0.01))
                        try:
                            left_num = float(re.sub(r"[^\d.\-]", "", str(target)))
                            right_num = float(re.sub(r"[^\d.\-]", "", str(right)))
                            cond_met = abs(left_num - right_num) <= tolerance
                            cond_detail = f"{left_num} {'=' if cond_met else '≠'} {right_num} (±{tolerance})"
                        except:
                            cond_detail = "✗ не числа"
                    
                    elif cond_type == "greater":
                        value = float(cond.get("value", 0))
                        try:
                            num = float(re.sub(r"[^\d.\-]", "", str(target)))
                            cond_met = num > value
                            cond_detail = f"{num} {'>' if cond_met else '≤'} {value}"
                        except:
                            cond_detail = "✗ не число"
                    
                    elif cond_type == "less":
                        value = float(cond.get("value", 0))
                        try:
                            num = float(re.sub(r"[^\d.\-]", "", str(target)))
                            cond_met = num < value
                            cond_detail = f"{num} {'<' if cond_met else '≥'} {value}"
                        except:
                            cond_detail = "✗ не число"
                    
                    elif cond_type == "length_between":
                        length = len(target)
                        min_len = int(cond.get("min", 0))
                        max_len = int(cond.get("max", 99999))
                        cond_met = min_len <= length <= max_len
                        cond_detail = f"len={length} {'✓' if cond_met else '✗'} [{min_len}, {max_len}]"
                    
                    elif cond_type == "length_min":
                        length = len(target)
                        min_len = int(cond.get("min", 0))
                        cond_met = length >= min_len
                        cond_detail = f"len={length} >= {min_len} {'✓' if cond_met else '✗'}"
                    
                    elif cond_type == "length_max":
                        length = len(target)
                        max_len = int(cond.get("max", 99999))
                        cond_met = length <= max_len
                        cond_detail = f"len={length} <= {max_len} {'✓' if cond_met else '✗'}"
                    
                except Exception as e:
                    cond_detail = f"✗ ошибка: {e}"
                
                conditions_results.append({"met": cond_met, "detail": cond_detail})
            
            result["conditions_met"] = conditions_results
            
            # Логика объединения
            logic = rule.get("condition_logic", "all")
            if not conditions_results:
                result["all_conditions_true"] = True  # Нет условий = всегда true
            elif logic == "all":
                result["all_conditions_true"] = all(c["met"] for c in conditions_results)
            else:  # any
                result["all_conditions_true"] = any(c["met"] for c in conditions_results)
            
            # 3. Вычисление reward
            formula = rule.get("reward_formula", "1.0") if result["all_conditions_true"] else rule.get("else_reward", "0.0")
            
            # Подстановка переменных в формулу
            formula = formula.replace("{{response}}", f"'''{response}'''")
            formula = formula.replace("{{reference}}", f"'''{reference}'''")
            formula = formula.replace("{{prompt}}", f"'''{prompt}'''")
            for k, v in result["extracted"].items():
                safe_v = str(v).replace("'", "\\'") if v else ""
                formula = formula.replace(f"{{{{extracted.{k}}}}}", f"'''{safe_v}'''")
            
            try:
                # Безопасное вычисление
                safe_globals = {"__builtins__": {"len": len, "min": min, "max": max, "abs": abs, "float": float, "int": int, "str": str, "bool": bool}}
                result["raw_reward"] = float(eval(formula, safe_globals))
            except Exception as e:
                result["raw_reward"] = 0.0
                result["details"] = f"Ошибка формулы: {e}"
            
            result["weighted_reward"] = result["raw_reward"] * result["weight"]
            results.append(result)
        
        return results
    
    if st.button("▶️ Рассчитать Reward", type="primary", use_container_width=True):
        results = evaluate_rules(
            current_response, 
            sample_reference, 
            sample_prompt, 
            st.session_state.reward_rules
        )
        
        total_reward = sum(r["weighted_reward"] for r in results)
        
        # Показываем результаты
        st.markdown("##### 📊 Результаты")
        
        for r in results:
            status = "✅" if r["all_conditions_true"] else "❌"
            
            with st.container():
                c1, c2, c3 = st.columns([3, 2, 2])
                with c1:
                    st.markdown(f"**{status} {r['name']}**")
                with c2:
                    st.write(f"Raw: `{r['raw_reward']:.3f}`")
                with c3:
                    color = "green" if r["weighted_reward"] > 0 else ("red" if r["weighted_reward"] < 0 else "gray")
                    st.markdown(f"×{r['weight']} = **:{color}[{r['weighted_reward']:.3f}]**")
                
                # Детали
                if r["extracted"]:
                    st.caption(f"📦 Извлечено: {r['extracted']}")
                
                for ci, cond in enumerate(r["conditions_met"]):
                    icon = "✓" if cond["met"] else "✗"
                    st.caption(f"  {icon} {cond['detail']}")
                
                if r["details"]:
                    st.warning(r["details"])
        
        st.markdown("---")
        color = "green" if total_reward > 0 else "red"
        st.markdown(f"### 🎯 Итоговый Reward: :{color}[**{total_reward:.3f}**]")
        
        # График
        if results:
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Bar(
                    x=[r["name"] for r in results],
                    y=[r["weighted_reward"] for r in results],
                    marker_color=["#22c55e" if r["weighted_reward"] > 0 else "#ef4444" for r in results],
                    text=[f"{r['weighted_reward']:.2f}" for r in results],
                    textposition="outside"
                )
            ])
            fig.update_layout(
                title="Вклад каждого правила",
                yaxis_title="Weighted Reward",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # Формат reasoning
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 📝 Формат Reasoning")
    
    reasoning_format = st.selectbox(
        "Формат тегов",
        ["deepseek", "simple", "russian"],
        format_func=lambda x: {
            "deepseek": "DeepSeek (<think>...</think>, <answer>...</answer>)",
            "simple": "Simple (<reasoning>...</reasoning>, <answer>...</answer>)",
            "russian": "Russian (на русском языке)",
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
    }
    
    with st.expander("📋 Пример формата ответа"):
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
    
    return {
        "grpo_dataset_source": dataset_source,
        "grpo_dataset_key": dataset_key,
        "grpo_dataset_path": grpo_dataset_path,
        "grpo_dataset_language": grpo_dataset_language,
        "grpo_max_samples": grpo_max_samples if grpo_max_samples > 0 else None,
        "grpo_reward_rules": reward_rules,
        "grpo_reasoning_format": reasoning_format,
    }


def render_model_config():
    """Конфигуратор модели в сайдбаре."""
    st.sidebar.header("🧠 Архитектура и Режим")
    
    # Режим обучения
    stage_options = {
        "pretrain": "Pretraining (с нуля)",
        "continual_pretrain": "Continual Pretraining (продолжение)",
        "sft": "SFT (Fine-Tuning)",
        "grpo": "🧠 GRPO (RL для Reasoning)"
    }
    selected_stage = st.sidebar.selectbox(
        "Этап обучения",
        options=list(stage_options.keys()),
        format_func=lambda x: stage_options[x],
        help="Выберите этап: обучение с нуля, продолжение pretrain, дообучение (SFT) или RL обучение (GRPO)"
    )
    
    # Имя модели (для папки эксперимента)
    if selected_stage == "pretrain":
        model_name_default = "home_pretrain"
    elif selected_stage == "continual_pretrain":
        model_name_default = "home_continual_pretrain"
    elif selected_stage == "grpo":
        model_name_default = "home_grpo"
    else:
        model_name_default = "home_sft"
    model_name = st.sidebar.text_input("Название эксперимента", value=model_name_default, help="Имя папки для сохранения")
    
    base_model_path = None
    
    if selected_stage in ("sft", "continual_pretrain", "grpo"):
        stage_label = {"sft": "SFT", "continual_pretrain": "Continual Pretraining", "grpo": "GRPO"}.get(selected_stage, selected_stage)
        st.sidebar.subheader("📦 Базовая модель")
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
                "Выберите модель", 
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
                st.sidebar.success("✅ HuggingFace модель")
            
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
                    st.sidebar.success("✅ Параметры загружены из run_config.json")
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
                st.sidebar.warning("⚠️ Конфиг не найден, введите параметры вручную")
        except Exception as e:
             st.sidebar.error(f"Ошибка чтения config: {e}")

    st.sidebar.subheader("⚙️ Параметры модели")

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
            
        arch_options = ["HomeModel (GPT-2 style)", "Llama (Custom)", "Mistral (Custom)", "Custom Blueprint (Visual Builder)"]
        
        model_type = st.sidebar.selectbox(
            "Архитектура модели",
            options=arch_options,
            index=0,
            help="Выберите архитектуру. Custom Blueprint позволяет использовать модели из Visual Builder."
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
        
        st.sidebar.info("🔒 Архитектура зафиксирована (от базовой модели)")
        
        # seq_len МОЖНО менять - тренировать на меньшем контексте
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚙️ Контекст для обучения**")
        
        # Опции seq_len: стандартные + максимум модели
        seq_len_opts = [512, 1024, 2048, 4096, 8192]
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
            st.sidebar.caption("⚠️ Убедитесь, что параметры совпадают с базовой моделью!")

        # Пресеты
        preset = st.sidebar.selectbox(
            "Пресет",
            ["Tiny (25M)", "Small (80M)", "Medium (200M)", "Large (400M)", "Custom"],
            index=0
        )
        
        presets = {
            "Tiny (25M)": (512, 8, 8),
            "Small (80M)": (768, 12, 12),
            "Medium (200M)": (1024, 16, 16),
            "Large (400M)": (1280, 20, 20),
        }
        
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
        
        seq_len_opts = [512, 1024, 2048, 4096, 6144, 8192]
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
    
    # Оценка параметров (HomeModel RoPE/SwiGLU)
    # Используем loaded_config если есть, иначе дефолты
    vocab_size = int(loaded_config.get("vocab_size", 50257)) if loaded_config else 50257
    intermediate_size = int(loaded_config.get("intermediate_size") or (int(hidden_size) * 4)) if loaded_config else (int(hidden_size) * 4)
    est_params = estimate_parameters(
        hidden_size,
        num_layers,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
    )
    st.sidebar.metric("Параметры (≈)", format_params(est_params))
    
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
    st.sidebar.subheader("🎯 Метод тюнинга")
    tuning_method = st.sidebar.selectbox(
        "Метод",
        ["full", "lora", "qlora"],
        index=0,
        help="full: полный fine-tuning, lora: LoRA, qlora: QLoRA (4-bit + LoRA)"
    )
    
    lora_r = None
    lora_alpha = None
    lora_dropout = None
    lora_target_modules = None
    
    if tuning_method in ("lora", "qlora"):
        st.sidebar.markdown("**LoRA параметры:**")
        lora_r = st.sidebar.slider("LoRA r", min_value=4, max_value=128, value=16, step=4)
        lora_alpha = st.sidebar.slider("LoRA alpha", min_value=4, max_value=256, value=32, step=4)
        lora_dropout = st.sidebar.slider("LoRA dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        
        lora_target_modules_input = st.sidebar.text_input(
            "Target modules (опционально)",
            placeholder="q_proj,k_proj,v_proj,o_proj",
            help="Модули для LoRA (через запятую). Если пусто - автодетект"
        )
        if lora_target_modules_input:
            lora_target_modules = [m.strip() for m in lora_target_modules_input.split(",")]
    
    # Сборка конфига
    config = {
        "stage": selected_stage,
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
    else:
        config["model_type"] = "home"
        if "Llama" in model_type: config["arch_preset"] = "llama"
        if "Mistral" in model_type: config["arch_preset"] = "mistral"
        
    return config


def render_training_config():
    """Конфигуратор обучения в сайдбаре."""
    st.sidebar.header("⚙️ Параметры обучения")
    
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
    
    warmup_steps = st.sidebar.number_input(
        "Warmup Steps",
        min_value=0,
        max_value=10000,
        value=1000
    )

    scheduler_resync_on_resume = st.sidebar.checkbox(
        "Resync LR scheduler при resume (фикс для старых чекпоинтов)",
        value=True,
        help=(
            "Если чекпоинт был сохранён со scheduler, который шагал по micro-batch (а не по update-step), "
            "LR при resume может стать почти 0. Этот флаг принудительно выставляет scheduler на global_step."
        ),
    )
    
    # Выбор: epochs или max_steps
    training_mode = st.sidebar.radio(
        "Режим тренировки",
        ["По эпохам", "По шагам"],
        help="Выберите как определять длительность тренировки"
    )
    
    if training_mode == "По эпохам":
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
    
    mixed_precision = st.sidebar.selectbox(
        "Mixed Precision",
        ["no", "fp16", "bf16"],
        index=2,
        help="bf16 рекомендуется для Ampere+ GPU"
    )

    # Sharding mode: гарантирует отсутствие двойного шардинга и корректную семантику resume
    st.sidebar.divider()
    st.sidebar.subheader("🧩 Шардирование данных")
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
    
    grad_checkpoint = st.sidebar.checkbox(
        "Gradient Checkpointing",
        value=False,
        help="Экономит VRAM, но медленнее"
    )
    
    max_grad_norm = st.sidebar.number_input(
        "Max Gradient Norm",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Gradient clipping для стабильности (0 = отключить)"
    )
    
    # Validation / Eval
    st.sidebar.divider()
    st.sidebar.subheader("📊 Валидация")
    
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
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "learning_rate": learning_rate,
        "lr_schedule": lr_schedule,
        "min_lr_ratio": min_lr_ratio,
        "warmup_steps": warmup_steps,
        "scheduler_resync_on_resume": scheduler_resync_on_resume,
        "epochs": epochs,
        "max_steps": max_steps,
        "mixed_precision": mixed_precision,
        "grad_checkpoint": grad_checkpoint,
        "max_grad_norm": max_grad_norm,
        "sharding_mode": sharding_mode,
        "val_ratio": val_ratio,
        "eval_every": eval_every,
        "eval_batches": eval_batches,
    }


def render_dataset_config(stage="pretrain"):
    """Выбор датасета (только выбор файла)."""
    st.sidebar.header("📁 Датасет")
    
    datasets = get_available_datasets()
    
    if datasets:
        dataset_options = [f"{name} ({size})" for name, size in datasets]
        selected = st.sidebar.selectbox("Выберите датасет", dataset_options)
        selected_name = selected.split(" (")[0]
        data_path = str(DATASET_DIR / selected_name)
    else:
        st.sidebar.warning("Датасеты не найдены в datasets/")
        data_path = st.sidebar.text_input("Путь к датасету", "datasets/data.jsonl")
    
    return {"data_path": data_path}


def render_output_config(model_name="training_run"):
    """Конфигурация вывода."""
    st.sidebar.header("💾 Сохранение")
    
    # Автоматический путь: out/{model_name}
    default_dir = f"out/{model_name}"
    
    output_dir = st.sidebar.text_input(
        "Output Directory (Experiment Root)",
        value=default_dir,
        help="Корневая папка для всех запусков этого эксперимента"
    )
    
    save_every = st.sidebar.number_input(
        "Save Checkpoint Every N Steps",
        min_value=100,
        max_value=50000,
        value=200,
        step=100,
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
                st.sidebar.caption("✅ Финальная модель сохранена")
    
    return {
        "output_dir": output_dir,
        "save_every": save_every,
        "export_on_checkpoint": export_on_checkpoint,
        "log_every": log_every,
        "tokenizer_path": "gpt2"
    }


def get_available_models():
    """Получить список доступных обученных моделей (рекурсивный поиск)."""
    models = []
    
    # 1. Ищем рекурсивно в out/ (обученные модели)
    if OUTPUT_DIR.exists():
        # Ищем любые config.json внутри out/
        for config_file in OUTPUT_DIR.rglob("config.json"):
            model_dir = config_file.parent
            
            # Игнорируем папки, которые не похожи на модели (например, логи)
            # Критерий модели: наличие config.json + (pytorch_model.bin или model.safetensors или adapter_model.bin)
            has_weights = (
                (model_dir / "pytorch_model.bin").exists() or 
                (model_dir / "model.safetensors").exists() or
                (model_dir / "adapter_model.bin").exists()
            )
            
            if has_weights:
                # Определяем тип (final или checkpoint)
                m_type = "checkpoint" if "checkpoint" in model_dir.name else "final"
                if model_dir.name == "final_model": m_type = "final"
                
                # Формируем красивое имя
                # Берем путь относительно OUTPUT_DIR
                rel_path = model_dir.relative_to(OUTPUT_DIR)
                models.append({
                    "name": str(rel_path),
                    "path": str(model_dir),
                    "type": m_type,
                    "time": model_dir.stat().st_mtime
                })
    
    # 2. Ищем в models/ (скачанные с HuggingFace)
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                config_file = model_dir / "config.json"
                if config_file.exists():
                    # Проверяем наличие весов
                    has_weights = (
                        (model_dir / "pytorch_model.bin").exists() or 
                        (model_dir / "model.safetensors").exists() or
                        any(model_dir.glob("*.safetensors")) or
                        any(model_dir.glob("pytorch_model*.bin"))
                    )
                    
                    if has_weights:
                        models.append({
                            "name": f"🤗 {model_dir.name}",
                            "path": str(model_dir),
                            "type": "hf",  # HuggingFace модель
                            "time": model_dir.stat().st_mtime
                        })
    
    # Сортируем по времени (новые сверху)
    models.sort(key=lambda x: x["time"], reverse=True)
    return models


def render_distributed_config(training_config: dict | None = None):
    """Конфигурация distributed training."""
    st.sidebar.header("🖥️ GPU и параллелизм")
    
    # Информация о GPU
    gpus = get_gpu_info()
    
    if gpus:
        st.sidebar.success(f"✅ Найдено GPU: {len(gpus)}")
        
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
                help="Выберите GPU для обучения"
            )
            num_gpus = len(selected_gpus)
            gpu_ids = [gpu_options.index(g) for g in selected_gpus]
        else:
            num_gpus = 1
            gpu_ids = [0]
            st.sidebar.info("Используется единственная GPU")
    else:
        st.sidebar.warning("⚠️ GPU не найдены, будет использован CPU")
        num_gpus = 0
        gpu_ids = []
    
    st.sidebar.markdown("---")
    
    # Выбор типа параллелизма
    st.sidebar.subheader("⚡ Тип параллелизма")
    
    # Определяем доступные опции и рекомендуемый режим
    if num_gpus == 0:
        available_modes = ["default"]
        recommended_idx = 0
    elif num_gpus == 1:
        available_modes = ["default", "deepspeed_zero3_offload"]
        recommended_idx = 0
    else:
        # При нескольких GPU рекомендуем multi_gpu или fsdp
        available_modes = ["multi_gpu", "fsdp", "deepspeed_zero2", "deepspeed_zero3", "deepspeed_zero3_offload", "default"]
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
    <b>Тип:</b> {mode_info['type']}<br>
    <small>{mode_info['description']}</small>
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
    st.sidebar.subheader("🚀 Конфигурация запуска")
    
    if num_gpus == 0:
        launch_info = "**Устройство:** CPU"
    elif selected_mode == "default":
        launch_info = f"**Устройство:** GPU {gpu_ids[0] if gpu_ids else 0}"
    else:
        launch_info = f"**Устройства:** {num_gpus} × GPU\n**Режим:** {mode_info['type']}"
    
    st.sidebar.info(launch_info)

    # Compute / precision (нужно и для GRPO, потому что training_config для GRPO пустой)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Precision & Memory")

    # Если training_config передан (SFT/Pretrain) — берём дефолт из него, иначе bf16 (GRPO дефолт)
    default_mp = (training_config.get("mixed_precision") if training_config else None) or "bf16"
    mixed_precision = st.sidebar.selectbox(
        "Mixed Precision",
        ["no", "fp16", "bf16"],
        index=["no", "fp16", "bf16"].index(default_mp) if default_mp in ("no", "fp16", "bf16") else 2,
        help="bf16 рекомендуется для Ampere+ GPU. Для FlashAttention нужен fp16/bf16.",
    )

    default_gc = bool(training_config.get("grad_checkpoint", False)) if training_config else False
    grad_checkpoint = st.sidebar.checkbox(
        "Gradient Checkpointing",
        value=default_gc,
        help="Экономит VRAM, но медленнее. Для GRPO (особенно full+длинные ответы) часто must-have.",
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
        "grad_checkpoint": grad_checkpoint,
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
        st.info("Выберите run для просмотра метрик")
        return
    
    run_id = st.session_state.current_run_id
    metrics = load_metrics(run_id)
    process_alive = is_process_running(run_id)
    
    # Статус
    if process_alive:
        st.success(f"🟢 Процесс запущен (Run: {run_id})")
    else:
        if metrics and metrics.get("status") == "completed":
            duration = metrics.get("training_duration", "unknown")
            st.success(f"✅ Тренировка завершена за {duration} (Run: {run_id})")
            clear_active_run()  # Очищаем active_run.json при завершении
            _close_run_log_files(run_id)  # Закрываем файловые дескрипторы
        elif metrics and metrics.get("status") == "error":
            st.error(f"❌ Ошибка (Run: {run_id})")
            clear_active_run()  # Очищаем active_run.json при ошибке
            _close_run_log_files(run_id)  # Закрываем файловые дескрипторы
        elif metrics and metrics.get("status") == "stopped":
            st.warning(f"⏹️ Тренировка остановлена (Run: {run_id})")
            clear_active_run()  # Очищаем active_run.json при остановке
            _close_run_log_files(run_id)  # Закрываем файловые дескрипторы
        else:
            st.info(f"📋 Просмотр метрик (Run: {run_id})")
    
    if metrics:
        render_metrics_dashboard(metrics)
    else:
        st.info("Метрики не найдены")


def render_metrics_dashboard(metrics: dict):
    """Дашборд с метриками обучения."""
    
    status = metrics.get("status", "unknown")
    
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
    
    st.subheader(f"{status_emoji} Статус: {status.upper()}")
    
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
        st.metric("Прогресс", f"{progress:.1f}%", suffix)

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
            st.metric("Параметры", format_params(model_params))
        else:
            st.metric("Параметры", "—")

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
        st.metric("Время", f"{format_time(elapsed)}", delta=f"Ост: {format_time(eta)}", delta_color="normal")
    
    # Progress bar
    st.progress(min(progress / 100, 1.0))
    
    # Charts
    # ВАЖНО: используем стабильный ключ по run_id, чтобы избежать утечки памяти
    rid = st.session_state.get("current_run_id", "active") or "active"
    
    # Специальная секция для GRPO
    if metrics.get("stage") == "grpo":
        st.markdown("---")
        st.subheader("🧠 GRPO Мониторинг")
        
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
                    st.info("Ожидание данных о loss...")
        
        # Окошко с семплами
        st.markdown("---")
        st.subheader("📝 Примеры генераций")
        
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
                            st.info("Ожидание генераций...")
            else:
                st.info("Семплы будут отображаться здесь после начала генераций. Проверьте файл `samples.jsonl` в директории run.")
        
        st.markdown("---")
    
    # Обычные графики для других стадий
    elif metrics.get("loss_history"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss chart
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=metrics["steps_history"],
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
            fig_lr = go.Figure()
            fig_lr.add_trace(go.Scatter(
                x=metrics["steps_history"],
                y=metrics["lr_history"],
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
        st.subheader("🖥️ Нагрузка GPU")
        
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
        st.error("❌ Произошла ошибка во время тренировки")
        with st.expander("Подробности ошибки (Traceback)", expanded=True):
            st.code(metrics['error'], language="python")
    
    # Логи процесса
    if st.session_state.current_run_id:
        run_dir = RUNS_DIR / st.session_state.current_run_id
        stderr_path = run_dir / "stderr.log"
        stdout_path = run_dir / "stdout.log"
        
        with st.expander("📋 Логи процесса"):
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
                
                # Проверка лимитов
                if limit_type == "Строки (Количество)" and count >= limit_val:
                    break
                if limit_type == "ГБ (Размер)" and (current_bytes + line_bytes) > limit_bytes:
                    break
                    
                f.write(line)
                count += 1
                current_bytes += line_bytes
                
                if count % 1000 == 0:
                    print(f"Downloaded {count} lines, {current_bytes / 1024**2:.2f} MB")

        st.success(f"Готово! Сохранено {count} строк ({current_bytes / 1024**2:.2f} MB) в {save_path}")
        return True

    except Exception as e:
        st.error(f"Ошибка: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def render_data_manager():
    """Вкладка управления данными."""
    st.header("💾 Управление данными")
    
    col_upload, col_list = st.columns([1, 2])
    
    with col_upload:
        # Секция 1: Загрузка локальных файлов
        with st.expander("📤 Загрузка локальных файлов", expanded=False):
            uploaded_files = st.file_uploader(
                "Перетащите файлы сюда", 
                type=["jsonl", "txt"],  # ВАЖНО: не включаем .json, т.к. это обычно массив, а не JSONL 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("📥 Сохранить файлы"):
                    for uploaded_file in uploaded_files:
                        save_path = DATASET_DIR / uploaded_file.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.toast(f"Файл {uploaded_file.name} сохранён!", icon="✅")
                    time.sleep(1)
                    st.rerun()

        # Секция 2: Загрузка с HuggingFace
        st.subheader("🤗 Скачать с HuggingFace")
        # Словарь пресетов: {название: (repo_id, subset, split)}
        presets = {
            # Pretrain датасеты
            "🟢 Pretrain: FineWeb-2 (Russian)": ("HuggingFaceFW/fineweb-2", "rus_Cyrl", "train"),
            "🟢 Pretrain: FineWeb-Edu (Educational)": ("HuggingFaceFW/fineweb-edu", "default", "train"),
            "🟢 Pretrain: Wikitext-103": ("wikitext", "wikitext-103-v1", "train"),
            # SFT датасеты
            "🔵 SFT: OpenOrca-ru": ("d0rj/OpenOrca-ru", "default", "train"),
            "🔵 SFT: ru-instruct": ("d0rj/ru-instruct", "default", "train"),
            "🔵 SFT: GrandMaster-PRO-MAX": ("Vikhrmodels/GrandMaster-PRO-MAX", "default", "train"),
            # Reasoning датасеты (для GRPO)
            "🧠 Reasoning: GSM8K (English)": ("gsm8k", "main", "train"),
            "🧠 Reasoning: GSM8K-RU (Русский)": ("d0rj/gsm8k-ru", "default", "train"),
            "🧠 Reasoning: MATH-RU (олимпиады)": ("d0rj/competition_math_ru", "default", "train"),
            "🧠 Reasoning: MGSM (многоязычный)": ("juletxara/mgsm", "ru", "train"),
            "🧠 Reasoning: ARC-Challenge": ("allenai/ai2_arc", "ARC-Challenge", "train"),
            # Ручной ввод
            "📝 Ввести вручную...": (None, None, None),
        }
        
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
            "📚 Популярные датасеты",
            options=list(presets.keys()),
            index=0,  # FineWeb-2 Russian первый
            key="dataset_preset_selector",
            on_change=on_preset_change,
            help="Выберите готовый датасет — все поля заполнятся автоматически"
        )

        repo_id = st.text_input("Репозиторий (ID)", key="hf_repo_id_input")
        
        # Кнопка проверки репозитория
        if st.button("🔍 Проверить репозиторий"):
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
                    
                    st.success(f"✅ Найдено {len(configs)} конфигураций, splits: {splits}")
            except Exception as e:
                st.error(f"Не удалось получить информацию: {e}")

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
        with st.expander("🛠️ Фильтры и Лимиты", expanded=True):
            # Лимиты (всегда доступны)
            col_lim1, col_lim2 = st.columns(2)
            with col_lim1:
                limit_type = st.radio("Тип лимита", ["ГБ (Размер)", "Строки (Количество)"], key="limit_type")
            
            with col_lim2:
                limit_val = 0
                limit_bytes = 0
                
                if limit_type == "Строки (Количество)":
                    limit_val = st.number_input("Кол-во строк", value=100000, step=10000, key="limit_val")
                else:
                    limit_gb = st.number_input("Размер (ГБ)", value=2.0, step=0.5, min_value=0.1, key="limit_gb")
                    limit_bytes = int(limit_gb * 1024**3)
            
            st.divider()
            
            # Фильтры на основе структуры (features)
            active_filters = {}
            
            if features:
                st.caption("🔍 Настройка фильтров по колонкам:")
                
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
                        selected_float_col = st.selectbox("Выберите колонку", ["(нет)"] + float_cols, key="sel_float_col")
                        if selected_float_col != "(нет)":
                            min_val = st.slider(f"Мин. значение {selected_float_col}", 0.0, 1.0, 0.0, key="val_float_col")
                            active_filters["score_col"] = selected_float_col
                            active_filters["min_score"] = min_val
                    else:
                        st.caption("Числовые колонки не найдены")

                with col_f2:
                    if string_cols:
                        st.markdown("**Фильтр по тексту (contains)**")
                        selected_str_col = st.selectbox("Выберите колонку", ["(нет)"] + string_cols, key="sel_str_col")
                        if selected_str_col != "(нет)":
                            target_str = st.text_input(f"Текст должен содержать:", key="val_str_col")
                            if target_str:
                                active_filters["lang_col"] = selected_str_col
                                active_filters["target_lang"] = target_str
                    else:
                        st.caption("Текстовые колонки не найдены")

            else:
                st.info("⚠️ Структура датасета не загружена. Доступны только лимиты по объему.")


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
            "Имя файла для сохранения", 
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

        st.button("Скачать и обработать", on_click=on_download_click, args=(active_filters,))
    
    with col_list:
        st.subheader("Доступные датасеты")
        
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
            st.info("Нет загруженных датасетов. Загрузите файлы слева.")
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
                            st.info("Файл пуст")
                        
                        col_del, col_info = st.columns([1, 4])
                        with col_del:
                            if st.button("🗑️ Удалить", key=f"del_{ds['name']}"):
                                ds['path'].unlink()
                                st.toast(f"Файл {ds['name']} удалён", icon="🗑️")
                                time.sleep(1)
                                st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка чтения файла: {e}")


def download_hf_model(repo_id: str, save_name: str, revision: str = "main"):
    """
    Скачивает модель с HuggingFace и сохраняет локально в MODELS_DIR.
    """
    from huggingface_hub import snapshot_download
    
    save_path = MODELS_DIR / save_name
    
    # Проверяем, не существует ли уже
    if save_path.exists():
        st.warning(f"⚠️ Модель `{save_name}` уже существует!")
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
                st.error("❌ Не найден config.json в скачанной модели")
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
        st.error(f"❌ Ошибка скачивания: {e}")
        import traceback
        print(traceback.format_exc())
        # Удаляем частично скачанное
        if save_path.exists():
            import shutil
            shutil.rmtree(save_path, ignore_errors=True)
        return False


def render_model_manager():
    """Вкладка управления моделями (скачивание с HuggingFace)."""
    st.header("🤖 Управление моделями")
    
    col_download, col_list = st.columns([1, 2])
    
    with col_download:
        st.subheader("🤗 Скачать с HuggingFace")
        
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
            "📚 Популярные модели",
            options=list(model_presets.keys()),
            index=0,
            key="model_preset_selector",
            on_change=on_model_preset_change,
            help="Выберите модель — repo_id заполнится автоматически"
        )
        
        repo_id = st.text_input("Репозиторий (ID)", key="model_repo_id")
        save_name = st.text_input("Название для сохранения", key="model_save_name", 
                                   help="Папка в models/, куда будет сохранена модель")
        
        # Информация о модели
        st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 12px; border-radius: 8px; margin: 10px 0;
            border: 1px solid #0f3460; color: #e8e8e8;">
<b style="color: #4fc3f7;">💡 Рекомендации:</b><br>
• <b style="color: #81d4fa;">SmolLM2</b> — современные компактные модели от HuggingFace<br>
• <b style="color: #81d4fa;">Pythia</b> — отличные для экспериментов, разные размеры<br>
• <b style="color: #81d4fa;">TinyLlama</b> — популярная, хорошо обучена на 3T токенов<br>
• <b style="color: #81d4fa;">Qwen2.5</b> — сильные модели от Alibaba
</div>
""", unsafe_allow_html=True)
        
        # Оценка размера
        size_estimates = {
            "70m": "~150 MB", "135m": "~300 MB", "160m": "~350 MB",
            "360m": "~800 MB", "410m": "~900 MB", "0.5b": "~1 GB",
            "1b": "~2 GB", "1.1b": "~2.5 GB", "1.5b": "~3 GB", "1.7b": "~3.5 GB",
            "124m": "~500 MB", "355m": "~1.5 GB", "125m": "~500 MB",
        }
        
        estimated_size = "Неизвестно"
        repo_lower = repo_id.lower()
        for size_key, size_val in size_estimates.items():
            if size_key in repo_lower:
                estimated_size = size_val
                break
        
        st.caption(f"📦 Примерный размер: **{estimated_size}**")
        
        if st.button("⬇️ Скачать модель", type="primary"):
            if not repo_id or not save_name:
                st.error("Укажите repo_id и название!")
            else:
                success = download_hf_model(repo_id, save_name)
                if success:
                    time.sleep(1)
                    st.rerun()
    
    with col_list:
        st.subheader("📁 Скачанные модели")
        
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
            st.info("Нет скачанных моделей. Скачайте модель слева для Continual Pretraining или SFT.")
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
                        if st.button("🚀 Использовать", key=f"use_{m['name']}", 
                                     help="Выбрать эту модель для Continual Pretrain / SFT"):
                            st.session_state.selected_base_model = str(m['path'])
                            st.toast(f"Модель {m['name']} выбрана! Перейдите в Запуск и выберите Continual Pretrain или SFT.", icon="✅")
                        
                        # Кнопка удаления
                        if st.button("🗑️ Удалить", key=f"del_model_{m['name']}"):
                            import shutil
                            shutil.rmtree(m['path'])
                            st.toast(f"Модель {m['name']} удалена", icon="🗑️")
                            time.sleep(1)
                            st.rerun()
        
        # Подсказка
        st.markdown("---")
        st.info("""
💡 **Как использовать скачанную модель:**
1. Нажмите **🚀 Использовать** на нужной модели
2. Перейдите на вкладку **🚀 Запуск**
3. В сайдбаре выберите режим **Continual Pretrain** или **SFT**
4. Модель автоматически подставится как базовая
""")


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


def render_model_preview(config: dict, distributed_config: dict = None):
    """Превью архитектуры модели и настроек параллелизма."""
    st.subheader("📐 Архитектура модели")
    
    stage = config.get("stage", "pretrain")
    if stage == "sft":
        st.info(f"🔄 **Режим SFT** (Fine-Tuning)\nБазовая модель: `{Path(config.get('base_model_path') or 'Unknown').name}`")
    elif stage == "continual_pretrain":
        st.info(f"🔄 **Режим Continual Pretraining** (Продолжение)\nБазовая модель: `{Path(config.get('base_model_path') or 'Unknown').name}`")
    elif stage == "grpo":
        st.info(f"🧠 **Режим GRPO** (RL для Reasoning)\nБазовая модель: `{Path(config.get('base_model_path') or 'Unknown').name}`")
    else:
        st.success("🏗️ **Режим Pretraining** (С нуля)")

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
        st.metric("Параметры", format_params(mem_info["params"]))
        
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
            st.warning("⚠️ Активации занимают много памяти! Включите Gradient Checkpointing или уменьшите Batch Size.")

    
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
        st.subheader("⚡ Параллелизм")
        
        mode = distributed_config.get("distributed_mode", "default")
        mode_info = PARALLEL_TYPES.get(mode, PARALLEL_TYPES["default"])
        num_gpus = distributed_config.get("num_gpus", 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Режим", mode_info["name"])
        
        with col2:
            st.metric("Тип", mode_info["type"])
        
        with col3:
            if num_gpus > 0:
                st.metric("GPU", f"{num_gpus} шт.")
            else:
                st.metric("Устройство", "CPU")
        
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
┌────────── FSDP (Fully Sharded) ─────────────┐
│                                              │
│  ┌─────────┐  ┌─────────┐       ┌─────────┐ │
│  │  GPU 0  │  │  GPU 1  │  ...  │  GPU N  │ │
│  │ Shard 0 │  │ Shard 1 │       │ Shard N │ │
│  │ Params  │  │ Params  │       │ Params  │ │
│  └────┬────┘  └────┬────┘       └────┬────┘ │
│       │            │                 │      │
│       └─── All-Gather for Forward ──┘       │
│       └─── Reduce-Scatter Backward ─┘       │
│                                              │
└──────────────────────────────────────────────┘
Модель распределена между GPU (экономия VRAM)
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
        st.error(f"Ошибка экспорта: {e}")
        return None


# ============================================================================
# Main App
# ============================================================================

def main():
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
        distributed_config = render_distributed_config(training_config=dummy_training_config)
    else:
        grpo_sidebar_config = {}
        training_config = render_training_config()
        distributed_config = render_distributed_config(training_config=training_config)
    
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
            # Копируем LoRA параметры из model_config в full_config для передачи в train_gsm8k.py
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
    # ВАЖНО: для GRPO training_config пустой, поэтому mixed_precision/grad_checkpoint берём из distributed_config.
    # Для остальных стадий эти параметры уже приходят из training_config — не перетираем их.
    if "mixed_precision" not in full_config:
        full_config["mixed_precision"] = distributed_config.get("mixed_precision", "bf16")
    if "grad_checkpoint" not in full_config:
        full_config["grad_checkpoint"] = distributed_config.get("grad_checkpoint", False)
    
    # Для SFT, Continual Pretrain и GRPO используем токенизатор базовой модели
    if model_config.get("stage") in ("sft", "continual_pretrain", "grpo") and model_config.get("base_model_path"):
        full_config["tokenizer_path"] = model_config["base_model_path"]
    
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🚀 Запуск", "📊 Мониторинг", "💬 Чат", "📜 История", "💾 Данные", "🤖 Модели", "📚 Учебник"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Передаем full_config, чтобы калькулятор памяти видел batch_size и grad_checkpoint
            render_model_preview(full_config, distributed_config)
            
            # SFT Config (Main Area)
            if model_config.get("stage") == "sft" and dataset_config.get("data_path"):
                st.markdown("---")
                # Вызываем функцию (даже если она дублирована, вызовется последняя определенная)
                sft_cfg = render_sft_main_config(dataset_config["data_path"])
                full_config.update(sft_cfg)
            
            # GRPO Config (Main Area) - настройка reward функций и датасета
            if model_config.get("stage") == "grpo":
                st.markdown("---")
                grpo_main_cfg = render_grpo_main_config(dataset_config.get("data_path"))
                full_config.update(grpo_main_cfg)
            
            st.subheader("📋 Конфигурация")
            st.json(full_config)
        
        with col2:
            st.subheader("🎮 Управление")
            
            if st.session_state.training_active:
                if st.button("⏹️ Остановить", type="primary"):
                    with st.spinner("Останавливаем тренировку..."):
                        stopped = stop_training()
                    if stopped:
                        st.success("✅ Тренировка остановлена")
                    else:
                        st.warning("⚠️ Не удалось остановить (возможно уже завершена)")
                    time.sleep(1)
                    st.rerun()
            else:
                # Для GRPO другая кнопка и логика
                if model_config.get("stage") == "grpo":
                    if st.button("🧠 Начать GRPO обучение", type="primary"):
                        with st.spinner("Запуск GRPO..."):
                            run_id, process = start_grpo_training(full_config)
                            st.session_state.current_run_id = run_id
                            st.session_state.training_process = process
                            st.session_state.training_active = True
                            
                            save_active_run(run_id, full_config)
                            
                            st.success(f"GRPO обучение запущено! Run ID: {run_id}")
                            time.sleep(1)
                            st.rerun()
                else:
                    if st.button("▶️ Начать тренировку", type="primary"):
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
    
    with tab2:
        # Используем fragment для автоматического обновления без перезагрузки страницы
        live_metrics_fragment()
    
    with tab4:
        st.header("📜 История запусков")
        st.markdown("---")
        
        # Фильтруем только директории (игнорируем файлы типа active_run.json)
        runs = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)
        
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
                                st.metric("Параметры", format_params(model_params))
                            else:
                                st.metric("Параметры", "—")
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
                            
                            for ckpt in checkpoints:
                                ckpt_loss = ckpt.get("loss")
                                # Если loss нет в чекпоинте, пытаемся найти в loss_history
                                if ckpt_loss is None and steps_history and loss_history:
                                    ckpt_step = ckpt.get("step")
                                    if ckpt_step in steps_history:
                                        idx = steps_history.index(ckpt_step)
                                        if idx < len(loss_history):
                                            ckpt_loss = loss_history[idx]
                                
                                if ckpt_loss is not None:
                                    st.caption(f"Step {ckpt['step']}: Loss {ckpt_loss:.4f} | {ckpt['path']}")
                                else:
                                    st.caption(f"Step {ckpt['step']}: {ckpt['path']}")
                        
                        # Кнопки
                        btn_col1, btn_col2, btn_col3 = st.columns(3)
                        with btn_col1:
                            if st.button(f"📊 Метрики", key=f"metrics_{run_id}"):
                                st.session_state.current_run_id = run_id
                                st.toast(f"✅ Выбран run: {run_id}. Перейдите на вкладку 📊 Мониторинг", icon="📊")
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
                                        if st.button("💬 Чат", key=f"chat_run_{run_id}"):
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
                                if st.button("▶️ Продолжить", key=f"continue_{run_id}", help="Продолжить обучение с последнего чекпоинта"):
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
                                st.button("⚠️ Файлы удалены", key=f"gone_{run_id}", disabled=True, help=f"Чекпоинт {checkpoints[-1]['path']} не найден на диске")
                        
                        # Показываем что выбрано
                        if st.session_state.current_run_id == run_id:
                            st.info("👆 Перейдите на вкладку **📊 Мониторинг**")
        else:
            st.info("Нет предыдущих запусков")
            
    with tab5:
        render_data_manager()
        
        # Подсказка про чат
        st.markdown("---")
        st.info("💡 Чтобы пообщаться с моделью, перейдите на вкладку **💬 Чат** (в верхней части страницы)")

    with tab6:
        render_model_manager()
    
    with tab7:
        render_docs()
    
    with tab3:
        st.header("💬 Чат с моделью")
        st.markdown("---")
        
        # Получаем список доступных моделей
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
                    "Выберите модель или чекпоинт",
                    options=model_options,
                    index=default_idx,
                    help="Выберите обученную модель для чата"
                )
                selected_model = next(m for m in available_models if m["name"] == selected_model_name)
                
                # Сбрасываем selected_chat_model после использования
                if st.session_state.selected_chat_model:
                    st.session_state.selected_chat_model = None
            
            with col2:
                # Для чата используем только final_model/export (HF формат)
                # Все модели сохраняются в HF формате, поэтому используем AutoModelForCausalLM
                model_type = selected_model["type"]
                if model_type == "final":
                    st.success("✅ Финальная модель")
                else:
                    st.info("📦 Чекпоинт")
            
            st.caption(f"Путь: `{selected_model['path']}`")
            
            # Параметры генерации
            with st.expander("⚙️ Параметры генерации"):
                gen_col1, gen_col2, gen_col3 = st.columns(3)
                with gen_col1:
                    max_tokens = st.slider("Max Tokens", 10, 500, 128)
                with gen_col2:
                    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
                with gen_col3:
                    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

                # Режим промпта (2 режима, но дефолт выбираем автоматически):
                # - если у модели есть chat_template -> Диалог
                # - если нет -> Completion
                if "chat_prompt_mode" not in st.session_state:
                    st.session_state.chat_prompt_mode = "completion"  # до загрузки модели
                prompt_mode_label = st.selectbox(
                    "Режим промпта",
                    options=["Диалог (chat_template)", "Completion (plain text)"],
                    index=0 if st.session_state.chat_prompt_mode == "chat" else 1,
                    help="По умолчанию: если у модели есть chat_template — включаем Диалог, иначе Completion. Можно переключить вручную.",
                    key="chat_prompt_mode_select",
                )
                prompt_mode = "chat" if prompt_mode_label.startswith("Диалог") else "completion"
                st.session_state.chat_prompt_mode = prompt_mode
            
            # Инициализация чата
            if "chat_model" not in st.session_state:
                st.session_state.chat_model = None
                st.session_state.chat_tokenizer = None
                st.session_state.chat_model_path = None
                st.session_state.chat_has_template = False
                st.session_state.chat_prompt_mode = "completion"
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Кнопка загрузки модели
            if st.session_state.chat_model_path != selected_model["path"]:
                if st.button("🔄 Загрузить модель", type="primary"):
                    with st.spinner("Загружаем модель..."):
                        try:
                            from transformers import AutoTokenizer, AutoModelForCausalLM
                            from homellm.models.adapters import detect_model_type
                            from homellm.models.home_model import HomeForCausalLM
                            
                            model_path = Path(selected_model["path"])
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            dtype = torch.float16 if device == "cuda" else torch.float32
                            
                            # Проверяем наличие config.json
                            config_json = model_path / "config.json"
                            if not config_json.exists():
                                raise ValueError(f"config.json не найден в {model_path}")
                            
                            # Определяем тип модели
                            model_type = detect_model_type(model_path)
                            st.info(f"Загружаем {model_type.upper()} модель...")
                            
                            # Загружаем токенизатор
                            try:
                                tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
                            except Exception:
                                # Для accelerate checkpoint'ов токенизатора обычно нет внутри checkpoint_stepXXXX.
                                # Пробуем взять tokenizer_path/base_model_path из run config.
                                tokenizer = None
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
                                except Exception:
                                    tokenizer = None

                                if tokenizer is None:
                                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                            
                            # Загружаем модель в зависимости от типа
                            if model_type == "home":
                                model = HomeForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype)
                            else:
                                model = AutoModelForCausalLM.from_pretrained(
                                    str(model_path), trust_remote_code=True, torch_dtype=dtype
                                )
                            
                            # Переносим на device
                            model = model.to(device)
                            model.eval()
                            
                            # Подготавливаем токенизатор (pad_token = eos_token)
                            if tokenizer.pad_token is None:
                                if tokenizer.eos_token:
                                    tokenizer.pad_token = tokenizer.eos_token
                            
                            st.session_state.chat_model = model
                            st.session_state.chat_tokenizer = tokenizer
                            st.session_state.chat_model_path = str(model_path)
                            st.session_state.messages = []
                            # Автоподтягивание режима: если у токенизатора есть chat_template, ставим "Диалог",
                            # иначе "Completion". Пользователь может переключить вручную после загрузки.
                            st.session_state.chat_has_template = bool(getattr(tokenizer, "chat_template", None))
                            st.session_state.chat_prompt_mode = "chat" if st.session_state.chat_has_template else "completion"
                            st.success("✅ Модель загружена!")
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
                st.success(f"✅ Модель загружена: {selected_model_name}")
                
                # Кнопка экспорта (для чекпоинтов особенно полезна)
                if st.button("💾 Экспортировать в HF формат", help="Сохранить как полноценную модель (с конфигом и токенизатором)"):
                    with st.spinner("Экспорт модели..."):
                        export_path = export_model_to_hf(
                            st.session_state.chat_model, 
                            st.session_state.chat_tokenizer, 
                            st.session_state.chat_model_path
                        )
                        if export_path:
                            st.success(f"Модель успешно экспортирована в:\n`{export_path}`")
                            time.sleep(2)
                            st.rerun() # Обновить список моделей чтобы увидеть экспорт

                # --- НАСТРОЙКИ СИСТЕМНОГО ПРОМПТА ---
                with st.expander("⚙️ Системный промпт", expanded=False):
                    system_prompt_input = st.text_area(
                        "Системный промпт (опционально):",
                        value=st.session_state.get("system_prompt", ""),
                        help="Если заполнено, будет использоваться вместо дефолтного системного промпта модели. Оставьте пустым для использования дефолтного.",
                        key="system_prompt_input"
                    )
                    st.session_state.system_prompt = system_prompt_input.strip()
                    if system_prompt_input.strip():
                        st.info("✅ Будет использован введенный системный промпт")
                    else:
                        st.caption("Используется дефолтный системный промпт из модели")
                
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
                                    tokenizer = st.session_state.chat_tokenizer
                                    model = st.session_state.chat_model
                                    device = next(model.parameters()).device
                                    
                                    # Формируем полный контекст
                                    # Пользователь контролирует режим: auto/chat_template/plain
                                    
                                    # Берем историю + новое сообщение
                                    conversation = st.session_state.messages.copy() # [{"role": "user", ...}, ...]
                                    
                                    has_template = bool(getattr(tokenizer, "chat_template", None))
                                    use_chat_template = (prompt_mode == "chat") and has_template
                                    if prompt_mode == "chat" and not has_template:
                                        st.warning("У выбранной модели нет chat_template — использую Completion.")
                                        use_chat_template = False

                                    # Обработка системного промпта (только для режима chat_template)
                                    if use_chat_template:
                                        system_prompt = st.session_state.get("system_prompt", "").strip()
                                        
                                        # Удаляем существующее системное сообщение из conversation (если есть)
                                        # чтобы не было конфликта с введенным системным промптом
                                        if conversation and conversation[0].get("role") == "system":
                                            conversation.pop(0)
                                        
                                        # Если указан системный промпт, добавляем его в начало
                                        if system_prompt:
                                            conversation.insert(0, {"role": "system", "content": system_prompt})
                                        # Если системный промпт пустой, шаблон использует дефолтный из модели
                                    
                                    if use_chat_template:
                                        # Для SFT модели: применяем шаблон с тегами
                                        prompt_text = tokenizer.apply_chat_template(
                                            conversation, 
                                            tokenize=False, 
                                            add_generation_prompt=True
                                        )
                                    else:
                                        # Для Base/Pretrain модели: просто текст
                                        # Обычно Base модели не понимают диалог, но попробуем просто слать последний промпт
                                        # или весь диалог текстом
                                        prompt_text = ""
                                        for m in conversation:
                                            prompt_text += f"{m['role']}: {m['content']}\n"
                                        prompt_text += "assistant: "
                                    
                                    # ВАЖНО:
                                    # - inputs должны быть на ТОМ ЖЕ устройстве, что и модель (модель может быть на cuda:1)
                                    # - заранее проверяем, что token ids не выходят за vocab модели (иначе будет CUDA assert)
                                    device = next(model.parameters()).device
                                    device_type = str(device.type)

                                    enc = tokenizer(prompt_text, return_tensors="pt")
                                    try:
                                        if hasattr(model, "get_input_embeddings") and model.get_input_embeddings() is not None:
                                            vocab_size = int(model.get_input_embeddings().weight.shape[0])
                                            max_id = int(enc["input_ids"].max().item())
                                            min_id = int(enc["input_ids"].min().item())
                                            if min_id < 0 or max_id >= vocab_size:
                                                raise ValueError(
                                                    f"Tokenizer выдаёт token_id вне vocab модели: min_id={min_id}, max_id={max_id}, "
                                                    f"vocab_size(model)={vocab_size}. "
                                                    f"Проверьте, что с моделью загружен правильный tokenizer (тот же, что был при обучении)."
                                                )
                                    except Exception as e:
                                        raise RuntimeError(f"Проблема токенизации/вокаба перед генерацией: {e}")

                                    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
                                    
                                    # Приводим dtype для стабильной генерации (особенно после SFT checkpoints bf16)
                                    model_dtype = next(model.parameters()).dtype
                                    autocast_enabled = (device_type == "cuda")
                                    autocast_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16

                                    # attention_mask должен оставаться int/bool (не bf16/fp16).

                                    # Ограничиваем контекстное окно (иначе возможен CUDA index out of bounds при позиционных индексах).
                                    max_ctx = None
                                    try:
                                        cfg = getattr(model, "config", None)
                                        for key in ("max_position_embeddings", "n_positions", "seq_len", "max_seq_len"):
                                            if cfg is not None and hasattr(cfg, key):
                                                v = getattr(cfg, key)
                                                if v is not None:
                                                    max_ctx = int(v)
                                                    break
                                    except Exception:
                                        max_ctx = None

                                    if max_ctx is not None and max_ctx > 0 and "input_ids" in inputs:
                                        in_len = int(inputs["input_ids"].shape[1])
                                        if in_len > max_ctx:
                                            cut = in_len - max_ctx
                                            inputs["input_ids"] = inputs["input_ids"][:, cut:]
                                            if "attention_mask" in inputs and hasattr(inputs["attention_mask"], "shape"):
                                                inputs["attention_mask"] = inputs["attention_mask"][:, cut:]
                                            in_len = int(inputs["input_ids"].shape[1])

                                        allowed_new = int(max_ctx - in_len)
                                        if allowed_new <= 0:
                                            raise RuntimeError(
                                                f"Контекст уже достиг максимума модели (max_ctx={max_ctx}). "
                                                f"Уменьшите историю/системный промпт или используйте модель с большим контекстом."
                                            )
                                        if int(max_tokens) > allowed_new:
                                            max_tokens = allowed_new

                                    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_enabled):
                                        outputs = model.generate(
                                            **inputs,
                                            max_new_tokens=max_tokens,
                                            temperature=temperature,
                                            top_p=top_p,
                                            do_sample=True,
                                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                            use_cache=False,  # Отключаем KV-cache для совместимости
                                        )
                                    
                                    response = tokenizer.decode(
                                        outputs[0][inputs["input_ids"].shape[1]:], 
                                        skip_special_tokens=True
                                    )
                                    
                                    st.write(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                except Exception as e:
                                    import traceback
                                    st.session_state.last_chat_error = traceback.format_exc()
                                    st.error(f"Ошибка генерации: {e}")
                                    st.code(st.session_state.last_chat_error)
                
                # Кнопка очистки чата
                if st.session_state.messages:
                    if st.button("🗑️ Очистить чат"):
                        st.session_state.messages = []
                        st.rerun()
        else:
            st.info("Нет обученных моделей. Запустите тренировку во вкладке 'Запуск'!")
            
            # Показываем где искать модели
            st.markdown("""
            **Модели будут доступны после тренировки:**
            - `out/*/final_model/` — финальные модели
            - `out/*/checkpoint_*/` — чекпоинты
            """)


if __name__ == "__main__":
    main()


