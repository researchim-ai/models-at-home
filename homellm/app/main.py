"""
Motels at Home Training Studio — Визуальное приложение для тренировки моделей
======================================================================

Запуск:
    streamlit run homellm/app/main.py
    
или:
    ./scripts/run_studio.sh
"""

import streamlit as st
import subprocess
import json
import time
import os
import signal
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import torch
from datasets import load_dataset  # Добавляем импорт

# Пути
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATASET_DIR = PROJECT_ROOT / "datasets"  # datasets с "s"!
OUTPUT_DIR = PROJECT_ROOT / "out"
RUNS_DIR = PROJECT_ROOT / ".runs"
RUNS_DIR.mkdir(exist_ok=True)

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
            # Проверяем жив ли процесс
            pid_path = run_dir / "pid"
            process_alive = False
            if pid_path.exists():
                try:
                    with open(pid_path) as f:
                        pid = int(f.read().strip())
                    os.kill(pid, 0)  # Проверка существования процесса
                    process_alive = True
                except PermissionError:
                    process_alive = True
                except (ProcessLookupError, ValueError, PermissionError):
                    pass
            
            # Восстанавливаем состояние
            st.session_state.current_run_id = run_id
            st.session_state.training_active = process_alive
            
            # Если процесс завершён, очищаем active_run
            if not process_alive:
                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                        if metrics.get("status") in ["completed", "error", "stopped"]:
                            clear_active_run()
                    except:
                        pass


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
        for f in DATASET_DIR.glob("*.txt"):
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


def estimate_parameters(hidden_size: int, num_layers: int, vocab_size: int = 50257) -> int:
    """Примерная оценка количества параметров."""
    # Embedding: vocab_size * hidden_size
    embed = vocab_size * hidden_size
    # Each layer: attention (4 * hidden^2) + FFN (8 * hidden^2) + norms
    per_layer = 4 * hidden_size ** 2 + 8 * hidden_size ** 2 + 2 * hidden_size
    # LM head is tied, so not counted
    total = embed + num_layers * per_layer
    return total


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
    metrics_path = RUNS_DIR / run_id / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                return json.load(f)
        except:
            pass
    return None


def start_training(config: dict) -> str:
    """Запустить тренировку в фоне."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
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
        cmd = [
            "accelerate", "launch",
            "--config_file", config_file,
            "--num_processes", str(num_gpus),
            "--gradient_accumulation_steps", str(config.get("gradient_accumulation", 1)),
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
    
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,  # Отделяем от родительского процесса
    )
    
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
                # Сначала SIGTERM, потом SIGKILL если не помогло
                os.kill(pid, signal.SIGTERM)
                stopped = True
                
                # Ждём немного и проверяем
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)  # Проверяем жив ли процесс
                    os.kill(pid, signal.SIGKILL)  # Принудительно убиваем
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
    
    return stopped


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


def render_model_config():
    """Конфигуратор модели в сайдбаре."""
    st.sidebar.header("🧠 Архитектура модели")
    
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
        default_h, default_l, default_n = 512, 8, 8
    
    hidden_size = st.sidebar.slider(
        "Hidden Size", 
        min_value=128, 
        max_value=2048, 
        value=default_h, 
        step=64,
        help="Размерность скрытого слоя"
    )
    
    num_layers = st.sidebar.slider(
        "Num Layers", 
        min_value=2, 
        max_value=32, 
        value=default_l,
        help="Количество слоёв трансформера"
    )
    
    n_heads = st.sidebar.slider(
        "Attention Heads", 
        min_value=2, 
        max_value=32, 
        value=default_n,
        help="Количество голов внимания"
    )
    
    seq_len = st.sidebar.selectbox(
        "Seq Length",
        [256, 512, 1024, 2048],
        index=1,
        help="Максимальная длина последовательности"
    )
    
    # Оценка параметров
    est_params = estimate_parameters(hidden_size, num_layers)
    st.sidebar.metric("Параметры (≈)", format_params(est_params))
    
    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "n_heads": n_heads,
        "seq_len": seq_len,
    }


def render_training_config():
    """Конфигуратор обучения в сайдбаре."""
    st.sidebar.header("⚙️ Параметры обучения")
    
    batch_size = st.sidebar.slider(
        "Batch Size",
        min_value=1,
        max_value=64,
        value=16,
        help="Размер батча"
    )
    
    grad_accum = st.sidebar.slider(
        "Gradient Accumulation",
        min_value=1,
        max_value=32,
        value=4,
        help="Шаги накопления градиента"
    )
    
    st.sidebar.caption(f"Effective batch: {batch_size * grad_accum}")
    
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
        value=5e-4,
        format_func=lambda x: f"{x:.0e}"
    )
    
    warmup_steps = st.sidebar.number_input(
        "Warmup Steps",
        min_value=0,
        max_value=10000,
        value=1000
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
            min_value=100,
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
    
    grad_checkpoint = st.sidebar.checkbox(
        "Gradient Checkpointing",
        value=False,
        help="Экономит VRAM, но медленнее"
    )
    
    return {
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "epochs": epochs,
        "max_steps": max_steps,
        "mixed_precision": mixed_precision,
        "grad_checkpoint": grad_checkpoint,
    }


def render_dataset_config():
    """Выбор датасета."""
    st.sidebar.header("📁 Датасет")
    
    datasets = get_available_datasets()
    
    if datasets:
        dataset_options = [f"{name} ({size})" for name, size in datasets]
        selected = st.sidebar.selectbox("Выберите датасет", dataset_options)
        selected_name = selected.split(" (")[0]
        data_path = str(DATASET_DIR / selected_name)
    else:
        st.sidebar.warning("Датасеты не найдены в dataset/")
        data_path = st.sidebar.text_input("Путь к датасету", "dataset/data.jsonl")
    
    return {"data_path": data_path}


def render_output_config():
    """Конфигурация вывода."""
    st.sidebar.header("💾 Сохранение")
    
    output_dir = st.sidebar.text_input(
        "Output Directory",
        value="out/training_run"
    )
    
    save_every = st.sidebar.number_input(
        "Save Checkpoint Every N Steps",
        min_value=100,
        max_value=50000,
        value=2000,
        step=500,
        help="Как часто сохранять чекпоинты"
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
        checkpoints = list(output_path.glob("checkpoint_*"))
        final_model = output_path / "final_model"
        
        if checkpoints or final_model.exists():
            st.sidebar.caption(f"📦 Найдено чекпоинтов: {len(checkpoints)}")
            if final_model.exists():
                st.sidebar.caption("✅ Финальная модель сохранена")
    
    return {
        "output_dir": output_dir,
        "save_every": save_every,
        "log_every": log_every,
        "tokenizer_path": "gpt2"
    }


def get_available_models():
    """Получить список доступных обученных моделей."""
    models = []
    
    # Ищем в out/
    if OUTPUT_DIR.exists():
        for model_dir in OUTPUT_DIR.iterdir():
            if model_dir.is_dir():
                # Проверяем есть ли final_model
                final_model = model_dir / "final_model"
                if final_model.exists() and (final_model / "config.json").exists():
                    models.append({
                        "name": f"{model_dir.name}/final_model",
                        "path": str(final_model),
                        "type": "final",
                    })
                
                # Ищем чекпоинты
                for ckpt in sorted(model_dir.glob("checkpoint_*"), reverse=True):
                    if ckpt.is_dir():
                        models.append({
                            "name": f"{model_dir.name}/{ckpt.name}",
                            "path": str(ckpt),
                            "type": "checkpoint",
                        })
    
    return models


def render_distributed_config():
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
    
    return {
        "distributed_mode": selected_mode,
        "num_gpus": num_gpus,
        "gpu_ids": gpu_ids,
        "config_file": config_file,
        "parallel_type": mode_info['type'],
    }


@st.fragment(run_every=3)  # Автообновление каждые 3 секунды
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
            st.success(f"✅ Тренировка завершена (Run: {run_id})")
        elif metrics and metrics.get("status") == "error":
            st.error(f"❌ Ошибка (Run: {run_id})")
        elif metrics and metrics.get("status") == "stopped":
            st.warning(f"⏹️ Тренировка остановлена (Run: {run_id})")
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
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_step = metrics.get("current_step", 0)
        total_steps = metrics.get("total_steps", 1)
        progress = current_step / total_steps * 100 if total_steps > 0 else 0
        st.metric("Прогресс", f"{progress:.1f}%", f"Step {current_step}/{total_steps}")
    
    with col2:
        loss = metrics.get("current_loss", 0)
        st.metric("Loss", f"{loss:.4f}")
    
    with col3:
        lr = metrics.get("current_lr", 0)
        st.metric("Learning Rate", f"{lr:.2e}")
    
    with col4:
        eta = metrics.get("eta_seconds", 0)
        st.metric("ETA", format_time(eta))
    
    # Progress bar
    st.progress(min(progress / 100, 1.0))
    
    # Charts
    if metrics.get("loss_history"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss chart
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=metrics["steps_history"],
                y=metrics["loss_history"],
                mode='lines',
                name='Loss',
                line=dict(color='#e94560', width=2)
            ))
            fig_loss.update_layout(
                title="Training Loss",
                xaxis_title="Step",
                yaxis_title="Loss",
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_loss, width="stretch")
        
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
            st.plotly_chart(fig_lr, width="stretch")
    
    # Checkpoints
    if metrics.get("checkpoints"):
        with st.expander("📦 Checkpoints"):
            for ckpt in metrics["checkpoints"]:
                st.text(f"Step {ckpt['step']}: {ckpt['path']}")
    
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



def download_hf_dataset(repo_id, subset, split, limit_val, limit_bytes, filter_lang, filter_score, filename):
    """Функция скачивания датасета (выполняется при нажатии кнопки)."""
    st.info(f"Начинаем скачивание: {repo_id}...")
    
    target_filename = filename if filename else f"{repo_id.split('/')[-1]}_{split}.jsonl"
    if not target_filename.endswith('.jsonl'):
        target_filename += '.jsonl'
    
    save_path = DATASET_DIR / target_filename
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    try:
        status_text.info(f"Подключение к {repo_id}...")
        
        # Обработка имени конфига
        config_name = subset if subset and subset != "default" else None
        
        # Пробуем загрузить
        try:
            ds = load_dataset(
                repo_id, 
                name=config_name, 
                split=split, 
                streaming=True,
                trust_remote_code=True
            )
        except ValueError as e:
            if "BuilderConfig" in str(e) and "not found" in str(e):
                # Пытаемся распарсить доступные конфиги из сообщения об ошибке
                import re
                match = re.search(r"Available: \[(.*?)\]", str(e))
                if match:
                    available = match.group(1).replace("'", "")
                    # Показываем подсказку
                    st.error(f"Конфиг '{subset}' не найден! Доступные конфиги (скопируйте нужный в поле Subset):")
                    st.code(available, language=None)
                    # Если есть rus_Cyrl, подсказываем его
                    if "rus_Cyrl" in available:
                        st.info("💡 Для русского языка попробуйте Subset: **rus_Cyrl**")
                    return
            raise e
        
        status_text.info(f"Скачивание в {target_filename}...")
        
        count = 0
        current_bytes = 0
        skipped_count = 0
        
        # Проверяем первый элемент
        iterator = iter(ds)
        try:
            first_item = next(iterator)
            items_to_process = [first_item]
        except StopIteration:
            st.error("Датасет пуст!")
            return
        
        with open(save_path, "w", encoding="utf-8") as f:
            import itertools
            for item in itertools.chain(items_to_process, iterator):
                # 1. Фильтрация
                if filter_lang:
                    item_lang = item.get("language")
                    if item_lang and filter_lang.lower() not in item_lang.lower():
                        skipped_count += 1
                        continue
                
                if filter_score > 0:
                    item_score = item.get("language_score")
                    if item_score is not None and float(item_score) < filter_score:
                        skipped_count += 1
                        continue
                
                # 2. Извлечение текста
                text = item.get("text") or item.get("content") or item.get("body")
                
                if text:
                    row_data = json.dumps({"text": text}, ensure_ascii=False)
                    row_bytes = len(row_data.encode('utf-8')) + 1
                    
                    f.write(row_data + "\n")
                    
                    count += 1
                    current_bytes += row_bytes
                    
                    if count % 100 == 0:
                        status_text.text(f"Скачано: {count} | {current_bytes / 1024**2:.1f} MB")
                        if limit_val > 0 and count >= limit_val:
                            break
                        if limit_bytes > 0 and current_bytes >= limit_bytes:
                            break
        
        status_text.success(f"✅ Готово! Сохранено в {target_filename}")
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        st.error(f"Ошибка: {e}")


def render_data_manager():
    """Вкладка управления данными."""
    st.header("💾 Управление данными")
    
    col_upload, col_list = st.columns([1, 2])
    
    with col_upload:
        # Секция 1: Загрузка локальных файлов
        with st.expander("📤 Загрузка локальных файлов", expanded=False):
            uploaded_files = st.file_uploader(
                "Перетащите файлы сюда", 
                type=["jsonl", "txt", "json"], 
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
        
        # Используем session_state для хранения значений формы
        if "hf_repo_id" not in st.session_state: st.session_state.hf_repo_id = "HuggingFaceFW/fineweb-2"
        if "hf_subset" not in st.session_state: st.session_state.hf_subset = "default"
        if "hf_split" not in st.session_state: st.session_state.hf_split = "train"
        if "hf_filename" not in st.session_state: st.session_state.hf_filename = ""
        
        # Инпуты обновляют session_state
        st.text_input("Репозиторий (ID)", key="hf_repo_id", help="Например: HuggingFaceFW/fineweb-2")
        st.text_input("Subset (конфиг)", key="hf_subset", help="Например: rus_Cyrl (для русского) или default. Если ошибка 'Config not found', смотрите список доступных в ошибке.")
        st.text_input("Split", key="hf_split")
        
        with st.expander("🛠️ Фильтры и Лимиты", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                filter_lang = st.text_input("Язык (rus)", value="rus", key="filter_lang")
                filter_score = st.slider("Мин. score", 0.0, 1.0, 0.0, key="filter_score")
            with col2:
                # По умолчанию - лимит в ГБ
                limit_type = st.radio("Тип лимита", ["ГБ (Размер)", "Строки (Количество)"], key="limit_type")
                
                limit_val = 0
                limit_bytes = 0
                
                if limit_type == "Строки (Количество)":
                    limit_val = st.number_input("Кол-во строк", value=100000, step=10000, key="limit_val")
                else:
                    limit_gb = st.number_input("Размер (ГБ)", value=2.0, step=0.5, min_value=0.1, key="limit_gb")
                    limit_bytes = int(limit_gb * 1024**3)
        
        st.text_input("Имя файла", key="hf_filename", placeholder="dataset.jsonl")
        
        # Кнопка запускает функцию
        if st.button("🚀 Скачать", type="primary"):
            if not st.session_state.hf_repo_id:
                st.error("Укажите ID репозитория!")
            else:
                download_hf_dataset(
                    st.session_state.hf_repo_id,
                    st.session_state.hf_subset,
                    st.session_state.hf_split,
                    limit_val,
                    limit_bytes,
                    filter_lang,
                    filter_score,
                    st.session_state.hf_filename
                )
    
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
                            head = [next(f).strip() for _ in range(5)]
                        st.markdown("**Preview (первые 5 строк):**")
                        st.code("\n".join(head), language="json" if "JSON" in ds['type'] else "text")
                        
                        col_del, col_info = st.columns([1, 4])
                        with col_del:
                            if st.button("🗑️ Удалить", key=f"del_{ds['name']}"):
                                ds['path'].unlink()
                                st.toast(f"Файл {ds['name']} удалён", icon="🗑️")
                                time.sleep(1)
                                st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка чтения файла: {e}")


def render_model_preview(config: dict, distributed_config: dict = None):
    """Превью архитектуры модели и настроек параллелизма."""
    st.subheader("📐 Архитектура модели")
    
    params = estimate_parameters(config["hidden_size"], config["num_layers"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Hidden Size", config["hidden_size"])
        st.metric("Layers", config["num_layers"])
    
    with col2:
        st.metric("Attention Heads", config["n_heads"])
        st.metric("Head Dim", config["hidden_size"] // config["n_heads"])
    
    with col3:
        st.metric("Параметры", format_params(params))
        vram_est = params * 4 / 1e9  # fp32
        st.metric("VRAM (≈ fp32)", f"{vram_est:.1f} GB")
    
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


# ============================================================================
# Main App
# ============================================================================

def main():
    render_header()
    
    # Sidebar configs
    model_config = render_model_config()
    training_config = render_training_config()
    distributed_config = render_distributed_config()
    dataset_config = render_dataset_config()
    output_config = render_output_config()
    
    # Merge configs
    full_config = {**model_config, **training_config, **dataset_config, **output_config}
    full_config["distributed_mode"] = distributed_config["distributed_mode"]
    full_config["num_gpus"] = distributed_config["num_gpus"]
    full_config["config_file"] = distributed_config["config_file"]
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Запуск", "📊 Мониторинг", "💬 Чат", "📜 История", "💾 Данные"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_model_preview(model_config, distributed_config)
            
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
    
    with tab3:
        st.header("📜 История запусков")
        st.markdown("---")
        
        runs = sorted(RUNS_DIR.glob("*"), reverse=True)
        
        if runs:
            for run_dir in runs[:10]:  # Last 10 runs
                run_id = run_dir.name
                metrics = load_metrics(run_id)
                
                if metrics:
                    status = metrics.get("status", "unknown")
                    status_emoji = {"training": "🟢", "completed": "✅", "error": "❌", "stopped": "⏹️"}.get(status, "⏳")
                    
                    with st.expander(f"{status_emoji} {run_id}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Steps", metrics.get("current_step", 0))
                        with col2:
                            st.metric("Final Loss", f"{metrics.get('current_loss', 0):.4f}")
                        with col3:
                            st.metric("Status", status)
                        
                        # Чекпоинты этого запуска
                        checkpoints = metrics.get("checkpoints", [])
                        if checkpoints:
                            st.markdown("**📦 Чекпоинты:**")
                            for ckpt in checkpoints[-5:]:  # Последние 5
                                st.caption(f"Step {ckpt['step']}: `{ckpt['path']}`")
                        
                        # Кнопки
                        btn_col1, btn_col2 = st.columns(2)
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
    
    with tab4:
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
            
            # Инициализация чата
            if "chat_model" not in st.session_state:
                st.session_state.chat_model = None
                st.session_state.chat_tokenizer = None
                st.session_state.chat_model_path = None
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Кнопка загрузки модели
            if st.session_state.chat_model_path != selected_model["path"]:
                if st.button("🔄 Загрузить модель", type="primary"):
                    with st.spinner("Загружаем модель..."):
                        try:
                            from transformers import AutoTokenizer
                            from homellm.models.home_model import HomeForCausalLM, HomeConfig
                            from safetensors.torch import load_file
                            
                            model_path = Path(selected_model["path"])
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            
                            # Определяем тип чекпоинта
                            config_json = model_path / "config.json"
                            model_safetensors = model_path / "model.safetensors"
                            tokenizer_json = model_path / "tokenizer.json"
                            tokenizer_config = model_path / "tokenizer_config.json"
                            
                            # HuggingFace формат = есть tokenizer файлы
                            is_hf_format = tokenizer_json.exists() or tokenizer_config.exists()
                            
                            if is_hf_format and config_json.exists():
                                # HuggingFace формат (final_model с tokenizer)
                                st.info("Загружаем HuggingFace модель...")
                                st.session_state.chat_tokenizer = AutoTokenizer.from_pretrained(
                                    str(model_path), 
                                    trust_remote_code=True
                                )
                                st.session_state.chat_model = HomeForCausalLM.from_pretrained(
                                    str(model_path)
                                ).to(device)
                            elif model_safetensors.exists():
                                # Accelerate checkpoint формат
                                st.info("Загружаем Accelerate чекпоинт...")
                                
                                # Загружаем токенизатор GPT-2 (по умолчанию)
                                st.session_state.chat_tokenizer = AutoTokenizer.from_pretrained("gpt2")
                                if st.session_state.chat_tokenizer.pad_token is None:
                                    st.session_state.chat_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                                
                                # Пытаемся загрузить конфиг модели из чекпоинта
                                if config_json.exists():
                                    config = HomeConfig.from_pretrained(str(model_path))
                                    st.info(f"Конфиг загружен: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
                                else:
                                    # Ищем конфиг в родительской директории (run config)
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
                                        st.info(f"Конфиг из run_config: hidden_size={config.hidden_size}")
                                    else:
                                        st.warning("⚠️ config.json не найден в чекпоинте, используем дефолтные параметры")
                                        config = HomeConfig(
                                            vocab_size=len(st.session_state.chat_tokenizer),
                                            hidden_size=512,
                                            num_hidden_layers=8,
                                            num_attention_heads=8,
                                            max_position_embeddings=512,
                                        )
                                
                                st.session_state.chat_model = HomeForCausalLM(config)
                                
                                # Загружаем веса
                                state_dict = load_file(str(model_safetensors))
                                st.session_state.chat_model.load_state_dict(state_dict)
                                st.session_state.chat_model = st.session_state.chat_model.to(device)
                            else:
                                raise ValueError(f"Не найден config.json или model.safetensors в {model_path}")
                            
                            st.session_state.chat_model.eval()
                            st.session_state.chat_model_path = str(model_path)
                            st.session_state.messages = []
                            st.success("✅ Модель загружена!")
                            st.rerun()
                        except Exception as e:
                            import traceback
                            st.error(f"Ошибка загрузки: {e}")
                            st.code(traceback.format_exc())
            else:
                st.success(f"✅ Модель загружена: {selected_model_name}")
                
                # Показываем историю чата
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                
                # Ввод пользователя
                if prompt := st.chat_input("Введите сообщение..."):
                    # Добавляем сообщение пользователя
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    # Генерируем ответ
                    with st.chat_message("assistant"):
                        with st.spinner("Генерация..."):
                            try:
                                tokenizer = st.session_state.chat_tokenizer
                                model = st.session_state.chat_model
                                device = next(model.parameters()).device
                                
                                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                                
                                with torch.no_grad():
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
                                st.error(f"Ошибка генерации: {e}")
                
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

