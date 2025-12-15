"""
HomeLLM Training Studio — Визуальное приложение для тренировки моделей
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
# Session State
# ============================================================================

if "training_process" not in st.session_state:
    st.session_state.training_process = None
if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None
if "training_active" not in st.session_state:
    st.session_state.training_active = False


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
    
    # Запускаем процесс с логированием в файлы
    cmd = [
        "python", "-m", "homellm.app.trainer_worker",
        "--config", str(config_path),
        "--metrics", str(metrics_path)
    ]
    
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
    if st.session_state.training_process:
        try:
            os.kill(st.session_state.training_process.pid, signal.SIGTERM)
        except:
            pass
        st.session_state.training_process = None
        st.session_state.training_active = False
    
    # Также попробуем убить по PID из файла
    if st.session_state.current_run_id:
        pid_path = RUNS_DIR / st.session_state.current_run_id / "pid"
        if pid_path.exists():
            try:
                with open(pid_path) as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGTERM)
            except:
                pass


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
    except (ProcessLookupError, ValueError, FileNotFoundError):
        return False


# ============================================================================
# UI Components
# ============================================================================

def render_header():
    st.markdown("# 🏠 HomeLLM Training Studio")
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
    
    epochs = st.sidebar.number_input(
        "Epochs",
        min_value=1,
        max_value=10,
        value=1
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
        "Save Every N Steps",
        min_value=100,
        max_value=50000,
        value=5000
    )
    
    log_every = st.sidebar.number_input(
        "Log Every N Steps",
        min_value=1,
        max_value=1000,
        value=10
    )
    
    return {
        "output_dir": output_dir,
        "save_every": save_every,
        "log_every": log_every,
        "tokenizer_path": "gpt2"
    }


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
    
    # Error
    if metrics.get("error"):
        st.error(f"Ошибка: {metrics['error']}")
    
    # Логи процесса
    if st.session_state.current_run_id:
        run_dir = RUNS_DIR / st.session_state.current_run_id
        stderr_path = run_dir / "stderr.log"
        stdout_path = run_dir / "stdout.log"
        
        with st.expander("📋 Логи процесса"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.caption("stdout")
                if stdout_path.exists():
                    with open(stdout_path) as f:
                        content = f.read()[-2000:]  # Последние 2000 символов
                        st.code(content if content else "(пусто)", language=None)
            
            with col2:
                st.caption("stderr")
                if stderr_path.exists():
                    with open(stderr_path) as f:
                        content = f.read()[-2000:]
                        st.code(content if content else "(пусто)", language=None)


def render_model_preview(config: dict):
    """Превью архитектуры модели."""
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


# ============================================================================
# Main App
# ============================================================================

def main():
    render_header()
    
    # Sidebar configs
    model_config = render_model_config()
    training_config = render_training_config()
    dataset_config = render_dataset_config()
    output_config = render_output_config()
    
    # Merge configs
    full_config = {**model_config, **training_config, **dataset_config, **output_config}
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🚀 Запуск", "📊 Мониторинг", "📜 История"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_model_preview(model_config)
            
            st.subheader("📋 Конфигурация")
            st.json(full_config)
        
        with col2:
            st.subheader("🎮 Управление")
            
            if st.session_state.training_active:
                if st.button("⏹️ Остановить", type="primary"):
                    stop_training()
                    st.success("Тренировка остановлена")
                    st.rerun()
            else:
                if st.button("▶️ Начать тренировку", type="primary"):
                    with st.spinner("Запуск..."):
                        run_id, process = start_training(full_config)
                        st.session_state.current_run_id = run_id
                        st.session_state.training_process = process
                        st.session_state.training_active = True
                        st.success(f"Тренировка запущена! Run ID: {run_id}")
                        time.sleep(1)
                        st.rerun()
    
    with tab2:
        if st.session_state.current_run_id:
            run_id = st.session_state.current_run_id
            metrics = load_metrics(run_id)
            process_alive = is_process_running(run_id)
            
            # Показываем статус процесса
            if process_alive:
                st.success(f"🟢 Процесс запущен (Run: {run_id})")
            else:
                if metrics and metrics.get("status") == "completed":
                    st.success(f"✅ Тренировка завершена (Run: {run_id})")
                elif metrics and metrics.get("status") == "error":
                    st.error(f"❌ Ошибка (Run: {run_id})")
                else:
                    st.warning(f"⚠️ Процесс не запущен (Run: {run_id})")
            
            if metrics:
                render_metrics_dashboard(metrics)
                
                # Auto-refresh пока процесс жив или статус training
                if process_alive or metrics.get("status") in ["training", "initializing", "loading_tokenizer", "loading_dataset", "building_model"]:
                    time.sleep(2)
                    st.rerun()
            else:
                st.info("Ожидание метрик...")
                if process_alive:
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("Запустите тренировку для просмотра метрик")
    
    with tab3:
        st.subheader("📜 История запусков")
        
        runs = sorted(RUNS_DIR.glob("*"), reverse=True)
        
        if runs:
            for run_dir in runs[:10]:  # Last 10 runs
                run_id = run_dir.name
                metrics = load_metrics(run_id)
                
                if metrics:
                    status = metrics.get("status", "unknown")
                    status_emoji = {"training": "🟢", "completed": "✅", "error": "❌"}.get(status, "⏳")
                    
                    with st.expander(f"{status_emoji} {run_id}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Steps", metrics.get("current_step", 0))
                        with col2:
                            st.metric("Final Loss", f"{metrics.get('current_loss', 0):.4f}")
                        with col3:
                            st.metric("Status", status)
                        
                        if st.button(f"Загрузить {run_id}", key=run_id):
                            st.session_state.current_run_id = run_id
                            st.rerun()
        else:
            st.info("Нет предыдущих запусков")


if __name__ == "__main__":
    main()

