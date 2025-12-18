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
                sys_val = default_system
                if sft_columns.get("system_field"):
                    field_sys = get_nested_value(sample, sft_columns["system_field"])
                    if field_sys: sys_val = str(field_sys)[:200]
                
                preview = f"{sys_val}{sep}{user_tag}\n{user_val}{sep}{assistant_tag}\n{asst_val}<|endoftext|>"
            
            with st.container(height=400):
                st.code(preview, language=None)
            
            st.success("✅ Готово!")
            
        except Exception as e:
            st.error(f"Ошибка: {e}")

    return {"sft_columns": sft_columns, "sft_template": sft_template}


def render_model_config():
    """Конфигуратор модели в сайдбаре."""
    st.sidebar.header("🧠 Архитектура и Режим")
    
    # Режим обучения
    stage_options = {
        "pretrain": "Pretraining (с нуля)",
        "sft": "SFT (Fine-Tuning)"
    }
    selected_stage = st.sidebar.selectbox(
        "Этап обучения",
        options=list(stage_options.keys()),
        format_func=lambda x: stage_options[x],
        help="Выберите этап: обучение с нуля или дообучение существующей модели"
    )
    
    # Имя модели (для папки эксперимента)
    model_name_default = "home_pretrain" if selected_stage == "pretrain" else "home_sft"
    model_name = st.sidebar.text_input("Название эксперимента", value=model_name_default, help="Имя папки для сохранения")
    
    base_model_path = None
    
    if selected_stage == "sft":
        st.sidebar.subheader("📦 Базовая модель")
        available = get_available_models()
        if not available:
            st.sidebar.warning("Нет доступных моделей для SFT. Сначала обучите Pretrain модель!")
            # Можно дать возможность ввести путь вручную
            base_model_path = st.sidebar.text_input("Путь к модели вручную", placeholder="/path/to/model")
        else:
            # Создаем список опций
            model_options = [m["name"] for m in available]
            selected_base_name = st.sidebar.selectbox("Выберите модель", options=model_options)
            # Находим путь
            base_model_path = next(m["path"] for m in available if m["name"] == selected_base_name)
            
            st.sidebar.caption(f"Путь: `{base_model_path}`")
    
    # Флаг, что параметры загружены из конфига
    loaded_config = None
    
    if selected_stage == "sft" and base_model_path:
        # Пытаемся загрузить конфиг
        try:
            base_path = Path(base_model_path)
            # Вариант 1: config.json прямо в папке (final_model)
            cfg_path = base_path / "config.json"
            # Вариант 2: это чекпоинт
            if not cfg_path.exists():
                # Пробуем найти run_config.json в родительской
                if (base_path.parent / "run_config.json").exists():
                    cfg_path = base_path.parent / "run_config.json"
                elif (base_path / "run_config.json").exists(): # иногда сохраняем так
                     cfg_path = base_path / "run_config.json"
            
            if cfg_path.exists():
                with open(cfg_path) as f:
                    loaded_config = json.load(f)
                st.sidebar.success("✅ Параметры загружены из базовой модели")
            else:
                st.sidebar.warning("⚠️ config.json не найден, введите параметры вручную")
        except Exception as e:
             st.sidebar.error(f"Ошибка чтения config.json: {e}")

    st.sidebar.subheader("⚙️ Параметры модели")
    
    if loaded_config:
        # Режим только чтения для SFT с загруженным конфигом
        # Используем значения из конфига (поддержка разных имен ключей)
        hidden_size = loaded_config.get("hidden_size", 512)
        # num_hidden_layers - HF, num_layers - наш конфиг
        num_layers = loaded_config.get("num_hidden_layers", loaded_config.get("num_layers", 8))
        num_attention_heads = loaded_config.get("num_attention_heads", loaded_config.get("n_heads", 8))
        max_position_embeddings = loaded_config.get("max_position_embeddings", loaded_config.get("seq_len", 512))
        
        # Для совместимости возвращаем имена переменных как ожидается
        n_heads = num_attention_heads
        seq_len = max_position_embeddings
        
        # Отображаем просто текстом/метриками
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Hidden Size", hidden_size)
        c2.metric("Layers", num_layers)
        c1.metric("Heads", n_heads)
        c2.metric("Seq Len", seq_len)
        
        st.sidebar.info("🔒 Параметры зафиксированы (наследуются от базы)")
        
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
        "stage": selected_stage,
        "base_model_path": base_model_path,
        "model_name_input": model_name,
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
    """Получить список доступных обученных моделей (рекурсивный поиск)."""
    models = []
    
    # Ищем рекурсивно в out/
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
    
    # Сортируем по времени (новые сверху)
    models.sort(key=lambda x: x["time"], reverse=True)
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
            duration = metrics.get("training_duration", "unknown")
            st.success(f"✅ Тренировка завершена за {duration} (Run: {run_id})")
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
        elapsed = metrics.get("elapsed_seconds", 0)
        st.metric("Время", f"{format_time(elapsed)}", delta=f"Ост: {format_time(eta)}", delta_color="normal")
    
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
            st.plotly_chart(fig_loss, key=f"loss_chart_{metrics.get('current_step')}")
        
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
            st.plotly_chart(fig_lr, key=f"lr_chart_{metrics.get('current_step')}")
    
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



def download_hf_dataset(repo_id, subset, split, limit_type, limit_val, limit_bytes, save_path, filters=None):
    """Функция скачивания и сохранения датасета."""
    try:
        status_text = f"Начинаем скачивание: {repo_id}..."
        st.toast(status_text)
        print(status_text)
        
        # Параметры stream=True чтобы не качать все в память
        ds = load_dataset(repo_id, subset, split=split, streaming=True)
        
        # Создаем файл
        save_path = DATASET_DIR / save_path
        save_path.parent.mkdir(exist_ok=True)
        
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
        # Словарь пресетов: {название: (repo_id, subset, split)}
        presets = {
            "🟢 Pretrain: FineWeb-2 (Russian)": ("HuggingFaceFW/fineweb-2", "rus_Cyrl", "train"),
            "🟢 Pretrain: FineWeb-Edu (Educational)": ("HuggingFaceFW/fineweb-edu", "default", "train"),
            "🟢 Pretrain: Wikitext-103": ("wikitext", "wikitext-103-v1", "train"),
            "🔵 SFT: OpenOrca-ru": ("d0rj/OpenOrca-ru", "default", "train"),
            "🔵 SFT: ru-instruct": ("d0rj/ru-instruct", "default", "train"),
            "🔵 SFT: GrandMaster-PRO-MAX": ("Vikhrmodels/GrandMaster-PRO-MAX", "default", "train"),
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


        save_filename = st.text_input("Имя файла для сохранения", "dataset.jsonl", key="save_filename")
        
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
            l_val = st.session_state.get('limit_val')
            
            l_gb = st.session_state.get('limit_gb', 2.0)
            l_bytes = int(l_gb * 1024**3)

            s_path = st.session_state.get('save_filename')
            
            # Собираем фильтры
            filters_to_pass = {}
            
            # 1. Динамические фильтры
            if active_filters_map:
                if "score_col" in active_filters_map:
                    filters_to_pass["score_col"] = active_filters_map["score_col"]
                    filters_to_pass["min_score"] = st.session_state.get("filter_score", 0.0)
                
                if "lang_col" in active_filters_map:
                    filters_to_pass["lang_col"] = active_filters_map["lang_col"]
                    filters_to_pass["target_lang"] = st.session_state.get("filter_lang", "ru")
            
            # 2. Фолбэк для FineWeb (удален, так как вызывал путаницу)
            # elif "fineweb" in r_id.lower():
            #      pass
            
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


def calculate_memory_footprint(config, batch_size, distributed_mode="default", num_gpus=1):
    """
    Рассчитывает потребление VRAM (в ГБ) для обучения.
    Учитывает: веса, оптимизатор, градиенты и активации.
    """
    try:
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        n_heads = config["n_heads"]
        seq_len = config["seq_len"]
        vocab_size = 50257  # Примерно для GPT-2 / Llama
        
        # 1. Параметры модели (P)
        embed_params = vocab_size * hidden_size
        layer_params = 12 * hidden_size**2 + 13 * hidden_size # Упрощенная формула для блока трансформера
        total_params = embed_params + num_layers * layer_params
        
        # 2. Статическая память (Веса + Градиенты + Оптимизатор)
        # Базовая Mixed Precision (fp16/bf16):
        # - Weights (fp16): 2 bytes
        # - Gradients (fp16): 2 bytes
        # - Optimizer (AdamW):
        #    - FP32 Master weights: 4 bytes
        #    - Momentum (fp32): 4 bytes
        #    - Variance (fp32): 4 bytes
        # Итого: ~16-18 байт на параметр.
        
        bytes_per_param = 18 
        
        # Учет Distributed стратегий
        if "deepspeed_zero3" in distributed_mode:
            # ZeRO-3 шардирует все (веса, градиенты, оптимизатор)
            static_mem_bytes = (total_params * bytes_per_param) / max(1, num_gpus)
        elif "deepspeed_zero2" in distributed_mode:
            # ZeRO-2 шардирует градиенты и оптимизатор (8+4+4=16 bytes), но веса (2 bytes) дублируются
            sharded_part = (total_params * 16) / max(1, num_gpus)
            replicated_part = total_params * 2
            static_mem_bytes = sharded_part + replicated_part
        elif distributed_mode == "fsdp":
            # FSDP похож на ZeRO-3
            static_mem_bytes = (total_params * bytes_per_param) / max(1, num_gpus)
        else:
            # DDP или Single GPU: полная копия у всех
            static_mem_bytes = total_params * bytes_per_param

        # 3. Динамическая память (Активации)
        # Зависит от Batch Size и Seq Len.
        # Формула: Batch * Seq * Hidden * Layers * Bytes * Overhead_Factor
        # Overhead_Factor для трансформеров без checkpointing ~34 (храним все промежуточные состояния)
        # С checkpointing ~4 (храним только входы слоев + пересчет)
        
        overhead_factor = 4 if config.get("grad_checkpoint") else 34
        activation_bytes = batch_size * seq_len * hidden_size * num_layers * 2 * overhead_factor
        
        # Конвертация в ГБ
        static_gb = static_mem_bytes / (1024**3)
        act_gb = activation_bytes / (1024**3)
        buffer_gb = 1.5  # Буфер для PyTorch context, cuda kernels fragmentation
        
        total_gb = static_gb + act_gb + buffer_gb
        
        return {
            "total_gb": round(total_gb, 2),
            "model_gb": round(static_gb, 2),
            "act_gb": round(act_gb, 2),
            "params": total_params
        }
    except Exception as e:
        print(f"Error calculating VRAM: {e}")
        return {"total_gb": 0, "model_gb": 0, "act_gb": 0, "params": 0}


def render_model_preview(config: dict, distributed_config: dict = None):
    """Превью архитектуры модели и настроек параллелизма."""
    st.subheader("📐 Архитектура модели")
    
    stage = config.get("stage", "pretrain")
    if stage == "sft":
        st.info(f"🔄 **Режим SFT** (Fine-Tuning)\nБазовая модель: `{Path(config.get('base_model_path') or 'Unknown').name}`")
    else:
        st.success("🏗️ **Режим Pretraining** (С нуля)")

    # Рассчитываем память
    # Нам нужен batch_size из конфига (это батч на девайс)
    batch_size = config.get("batch_size", 1)
    dist_mode = distributed_config.get("distributed_mode", "default") if distributed_config else "default"
    n_gpus = distributed_config.get("num_gpus", 1) if distributed_config else 1
    
    mem_info = calculate_memory_footprint(config, batch_size, dist_mode, n_gpus)
    
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
        
        st.metric(
            "VRAM (Estimate)", 
            f"{val:.1f} GB", 
            delta=f"M: {mem_info['model_gb']} + A: {mem_info['act_gb']} GB",
            delta_color=color,
            help="M: Static Model Memory (Weights+Optim)\nA: Activations (Batch Size dependent)"
        )
    
    # Визуализация использования памяти
    if mem_info["total_gb"] > 0:
        st.caption("📊 Примерное распределение памяти GPU:")
        
        # Создаем простой бар чарт через HTML/CSS для наглядности
        total = mem_info["total_gb"]
        p_model = (mem_info["model_gb"] / total) * 100
        p_act = (mem_info["act_gb"] / total) * 100
        p_buff = 100 - p_model - p_act
        
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
    """Экспортирует модель и токенизатор в стандартный HF формат."""
    try:
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
        
        # Сохраняем
        model.save_pretrained(export_dir)
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
    
    training_config = render_training_config()
    distributed_config = render_distributed_config()
    
    # Передаем stage в dataset_config
    dataset_config = render_dataset_config(stage=model_config.get("stage", "pretrain"))
    
    output_config = render_output_config(st.session_state.current_model_name)
    
    # Merge configs
    full_config = {**model_config, **training_config, **dataset_config, **output_config}
    full_config["distributed_mode"] = distributed_config["distributed_mode"]
    full_config["num_gpus"] = distributed_config["num_gpus"]
    full_config["config_file"] = distributed_config["config_file"]
    
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚀 Запуск", "📊 Мониторинг", "💬 Чат", "📜 История", "💾 Данные", "📚 Учебник"])
    
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
    
    with tab4:
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
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Steps", metrics.get("current_step", 0))
                        with col2:
                            st.metric("Final Loss", f"{metrics.get('current_loss', 0):.4f}")
                        with col3:
                            st.metric("Status", status)
                        with col4:
                            st.metric("Duration", metrics.get("training_duration", "-"))
                        
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

    with tab6:
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
                                missing, unexpected = st.session_state.chat_model.load_state_dict(state_dict, strict=False)
                                
                                # Умная проверка пропущенных ключей
                                if missing:
                                    # Игнорируем lm_head.weight, так как он связан с embed_tokens
                                    real_missing = [k for k in missing if k != "lm_head.weight"]
                                    if real_missing:
                                        st.warning(f"⚠️ Внимание! Отсутствуют веса: {real_missing[:5]}... (всего {len(real_missing)})")
                                        logger.warning(f"Missing keys: {real_missing}")
                                    else:
                                        # Если не хватает только lm_head, значит все ок
                                        logger.info("Missing only lm_head.weight (expected for tied weights)")
                                
                                if unexpected:
                                    st.warning(f"⚠️ Найдены лишние ключи в чекпоинте: {unexpected[:5]}...")

                                # Явно связываем веса после загрузки
                                if hasattr(st.session_state.chat_model, "tie_weights"):
                                    st.session_state.chat_model.tie_weights()
                                    
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
                                    # Используем chat_template если он есть (для SFT моделей)
                                    # Иначе просто склеиваем текст (для Pretrain моделей)
                                    
                                    # Берем историю + новое сообщение
                                    conversation = st.session_state.messages # [{"role": "user", ...}, ...]
                                    
                                    if tokenizer.chat_template:
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
                                    
                                    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                                    
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


