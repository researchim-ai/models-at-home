"""Notebook endpoints: list local .ipynb, expose templates, check JupyterLab."""
from __future__ import annotations

import urllib.request
import urllib.error
import os
import signal
import subprocess
import time
import sys
import secrets
import threading
import json
from urllib.parse import urlsplit
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from backend.paths import get_project_root
from backend.services import processes

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
JUPYTER_RUN = PROJECT_ROOT / ".runs" / "jupyter"
JUPYTER_PID = JUPYTER_RUN / "pid"
JUPYTER_LOG = JUPYTER_RUN / "stdout.log"
JUPYTER_TOKEN = os.environ.get("JUPYTER_TOKEN") or ""
JUPYTER_PORT = int(os.environ.get("JUPYTER_PORT", "8888"))
JUPYTER_PUBLIC_URL = os.environ.get("JUPYTER_PUBLIC_URL", f"http://127.0.0.1:{JUPYTER_PORT}")
_start_lock = threading.Lock()


def _connection_token() -> str:
    global JUPYTER_TOKEN
    connection = JUPYTER_RUN / 'connection.json'
    saved = None
    try:
        value = json.loads(connection.read_text()).get('token')
        if isinstance(value, str) and value:
            saved = value
    except (OSError, ValueError, AttributeError):
        pass
    desired = JUPYTER_TOKEN or saved or secrets.token_urlsafe(32)
    if _pid_running():
        if not processes.is_owned(JUPYTER_RUN):
            raise HTTPException(409, 'Jupyter process ownership cannot be verified')
        if saved != desired:
            processes.stop(JUPYTER_RUN)
    from backend.paths import atomic_json
    JUPYTER_TOKEN = desired
    atomic_json(connection, {'token': desired})
    return JUPYTER_TOKEN

# Curated templates that ship with the project
TEMPLATES = [
    {
        "name": "Inference Playground",
        "description": "Интерактивный inference: попробовать любую модель",
        "file": "inference_playground.ipynb",
    },
    {
        "name": "Dataset Download",
        "description": "Скачать датасет с HuggingFace и положить в datasets/",
        "file": "dataset_download.ipynb",
    },
    {
        "name": "SFT Playground",
        "description": "Fine-tune модели на chat-данных",
        "file": "sft_playground.ipynb",
    },
    {
        "name": "GRPO Mini Demo",
        "description": "RL-обучение через GRPO на небольшом примере",
        "file": "grpo_mini_demo.ipynb",
    },
    {
        "name": "Memory Estimator Demo",
        "description": "Оценка VRAM для разных архитектур",
        "file": "memory_estimator_demo.ipynb",
    },
    {
        "name": "Blueprint from JSON",
        "description": "Собрать модель из blueprint JSON",
        "file": "blueprint_from_json.ipynb",
    },
    {
        "name": "Custom Block Compare",
        "description": "Сравнение кастомных блоков (норм, активаций, MoE)",
        "file": "custom_block_compare.ipynb",
    },
    {
        "name": "MoE Playground",
        "description": "Эксперименты с Mixture of Experts",
        "file": "moe_playground.ipynb",
    },
    {
        "name": "Llama Default Train",
        "description": "Базовая тренировка Llama-архитектуры",
        "file": "llama_default_train.ipynb",
    },
    {
        "name": "Llama Module Playground",
        "description": "Экспериментировать с отдельными модулями Llama",
        "file": "llama_module_playground.ipynb",
    },
    {
        "name": "Optimizers Compare",
        "description": "Сравнение AdamW / SGD / Lion / ...",
        "file": "optimizers_compare.ipynb",
    },
    {
        "name": "Data Formats (SFT/RL)",
        "description": "Форматы данных для SFT и RL тренировки",
        "file": "data_formats_sft_rl.ipynb",
    },
]


@router.get("/list")
def list_notebooks():
    """List .ipynb files in the notebooks directory."""
    items = []
    if not NOTEBOOKS_DIR.exists():
        return {"notebooks": []}

    for f in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        try:
            stat = f.stat()
            items.append({
                "name": f.name,
                "path": f.name,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        except Exception:
            continue

    return {"notebooks": items}


@router.get("/templates")
def notebook_templates():
    """Expose curated templates. Only returns those that exist on disk."""
    available = []
    for tmpl in TEMPLATES:
        if (NOTEBOOKS_DIR / tmpl["file"]).exists():
            available.append(tmpl)
    return {"templates": available}


@router.get("/jupyter-status")
def jupyter_status(url: str = Query(...)):
    """Check if JupyterLab is running by hitting /api/status."""
    try:
        endpoint = f"{_probe_url(url).rstrip('/')}/api/status"
        req = urllib.request.Request(endpoint, headers={"User-Agent": "mah-studio"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            return {"running": resp.status == 200, "url": url}
    except urllib.error.HTTPError as e:
        return {"running": e.code in (401, 403), "url": url}
    except Exception:
        return {"running": False, "url": url}


@router.post("/jupyter-start")
def jupyter_start():
    """Start a managed JupyterLab process rooted at models-at-home/notebooks."""
    with _start_lock:
        return _start_jupyter()


def _start_jupyter():
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    JUPYTER_PID.parent.mkdir(parents=True, exist_ok=True)
    _connection_token()

    if _pid_running():
        return {
            "running": True,
            "url": JUPYTER_PUBLIC_URL,
            "token": JUPYTER_TOKEN,
            "root_dir": str(NOTEBOOKS_DIR),
        }

    cmd = [
        sys.executable,
        "-m",
        "jupyterlab",
        "--no-browser",
        f"--ip={os.environ.get('JUPYTER_BIND_HOST', '127.0.0.1')}",
        f"--port={JUPYTER_PORT}",
        f"--IdentityProvider.token={JUPYTER_TOKEN}",
        "--ServerApp.port_retries=0",
        f"--ServerApp.root_dir={NOTEBOOKS_DIR}",
        "--ServerApp.allow_remote_access=True",
        "--allow-root",
    ]

    try:
        proc = processes.launch(JUPYTER_RUN, cmd, cwd=PROJECT_ROOT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except Exception as exc:
        raise HTTPException(500, f"Failed to start JupyterLab: {exc}")

    if not _wait_for_jupyter_ready(f"http://127.0.0.1:{JUPYTER_PORT}", timeout=30, proc=proc):
        tail = ""
        if JUPYTER_LOG.exists():
            tail = "\n".join(JUPYTER_LOG.read_text(errors="ignore").splitlines()[-20:])
        processes.stop(JUPYTER_RUN)
        raise HTTPException(500, f"JupyterLab did not become ready. Log tail:\n{tail}")

    return {
        "running": True,
        "url": JUPYTER_PUBLIC_URL,
        "token": JUPYTER_TOKEN,
        "root_dir": str(NOTEBOOKS_DIR),
    }


@router.post("/jupyter-stop")
def jupyter_stop():
    """Stop the managed JupyterLab process if it was started by the backend."""
    processes.stop(JUPYTER_RUN)
    return {"success": True}


def _pid_running() -> bool:
    return processes.is_running(JUPYTER_RUN)


def _wait_for_jupyter_ready(url: str, timeout: int = 30, proc=None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            endpoint = f"{url.rstrip('/')}/api/status"
            req = urllib.request.Request(endpoint, headers={"User-Agent": "mah-studio", "Authorization": f"token {JUPYTER_TOKEN}"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return False
        except Exception:
            pass
        time.sleep(1)
    return False


def _probe_url(url: str) -> str:
    """Use the container-internal URL when checking the managed Jupyter server."""
    if url.rstrip("/") == JUPYTER_PUBLIC_URL.rstrip("/"):
        return f"http://127.0.0.1:{JUPYTER_PORT}"
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise HTTPException(400, "Expected an HTTP(S) Jupyter URL")
    return url

# Native Studio editor. Legacy JupyterLab endpoints remain for older clients.
import hashlib
from pydantic import BaseModel, Field
from backend.paths import safe_child, atomic_json
from backend.services import notebook_runtime

_document_lock = threading.Lock()


class NotebookWrite(BaseModel):
    path: str = Field(max_length=240)
    notebook: dict
    revision: str | None = None


class NotebookSession(BaseModel):
    path: str = Field(max_length=240)


class CellExecution(BaseModel):
    code: str = Field(max_length=100_000)
    cell_id: str = Field(min_length=1, max_length=100)


def _notebook_path(name: str) -> Path:
    path = safe_child(NOTEBOOKS_DIR, name)
    if path.suffix != '.ipynb':
        raise HTTPException(400, 'Expected an .ipynb file')
    return path


def _read_notebook(path: Path):
    if not path.is_file():
        raise HTTPException(404, 'Notebook not found')
    if path.stat().st_size > 10_000_000:
        raise HTTPException(413, 'Notebook exceeds the 10 MB editor limit')
    raw = path.read_bytes()
    try:
        notebook = json.loads(raw)
        _validate_notebook(notebook)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, f'Invalid notebook: {exc}')
    return notebook, hashlib.sha256(raw).hexdigest()


def _validate_notebook(notebook):
    if not isinstance(notebook, dict) or notebook.get('nbformat') != 4:
        raise ValueError('Only nbformat 4 notebooks are supported')
    cells = notebook.get('cells')
    if not isinstance(cells, list) or len(cells) > 500:
        raise ValueError('Expected at most 500 cells')
    if not isinstance(notebook.get('metadata'), dict):
        raise ValueError('Notebook metadata must be an object')
    for cell in cells:
        if not isinstance(cell, dict) or cell.get('cell_type') not in ('code', 'markdown', 'raw'):
            raise ValueError('Invalid cell type')
        source = cell.get('source')
        if not isinstance(source, str) and not (isinstance(source, list) and all(isinstance(s, str) for s in source)):
            raise ValueError('Invalid cell source')
        if cell['cell_type'] == 'code' and not isinstance(cell.get('outputs', []), list):
            raise ValueError('Invalid outputs')
        for output in cell.get('outputs', []):
            if not isinstance(output, dict) or not isinstance(output.get('output_type'), str):
                raise ValueError('Invalid output item')
            if 'data' in output and not isinstance(output['data'], dict):
                raise ValueError('Invalid output data')
            if 'traceback' in output and not (isinstance(output['traceback'], list) and all(isinstance(line, str) for line in output['traceback'])):
                raise ValueError('Invalid traceback')


class NotebookValidation(BaseModel):
    notebook: dict


@router.post('/validate')
def validate_notebook(payload: NotebookValidation):
    try:
        _validate_notebook(payload.notebook)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    return {'valid': True}


@router.get('/library')
def notebook_library(lang: str = 'ru'):
    items = []
    if NOTEBOOKS_DIR.exists():
        for path in sorted(NOTEBOOKS_DIR.rglob('*.ipynb')):
            if '.ipynb_checkpoints' in path.parts:
                continue
            try:
                notebook, _ = _read_notebook(_notebook_path(str(path.relative_to(NOTEBOOKS_DIR))))
                info = notebook.get('metadata', {}).get('studio', {})
                if info.get('language') and info['language'] != ('ru' if lang.startswith('ru') else 'en'):
                    continue
                items.append({'path': str(path.relative_to(NOTEBOOKS_DIR)), 'title': info.get('title', path.stem),
                              'description': info.get('description', ''), 'topic': info.get('topic', 'custom'),
                              'minutes': info.get('minutes'), 'level': info.get('level', ''),
                              'track': info.get('track', 'foundations' if info.get('lesson') else 'custom'),
                              'paper': info.get('paper') if isinstance(info.get('paper'), dict) else None,
                              'lesson': bool(info.get('lesson')), 'cells': len(notebook['cells'])})
            except (HTTPException, OSError, ValueError):
                continue
    items.sort(key=lambda item: (not item['lesson'], item['path']))
    return {'notebooks': items, 'runtime': {'available': notebook_runtime.available(),
            'python': sys.version.split()[0], 'executable': sys.executable,
            'install': 'python -m pip install jupyter-client ipykernel matplotlib'}}


@router.get('/document')
def notebook_document(path: str):
    notebook, revision = _read_notebook(_notebook_path(path))
    return {'notebook': notebook, 'revision': revision, 'path': path}


@router.put('/document')
def save_notebook(payload: NotebookWrite):
    path = _notebook_path(payload.path)
    try:
        _validate_notebook(payload.notebook)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc))
    if len(json.dumps(payload.notebook).encode()) > 10_000_000:
        raise HTTPException(413, 'Notebook exceeds the 10 MB editor limit')
    with _document_lock:
        if path.exists():
            _, revision = _read_notebook(path)
            if payload.revision != revision:
                raise HTTPException(409, 'Notebook changed on disk. Reopen it or save a copy.')
        elif payload.revision:
            raise HTTPException(409, 'Notebook was removed. Save a copy instead.')
        atomic_json(path, payload.notebook)
        _, revision = _read_notebook(path)
    return {'revision': revision, 'path': payload.path}


@router.post('/sessions')
def start_notebook_session(payload: NotebookSession):
    _read_notebook(_notebook_path(payload.path))
    return notebook_runtime.start(PROJECT_ROOT, payload.path)


@router.get('/sessions/{session_id}')
def notebook_session_status(session_id: str):
    return notebook_runtime.get(session_id).snapshot()


@router.post('/sessions/{session_id}/execute')
def execute_notebook_cell(session_id: str, payload: CellExecution):
    return notebook_runtime.get(session_id).execute(payload.code, payload.cell_id)


@router.post('/sessions/{session_id}/interrupt')
def interrupt_notebook_cell(session_id: str):
    return notebook_runtime.get(session_id).interrupt()


@router.delete('/sessions/{session_id}')
def close_notebook_session(session_id: str):
    notebook_runtime.stop(session_id)
    return {'success': True}
