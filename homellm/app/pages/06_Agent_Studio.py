"""Agent Studio page with local llama.cpp inference and training tools."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

try:
    from homellm.app.agent_runtime import run_agent_turn
    from homellm.app.agent_tools import (
        RUNS_DIR,
        get_tool_specs,
        get_training_presets,
        list_training_capabilities,
        stop_run,
    )
    from homellm.app.hf_gguf import (
        DEFAULT_AGENT_GGUF_QUANT,
        DEFAULT_AGENT_GGUF_REPO,
        download_gguf_to_models_dir,
        find_matching_gguf_filename,
        get_agent_quant,
        get_agent_repo_id,
    )
    from homellm.app.llama_cpp_chat import LlamaCppChatBackend, is_llama_cpp_available
    from homellm.app.ui_preferences import DEFAULT_THEME, apply_theme_css, init_user_preferences
except ImportError:
    from ..agent_runtime import run_agent_turn
    from ..agent_tools import RUNS_DIR, get_tool_specs, get_training_presets, list_training_capabilities, stop_run
    from ..hf_gguf import (
        DEFAULT_AGENT_GGUF_QUANT,
        DEFAULT_AGENT_GGUF_REPO,
        download_gguf_to_models_dir,
        find_matching_gguf_filename,
        get_agent_quant,
        get_agent_repo_id,
    )
    from ..llama_cpp_chat import LlamaCppChatBackend, is_llama_cpp_available
    from ..ui_preferences import DEFAULT_THEME, apply_theme_css, init_user_preferences

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
AGENT_SESSIONS_DIR = RUNS_DIR / "agent_sessions"
USER_PREFS_FILE = RUNS_DIR / "ui_preferences.json"
AGENT_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AGENT_PROMPT = """Ты агент Models at Home Studio.
Помогай проектировать, конфигурировать и запускать обучение моделей.
Думай прагматично: сначала проверяй локальные модели, датасеты и статусы run через tools.
Если можешь запустить training безопасно и обоснованно, делай это.
Если информации мало, уточняй ровно недостающие поля."""


def _discover_gguf_models() -> List[Path]:
    if not MODELS_DIR.exists():
        return []
    paths = sorted(MODELS_DIR.rglob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def _preferred_model(models: List[Path]) -> Path | None:
    env_path = os.environ.get("MAH_AGENT_MODEL", "").strip()
    if env_path:
        candidate = Path(env_path)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / env_path).resolve()
        if candidate.exists():
            return candidate

    preferred_tokens = ["qwen", "3.5", "9b", "4k", "xl"]
    for model in models:
        name = model.name.lower().replace("-", "_")
        if all(token in name for token in preferred_tokens):
            return model
    return models[0] if models else None


def _ensure_default_gguf_downloaded(repo_id: str, quant: str) -> Path:
    explicit_filename = os.environ.get("MAH_AGENT_MODEL_FILENAME", "").strip() or None
    return download_gguf_to_models_dir(
        models_dir=MODELS_DIR,
        repo_id=repo_id,
        quant=quant,
        filename=explicit_filename,
    )


def _safe_relpath(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _serialize_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serializable: List[Dict[str, Any]] = []
    for item in trace:
        serializable.append(
            {
                "step": item.get("step"),
                "parsed": item.get("parsed"),
                "tool_results": item.get("tool_results"),
                "raw_response": item.get("raw_response"),
            }
        )
    return serializable


def _persist_session() -> None:
    session_id = st.session_state.agent_session_id
    session_path = AGENT_SESSIONS_DIR / f"{session_id}.json"
    payload = {
        "session_id": session_id,
        "updated_at": datetime.now().isoformat(),
        "model_path": st.session_state.get("agent_model_path_loaded"),
        "messages": st.session_state.get("agent_messages", []),
        "trace": _serialize_trace(st.session_state.get("agent_trace", [])),
        "agent_prompt": st.session_state.get("agent_prompt", DEFAULT_AGENT_PROMPT),
    }
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _new_session() -> None:
    st.session_state.agent_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.agent_messages = []
    st.session_state.agent_trace = []
    _persist_session()


def _load_backend(model_path: Path, n_ctx: int, n_gpu_layers: int, n_batch: int, n_threads: int) -> None:
    backend = LlamaCppChatBackend(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_threads=n_threads,
        verbose=False,
    )
    st.session_state.agent_backend = backend
    st.session_state.agent_model_path_loaded = str(model_path)


def _unload_backend() -> None:
    st.session_state.agent_backend = None
    st.session_state.agent_model_path_loaded = None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"error": str(exc)}


def _collect_agent_runs() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.startswith("agent_"):
            continue
        metrics = _read_json(run_dir / "metrics.json")
        config = _read_json(run_dir / "config.json")
        runs.append(
            {
                "run_id": run_dir.name,
                "stage": config.get("stage"),
                "status": metrics.get("status", "unknown"),
                "current_step": metrics.get("current_step"),
                "total_steps": metrics.get("total_steps"),
                "output_dir": config.get("output_dir"),
                "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                "stdout_path": _safe_relpath(run_dir / "stdout.log"),
                "stderr_path": _safe_relpath(run_dir / "stderr.log"),
            }
        )
    runs.sort(key=lambda item: item["updated_at"], reverse=True)
    return runs


def _init_state() -> None:
    if "agent_session_id" not in st.session_state:
        st.session_state.agent_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "agent_trace" not in st.session_state:
        st.session_state.agent_trace = []
    if "agent_backend" not in st.session_state:
        st.session_state.agent_backend = None
    if "agent_model_path_loaded" not in st.session_state:
        st.session_state.agent_model_path_loaded = None
    if "agent_prompt" not in st.session_state:
        st.session_state.agent_prompt = DEFAULT_AGENT_PROMPT
    if "agent_autoload_attempted" not in st.session_state:
        st.session_state.agent_autoload_attempted = False
    if "agent_download_attempted" not in st.session_state:
        st.session_state.agent_download_attempted = False


def main() -> None:
    st.set_page_config(page_title="Agent Studio", page_icon="🧠", layout="wide")
    init_user_preferences(USER_PREFS_FILE)
    apply_theme_css(st.session_state.get("ui_theme", DEFAULT_THEME))
    _init_state()

    st.title("🧠 Agent Studio")
    st.caption(
        "Локальный агент внутри студии: `llama.cpp` для общения и контролируемые tools для запуска text/VLM training поверх уже существующих воркеров."
    )

    available = is_llama_cpp_available()
    gguf_models = _discover_gguf_models()
    preferred = _preferred_model(gguf_models)
    default_repo_id = get_agent_repo_id()
    default_quant = get_agent_quant()

    with st.sidebar:
        st.header("Локальный агент")
        if not available:
            st.error("`llama-cpp-python` не установлен в контейнере/окружении.")
        if not gguf_models:
            st.warning("GGUF-модели не найдены в `models/`.")

        repo_id = st.text_input(
            "HF repo",
            value=default_repo_id,
            help="Источник GGUF. По умолчанию используется тот же repo, что и в one-click-coding-agent.",
        ).strip()
        quant_name = st.text_input(
            "Quant",
            value=default_quant,
            help="По умолчанию: 9B-UD-Q4_K_XL из `unsloth/Qwen3.5-9B-GGUF`.",
        ).strip()

        resolved_filename = ""
        try:
            resolved_filename = find_matching_gguf_filename(repo_id=repo_id, quant=quant_name)
            st.caption(f"Будет использован файл: `{resolved_filename}`")
        except Exception as exc:
            st.caption(f"Не удалось заранее определить файл: {exc}")

        model_options = [str(path) for path in gguf_models]
        default_model = str(preferred) if preferred else None
        selected_model = st.selectbox(
            "GGUF модель",
            options=model_options if model_options else ["<не найдена>"],
            index=model_options.index(default_model) if default_model in model_options else 0,
            disabled=not model_options,
        )

        n_ctx = st.number_input("Context", min_value=1024, max_value=32768, value=int(os.environ.get("MAH_AGENT_CTX_SIZE", "4096")), step=512)
        n_gpu_layers = st.number_input("GPU layers", min_value=-1, max_value=999, value=int(os.environ.get("MAH_AGENT_GPU_LAYERS", "-1")), step=1)
        n_batch = st.number_input("Batch", min_value=64, max_value=4096, value=512, step=64)
        n_threads = st.number_input("Threads", min_value=1, max_value=128, value=max(1, (os.cpu_count() or 8) // 2), step=1)

        st.markdown("---")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.2, value=0.2, step=0.05)
        top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.95, step=0.05)
        top_k = st.slider("Top-k", min_value=1, max_value=200, value=40, step=1)
        max_tokens = st.slider("Max output tokens", min_value=128, max_value=2048, value=700, step=64)
        max_steps = st.slider("Max agent steps", min_value=1, max_value=10, value=6, step=1)

        st.markdown("---")
        autoload = st.toggle("Автозапуск при входе", value=os.environ.get("MAH_AGENT_AUTOLOAD", "1") not in {"0", "false", "False"})

        if st.button("Скачать рекомендуемую GGUF", disabled=not available):
            try:
                with st.spinner(f"Скачиваю {quant_name} из {repo_id}..."):
                    downloaded = download_gguf_to_models_dir(MODELS_DIR, repo_id=repo_id, quant=quant_name)
                st.success(f"Скачано: `{_safe_relpath(downloaded)}`")
                st.rerun()
            except Exception as exc:
                st.error(f"Ошибка скачивания: {exc}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Запустить", type="primary", disabled=not (available and model_options)):
                with st.spinner("Загружаю llama.cpp модель..."):
                    _load_backend(Path(selected_model), int(n_ctx), int(n_gpu_layers), int(n_batch), int(n_threads))
                st.rerun()
        with col2:
            if st.button("Выгрузить"):
                _unload_backend()
                st.rerun()

        if st.button("Новая сессия"):
            _new_session()
            st.rerun()

        loaded_model = st.session_state.get("agent_model_path_loaded")
        if loaded_model:
            st.success(f"Активна модель: `{_safe_relpath(Path(loaded_model))}`")
        else:
            st.info("Модель агента пока не загружена.")

    if autoload and available and not gguf_models and not st.session_state.agent_download_attempted:
        st.session_state.agent_download_attempted = True
        try:
            with st.spinner(f"GGUF не найдена, скачиваю {default_quant} из {default_repo_id}..."):
                downloaded_model = _ensure_default_gguf_downloaded(default_repo_id, default_quant)
            st.success(f"Автоматически скачана модель: `{_safe_relpath(downloaded_model)}`")
            st.rerun()
        except Exception as exc:
            st.warning(
                "Не удалось автоматически скачать дефолтную GGUF. "
                f"Проверь Hugging Face доступ и repo/quant в sidebar. Детали: {exc}"
            )

    if autoload and available and model_options and st.session_state.agent_backend is None and not st.session_state.agent_autoload_attempted:
        st.session_state.agent_autoload_attempted = True
        with st.spinner("Автозапуск локального агента..."):
            _load_backend(Path(selected_model), int(n_ctx), int(n_gpu_layers), int(n_batch), int(n_threads))
        st.rerun()

    tabs = st.tabs(["💬 Чат", "🏃 Запуски", "🛠️ Tools", "📚 Архитектура"])

    with tabs[0]:
        st.subheader("Системная роль агента")
        st.text_area(
            "Agent prompt",
            key="agent_prompt",
            height=140,
            help="Этот промпт добавляется как системная роль перед пользовательским диалогом.",
        )

        if not available:
            st.info("Чтобы страница заработала, нужно установить `llama-cpp-python` с CUDA-поддержкой в Docker-образ.")
            return

        if st.session_state.agent_backend is None:
            st.warning("Загрузи GGUF-модель агента через sidebar.")
        else:
            for idx, message in enumerate(st.session_state.agent_messages):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant":
                        trace = message.get("trace") or []
                        if trace:
                            with st.expander(f"Tool trace #{idx}", expanded=False):
                                st.json(_serialize_trace(trace))

            prompt = st.chat_input("Опиши задачу: подготовить конфиг, подобрать пресет, запустить train, проверить run...")
            if prompt:
                st.session_state.agent_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                system_prompt = st.session_state.get("agent_prompt", "").strip()
                conversation: List[Dict[str, str]] = []
                if system_prompt:
                    conversation.append({"role": "system", "content": system_prompt})
                conversation.extend(
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.agent_messages
                    if msg["role"] in {"user", "assistant"}
                )

                with st.chat_message("assistant"):
                    with st.spinner("Агент думает, проверяет контекст и при необходимости вызывает tools..."):
                        answer, trace = run_agent_turn(
                            backend=st.session_state.agent_backend,
                            conversation=conversation,
                            max_steps=max_steps,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                        )
                        st.write(answer)
                        with st.expander("Подробный trace", expanded=False):
                            st.json(_serialize_trace(trace))

                st.session_state.agent_messages.append({"role": "assistant", "content": answer, "trace": trace})
                st.session_state.agent_trace = trace
                _persist_session()

    with tabs[1]:
        st.subheader("Agent-initiated runs")
        runs = _collect_agent_runs()
        if not runs:
            st.info("Пока нет запусков, созданных агентом.")
        for run in runs[:25]:
            cols = st.columns([2, 1, 1, 1])
            with cols[0]:
                st.markdown(f"**{run['run_id']}**")
                st.caption(f"stage={run['stage']} | status={run['status']} | output={run['output_dir']}")
            with cols[1]:
                st.caption(f"step: {run['current_step']} / {run['total_steps']}")
            with cols[2]:
                if st.button("Стоп", key=f"stop_{run['run_id']}"):
                    result = stop_run(run["run_id"])
                    if result.get("stopped"):
                        st.success(f"{run['run_id']} остановлен")
                    else:
                        st.error(result.get("reason", "Не удалось остановить"))
                    st.rerun()
            with cols[3]:
                st.caption(run["updated_at"])
            with st.expander(f"Логи {run['run_id']}", expanded=False):
                stdout_path = PROJECT_ROOT / run["stdout_path"]
                stderr_path = PROJECT_ROOT / run["stderr_path"]
                stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
                st.code(stdout_text[-4000:] or "(stdout empty)")
                if stderr_text.strip():
                    st.code(stderr_text[-4000:])

    with tabs[2]:
        st.subheader("Доступные инструменты")
        st.json(get_tool_specs())
        st.markdown("### Готовые пресеты")
        st.json(get_training_presets("all"))

    with tabs[3]:
        st.subheader("Что уже реализовано")
        st.markdown(
            """
            - Локальный inference через `llama.cpp` и GGUF-модели из `models/`.
            - Автозапуск preferred-модели при заходе на страницу.
            - Агентный цикл с JSON tool-calling и ограниченным набором безопасных действий.
            - Запуск text/VLM training через существующие worker-модули проекта.
            - Мониторинг agent-run'ов через `.runs/agent_*`.
            """
        )
        st.markdown("### Training capabilities snapshot")
        st.json(list_training_capabilities())
        st.markdown(
            """
            ### Рекомендуемый путь для Qwen 3.5 9B 4k_xl
            1. По умолчанию страница сама качает `9B-UD-Q4_K_XL` из `unsloth/Qwen3.5-9B-GGUF`.
            2. При необходимости можно переопределить источник через `MAH_AGENT_MODEL_REPO`, `MAH_AGENT_MODEL_QUANT`, `MAH_AGENT_MODEL_FILENAME`.
            3. После скачивания агент автоматически загрузит GGUF при заходе на страницу.
            4. Дальше агент сможет подобрать пресет, проверить датасеты и запустить обучение через имеющиеся воркеры.
            """
        )


if __name__ == "__main__":
    main()
