"""Agent Studio page with local llama.cpp inference and training tools."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

try:
    from homellm.app.agent_runtime import run_agent_turn
    from homellm.app.agent_tools import (
        RUNS_DIR,
        get_tool_groups,
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
    from homellm.app.llama_server_backend import LLAMA_SERVER_LOG, LlamaServerBackend, is_llama_server_supported
    from homellm.app.ui_preferences import DEFAULT_THEME, apply_theme_css, init_user_preferences
except ImportError:
    from ..agent_runtime import run_agent_turn
    from ..agent_tools import RUNS_DIR, get_tool_groups, get_training_presets, list_training_capabilities, stop_run
    from ..hf_gguf import (
        DEFAULT_AGENT_GGUF_QUANT,
        DEFAULT_AGENT_GGUF_REPO,
        download_gguf_to_models_dir,
        find_matching_gguf_filename,
        get_agent_quant,
        get_agent_repo_id,
    )
    from ..llama_server_backend import LLAMA_SERVER_LOG, LlamaServerBackend, is_llama_server_supported
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

RUNTIME_MODE_PRESETS = {
    "auto": {"label": "Авто", "gpu_layers": -1, "description": "Пытается использовать GPU/offload максимально эффективно."},
    "cpu": {"label": "CPU only", "gpu_layers": 0, "description": "Полностью CPU-режим. Самый совместимый и переносимый."},
    "balanced": {"label": "GPU баланс", "gpu_layers": 24, "description": "Частичный offload на GPU, если сборка это поддерживает."},
    "max_gpu": {"label": "GPU максимум", "gpu_layers": -1, "description": "Максимальный offload на GPU."},
}

CONTEXT_PRESETS = {
    "32k": 32768,
    "128k": 131072,
    "256k": 262144,
}


def _recommended_output_tokens(n_ctx: int) -> int:
    """Reasonable default response budget based on active context size."""
    if n_ctx <= 32768:
        return 4096
    if n_ctx <= 131072:
        return 8192
    return 16384


def _render_agent_styles() -> None:
    st.markdown(
        """
<style>
div[data-testid="stAppViewBlockContainer"] {
    max-width: 1320px;
    padding-top: 1.25rem;
}

.agent-hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(120, 119, 198, 0.24);
    background:
        radial-gradient(circle at top left, rgba(91, 141, 239, 0.30), transparent 34%),
        radial-gradient(circle at top right, rgba(139, 92, 246, 0.24), transparent 32%),
        linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.93));
    border-radius: 24px;
    padding: 1.6rem 1.6rem 1.25rem 1.6rem;
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.28);
    margin-bottom: 1rem;
}

.agent-hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #f8fafc !important;
}

.agent-hero p {
    margin: 0;
    max-width: 900px;
    color: rgba(226, 232, 240, 0.9) !important;
    font-size: 1rem;
    line-height: 1.55;
}

.agent-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-top: 1rem;
}

.agent-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.10);
    color: #e2e8f0 !important;
    font-size: 0.86rem;
    backdrop-filter: blur(8px);
}

.agent-card-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 1rem 0 1.2rem 0;
}

.agent-card {
    border-radius: 18px;
    padding: 1rem 1rem 0.9rem 1rem;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.80), rgba(15, 23, 42, 0.68));
    box-shadow: 0 8px 28px rgba(2, 6, 23, 0.18);
}

.agent-card-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(148, 163, 184, 0.9) !important;
    margin-bottom: 0.4rem;
}

.agent-card-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc !important;
}

.agent-card-sub {
    margin-top: 0.35rem;
    font-size: 0.84rem;
    color: rgba(203, 213, 225, 0.86) !important;
}

.agent-section-title {
    font-size: 1.08rem;
    font-weight: 700;
    margin: 0.2rem 0 0.8rem 0;
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(15, 23, 42, 0.35);
    padding: 0.4rem 0.6rem;
    margin-bottom: 0.35rem;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button,
[data-testid="stSidebar"] button[kind="primary"] {
    border-radius: 14px !important;
    min-height: 2.8rem;
}

[data-testid="stSidebar"] [data-testid="stNumberInputContainer"],
[data-testid="stSidebar"] .stTextInput > div,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    border-radius: 14px !important;
}

div[data-testid="stTabs"] button {
    border-radius: 12px 12px 0 0 !important;
    font-weight: 600 !important;
}

div[data-testid="stExpander"] {
    border-radius: 18px !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    overflow: hidden;
}

div[data-testid="stChatMessage"] {
    border-radius: 20px;
    border: 1px solid rgba(148, 163, 184, 0.12);
    background: rgba(15, 23, 42, 0.34);
    box-shadow: 0 10px 30px rgba(2, 6, 23, 0.10);
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}

div[data-testid="stChatInput"] {
    border-top: 0 !important;
}

div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] input {
    border-radius: 16px !important;
}

div[data-testid="metric-container"] {
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
}

.agent-empty-state {
    border-radius: 22px;
    padding: 1.15rem 1.2rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 30%),
        linear-gradient(180deg, rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.60));
    box-shadow: 0 16px 42px rgba(2, 6, 23, 0.14);
}

.agent-empty-state h3 {
    margin: 0 0 0.45rem 0;
    color: #f8fafc !important;
    font-size: 1.1rem;
}

.agent-empty-state p {
    margin: 0;
    color: rgba(226, 232, 240, 0.86) !important;
    line-height: 1.55;
}

.agent-side-card {
    border-radius: 20px;
    padding: 1rem 1rem 0.9rem 1rem;
    margin-bottom: 0.9rem;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.78), rgba(15, 23, 42, 0.60));
    box-shadow: 0 12px 34px rgba(2, 6, 23, 0.12);
}

.agent-side-card h3 {
    margin: 0 0 0.55rem 0;
    font-size: 1rem;
    color: #f8fafc !important;
}

.agent-side-card p,
.agent-side-card li {
    color: rgba(226, 232, 240, 0.84) !important;
}

.agent-side-card ul {
    margin: 0.35rem 0 0 1rem;
    padding: 0;
}

.agent-form-note {
    margin-top: -0.25rem;
    margin-bottom: 0.8rem;
    color: rgba(148, 163, 184, 0.95) !important;
    font-size: 0.84rem;
}

div[data-testid="stForm"] {
    border-radius: 20px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background: rgba(15, 23, 42, 0.32);
    padding: 0.85rem 0.9rem 0.45rem 0.9rem;
}

div[data-testid="stForm"] textarea {
    min-height: 110px !important;
}

div[data-testid="stFormSubmitButton"] button {
    border-radius: 14px !important;
    min-height: 2.9rem;
    font-weight: 700 !important;
}

@media (max-width: 1000px) {
    .agent-card-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 640px) {
    .agent-card-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_hero(available: bool, installed_backend: str, gguf_models: List[Path], run_count: int) -> None:
    status_label = "Готов к запуску" if available else "Нужна сборка llama.cpp"
    st.markdown(
        f"""
<div class="agent-hero">
  <h1>Agent Studio</h1>
  <p>
    Агент для тренировки и настройки ваших моделей внутри Models at Home.
    Он помогает готовить конфиги, запускать обучение и работать с локальными GGUF-моделями.
  </p>
  <div class="agent-chip-row">
    <span class="agent-chip">Статус: {status_label}</span>
    <span class="agent-chip">Backend: {installed_backend}</span>
    <span class="agent-chip">GGUF моделей: {len(gguf_models)}</span>
    <span class="agent-chip">Agent runs: {run_count}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_overview_cards(
    *,
    selected_model_name: str,
    runtime_mode: str,
    n_ctx: int,
    max_tokens: int,
    installed_backend: str,
    device_summary: str,
) -> None:
    st.markdown(
        f"""
<div class="agent-card-grid">
  <div class="agent-card">
    <div class="agent-card-label">Модель</div>
    <div class="agent-card-value">{selected_model_name}</div>
    <div class="agent-card-sub">Локальная модель для работы агента</div>
  </div>
  <div class="agent-card">
    <div class="agent-card-label">Режим</div>
    <div class="agent-card-value">{RUNTIME_MODE_PRESETS[runtime_mode]["label"]}</div>
    <div class="agent-card-sub">{RUNTIME_MODE_PRESETS[runtime_mode]["description"]}</div>
  </div>
  <div class="agent-card">
    <div class="agent-card-label">Контекст</div>
    <div class="agent-card-value">{n_ctx:,} токенов</div>
    <div class="agent-card-sub">Авто-бюджет ответа: ~{max_tokens:,} токенов</div>
  </div>
  <div class="agent-card">
    <div class="agent-card-label">Инференс backend</div>
    <div class="agent-card-value">{installed_backend.upper()}</div>
    <div class="agent-card-sub">{device_summary}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _read_llama_runtime_summary() -> Dict[str, Any]:
    if not LLAMA_SERVER_LOG.exists():
        return {}

    try:
        lines = LLAMA_SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}

    # The server log is append-only across restarts, so only inspect the most
    # recent model load block to avoid mixing runtime info from older sessions.
    start_markers = [
        idx
        for idx, line in enumerate(lines)
        if "main: loading model" in line or "srv    load_model: loading model" in line
    ]
    if start_markers:
        lines = lines[start_markers[-1] :]

    summary: Dict[str, Any] = {}
    for line in lines:
        if "using device " in line:
            summary["device_line"] = line.strip()
        if "offloaded " in line and "layers to GPU" in line:
            summary["offload_line"] = line.strip()
        if "model buffer size" in line and ("Vulkan" in line or "CUDA" in line or "GPU" in line):
            summary["model_buffer_line"] = line.strip()
        if "KV buffer size" in line:
            summary["kv_buffer_line"] = line.strip()
        if "compute buffer size" in line and "sched_reserve" in line:
            buffers = summary.setdefault("compute_buffer_lines", [])
            buffers.append(line.strip())
        if "n_ctx_slot =" in line and "task.n_tokens =" in line:
            ctx_match = re.search(r"n_ctx_slot = (\d+)", line)
            token_match = re.search(r"task\.n_tokens = (\d+)", line)
            if ctx_match and token_match:
                summary["context_capacity"] = int(ctx_match.group(1))
                summary["context_used"] = int(token_match.group(1))

    return summary


def _runtime_context_ratio(runtime_summary: Dict[str, Any], configured_ctx: int) -> float | None:
    used = runtime_summary.get("context_used")
    capacity = runtime_summary.get("context_capacity") or configured_ctx
    try:
        used_value = int(used)
        capacity_value = int(capacity)
    except Exception:
        return None
    if capacity_value <= 0:
        return None
    return min(1.0, max(0.0, used_value / capacity_value))


def _render_runtime_banner(runtime_summary: Dict[str, Any], configured_ctx: int) -> None:
    if not runtime_summary:
        return

    st.markdown('<div class="agent-section-title">Runtime инференса</div>', unsafe_allow_html=True)
    info_cols = st.columns(3)
    info_cols[0].metric("Устройство", runtime_summary.get("device_line", "не определено").split("using device ", 1)[-1])
    info_cols[1].metric("Offload", runtime_summary.get("offload_line", "нет данных").split("load_tensors: ", 1)[-1])
    info_cols[2].metric("KV cache", runtime_summary.get("kv_buffer_line", "нет данных").split(": ", 1)[-1])

    ratio = _runtime_context_ratio(runtime_summary, configured_ctx)
    context_used = runtime_summary.get("context_used")
    context_capacity = runtime_summary.get("context_capacity") or configured_ctx
    if ratio is not None and context_used is not None:
        st.progress(ratio, text=f"Контекст: {int(context_used):,} / {int(context_capacity):,} токенов ({ratio * 100:.1f}%)")
    else:
        st.caption("Контекст будет отображён после первого реального запроса к агенту.")

    if runtime_summary.get("model_buffer_line"):
        st.caption(runtime_summary["model_buffer_line"])
    compute_lines = runtime_summary.get("compute_buffer_lines") or []
    for line in compute_lines[-2:]:
        st.caption(line)


def _is_pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _format_seconds(seconds: Any) -> str:
    try:
        total = max(0, int(seconds or 0))
    except Exception:
        return "0с"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}ч {minutes}м"
    if minutes:
        return f"{minutes}м {secs}с"
    return f"{secs}с"


def _progress_ratio(current_step: Any, total_steps: Any) -> float | None:
    try:
        cur = float(current_step or 0)
        total = float(total_steps or 0)
        if total <= 0:
            return None
        return min(1.0, max(0.0, cur / total))
    except Exception:
        return None


def _render_active_run_banner(runs: List[Dict[str, Any]]) -> None:
    active_runs = [
        run
        for run in runs
        if run.get("is_running") and str(run.get("status", "")).lower() not in {"error", "completed", "stopped"}
    ]
    if not active_runs:
        return

    st.markdown('<div class="agent-section-title">Активные процессы обучения</div>', unsafe_allow_html=True)
    for run in active_runs[:3]:
        ratio = _progress_ratio(run.get("current_step"), run.get("total_steps"))
        title_cols = st.columns([3, 1, 1])
        with title_cols[0]:
            st.markdown(f"**{run['run_id']}**")
            st.caption(
                f"stage={run.get('stage')} | status={run.get('status')} | backend={run.get('training_backend') or '-'}"
            )
        with title_cols[1]:
            st.metric("Шаг", f"{run.get('current_step', 0)} / {run.get('total_steps', 0)}")
        with title_cols[2]:
            st.metric("ETA", _format_seconds(run.get("eta_seconds")))
        if ratio is not None:
            st.progress(ratio, text=f"Прогресс: {ratio * 100:.1f}%")
        else:
            st.info("Процесс запущен, но общий прогресс пока не рассчитан.")


def _plot_metric_chart(title: str, x_values: List[Any], y_values: List[Any], y_label: str):
    import plotly.graph_objects as go

    if not x_values or not y_values or len(x_values) != len(y_values):
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line={"width": 2},
            marker={"size": 5},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title=y_label,
        height=280,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
    )
    return fig


def _render_run_detail(run: Dict[str, Any]) -> None:
    config = run.get("config") or {}
    metrics = run.get("metrics") or {}
    tabs = st.tabs(["Обзор", "Конфиг", "Логи", "Графики", "Артефакты"])

    with tabs[0]:
        overview_cols = st.columns(4)
        overview_cols[0].metric("Статус", str(run.get("status", "unknown")))
        overview_cols[1].metric("Текущий шаг", str(metrics.get("current_step", 0)))
        overview_cols[2].metric("Loss", f"{float(metrics.get('current_loss', 0.0)):.4f}" if metrics.get("current_loss") is not None else "-")
        overview_cols[3].metric("ETA", _format_seconds(metrics.get("eta_seconds")))
        st.caption(
            f"stage={config.get('stage')} | base_model={config.get('base_model_path')} | output={config.get('output_dir')}"
        )
        ratio = _progress_ratio(metrics.get("current_step"), metrics.get("total_steps"))
        if ratio is not None:
            st.progress(ratio, text=f"{ratio * 100:.1f}%")
        if metrics.get("error"):
            st.error(str(metrics["error"]))

    with tabs[1]:
        st.json(config)

    with tabs[2]:
        log_tabs = st.tabs(["stdout", "stderr"])
        with log_tabs[0]:
            stdout_text = "\n".join(run.get("stdout_tail", []))
            st.code(stdout_text or "(stdout empty)")
        with log_tabs[1]:
            stderr_text = "\n".join(run.get("stderr_tail", []))
            st.code(stderr_text or "(stderr empty)")

    with tabs[3]:
        steps_history = metrics.get("steps_history") or []
        loss_history = metrics.get("loss_history") or []
        lr_history = metrics.get("lr_history") or []
        reward_history = metrics.get("reward_history") or []
        fig1 = _plot_metric_chart("Loss", steps_history, loss_history, "Loss")
        fig2 = _plot_metric_chart("Learning Rate", steps_history, lr_history, "LR")
        fig3 = _plot_metric_chart("Reward", steps_history[: len(reward_history)], reward_history, "Reward")
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        if not any([fig1, fig2, fig3]):
            st.info("Графики появятся, когда run запишет первые метрики.")

    with tabs[4]:
        artifact_tabs = st.tabs(["checkpoints", "sample_outputs", "gpu"])
        with artifact_tabs[0]:
            st.json(metrics.get("checkpoints", []))
        with artifact_tabs[1]:
            st.json(metrics.get("sample_outputs", []))
        with artifact_tabs[2]:
            st.json(metrics.get("gpu_stats", []))


def _render_prompt_and_tools() -> None:
    st.markdown('<div class="agent-section-title">Что умеет агент</div>', unsafe_allow_html=True)
    capabilities = list_training_capabilities()
    capability_cols = st.columns(2)
    text_training = capabilities.get("text_training", {})
    vlm_training = capabilities.get("vlm_training", {})

    with capability_cols[0]:
        st.markdown("### Text training")
        text_stages = text_training.get("stages") or []
        text_supports = text_training.get("supports") or []
        if text_stages:
            st.markdown("Стадии: " + ", ".join(f"`{stage}`" for stage in text_stages))
        if text_supports:
            st.markdown("Поддержка:")
            for item in text_supports:
                st.markdown(f"- {item}")
        if text_training.get("entrypoint"):
            st.caption(f"Entrypoint: `{text_training['entrypoint']}`")

    with capability_cols[1]:
        st.markdown("### VLM training")
        vlm_stages = vlm_training.get("stages") or []
        vlm_entrypoints = vlm_training.get("entrypoints") or []
        if vlm_stages:
            st.markdown("Стадии: " + ", ".join(f"`{stage}`" for stage in vlm_stages))
        if vlm_entrypoints:
            st.markdown("Воркеры:")
            for item in vlm_entrypoints:
                st.markdown(f"- `{item}`")
        supports = vlm_training.get("supports") or []
        if supports:
            st.markdown("Поддержка:")
            for item in supports:
                st.markdown(f"- {item}")

    st.markdown("### Инструменты агента")
    for group in get_tool_groups():
        with st.expander(group["title"], expanded=False):
            for tool in group["tools"]:
                st.markdown(f"**`{tool['name']}`**")
                st.caption(tool["description"])
                if tool.get("arguments"):
                    args_label = ", ".join(f"`{name}`" for name in tool["arguments"].keys())
                    st.caption(f"Аргументы: {args_label}")


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


def _installed_llama_backend() -> str:
    return (os.environ.get("LLAMA_CPP_BACKEND", "vulkan") or "vulkan").strip().lower()


def _detect_available_gpus(installed_backend: str) -> List[Dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return []

    devices: List[Dict[str, Any]] = []
    uuid_to_backend_id: Dict[str, int] = {}
    if str(installed_backend or "").lower() == "vulkan":
        try:
            vk_result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            current_backend_id: int | None = None
            for raw_line in vk_result.stdout.splitlines():
                line = raw_line.strip()
                match_gpu = re.match(r"GPU(\d+):", line)
                if match_gpu:
                    current_backend_id = int(match_gpu.group(1))
                    continue
                if line.startswith("deviceUUID") and current_backend_id is not None:
                    _, value = line.split("=", 1)
                    uuid_to_backend_id[value.strip().lower()] = current_backend_id
        except Exception:
            uuid_to_backend_id = {}

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 4:
            continue
        idx_raw, uuid_raw, name, memory_raw = parts
        try:
            idx = int(idx_raw)
            memory_mb = int(float(memory_raw))
        except ValueError:
            continue
        backend_id = uuid_to_backend_id.get(uuid_raw.lower(), idx)
        devices.append(
            {
                "id": idx,
                "backend_id": backend_id,
                "uuid": uuid_raw,
                "name": name,
                "memory_mb": memory_mb,
                "label": (
                    f"GPU {idx} - {name} ({memory_mb / 1024:.1f} GB)"
                    if backend_id == idx
                    else f"GPU {idx} / Vulkan {backend_id} - {name} ({memory_mb / 1024:.1f} GB)"
                ),
            }
        )
    return devices


def _selected_backend_device_ids(selected_ids: List[int], gpu_inventory: List[Dict[str, Any]]) -> List[int]:
    selected_set = {int(device_id) for device_id in selected_ids}
    backend_ids: List[int] = []
    for device in gpu_inventory:
        if int(device.get("id", -1)) in selected_set:
            backend_ids.append(int(device.get("backend_id", device["id"])))
    return backend_ids


def _device_selection_summary(visible_devices: List[int], gpu_inventory: List[Dict[str, Any]], use_gpu: bool) -> str:
    if not use_gpu:
        return "CPU only"
    if not visible_devices:
        return "Все доступные GPU"

    label_map = {int(device["id"]): str(device["label"]) for device in gpu_inventory}
    if len(visible_devices) == 1:
        return label_map.get(int(visible_devices[0]), f"GPU {visible_devices[0]}")
    return ", ".join(label_map.get(int(device_id), f"GPU {device_id}") for device_id in visible_devices)


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


def _load_backend(
    model_path: Path,
    n_ctx: int,
    n_gpu_layers: int,
    n_batch: int,
    n_threads: int,
    visible_devices: List[int],
    backend_name: str,
) -> None:
    requested_backend = str(backend_name or "cpu").lower()
    fallback_order = [requested_backend]
    if requested_backend == "auto":
        fallback_order.extend(["cuda", "vulkan", "cpu"])
    elif requested_backend == "cuda":
        fallback_order.extend(["vulkan", "cpu"])
    elif requested_backend == "vulkan":
        fallback_order.append("cpu")

    errors: List[str] = []
    backend = None
    effective_backend = None
    runtime_notice = None
    loaded_gpu_layers = 0
    loaded_visible_devices: List[int] = []

    for candidate in list(dict.fromkeys(fallback_order)):
        candidate_gpu_layers = n_gpu_layers if candidate != "cpu" else 0
        candidate_devices = visible_devices if candidate != "cpu" else []
        try:
            backend = LlamaServerBackend(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=candidate_gpu_layers,
                n_batch=n_batch,
                n_threads=n_threads,
                visible_devices=candidate_devices,
                backend_name=candidate,
                verbose=False,
            )
            effective_backend = getattr(backend, "backend_name", candidate)
            if effective_backend != requested_backend:
                runtime_notice = (
                    f"Запрошенный backend `{requested_backend}` недоступен, "
                    f"поэтому агент переключился на `{effective_backend}`."
                )
            loaded_gpu_layers = int(candidate_gpu_layers)
            loaded_visible_devices = [int(device_id) for device_id in candidate_devices]
            break
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    if backend is None or effective_backend is None:
        raise RuntimeError(" ; ".join(errors) or "Не удалось запустить llama-server.")

    st.session_state.agent_backend = backend
    st.session_state.agent_model_path_loaded = str(model_path)
    st.session_state.agent_runtime_backend_effective = effective_backend
    st.session_state.agent_runtime_notice = runtime_notice
    st.session_state.agent_loaded_backend_config = {
        "model_path": str(model_path),
        "n_ctx": int(n_ctx),
        "n_gpu_layers": int(loaded_gpu_layers),
        "visible_devices": loaded_visible_devices,
        "requested_backend": requested_backend,
        "effective_backend": effective_backend,
    }


def _unload_backend() -> None:
    backend = st.session_state.get("agent_backend")
    if backend is not None and hasattr(backend, "stop"):
        try:
            backend.stop()
        except Exception:
            pass
    st.session_state.agent_backend = None
    st.session_state.agent_model_path_loaded = None
    st.session_state.agent_runtime_backend_effective = None
    st.session_state.agent_runtime_notice = None
    st.session_state.agent_loaded_backend_config = None


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
        pid = None
        pid_path = run_dir / "pid"
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
            except Exception:
                pid = None
        runs.append(
            {
                "run_id": run_dir.name,
                "stage": config.get("stage"),
                "status": metrics.get("status", "unknown"),
                "current_step": metrics.get("current_step"),
                "total_steps": metrics.get("total_steps"),
                "eta_seconds": metrics.get("eta_seconds"),
                "training_backend": config.get("training_backend"),
                "output_dir": config.get("output_dir"),
                "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                "stdout_path": _safe_relpath(run_dir / "stdout.log"),
                "stderr_path": _safe_relpath(run_dir / "stderr.log"),
                "stdout_tail": (run_dir / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines()[-160:]
                if (run_dir / "stdout.log").exists()
                else [],
                "stderr_tail": (run_dir / "stderr.log").read_text(encoding="utf-8", errors="replace").splitlines()[-160:]
                if (run_dir / "stderr.log").exists()
                else [],
                "config": config,
                "metrics": metrics,
                "pid": pid,
                "is_running": _is_pid_alive(pid),
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
    if "agent_runtime_mode" not in st.session_state:
        st.session_state.agent_runtime_mode = "max_gpu"
    if "agent_runtime_mode_migrated" not in st.session_state:
        if st.session_state.get("agent_runtime_mode") == "auto":
            st.session_state.agent_runtime_mode = "max_gpu"
        st.session_state.agent_runtime_mode_migrated = True
    if "agent_load_error" not in st.session_state:
        st.session_state.agent_load_error = None
    if "agent_show_debug" not in st.session_state:
        st.session_state.agent_show_debug = False
    if "agent_input_draft" not in st.session_state:
        st.session_state.agent_input_draft = ""
    if "agent_selected_gpus" not in st.session_state:
        st.session_state.agent_selected_gpus = []
    if "agent_runtime_backend_effective" not in st.session_state:
        st.session_state.agent_runtime_backend_effective = None
    if "agent_runtime_notice" not in st.session_state:
        st.session_state.agent_runtime_notice = None
    if "agent_loaded_backend_config" not in st.session_state:
        st.session_state.agent_loaded_backend_config = None


def main() -> None:
    st.set_page_config(page_title="Agent Studio", page_icon="🧠", layout="wide")
    init_user_preferences(USER_PREFS_FILE)
    apply_theme_css(st.session_state.get("ui_theme", DEFAULT_THEME))
    _init_state()
    _render_agent_styles()

    available = is_llama_server_supported()
    runs = _collect_agent_runs()
    gguf_models = _discover_gguf_models()
    preferred = _preferred_model(gguf_models)
    default_repo_id = get_agent_repo_id()
    default_quant = get_agent_quant()
    installed_backend = _installed_llama_backend()
    gpu_inventory = _detect_available_gpus(installed_backend)
    backend_caption = installed_backend
    _render_hero(available=available, installed_backend=backend_caption, gguf_models=gguf_models, run_count=len(runs))

    with st.sidebar:
        st.header("Локальный агент")
        if not available:
            st.error("Текущая платформа не поддерживается для запуска prebuilt llama-server.")
        if not gguf_models:
            st.warning("GGUF-модели не найдены в `models/`.")
        st.caption(f"Сборка `llama.cpp`: `{backend_caption}`")

        runtime_mode = st.radio(
            "Режим запуска",
            options=list(RUNTIME_MODE_PRESETS.keys()),
            index=list(RUNTIME_MODE_PRESETS.keys()).index(st.session_state.get("agent_runtime_mode", "auto")),
            format_func=lambda key: RUNTIME_MODE_PRESETS[key]["label"],
            help="Простой режим для обычных пользователей: выбери CPU или GPU-режим без ручной настройки параметров.",
            key="agent_runtime_mode_radio",
        )
        st.session_state.agent_runtime_mode = runtime_mode
        st.caption(RUNTIME_MODE_PRESETS[runtime_mode]["description"])

        if installed_backend == "cpu" and runtime_mode != "cpu":
            st.info("Текущая сборка образа поддерживает только CPU backend для llama.cpp, поэтому GPU-режим здесь недоступен, даже если выбран в UI.")

        st.markdown("### Контекст")
        preset_cols = st.columns(5)
        for idx, (preset_name, preset_value) in enumerate(CONTEXT_PRESETS.items()):
            if preset_cols[idx].button(preset_name, key=f"ctx_preset_{preset_name}"):
                st.session_state.agent_ctx_size = preset_value
        if "agent_ctx_size" not in st.session_state:
            st.session_state.agent_ctx_size = int(os.environ.get("MAH_AGENT_CTX_SIZE", "262144"))

        model_options = [str(path) for path in gguf_models]
        default_model = str(preferred) if preferred else None
        selected_model = st.selectbox(
            "GGUF модель",
            options=model_options if model_options else ["<не найдена>"],
            index=model_options.index(default_model) if default_model in model_options else 0,
            disabled=not model_options,
        )

        n_ctx = st.number_input(
            "Context",
            min_value=32768,
            max_value=262144,
            value=int(st.session_state.get("agent_ctx_size", 262144)),
            step=1024,
            help="Размер контекста. Для некоторых Qwen GGUF возможен long-context до 256k, но реальный лимит зависит от конкретной модели и памяти.",
        )
        st.session_state.agent_ctx_size = int(n_ctx)
        st.markdown("### GPU устройства")
        inventory_ids = [int(device["id"]) for device in gpu_inventory]
        persisted_ids = [int(device_id) for device_id in st.session_state.get("agent_selected_gpus", [])]
        fallback_last_gpu = [max(inventory_ids)] if inventory_ids else []
        default_gpu_ids = [device_id for device_id in persisted_ids if device_id in inventory_ids] or fallback_last_gpu

        if gpu_inventory:
            if not persisted_ids:
                st.session_state.agent_selected_gpus = default_gpu_ids
            selected_gpu_ids = st.multiselect(
                "Какие GPU использовать для инференса",
                options=inventory_ids,
                default=default_gpu_ids,
                format_func=lambda device_id: next(
                    (
                        str(device["label"])
                        for device in gpu_inventory
                        if int(device["id"]) == int(device_id)
                    ),
                    f"GPU {device_id}",
                ),
                help="По умолчанию выбирается последняя GPU по ID. Можно отметить одну карту, несколько карт или все сразу.",
            )
            st.session_state.agent_selected_gpus = [int(device_id) for device_id in selected_gpu_ids]
            st.caption(
                "Обнаружено GPU: " + ", ".join(str(device["label"]) for device in gpu_inventory)
            )
        else:
            selected_gpu_ids = []
            st.info("GPU в контейнере не обнаружены или `nvidia-smi` недоступен.")

        backend_visible_device_ids = _selected_backend_device_ids(selected_gpu_ids, gpu_inventory)

        backend_supports_gpu = installed_backend in {"auto", "cuda", "vulkan", "hip", "rocm"}
        wants_gpu = backend_supports_gpu and runtime_mode != "cpu"
        use_gpu = wants_gpu
        if installed_backend == "cpu" and gpu_inventory:
            st.info("GPU обнаружены, но текущая сборка `llama.cpp` CPU-only. После GPU-сборки можно будет запускать агент именно на выбранных картах.")
        if wants_gpu and gpu_inventory and not selected_gpu_ids:
            st.warning("GPU-режим выбран, но ни одно устройство не отмечено. Будут использованы все доступные GPU.")

        effective_gpu_layers = int(RUNTIME_MODE_PRESETS[runtime_mode]["gpu_layers"]) if use_gpu else 0

        desired_backend_config = {
            "model_path": str(selected_model) if model_options else None,
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(effective_gpu_layers),
            "visible_devices": [int(device_id) for device_id in (backend_visible_device_ids if effective_gpu_layers != 0 else [])],
            "requested_backend": str(installed_backend),
        }
        loaded_backend_config = st.session_state.get("agent_loaded_backend_config") or {}
        loaded_backend_mismatch = bool(
            st.session_state.get("agent_model_path_loaded")
            and any(
                loaded_backend_config.get(key) != desired_backend_config.get(key)
                for key in ("model_path", "n_ctx", "n_gpu_layers", "visible_devices", "requested_backend")
            )
        )

        st.caption(f"Текущий режим инференса: `{'CPU only' if effective_gpu_layers == 0 else f'GPU/offload ({effective_gpu_layers})'}`")
        st.caption(f"Устройства: `{_device_selection_summary(selected_gpu_ids, gpu_inventory, use_gpu and effective_gpu_layers != 0)}`")

        st.markdown("---")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.2, value=0.2, step=0.05)
        max_tokens = _recommended_output_tokens(int(n_ctx))
        st.caption(f"Ответ агента рассчитывается автоматически: примерно до `{max_tokens}` токенов при текущем контексте.")

        st.markdown("---")
        autoload = st.toggle("Автозапуск при входе", value=os.environ.get("MAH_AGENT_AUTOLOAD", "1") not in {"0", "false", "False"})

        with st.expander("Продвинутые настройки модели", expanded=False):
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

            try:
                resolved_filename = find_matching_gguf_filename(repo_id=repo_id, quant=quant_name)
                st.caption(f"Будет использован файл: `{resolved_filename}`")
            except Exception as exc:
                st.caption(f"Не удалось заранее определить файл: {exc}")

            if st.button("Скачать рекомендуемую GGUF", disabled=not available):
                try:
                    with st.spinner(f"Скачиваю {quant_name} из {repo_id}..."):
                        downloaded = download_gguf_to_models_dir(MODELS_DIR, repo_id=repo_id, quant=quant_name)
                    st.success(f"Скачано: `{_safe_relpath(downloaded)}`")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ошибка скачивания: {exc}")

            n_gpu_layers = st.number_input(
                "GPU layers",
                min_value=-1,
                max_value=999,
                value=effective_gpu_layers,
                step=1,
                disabled=True,
                help="-1 = максимально возможный offload, 0 = CPU only, положительное число = ручное число offloaded слоёв.",
            )
            n_batch = st.number_input("Batch", min_value=64, max_value=4096, value=512, step=64)
            n_threads = st.number_input("Threads", min_value=1, max_value=128, value=max(1, (os.cpu_count() or 8) // 2), step=1)
            max_steps = st.slider("Max agent steps", min_value=1, max_value=10, value=6, step=1)

        with st.expander("Настройки агента", expanded=False):
            st.text_area(
                "Системный prompt",
                key="agent_prompt",
                height=180,
                help="Скрытая системная роль агента. Здесь ее можно настроить, не засоряя основной экран чата.",
            )
            st.toggle(
                "Режим отладки",
                key="agent_show_debug",
                help="Показывает tool trace и служебные детали только когда это действительно нужно.",
            )

        col1, col2 = st.columns(2)
        with col1:
            launch_disabled = not (available and model_options)
            launch_label = "Перезапустить с текущими настройками" if loaded_backend_mismatch else "Запустить"
            if st.button(launch_label, type="primary", disabled=launch_disabled):
                try:
                    with st.spinner("Загружаю модель агента..."):
                        _load_backend(
                            Path(selected_model),
                            int(n_ctx),
                            int(effective_gpu_layers),
                            int(n_batch),
                            int(n_threads),
                            visible_devices=backend_visible_device_ids,
                            backend_name=installed_backend,
                        )
                    st.session_state.agent_load_error = None
                    st.rerun()
                except Exception as exc:
                    st.session_state.agent_backend = None
                    st.session_state.agent_model_path_loaded = None
                    st.session_state.agent_load_error = (
                        "Не удалось загрузить GGUF-модель. "
                        "Проверь, что файл скачался полностью и что текущий backend поддерживает этот режим. "
                        f"Детали: {exc}"
                    )
        with col2:
            if st.button("Выгрузить"):
                _unload_backend()
                st.session_state.agent_load_error = None
                st.rerun()

        if st.button("Новая сессия"):
            _new_session()
            st.rerun()

        loaded_model = st.session_state.get("agent_model_path_loaded")
        if loaded_model:
            st.success(f"Активна модель: `{_safe_relpath(Path(loaded_model))}`")
            effective_backend = st.session_state.get("agent_runtime_backend_effective")
            if effective_backend:
                st.caption(f"Runtime backend: `{effective_backend}`")
            if loaded_backend_mismatch:
                st.warning(
                    "Текущая загруженная модель работает не с теми параметрами, которые сейчас выбраны в UI. "
                    "Чтобы применить `GPU максимум`, контекст или выбор GPU, нажмите `Перезапустить с текущими настройками`."
                )
        else:
            st.info("Модель агента пока не загружена.")
        if st.session_state.get("agent_runtime_notice"):
            st.info(st.session_state.agent_runtime_notice)
        if st.session_state.get("agent_load_error"):
            st.error(st.session_state.agent_load_error)

    _render_overview_cards(
        selected_model_name=Path(selected_model).name if model_options else "GGUF не найдена",
        runtime_mode=runtime_mode,
        n_ctx=int(n_ctx),
        max_tokens=int(max_tokens),
        installed_backend=backend_caption,
        device_summary=_device_selection_summary(selected_gpu_ids, gpu_inventory, use_gpu and effective_gpu_layers != 0),
    )
    _render_runtime_banner(_read_llama_runtime_summary(), int(n_ctx))
    _render_active_run_banner(runs)

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

    if (
        autoload
        and available
        and model_options
        and st.session_state.agent_backend is None
        and not st.session_state.agent_autoload_attempted
    ):
        st.session_state.agent_autoload_attempted = True
        try:
            with st.spinner("Запускаю агента..."):
                _load_backend(
                    Path(selected_model),
                    int(n_ctx),
                    int(effective_gpu_layers),
                    int(n_batch),
                    int(n_threads),
                    visible_devices=backend_visible_device_ids,
                    backend_name=installed_backend,
                )
            st.session_state.agent_load_error = None
            st.rerun()
        except Exception as exc:
            st.session_state.agent_backend = None
            st.session_state.agent_model_path_loaded = None
            st.session_state.agent_load_error = (
                "Автозапуск модели не удался. "
                "Попробуй режим `CPU only`, меньший контекст или перескачай GGUF. "
                f"Детали: {exc}"
            )

    tabs = st.tabs(["💬 Диалог", "🏃 Запуски", "🧰 Возможности", "📚 Архитектура"])

    with tabs[0]:
        chat_col = st.container()

        with chat_col:
            st.markdown('<div class="agent-section-title">Диалог с агентом</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="agent-form-note">Служебный prompt и debug скрыты в боковых настройках, чтобы чат оставался чистым и удобным.</div>',
                unsafe_allow_html=True,
            )

            if not st.session_state.agent_messages:
                st.markdown(
                    """
<div class="agent-empty-state">
  <h3>Нормальный рабочий диалог без инженерного мусора</h3>
  <p>
    Просто опишите цель: подобрать пресет, проверить датасеты, запустить SFT/RL/pretrain,
    разобрать активный run или помочь с текущим экспериментом.
  </p>
</div>
""",
                    unsafe_allow_html=True,
                )

            for idx, message in enumerate(st.session_state.agent_messages):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if st.session_state.get("agent_show_debug") and message["role"] == "assistant":
                        trace = message.get("trace") or []
                        if trace:
                            with st.expander(f"Debug trace #{idx}", expanded=False):
                                st.json(_serialize_trace(trace))

            composer_disabled = not available or st.session_state.agent_backend is None
            with st.form("agent_chat_form", clear_on_submit=True):
                prompt = st.text_area(
                    "Сообщение агенту",
                    key="agent_input_draft",
                    label_visibility="collapsed",
                    placeholder="Например: подбери безопасный SFT-конфиг под мою систему и проверь, какой датасет лучше взять для старта.",
                    disabled=composer_disabled,
                )
                submitted = st.form_submit_button(
                    "Отправить агенту",
                    type="primary",
                    use_container_width=True,
                    disabled=composer_disabled,
                )

            if not available:
                st.info("Для чата нужен поддерживаемый prebuilt `llama-server` для текущей платформы.")
            elif st.session_state.agent_backend is None:
                st.warning("Сначала загрузи GGUF-модель агента через левую панель.")

            if submitted and prompt.strip():
                st.session_state.agent_messages.append({"role": "user", "content": prompt.strip()})

                system_prompt = st.session_state.get("agent_prompt", "").strip()
                conversation: List[Dict[str, str]] = []
                if system_prompt:
                    conversation.append({"role": "system", "content": system_prompt})
                conversation.extend(
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.agent_messages
                    if msg["role"] in {"user", "assistant"}
                )

                with st.spinner("Агент думает..."):
                    answer, trace = run_agent_turn(
                        backend=st.session_state.agent_backend,
                        conversation=conversation,
                        max_steps=max_steps,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        top_k=40,
                    )

                st.session_state.agent_messages.append({"role": "assistant", "content": answer, "trace": trace})
                st.session_state.agent_trace = trace
                _persist_session()
                st.rerun()

    with tabs[1]:
        st.markdown('<div class="agent-section-title">Agent-initiated runs</div>', unsafe_allow_html=True)
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
            with st.expander(f"Открыть run {run['run_id']}", expanded=False):
                _render_run_detail(run)

    with tabs[2]:
        _render_prompt_and_tools()
        st.markdown("### Готовые пресеты")
        st.json(get_training_presets("all"))

    with tabs[3]:
        st.markdown('<div class="agent-section-title">Что уже реализовано</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - Локальный inference через `llama.cpp` и GGUF-модели из `models/`.
            - Поддержка режима `CPU only` прямо из UI.
            - Настройка контекста до `256k` при наличии long-context GGUF и достаточной памяти.
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
