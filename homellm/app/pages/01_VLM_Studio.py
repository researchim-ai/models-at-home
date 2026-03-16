"""VLM Studio: Vision-Language Model tuning and inference (same style as LLM page)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

try:
    from homellm.i18n import t, load_translations, get_current_language
except ImportError:
    def t(key, **kwargs):
        return key
    def load_translations():
        pass
    def get_current_language():
        return "en"

try:
    from homellm.app.ui_preferences import (
        DEFAULT_THEME,
        init_user_preferences,
        apply_theme_css,
    )
except ImportError:
    from ..ui_preferences import (
        DEFAULT_THEME,
        init_user_preferences,
        apply_theme_css,
    )

load_translations()

# Paths (same as main app)
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
RUNS_DIR.mkdir(exist_ok=True)
DATASET_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Default VLM presets (HuggingFace)
VLM_HF_PRESETS = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]

# HuggingFace VLM datasets (id, name for UI, description, split, format)
VLM_HF_DATASETS = [
    {
        "id": "HuggingFaceH4/llava-instruct-mix-vsft",
        "name": "LLaVA Instruct Mix (VSFT)",
        "description": "~273k image+conversation, SFT. 11.4 GB.",
        "split": "train",
        "format": "image_messages",
    },
    {
        "id": "liuhaotian/LLaVA-Pretrain",
        "name": "LLaVA-Pretrain (558k)",
        "description": "Image–caption for pretrain (LCS-558K).",
        "split": "train",
        "format": "image_caption",
    },
    {
        "id": "HuggingFaceM4/COCO",
        "name": "COCO (M4)",
        "description": "COCO captions, pretrain/SFT.",
        "split": "train",
        "format": "coco",
    },
    {
        "id": "nielsr/caption-the-image",
        "name": "Caption the Image",
        "description": "Image captioning for VLM.",
        "split": "train",
        "format": "image_caption",
    },
]

VLM_STAGES = [
    ("vlm_pretrain", "vlm.stage.pretrain"),
    ("vlm_sft", "vlm.stage.sft"),
    ("vlm_grpo", "vlm.stage.grpo"),
]


def _load_metrics(run_id: str) -> dict | None:
    """Load metrics.json for a run."""
    path = RUNS_DIR / run_id / "metrics.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_process_running(run_id: str) -> bool:
    pid_path = RUNS_DIR / run_id / "pid"
    if not pid_path.exists():
        return False
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, OSError):
        return False


def _list_vlm_runs() -> list[Path]:
    """List run dirs that are VLM (config has stage vlm_pretrain, vlm_sft, or vlm_grpo)."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for p in RUNS_DIR.iterdir():
        if not p.is_dir():
            continue
        cfg = p / "config.json"
        if cfg.exists():
            try:
                with open(cfg) as f:
                    c = json.load(f)
                stage = c.get("stage", "")
                if stage in ("vlm_pretrain", "vlm_sft", "vlm_grpo"):
                    runs.append(p)
            except Exception:
                pass
    return sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)


def _download_hf_vlm_dataset(hf_id: str, split: str, out_name: str, format_type: str) -> str | None:
    """Download HF dataset and save as JSONL to DATASET_DIR. Returns path or error message."""
    try:
        from datasets import load_dataset
    except ImportError:
        return "datasets library not installed"
    out_path = DATASET_DIR / (out_name or "vlm_data.jsonl")
    try:
        ds = load_dataset(hf_id, split=split, trust_remote_code=True)
    except Exception as e:
        return str(e)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            try:
                if format_type == "image_messages":
                    img = row.get("image")
                    messages = row.get("messages", row.get("conversations", []))
                    if img is None or not messages:
                        continue
                    if hasattr(img, "save"):
                        rel = f"vlm_{hf_id.replace('/', '_')}_{i}.png"
                        img_path = DATASET_DIR / rel
                        img.save(img_path)
                        img_ref = rel
                    else:
                        img_ref = str(img) if img else ""
                    rec = {"image": img_ref, "conversations": messages}
                elif format_type == "image_caption":
                    img = row.get("image")
                    cap = row.get("caption", row.get("caption_gt", row.get("caption_gt_cleaned", row.get("caption_text", ""))))
                    if img is None or not cap:
                        continue
                    if hasattr(img, "save"):
                        rel = f"vlm_{hf_id.replace('/', '_')}_{i}.png"
                        img_path = DATASET_DIR / rel
                        img.save(img_path)
                        img_ref = rel
                    else:
                        img_ref = str(img) if img else ""
                    rec = {"image": img_ref, "caption": cap}
                elif format_type == "coco":
                    img = row.get("image")
                    caps = row.get("caption", row.get("captions", [row.get("caption_gt", "")]))
                    cap = caps[0] if isinstance(caps, list) and caps else (caps or "")
                    if img is None or not cap:
                        continue
                    if hasattr(img, "save"):
                        rel = f"vlm_coco_{i}.png"
                        img_path = DATASET_DIR / rel
                        img.save(img_path)
                        img_ref = rel
                    else:
                        img_ref = str(img) if img else ""
                    rec = {"image": img_ref, "caption": cap}
                else:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            except Exception:
                continue
    return str(out_path) if written > 0 else None


def _get_vlm_datasets() -> list[tuple[str, str]]:
    """List JSONL files in datasets/ (for VLM: image + text)."""
    out = []
    if DATASET_DIR.exists():
        for f in DATASET_DIR.glob("*.jsonl"):
            try:
                size_mb = f.stat().st_size / (1024 * 1024)
                out.append((f.name, f"{size_mb:.1f} MB"))
            except Exception:
                out.append((f.name, "?"))
    return out


def get_available_vlm_models() -> list[dict]:
    """List available VLM: HF presets + local out/ and models/ (like LLM get_available_models)."""
    models = []
    for name in VLM_HF_PRESETS:
        models.append({"name": f"🤗 {name}", "path": name, "type": "hf"})
    # out/*/final_model, out/*/*/final_model
    if OUTPUT_DIR.exists():
        for final in OUTPUT_DIR.rglob("final_model"):
            if final.is_dir() and (final / "config.json").exists():
                try:
                    rel = final.relative_to(OUTPUT_DIR)
                    models.append({
                        "name": f"📁 {rel}",
                        "path": str(final),
                        "type": "local",
                    })
                except ValueError:
                    models.append({"name": f"📁 {final.name}", "path": str(final), "type": "local"})
    # models/ (downloaded HF)
    if MODELS_DIR.exists():
        for d in MODELS_DIR.iterdir():
            if d.is_dir() and (d / "config.json").exists():
                cfg = d / "config.json"
                try:
                    with open(cfg) as f:
                        c = json.load(f)
                    if c.get("model_type") in ("llava", "llava_next", "qwen2_vl") or "vision" in str(c).lower():
                        models.append({"name": f"📂 {d.name}", "path": str(d), "type": "local"})
                except Exception:
                    models.append({"name": f"📂 {d.name}", "path": str(d), "type": "local"})
    return models


def _get_vlm_models() -> list[dict]:
    """Alias for get_available_vlm_models."""
    return get_available_vlm_models()


def _start_vlm_training(config: dict) -> tuple[str, subprocess.Popen]:
    """Start VLM SFT in background (stage must be vlm_sft for worker to accept)."""
    run_id = datetime.now().strftime("vlm_%Y%m%d_%H%M%S")
    experiment_root = Path(config.get("output_dir", "out/vlm_sft"))
    if not str(experiment_root).startswith("out"):
        experiment_root = OUTPUT_DIR / "vlm_sft"
    run_output_dir = PROJECT_ROOT / experiment_root / run_id
    config["output_dir"] = str(run_output_dir)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    with open(metrics_path, "w") as f:
        json.dump({"status": "starting", "current_step": 0}, f)
    cmd = [
        sys.executable, "-m", "homellm.training.vlm_sft",
        "--config", str(config_path),
        "--metrics", str(metrics_path),
    ]
    env = os.environ.copy()
    gpu_ids = config.get("gpu_ids") or []
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    stdout_file = open(stdout_path, "w")
    stderr_file = open(stderr_path, "w")
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=True,
        env=env,
    )
    with open(run_dir / "pid", "w") as f:
        f.write(str(process.pid))
    if "vlm_stdout_file" not in st.session_state:
        st.session_state.vlm_stdout_file = {}
    st.session_state.vlm_stdout_file[run_id] = stdout_file
    st.session_state.vlm_stderr_file = st.session_state.get("vlm_stderr_file") or {}
    st.session_state.vlm_stderr_file[run_id] = stderr_file
    return run_id, process


def _render_quick_summary_vlm(base_model_display: str, data_display: str, stage_display: str, all_ready: bool) -> bool:
    """Quick summary 3 columns (Model / Data / Stage) like LLM page."""
    st.markdown("""
    <style>
    .vlm-summary { background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%); border: 2px solid #444; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
    .vlm-summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
    .vlm-summary-item { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 0.8rem; text-align: center; }
    .vlm-summary-label { color: #888; font-size: 0.85rem; margin-bottom: 0.3rem; }
    .vlm-summary-value { color: #fff; font-size: 1rem; word-break: break-word; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="vlm-summary">
        <div class="vlm-summary-grid">
            <div class="vlm-summary-item">
                <div class="vlm-summary-label">Модель</div>
                <div class="vlm-summary-value">{base_model_display}</div>
                <div>{'✅' if base_model_display != '❌ Не выбрано' else '⚠️'}</div>
            </div>
            <div class="vlm-summary-item">
                <div class="vlm-summary-label">Данные</div>
                <div class="vlm-summary-value">{data_display}</div>
                <div>{'✅' if data_display != '❌ Не выбрано' else '⚠️'}</div>
            </div>
            <div class="vlm-summary-item">
                <div class="vlm-summary-label">Этап</div>
                <div class="vlm-summary-value">{stage_display}</div>
                <div>{'✅' if all_ready else '⚠️'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return all_ready


def _render_model_preview_vlm(config: dict) -> None:
    """Short preview: base model, tuning method, key hyperparams."""
    stage = config.get("stage", "vlm_sft")
    base = config.get("base_model_path") or config.get("model_name_or_path") or "—"
    base_name = Path(base).name if base and "/" in str(base) else base
    st.info(f"**Этап:** {stage} • **База:** `{base_name}` • **Тюнинг:** {config.get('tuning_method', 'lora')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Batch size", config.get("batch_size", 2))
    with c2:
        st.metric("Seq length", config.get("seq_len", 2048))
    with c3:
        st.metric("LR", config.get("learning_rate", 2e-5))


def _download_hf_vlm_model(repo_id: str, save_name: str) -> bool:
    """Download VLM from HuggingFace to MODELS_DIR (like LLM page)."""
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
                revision="main",
                local_dir=str(save_path),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt", "*.gitattributes", ".git*"],
            )
        if not (save_path / "config.json").exists():
            st.error("config.json не найден после загрузки")
            return False
        st.success(f"Модель сохранена: {save_path}")
        return True
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        if save_path.exists():
            import shutil
            shutil.rmtree(save_path, ignore_errors=True)
        return False


def _stop_vlm_training(run_id: str) -> bool:
    pid_path = RUNS_DIR / run_id / "pid"
    if not pid_path.exists():
        return False
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 15)
        return True
    except Exception:
        return False


def main() -> None:
    st.set_page_config(
        page_title=t("vlm.title"),
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_user_preferences(RUNS_DIR / "ui_preferences.json")
    apply_theme_css(st.session_state.get("ui_theme", DEFAULT_THEME))

    st.title(f"🖼️ {t('vlm.title')}")
    st.caption(t("vlm.subtitle"))

    # Sidebar: stage (pretrain / SFT / GRPO) — like LLM page
    stage_options = [(sid, t(label_key)) for sid, label_key in VLM_STAGES]
    stage_choice = st.sidebar.selectbox(
        t("vlm.sidebar.stage"),
        range(len(stage_options)),
        format_func=lambda i: stage_options[i][1],
        key="vlm_stage_sel",
    )
    selected_stage = stage_options[stage_choice][0]

    st.sidebar.header(f"🖼️ {t('vlm.sidebar.base_model')}")
    available_models = get_available_vlm_models()
    if not available_models:
        base_model_path = st.sidebar.text_input("HF ID or path", value=VLM_HF_PRESETS[0], key="vlm_base_model")
    else:
        model_opt = [m["name"] for m in available_models]
        default_sel = 0
        if st.session_state.get("vlm_selected_base_path"):
            for i, m in enumerate(available_models):
                if m["path"] == st.session_state.vlm_selected_base_path:
                    default_sel = i
                    break
        model_idx = st.sidebar.selectbox(
            t("vlm.sidebar.base_model"),
            range(len(model_opt)),
            index=default_sel,
            format_func=lambda i: model_opt[i],
            key="vlm_base_sel",
        )
        base_model_path = available_models[model_idx]["path"] if model_idx is not None else (available_models[0]["path"])
    experiment_name = st.sidebar.text_input(
        t("vlm.sidebar.experiment_name"),
        value="vlm_sft" if selected_stage == "vlm_sft" else selected_stage.replace("vlm_", "vlm_"),
        key="vlm_experiment_name",
    )
    tuning_method = st.sidebar.selectbox(
        t("vlm.sidebar.tuning_method"),
        options=["lora", "qlora", "full"],
        index=0,
        key="vlm_tuning",
    )
    if tuning_method in ("lora", "qlora"):
        st.sidebar.caption("LoRA")
        lora_r = st.sidebar.slider("LoRA r", 8, 128, 32, step=8, key="vlm_lora_r")
        lora_alpha = st.sidebar.slider("LoRA alpha", 8, 256, 32, step=8, key="vlm_lora_alpha")
        lora_dropout = st.sidebar.slider("LoRA dropout", 0.0, 0.2, 0.05, step=0.01, key="vlm_lora_dropout")
    else:
        lora_r = lora_alpha = 32
        lora_dropout = 0.05
    st.sidebar.markdown("---")
    st.sidebar.subheader(t("vlm.sidebar.dataset"))
    datasets = _get_vlm_datasets()
    dataset_options = ["-- " + t("data.select_dataset") + " --"]
    if datasets:
        dataset_options += [f"{name} ({size})" for name, size in datasets]
    sel = st.sidebar.selectbox(t("data.select_dataset"), dataset_options, key="vlm_dataset_sel")
    if sel and not sel.startswith("--"):
        data_path = str(DATASET_DIR / sel.split(" (")[0].strip())
    else:
        data_path = st.sidebar.text_input("Path to JSONL", placeholder="datasets/my_vlm_data.jsonl", key="vlm_data_path") or ""
        if data_path and not os.path.isabs(data_path) and (DATASET_DIR / data_path).exists():
            data_path = str(DATASET_DIR / data_path)
    st.sidebar.markdown("---")
    st.sidebar.subheader(t("sidebar.hyperparams"))
    batch_size = st.sidebar.slider("Batch size", 1, 16, 2, key="vlm_batch")
    gradient_accumulation = st.sidebar.slider("Gradient accumulation", 1, 32, 4, key="vlm_grad_accum")
    learning_rate = st.sidebar.select_slider("Learning rate", options=["1e-5", "2e-5", "5e-5", "1e-4"], value="2e-5", key="vlm_lr")
    num_epochs = st.sidebar.slider("Epochs", 1, 10, 3, key="vlm_epochs")
    seq_len = st.sidebar.slider("Max length", 512, 4096, 2048, step=256, key="vlm_seq_len")
    save_every = st.sidebar.number_input("Save every N steps", 100, 5000, 500, step=100, key="vlm_save_every")
    log_every = st.sidebar.number_input("Log every N steps", 1, 100, 10, key="vlm_log_every")
    warmup_steps = st.sidebar.number_input("Warmup steps", 0, 2000, 100, key="vlm_warmup")
    st.sidebar.caption("GPU")
    num_gpus = st.sidebar.slider("Number of GPUs", 1, 8, 1, key="vlm_num_gpus")
    if st.sidebar.button(t("vlm.preset.2x3090"), key="vlm_preset_2x3090"):
        st.sidebar.info(t("vlm.best_practices.body"))
        st.session_state.vlm_tuning = "qlora"
        st.session_state.vlm_batch = 2
        st.session_state.vlm_seq_len = 2048
        st.session_state.vlm_grad_accum = 8
        st.rerun()

    data_base_dir = str(Path(data_path).parent) if data_path and Path(data_path).exists() else str(DATASET_DIR)
    full_config = {
        "stage": selected_stage,
        "base_model_path": base_model_path or VLM_HF_PRESETS[0],
        "model_name_or_path": base_model_path or VLM_HF_PRESETS[0],
        "output_dir": str(OUTPUT_DIR / (experiment_name or "vlm_sft")),
        "data_path": data_path or "",
        "data_base_dir": data_base_dir,
        "tuning_method": tuning_method,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": float(learning_rate),
        "num_epochs": num_epochs,
        "seq_len": seq_len,
        "save_every": int(save_every),
        "log_every": int(log_every),
        "warmup_steps": int(warmup_steps),
        "num_gpus": num_gpus,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }

    # Tabs (same order as LLM: Launch, Monitoring, Chat, History, Data, Models, Docs)
    tab_launch, tab_monitor, tab_chat, tab_history, tab_data, tab_models, tab_docs = st.tabs([
        f"🚀 {t('vlm.tabs.launch')}",
        f"📊 {t('vlm.tabs.monitoring')}",
        f"💬 {t('vlm.tabs.chat')}",
        f"📜 {t('history.title')}",
        f"💾 {t('vlm.tabs.data')}",
        f"🤖 {t('vlm.tabs.models')}",
        f"📚 {t('vlm.tabs.docs')}",
    ])

    with tab_launch:
        with st.expander(t("vlm.best_practices.title"), expanded=False):
            st.markdown(t("vlm.best_practices.body"))
        base_display = Path(base_model_path).name if base_model_path else "❌ Не выбрано"
        if not base_model_path:
            base_display = "❌ Не выбрано"
        data_display = Path(data_path).name if (data_path and Path(data_path).exists()) else "❌ Не выбрано"
        stage_label = dict(VLM_STAGES).get(selected_stage, selected_stage)
        stage_display = t(stage_label) if stage_label.startswith("vlm.") else stage_label
        only_sft_implemented = selected_stage != "vlm_sft"
        all_ready = (
            bool(base_model_path)
            and bool(data_path and Path(data_path).exists())
            and (not only_sft_implemented)
        )
        _render_quick_summary_vlm(base_display, data_display, stage_display, all_ready)
        st.subheader(f"📐 {t('model.architecture')}")
        _render_model_preview_vlm(full_config)
        st.markdown("---")
        st.subheader("Полный конфиг")
        st.json(full_config)
        if only_sft_implemented:
            st.info(t("vlm.stage.coming_soon"))
        elif not (base_model_path and data_path):
            st.warning("Укажите базовую модель и датасет (JSONL с полями image и conversations/caption).")
        if st.button(t("vlm.launch.button"), type="primary", disabled=not all_ready):
            full_config["stage"] = "vlm_sft"  # worker accepts only vlm_sft
            run_id, proc = _start_vlm_training(full_config)
            st.session_state.vlm_current_run_id = run_id
            st.session_state.vlm_training_active = True
            st.success(f"VLM SFT запущен: {run_id}. Перейдите на вкладку Мониторинг.")
            st.rerun()

    with tab_monitor:
        runs = _list_vlm_runs()
        run_options = [r.name for r in runs[:30]]
        current_run = st.session_state.get("vlm_current_run_id")
        default_idx = run_options.index(current_run) if current_run and current_run in run_options else 0
        selected_run = st.selectbox(
            "Выберите run",
            run_options,
            index=min(default_idx, len(run_options) - 1) if run_options else 0,
            key="vlm_monitor_run",
        )
        if not runs:
            st.info("Нет запусков VLM. Запустите обучение во вкладке «Запуск».")
        elif selected_run:
            if selected_run != st.session_state.get("vlm_current_run_id"):
                st.session_state.vlm_current_run_id = selected_run
            metrics = _load_metrics(selected_run)
            alive = _is_process_running(selected_run)
            status = (metrics or {}).get("status", "running" if alive else "unknown")
            status_emoji = {"training": "🟢", "running": "🟢", "completed": "✅", "error": "❌", "stopped": "⏹️"}.get(status, "⏳")
            st.subheader(f"{status_emoji} {selected_run}")
            col_stop, col_refresh, _ = st.columns([1, 1, 4])
            with col_stop:
                if alive:
                    if st.button("⏹ Остановить", key="vlm_stop_btn"):
                        _stop_vlm_training(selected_run)
                        st.rerun()
            with col_refresh:
                if st.button("🔄 Обновить", key="vlm_refresh_metrics"):
                    st.rerun()
            if metrics:
                step = metrics.get("current_step", 0)
                total = metrics.get("total_steps") or metrics.get("planned_total_steps")
                progress = (step / total * 100) if total and int(total) > 0 else 0
                st.caption(f"Шаг {step}" + (f" / {total}" if total else ""))
                st.progress(min(progress / 100.0, 1.0))
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.metric("Step", step)
                with c2:
                    st.metric("Loss", f"{metrics.get('current_loss', 0):.4f}")
                with c3:
                    st.metric("LR", f"{metrics.get('current_lr', 0):.2e}")
                with c4:
                    eta = metrics.get("eta_seconds", 0)
                    elapsed = metrics.get("elapsed_seconds", 0)
                    st.metric("Время", f"{int(elapsed)//60} мин", delta=f"ETA {int(eta)//60} мин" if eta else None)
                with c5:
                    gpu = metrics.get("gpu_memory_used_mb")
                    st.metric("GPU MB", f"{gpu:.0f}" if gpu is not None else "—")
                if metrics.get("loss_history"):
                    try:
                        import plotly.graph_objects as go
                        steps_h = metrics.get("steps_history", list(range(len(metrics["loss_history"]))))
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=steps_h, y=metrics["loss_history"], mode="lines", name="Loss", line=dict(color="#e94560")))
                        fig.update_layout(title="Loss", xaxis_title="Step", template="plotly_dark", height=280, margin=dict(l=0, r=0, t=30, b=0))
                        st.plotly_chart(fig, use_container_width=True, key=f"vlm_loss_{selected_run}")
                    except Exception:
                        pass
                checkpoints = metrics.get("checkpoints", [])
                if checkpoints:
                    with st.expander("Чекпоинты"):
                        for ckpt in checkpoints:
                            st.caption(f"Step {ckpt.get('step')}: {ckpt.get('path', '')}")
            log_path = RUNS_DIR / selected_run / "stdout.log"
            with st.expander("Лог вывода"):
                if log_path.exists():
                    content = log_path.read_text(encoding="utf-8", errors="replace")
                    st.text(content[-50000:])
                else:
                    st.caption("Лог пока пуст.")

    with tab_chat:
        st.subheader(t("vlm.chat.select_model"))
        models = _get_vlm_models()
        if not models:
            st.info("Нет доступных VLM. Скачайте модель во вкладке «Модели» или обучите во вкладке «Запуск».")
        else:
            model_options = [m["name"] for m in models]
            default_chat_idx = 0
            preselected = st.session_state.get("vlm_selected_chat_model")
            if preselected:
                for i, m in enumerate(models):
                    if m["path"] == preselected:
                        default_chat_idx = i
                        break
            chat_model_idx = st.selectbox(
                "Модель",
                range(len(model_options)),
                index=default_chat_idx,
                format_func=lambda i: model_options[i],
                key="vlm_chat_model",
            )
            chat_model_path = models[chat_model_idx]["path"] if chat_model_idx is not None else models[0]["path"]
            max_new_tokens = st.slider("Max new tokens", 64, 1024, 256, key="vlm_chat_max_tokens")
            temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.1, key="vlm_chat_temp")
            img_file = st.file_uploader(t("vlm.chat.upload_image"), type=["png", "jpg", "jpeg", "webp"], key="vlm_chat_img")
            prompt_text = st.text_input(t("vlm.chat.prompt"), key="vlm_chat_prompt")
            if st.button(t("vlm.chat.generate"), key="vlm_chat_go") and img_file and prompt_text and chat_model_path:
                try:
                    import torch
                    from PIL import Image
                    from transformers import AutoProcessor, AutoModelForVision2Seq
                    cache_key = "vlm_chat_cache"
                    if cache_key not in st.session_state:
                        st.session_state[cache_key] = {}
                    cache = st.session_state[cache_key]
                    if chat_model_path not in cache:
                        with st.spinner("Загрузка модели..."):
                            processor = AutoProcessor.from_pretrained(chat_model_path, trust_remote_code=True)
                            model = AutoModelForVision2Seq.from_pretrained(
                                chat_model_path,
                                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                device_map="auto" if torch.cuda.is_available() else None,
                                trust_remote_code=True,
                            )
                            cache[chat_model_path] = {"processor": processor, "model": model}
                    proc = cache[chat_model_path]["processor"]
                    model = cache[chat_model_path]["model"]
                    img = Image.open(img_file).convert("RGB")
                    inputs = proc(images=img, text=prompt_text, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                    out = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=temperature > 0.01, temperature=temperature if temperature > 0.01 else 1.0)
                    response = proc.decode(out[0], skip_special_tokens=True)
                    if "vlm_chat_messages" not in st.session_state:
                        st.session_state.vlm_chat_messages = []
                    st.session_state.vlm_chat_messages.append({"role": "user", "content": prompt_text, "image": True})
                    st.session_state.vlm_chat_messages.append({"role": "assistant", "content": response})
                    st.write("**Ответ:**")
                    st.write(response)
                except Exception as e:
                    st.error(str(e))
                    import traceback
                    st.code(traceback.format_exc())
            if st.session_state.get("vlm_chat_messages"):
                with st.expander("История сообщений"):
                    for msg in st.session_state.vlm_chat_messages[-10:]:
                        role = msg.get("role", "")
                        st.caption(f"{role}: {msg.get('content', '')[:200]}{'...' if len(msg.get('content', '')) > 200 else ''}")

    with tab_history:
        st.header(f"📜 {t('history.title')}")
        runs_h = _list_vlm_runs()
        if not runs_h:
            st.info("Нет запусков VLM. Запустите обучение во вкладке «Запуск».")
        else:
            for run_dir in runs_h[:30]:
                run_id = run_dir.name
                metrics_h = _load_metrics(run_id)
                if not metrics_h:
                    continue
                status_h = metrics_h.get("status", "unknown")
                status_emoji_h = {"training": "🟢", "running": "🟢", "completed": "✅", "error": "❌", "stopped": "⏹️"}.get(status_h, "⏳")
                config_path = run_dir / "config.json"
                display_name = run_id
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            rc = json.load(f)
                        display_name = f"{run_id} | {Path(rc.get('base_model_path', '')).name or run_id}"
                    except Exception:
                        pass
                with st.expander(f"{status_emoji_h} {display_name}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Steps", metrics_h.get("current_step", 0))
                    with col2:
                        st.metric("Loss", f"{metrics_h.get('current_loss', 0):.4f}")
                    with col3:
                        st.metric("Status", status_h)
                    with col4:
                        st.metric("Duration", metrics_h.get("training_duration", "—"))
                    btn1, btn2, btn3 = st.columns(3)
                    with btn1:
                        if st.button("📊 Мониторинг", key=f"hist_mon_{run_id}"):
                            st.session_state.vlm_current_run_id = run_id
                            st.toast(f"Выбран run: {run_id}. Перейдите на вкладку Мониторинг.", icon="📊")
                            st.rerun()
                    with btn2:
                        try:
                            if config_path.exists():
                                with open(config_path) as f:
                                    rc = json.load(f)
                                out_dir = Path(rc.get("output_dir", ""))
                                final_model = out_dir / "final_model"
                                if not final_model.is_absolute():
                                    final_model = PROJECT_ROOT / final_model
                                if final_model.exists():
                                    if st.button("💬 Чат", key=f"hist_chat_{run_id}"):
                                        st.session_state.vlm_selected_chat_model = str(final_model)
                                        st.toast("Модель выбрана. Перейдите на вкладку Чат.", icon="💬")
                                        st.rerun()
                        except Exception:
                            pass
                    with btn3:
                        pass

    with tab_data:
        st.subheader(t("vlm.data.download_hf"))
        hf_ds_opt = [f"{d['name']} — {d['id']}" for d in VLM_HF_DATASETS]
        hf_sel_idx = st.selectbox(t("vlm.data.hf_dataset"), range(len(hf_ds_opt)), format_func=lambda i: hf_ds_opt[i], key="vlm_hf_ds")
        out_filename = st.text_input("Save as (filename in datasets/)", value="vlm_hf_data.jsonl", key="vlm_hf_out_name")
        if st.button(t("vlm.data.download_button"), key="vlm_hf_download_btn"):
            d = VLM_HF_DATASETS[hf_sel_idx]
            with st.spinner(f"Downloading {d['id']}..."):
                result = _download_hf_vlm_dataset(
                    d["id"], d.get("split", "train"), out_filename or "vlm_data.jsonl", d.get("format", "image_messages")
                )
            if result and Path(result).exists():
                st.success(f"Saved to {result}")
                st.rerun()
            elif result:
                st.error(result)
            else:
                st.warning("No records written (wrong format or empty split).")
        st.markdown("---")
        st.subheader(t("vlm.data.preview"))
        st.caption(t("vlm.data.format_hint"))
        datasets_list = _get_vlm_datasets()
        if not datasets_list:
            st.info(t("data.no_datasets"))
        else:
            chosen = st.selectbox("Датасет", [n for n, _ in datasets_list], key="vlm_data_preview")
            if chosen:
                path = DATASET_DIR / chosen
                if path.exists():
                    with open(path, encoding="utf-8") as f:
                        first_line = f.readline()
                    if first_line.strip():
                        try:
                            rec = json.loads(first_line)
                            img_path = rec.get("image", "")
                            if img_path and not img_path.startswith("http"):
                                full_img = path.parent / img_path
                                if full_img.exists():
                                    from PIL import Image
                                    st.image(Image.open(full_img).convert("RGB"), caption="First image", width=200)
                            if "conversations" in rec:
                                st.json(rec["conversations"])
                            elif "caption" in rec:
                                st.write(rec["caption"])
                        except Exception as e:
                            st.warning(f"Превью не удалось: {e}")

    with tab_models:
        st.header(f"🤖 {t('vlm.models.available')}")
        col_download, col_list = st.columns([1, 2])
        with col_download:
            st.subheader("🤗 Скачать с HuggingFace")
            vlm_presets = {
                "LLaVA 1.5 7B": ("llava-hf/llava-1.5-7b-hf", "llava-1.5-7b"),
                "LLaVA 1.5 13B": ("llava-hf/llava-1.5-13b-hf", "llava-1.5-13b"),
                "Qwen2-VL 2B Instruct": ("Qwen/Qwen2-VL-2B-Instruct", "Qwen2-VL-2B-Instruct"),
                "Qwen2-VL 7B Instruct": ("Qwen/Qwen2-VL-7B-Instruct", "Qwen2-VL-7B-Instruct"),
                "Ввести вручную": (None, None),
            }
            preset_sel = st.selectbox("Пресет", list(vlm_presets.keys()), key="vlm_dl_preset")
            repo_id = st.text_input("Repo ID", value=vlm_presets[preset_sel][0] or "", key="vlm_dl_repo")
            save_name = st.text_input("Имя папки в models/", value=vlm_presets[preset_sel][1] or "", key="vlm_dl_name")
            if st.button("Скачать модель", key="vlm_dl_btn") and repo_id and save_name:
                if _download_hf_vlm_model(repo_id.strip(), save_name.strip()):
                    st.rerun()
        with col_list:
            st.subheader(t("vlm.models.available"))
            models_list = get_available_vlm_models()
            if not models_list:
                st.info("Нет локальных VLM. Скачайте модель слева или обучите во вкладке «Запуск».")
            else:
                for i, m in enumerate(models_list):
                    with st.expander(f"{m['name']}"):
                        st.caption(f"📂 {m['path']}")
                        if st.button("🚀 Использовать", key=f"use_vlm_{i}"):
                            st.session_state.vlm_selected_base_path = m["path"]
                            st.toast(f"Модель выбрана: {m['name']}. Перейдите в сайдбар.", icon="✅")
                            st.rerun()

    with tab_docs:
        st.caption(t("vlm.docs.source"))
        lang = get_current_language()
        if lang not in ("ru", "en"):
            lang = "en"
        vlm_md = STUDY_MATERIALS / lang / "VLM.md"
        if vlm_md.exists():
            st.markdown(vlm_md.read_text(encoding="utf-8"))
        else:
            fallback = STUDY_MATERIALS / "en" / "VLM.md"
            if fallback.exists():
                st.markdown(fallback.read_text(encoding="utf-8"))
            else:
                st.info("Файл study_materials/.../VLM.md не найден.")


if __name__ == "__main__":
    main()
