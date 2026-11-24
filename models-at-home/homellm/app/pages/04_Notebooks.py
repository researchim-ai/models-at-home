"""Streamlit page that links to a dedicated JupyterLab service."""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

try:
    from homellm.app.ui_preferences import DEFAULT_THEME, apply_theme_css, init_user_preferences
except ImportError:
    from ..ui_preferences import DEFAULT_THEME, apply_theme_css, init_user_preferences


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = PROJECT_ROOT / ".runs"
RUNS_DIR.mkdir(exist_ok=True)
USER_PREFS_FILE = RUNS_DIR / "ui_preferences.json"

DEFAULT_PUBLIC_URL = os.getenv("NOTEBOOKS_PUBLIC_URL", "http://localhost:8888")
DEFAULT_INTERNAL_URL = os.getenv("NOTEBOOKS_INTERNAL_URL", "http://notebooks:8888")
DEFAULT_JUPYTER_TOKEN = os.getenv("JUPYTER_TOKEN", "mah-local")


def _join_url(base: str, path: str, token: str) -> str:
    clean_base = base.rstrip("/")
    clean_path = "/" + path.lstrip("/")
    url = f"{clean_base}{clean_path}"
    if token.strip():
        query = urllib.parse.urlencode({"token": token.strip()})
        url = f"{url}?{query}"
    return url


def _healthcheck(internal_base_url: str, token: str) -> tuple[bool, str]:
    status_url = _join_url(internal_base_url, "/api/status", token)
    try:
        request = urllib.request.Request(status_url, method="GET")
        with urllib.request.urlopen(request, timeout=2.0) as response:
            if response.status == 200:
                return True, "JupyterLab service is online."
            return False, f"Unexpected status code: {response.status}"
    except urllib.error.URLError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - UI safety net
        return False, str(exc)


def main() -> None:
    st.set_page_config(page_title="Notebooks", page_icon="📓", layout="wide")
    init_user_preferences(USER_PREFS_FILE)
    apply_theme_css(st.session_state.get("ui_theme", DEFAULT_THEME))

    st.title("📓 Notebook Studio")
    st.caption("Run and tune models in a dedicated JupyterLab workspace.")

    with st.expander("How to use", expanded=True):
        st.markdown(
            "\n".join(
                [
                    "1. Start services with `docker compose up --build`.",
                    "2. Open JupyterLab from this page.",
                    "3. New notebooks auto-import core `homellm` symbols.",
                    "4. Save your experiments in `/app/notebooks` (mapped to `./notebooks`).",
                ]
            )
        )

    col1, col2 = st.columns(2)
    with col1:
        public_url = st.text_input("Public Jupyter URL", value=DEFAULT_PUBLIC_URL)
    with col2:
        token = st.text_input("Jupyter token", value=DEFAULT_JUPYTER_TOKEN, type="password")

    internal_url = st.text_input(
        "Internal healthcheck URL (service-to-service)",
        value=DEFAULT_INTERNAL_URL,
        help="Used by Streamlit container to check Jupyter status.",
    )

    ok, detail = _healthcheck(internal_url, token)
    if ok:
        st.success(detail)
    else:
        st.warning(f"Jupyter status check failed: {detail}")

    lab_url = _join_url(public_url, "/lab", token)
    files_url = _join_url(public_url, "/lab/tree", token)
    default_train_nb_url = _join_url(public_url, "/lab/tree/llama_default_train.ipynb", token)
    module_playground_nb_url = _join_url(public_url, "/lab/tree/llama_module_playground.ipynb", token)
    custom_compare_nb_url = _join_url(public_url, "/lab/tree/custom_block_compare.ipynb", token)
    sft_playground_url = _join_url(public_url, "/lab/tree/sft_playground.ipynb", token)
    inference_playground_url = _join_url(public_url, "/lab/tree/inference_playground.ipynb", token)
    memory_estimator_url = _join_url(public_url, "/lab/tree/memory_estimator_demo.ipynb", token)
    blueprint_from_json_url = _join_url(public_url, "/lab/tree/blueprint_from_json.ipynb", token)
    moe_playground_url = _join_url(public_url, "/lab/tree/moe_playground.ipynb", token)
    optimizers_compare_url = _join_url(public_url, "/lab/tree/optimizers_compare.ipynb", token)
    data_formats_url = _join_url(public_url, "/lab/tree/data_formats_sft_rl.ipynb", token)
    grpo_mini_url = _join_url(public_url, "/lab/tree/grpo_mini_demo.ipynb", token)
    dataset_download_url = _join_url(public_url, "/lab/tree/dataset_download.ipynb", token)

    open_col1, open_col2 = st.columns(2)
    with open_col1:
        st.link_button("Open JupyterLab", lab_url, type="primary")
    with open_col2:
        st.link_button("Open notebook files", files_url)

    st.subheader("Notebook templates")
    r1_1, r1_2, r1_3, r1_4 = st.columns(4)
    with r1_1:
        st.link_button("LLaMA train", default_train_nb_url)
    with r1_2:
        st.link_button("nn.Module playground", module_playground_nb_url)
    with r1_3:
        st.link_button("Custom block compare", custom_compare_nb_url)
    with r1_4:
        st.link_button("SFT playground", sft_playground_url)

    r2_1, r2_2, r2_3, r2_4 = st.columns(4)
    with r2_1:
        st.link_button("Inference", inference_playground_url)
    with r2_2:
        st.link_button("Memory estimator", memory_estimator_url)
    with r2_3:
        st.link_button("Blueprint from JSON", blueprint_from_json_url)
    with r2_4:
        st.link_button("MoE playground", moe_playground_url)

    r3_1, r3_2, r3_3, _ = st.columns(4)
    with r3_1:
        st.link_button("Optimizers compare", optimizers_compare_url)
    with r3_2:
        st.link_button("Data formats (SFT/RL)", data_formats_url)
    with r3_3:
        st.link_button("GRPO mini demo", grpo_mini_url)
    st.caption("Если датасета нет — в ноутбуках с pretrain вызывается ensure_pretrain_dataset(); отдельно: «Загрузка датасетов».")
    r4_1, _, _ = st.columns([1, 2, 1])
    with r4_1:
        st.link_button("📥 Загрузка датасетов", dataset_download_url)

    st.caption(
        "If your browser blocks iframe embedding because of Jupyter CSP, use 'Open JupyterLab' in a new tab."
    )
    embed = st.checkbox("Embed JupyterLab here (experimental)", value=False)
    if embed:
        components.iframe(lab_url, height=980, scrolling=True)

    st.info(
        "Tip: create custom model code inside `notebooks/` and import it directly from notebook cells."
    )


if __name__ == "__main__":
    main()
