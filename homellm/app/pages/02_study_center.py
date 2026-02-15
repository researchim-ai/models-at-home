"""Study Center page for browsing State-of-AI markdown resources."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

# Internationalization (i18n)
try:
    from homellm.i18n import t
except ImportError:
    # Fallback for direct run
    def t(key, **kwargs):
        return key

REPO_OWNER = "researchim-ai"
REPO_NAME = "state-of-ai"
REPO_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}"
TREE_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/git/trees"
BRANCH_CANDIDATES = ("main", "master")


def _apply_page_typography() -> None:
    st.markdown(
        """
        <style>
        /* Larger readable typography for study_center page */
        .block-container .stMarkdown p,
        .block-container .stMarkdown li {
            font-size: 1.08rem;
            line-height: 1.7;
        }
        .block-container .stMarkdown h1 { font-size: 2.0rem; }
        .block-container .stMarkdown h2 { font-size: 1.6rem; }
        .block-container .stMarkdown h3 { font-size: 1.35rem; }
        .block-container .stMarkdown h4 { font-size: 1.15rem; }
        .block-container .stCaptionContainer p {
            font-size: 0.98rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parent


PROJECT_ROOT = _find_project_root(Path(__file__).parent)
LOCAL_STATE_OF_AI_DIR = PROJECT_ROOT / "state-of-ai"


def _http_get_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": "models-at-home-study-center"})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "models-at-home-study-center"})
    with urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8")


def _display_title_from_path(path: str) -> str:
    name = Path(path).stem.replace("_", " ").replace("-", " ").strip()
    return re.sub(r"\s+", " ", name)


def _sort_docs(paths: List[str]) -> List[str]:
    return sorted(paths, key=lambda p: (0 if Path(p).name.lower() == "readme.md" else 1, p.lower()))


@st.cache_data(ttl=900, show_spinner=False)
def load_remote_markdown_index() -> Tuple[str, List[Dict[str, str]]]:
    errors: List[str] = []
    for branch in BRANCH_CANDIDATES:
        try:
            tree_url = f"{TREE_API_URL}/{branch}?recursive=1"
            payload = _http_get_json(tree_url)
            tree = payload.get("tree", [])
            md_paths: List[str] = []
            for node in tree:
                if node.get("type") != "blob":
                    continue
                path = str(node.get("path", ""))
                if path.lower().endswith(".md"):
                    md_paths.append(path)
            if md_paths:
                docs = [
                    {
                        "path": p,
                        "title": _display_title_from_path(p),
                        "raw_url": f"{RAW_BASE_URL}/{branch}/{p}",
                    }
                    for p in _sort_docs(md_paths)
                ]
                return branch, docs
            errors.append(f"{branch}: markdown files not found")
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{branch}: {exc}")
    raise RuntimeError(" | ".join(errors))


@st.cache_data(ttl=900, show_spinner=False)
def load_remote_markdown(raw_url: str) -> str:
    return _http_get_text(raw_url)


def load_local_markdown_index() -> List[Dict[str, str]]:
    if not LOCAL_STATE_OF_AI_DIR.exists():
        return []
    md_paths = [str(p.relative_to(LOCAL_STATE_OF_AI_DIR)) for p in LOCAL_STATE_OF_AI_DIR.rglob("*.md")]
    docs = []
    for rel_path in _sort_docs(md_paths):
        docs.append({"path": rel_path, "title": _display_title_from_path(rel_path)})
    return docs


def load_local_markdown(rel_path: str) -> str:
    target = LOCAL_STATE_OF_AI_DIR / rel_path
    return target.read_text(encoding="utf-8")


def _init_page_state(default_doc_id: str) -> None:
    if "study_selected_doc" not in st.session_state:
        st.session_state.study_selected_doc = default_doc_id


def _render_sidebar_docs(source_label: str, docs: List[Dict[str, str]]) -> str:
    st.sidebar.header(f"🎓 {t('study_center.title')}")
    st.sidebar.caption(t("study_center.sidebar.source", source=source_label))
    st.sidebar.markdown(f"### {t('study_center.sidebar.section_state_of_ai')}")
    st.sidebar.caption(t("study_center.sidebar.documents_count", count=len(docs)))

    options = [doc["path"] for doc in docs]
    labels = {doc["path"]: doc["title"] for doc in docs}

    selected = st.sidebar.radio(
        t("study_center.sidebar.materials"),
        options=options,
        format_func=lambda p: labels[p],
        key="study_selected_doc",
    )
    return selected


def main() -> None:
    st.set_page_config(page_title=t("study_center.title"), page_icon="🎓", layout="wide")
    _apply_page_typography()

    st.title(f"🎓 {t('study_center.title')}")
    st.caption(t("study_center.subtitle"))
    st.markdown(f"[{t('study_center.repo_link_label')}]({REPO_URL})")

    refresh = st.sidebar.button(f"🔄 {t('study_center.refresh_button')}", use_container_width=True)
    if refresh:
        load_remote_markdown_index.clear()
        load_remote_markdown.clear()
        st.sidebar.success(t("study_center.cache_refreshed"))

    docs: List[Dict[str, str]]
    source = t("study_center.source.github")
    branch = "main"
    remote_error = None

    with st.spinner(t("study_center.loading")):
        try:
            branch, docs = load_remote_markdown_index()
        except Exception as exc:  # noqa: BLE001 - user-facing fallback logic
            remote_error = str(exc)
            docs = load_local_markdown_index()
            source = t("study_center.source.local_copy")

    if not docs:
        st.error(t("study_center.error.no_docs"))
        if remote_error:
            st.caption(t("study_center.error.github_details", error=remote_error))
        return

    _init_page_state(docs[0]["path"])
    source_label = source if source != t("study_center.source.github") else f"{t('study_center.source.github')} ({branch})"
    selected_path = _render_sidebar_docs(source_label, docs)
    current_doc = next(doc for doc in docs if doc["path"] == selected_path)

    if source == t("study_center.source.local_copy"):
        st.warning(t("study_center.warning.local_fallback"))
        if remote_error:
            with st.expander(t("study_center.github_error_details_title")):
                st.code(remote_error)

    try:
        if source == "GitHub":
            content = load_remote_markdown(current_doc["raw_url"])
        else:
            content = load_local_markdown(current_doc["path"])
    except Exception as exc:  # noqa: BLE001 - user-facing message
        st.error(t("study_center.error.open_doc", path=current_doc["path"], error=exc))
        return

    st.subheader(f"📄 {current_doc['title']}")
    st.caption(t("study_center.file_label", path=current_doc["path"]))
    st.markdown("---")
    st.markdown(content)


if __name__ == "__main__":
    main()
