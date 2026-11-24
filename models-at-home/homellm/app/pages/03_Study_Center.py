"""Study Center page for browsing State-of-AI markdown resources."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

# Internationalization (i18n)
try:
    from homellm.i18n import t, get_current_language
except ImportError:
    # Fallback for direct run
    def t(key, **kwargs):
        return key

    def get_current_language() -> str:
        return "en"
try:
    from homellm.app.ui_preferences import DEFAULT_THEME, init_user_preferences, apply_theme_css
except ImportError:
    from ..ui_preferences import DEFAULT_THEME, init_user_preferences, apply_theme_css

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
# Внутренние учебные материалы (свои MD в репозитории)
STUDY_MATERIALS_DIR = PROJECT_ROOT / "study_materials"
# Внешний репозиторий state-of-ai (клонируется отдельно или грузится с GitHub)
LOCAL_STATE_OF_AI_DIR = PROJECT_ROOT / "state-of-ai"
RUNS_DIR = PROJECT_ROOT / ".runs"
RUNS_DIR.mkdir(exist_ok=True)
USER_PREFS_FILE = RUNS_DIR / "ui_preferences.json"


def _http_get_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": "models-at-home-study-center"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "models-at-home-study-center"})
    with urlopen(req, timeout=10) as resp:
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


def _internal_materials_dir_for_lang(lang: str) -> Path:
    """Папка с учебными материалами для языка: study_materials/{lang}/."""
    return STUDY_MATERIALS_DIR / lang


def load_internal_markdown_index(lang: Optional[str] = None) -> List[Dict[str, str]]:
    """Внутренние учебные материалы из study_materials/{lang}/. Язык — из i18n, если не передан."""
    if lang is None:
        lang = get_current_language()
    root = _internal_materials_dir_for_lang(lang)
    if not root.exists():
        # Fallback: другой поддерживаемый язык
        other = "ru" if lang == "en" else "en"
        root = _internal_materials_dir_for_lang(other)
    if not root.exists():
        return []
    md_paths = [str(p.relative_to(root)) for p in root.rglob("*.md")]
    docs = []
    for rel_path in _sort_docs(md_paths):
        docs.append({"path": rel_path, "title": _display_title_from_path(rel_path)})
    return docs


def load_internal_markdown(rel_path: str, lang: Optional[str] = None) -> str:
    """Читает markdown из study_materials/{lang}/{rel_path}."""
    if lang is None:
        lang = get_current_language()
    root = _internal_materials_dir_for_lang(lang)
    if not root.exists():
        root = _internal_materials_dir_for_lang("ru" if lang == "en" else "en")
    target = root / rel_path
    return target.read_text(encoding="utf-8")


def rewrite_internal_md_links(content: str, internal_doc_paths: List[str]) -> str:
    """Заменяет ссылки на внутренние .md файлы на query-параметры, чтобы по клику открывался нужный документ в Study Center."""
    if not internal_doc_paths:
        return content
    # Сортируем по длине (убывание), чтобы сначала матчить более длинные пути (на случай подстрок)
    for path in sorted(internal_doc_paths, key=len, reverse=True):
        escaped = re.escape(path)
        # Матчим ](path) или ](path#anchor)
        pattern = rf"\]\({escaped}(#.*?)?\)"
        replacement = rf"](?internal_doc={path}\1)"
        content = re.sub(pattern, replacement, content)
    return content


def _apply_query_params(
    internal_docs: List[Dict[str, str]],
    state_of_ai_docs: List[Dict[str, str]],
    internal_section_label: str,
    state_of_ai_section_label: str,
) -> None:
    """Читает query-параметры (?internal_doc=LLM.md или ?state_of_ai_doc=...) и выставляет выбор документа в session_state."""
    try:
        q = st.query_params
    except Exception:
        return
    internal_paths = {d["path"] for d in internal_docs}
    state_of_ai_paths = {d["path"] for d in state_of_ai_docs}
    if "internal_doc" in q:
        doc = q["internal_doc"]
        if doc in internal_paths:
            st.session_state.study_section = internal_section_label
            st.session_state.study_selected_doc_internal = doc
    if "state_of_ai_doc" in q:
        doc = q["state_of_ai_doc"]
        if doc in state_of_ai_paths:
            st.session_state.study_section = state_of_ai_section_label
            st.session_state.study_selected_doc_state_of_ai = doc


def _init_page_state(
    default_section: str,
    internal_docs: List[Dict[str, str]],
    state_of_ai_docs: List[Dict[str, str]],
) -> None:
    if "study_section" not in st.session_state:
        st.session_state.study_section = default_section
    if "study_selected_doc_internal" not in st.session_state:
        st.session_state.study_selected_doc_internal = internal_docs[0]["path"] if internal_docs else ""
    if "study_selected_doc_state_of_ai" not in st.session_state:
        st.session_state.study_selected_doc_state_of_ai = state_of_ai_docs[0]["path"] if state_of_ai_docs else ""


def _render_sidebar(
    internal_docs: List[Dict[str, str]],
    state_of_ai_docs: List[Dict[str, str]],
    state_of_ai_source_label: str,
) -> Tuple[str, str]:
    """Возвращает (section, selected_path)."""
    st.sidebar.header(f"🎓 {t('study_center.title')}")

    section_options = [t("study_center.section.internal"), t("study_center.section.state_of_ai")]
    section = st.sidebar.radio(
        t("study_center.sidebar.choose_section"),
        options=section_options,
        key="study_section",
    )

    if section == t("study_center.section.internal"):
        st.sidebar.markdown(f"### {t('study_center.sidebar.section_internal')}")
        st.sidebar.caption(t("study_center.sidebar.documents_count", count=len(internal_docs)))
        if not internal_docs:
            st.sidebar.info(t("study_center.sidebar.no_internal_docs"))
            return section, ""
        options = [doc["path"] for doc in internal_docs]
        labels = {doc["path"]: doc["title"] for doc in internal_docs}
        selected = st.sidebar.radio(
            t("study_center.sidebar.materials"),
            options=options,
            format_func=lambda p: labels[p],
            key="study_selected_doc_internal",
        )
        return section, selected
    else:
        st.sidebar.caption(t("study_center.sidebar.source", source=state_of_ai_source_label))
        st.sidebar.markdown(f"### {t('study_center.sidebar.section_state_of_ai')}")
        st.sidebar.caption(t("study_center.sidebar.documents_count", count=len(state_of_ai_docs)))
        if not state_of_ai_docs:
            st.sidebar.warning(t("study_center.sidebar.no_state_of_ai_docs"))
            return section, ""
        options = [doc["path"] for doc in state_of_ai_docs]
        labels = {doc["path"]: doc["title"] for doc in state_of_ai_docs}
        selected = st.sidebar.radio(
            t("study_center.sidebar.materials"),
            options=options,
            format_func=lambda p: labels[p],
            key="study_selected_doc_state_of_ai",
        )
        return section, selected


def main() -> None:
    st.set_page_config(page_title=t("study_center.title"), page_icon="🎓", layout="wide")
    init_user_preferences(USER_PREFS_FILE)
    apply_theme_css(st.session_state.get("ui_theme", DEFAULT_THEME))
    _apply_page_typography()

    st.title(f"🎓 {t('study_center.title')}")
    st.caption(t("study_center.subtitle"))
    st.markdown(f"[{t('study_center.repo_link_label')}]({REPO_URL})")

    # Внутренние материалы — из study_materials/{lang}/ по текущему языку интерфейса
    current_lang = get_current_language()
    internal_docs = load_internal_markdown_index(current_lang)

    # state-of-ai — загрузка только по кнопке, одна попытка
    if "study_state_of_ai_remote" not in st.session_state:
        st.session_state.study_state_of_ai_remote = None  # (branch, docs) or None
    if "study_state_of_ai_remote_error" not in st.session_state:
        st.session_state.study_state_of_ai_remote_error = None

    # Берём данные из session_state (уже загружены по кнопке) или локальную копию; без автоматического запроса
    state_of_ai_docs: List[Dict[str, str]] = []
    state_of_ai_source = t("study_center.source.github")
    branch = "main"
    remote_error = st.session_state.study_state_of_ai_remote_error

    if st.session_state.study_state_of_ai_remote is not None:
        branch, state_of_ai_docs = st.session_state.study_state_of_ai_remote
    else:
        state_of_ai_docs = load_local_markdown_index()
        if state_of_ai_docs:
            state_of_ai_source = t("study_center.source.local_copy")

    state_of_ai_source_label = (
        state_of_ai_source if state_of_ai_source != t("study_center.source.github")
        else f"{t('study_center.source.github')} ({branch})"
    )

    internal_section_label = t("study_center.section.internal")
    state_of_ai_section_label = t("study_center.section.state_of_ai")
    _apply_query_params(
        internal_docs,
        state_of_ai_docs,
        internal_section_label,
        state_of_ai_section_label,
    )

    default_section = internal_section_label if internal_docs else state_of_ai_section_label
    _init_page_state(default_section, internal_docs, state_of_ai_docs)

    # Кнопка «Загрузить с GitHub» только когда выбран раздел state-of-ai
    if st.session_state.study_section == state_of_ai_section_label:
        load_clicked = st.sidebar.button(f"📥 {t('study_center.load_button')}", use_container_width=True)
        if load_clicked:
            st.session_state.study_state_of_ai_remote_error = None
            with st.spinner(t("study_center.loading")):
                try:
                    branch, docs = load_remote_markdown_index()
                    st.session_state.study_state_of_ai_remote = (branch, docs)
                except Exception as exc:  # noqa: BLE001 - user-facing fallback logic
                    st.session_state.study_state_of_ai_remote_error = str(exc)

    if remote_error:
        st.sidebar.error(t("study_center.error.load_failed"))

    if not internal_docs and not state_of_ai_docs:
        st.error(t("study_center.error.no_docs"))
        if remote_error:
            st.caption(t("study_center.error.github_details", error=remote_error))
        st.info(t("study_center.sidebar.no_internal_docs"))
        return

    section, selected_path = _render_sidebar(
        internal_docs, state_of_ai_docs, state_of_ai_source_label
    )

    if not selected_path:
        if section == t("study_center.section.internal"):
            st.info(t("study_center.sidebar.no_internal_docs"))
        else:
            st.warning(t("study_center.sidebar.no_state_of_ai_docs"))
        return

    if section == t("study_center.section.internal"):
        current_doc = next(doc for doc in internal_docs if doc["path"] == selected_path)
        try:
            content = load_internal_markdown(current_doc["path"], current_lang)
            content = rewrite_internal_md_links(
                content, [d["path"] for d in internal_docs]
            )
        except Exception as exc:  # noqa: BLE001
            st.error(t("study_center.error.open_doc", path=current_doc["path"], error=exc))
            return
    else:
        current_doc = next(doc for doc in state_of_ai_docs if doc["path"] == selected_path)
        if state_of_ai_source == t("study_center.source.local_copy"):
            st.warning(t("study_center.warning.local_fallback"))
            if remote_error:
                with st.expander(t("study_center.github_error_details_title")):
                    st.code(remote_error)
        try:
            if state_of_ai_source == t("study_center.source.github"):
                content = load_remote_markdown(current_doc["raw_url"])
            else:
                content = load_local_markdown(current_doc["path"])
        except Exception as exc:  # noqa: BLE001
            st.error(t("study_center.error.open_doc", path=current_doc["path"], error=exc))
            return

    st.subheader(f"📄 {current_doc['title']}")
    st.caption(t("study_center.file_label", path=current_doc["path"]))
    st.markdown("---")
    st.markdown(content)


if __name__ == "__main__":
    main()
