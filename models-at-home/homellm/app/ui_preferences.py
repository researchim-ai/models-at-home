"""UI preferences and first-run onboarding for Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

try:
    from homellm.i18n import t, get_languages, get_language_name
except ImportError:
    from ..i18n import t, get_languages, get_language_name

DEFAULT_THEME = "streamlit_dark"

THEME_PRESETS = {
    "streamlit_dark": {
        "label_key": "settings.theme.streamlit_dark",
        "theme": {"base": "dark"},
    },
    "streamlit_light": {
        "label_key": "settings.theme.streamlit_light",
        "theme": {"base": "light"},
    },
    "ocean_mist": {
        "label_key": "settings.theme.ocean_mist",
        "theme": {
            "base": "light",
            "primaryColor": "#0EA5A4",
            "backgroundColor": "#F5FBFB",
            "secondaryBackgroundColor": "#E6F4F4",
            "textColor": "#102A43",
        },
    },
    "forest_night": {
        "label_key": "settings.theme.forest_night",
        "theme": {
            "base": "dark",
            "primaryColor": "#34D399",
            "backgroundColor": "#0B1410",
            "secondaryBackgroundColor": "#12201A",
            "textColor": "#E5F3EA",
        },
    },
    "sunset_warm": {
        "label_key": "settings.theme.sunset_warm",
        "theme": {
            "base": "light",
            "primaryColor": "#F97316",
            "backgroundColor": "#FFF8F1",
            "secondaryBackgroundColor": "#FDEEDC",
            "textColor": "#3A2A1B",
        },
    },
}


def _normalize_theme_key(theme: str) -> str:
    # Migration for old values from previous implementation.
    if theme == "dark":
        return "streamlit_dark"
    if theme == "light":
        return "streamlit_light"
    return theme if theme in THEME_PRESETS else "streamlit_dark"


def _theme_label(theme: str) -> str:
    if theme in THEME_PRESETS:
        return t(THEME_PRESETS[theme]["label_key"])
    return theme


def load_user_preferences(user_prefs_file: Path) -> dict:
    """Load UI preferences from disk."""
    default = {"lang": "en", "theme": "streamlit_dark", "onboarding_done": False}
    if not user_prefs_file.exists():
        return default

    try:
        with open(user_prefs_file, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default

    lang = str(data.get("lang", default["lang"]))
    if lang not in {"en", "ru"}:
        lang = default["lang"]

    theme = _normalize_theme_key(str(data.get("theme", default["theme"])))

    onboarding_done = bool(data.get("onboarding_done", False))
    return {"lang": lang, "theme": theme, "onboarding_done": onboarding_done}


def save_user_preferences(user_prefs_file: Path, lang: str, theme: str, onboarding_done: bool = True) -> None:
    """Persist UI preferences to disk."""
    theme_key = _normalize_theme_key(theme)
    data = {"lang": lang, "theme": theme_key, "onboarding_done": onboarding_done}
    with open(user_prefs_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def init_user_preferences(user_prefs_file: Path) -> None:
    """Initialize user preferences in session_state once."""
    if st.session_state.get("_user_prefs_initialized"):
        return

    prefs = load_user_preferences(user_prefs_file)
    if "lang" not in st.session_state:
        st.session_state.lang = prefs["lang"]
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = _normalize_theme_key(prefs["theme"])
    if "onboarding_done" not in st.session_state:
        st.session_state.onboarding_done = prefs["onboarding_done"]

    st.session_state._user_prefs_snapshot = {
        "lang": st.session_state.get("lang", "en"),
        "theme": _normalize_theme_key(st.session_state.get("ui_theme", "streamlit_dark")),
        "onboarding_done": st.session_state.get("onboarding_done", False),
    }
    st.session_state._user_prefs_initialized = True


def persist_user_preferences_if_changed(user_prefs_file: Path) -> None:
    """Save preferences if current session values changed."""
    if not st.session_state.get("onboarding_done", False):
        return

    snapshot = st.session_state.get("_user_prefs_snapshot", {})
    current = {
        "lang": st.session_state.get("lang", "en"),
        "theme": _normalize_theme_key(st.session_state.get("ui_theme", "streamlit_dark")),
        "onboarding_done": st.session_state.get("onboarding_done", False),
    }
    if current != snapshot:
        save_user_preferences(
            user_prefs_file=user_prefs_file,
            lang=current["lang"],
            theme=current["theme"],
            onboarding_done=current["onboarding_done"],
        )
        st.session_state._user_prefs_snapshot = current


def apply_theme_css(theme: str) -> None:
    """Apply selected palette immediately without app restart."""
    theme_key = _normalize_theme_key(theme)
    cfg = THEME_PRESETS.get(theme_key, THEME_PRESETS[DEFAULT_THEME])["theme"]
    base = cfg.get("base", "dark")
    bg = cfg.get("backgroundColor", "#0e1117" if base == "dark" else "#ffffff")
    secondary = cfg.get("secondaryBackgroundColor", "#262730" if base == "dark" else "#f0f2f6")
    text = cfg.get("textColor", "#fafafa" if base == "dark" else "#262730")
    primary = cfg.get("primaryColor", "#ff4b4b")
    border = "#2f3542" if base == "dark" else "#d6d6d6"
    input_bg = "#111827" if base == "dark" else "#ffffff"
    code_bg = "#0d1117" if base == "dark" else "#f6f8fa"

    st.markdown(
        f"""
<style>
    :root {{
        --mth-bg: {bg};
        --mth-secondary: {secondary};
        --mth-text: {text};
        --mth-primary: {primary};
        --mth-border: {border};
        --mth-input-bg: {input_bg};
        --mth-code-bg: {code_bg};
    }}
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {{
        background: var(--mth-bg) !important;
        color: var(--mth-text) !important;
    }}
    [data-testid="stSidebar"] {{
        background: var(--mth-secondary) !important;
        border-right: 1px solid var(--mth-border);
    }}
    [data-testid="stSidebar"] *, .stApp, .stMarkdown, .stCaption, label, p, h1, h2, h3, h4, h5, h6 {{
        color: var(--mth-text) !important;
    }}
    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {{
        background: var(--mth-input-bg) !important;
        color: var(--mth-text) !important;
        border-color: var(--mth-border) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: var(--mth-primary) !important;
        color: #ffffff !important;
        border-color: var(--mth-primary) !important;
    }}
    div[data-testid="metric-container"] {{
        background: var(--mth-secondary);
        border: 1px solid var(--mth-border);
        border-radius: 8px;
        padding: 0.5rem;
    }}
    pre {{
        background: var(--mth-code-bg) !important;
        border: 1px solid var(--mth-border) !important;
    }}
    .main-header {{ font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 0.5rem; }}
    .sub-header {{ text-align: center; font-size: 1.1rem; margin-bottom: 2rem; }}
    .status-running, .status-completed, .status-error {{ font-weight: bold; }}
    .model-ascii {{
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        white-space: pre;
        overflow-x: auto;
        background: var(--mth-secondary);
        border: 1px solid var(--mth-border);
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def render_first_run_setup(user_prefs_file: Path) -> None:
    """Render first-run onboarding and persist selected settings."""
    st.title(f"👋 {t('onboarding.title')}")
    st.caption(t("onboarding.subtitle"))
    st.info(t("onboarding.info"))

    langs = get_languages() or ["en", "ru"]
    selected_lang = st.selectbox(
        t("onboarding.language"),
        options=langs,
        index=langs.index(st.session_state.get("lang", "en")) if st.session_state.get("lang", "en") in langs else 0,
        format_func=get_language_name,
    )
    selected_theme = st.radio(
        t("onboarding.theme"),
        options=list(THEME_PRESETS.keys()),
        index=list(THEME_PRESETS.keys()).index(_normalize_theme_key(st.session_state.get("ui_theme", "streamlit_dark")))
        if _normalize_theme_key(st.session_state.get("ui_theme", "streamlit_dark")) in THEME_PRESETS
        else 0,
        format_func=_theme_label,
        horizontal=False,
    )

    if st.button(t("onboarding.start_button"), type="primary"):
        st.session_state.lang = selected_lang
        st.session_state.ui_theme = selected_theme
        st.session_state.onboarding_done = True
        save_user_preferences(
            user_prefs_file=user_prefs_file,
            lang=selected_lang,
            theme=selected_theme,
            onboarding_done=True,
        )
        st.session_state._user_prefs_snapshot = {
            "lang": selected_lang,
            "theme": _normalize_theme_key(selected_theme),
            "onboarding_done": True,
        }
        st.rerun()


def render_settings_panel(user_prefs_file: Path) -> None:
    """Render settings tab with language/theme controls."""
    st.header(f"⚙️ {t('settings.title')}")
    st.caption(t("settings.subtitle"))

    langs = get_languages() or ["en", "ru"]
    current_lang = st.session_state.get("lang", "en")
    current_theme = _normalize_theme_key(st.session_state.get("ui_theme", "streamlit_dark"))

    new_lang = st.selectbox(
        t("settings.language"),
        options=langs,
        index=langs.index(current_lang) if current_lang in langs else 0,
        format_func=get_language_name,
        key="settings_lang_select",
    )
    new_theme = st.radio(
        t("settings.theme"),
        options=list(THEME_PRESETS.keys()),
        index=list(THEME_PRESETS.keys()).index(current_theme) if current_theme in THEME_PRESETS else 0,
        format_func=_theme_label,
        horizontal=False,
        key="settings_theme_select",
    )

    changed = (new_lang != current_lang) or (_normalize_theme_key(new_theme) != current_theme)
    if changed:
        st.session_state.lang = new_lang
        st.session_state.ui_theme = _normalize_theme_key(new_theme)
        st.session_state.onboarding_done = True
        save_user_preferences(
            user_prefs_file=user_prefs_file,
            lang=new_lang,
            theme=_normalize_theme_key(new_theme),
            onboarding_done=True,
        )
        st.session_state._user_prefs_snapshot = {
            "lang": new_lang,
            "theme": _normalize_theme_key(new_theme),
            "onboarding_done": True,
        }
        st.success(t("settings.saved"))
        st.rerun()
