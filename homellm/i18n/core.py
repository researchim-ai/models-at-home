"""
Internationalization (i18n) module for HomeLLM.

Provides multi-language support using JSON translation files.
"""

import json
from pathlib import Path
from typing import Optional
import streamlit as st

LOCALES_DIR = Path(__file__).parent / "locales"
DEFAULT_LANG = "en"
SUPPORTED_LANGS = {"en": "English", "ru": "Русский"}

_translations: dict[str, dict] = {}
_loaded = False


def load_translations() -> None:
    """Load all available translation files."""
    global _translations, _loaded
    
    if _loaded:
        return
    
    for file in LOCALES_DIR.glob("*.json"):
        lang = file.stem
        try:
            _translations[lang] = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to load translation file {file}: {e}")
    
    _loaded = True


def t(key: str, **kwargs) -> str:
    """
    Get translation by key with optional interpolation.
    
    Args:
        key: Translation key (e.g., "app.title", "button.start")
        **kwargs: Variables for string interpolation
    
    Returns:
        Translated string, or key itself if not found
    
    Example:
        t("error.file_read", error="File not found")
        # Returns: "Error reading file: File not found"
    """
    load_translations()
    
    lang = st.session_state.get("lang", DEFAULT_LANG)
    
    # Fallback chain: current lang -> default lang -> key itself
    text = _translations.get(lang, {}).get(key)
    if text is None:
        text = _translations.get(DEFAULT_LANG, {}).get(key, key)
    
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    
    return text


def get_languages() -> list[str]:
    """Get list of available language codes."""
    load_translations()
    return sorted(_translations.keys())


def get_language_name(code: str) -> str:
    """Get human-readable language name."""
    return SUPPORTED_LANGS.get(code, code.upper())


def get_current_language() -> str:
    """Get current language code from session state."""
    return st.session_state.get("lang", DEFAULT_LANG)


def set_language(lang: str) -> None:
    """Set current language in session state."""
    st.session_state.lang = lang


def language_selector(label: str = "🌐") -> str:
    """
    Render language selector widget in Streamlit sidebar.
    
    Args:
        label: Label for the selector
    
    Returns:
        Selected language code
    """
    load_translations()
    
    langs = get_languages()
    if not langs:
        langs = [DEFAULT_LANG]
    
    current = st.session_state.get("lang", DEFAULT_LANG)
    if current not in langs:
        current = langs[0]
    
    idx = langs.index(current)
    
    selected = st.selectbox(
        label,
        langs,
        index=idx,
        format_func=get_language_name,
        key="lang_selector"
    )
    
    if selected != st.session_state.get("lang"):
        st.session_state.lang = selected
        st.rerun()
    
    return selected
