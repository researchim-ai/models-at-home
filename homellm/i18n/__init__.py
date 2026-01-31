"""
HomeLLM Internationalization (i18n) Module.

Usage:
    from homellm.i18n import t, language_selector, load_translations
    
    # Load translations (call once at app start)
    load_translations()
    
    # Get translated string
    title = t("app.title")
    error_msg = t("error.file_read", error=str(e))
    
    # Add language selector to sidebar
    with st.sidebar:
        language_selector()
"""

from .core import (
    t,
    load_translations,
    get_languages,
    get_language_name,
    get_current_language,
    set_language,
    language_selector,
    DEFAULT_LANG,
    SUPPORTED_LANGS,
)

__all__ = [
    "t",
    "load_translations",
    "get_languages",
    "get_language_name", 
    "get_current_language",
    "set_language",
    "language_selector",
    "DEFAULT_LANG",
    "SUPPORTED_LANGS",
]
