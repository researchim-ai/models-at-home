"""Streamlit page for app language/theme settings."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    from homellm.i18n import t, load_translations
except ImportError:
    from ..i18n import t, load_translations

try:
    from homellm.app.ui_preferences import (
        DEFAULT_THEME,
        init_user_preferences,
        apply_theme_css,
        render_settings_panel,
    )
except ImportError:
    from ..ui_preferences import (
        DEFAULT_THEME,
        init_user_preferences,
        apply_theme_css,
        render_settings_panel,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = PROJECT_ROOT / ".runs"
RUNS_DIR.mkdir(exist_ok=True)
USER_PREFS_FILE = RUNS_DIR / "ui_preferences.json"


def main() -> None:
    load_translations()
    st.set_page_config(page_title=t("settings.title"), page_icon="⚙️", layout="wide")

    init_user_preferences(USER_PREFS_FILE)
    apply_theme_css(st.session_state.get("ui_theme", DEFAULT_THEME))
    render_settings_panel(USER_PREFS_FILE)


if __name__ == "__main__":
    main()
