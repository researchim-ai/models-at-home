import sys

import pytest

from backend.services import vlm_environment


def test_explicit_vlm_python_does_not_change_main_interpreter(monkeypatch):
    monkeypatch.setenv('MAH_VLM_PYTHON', sys.executable)
    assert vlm_environment.python_command() == sys.executable


def test_invalid_vlm_python_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv('MAH_VLM_PYTHON', str(tmp_path / 'missing-python'))
    with pytest.raises(ValueError, match='MAH_VLM_PYTHON'):
        vlm_environment.python_command()


def test_vlm_environment_prefers_modern_venv_and_falls_back(monkeypatch, tmp_path):
    monkeypatch.delenv('MAH_VLM_PYTHON', raising=False)
    monkeypatch.setattr(vlm_environment, '__file__', str(tmp_path / 'backend' / 'services' / 'vlm_environment.py'))
    assert vlm_environment.python_command() == sys.executable
    path = tmp_path / '.venv-vlm' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    path.parent.mkdir(parents=True)
    path.write_text('placeholder')
    assert vlm_environment.python_command() == str(path)
