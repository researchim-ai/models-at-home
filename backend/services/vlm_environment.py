"""Keep modern VLM dependencies separate from the legacy text runtime."""
import os
from pathlib import Path
import sys


def python_command():
    explicit = os.environ.get('MAH_VLM_PYTHON')
    if explicit:
        path = Path(explicit).expanduser().absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f'MAH_VLM_PYTHON is not an executable: {path}')
        return str(path)
    root = Path(__file__).resolve().parents[2]
    candidate = root / '.venv-vlm' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    return str(candidate) if candidate.is_file() else sys.executable
