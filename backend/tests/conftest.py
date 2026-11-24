import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Route imports create workspace folders. Never import against the user's ML data.
_session_root = tempfile.mkdtemp(prefix="mah-tests-")
os.environ["MODELS_AT_HOME_ROOT"] = _session_root

from fastapi.testclient import TestClient
from backend.api import app
from backend.routes import training, vlm, vla, world_models, world_foundation, world_play, world_resources, datasets, models, study, agent, notebooks, blueprints
from backend import ws
from backend.services import processes


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for module in (training, vlm, vla, world_models, world_foundation, world_play, world_resources, datasets, models, study, agent, notebooks, blueprints, ws):
        monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    for module, field, name in [
        (training, "RUNS_DIR", ".runs"), (vlm, "RUNS_DIR", ".runs"), (vla, "RUNS_DIR", ".runs"), (world_models, "RUNS_DIR", ".runs"), (world_foundation, "RUNS_DIR", ".runs"), (ws, "RUNS_DIR", ".runs"),
        (training, "OUTPUT_DIR", "out"), (vlm, "OUTPUT_DIR", "out"), (models, "OUTPUT_DIR", "out"),
        (models, "MODELS_DIR", "models"), (datasets, "DATASET_DIR", "datasets"),
        (study, "STUDY_DIR", "study_materials"), (agent, "SESSIONS_DIR", ".runs/agent_sessions"),
        (blueprints, 'BLUEPRINTS_DIR', 'blueprints'),
    ]:
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, field, path)
    yield tmp_path
    world_play.service.close()
    processes.shutdown()


@pytest.fixture
def client(workspace):
    # No lifespan: route tests must not import/unload GPU runtimes.
    return TestClient(app)


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_session_root, ignore_errors=True)
