import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest
from fastapi import HTTPException

from backend.paths import atomic_json
from backend.routes import notebooks
from backend.services import processes


def test_recover_worker_after_backend_crash(workspace):
    run = workspace / '.runs' / 'orphan'
    run.mkdir()
    atomic_json(run / 'metrics.json', {'status': 'training', 'step': 1})
    # A separate supervisor exits without its normal shutdown, as in a crash.
    source = (
        'import os,sys\nfrom pathlib import Path\nfrom backend.services import processes\n'
        f'run=Path({str(run)!r})\n'
        'processes.launch(run,[sys.executable,"-c","import time; time.sleep(60)"],cwd=run,env=os.environ.copy())\n'
        'os._exit(0)\n'
    )
    subprocess.run([sys.executable, '-c', source], cwd=Path(__file__).resolve().parents[2], check=True, timeout=15)
    record = json.loads((run / 'process.json').read_text())
    child = psutil.Process(record['pid'])
    try:
        processes.recover(workspace / '.runs')
        assert processes.is_owned(run)
        assert str(run.resolve()) in processes._children
        processes.stop(run)
        assert not processes.is_running(run)
        assert json.loads((run / 'metrics.json').read_text())['status'] == 'stopped'
    finally:
        if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            child.kill()


def test_forged_record_never_signals_unrelated_process(workspace):
    run = workspace / '.runs' / 'forged'
    run.mkdir()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        atomic_json(run / 'process.json', {'pid': child.pid, 'created_at': psutil.Process(child.pid).create_time(), 'token': 'wrong'})
        (run / 'pid').write_text(str(child.pid))
        assert not processes.is_owned(run)
        with pytest.raises(HTTPException) as error:
            processes.stop(run)
        assert error.value.status_code == 409
        assert child.poll() is None
        processes.recover(workspace / '.runs')
        assert json.loads((run / 'metrics.json').read_text())['status'] == 'error'
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_jupyter_token_survives_backend_restart(workspace, monkeypatch):
    run = workspace / '.runs' / 'jupyter'
    run.mkdir()
    monkeypatch.setattr(notebooks, 'JUPYTER_RUN', run)
    monkeypatch.setattr(notebooks, 'JUPYTER_TOKEN', '')
    token = notebooks._connection_token()
    assert len(token) >= 32
    monkeypatch.setattr(notebooks, 'JUPYTER_TOKEN', '')
    assert notebooks._connection_token() == token
    monkeypatch.setattr(notebooks, 'JUPYTER_TOKEN', 'explicit-environment-token')
    assert notebooks._connection_token() == 'explicit-environment-token'
    assert json.loads((run / 'connection.json').read_text())['token'] == 'explicit-environment-token'


def test_jupyter_rejects_unverified_live_pid(workspace, monkeypatch):
    run = workspace / '.runs' / 'jupyter'
    run.mkdir()
    (run / 'pid').write_text(str(os.getpid()))
    monkeypatch.setattr(notebooks, 'JUPYTER_RUN', run)
    monkeypatch.setattr(notebooks, 'JUPYTER_TOKEN', '')
    with pytest.raises(HTTPException) as error:
        notebooks._connection_token()
    assert error.value.status_code == 409


@pytest.mark.parametrize('failed_file', ['process.json', 'pid'])
def test_failed_ownership_write_stops_new_worker(workspace, monkeypatch, failed_file):
    run = workspace / '.runs' / 'write-failure'
    run.mkdir()
    original_write = Path.write_text
    def write(path, *args, **kwargs):
        if path.name == failed_file or (failed_file == 'process.json' and path.name.startswith('process.')):
            raise OSError('synthetic disk failure')
        return original_write(path, *args, **kwargs)
    if failed_file == 'process.json':
        original_atomic = processes.atomic_json
        def fail_atomic(path, value):
            if path.name == failed_file:
                raise OSError('synthetic disk failure')
            return original_atomic(path, value)
        monkeypatch.setattr(processes, 'atomic_json', fail_atomic)
    else:
        monkeypatch.setattr(Path, 'write_text', write)
    with pytest.raises(HTTPException) as error:
        processes.launch(run, [sys.executable, '-c', 'import time; time.sleep(60)'], cwd=workspace, env=os.environ.copy())
    assert error.value.status_code == 500
    assert str(run.resolve()) not in processes._children
    assert not processes.is_running(run)
    assert json.loads((run / 'metrics.json').read_text())['status'] == 'error'
