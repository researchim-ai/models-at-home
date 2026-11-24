"""Supervise workers and recover ownership without trusting a reusable PID alone."""
from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import threading
from pathlib import Path

import psutil
from fastapi import HTTPException
from backend.paths import atomic_json

_children: dict[str, subprocess.Popen | psutil.Process] = {}
_lock = threading.RLock()
_terminal = {"completed", "error", "stopped"}


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _verified(run_dir: Path) -> psutil.Process | None:
    record = _read(run_dir / "process.json")
    try:
        pid = record["pid"]
        if not isinstance(pid, int) or pid <= 0:
            return None
        proc = psutil.Process(pid)
        if abs(proc.create_time() - record["created_at"]) > 0.001:
            return None
        if proc.status() == psutil.STATUS_ZOMBIE:
            return None
        env = proc.environ()
        token = record.get("token")
        if (not isinstance(token, str) or not token
                or not secrets.compare_digest(env.get("MAH_WORKER_TOKEN", ""), token)
                or env.get("MAH_WORKER_RUN_DIR") != str(run_dir.resolve())):
            return None
        return proc
    except (KeyError, TypeError, ValueError, psutil.Error):
        return None


def _live(proc: subprocess.Popen | psutil.Process) -> bool:
    if isinstance(proc, subprocess.Popen):
        return proc.poll() is None
    try:
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _finish(run_dir: Path, proc, code: int | None) -> None:
    with _lock:
        key = str(run_dir.resolve())
        if _children.get(key) is not proc:
            return
        metrics = _read(run_dir / "metrics.json")
        if metrics.get("status") not in _terminal:
            metrics["status"] = "completed" if code == 0 else "error"
            if code != 0:
                metrics["error"] = (f"Worker exited with code {code}; see stderr.log" if code is not None
                                    else "Recovered worker exited without a terminal status; see stderr.log")
            atomic_json(run_dir / "metrics.json", metrics)
        (run_dir / "pid").unlink(missing_ok=True)
        _children.pop(key, None)


def _watch(run_dir: Path, proc) -> None:
    def reap():
        try:
            code = proc.wait()
        except psutil.Error:
            code = None
        _finish(run_dir, proc, code)
    threading.Thread(target=reap, daemon=True).start()


def launch(run_dir: Path, cmd: list[str], *, cwd: Path, env: dict) -> subprocess.Popen:
    with _lock:
        if is_running(run_dir):
            raise HTTPException(409, "A worker already uses this run directory")
        token = secrets.token_hex(32)
        worker_env = {**env, "MAH_WORKER_TOKEN": token, "MAH_WORKER_RUN_DIR": str(run_dir.resolve())}
        with (run_dir / "stdout.log").open("a") as out, (run_dir / "stderr.log").open("a") as err:
            try:
                proc = subprocess.Popen(cmd, cwd=str(cwd), env=worker_env, stdout=out, stderr=err,
                                        start_new_session=os.name != "nt")
            except OSError as exc:
                atomic_json(run_dir / "metrics.json", {"status": "error", "error": str(exc)})
                raise HTTPException(500, f"Could not start worker: {exc}") from exc
        key = str(run_dir.resolve())
        _children[key] = proc
        try:
            try:
                created = psutil.Process(proc.pid).create_time()
            except psutil.NoSuchProcess:
                # A very short command can finish before metadata is captured.
                created = None
            if created is not None:
                atomic_json(run_dir / 'process.json', {'pid': proc.pid, 'created_at': created, 'token': token})
            (run_dir / 'pid').write_text(str(proc.pid))
        except (OSError, psutil.Error) as exc:
            # Failure to persist ownership must not leave an untracked worker
            # (or its distributed children) consuming GPUs in the background.
            try:
                _terminate(proc)
            finally:
                if not _live(proc):
                    _children.pop(key, None)
                else:
                    _watch(run_dir, proc)
            try:
                metrics = _read(run_dir / 'metrics.json')
                metrics.update(status='error', error=f'Could not persist worker ownership: {exc}')
                atomic_json(run_dir / 'metrics.json', metrics)
            except OSError:
                pass  # The same disk failure can also prevent status persistence.
            raise HTTPException(500, f'Could not persist worker ownership: {exc}') from exc
        _watch(run_dir, proc)
        return proc


def is_running(run_dir: Path) -> bool:
    with _lock:
        proc = _children.get(str(run_dir.resolve()))
    if proc is not None:
        return _live(proc)
    if (run_dir / "process.json").exists() and _verified(run_dir) is not None:
        return True
    # Legacy PID files carry no ownership proof. Protect live PIDs from delete,
    # but never send them a signal.
    try:
        pid = int((run_dir / "pid").read_text())
        return pid > 0 and psutil.pid_exists(pid)
    except (OSError, ValueError):
        return False


def is_owned(run_dir: Path) -> bool:
    with _lock:
        proc = _children.get(str(run_dir.resolve()))
        return _live(proc) if proc is not None else _verified(run_dir) is not None


def recover(runs_dir: Path) -> None:
    """Adopt workers from a previous backend; reject forged/stale ownership."""
    if not runs_dir.exists():
        return
    with _lock:
        for path in runs_dir.glob("*/process.json"):
            run_dir = path.parent
            if path.is_symlink() or not run_dir.resolve().is_relative_to(runs_dir.resolve()):
                continue
            key = str(run_dir.resolve())
            if key in _children:
                continue
            proc = _verified(run_dir)
            if proc is not None:
                _children[key] = proc
                _watch(run_dir, proc)
            else:
                metrics = _read(run_dir / "metrics.json")
                if metrics.get("status") not in _terminal:
                    metrics.update(status="error", error="Worker is no longer running or its identity could not be verified")
                    atomic_json(run_dir / "metrics.json", metrics)
                if not is_running(run_dir):
                    (run_dir / "pid").unlink(missing_ok=True)


def _terminate(proc: subprocess.Popen | psutil.Process) -> None:
    if not _live(proc):
        return
    try:
        root = psutil.Process(proc.pid) if isinstance(proc, subprocess.Popen) else proc
        targets = root.children(recursive=True) + [root]
    except psutil.NoSuchProcess:
        return
    # psutil checks creation time when signalling, protecting against PID reuse.
    for target in reversed(targets):
        try:
            target.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=5)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
        raise HTTPException(500, "Worker did not stop")
    if isinstance(proc, subprocess.Popen):
        proc.wait(timeout=5)


def stop(run_dir: Path) -> None:
    key = str(run_dir.resolve())
    with _lock:
        proc = _children.get(key) or _verified(run_dir)
        if proc is None and is_running(run_dir):
            raise HTTPException(409, "Worker has no valid ownership record; refusing to signal an unverified PID")
        if proc is None:
            try:
                pid = int((run_dir / "pid").read_text())
                live_pid = pid > 0 and psutil.pid_exists(pid)
            except (OSError, ValueError):
                live_pid = False
            if live_pid:
                raise HTTPException(409, "Worker identity cannot be verified; refusing to signal its PID")
        else:
            _terminate(proc)
            _children.pop(key, None)
        (run_dir / "pid").unlink(missing_ok=True)
        metrics = _read(run_dir / "metrics.json")
        if metrics.get("status") not in ("completed", "error"):
            metrics["status"] = "stopped"
            atomic_json(run_dir / "metrics.json", metrics)


def shutdown() -> None:
    with _lock:
        for key in list(_children):
            try:
                stop(Path(key))
            except (OSError, psutil.Error, subprocess.SubprocessError, HTTPException):
                pass
