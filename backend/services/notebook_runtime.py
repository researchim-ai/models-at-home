"""Local Jupyter kernels for Studio's native notebook editor.

Each client is owned by one worker thread (ZMQ sockets are not thread safe).
Only plain text, PNG/JPEG and JSON are exposed to the renderer: saved HTML/JS
and widget payloads are never executed by the desktop renderer.
"""
from __future__ import annotations

import copy
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException

MAX_OUTPUT = 2_000_000
MAX_SECONDS = 600


class Session:
    def __init__(self, root: Path, path: str):
        self.id = uuid.uuid4().hex
        self.path = path
        self.root = root
        self.lock = threading.RLock()
        self.jobs = queue.Queue()
        self.ready = threading.Event()
        self.closed = False
        self.state = 'starting'
        self.error = None
        self.result = None
        self.manager = None
        self.last_used = time.monotonic()
        self.thread = threading.Thread(target=self._worker, daemon=True, name=f'notebook-{self.id[:8]}')
        self.thread.start()

    def snapshot(self):
        with self.lock:
            self.last_used = time.monotonic()
            return copy.deepcopy({'id': self.id, 'path': self.path, 'state': self.state,
                                  'error': self.error, 'execution': self.result})

    def execute(self, code: str, cell_id: str):
        with self.lock:
            if self.closed or self.state != 'idle':
                raise HTTPException(409, 'Kernel is busy or unavailable')
            self.state = 'busy'
            self.last_used = time.monotonic()
            self.result = {'id': uuid.uuid4().hex, 'cell_id': cell_id, 'status': 'running',
                           'outputs': [], 'execution_count': None}
            self.jobs.put(code)
            return self.snapshot()

    def interrupt(self):
        with self.lock:
            if self.manager and self.state == 'busy':
                self.manager.interrupt_kernel()
        return self.snapshot()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.manager and self.state == 'busy':
                self.manager.interrupt_kernel()
            self.jobs.put(None)
        self.thread.join(timeout=8)
        if self.thread.is_alive() and self.manager:
            self.manager.shutdown_kernel(now=True)
            self.thread.join(timeout=3)

    def _worker(self):
        client = None
        try:
            from jupyter_client import KernelManager
            self.manager = KernelManager(kernel_name='python3')
            # Use the API interpreter, not an unrelated globally registered kernel.
            self.manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
            env = dict(os.environ)
            env.pop('MAH_API_TOKEN', None)
            env.pop('JUPYTER_TOKEN', None)
            repo = Path(__file__).resolve().parents[2]
            env['PYTHONPATH'] = os.pathsep.join([str(repo), str(self.root), env.get('PYTHONPATH', '')])
            self.manager.start_kernel(cwd=str(self.root), env=env, stdout=subprocess_devnull(), stderr=subprocess_devnull())
            client = self.manager.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=25)
            with self.lock:
                self.state = 'idle'
            self.ready.set()
            while not self.closed:
                try:
                    code = self.jobs.get(timeout=30)
                except queue.Empty:
                    if time.monotonic() - self.last_used > 3600:
                        break
                    if not self.manager.is_alive():
                        raise RuntimeError('Python kernel exited. Restart the kernel to continue.')
                    continue
                if code is None:
                    break
                self._run(client, code)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.state = 'error'
                if self.result:
                    self.result['status'] = 'error'
            self.ready.set()
        finally:
            if client:
                client.stop_channels()
            if self.manager and self.manager.has_kernel:
                self.manager.shutdown_kernel(now=True)
            with self.lock:
                if self.state != 'error':
                    self.state = 'closed'

    def _run(self, client, code):
        msg_id = client.execute(code, allow_stdin=False, stop_on_error=True)
        deadline = time.monotonic() + MAX_SECONDS
        size = 0
        failed = False
        truncated = False
        clear_pending = False
        while True:
            if time.monotonic() > deadline:
                self.manager.interrupt_kernel()
                raise TimeoutError(f'Cell exceeded {MAX_SECONDS}s. Restart the kernel to continue.')
            try:
                msg = client.get_iopub_msg(timeout=0.25)
            except queue.Empty:
                if not self.manager.is_alive():
                    raise RuntimeError('Python kernel exited during execution')
                continue
            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue
            kind, data = msg['header']['msg_type'], msg['content']
            with self.lock:
                if kind == 'status' and data['execution_state'] == 'idle':
                    self.result['status'] = 'error' if failed else 'ok'
                    self.state = 'idle'
                    break
                if kind == 'execute_input':
                    self.result['execution_count'] = data['execution_count']
                if kind == 'clear_output':
                    if data.get('wait'):
                        clear_pending = True
                    else:
                        self.result['outputs'] = []
                    continue
                output = None
                if kind == 'stream':
                    output = {'output_type': 'stream', 'name': data['name'], 'text': data['text']}
                elif kind == 'error':
                    failed = True
                    output = {'output_type': 'error', 'ename': data['ename'], 'evalue': data['evalue'], 'traceback': data['traceback']}
                elif kind in ('display_data', 'execute_result'):
                    safe = {k: v for k, v in data.get('data', {}).items()
                            if k in ('text/plain', 'image/png', 'image/jpeg', 'application/json')}
                    output = {'output_type': kind, 'data': safe, 'metadata': {}}
                    if kind == 'execute_result':
                        output['execution_count'] = data.get('execution_count')
                if output:
                    if clear_pending:
                        self.result['outputs'] = []
                        clear_pending = False
                    length = len(str(output))
                    if size + length <= MAX_OUTPUT:
                        self.result['outputs'].append(output)
                        size += length
                    elif not truncated:
                        self.result['outputs'].append({'output_type': 'stream', 'name': 'stderr',
                                                       'text': '\n[Studio: output truncated at 2 MB]\n'})
                        truncated = True
        # Drain shell replies to avoid accumulating messages during long sessions.
        while True:
            try:
                reply = client.get_shell_msg(timeout=1)
            except queue.Empty:
                break
            if reply.get('parent_header', {}).get('msg_id') == msg_id:
                break


def subprocess_devnull():
    import subprocess
    return subprocess.DEVNULL


_sessions: dict[str, Session] = {}
_lock = threading.RLock()


def available():
    import importlib.util
    return all(importlib.util.find_spec(name) is not None for name in ('jupyter_client', 'ipykernel'))


def start(root: Path, path: str):
    if not available():
        raise HTTPException(503, 'Install notebook dependencies in the backend environment: python -m pip install jupyter-client ipykernel matplotlib')
    with _lock:
        for key, value in list(_sessions.items()):
            if value.state in ('closed', 'error'):
                value.close()
                del _sessions[key]
        if len(_sessions) >= 4:
            raise HTTPException(409, 'Close a notebook kernel before opening another (limit: 4)')
        session = Session(root, path)
        _sessions[session.id] = session
    session.ready.wait(timeout=30)
    if session.state != 'idle':
        stop(session.id)
        raise HTTPException(503, session.error or 'Python kernel did not become ready')
    return session.snapshot()


def get(session_id: str):
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, 'Notebook session not found; start a new kernel')
    return session


def stop(session_id: str):
    with _lock:
        session = _sessions.pop(session_id, None)
    if session:
        session.close()


def shutdown():
    with _lock:
        ids = list(_sessions)
    for key in ids:
        stop(key)
