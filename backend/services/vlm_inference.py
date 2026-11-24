"""Bounded synchronous IPC for the persistent isolated VLM inference worker."""
from collections import deque
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading

from fastapi import HTTPException


class InferenceWorker:
    def __init__(self, python, cwd):
        env = os.environ.copy()
        env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        # Force the child to serve inference locally, including externally set overrides.
        env['MAH_VLM_PYTHON'] = python
        self.process = subprocess.Popen([python, str(Path(__file__).with_name('vlm_inference_worker.py'))],
            cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        self.responses = Queue(maxsize=2)
        self.errors = deque(maxlen=30)
        def read():
            for line in self.process.stdout:
                try:
                    self.responses.put_nowait(json.loads(line))
                except Exception:
                    # A broken protocol must never leave a live GPU worker behind.
                    self.close()
                    return
            try:
                self.responses.put_nowait(None)
            except Exception:
                pass
        def drain():
            for line in self.process.stderr:
                self.errors.append(line[-2000:])
        self.readers = [threading.Thread(target=read, daemon=True), threading.Thread(target=drain, daemon=True)]
        for reader in self.readers:
            reader.start()

    @property
    def alive(self):
        return self.process.poll() is None

    def request(self, operation, payload, timeout=600):
        try:
            self.process.stdin.write(json.dumps({'operation': operation, 'payload': payload}) + '\n')
            self.process.stdin.flush()
            response = self.responses.get(timeout=timeout)
            if response is None:
                raise RuntimeError('VLM worker exited: ' + ''.join(self.errors)[-3000:])
        except (OSError, Empty, RuntimeError) as exc:
            self.close()
            raise HTTPException(500, f'VLM inference worker failed or timed out: {exc}') from exc
        if not response.get('ok'):
            raise HTTPException(response.get('status', 500), response.get('error', 'Invalid worker response'))
        return response['value']

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdin:
            self.process.stdin.close()
