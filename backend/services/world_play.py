"""Single owned local-world inference process, serialized RPC and idle cleanup."""
from collections import deque
import json
from pathlib import Path
from queue import Queue, Empty
import os
import subprocess
import threading
import time
import uuid
from fastapi import HTTPException

_lock=threading.RLock()
_worker=None
_session=None
_touched=0.


class LocalWorker:
    def __init__(self,python,cwd,env):
        self.proc=subprocess.Popen([python,str(Path(__file__).with_name('world_play_worker.py'))],cwd=cwd,env=env,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        self.responses=Queue(maxsize=2);self.errors=deque(maxlen=20)
        def read():
            try:
                for line in self.proc.stdout:self.responses.put_nowait(json.loads(line))
            except Exception:pass
            try:self.responses.put_nowait(None)
            except Exception:pass
        def drain():
            for line in self.proc.stderr:self.errors.append(line[-2000:])
        threading.Thread(target=read,daemon=True).start();threading.Thread(target=drain,daemon=True).start()
    def request(self,op,payload,timeout=180):
        try:
            self.proc.stdin.write(json.dumps({'operation':op,'payload':payload})+'\n');self.proc.stdin.flush()
            result=self.responses.get(timeout=timeout)
            if result is None:raise RuntimeError(''.join(self.errors)[-3000:] or 'Worker exited')
        except (OSError,Empty,RuntimeError) as exc:
            self.close();raise HTTPException(500,f'Local world worker failed: {exc}') from exc
        if not result.get('ok'):raise HTTPException(400,result.get('error','Local world error'))
        return result['value']
    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(5)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(5)
        for stream in (self.proc.stdin,self.proc.stdout,self.proc.stderr):
            if stream:
                try:stream.close()
                except OSError:pass


def close(session_id=None):
    global _worker,_session
    with _lock:
        if session_id and _session and session_id!=_session:raise HTTPException(409,'This is a different local session')
        if _worker:_worker.close()
        _worker=_session=None


def start(python,cwd,env,config):
    global _worker,_session,_touched
    with _lock:
        if _worker and _worker.proc.poll() is None:raise HTTPException(409,'Stop the current local world session first')
        close()
        _worker=LocalWorker(python,cwd,env)
        try:
            value=_worker.request('load',config)
        except Exception:
            close();raise
        _session=uuid.uuid4().hex;_touched=time.monotonic()
        return dict(session_id=_session,**value)


def request(session_id,operation,payload):
    global _touched
    with _lock:
        if not _worker or _session!=session_id or _worker.proc.poll() is not None:
            raise HTTPException(410,'Local world session ended; start it again')
        _touched=time.monotonic()
        value=_worker.request(operation,payload)
        _touched=time.monotonic()
        return value


def status():
    with _lock:
        return {'session_id':_session if _worker and _worker.proc.poll() is None else None}


def reap_idle():
    while True:
        time.sleep(30)
        with _lock:
            if _worker and time.monotonic()-_touched>300:close()

threading.Thread(target=reap_idle,daemon=True).start()
