"""Persistent JSON-line inference bridge, isolated in the VLM Python."""
import contextlib
import json
from pathlib import Path
import sys
import os

if sys.platform == 'linux':
    # Do not retain GPU allocations if the API is killed without its lifespan.
    import ctypes
    import signal
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if parent == 1 or os.getppid() != parent:
        raise SystemExit(1)

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == '__main__':
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from backend.routes import vlm
        from fastapi import HTTPException
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request['operation'] == 'load':
                    value = vlm.load_vlm(vlm.VLMLoadRequest(**request['payload']))
                elif request['operation'] == 'generate':
                    value = vlm.generate_vlm(vlm.VLMGenerateRequest(**request['payload']))
                else:
                    raise ValueError('Unknown inference bridge operation')
            response = {'ok': True, 'value': value}
        except HTTPException as exc:
            response = {'ok': False, 'status': exc.status_code, 'error': str(exc.detail)}
        except Exception as exc:
            response = {'ok': False, 'status': 500, 'error': str(exc)}
        print(json.dumps(response, ensure_ascii=False), file=output, flush=True)
