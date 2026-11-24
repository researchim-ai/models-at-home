"""Persistent headless local-world engine; no cloud API, pygame or shell launcher."""
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

if sys.platform=='linux':
    import ctypes
    parent=os.getppid()
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:sys.exit(0)

world=None
for line in sys.stdin:
    try:
        message=json.loads(line);op=message['operation'];p=message['payload']
        with contextlib.redirect_stdout(sys.stderr):
            if op=='load':
                from backend.services.world_local import LocalWorld
                world=LocalWorld(p);value=world.reset()
            elif op=='step' and world is not None:value=world.step(p)
            elif op=='reset' and world is not None:value=world.reset(p.get('spawn'))
            else:raise ValueError('Unknown world operation or model not loaded')
        print(json.dumps({'ok':True,'value':value},allow_nan=False),flush=True)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'ok':False,'error':str(exc)}),flush=True)
