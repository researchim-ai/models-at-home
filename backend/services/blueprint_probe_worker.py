"""Isolate synthetic probes so the API can enforce an execution deadline."""
import contextlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
if __name__=='__main__':
    payload=json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        from backend.services.blueprint_analysis import synthetic_probe
        result=synthetic_probe(payload['blueprint'],payload.get('profile',{}))
    print(json.dumps(result,allow_nan=False))
