"""WebSocket endpoint for real-time training metrics streaming."""
from __future__ import annotations

import asyncio
import os
import secrets
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from backend.paths import get_project_root, safe_id
from backend.services import metrics
from starlette.concurrency import run_in_threadpool

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent / "models-at-home")
RUNS_DIR = PROJECT_ROOT / ".runs"


@router.websocket("/ws/metrics/{run_id}")
async def metrics_ws(websocket: WebSocket, run_id: str):
    """Stream training metrics from .runs/<run_id>/metrics.json via WebSocket."""
    origin = websocket.headers.get("origin")
    own_origin = f"{('https' if websocket.url.scheme == 'wss' else 'http')}://{websocket.url.netloc}"
    if origin and origin not in {"null", "http://localhost:5173", "http://127.0.0.1:5173", own_origin}:
        await websocket.close(code=1008)
        return
    try:
        metrics_path = safe_id(RUNS_DIR, run_id) / "metrics.json"
    except HTTPException:
        await websocket.close(code=1008)
        return
    token = os.environ.get("MAH_API_TOKEN")
    offered = [value.strip() for value in websocket.headers.get("sec-websocket-protocol", "").split(",")]
    protocol = f"mah.{token}" if token else None
    if (token and not any(secrets.compare_digest(value, protocol) for value in offered)) or (origin == "null" and not token):
        await websocket.close(code=1008)
        return
    await websocket.accept(subprotocol=protocol)

    last_version = None
    last_data = None

    try:
        while True:
            if metrics_path.exists():
                stat = metrics_path.stat()
                version = (stat.st_mtime_ns, stat.st_size)
                if version != last_version:
                    try:
                        raw = await run_in_threadpool(metrics.read, metrics_path)
                        if raw is None:
                            await asyncio.sleep(1)
                            continue
                        data = _normalize_metrics(raw)
                        last_version = version
                        if data != last_data:
                            await websocket.send_json(data)
                            last_data = data
                    except (json.JSONDecodeError, IOError):
                        pass
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()


def _normalize_metrics(data: dict) -> dict:
    return metrics.snapshot(data)
