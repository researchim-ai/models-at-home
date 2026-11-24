"""
FastAPI backend — thin API layer between Electron/React frontend and Python ML code.
Replaces Streamlit as the UI communication layer.
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.paths import get_project_root
from backend.services import processes
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent / "models-at-home")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes import (
    training, models, datasets, chat, agent, system, vlm, vla, world_models, world_foundation, world_play, world_resources, study,
    blueprints, notebooks,
)
from backend.ws import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(processes.recover, PROJECT_ROOT / ".runs")
    try:
        yield
    finally:
        await run_in_threadpool(world_play.service.close)
        await run_in_threadpool(processes.shutdown)
        await run_in_threadpool(agent.stop_server)
        await run_in_threadpool(chat.unload_model)
        await run_in_threadpool(vlm.unload_vlm)


app = FastAPI(
    title="Models at Home Studio API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def check_origin(request, call_next):
    origin = request.headers.get("origin")
    allowed = {"null", "http://localhost:5173", "http://127.0.0.1:5173", str(request.base_url).rstrip("/")}
    if request.method == "OPTIONS":
        return await call_next(request)
    token = os.environ.get("MAH_API_TOKEN")
    if token and not secrets.compare_digest(request.headers.get("x-studio-token", ""), token):
        return JSONResponse({"detail": "Invalid Studio session token"}, status_code=401)
    if origin == "null" and not token and request.url.path != "/api/system/health":
        return JSONResponse({"detail": "File origins require a Studio session token"}, status_code=403)
    if origin and origin not in allowed:
        return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
    return await call_next(request)

app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(agent.router, prefix="/api/agent", tags=["agent"])
app.include_router(vla.router, prefix="/api/vla", tags=["vla"])
app.include_router(world_models.router, prefix="/api/world-models", tags=["world-models"])
app.include_router(world_resources.router, prefix="/api/world-models/resources", tags=["world-models"])
app.include_router(world_play.router, prefix="/api/world-models/play", tags=["world-models"])
app.include_router(world_foundation.router, prefix="/api/world-models/foundation", tags=["world-models"])
app.include_router(vlm.router, prefix="/api/vlm", tags=["vlm"])
app.include_router(study.router, prefix="/api/study", tags=["study"])
app.include_router(blueprints.router, prefix="/api/blueprints", tags=["blueprints"])
app.include_router(notebooks.router, prefix="/api/notebooks", tags=["notebooks"])
app.include_router(ws_router)
