"""System endpoints: health, GPU info, configs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi import APIRouter
from backend.paths import get_project_root

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
CONFIGS_DIR = PROJECT_ROOT / "configs"


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/gpu")
def gpu_info():
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "id": int(parts[0]),
                        "memory_used_gb": round(float(parts[1]) / 1024, 2),
                        "memory_total_gb": round(float(parts[2]) / 1024, 2),
                        "memory_percent": round(float(parts[1]) / float(parts[2]) * 100, 1) if float(parts[2]) > 0 else 0,
                        "utilization": int(parts[3]) if parts[3].strip().isdigit() else None,
                    })
    except Exception:
        pass
    return {"gpus": gpus}


@router.get("/configs")
def list_configs():
    accelerate = []
    deepspeed = []

    if CONFIGS_DIR.exists():
        for f in sorted(CONFIGS_DIR.iterdir()):
            if f.suffix in (".yaml", ".yml") and "accelerate" in f.name:
                accelerate.append(f.name)
            elif f.suffix == ".json" and "deepspeed" in f.name:
                deepspeed.append(f.name)

    return {"accelerate": accelerate, "deepspeed": deepspeed}
