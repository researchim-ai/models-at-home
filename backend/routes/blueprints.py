"""Blueprint endpoints: list, get, save, delete + expose block registry."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.paths import get_project_root, safe_child, atomic_json
from backend.services.blueprints import validate

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
BLUEPRINTS_DIR = PROJECT_ROOT / "blueprints"
BLUEPRINTS_DIR.mkdir(parents=True, exist_ok=True)


class SaveBlueprintRequest(BaseModel):
    name: str
    blueprint: dict


@router.post('/validate')
def validate_blueprint(blueprint: dict):
    try:
        return {'blueprint': validate(blueprint)}
    except ValueError as exc:
        raise HTTPException(400, f'Invalid blueprint: {exc}') from exc


def _safe_name(name: str) -> str:
    """Prevent path traversal."""
    name = name.strip()
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise HTTPException(400, 'Blueprint name must be a nonempty filename')
    if not name.endswith(".json"):
        name = f"{name}.json"
    return name


@router.get("/list")
def list_blueprints():
    """List saved blueprints with summary info."""
    items = []
    if not BLUEPRINTS_DIR.exists():
        return {"blueprints": []}

    for f in sorted(BLUEPRINTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            items.append({
                "name": f.stem,
                "path": str(f),
                "hidden_size": data.get("hidden_size", 0),
                "vocab_size": data.get("vocab_size", 0),
                "num_blocks": len(data.get("blocks", [])),
                "modified_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        except Exception:
            continue

    return {"blueprints": items}


@router.get("/get/{name}")
def get_blueprint(name: str):
    """Load a saved blueprint by name."""
    safe = _safe_name(name)
    path = safe_child(BLUEPRINTS_DIR, safe)
    if not path.exists():
        raise HTTPException(404, f"Blueprint {name} not found")

    try:
        data = json.loads(path.read_text())
        return {"blueprint": validate(data)}
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Invalid saved blueprint: {exc}") from exc
    except Exception as e:
        raise HTTPException(500, f"Failed to read blueprint: {e}")


@router.post("/save")
def save_blueprint(req: SaveBlueprintRequest):
    """Save a blueprint to disk. Validates using homellm.models.blueprint.Blueprint if available."""
    safe = _safe_name(req.name)
    path = safe_child(BLUEPRINTS_DIR, safe)

    try:
        blueprint = validate(req.blueprint)
    except ValueError as exc:
        raise HTTPException(400, f'Invalid blueprint: {exc}') from exc

    try:
        atomic_json(path, blueprint)
        return {"success": True, "path": str(path)}
    except Exception as e:
        raise HTTPException(500, f"Failed to save: {e}")


@router.delete("/delete/{name}")
def delete_blueprint(name: str):
    safe = _safe_name(name)
    path = safe_child(BLUEPRINTS_DIR, safe)
    path.unlink(missing_ok=True)
    return {"success": True}


@router.get('/blocks')
def list_blocks():
    from backend.services.blueprint_catalog import catalog
    return {'blocks': catalog()}


@router.post('/analyze')
def analyze_blueprint(blueprint: dict):
    try:
        from backend.services.blueprint_analysis import analyze
        return analyze(blueprint)
    except ImportError as exc:
        raise HTTPException(503, f'Architecture analysis needs the ML runtime and Torch: {exc}') from exc
    except (ValueError, TypeError, KeyError, RuntimeError, AssertionError) as exc:
        raise HTTPException(400, f'Invalid architecture: {exc}') from exc


class ResearchRequest(BaseModel):
    blueprint: dict
    profile: dict = {}


@router.post('/research/analyze')
def research_analyze(req: ResearchRequest):
    try:
        from backend.services.blueprint_analysis import analyze
        return analyze(req.blueprint, req.profile)
    except ImportError as exc:
        raise HTTPException(503, f'Research analysis needs the ML runtime: {exc}') from exc
    except (ValueError, TypeError, KeyError, RuntimeError, AssertionError) as exc:
        raise HTTPException(400, f'Research analysis: {exc}') from exc


@router.post('/research/probe')
def research_probe(req: ResearchRequest):
    import subprocess
    import sys
    import os
    try:
        from backend.services.blueprint_analysis import analyze
        result=analyze(req.blueprint, req.profile)
        if result['parameters']>8_000_000 or result['output_tensor_volume_mib']>32:
            raise ValueError('Synthetic CPU probe budget exceeded: max 8M parameters and 32 MiB output-tensor volume; use a smaller research recipe')
        env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'services'/'blueprint_probe_worker.py')],
            input=json.dumps({'blueprint':result['blueprint'],'profile':result['profile']}),text=True,capture_output=True,
            cwd=PROJECT_ROOT,env=env,timeout=90)
        if completed.returncode:
            raise ValueError(completed.stderr[-3000:] or 'Synthetic probe failed')
        return json.loads(completed.stdout)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(408,'Synthetic CPU probe exceeded 90 seconds; reduce depth or sequence size') from exc
    except ImportError as exc:
        raise HTTPException(503,f'Synthetic probe requires ML dependencies: {exc}') from exc
    except (ValueError,TypeError,KeyError,RuntimeError,AssertionError) as exc:
        raise HTTPException(400,f'Synthetic probe: {exc}') from exc


@router.get('/research/examples')
def research_examples():
    directory=Path(__file__).resolve().parents[2]/'examples'/'research-lab'
    return {'train':str(directory/'train.jsonl'),'validation':str(directory/'validation.jsonl')}
