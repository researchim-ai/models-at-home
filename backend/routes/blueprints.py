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
        return {"blueprint": data}
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


@router.get("/blocks")
def list_blocks():
    """List all available block types from the BLOCK_REGISTRY."""
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from homellm.models.blocks import BLOCK_REGISTRY

        blocks = []
        for name, builder in BLOCK_REGISTRY.items():
            blocks.append({
                "type": name,
                "description": getattr(builder, "description", "") or "",
            })
        return {"blocks": blocks}
    except Exception as e:
        fallback = [
            {"type": "token_embedding", "description": "Token embedding layer"},
            {"type": "positional_embedding", "description": "Learnable positional embeddings"},
            {"type": "attention", "description": "Self-attention"},
            {"type": "causal_self_attention", "description": "Flash causal self-attention"},
            {"type": "mlp", "description": "Two-layer MLP"},
            {"type": "swiglu", "description": "SwiGLU feed-forward"},
            {"type": "rmsnorm", "description": "RMS normalization"},
            {"type": "layernorm", "description": "Layer normalization"},
            {"type": "moe", "description": "Mixture of Experts"},
            {"type": "llama_block", "description": "Full Llama transformer block"},
            {"type": "add", "description": "Elementwise sum"},
            {"type": "dropout", "description": "Dropout"},
            {"type": "linear", "description": "Linear projection"},
        ]
        return {"blocks": fallback, "error": str(e)}
