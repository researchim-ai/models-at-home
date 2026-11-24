"""Model management endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.paths import get_project_root, safe_child

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "out"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class DownloadRequest(BaseModel):
    repo_id: str
    save_name: str = ""


@router.get("/local")
def list_local_models():
    """List models downloaded to models/ directory."""
    models = []
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if not d.is_symlink() and not d.name.startswith(".") and (d.is_dir() or d.suffix.lower() == ".gguf"):
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.is_dir() else d.stat().st_size
                arch = _detect_architecture(d) if d.is_dir() else "gguf"
                models.append({
                    "name": d.name,
                    "path": str(d),
                    "size_bytes": size,
                    "type": "local",
                    "architecture": arch,
                    "modified_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                })
    return {"models": models}


@router.get("/trained")
def list_trained_models():
    """List trained models from out/ directory."""
    models = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir()):
            if d.is_dir() and not d.is_symlink() and not d.name.startswith("."):
                final = d / "final_model"
                target = final if final.exists() else d
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                arch = _detect_architecture(target)

                model_type = "trained"
                if "lora" in d.name.lower() or (d / "adapter_config.json").exists():
                    model_type = "lora"

                models.append({
                    "name": d.name,
                    "path": str(target),
                    "size_bytes": size,
                    "type": model_type,
                    "architecture": arch,
                    "modified_at": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                })
    return {"models": models}


@router.post("/download")
def download_model(req: DownloadRequest):
    """Download a model from HuggingFace Hub."""
    save_name = req.save_name or req.repo_id.split("/")[-1]
    save_path = safe_child(MODELS_DIR, save_name)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise HTTPException(500, "huggingface_hub not installed")


    try:
        snapshot_download(
            repo_id=req.repo_id,
            local_dir=str(save_path),
        )
        return {"success": True, "path": str(save_path)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/local/{name:path}")
def delete_local_model(name: str):
    """Delete a downloaded model from the models/ directory."""
    import shutil
    path = safe_child(MODELS_DIR, name)
    if not path.exists():
        raise HTTPException(404, "Model not found")
    shutil.rmtree(path) if path.is_dir() else path.unlink()
    return {"success": True}


@router.post("/estimate-memory")
def estimate_memory(config: dict):
    """Estimate VRAM usage (simple)."""
    if config.get('model_type') == 'blueprint':
        raise HTTPException(400, 'VRAM estimation for arbitrary blueprint graphs is not supported')
    try:
        from homellm.models.memory_estimator import estimate_memory_footprint
        result = estimate_memory_footprint(
            _memory_config(config),
            batch_size=int(config.get("batch_size", 1) or 1),
            distributed_mode=config.get("parallel_mode", "default") or "default",
            num_gpus=max(1, len(config.get("gpu_ids") or [])) or 1,
        )
        return result
    except Exception as e:
        return {"vram_gb": 0, "params": 0, "error": str(e)}


class EstimateMemoryDetailedRequest(BaseModel):
    config: dict
    batch_size: int = Field(default=1, gt=0)
    distributed_mode: str = "default"
    num_gpus: int = Field(default=1, gt=0)


@router.post("/estimate-memory-detailed")
def estimate_memory_detailed(req: EstimateMemoryDetailedRequest):
    """Full memory estimator — returns breakdown (weights/grads/optim/activations)."""
    if req.config.get('model_type') == 'blueprint':
        raise HTTPException(400, 'VRAM estimation for arbitrary blueprint graphs is not supported')
    try:
        from homellm.models.memory_estimator import estimate_memory_footprint

        cfg = _memory_config(req.config)

        return estimate_memory_footprint(
            cfg,
            batch_size=req.batch_size,
            distributed_mode=req.distributed_mode,
            num_gpus=req.num_gpus,
        )
    except Exception as e:
        return {"error": str(e)}


def _memory_config(value: dict) -> dict:
    cfg = dict(value)
    # Architecture of a loaded model is authoritative. UI defaults for the
    # previous from-scratch model must not replace its actual parameter count.
    source = cfg.get('base_model_path') or cfg.get('model_path')
    if cfg.get('model_type') in ('hf', 'checkpoint') and source:
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        config_file = path / 'config.json'
        if config_file.is_file():
            saved = json.loads(config_file.read_text())
            # Multimodal configs commonly nest the language model dimensions.
            saved = saved.get('text_config', saved)
            aliases = {'hidden_size': ('hidden_size', 'n_embd'), 'num_layers': ('num_hidden_layers', 'n_layer'),
                       'n_heads': ('num_attention_heads', 'n_head'), 'vocab_size': ('vocab_size',),
                       'intermediate_size': ('intermediate_size',)}
            for target, candidates in aliases.items():
                for key in candidates:
                    if saved.get(key) is not None:
                        cfg[target] = saved[key]
                        break
            if cfg.get('hidden_size') and not saved.get('intermediate_size'):
                cfg['intermediate_size'] = 4 * cfg['hidden_size']
    for key, default in dict(hidden_size=512, num_layers=8, n_heads=8, vocab_size=50257).items():
        cfg.setdefault(key, default)
    cfg['seq_len'] = cfg.get('sft_max_seq_length', cfg.get('seq_len', 2048)) if cfg.get('stage') == 'sft' else cfg.get('seq_len', 2048)
    return cfg


@router.get("/chat-template")
def get_chat_template(model_path: str):
    """Return the chat_template from a model's tokenizer_config.json if present."""
    p = Path(model_path)
    if not p.exists():
        return {"has_template": False, "template": None, "error": "Path not found"}

    candidates = [p / "tokenizer_config.json", p / "chat_template.json"]
    for c in candidates:
        if c.exists():
            try:
                data = json.loads(c.read_text())
                tmpl = data.get("chat_template")
                if tmpl:
                    return {"has_template": True, "template": tmpl}
            except Exception:
                continue
    return {"has_template": False, "template": None}


def _detect_architecture(path: Path) -> str | None:
    config_file = path / "config.json"
    if config_file.exists():
        try:
            cfg = json.loads(config_file.read_text())
            return cfg.get("model_type") or cfg.get("architectures", [None])[0]
        except Exception:
            pass
    return None
