"""Dataset management endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.paths import get_project_root, safe_child

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
DATASET_DIR = PROJECT_ROOT / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


class DatasetDownloadRequest(BaseModel):
    repo_id: str
    subset: str = ""
    split: str = "train"
    save_as: str = ""
    limit: int | None = Field(default=None, ge=1, le=10000000)


@router.get("/local")
def list_datasets():
    """List all local datasets."""
    datasets = []
    if DATASET_DIR.exists():
        for f in sorted(DATASET_DIR.iterdir()):
            if f.name.startswith(".") or f.is_symlink():
                continue
            if f.is_file():
                ext = f.suffix.lstrip(".")
                if ext in ("jsonl", "txt", "parquet", "arrow", "csv"):
                    datasets.append({
                        "name": f.name,
                        "path": str(f),
                        "size_bytes": f.stat().st_size,
                        "format": ext,
                        "modified_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    })
            elif f.is_dir():
                size = sum(p.stat().st_size for p in f.rglob("*") if p.is_file() and not p.is_symlink())
                fmt = "jsonl"
                for p in f.rglob("*"):
                    if p.suffix in (".parquet",):
                        fmt = "parquet"
                        break
                    if p.suffix in (".arrow",):
                        fmt = "arrow"
                        break
                datasets.append({
                    "name": f.name,
                    "path": str(f),
                    "size_bytes": size,
                    "format": fmt,
                    "modified_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
    return {"datasets": datasets}


@router.post("/download")
def download_dataset(req: DatasetDownloadRequest):
    """Download a dataset from HuggingFace Datasets."""
    save_name = req.save_as or req.repo_id.split("/")[-1]
    save_path = safe_child(DATASET_DIR, save_name)

    try:
        from datasets import load_dataset
    except ImportError:
        raise HTTPException(500, "datasets library not installed")


    try:
        kwargs: dict = {"path": req.repo_id, "split": req.split}
        if req.subset:
            kwargs["name"] = req.subset
        if req.limit:
            kwargs["streaming"] = True

        ds = load_dataset(**kwargs)

        if req.limit and hasattr(ds, "take"):
            ds = ds.take(req.limit)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / "train.jsonl", "w") as f:
                for row in ds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            ds.save_to_disk(str(save_path))

        return {"success": True}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/{name:path}")
def delete_dataset(name: str):
    """Delete a local dataset (file or directory)."""
    import shutil
    path = safe_child(DATASET_DIR, name)
    if not path.exists():
        raise HTTPException(404, "Dataset not found")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return {"success": True}


@router.get("/preview/{name:path}")
def preview_dataset(name: str, limit: int = 10):
    """Preview first N rows of a dataset."""
    path = safe_child(DATASET_DIR, name)
    if not 1 <= limit <= 100:
        raise HTTPException(400, "limit must be between 1 and 100")
    if not path.exists():
        raise HTTPException(404, "Dataset not found")
    if path.is_dir():
        if (path / "state.json").is_file() or (path / "dataset_dict.json").is_file():
            try:
                from datasets import load_from_disk
                ds = load_from_disk(str(path))
                if hasattr(ds, "keys"):
                    ds = ds["train"] if "train" in ds else ds[next(iter(ds))]
                return {"rows": list(ds.select(range(min(limit, len(ds)))))}
            except ImportError:
                raise HTTPException(503, "datasets library is required to preview Arrow datasets")
        files = sorted(p for p in path.iterdir() if p.is_file() and not p.is_symlink()
                       and p.suffix in (".jsonl", ".txt", ".csv", ".parquet", ".arrow"))
        if not files:
            raise HTTPException(400, "No supported dataset files found")
        path = safe_child(path, files[0].name)
    rows = []
    if path.suffix in (".jsonl", ".txt"):
        with path.open(encoding="utf-8") as source:
            for i, line in enumerate(source):
                if i >= limit:
                    break
                if path.suffix == ".txt":
                    rows.append({"text": line.strip()})
                else:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    elif path.suffix == ".csv":
        import csv
        from itertools import islice
        with path.open(encoding="utf-8", newline="") as source:
            rows = list(islice(csv.DictReader(source), limit))
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
            batch = next(parquet.ParquetFile(path).iter_batches(batch_size=limit), None)
            rows = batch.to_pylist() if batch is not None else []
        except ImportError:
            raise HTTPException(503, "pyarrow is required to preview Parquet datasets")
    elif path.suffix == ".arrow":
        try:
            from datasets import Dataset
            ds = Dataset.from_file(str(path))
            rows = list(ds.select(range(min(limit, len(ds)))))
        except ImportError:
            raise HTTPException(503, "datasets library is required to preview Arrow datasets")
    else:
        raise HTTPException(400, "Unsupported dataset format")
    return {"rows": rows}
