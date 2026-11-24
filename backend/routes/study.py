"""Study Center endpoints: list and serve markdown documents."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from backend.paths import get_project_root, safe_child

router = APIRouter()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
STUDY_DIR = PROJECT_ROOT / "study_materials"


@router.get("/index")
def document_index(lang: str = Query("en")):
    """List available study documents."""
    documents = []

    if lang not in ("en", "ru"):
        raise HTTPException(400, "Unsupported language")
    lang_dir = safe_child(STUDY_DIR, lang)
    if not lang_dir.exists():
        lang_dir = STUDY_DIR / "en"

    if lang_dir.exists():
        for f in sorted(lang_dir.rglob("*.md")):
            rel = f.relative_to(STUDY_DIR)
            documents.append({
                "name": f.stem.replace("_", " ").replace("-", " ").title(),
                "path": str(rel),
                "source": "local",
            })

    return {"documents": documents}


@router.get("/document")
def get_document(path: str = Query(...)):
    """Get content of a specific document."""
    full_path = safe_child(STUDY_DIR, path)
    if full_path.suffix != ".md":
        raise HTTPException(400, "Expected a Markdown document")

    if not full_path.is_file():
        raise HTTPException(404, "Document not found")

    content = full_path.read_text(encoding="utf-8")
    return {"content": content}
