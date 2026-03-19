"""Helpers for downloading GGUF models from Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_AGENT_GGUF_REPO = "unsloth/Qwen3.5-9B-GGUF"
DEFAULT_AGENT_GGUF_QUANT = "9B-UD-Q4_K_XL"


def get_agent_repo_id() -> str:
    return os.environ.get("MAH_AGENT_MODEL_REPO", DEFAULT_AGENT_GGUF_REPO).strip() or DEFAULT_AGENT_GGUF_REPO


def get_agent_quant() -> str:
    return os.environ.get("MAH_AGENT_MODEL_QUANT", DEFAULT_AGENT_GGUF_QUANT).strip() or DEFAULT_AGENT_GGUF_QUANT


def normalize_quant_name(value: str) -> str:
    return value.lower().replace("-", "_")


def find_matching_gguf_filename(repo_id: str, quant: str) -> str:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model")
    quant_norm = normalize_quant_name(quant)
    ggufs = [file for file in files if file.endswith(".gguf")]
    for file in ggufs:
        if quant_norm in normalize_quant_name(Path(file).name):
            return Path(file).name
    raise FileNotFoundError(
        f"GGUF quant '{quant}' не найден в {repo_id}. Доступно: {', '.join(Path(file).name for file in ggufs[:50])}"
    )


def download_gguf_to_models_dir(models_dir: Path, repo_id: str, quant: str, filename: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    models_dir.mkdir(parents=True, exist_ok=True)
    target_filename = filename or find_matching_gguf_filename(repo_id=repo_id, quant=quant)
    target_path = models_dir / Path(target_filename).name
    if target_path.exists():
        return target_path

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=target_filename,
        repo_type="model",
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded_path)
