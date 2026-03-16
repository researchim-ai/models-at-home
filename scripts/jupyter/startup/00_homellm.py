"""Auto-import core Models-at-Home symbols for Jupyter notebooks."""

from __future__ import annotations

import json
from pathlib import Path
import os
import shlex
import subprocess
import sys


PROJECT_ROOT = Path("/app")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

try:
    from homellm.models import HomeConfig, HomeForCausalLM
    from homellm.models.blueprint import Blueprint
    from homellm.models.blueprint_model import BlueprintForCausalLM, BlueprintLMConfig
    from homellm.models.home_model_moe import HomeMoEConfig, HomeMoEForCausalLM
    from homellm.models.memory_estimator import get_architecture_profile, estimate_memory_footprint
    from homellm.training.pretrain import StreamingTextDataset
    from homellm.training.sft import SFTDataset
    from homellm.training.optimizers import MagmaAdamW
except Exception as exc:  # pragma: no cover - runtime helper for notebooks
    print(f"[models-at-home] startup imports failed: {exc}")
else:
    print("[models-at-home] notebook helpers loaded:")
    print("  - HomeConfig, HomeForCausalLM | HomeMoEConfig, HomeMoEForCausalLM")
    print("  - Blueprint, BlueprintForCausalLM, BlueprintLMConfig")
    print("  - StreamingTextDataset, SFTDataset | MagmaAdamW")
    print("  - get_architecture_profile, estimate_memory_footprint")
    print("  - project_paths(), train_default_llama(...), ensure_pretrain_dataset(...)")


def project_paths() -> dict[str, Path]:
    """Convenience helper to discover mounted workspace paths."""
    return {
        "root": PROJECT_ROOT,
        "datasets": PROJECT_ROOT / "datasets",
        "models": PROJECT_ROOT / "models",
        "out": PROJECT_ROOT / "out",
        "runs": PROJECT_ROOT / ".runs",
        "blueprints": PROJECT_ROOT / "blueprints",
        "notebooks": PROJECT_ROOT / "notebooks",
    }


def train_default_llama(
    data_path: str = "/app/datasets/dataset.jsonl",
    output_dir: str = "/app/out/notebook_llama_default",
    tokenizer_path: str = "gpt2",
    hidden_size: int = 512,
    num_layers: int = 8,
    n_heads: int = 8,
    seq_len: int = 2048,
    batch_size: int = 8,
    gradient_accumulation: int = 4,
    epochs: int = 1,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
    optimizer: str = "adamw",
    bf16: bool = True,
    grad_checkpoint: bool = True,
    flash_attention: bool = True,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> str | int:
    """
    Run pretraining with default HomeModel (LLaMA-style) settings from notebooks.

    Returns:
        str: shell command when dry_run=True
        int: process return code when dry_run=False
    """
    cmd = [
        sys.executable,
        "-m",
        "homellm.training.pretrain",
        "--data_path",
        data_path,
        "--output_dir",
        output_dir,
        "--tokenizer_path",
        tokenizer_path,
        "--arch",
        "home",
        "--hidden_size",
        str(hidden_size),
        "--num_layers",
        str(num_layers),
        "--n_heads",
        str(n_heads),
        "--seq_len",
        str(seq_len),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation",
        str(gradient_accumulation),
        "--epochs",
        str(epochs),
        "--learning_rate",
        str(learning_rate),
        "--warmup_steps",
        str(warmup_steps),
        "--optimizer",
        optimizer,
    ]

    if bf16:
        cmd.append("--bf16")
    if grad_checkpoint:
        cmd.append("--grad_checkpoint")
    if flash_attention:
        cmd.append("--flash_attention")
    if extra_args:
        cmd.extend(extra_args)

    quoted = " ".join(shlex.quote(part) for part in cmd)
    print(f"[models-at-home] pretrain command:\n{quoted}")
    if dry_run:
        return quoted

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode != 0:
        print(f"[models-at-home] training finished with code {result.returncode}")
    return int(result.returncode)


def ensure_pretrain_dataset(
    path: str | Path = "/app/datasets/fineweb-2_train.jsonl",
    max_rows: int = 10_000,
    repo_id: str = "HuggingFaceFW/fineweb-edu",
    subset: str | None = "default",
    split: str = "train",
) -> str:
    """
    Если файл датасета не существует — скачивает с HuggingFace (streaming) и сохраняет.
    Подходит для pretrain (JSONL с полем «text» или аналогичным).

    Args:
        path: Путь к JSONL файлу (например /app/datasets/fineweb-2_train.jsonl).
        max_rows: Максимум строк при автозагрузке.
        repo_id: HF репозиторий (fineweb-edu меньше по размеру, fineweb-2 — полный).
        subset: Конфиг датасета (None или "default" для fineweb-edu).
        split: Сплит (train/validation).

    Returns:
        Путь к файлу (существующему или только что скачанному).
    """
    path = Path(path)
    if path.exists():
        return str(path)
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установите datasets: pip install datasets")
    path.parent.mkdir(parents=True, exist_ok=True)
    subset_arg = None if (not subset or subset.strip() == "" or subset.lower() == "default") else subset
    ds = load_dataset(repo_id, subset_arg, split=split, streaming=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
            if count % 1000 == 0:
                print(f"[models-at-home] downloaded {count} rows -> {path}")
    print(f"[models-at-home] saved {count} rows to {path}")
    return str(path)
