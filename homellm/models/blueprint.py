"""Blueprint schema and helpers for composable HomeLLM models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class BlockSpec(BaseModel):
    """Single block in a model blueprint."""

    id: str = Field(..., description="Unique block identifier")
    type: str = Field(..., description="Block type registered in BLOCK_REGISTRY")
    params: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[str] = Field(default_factory=list, description="IDs of input blocks (for graph mode)")


class Blueprint(BaseModel):
    """Top-level blueprint description."""

    model_type: str = Field(default="homellm_blueprint")
    vocab_size: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    max_position_embeddings: int = Field(default=2048, gt=0)
    blocks: List[BlockSpec] = Field(..., min_items=1)
    auto_project: bool = Field(default=True, description="Insert projections on hidden size mismatch")
    dtype: Optional[str] = Field(default=None, description="Optional dtype hint (float32/bfloat16/float16)")

    @validator("blocks")
    def check_unique_ids(cls, blocks: List[BlockSpec]) -> List[BlockSpec]:
        ids = [b.id for b in blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("Block ids must be unique")
        return blocks

    def hash(self) -> str:
        data = self.dict()
        blob = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def load(cls, path: str | Path) -> "Blueprint":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as e:
                raise RuntimeError("PyYAML is required to load YAML blueprints") from e
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.parse_obj(data)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as e:
                raise RuntimeError("PyYAML is required to save YAML blueprints") from e
            path.write_text(yaml.safe_dump(self.dict(), allow_unicode=True), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.dict(), indent=2, ensure_ascii=False), encoding="utf-8")


class BlueprintConfig(BaseModel):
    """Serializable config used inside HF PretrainedConfig."""

    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    auto_project: bool = True
    dtype: Optional[str] = None
    blueprint: Dict[str, Any]

    @classmethod
    def from_blueprint(cls, bp: Blueprint) -> "BlueprintConfig":
        return cls(
            vocab_size=bp.vocab_size,
            hidden_size=bp.hidden_size,
            max_position_embeddings=bp.max_position_embeddings,
            auto_project=bp.auto_project,
            dtype=bp.dtype,
            blueprint=bp.dict(),
        )
