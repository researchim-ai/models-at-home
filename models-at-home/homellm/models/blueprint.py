"""Blueprint schema and helpers for composable HomeLLM models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class TrainingConfig(BaseModel):
    """Training hyperparameters embedded in the blueprint."""
    
    optimizer: str = Field(default="adamw", description="Optimizer: adamw, adam, sgd, rmsprop, etc.")
    lr: float = Field(default=1e-3, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay (L2 regularization)")
    momentum: Optional[float] = Field(default=0.0, description="Momentum (for SGD)")
    loss_fn: str = Field(default="cross_entropy", description="Loss function")
    label_smoothing: float = Field(default=0.0, ge=0, le=1, description="Label smoothing for CrossEntropy")
    
    # Extra optimizer params
    betas: Optional[tuple] = Field(default=None, description="Betas for Adam/AdamW")
    eps: float = Field(default=1e-8, description="Epsilon for numerical stability")


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
    training: Optional[TrainingConfig] = Field(default=None, description="Training configuration")

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


def create_optimizer(model, training_config: TrainingConfig):
    """Create optimizer from TrainingConfig."""
    import torch
    
    params = model.parameters()
    opt_name = training_config.optimizer.lower()
    lr = training_config.lr
    wd = training_config.weight_decay
    
    if opt_name == "adamw":
        betas = training_config.betas or (0.9, 0.999)
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=training_config.eps, weight_decay=wd)
    elif opt_name == "adam":
        betas = training_config.betas or (0.9, 0.999)
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=training_config.eps, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=training_config.momentum or 0.0, weight_decay=wd)
    elif opt_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd, eps=training_config.eps)
    elif opt_name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=wd, eps=training_config.eps)
    elif opt_name == "adadelta":
        return torch.optim.Adadelta(params, lr=lr, weight_decay=wd, eps=training_config.eps)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def create_loss_fn(training_config: TrainingConfig):
    """Create loss function from TrainingConfig."""
    import torch.nn as nn
    
    loss_name = training_config.loss_fn.lower()
    
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    elif loss_name == "huber":
        return nn.HuberLoss()
    elif loss_name == "nll":
        return nn.NLLLoss()
    elif loss_name == "bce":
        return nn.BCELoss()
    elif loss_name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "kl_div":
        return nn.KLDivLoss(reduction="batchmean")
    elif loss_name == "cosine_embedding":
        return nn.CosineEmbeddingLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
