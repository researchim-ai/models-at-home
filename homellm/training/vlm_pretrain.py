from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from .vlm_common import MetricsLogger
from .vlm_sft import run_vlm_sft

logger = logging.getLogger(__name__)


def run_vlm_pretrain(config: Dict[str, Any], metrics_logger: MetricsLogger) -> None:
    """Home-scale continued multimodal pretraining/alignment."""
    if config.get("stage") not in (None, "vlm_pretrain"):
        raise ValueError(f"vlm_pretrain worker supports only stage=vlm_pretrain, got {config.get('stage')}")

    pretrain_config = dict(config)
    pretrain_config["stage"] = "vlm_sft"
    pretrain_config.setdefault("assistant_only_loss", False)
    pretrain_config.setdefault("caption_prompt", "Describe this image in detail.")
    pretrain_config.setdefault("learning_rate", 1e-5)
    pretrain_config.setdefault("freeze_vision_tower", True)
    pretrain_config.setdefault("gradient_checkpointing", True)
    pretrain_config.setdefault("tuning_method", "qlora")
    metrics_logger.update(stage="vlm_pretrain", status="starting")
    run_vlm_sft(pretrain_config, metrics_logger)
    metrics_logger.update(stage="vlm_pretrain")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    metrics_path = Path(args.metrics)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    metrics_logger = MetricsLogger(metrics_path, enabled=True)
    run_vlm_pretrain(config, metrics_logger)


if __name__ == "__main__":
    main()
