"""Bridge Studio's config-file protocol to the legacy GRPO JSON CLI."""
import argparse
import json
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    config["ui_run_dir"] = str(config_path.parent)
    if config.get("training_backend") != "unsloth" and config.get("grpo_algorithm") == "dr_grpo":
        config["grpo_algorithm"] = "drgrpo"
    # The original CLI only accepts --config_json, not --config <filename>.
    # Assigning argv in-process also avoids OS command-line length limits.
    sys.path.insert(0, str(Path.cwd()))
    sys.argv = ["homellm.training.rl.train_rl", "--config_json", json.dumps(config)]
    runpy.run_module("homellm.training.rl.train_rl", run_name="__main__")


if __name__ == "__main__":
    main()
