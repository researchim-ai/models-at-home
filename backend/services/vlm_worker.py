"""Studio VLM entrypoint, keeping the external ML source tree unchanged."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--metrics', required=True)
    args = parser.parse_args()
    try:
        from backend.services.vlm_runtime import configure
        config = json.loads(Path(args.config).read_text())
        if config.get('training_backend') == 'unsloth':
            import unsloth  # Must patch Transformers/TRL before either is imported.
        configure(config)
        from homellm.training.vlm_common import MetricsLogger
        if config['stage'] == 'vlm_pretrain':
            from homellm.training.vlm_pretrain import run_vlm_pretrain as run
        elif config['stage'] == 'vlm_sft':
            from homellm.training.vlm_sft import run_vlm_sft as run
        elif config['stage'] == 'vlm_grpo':
            from backend.services.vlm_grpo import run
        else:
            raise ValueError(f"Unsupported VLM stage: {config['stage']}")
        run(config, MetricsLogger(Path(args.metrics)))
    except Exception as exc:
        import traceback
        from backend.paths import atomic_json
        path = Path(args.metrics)
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = {}
        atomic_json(path, {**previous, 'status': 'error', 'error': str(exc) + '\n' + traceback.format_exc()})
        raise



if __name__ == '__main__':
    main()
