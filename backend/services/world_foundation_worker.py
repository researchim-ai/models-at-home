"""Managed official world foundation generation and weight download entrypoint."""
import argparse
import asyncio
import contextlib
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'download', 'generate'], required=True)
    parser.add_argument('--config'); parser.add_argument('--metrics')
    args = parser.parse_args()
    from backend.paths import atomic_json
    from backend.services.world_foundation_runtime import preflight, download, cosmos, save_video
    from backend.services.world_catalog import model_info
    metrics = {'status': 'initializing', 'current_step': 0}
    def progress(**fields):
        metrics.update(fields)
        if args.metrics:
            atomic_json(Path(args.metrics), metrics)
    try:
        c = json.loads(Path(args.config).read_text()) if args.config else json.load(sys.stdin)
        if args.mode == 'preflight':
            with contextlib.redirect_stdout(sys.stderr):
                result = preflight(c)
            print(json.dumps(result, allow_nan=False)); return
        output = Path(c['output_dir']); output.mkdir(parents=True, exist_ok=True)
        c['run_dir'] = str(Path(args.config).parent)
        provider = model_info(c['model_id'])['provider']
        if args.mode == 'download':
            result = download(c, progress)
        elif provider == 'diamond':
            from backend.services.world_local import LocalWorld
            c['denoising_steps']=min(c['inference_steps'],10)
            engine=LocalWorld(c)
            from PIL import Image
            from io import BytesIO
            import base64
            frames=[]
            for i in range(c['num_frames']):
                value=engine.reset() if i==0 else engine.step(dict(keys=c['keys'],mouse_x=0,mouse_y=0,action=0))
                frames.append(Image.open(BytesIO(base64.b64decode(value['frame'].split(',')[-1]))).convert('RGB'))
                progress(status='generating',current_step=i+1)
            result=save_video(frames,output,c['fps'])
        else:
            result = save_video(cosmos(c, progress), output, c['fps'])
        result.update(model_id=c['model_id'], task=c['task'])
        atomic_json(output / 'result.json', result)
        progress(status='completed', result_path=str(output / 'result.json'))
    except Exception as exc:
        if args.mode != 'preflight':
            progress(status='error', error=str(exc))
        traceback.print_exc(); sys.exit(1)


if __name__ == '__main__':
    main()
