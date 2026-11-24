"""Bounded shape-matched native training trial using the actual training loop.
All generated data/checkpoints live in a temporary directory; no user run is modified.
"""
import contextlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def benchmark(c):
    import torch
    from PIL import Image
    import psutil
    from backend.services.world_training import train,device
    torch.set_num_threads(min(4,torch.get_num_threads()))
    dev=device(c)
    if dev.type=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='mah-world-capacity-') as temp:
        root=Path(temp)
        length,size=c['sequence_length'],c['image_size']
        state=[[0.]*c['state_dim'] for _ in range(length+1)]
        if c['action_type']=='continuous':action=[0.]*c['action_dim']
        elif c['action_type']=='discrete':action=0
        else:action={'continuous':[0.]*c['action_dim'],'discrete':[0]*len(c['discrete_classes'])}
        for split in ('train','val'):
            frames=[]
            for t in range(length+1):
                path=root/f'{split}-{t}.png';Image.new('RGB',(size,size),(30+t%200,70,110)).save(path);frames.append(str(path))
            rows=[dict(episode_id=f'{split}-{i}',frames=frames,actions=[action]*length,
                **({'states':state} if c['state_dim'] else {}),rewards=[0.]*length,dones=[False]*(length-1)+[True])
                for i in range(c['batch_size'] if split=='val' else 1)]
            (root/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        config={**c,'dataset_path':str(root/'train.jsonl'),'val_data_path':str(root/'val.jsonl'),
            'max_steps':5,'save_every':5,'eval_every':5,'output_dir':str(root/'out'),
            'resume_from_checkpoint':'','resumed_from_run':''}
        with contextlib.redirect_stdout(sys.stderr):metrics=train(config,root/'metrics.json')
        if dev.type=='cuda':torch.cuda.synchronize()
        rss=psutil.Process().memory_info().rss
        try:
            import resource
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss=max(rss,peak if sys.platform=='darwin' else peak*1024)
        except ImportError:pass
        return dict(status='ok',device=str(dev),gpu_name=torch.cuda.get_device_name(0) if dev.type=='cuda' else None,
            torch_version=torch.__version__,optimizer='AdamW FP32 master parameters',warmup_steps=2,measured_steps=3,
            train_step_seconds=statistics.median(metrics['train_step_seconds_history'][2:]),
            validation_batch_seconds=metrics['validation_seconds_history'][-1],checkpoint_seconds=metrics['checkpoint_seconds_history'][-1],
            final_export_seconds=metrics['final_export_seconds'],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if dev.type=='cuda' else 0,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if dev.type=='cuda' else 0,peak_rss_gib=rss/2**30,
            wall_seconds=time.perf_counter()-started,measured_at=time.time(),data_basis='synthetic shapes; actual forward/backward/optimizer/validation/checkpoint')


if __name__=='__main__':
    try:
        result=benchmark(json.load(sys.stdin))
    except Exception as exc:
        import torch
        oom=isinstance(exc,(torch.OutOfMemoryError,MemoryError)) or 'out of memory' in str(exc).lower()
        result=dict(status='oom' if oom else 'error',error=str(exc))
    print(json.dumps(result,allow_nan=False))
