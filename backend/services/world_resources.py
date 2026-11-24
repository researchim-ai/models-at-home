"""Conservative world-model capacity planning; measured native training calibration.
Estimates are explicitly distinct from successful allocation tests and unsupported finetuning.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
import psutil

GIB=2**30
_CACHE={}
_DATA={}
_lock=threading.RLock()
benchmark_lock=threading.Lock()
TTL=900


def _number(value):
    try:
        n=float(value)
        return n if math.isfinite(n) and n>=0 else None
    except (ValueError,TypeError):return None


def cgroup_available(root=Path('/sys/fs/cgroup')):
    """Use the tightest finite cgroup limit/headroom, including ancestors."""
    paths=[root]
    if root==Path('/sys/fs/cgroup'):
        try:
            for row in Path('/proc/self/cgroup').read_text().splitlines():
                _,controllers,relative=row.split(':',2)
                base=root if not controllers else root/'memory' if 'memory' in controllers.split(',') else None
                if base is not None:
                    current=base/relative.lstrip('/')
                    if current.is_relative_to(base):
                        paths.extend([current,*[p for p in current.parents if p.is_relative_to(base)]])
        except (OSError,ValueError):pass
    caps=[];limits=[]
    for path in paths:
        for limit_name,used_name in [('memory.max','memory.current'),('memory.limit_in_bytes','memory.usage_in_bytes'),('memory/memory.limit_in_bytes','memory/memory.usage_in_bytes')]:
            try:
                limit=int((path/limit_name).read_text());used=int((path/used_name).read_text())
                if 0<limit<2**60:caps.append(max(0,limit-used));limits.append(limit)
            except (OSError,ValueError):pass
    return (min(caps),min(limits)) if caps else (None,None)


def cpu_threads():
    # The native trainer and the benchmark both cap Torch intra-op threads at four.
    try:return max(1,min(4,len(os.sched_getaffinity(0))))
    except AttributeError:return min(4,psutil.cpu_count() or 1)


def _disk(path):
    p=Path(path).expanduser()
    while not p.exists() and p.parent!=p:p=p.parent
    try:return shutil.disk_usage(p).free/GIB
    except OSError:return None


def hardware(root):
    ram=psutil.virtual_memory();cap,limit=cgroup_available()
    gpus=[];probe='unavailable';error=None
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        if result.returncode==0:
            probe='ok'
            for row in csv.reader(result.stdout.splitlines()):
                if len(row)!=7:continue
                try:index=int(row[0].strip())
                except ValueError:continue
                total,used,free,util=map(_number,row[3:])
                gpus.append(dict(id=index,name=row[1].strip(),uuid=row[2].strip(),total_gib=total/1024 if total is not None else None,
                    used_gib=used/1024 if used is not None else None,free_gib=free/1024 if free is not None else None,utilization=util))
        else:error='nvidia-smi не смог опросить GPU'
    except (OSError,subprocess.TimeoutExpired):error='Опрос NVIDIA недоступен; это не доказательство отсутствия CUDA в ML Python'
    cache=os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE') or str(Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'hub')
    return dict(cpu_name=platform.processor() or platform.machine(),cpu_threads=cpu_threads(),
        ram_total_gib=min(ram.total,limit)/GIB if limit is not None else ram.total/GIB,
        ram_available_gib=min(ram.available,cap)/GIB if cap is not None else ram.available/GIB,
        cgroup_limited=limit is not None,gpus=gpus,gpu_probe=probe,gpu_probe_detail=error,
        disk_free_gib=_disk(root),cache_disk_free_gib=_disk(cache),sampled_at=time.time())


def native_parameters(c):
    h,z,s=c['hidden_dim'],c['latent_dim'],c['state_dim']
    a=c['action_dim']+(sum(c['discrete_classes']) if c['action_type']=='hybrid' else 0)
    m=2 if c['architecture']=='rssm' else 1
    return (784+8224+513*h + h*(h+s+1) + 3*h*(z+a)+3*h*h+6*h
            +(h+1)*z*m+(2*h+1)*z*m+2048*(h+z+1)+8208+771
            +(h+z+1)*s+2*(h+z+1))


def _identity(c,hw):
    fallback=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    selected=c.get('gpu_ids') or [int(fallback) if fallback.isdigit() else 0]
    gpu=next((g for g in hw['gpus'] if g['id']==selected[0]),None)
    device=c.get('device','auto')
    if device=='auto':device='cuda' if gpu is not None else ('cpu' if hw['gpu_probe']=='ok' else 'unknown')
    return device,gpu


def profile_key(model_id,operation,c,hw):
    device,gpu=_identity(c,hw)
    fields=('architecture','action_type','action_dim','discrete_classes','state_dim','image_size','sequence_length','latent_dim','hidden_dim','batch_size','mixed_precision','upsample','denoising_steps','num_frames','height','width','cpu_offload','task','game')
    return hashlib.sha256(json.dumps(dict(version=1,model=model_id,operation=operation,device=device,
        hardware=gpu['uuid'] if device=='cuda' and gpu else hw['cpu_name']+str(hw['cpu_threads']),
        gpu_ids=c.get('gpu_ids') or [0],cpu_threads=hw['cpu_threads'],recipe={k:c[k] for k in fields if k in c}),sort_keys=True).encode()).hexdigest()


def _dataset_key(c):
    out=[]
    for key in ('dataset_path','val_data_path'):
        value=c.get(key,'')
        if not value:out.append(None);continue
        try:
            p=Path(value);stat=p.stat();out.append((str(p.resolve()),stat.st_size,stat.st_mtime_ns))
        except OSError:return None
    return str((out,c.get("sequence_length"),c.get("action_type"),c.get("action_dim"),c.get("state_dim"),c.get("discrete_classes")))


def remember_data(c,summary):
    key=_dataset_key(c)
    if key:
        with _lock:
            _DATA[key]=(time.monotonic(),summary)
            while len(_DATA)>32:_DATA.pop(next(iter(_DATA)))


def store_measurement(key,value):
    with _lock:
        _CACHE[key]=(time.monotonic(),value)
        while len(_CACHE)>32:_CACHE.pop(next(iter(_CACHE)))


def _measurement(key):
    with _lock:
        cached=_CACHE.get(key)
        return cached[1] if cached and time.monotonic()-cached[0]<TTL else None


def _counts(c):
    with _lock:
        data=_DATA.get(_dataset_key(c))
        return data[1] if data and time.monotonic()-data[0]<TTL else None


def _events(total,start,every):
    return max(0,total//every-start//every)+(1 if total>start and total%every else 0)


def duration(c,measured,summary):
    if not measured or measured.get('status')!='ok':return None
    start=0
    if c.get('resume_from_checkpoint'):
        try:start=int(json.loads((Path(c['resume_from_checkpoint'])/'complete.json').read_text())['step'])
        except (OSError,ValueError,KeyError,TypeError):return None
    steps=max(0,c['max_steps']-start)
    train=steps*measured['train_step_seconds']
    checkpoints=_events(c['max_steps'],start,c['save_every'])
    save=checkpoints*measured['checkpoint_seconds']+measured['final_export_seconds']
    evaluation=0.;missing=False
    if c.get('val_data_path'):
        if summary is None:missing=True
        else:
            batches=math.ceil(summary['validation_windows']/c['batch_size'])
            evaluation=_events(c['max_steps'],start,c['eval_every'])*batches*measured['validation_batch_seconds']
    total=train+save+evaluation
    return dict(remaining_steps=steps,training_seconds=train,validation_seconds=None if missing else evaluation,
        checkpoint_seconds=save,total_seconds_low=total*.8,total_seconds_high=None if missing else total*1.8,
        basis='Короткая проба на синтетических данных той же формы; скорость реального чтения данных может отличаться',
        validation_included=not missing,note='Сначала проверьте данные: время validation ещё не учтено' if missing else 'Включены validation и сохранения; подготовка полного датасета, скорость целевого диска и будущая нагрузка не измерены')


def estimate(model_id,operation,c,hw,use_measurement=True):
    device,gpu=_identity(c,hw)
    native=model_id in ('latent_gru','rssm')
    supported=native and operation=='train' or not native and operation=='inference'
    hints=[];reasons=[];unknown=[]
    parameters=0;disk=None;train_lower=None
    if native:
        parameters=native_parameters(c)
        b,t,size=c['batch_size'],c['sequence_length'],c['image_size']
        # FP32 parameters/gradients + 2 FP32 AdamW moments. Activations are bounded heuristically.
        states=parameters*16/GIB
        pixels=b*(t+1)*(12*size*size+16*(size//2)**2+32*(size//4)**2+10000)
        factor=9 if c['architecture']=='rssm' else 5
        activation=pixels*(2 if device=='cuda' and c['mixed_precision']!='no' else 4)*factor/GIB
        memory=[states+activation*.6+.25,states+activation*1.7+.7]
        host_batch=b*(t+1)*3*size*size*4/GIB
        ram=[.7+parameters*4/GIB+host_batch*2,1.5+states+host_batch*4] if device=='cuda' else [.7+memory[0],1.2+memory[1]]
        checkpoint=parameters*12/GIB+.001
        disk=(math.ceil(c['max_steps']/c['save_every'])*checkpoint+parameters*4/GIB)*1.2
        hints+=['Уменьшите batch, длину окна или размер кадра при нехватке памяти.','Несколько GPU не суммируются: текущий trainer использует одну выбранную GPU.']
    else:
        parameters={'diamond_csgo':381638406 if c.get('upsample',True) else 330000000,'diamond_atari':4405955,
            'cosmos3_edge':3369657024,'cosmos3_nano':16000000000,'cosmos3_super':64000000000}[model_id]
        train_lower=parameters*16/GIB
        if model_id=='diamond_atari':memory=[.25,.9];ram=[.45,1.2];disk=.12
        elif model_id=='diamond_csgo':memory=[1.9,3.5] if c.get('upsample',True) else [1.6,2.7];ram=[2.,4.5];disk=3.1
        else:
            weights={'cosmos3_edge':8.2,'cosmos3_nano':36.,'cosmos3_super':130.}[model_id]
            work=(c.get('height',256)*c.get('width',448)/(256*448))*max(1,c.get('num_frames',49)/49)
            memory=[(3. if c.get('cpu_offload',True) else weights)+.3*work,(8. if c.get('cpu_offload',True) else weights)+2.*work]
            ram=[weights+2,weights*1.5+8] if c.get('cpu_offload',True) else [3.,8.]
            disk=weights*2.2
            hints+=['Диапазон Cosmos приблизительный: память guardrail, attention/VAE workspace и конкретный checkpoint требуют проверки реальным запуском.','CPU offload переносит веса в RAM, но не делает вычисления полностью CPU.']
        if model_id.startswith('cosmos3') and device=='cpu':reasons.append('Cosmos в Studio требует CUDA с BF16')
        if operation=='train':
            memory=[None,None];ram=[None,None];disk=None
            reasons.append('Дообучение pretrained-модели не подключено. Реальный пик памяти и время тюнинга ещё не измеряются.')
        hints+=['Нижняя граница Full/AdamW — веса, градиенты и moments в FP32; активации и дополнительная AMP-копия сюда не входят.']
    key=profile_key(model_id,operation,c,hw)
    measured=_measurement(key) if use_measurement else None
    if native and measured and measured.get('status')=='ok':
        if device=='cuda':memory=[measured['peak_reserved_gib']+.35,measured['peak_reserved_gib']*1.2+.75]
        ram=[measured['peak_rss_gib'],measured['peak_rss_gib']*1.25+.25]
        if device=='cpu':memory=list(ram)
    if device=='cuda':
        if gpu is None:
            (reasons if hw['gpu_probe']=='ok' else unknown).append('Выбранная GPU не обнаружена' if hw['gpu_probe']=='ok' else 'Не удалось измерить свободную VRAM выбранной GPU')
        elif memory[0] is None:
            pass
        elif gpu['free_gib'] is not None:
            if memory[0]>gpu['free_gib']:reasons.append(f"Недостаточно свободной VRAM GPU {gpu['id']}: минимум {memory[0]:.2f} GiB, доступно {gpu['free_gib']:.2f} GiB")
            elif memory[1]+max(.5,(gpu['total_gib'] or 0)*.05)>gpu['free_gib']:unknown.append('Малый запас VRAM: уменьшите параметры или выполните пробу')
        else:unknown.append('Свободная VRAM неизвестна')
        if gpu and (gpu.get('utilization') or 0)>70:hints.append('GPU сейчас нагружена; время может быть больше результата короткой пробы.')
    if device=='unknown':unknown.append('Устройство auto не определено; выберите CPU/CUDA или выполните пробу в ML Python')
    if ram[0] is not None and ram[0]>hw['ram_available_gib']:reasons.append('Недостаточно доступной RAM (учтён лимит контейнера)')
    elif ram[1] is not None and ram[1]>hw['ram_available_gib']*.8:unknown.append('Запас доступной RAM небольшой')
    disk_free=hw['disk_free_gib'] if native else hw['cache_disk_free_gib']
    if not native and operation=='inference' and c.get('model_path') and Path(c['model_path']).expanduser().is_dir():
        disk=.5 if model_id.startswith('cosmos3') else .05
        disk_free=hw['disk_free_gib']
        hints.append('Локальные веса уже существуют: диск оценён только под результаты; полнота checkpoint проверяется при загрузке.')
    if disk is not None and disk_free is not None and disk>disk_free:
        (reasons if native else unknown).append('Недостаточно диска для всех checkpoints' if native else 'Мало диска для загрузки weights/cache; уже скачанные веса не требуют этого места повторно')
    if native and measured and measured.get('status')=='oom':reasons.append('Последняя проба этой конфигурации завершилась OOM; освободите память и повторите пробу или уменьшите параметры')
    status='unsupported' if not supported else 'blocked' if reasons else 'unknown' if device=='unknown' else 'risk' if unknown else 'likely_fit'
    return dict(model_id=model_id,operation=operation,supported=supported,training_supported=native,status=status,
        device=device,gpu=gpu,hardware=hw,parameters=parameters,memory_gib=memory,ram_gib=ram,disk_required_gib=disk,disk_available_gib=disk_free,
        training_state_lower_bound_gib=train_lower,measurement=measured,eta=duration(c,measured,_counts(c)) if native else None,
        reasons=reasons,warnings=unknown,hints=hints,profile_key=key,benchmark_supported=native and operation=='train',
        benchmark_running=benchmark_lock.locked(),basis='Измерено на этой машине + запас' if measured and measured.get('status')=='ok' else 'Предварительная оценка, не гарантия отсутствия OOM')


def enforce(model_id,operation,c,root):
    if benchmark_lock.locked():raise ValueError('Идёт ресурсная проба; дождитесь её завершения')
    target=Path(c.get('output_dir') or root).expanduser()
    if not target.is_absolute():target=Path(root)/target
    report=estimate(model_id,operation,c,hardware(target))
    if report['status']=='blocked':raise ValueError('Ресурсы: '+'; '.join(report['reasons']))
    return report
