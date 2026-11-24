import pytest
from backend.services import world_resources as r
from backend.routes.world_models import WorldRequest


def hw(free=20):
    return dict(cpu_name='test',cpu_threads=4,gpu_probe='ok',gpus=[dict(id=0,name='A',uuid='a',total_gib=24,free_gib=free,utilization=0),dict(id=1,name='B',uuid='b',total_gib=24,free_gib=23,utilization=0)],ram_available_gib=64,ram_total_gib=128,disk_free_gib=100,cache_disk_free_gib=100)


def recipe(**values):return WorldRequest(**values).model_dump()


@pytest.mark.parametrize('architecture,state,kind,classes',[('latent_gru',0,'continuous',[]),('rssm',3,'hybrid',[3,4]),('rssm',0,'discrete',[])])
def test_exact_parameter_count(architecture,state,kind,classes):
    from backend.services.world_model import WorldModel
    c=recipe(architecture=architecture,state_dim=state,action_type=kind,discrete_classes=classes)
    assert r.native_parameters(c)==sum(p.numel() for p in WorldModel(c).parameters())


def test_free_selected_gpu_not_total_or_sum():
    c=recipe(device='cuda',gpu_ids=[0]);h=hw(.1)
    assert r.estimate('latent_gru','train',c,h)['status']=='blocked'
    c['gpu_ids']=[1]
    assert r.estimate('latent_gru','train',c,h)['status']=='likely_fit'
    c['gpu_ids']=[2]
    assert r.estimate('latent_gru','train',c,h)['status']=='blocked'


def test_unknown_hardware_and_scaling():
    c=recipe();h=hw();h.update(gpus=[],gpu_probe='unavailable')
    assert r.estimate('rssm','train',c,h)['status']=='unknown'
    c['device']='cpu'
    small=r.estimate('latent_gru','train',c,h)
    c.update(batch_size=32,sequence_length=16,image_size=128)
    large=r.estimate('latent_gru','train',c,h)
    assert large['memory_gib'][1]>small['memory_gib'][1]
    h['ram_available_gib']=.1
    assert r.estimate('latent_gru','train',c,h)['status']=='blocked'


def test_pretrained_tuning_is_not_claimed():
    result=r.estimate('diamond_csgo','train',{},hw())
    assert result['status']=='unsupported' and result['eta'] is None
    assert result['memory_gib']==[None,None] and result['training_state_lower_bound_gib']>5


def test_measurement_reuse_invalidation_and_eta(tmp_path):
    c=recipe(device='cpu',max_steps=100,save_every=25,eval_every=10,val_data_path=str(tmp_path/'val.jsonl'))
    key=r.profile_key('rssm','train',c,hw())
    measured=dict(status='ok',train_step_seconds=.1,checkpoint_seconds=1,final_export_seconds=1,validation_batch_seconds=.2,peak_rss_gib=1)
    r.store_measurement(key,measured)
    c['max_steps']=200
    report=r.estimate('rssm','train',c,hw())
    assert report['measurement']==measured and report['eta']['total_seconds_high'] is None
    eta=r.duration(c,measured,dict(validation_windows=16))
    assert eta['validation_seconds']==8 and eta['checkpoint_seconds']==9
    c['batch_size']=16
    assert r.profile_key('rssm','train',c,hw())!=key


def test_container_limit(tmp_path):
    (tmp_path/'memory.max').write_text('1024');(tmp_path/'memory.current').write_text('768')
    assert r.cgroup_available(tmp_path)==(256,1024)
    (tmp_path/'memory.max').write_text('max')
    assert r.cgroup_available(tmp_path)==(None,None)


def test_api_benchmark_and_start_gate(client,monkeypatch,workspace):
    from backend.routes import world_resources as route,world_models
    monkeypatch.setattr(r,'hardware',lambda root:hw())
    monkeypatch.setattr(route,'probe',lambda c:dict(status='oom',error='out of memory'))
    response=client.post('/api/world-models/resources/benchmark',json=dict(model_id='latent_gru',config=dict(device='cuda')))
    assert response.status_code==200 and response.json()['status']=='blocked'
    assert client.post('/api/world-models/resources/benchmark',json=dict(model_id='diamond_atari')).status_code==400
    data=workspace/'data.jsonl';data.write_text('{}')
    monkeypatch.setattr(r,'hardware',lambda root:hw(.01))
    def never(*a,**kw):pytest.fail('must not launch on insufficient VRAM')
    monkeypatch.setattr(world_models.processes,'launch',never)
    result=client.post('/api/world-models/start',json=dict(dataset_path=str(data),device='cuda'))
    assert result.status_code==400 and 'VRAM' in result.text
    r._CACHE.clear()


def test_validation_counts_follow_window_shape(tmp_path):
    path=tmp_path/'data.jsonl';path.write_text('{}')
    c=recipe(dataset_path=str(path),sequence_length=4)
    r.remember_data(c,dict(validation_windows=10))
    assert r._counts(c)['validation_windows']==10
    c['sequence_length']=8
    assert r._counts(c) is None


def test_existing_local_weights_do_not_need_downloading_twice(tmp_path):
    h=hw();h['cache_disk_free_gib']=.001
    report=r.estimate('diamond_csgo','inference',dict(device='cpu',model_path=str(tmp_path)),h)
    assert report['status']=='likely_fit' and report['disk_required_gib']==.05
