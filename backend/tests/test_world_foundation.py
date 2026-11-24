import json
from pathlib import Path
import pytest
from backend.routes import world_foundation as wf


def test_catalog_is_local_only(client):
    rows={r['id']:r for r in client.get('/api/world-models/foundation/models').json()['models']}
    assert set(rows)=={'diamond_csgo','diamond_atari','cosmos3_edge','cosmos3_nano','cosmos3_super'}
    assert all(r['local_weights'] and len(r['revision'])==40 for r in rows.values())
    assert rows['diamond_csgo']['provider']=='diamond'
    assert 'local_play' in rows['diamond_atari']['tasks']
    assert not any(r['training'] for r in rows.values())

@pytest.mark.parametrize('payload',[
    {'model_id':'genie4','task':'interactive_video'}, {'model_id':'matrix_game4','task':'video_render'},
    {'num_frames':50}, {'width':450}, {'gpu_ids':[0,1]},
    {'task':'forward_dynamics','domain':'umi','actions':[[0.]*7]*4},
    {'task':'forward_dynamics','domain':'umi','actions':[[0.]*10]*3},
    {'task':'forward_dynamics','domain':'umi','actions':[[float('inf')]*10]*4},
])
def test_contract_rejects_cloud_and_bad_shapes(payload):
    with pytest.raises(ValueError):wf.FoundationRequest(**payload)


def test_download_and_video_jobs(client,workspace,monkeypatch):
    monkeypatch.setattr(wf.processes,'launch',lambda *a,**k:None)
    r=client.post('/api/world-models/foundation/download',json={'model_id':'diamond_csgo','task':'local_play'}).json()
    run=workspace/'.runs'/r['run_id'];c=json.loads((run/'config.json').read_text())
    assert c['model_id']=='diamond_csgo' and c['operation']=='download'
    assert len(client.get('/api/world-models/foundation/jobs').json()['runs'])==1
    assert client.post(f"/api/training/runs/{r['run_id']}/continue",json={'max_steps':100}).status_code==400
    out=Path(c['output_dir']);(out/'generated.mp4').write_bytes(b'mp4-test')
    (out/'result.json').write_text(json.dumps({'video_path':str(out/'generated.mp4')}))
    assert client.get(f"/api/world-models/foundation/jobs/{r['run_id']}/video").content==b'mp4-test'
    assert client.get(f"/api/world-models/foundation/jobs/{r['run_id']}/result").status_code==200
    assert client.get('/api/world-models/foundation/jobs/unknown/result').status_code==404


def test_seed_not_embedded_in_listing(client,workspace,monkeypatch):
    monkeypatch.setattr(wf.processes,'launch',lambda *a,**k:None)
    r=client.post('/api/world-models/foundation/generate',json={'task':'image2video','prompt':'room','image_base64':'seed-bytes'}).json()
    c=json.loads((workspace/'.runs'/r['run_id']/'config.json').read_text())
    assert 'image_base64' not in c and Path(c['image_file']).read_text()=='seed-bytes'


def test_stop_refuses_other_stage(client,workspace):
    p=workspace/'.runs'/'other';p.mkdir();(p/'config.json').write_text(json.dumps({'stage':'vla_sft'}))
    assert client.post('/api/world-models/foundation/stop/other').status_code==400


def test_python_venv_symlink_preserved(workspace,monkeypatch):
    import sys
    p=workspace/'isolated'/'bin'/'python';p.parent.mkdir(parents=True);p.symlink_to(sys.executable)
    monkeypatch.setenv('MAH_WORLD_PYTHON',str(p));assert wf.python_command()==str(p)
    monkeypatch.delenv('MAH_WORLD_PYTHON');auto=workspace/'.venv-world'/'bin'/'python';auto.parent.mkdir(parents=True);auto.symlink_to(sys.executable)
    assert wf.python_command()==str(auto)
