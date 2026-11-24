import pytest
from backend.routes import world_play as route


def test_play_validation_and_stale_sessions(client):
    assert client.post('/api/world-models/play/start',json={'model_id':'genie4'}).status_code==422
    assert client.post('/api/world-models/play/start',json={'model_id':'diamond_atari','game':'Pong'}).status_code==400
    assert client.post('/api/world-models/play/step',json={'session_id':'a'*32,'keys':['exec']}).status_code==422
    with pytest.raises(ValueError):route.StepRequest(session_id='a'*32,mouse_x=float('nan'))
    assert client.post('/api/world-models/play/step',json={'session_id':'a'*32}).status_code==410
    assert client.post('/api/world-models/play/reset',json={'session_id':'a'*32,'spawn':99}).status_code==422
    assert client.get('/api/world-models/play/status').json()['session_id'] is None


def test_play_uses_embedded_worker_contract(client,monkeypatch):
    captured=[]
    def start(python,cwd,env,c):captured.append(c);return {'session_id':'b'*32,'frame':'data:image/png;base64,x','step':0}
    monkeypatch.setattr(route.service,'start',start)
    r=client.post('/api/world-models/play/start',json={'model_id':'diamond_csgo','spawn':3,'gpu_ids':[0]})
    assert r.status_code==200 and captured[0]['spawn']==3
    assert captured[0]['upsample'] is True and 'runtime_path' not in captured[0]
