"""Standalone action-policy boundaries and preprocessing invariants, no large weights."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from backend.routes.vla import VLATrainingRequest
from backend.services.vla_policies import POLICIES,python_for


@pytest.mark.parametrize('kind',['openvla','smolvla','act','pi0'])
def test_catalog_describes_distinct_objectives_and_sources(client,kind):
    row=next(p for p in client.get('/api/vla/policies').json()['policies'] if p['id']==kind)
    assert row['paper'].startswith('https://arxiv.org/') and row['objective']
    assert 'unsloth' not in row['methods']


@pytest.mark.parametrize('extra',[
    {'architecture':'openvla','model_path':'openvla/openvla-7b'},
    {'architecture':'openvla','model_path':'x','state_dim':0,'action_horizon':1,'action_dim':6},
    {'architecture':'smolvla','model_path':'x','action_type':'discrete'},
    {'architecture':'pi0','model_path':'x','state_dim':0},
    {'architecture':'act','tuning_method':'lora'},
    {'architecture':'smolvla','model_path':'x','training_backend':'unsloth'},
    {'architecture':'pi0','model_path':'x','mixed_precision':'fp16'},
])
def test_invalid_standalone_combinations_fail_before_worker(extra):
    with pytest.raises(ValueError):VLATrainingRequest(**extra)


def test_standalone_interpreters_are_isolated_and_missing_environment_is_actionable(monkeypatch):
    monkeypatch.setenv('MAH_OPENVLA_PYTHON','/missing/openvla/python')
    with pytest.raises(ValueError,match='MAH_OPENVLA_PYTHON'):python_for('openvla')
    monkeypatch.setenv('MAH_OPENVLA_PYTHON',sys.executable)
    monkeypatch.setenv('MAH_LEROBOT_PYTHON',sys.executable)
    for kind in POLICIES:assert python_for(kind)==sys.executable


def test_standalone_start_uses_selected_environment_and_shared_run_lifecycle(client,workspace,monkeypatch):
    from backend.routes import vla
    data=workspace/'datasets'/'actions.jsonl';data.write_text('{}\n')
    monkeypatch.setenv('MAH_LEROBOT_PYTHON',sys.executable)
    calls=[];monkeypatch.setattr(vla.processes,'launch',lambda run,command,**kw:calls.append((run,command,kw)))
    res=client.post('/api/vla/start',json={'architecture':'act','dataset_path':str(data),'mixed_precision':'no','device':'cpu'})
    assert res.status_code==200,res.text
    run,command,kw=calls[0];assert command[0]==sys.executable
    config=json.loads((run/'config.json').read_text());assert config['architecture']=='act' and config['stage']=='vla_sft'
    assert config['model_path']=='', 'Scratch ACT must not reinterpret an empty path as the workspace directory'
    assert kw['env']['CUDA_VISIBLE_DEVICES']=='0'


def test_pi0_does_not_learn_fake_terminal_padding():
    from backend.services.vla_policy_runtime import validate_rows
    data=SimpleNamespace(rows=[{'actions':np.zeros((1,7))}])
    with pytest.raises(ValueError,match='full action chunks'):validate_rows(data,{'architecture':'pi0','action_horizon':2})
    validate_rows(data,{'architecture':'smolvla','action_horizon':2})


def test_openvla_uses_upstream_bins_quantiles_and_gripper_contract():
    pytest.importorskip('torch')
    from backend.services.vla_openvla import encode_actions,decode_actions,normalize_actions,action_statistics
    values=np.array([[-2,-1,0,1,2,3,0],[2,1,1,2,3,4,1]],dtype=np.float32)
    st=action_statistics([{'actions':values}]);assert st['mask']==[True]*6+[False]
    normalized=normalize_actions(values,st)
    assert (np.abs(normalized[:,:6])<=1).all() and np.array_equal(normalized[:,-1],values[:,-1])
    ids=encode_actions(np.array([-1.,0.,1.]),32000)
    assert ids.tolist()==[31999,31872,31744]
    assert np.max(np.abs(decode_actions(ids,32000)-[-1,0,1]))<.008
    with pytest.raises(ValueError,match='gripper'):action_statistics([{'actions':values+2}])


def test_openvla_supervises_only_action_tokens_and_eos_without_truncating():
    torch=pytest.importorskip('torch')
    from backend.services.vla_openvla import Adapter
    # Processor fixture only: token supervision contract is independent of model weights.
    class Processor:
        tokenizer=SimpleNamespace(vocab_size=32000,pad_token_id=0,eos_token_id=2)
        def __call__(self,*args,**kwargs):return {'input_ids':torch.tensor([[1,123]]),'pixel_values':torch.zeros(1,3,32,32)}
    obj=Adapter.__new__(Adapter);torch.nn.Module.__init__(obj);obj.processor=Processor()
    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'i.png';Image.new('RGB',(32,32)).save(path)
        stats={'openvla':{'q01':[-1]*7,'q99':[1]*7,'mask':[True]*6+[False]}}
        inputs,_,targets,mask=obj.prepare([{'image':str(path),'instruction':'reach','actions':np.zeros((1,7))}],{'seq_len':32},stats)
        assert inputs['input_ids'][0,:3].tolist()==[1,123,29871]
        assert inputs['labels'][0,:3].tolist()==[-100]*3
        assert (inputs['labels'][0,3:]!=-100).all() and inputs['labels'][0,-1]==2
        assert targets.shape==(1,1,7) and mask.all()
        with pytest.raises(ValueError,match='no silent truncation'):obj.prepare([{'image':str(path),'instruction':'reach','actions':np.zeros((1,7))}],{'seq_len':4},stats)


def test_lora_resume_fingerprint_covers_dropout_and_adapter_targets():
    from backend.services.vla_training import signature
    base={'architecture':'openvla','lora_dropout':0.,'lora_target_modules':['q_proj']}
    assert signature(base)!=signature({**base,'lora_dropout':.1})
    assert signature(base)!=signature({**base,'lora_target_modules':['v_proj']})
