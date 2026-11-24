import copy
import json
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('homellm')
from backend.services.blueprint_analysis import analyze,synthetic_probe
from backend.services.blueprint_multimodal import normalize,BlueprintVisionLM


def spec():
    value=json.loads((Path(__file__).resolve().parents[2]/'examples/model-builder/looped-transformer.json').read_text())
    value['max_position_embeddings']=128
    return value


def test_catalog_has_primary_references_and_honest_primitive_attribution(client):
    blocks=client.get('/api/blueprints/blocks').json()['blocks']
    assert all(row.get('research',{}).get('summary',{}).get('ru') for row in blocks)
    assert all(ref['url'].startswith('https://') for row in blocks for ref in row['research']['references'])
    rms=next(row for row in blocks if row['type']=='rmsnorm')['research']
    assert any('1910.07467' in ref['url'] and ref['year']==2019 for ref in rms['references'])
    add=next(row for row in blocks if row['type']=='add')['research']
    assert all(ref['relationship']=='implementation' for ref in add['references'])


@pytest.mark.parametrize('task',['llm','vlm','vla'])
def test_structural_analysis_matches_the_actual_task_model_and_tracks_shapes(task):
    from backend.services.blueprint_analysis import LabProfile,build
    value=spec(); profile=LabProfile(task=task,seq_len=8)
    result=analyze(value,profile)
    model,_,effective,_=build(value,profile)
    assert result['parameters']==sum(p.numel() for p in model.parameters())
    assert result['blocks'][0]['input_shape']==[1,8+(16 if task!='llm' else 0)]
    assert result['contracts']['vision_tokens']==(16 if task!='llm' else 0)
    assert result['contracts']['vocabulary']==(100 if task=='llm' else 259)
    assert result['source_hash']==analyze(value)['source_hash']
    assert result['fp32_adam_parameter_floor_mib']==result['parameters']*16/2**20
    assert result['dense_macs_estimate']>0


def test_more_loops_change_compute_but_not_the_number_of_shared_parameters():
    value=spec(); first=analyze(value)
    value['blocks'][2]['params']['iterations']=8
    value['blocks'][2]['params']['eval_iterations']=12
    other=analyze(value)
    assert first['parameters']==other['parameters']
    assert first['dense_macs_estimate']<other['dense_macs_estimate']<other['eval_dense_macs_estimate']
    core=next(row for row in other['blocks'] if row['path']=='core.body[0]')
    assert core['train_calls_max']==8


@pytest.mark.parametrize('task',['llm','vlm','vla'])
def test_real_probe_reports_updates_activations_gradients_and_causality(task):
    torch.set_num_threads(2)
    result=synthetic_probe(spec(),{'task':task,'seq_len':8,'steps':1})
    assert result['optimizer_steps']==1 and result['weight_update_norm']>0
    assert result['causality']['passed']
    assert all(row['last_activation']['finite_fraction']==1 for row in result['blocks'])
    assert any(row['gradient_norm'] is not None and row['gradient_norm']>0 for row in result['blocks'])
    assert result['operators'] and result['environment']['synthetic_data_version']==1
    if task!='llm':assert result['image_sensitivity']>0


def test_groupnorm_future_leak_is_detected_instead_of_claiming_a_causal_model():
    value=spec();value['blocks']=[value['blocks'][0],{'id':'norm','type':'groupnorm','params':{'num_groups':4}}]
    result=synthetic_probe(value,{'seq_len':8,'steps':1})
    assert not result['causality']['passed']
    assert any('Causality check failed' in warning for warning in result['warnings'])
    assert not analyze(value)['capabilities']['vlm']['supported']


def test_probe_does_not_execute_custom_operators(client):
    value=spec();value['blocks'][1]={'id':'prelude','type':'custom_op','params':{'path':'must_not_import.Module','out_features':32}}
    response=client.post('/api/blueprints/research/probe',json={'blueprint':value})
    assert response.status_code==400 and 'does not execute custom code' in response.text


def test_big_architecture_is_rejected_before_allocating_probe_weights(client):
    value=spec();value['vocab_size']=200000;value['blocks'][0]['params']['vocab_size']=200000
    response=client.post('/api/blueprints/research/probe',json={'blueprint':value})
    assert response.status_code==400 and 'budget exceeded' in response.text


def test_multimodal_adapter_preserves_source_and_requires_cross_token_mixing():
    value=spec(); untouched=copy.deepcopy(value)
    config=normalize({'model_blueprint_spec':value,'tuning_method':'full'},'.',task='vlm')
    assert value==untouched and config['source_blueprint_spec']['vocab_size']==100
    assert config['model_blueprint_spec']['vocab_size']==259
    value['blocks']=[value['blocks'][0]]
    with pytest.raises(ValueError,match='cross-token mixer'):
        normalize({'model_blueprint_spec':value,'tuning_method':'full'},'.',task='vlm')


def test_byte_sft_masks_and_image_conditioning_have_real_gradients(tmp_path):
    from backend.services.blueprint_vlm import encode_messages
    messages=[{'role':'user','content':'color?'},{'role':'assistant','content':'red'}]
    ids,labels=encode_messages(messages)
    assert sum(value!=-100 for value in labels)==4  # three answer bytes and EOS
    assert labels[-1]==2
    c=normalize({'model_blueprint_spec':spec(),'tuning_method':'full'},'.',task='vlm')
    model=BlueprintVisionLM(c)
    tokens=torch.tensor([ids]); targets=torch.tensor([labels])
    result=model(torch.rand(1,3,32,32),tokens,torch.ones_like(tokens),labels=targets)
    result['loss'].backward()
    assert model.core.patch_projection.weight.grad.abs().sum()>0
    assert model.core.graph.modules_dict['core'].body.layers[0].attn.q_proj.weight.grad.abs().sum()>0


def test_runtime_signals_capture_actual_batch_without_retaining_a_tensor_graph():
    from backend.services.research_signals import RuntimeSignals
    from backend.tests.test_looped_blueprints import lm
    value=spec();model=lm(value)
    observer=RuntimeSignals(model.model,model.config.blueprint)
    observer.begin(1)
    tokens=torch.tensor([[1,2,3,4]])
    model(tokens,labels=tokens).loss.backward()
    packet=observer.collect()
    assert packet['step']==1 and packet['blocks']
    assert all(isinstance(row['shape'],list) for row in packet['blocks'])
    assert any(row['gradient_rms_sampled'] is not None for row in packet['blocks'])
    json.dumps(packet,allow_nan=False)
    observer.begin(2)
    assert observer.collect() is None
    observer.close()


def test_unsupported_native_vlm_modes_fail_before_launch(client,workspace):
    data=workspace/'tiny.jsonl';data.write_text('{}\n')
    common={'architecture':'blueprint_vlm','model_blueprint_spec':spec(),'dataset_path':str(data),'use_lora':False,'tuning_method':'full','max_steps':2}
    for update in [{'stage':'vlm_grpo'},{'training_backend':'unsloth'},{'tuning_method':'qlora'}]:
        response=client.post('/api/vlm/start',json={**common,**update})
        assert response.status_code==400,response.text
    assert not list((workspace/'.runs').glob('*/config.json'))


def test_synthetic_inputs_stay_fixed_when_architecture_initialization_changes():
    torch.set_num_threads(2)
    value=spec()
    first=synthetic_probe(value,{'seq_len':8,'steps':1})
    value['blocks'][1]['params']['num_kv_heads']=4
    second=synthetic_probe(value,{'seq_len':8,'steps':1})
    assert first['data_fingerprint']==second['data_fingerprint']
    assert first['source_hash']!=second['source_hash']
