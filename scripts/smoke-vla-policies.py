"""Real ACT or reduced SmolVLA lifecycle through API and isolated LeRobot worker."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--architecture',choices=['act','smolvla'],default='act')
    parser.add_argument('--model-path',default='')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mah-act-smoke-') as tmp:
        root=Path(tmp);os.environ['MODELS_AT_HOME_ROOT']=tmp
        from PIL import Image
        from fastapi.testclient import TestClient
        from backend.api import app
        from backend.services import processes
        from safetensors.torch import load_file
        import torch
        rows=[]
        for i in range(6):
            Image.new('RGB',(32,32),'red' if i%2 else 'blue').save(root/f'{i}.png')
            rows.append({'image':f'{i}.png','instruction':'reach target','state':[float(i)/6],
                'actions':[[.1*i,-.1*i]]*(1 if i==0 else 2)})
        train=root/'train.jsonl';train.write_text('\n'.join(json.dumps(r) for r in rows[:4]))
        val=root/'val.jsonl';val.write_text('\n'.join(json.dumps(r) for r in rows[4:]))
        settings={'architecture':args.architecture,'model_path':args.model_path,'dataset_path':str(train),'val_data_path':str(val),'action_dim':2,'state_dim':1,'action_horizon':2,
            'device':'cpu','mixed_precision':'no','batch_size':2,'max_steps':2,'save_every':1,'eval_every':1,'image_size':32,
            'act_dim_model':32,'act_encoder_layers':1,'learning_rate':.001,'tuning_method':'full'}
        def done(client,run):
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                value=client.get(f'/api/training/runs/{run}').json()
                if value['status']=='error':raise AssertionError((root/'.runs'/run/'stderr.log').read_text()[-7000:])
                if value['status']=='completed' and not processes.is_running(root/'.runs'/run):return value
                time.sleep(.15)
            raise AssertionError('ACT worker timed out')
        with TestClient(app) as client:
            checked=client.post('/api/vla/preflight',json=settings)
            assert checked.status_code==200 and checked.json()['ready'],checked.text
            started=client.post('/api/vla/start',json=settings);assert started.status_code==200,started.text
            run=started.json()['run_id'];result=done(client,run);output=Path(result['config']['output_dir'])
            old=load_file(str(output/'checkpoint-1'/'policy'/'model.safetensors'))
            new=load_file(str(output/'checkpoint-2'/'policy'/'model.safetensors'))
            assert any(not torch.equal(old[k],new[k]) for k in old),'ACT did not update weights'
            print(f'PASS actual LeRobot {args.architecture} optimizer updates, own objective, holdout validation and checkpoint',flush=True)
            # Resume from step 1 to step 2 with the same total plan: exact RNG/optimizer/data replay.
            resumed=client.post('/api/vla/start',json={**settings,'resume_from_checkpoint':str(output/'checkpoint-1')})
            assert resumed.status_code==200,resumed.text
            replay=done(client,resumed.json()['run_id']);replayed=Path(replay['config']['output_dir'])
            actual=load_file(str(replayed/'final_model'/'policy'/'model.safetensors'))
            expected=load_file(str(output/'final_model'/'policy'/'model.safetensors'))
            assert all(torch.equal(expected[k],actual[k]) for k in expected),'ACT resumed weights diverged'
            pred=client.post('/api/vla/predict',json={'model_path':str(output/'final_model'),'instruction':'reach target','state':[.3],
                'image_base64':base64.b64encode((root/'0.png').read_bytes()).decode(),'device':'cpu'})
            assert pred.status_code==200,pred.text
            assert len(pred.json()['actions'])==2 and len(pred.json()['actions'][0])==2
            assert client.get(f'/api/training/runs/{run}/metrics').json()['metrics']['action_mae']>=0
            print(f'PASS {args.architecture} exact resume, exported official policy reload and action-chunk inference',flush=True)

if __name__=='__main__':main()
