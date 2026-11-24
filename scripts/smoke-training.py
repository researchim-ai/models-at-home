"""Offline GPU smoke: tiny local tokenizer/data -> train -> resume -> inference.

Requires the installed homellm runtime and its ML dependencies. All files live
in a temporary directory. No model, tokenizer or dataset is downloaded.
"""
import json
import os
import sys
import tempfile
import time
import argparse
import socket
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--architecture', choices=['home', 'gpt2', 'home_moe', 'blueprint'], default='home')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mah-training-smoke-") as temporary:
        root = Path(temporary)
        os.environ.update(MODELS_AT_HOME_ROOT=temporary, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from fastapi.testclient import TestClient
        from backend.api import app
        from backend.services import processes
        from backend.routes import chat

        vocabulary = {word: i for i, word in enumerate(['[UNK]', '[PAD]', '[EOS]', 'hello', 'tiny', 'world', 'training', 'model', 'test'])}
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
        tokenizer_dir = root / 'tokenizer'
        tokenizer.save_pretrained(tokenizer_dir)
        # A valid HF config next to the tokenizer used to override Home selection.
        (tokenizer_dir / 'config.json').write_text(json.dumps({'model_type': 'gpt2', 'n_embd': 64, 'n_layer': 2, 'n_head': 4}))
        if args.multi_gpu:
            (root / 'configs').mkdir()
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            (root / 'configs' / 'accelerate_multi_gpu.yaml').write_text(
                f'compute_environment: LOCAL_MACHINE\ndistributed_type: MULTI_GPU\nmixed_precision: "no"\n'
                f'num_processes: 2\nnum_machines: 1\nmain_process_port: {port}\ndynamo_backend: "no"\n')
        data = root / 'datasets' / 'tiny.jsonl'
        data.parent.mkdir(exist_ok=True)
        data.write_text(''.join(json.dumps({'text': 'hello tiny world training model test ' * 32}) + '\n' for _ in range(80)))
        client = TestClient(app)

        def completed(run_id):
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                result = client.get(f'/api/training/runs/{run_id}').json()
                if result['status'] == 'completed' and not processes.is_running(root / '.runs' / run_id):
                    return result
                if result['status'] == 'error':
                    raise AssertionError((root / '.runs' / run_id / 'stderr.log').read_text()[-5000:])
                time.sleep(0.25)
            raise AssertionError('Tiny training did not finish within 120 seconds')

        try:
            settings = {
                'stage': 'pretrain', 'model_type': args.architecture, 'model_id': str(tokenizer_dir),
                'tokenizer_path': str(tokenizer_dir), 'dataset_path': str(data),
                'hidden_size': 32, 'num_layers': 1, 'n_heads': 4, 'seq_len': 64,
                'num_experts': 2, 'num_experts_per_tok': 1, 'intermediate_size': 64,
                'batch_size': 1, 'gradient_accumulation': 1, 'max_steps': 2,
                'warmup_steps': 0, 'save_every': 1, 'log_every': 1, 'mixed_precision': 'no',
                'use_liger': False, 'liger_fused_ce': False, 'grad_checkpoint': False,
                'use_flash_attention': False, 'export_on_checkpoint': False,
                'gpu_ids': [0, 1] if args.multi_gpu else [1],
                'parallel_mode': 'multi_gpu' if args.multi_gpu else 'single',
            }
            if args.architecture == 'blueprint':
                blueprint = {'hidden_size': 32, 'vocab_size': 50257, 'max_position_embeddings': 64,
                    'blocks': [
                        {'id': 'embedding', 'type': 'token_embedding', 'params': {'hidden_size': 32, 'vocab_size': 50257}},
                        {'id': 'positions', 'type': 'positional_embedding'},
                        {'id': 'projection', 'type': 'linear', 'params': {'out_features': 32, 'bias': True}},
                        {'id': 'attention', 'type': 'causal_self_attention', 'params': {'num_heads': 4, 'use_rope': True}},
                        {'id': 'norm', 'type': 'layernorm'},
                    ]}
                response = client.post('/api/blueprints/save', json={'name': 'tiny', 'blueprint': blueprint})
                assert response.status_code == 200, response.text
                settings.update(model_blueprint=response.json()['path'], grad_checkpoint=True)
            response = client.post('/api/training/start', json=settings)
            assert response.status_code == 200, response.text
            first_id = response.json()['run_id']
            first = completed(first_id)
            assert first['current_step'] == 2, first
            print('PASS local pretrain: step 2', flush=True)
            response = client.post(f'/api/training/runs/{first_id}/continue', json={'max_steps': 4})
            assert response.status_code == 200, response.text
            resumed_id = response.json()['run_id']
            resumed = completed(resumed_id)
            assert resumed['current_step'] == 4, resumed
            assert resumed['config']['resume_from_checkpoint'].endswith('checkpoint_step2')
            assert resumed['config']['output_dir'] != first['config']['output_dir']
            print('PASS checkpoint resume: step 2 -> 4, separate output directories', flush=True)
            final_model = str(Path(resumed['config']['output_dir']) / 'final_model')
            saved = json.loads((Path(final_model) / 'config.json').read_text())
            assert saved['model_type'] == {'home': 'homellm', 'gpt2': 'gpt2_home', 'home_moe': 'homellm_moe', 'blueprint': 'homellm_blueprint'}[args.architecture], saved
            assert saved['hidden_size'] == 32, saved
            if args.architecture != 'blueprint':
                assert saved['intermediate_size'] == 64, saved
            else:
                assert saved['vocab_size'] == len(vocabulary)
                assert saved['blueprint']['blocks'][0]['params']['vocab_size'] == len(vocabulary)
            response = client.post('/api/chat/generate', json={
                'model_path': final_model, 'messages': [{'role': 'user', 'content': 'hello tiny world'}],
                'max_tokens': 4, 'temperature': 0, 'stream': False,
            })
            assert response.status_code == 200, response.text
            assert isinstance(response.json()['text'], str)
            print('PASS inference from exported Home model', flush=True)
            if args.architecture == 'blueprint':
                import torch
                loaded = chat._loaded_backend['model']
                with torch.inference_mode():
                    first = loaded.model(torch.tensor([[3, 4]], device=loaded.device))
                    second = loaded.model(torch.tensor([[4, 3]], device=loaded.device))
                    assert not torch.allclose(first, second), 'Positional embedding discarded the token representation'
                assert isinstance(loaded.model.modules_dict['projection'], torch.nn.Linear)
                print('PASS positional encoding preserves tokens; same-width linear is trainable', flush=True)
            chat.unload_model()
            for fmt in ('chat', 'instruct'):
                sft_data = root / 'datasets' / f'{fmt}.jsonl'
                row = {'messages': [{'role': 'user', 'content': 'hello tiny world'},
                                    {'role': 'assistant', 'content': 'training model test'}]} if fmt == 'chat' else {
                                        'instruction': 'hello tiny world', 'output': 'training model test'}
                sft_data.write_text((json.dumps(row) + '\n') * 40)
                response = client.post('/api/training/start', json={**settings,
                    'stage': 'sft', 'model_path': final_model, 'dataset_path': str(sft_data),
                    'tokenizer_path': final_model, 'sft_max_seq_length': 32,
                    'sft_data_format': fmt, 'sft_packing': fmt == 'chat',
                    'sft_train_on_completions_only': fmt == 'instruct',
                })
                assert response.status_code == 200, response.text
                result = completed(response.json()['run_id'])
                assert result['current_step'] == 2, result
                print(f'PASS SFT {fmt}, packing={fmt == "chat"}, assistant-only={fmt == "instruct"}', flush=True)
        finally:
            processes.shutdown()
            chat.unload_model()


if __name__ == '__main__':
    main()
