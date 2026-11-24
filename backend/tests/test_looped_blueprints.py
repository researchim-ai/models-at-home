"""Real CPU autograd checks, not mocked model-training assertions."""
import copy
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("homellm")
from homellm.models.blocks import get_block
from homellm.models.blocks.recurrent import auxiliary_losses
from backend.services.blueprints import validate
from backend.services.blueprint_model import StudioBlueprintConfig, StudioBlueprintForCausalLM


def body():
    return [{'type': 'llama_block', 'params': {'num_heads': 4, 'num_kv_heads': 2, 'intermediate_size': 32, 'use_liger': False}}]


def loop(**kwargs):
    return get_block('loop')({'iterations': 3, 'body': body(), **kwargs}, 16, False)[0]


def spec(**kwargs):
    return {'hidden_size': 16, 'vocab_size': 20, 'max_position_embeddings': 16, 'blocks': [
        {'id': 'embed', 'type': 'token_embedding'},
        {'id': 'core', 'type': 'loop', 'params': {'iterations': 3, 'body': body(), **kwargs}},
        {'id': 'norm', 'type': 'rmsnorm', 'params': {'use_liger': False}},
    ]}


def lm(value):
    value = validate(value)
    return StudioBlueprintForCausalLM(StudioBlueprintConfig(vocab_size=value['vocab_size'], hidden_size=value['hidden_size'],
        max_position_embeddings=value['max_position_embeddings'], blueprint=value, tie_word_embeddings=False, use_cache=False, studio_graph_version=1))


def test_weight_count_does_not_grow_with_loop_depth():
    once, repeated = loop(iterations=1), loop(iterations=9)
    assert sum(p.numel() for p in once.parameters()) == sum(p.numel() for p in repeated.parameters())
    independent, _ = get_block('repeater')({'num_repeats': 3, 'block_type': 'llama_block', 'block_params': body()[0]['params']}, 16, False)
    assert sum(p.numel() for p in independent.parameters()) == 3 * sum(p.numel() for p in once.parameters())
    assert independent.layers[0].attn.q_proj.weight is not independent.layers[1].attn.q_proj.weight


def test_full_bptt_matches_manual_unroll_outputs_and_gradients():
    torch.manual_seed(3)
    shared = loop()
    manual = copy.deepcopy(shared)
    x = torch.randn(2, 7, 16, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    expected = reference
    for _ in range(3):
        expected = manual.body(expected)
    actual = shared(x)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward(); expected.square().mean().backward()
    torch.testing.assert_close(x.grad, reference.grad)
    for p, q in zip(shared.parameters(), manual.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize('injection', ['none', 'add', 'concat'])
def test_checkpoint_recomputes_the_same_step_and_gradients(injection):
    torch.manual_seed(4)
    regular = loop(input_injection=injection, step_embedding='sinusoidal')
    checked = copy.deepcopy(regular); checked.checkpoint = True
    a = torch.randn(1, 5, 16, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    result = regular(a); restored = checked(b)
    torch.testing.assert_close(result, restored)
    result.square().mean().backward(); restored.square().mean().backward()
    torch.testing.assert_close(a.grad, b.grad)
    for p, q in zip(regular.parameters(), checked.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_truncated_bptt_keeps_the_input_injection_gradient():
    model = loop(input_injection='concat', state_init='zeros', gradient_steps=1, checkpoint=True)
    x = torch.randn(1, 5, 16, requires_grad=True)
    model(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_random_depth_is_seeded_and_eval_depth_is_deterministic():
    model = loop(iterations=5, min_iterations=2, random_depth=True, eval_iterations=7)
    x = torch.randn(1, 3, 16)
    def depths():
        torch.manual_seed(9)
        result = []
        for _ in range(12):
            model(x); result.append(model.last_iterations)
        return result
    first = depths()
    assert first == depths() and min(first) >= 2 and max(first) <= 5 and len(set(first)) > 1
    model.eval()
    model(x); assert model.last_iterations == 7
    model(x, loop_iterations=2); assert model.last_iterations == 2


@pytest.mark.parametrize('kind', ['llama_block', 'causal_self_attention', 'attention'])
def test_causality_and_left_padding_are_preserved(kind):
    params = body()[0]['params'] if kind == 'llama_block' else {'num_heads': 4}
    layer, _ = get_block(kind)(params, 16, False)
    layer.eval()
    x = torch.randn(2, 6, 16, requires_grad=True)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    result = layer(x, attention_mask=mask)
    changed = x.detach().clone(); changed[:, 4:] += 10; changed[0, :2] -= 20
    other = layer(changed, attention_mask=mask)
    torch.testing.assert_close(result[0, 2:4], other[0, 2:4])
    torch.testing.assert_close(result[1, :4], other[1, :4])
    result.square().mean().backward()
    assert torch.isfinite(result).all() and torch.isfinite(x.grad).all()


def test_residual_rezero_and_arbitrary_sequential_body():
    residual, _ = get_block('residual')({'scale': 0, 'learnable_scale': True,
        'body': [{'type': 'rmsnorm', 'params': {'use_liger': False}}, {'type': 'mlp', 'params': {'intermediate_size': 32}}]}, 16, False)
    x = torch.randn(1, 3, 16)
    torch.testing.assert_close(residual(x), x)
    residual(x).square().sum().backward()
    assert residual.scale.grad is not None
    repeated, _ = get_block('repeater')({'num_repeats': 2, 'block_type': 'linear', 'block_params': {'out_features': 16}}, 16, False)
    assert repeated(x).shape == x.shape
    old_residual, _ = get_block('residual_mlp')({}, 16, False)
    assert old_residual(x, attention_mask=torch.ones(1, 3)).shape == x.shape


def test_moe_auxiliary_losses_include_the_recurrent_body_without_double_counting():
    model = loop(body=[{'type': 'moe', 'params': {'num_experts': 2, 'num_select': 1}}])
    model(torch.randn(2, 4, 16)).square().mean().backward()
    assert len(auxiliary_losses(model)) == 1
    assert torch.isfinite(model.aux_loss) and model.aux_loss.requires_grad


def test_real_optimizer_save_reload_generation_and_override(tmp_path):
    torch.manual_seed(8)
    torch.set_num_threads(2)
    model = lm(spec(input_injection='concat', state_init='zeros', checkpoint=True))
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]] * 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
    first = model(tokens, labels=tokens).loss.item()
    original = model.model.modules_dict['core'].body.layers[0].attn.q_proj.weight.detach().clone()
    for _ in range(12):
        optimizer.zero_grad(); output = model(tokens, labels=tokens)
        assert torch.isfinite(output.loss)
        output.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
    model.eval()
    assert model(tokens, labels=tokens).loss.item() < first
    assert not torch.equal(original, model.model.modules_dict['core'].body.layers[0].attn.q_proj.weight)
    model.save_pretrained(tmp_path)
    restored = StudioBlueprintForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(model(tokens).logits, restored(tokens).logits)
    generated = restored.generate(tokens[:1], max_new_tokens=2, do_sample=False, use_cache=False, pad_token_id=0, eos_token_id=None, loop_iterations=5)
    assert generated.shape[1] == 8
    assert restored.model.modules_dict['core'].last_iterations == 5
    restored(tokens, loop_iterations=5)
    assert restored.model.modules_dict['core'].last_iterations == 5


def test_graph_concat_gate_and_projection_have_correct_dimensions():
    value = spec(); value['blocks'] = [value['blocks'][0],
        {'id': 'value', 'type': 'linear', 'inputs': ['embed'], 'params': {'out_features': 8}},
        {'id': 'gate', 'type': 'activation', 'inputs': ['value'], 'params': {'activation': 'sigmoid'}},
        {'id': 'gated', 'type': 'multiply', 'inputs': ['value', 'gate']},
        {'id': 'concat', 'type': 'concat', 'inputs': ['embed', 'gated']},
        {'id': 'project', 'type': 'linear', 'params': {'in_features': 24, 'out_features': 16}},
    ]
    result = lm(value)(torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]]))
    assert result.logits.shape == (1, 3, 20) and torch.isfinite(result.loss)
    result.loss.backward()


@pytest.mark.parametrize('params', [
    {'iterations': 0}, {'iterations': True}, {'min_iterations': 5}, {'eval_iterations': 257},
    {'gradient_steps': 4}, {'body': []}, {'body': [{'type': 'linear', 'params': {'out_features': 8}}]},
    {'input_injection': 'unknown'}, {'body': [{'type': 'llama_block', 'params': {'num_heads': 3}}]},
    {'body': [{'type': 'causal_self_attention', 'params': {'num_heads': 4, 'num_kv_heads': 3}}]},
    {'body': [{'type': 'token_embedding'}]}, {'checkpoint': 'true'}, {'state_init': 'zeros', 'input_injection': 'none'},
])
def test_invalid_loop_never_reaches_training(client, workspace, params):
    value = spec(**params)
    response = client.post('/api/blueprints/save', json={'name': 'invalid-loop', 'blueprint': value})
    assert response.status_code == 400, response.text
    assert not (workspace / 'blueprints' / 'invalid-loop.json').exists()


def test_api_catalog_and_meta_count_match_actual_model(client):
    assert {'loop', 'residual', 'repeater', 'moe'} <= {x['type'] for x in client.get('/api/blueprints/blocks').json()['blocks']}
    value = spec(input_injection='concat')
    response = client.post('/api/blueprints/analyze', json=value)
    assert response.status_code == 200, response.text
    assert response.json()['parameters'] == sum(p.numel() for p in lm(value).parameters())
    assert response.json()['max_block_applications'] == 6

@pytest.mark.parametrize('checkpoint', [False, True])
def test_truncated_amp_does_not_reuse_detached_weight_casts(checkpoint):
    model = loop(input_injection='concat', state_init='zeros', gradient_steps=1, checkpoint=checkpoint)
    x = torch.randn(1, 5, 16, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f'AMP lost the gradient for {name}'
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize('checkpoint', [False, True])
def test_recurrent_moe_lm_auxiliary_loss_backpropagates(checkpoint):
    model = lm(spec(checkpoint=checkpoint, body=[{'type': 'moe', 'params': {'num_experts': 2, 'num_select': 1}}]))
    tokens = torch.tensor([[1, 2, 3, 4]])
    output = model(tokens, labels=tokens)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    gate = model.model.modules_dict['core'].body.layers[0].gate
    assert gate.weight.grad is not None and torch.isfinite(gate.weight.grad).all()


def test_runtime_head_tying_preserves_an_explicit_output_width():
    from homellm.models.blueprint_model import BlueprintForCausalLM, BlueprintLMConfig
    value = spec(); value['blocks'] = [value['blocks'][0], {'id': 'project', 'type': 'linear', 'params': {'out_features': 8}}]
    model = BlueprintForCausalLM(BlueprintLMConfig(vocab_size=20, hidden_size=16, blueprint=value, tie_word_embeddings=True))
    assert model.lm_head.weight.shape == (20, 8)
    assert model(torch.tensor([[1, 2, 3]])).logits.shape == (1, 3, 20)


def test_existing_advanced_blocks_remain_available_without_executing_them_in_analysis(client):
    from backend.services.blueprint_catalog import SUPPORTED
    from homellm.models.blocks import BLOCK_REGISTRY
    assert set(BLOCK_REGISTRY) == SUPPORTED
    value = spec(); value['blocks'][1] = {'id': 'custom', 'type': 'custom_op', 'params': {'path': 'not_imported.Module', 'out_features': 16}}
    assert client.post('/api/blueprints/save', json={'name': 'custom', 'blueprint': value}).status_code == 200
    response = client.post('/api/blueprints/analyze', json=value)
    assert response.status_code == 400 and 'does not execute custom code' in response.text
