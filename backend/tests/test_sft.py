from backend.services.sft import pack_records


def test_packing_preserves_tokens_and_masks_and_pads_tail():
    samples = [dict(input_ids=[1, 2, 0, 0], labels=[-100, 2, -100, -100], attention_mask=[1, 1, 0, 0]),
               dict(input_ids=[3, 4, 5, 6], labels=[-100, 4, 5, 6], attention_mask=[1, 1, 1, 1])]
    packed = list(pack_records(samples, 4, 0))
    assert packed == [dict(input_ids=[1, 2, 3, 4], labels=[-100, 2, -100, 4], attention_mask=[1, 1, 1, 1]),
                      dict(input_ids=[5, 6, 0, 0], labels=[5, 6, -100, -100], attention_mask=[1, 1, 0, 0])]


def test_packing_drops_blocks_with_no_next_token_targets():
    samples = [dict(input_ids=[1, 2], labels=[1, -100], attention_mask=[1, 1])]
    assert list(pack_records(samples, 2, 0)) == []
