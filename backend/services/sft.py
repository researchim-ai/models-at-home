"""Apply Studio SFT controls while retaining the runtime's formatting/sharding."""


def pack_records(records, length, pad_id):
    ids, labels = [], []
    for record in records:
        for token, label, mask in zip(record['input_ids'], record['labels'], record['attention_mask']):
            if mask:
                ids.append(token)
                labels.append(label)
        while len(ids) >= length:
            if any(label != -100 for label in labels[1:length]):
                yield dict(input_ids=ids[:length], labels=labels[:length], attention_mask=[1] * length)
            ids, labels = ids[length:], labels[length:]
    if len(ids) > 1 and any(label != -100 for label in labels[1:]):
        padding = length - len(ids)
        yield dict(input_ids=ids + [pad_id] * padding, labels=labels + [-100] * padding,
                   attention_mask=[1] * len(ids) + [0] * padding)


def dataset_class(base, config):
    class StudioSFTDataset(base):
        def __iter__(self):
            def records():
                for sample in super(StudioSFTDataset, self).__iter__():
                    if not config.get('sft_train_on_completions_only', False):
                        sample['labels'] = sample['input_ids'].clone().masked_fill(sample['attention_mask'] == 0, -100)
                    yield sample
            if not config.get('sft_packing'):
                yield from records()
                return
            import torch
            lists = ({key: value.tolist() for key, value in sample.items()} for sample in records())
            for sample in pack_records(lists, self.seq_len, self.tokenizer.pad_token_id):
                yield {key: torch.tensor(value, dtype=torch.long) for key, value in sample.items()}
    return StudioSFTDataset
