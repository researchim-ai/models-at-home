"""Native action-chunk policies, trained with our supervised loop."""
from pathlib import Path
import json
import torch
from torch import nn


class ActionPolicy(nn.Module):
    def __init__(self, config, backbone=None):
        super().__init__()
        self.config, self.backbone = config, backbone
        self.output_dim = config['action_dim'] + (sum(config['discrete_classes']) if config['action_type'] == 'hybrid' else 0)
        if backbone is None:
            grid = config.get('vision_grid', 1)  # Old exports used global pooling.
            self.vision = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.GELU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((grid, grid)), nn.Flatten())
            self.text_embedding = nn.Embedding(257, 32, padding_idx=0)
            self.text_encoder = nn.GRU(32, 64, batch_first=True)
            width = 32 * grid * grid + 64
        else:
            width = getattr(backbone.config, 'text_config', backbone.config).hidden_size
        self.head = nn.Sequential(nn.Linear(width + config['state_dim'], config['hidden_dim']), nn.GELU(),
            nn.Linear(config['hidden_dim'], config['action_horizon'] * self.output_dim))

    def forward(self, inputs, states):
        if self.backbone is None:
            tokens = [list(text.encode('utf-8')) for text in inputs['instructions']]
            lengths = torch.tensor([len(row) for row in tokens])
            ids = torch.zeros(len(tokens), int(lengths.max()), dtype=torch.long, device=states.device)
            for index, row in enumerate(tokens):
                ids[index, :len(row)] = torch.tensor(row, device=states.device) + 1
            packed = nn.utils.rnn.pack_padded_sequence(self.text_embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, text = self.text_encoder(packed)
            features = torch.cat([self.vision(inputs['images']), text[-1]], dim=-1)
        else:
            options = {'logits_to_keep': 1} if self.backbone.config.model_type in ('qwen3_vl', 'qwen3_5') else {}
            output = self.backbone(**inputs, output_hidden_states=True, use_cache=False, **options)
            hidden = output.hidden_states[-1]
            # Last non-padding token, valid for either padding side.
            positions = torch.arange(hidden.shape[1], device=hidden.device).expand(hidden.shape[0], -1)
            indices = positions.masked_fill(inputs['attention_mask'] == 0, -1).max(-1).values
            features = hidden[torch.arange(len(hidden), device=hidden.device), indices]
        features = torch.cat([features.float(), states.float()], dim=-1)
        return self.head(features).view(-1, self.config['action_horizon'], self.output_dim)


def create(config):
    if config['architecture'] == 'native_vla':
        return ActionPolicy(config), None
    from homellm.training.vlm_common import load_vlm_processor
    from backend.services.vlm_runtime import load_model, apply_adapters
    processor = load_vlm_processor(config['model_path'], config)
    backbone = apply_adapters(load_model(config['model_path'], config, 'cuda'), config)
    if config['gradient_checkpointing']:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    backbone.config.use_cache = False
    return ActionPolicy(config, backbone), processor


def save(directory, model, processor, stats, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if model.backbone is None:
        from safetensors.torch import save_file
        save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(directory / 'policy.safetensors'))
    else:
        from safetensors.torch import save_file
        model.backbone.save_pretrained(directory / 'backbone')
        processor.save_pretrained(directory / 'backbone')
        save_file({key: value.detach().cpu().contiguous() for key, value in model.head.state_dict().items()}, str(directory / 'action_head.safetensors'))
    (directory / 'policy.json').write_text(json.dumps({'format': 'mah_vla_v1', 'config': config, 'statistics': stats}, indent=2))


def restore_weights(model, directory):
    from safetensors.torch import load_file
    directory = Path(directory)
    if model.backbone is None:
        model.load_state_dict(load_file(str(directory / 'policy.safetensors')), strict=True)
    else:
        model.head.load_state_dict(load_file(str(directory / 'action_head.safetensors')), strict=True)
        if hasattr(model.backbone, 'peft_config'):
            from peft import set_peft_model_state_dict
            from peft.utils.save_and_load import load_peft_weights
            result = set_peft_model_state_dict(model.backbone, load_peft_weights(str(directory / 'backbone')))
            if result.unexpected_keys:
                raise ValueError(f'Unexpected VLA adapter keys: {result.unexpected_keys}')
        else:
            weights = {}
            for path in (directory / 'backbone').glob('*.safetensors'):
                weights.update(load_file(str(path)))
            # HF serialization may omit tied aliases: compare missing names by parameter identity.
            result = model.backbone.load_state_dict(weights, strict=False)
            parameters = dict(model.backbone.named_parameters(remove_duplicate=False))
            for name in result.missing_keys:
                if name not in parameters or not any(parameters[name] is parameters.get(key) for key in weights):
                    raise ValueError(f'Missing backbone weight: {name}')
            if result.unexpected_keys:
                raise ValueError(f'Unexpected backbone weights: {result.unexpected_keys}')
