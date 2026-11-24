"""Small, local multimodal adapters for Research Lab blueprints, trained from scratch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from backend.services.blueprints import validate, with_tokenizer

BYTE_VOCAB = 259  # PAD=0, BOS=1, EOS=2, byte values + 3.


def digest(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_core(spec):
    spec = validate(spec)
    embeddings = [b for b in spec['blocks'] if b['type'] == 'token_embedding']
    if len(embeddings) != 1:
        raise ValueError('Multimodal adapters require exactly one token_embedding')
    mixers=[]
    blocked = {'custom_op', 'inline_code', 'permute', 'adaptive_avg_pool', 'groupnorm'}
    def inspect(block):
        kind=block['type'].removeprefix('liger_')
        if kind in ('attention','causal_self_attention','llama_block') or (kind=='causal_conv1d' and block.get('params',{}).get('kernel_size',3)>1):
            mixers.append(kind)
        if block['type'].removeprefix('liger_') in blocked:
            raise ValueError(f"{block['type']}: multimodal causal/sequence contract is not verified")
        for child in block.get('params', {}).get('body', []):
            inspect(child)
        if block['type'] == 'repeater':
            inspect({'type': block['params']['block_type'], 'params': block['params'].get('block_params', {})})
    for block in spec['blocks']:
        inspect(block)
    if not mixers:
        raise ValueError('Multimodal backbone needs a verified cross-token mixer (attention or causal convolution) so text can depend on image tokens')
    return spec


def normalize(config, root, *, task):
    """Capture source and effective specs. A saved file is never read again on resume."""
    source = config.get('source_blueprint_spec') or config.get('model_blueprint_spec')
    if source is None:
        path = Path(config.get('model_blueprint', '')).expanduser()
        path = path if path.is_absolute() else Path(root) / path
        if not path.is_file():
            raise ValueError('Choose a Research Lab blueprint')
        source = json.loads(path.read_text())
    source = check_core(source)
    effective = with_tokenizer(source, BYTE_VOCAB)
    image_size = config.get('vision_image_size', config.get('image_size', 32))
    patch_size = config.get('vision_patch_size', 8)
    if isinstance(image_size, bool) or not isinstance(image_size, int) or not 16 <= image_size <= 256:
        raise ValueError('Research adapter image size must be in [16, 256]')
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or not 2 <= patch_size <= 32 or image_size % patch_size:
        raise ValueError('vision_patch_size must divide image size (2..32)')
    patches = (image_size // patch_size) ** 2
    if patches > 256 or effective['max_position_embeddings'] <= patches + 2:
        raise ValueError('Image patches + text must fit the blueprint context; reduce image size or increase context')
    if config.get('training_backend', 'transformers') != 'transformers' or config.get('tuning_method', 'full') != 'full':
        raise ValueError('Research blueprint adapters currently train full weights with native PyTorch, without Unsloth/LoRA/QLoRA')
    if task == 'vlm' and config.get('stage', 'vlm_sft') not in ('vlm_sft', 'vlm_pretrain'):
        raise ValueError('Research VLM blueprints support SFT/pretrain; GRPO is not implemented for this adapter')
    config.update(source_blueprint_spec=source, source_blueprint_hash=digest(source), model_blueprint_spec=effective,
        effective_blueprint_hash=digest(effective), vision_image_size=image_size, vision_patch_size=patch_size,
        vision_tokens=patches, byte_tokenizer_version=1, seq_len=effective['max_position_embeddings'])
    if task == 'vla':
        config['image_size'] = image_size
        config['max_instruction_bytes'] = min(config.get('max_instruction_bytes', 512), effective['max_position_embeddings'] - patches - 2)
    return config


def byte_ids(text):
    return [value + 3 for value in text.encode('utf-8')]


def instruction_batch(instructions, device):
    rows = [[1, *byte_ids(text), 2] for text in instructions]
    ids = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row, device=device)
        mask[index, :len(row)] = 1
    return ids, mask


class BlueprintMultimodalCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        from backend.services.blueprint_model import StudioBlueprintConfig, StudioBlueprintForCausalLM
        spec = check_core(config['model_blueprint_spec'])
        if spec['vocab_size'] != BYTE_VOCAB:
            raise ValueError('Research multimodal models require the explicit byte tokenizer profile (259 tokens)')
        language = StudioBlueprintForCausalLM(StudioBlueprintConfig(vocab_size=BYTE_VOCAB, hidden_size=spec['hidden_size'],
            blueprint=spec, max_position_embeddings=spec['max_position_embeddings'], tie_word_embeddings=False,
            use_cache=False, studio_graph_version=1))
        self.graph = language.model
        self.embedding_id = next(b['id'] for b in spec['blocks'] if b['type'] == 'token_embedding')
        self.input_width = self.graph.modules_dict[self.embedding_id].embedding_dim
        self.output_width = language.lm_head.in_features
        self.context = spec['max_position_embeddings']
        self.image_size = config['vision_image_size']
        patch = config['vision_patch_size']
        self.patch_projection = nn.Conv2d(3, self.input_width, kernel_size=patch, stride=patch)
        self.patch_positions = nn.Parameter(torch.zeros(1, (self.image_size // patch) ** 2, self.input_width))
        nn.init.normal_(self.patch_positions, std=.02)

    def forward(self, images, input_ids, attention_mask, loop_iterations=None):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError('Research vision input must be RGB at the configured square resolution')
        vision = self.patch_projection(images).flatten(2).transpose(1, 2) + self.patch_positions
        tokens = self.graph.modules_dict[self.embedding_id](input_ids)
        embeddings = torch.cat((vision, tokens), dim=1)
        if embeddings.shape[1] > self.context:
            raise ValueError(f'Image patches + byte text ({embeddings.shape[1]}) exceed context ({self.context}); no silent truncation')
        prefix_ids = input_ids.new_zeros(input_ids.shape[0], vision.shape[1])
        ids = torch.cat((prefix_ids, input_ids), dim=1)
        mask = torch.cat((attention_mask.new_ones(attention_mask.shape[0], vision.shape[1]), attention_mask), dim=1)
        hidden = self.graph(ids, attention_mask=mask, inputs_embeds=embeddings, loop_iterations=loop_iterations)
        return hidden, mask, vision.shape[1]


class BlueprintVisionLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = BlueprintMultimodalCore(config)
        self.lm_head = nn.Linear(self.core.output_width, BYTE_VOCAB, bias=False)

    def forward(self, images, input_ids, attention_mask, labels=None, loop_iterations=None):
        hidden, _, prefix = self.core(images, input_ids, attention_mask, loop_iterations)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            full_labels = torch.cat((labels.new_full((len(labels), prefix), -100), labels), dim=1)
            if not (full_labels[:, 1:] != -100).any():
                raise ValueError('No supervised next-token targets')
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, BYTE_VOCAB), full_labels[:, 1:].reshape(-1), ignore_index=-100)
            from homellm.models.blocks.recurrent import auxiliary_losses
            aux = auxiliary_losses(self.core.graph)
            if aux and self.training:
                loss = loss + .01 * torch.stack(aux).sum()
        return {'logits': logits[:, prefix:], 'loss': loss}

    @torch.inference_mode()
    def generate_text(self, image, prompt, max_tokens=64, temperature=0, loop_iterations=None):
        if not prompt.strip():
            raise ValueError('Prompt is empty')
        device = next(self.parameters()).device
        ids = torch.tensor([[1, *byte_ids('User: ' + prompt + '\nAssistant: ')]], device=device)
        if ids.shape[1] + self.config['vision_tokens'] >= self.core.context:
            raise ValueError('Image and prompt do not leave generation budget')
        generated = []
        for _ in range(min(max_tokens, self.core.context - self.config['vision_tokens'] - ids.shape[1])):
            logits = self(image.to(device), ids, torch.ones_like(ids), loop_iterations=loop_iterations)['logits'][0, -1].float()
            # PAD and BOS are framing tokens, not generated text.
            logits[:2] = -torch.inf
            token = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1)) if temperature > 0 else int(logits.argmax())
            if token == 2:
                break
            generated.append(token - 3)
            ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
        return bytes(generated).decode('utf-8', errors='replace')


def save_vlm(directory, model, config):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(directory / 'model.safetensors'))
    (directory / 'blueprint_vlm.json').write_text(json.dumps({'format': 'mah_blueprint_vlm_v1', 'config': config}, indent=2))


def load_vlm(directory, device='cpu'):
    from safetensors.torch import load_file
    directory = Path(directory)
    meta = json.loads((directory / 'blueprint_vlm.json').read_text())
    if meta.get('format') != 'mah_blueprint_vlm_v1':
        raise ValueError('Unsupported Research VLM export')
    model = BlueprintVisionLM(meta['config'])
    model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
    return model.to(device).eval()
