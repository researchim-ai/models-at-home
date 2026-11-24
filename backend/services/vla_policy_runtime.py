"""Shared lifecycle hooks for standalone action policies."""
from pathlib import Path
from backend.services.vla_policies import POLICIES


def standalone(config):return config.get('architecture') in POLICIES


def create(c):
    if c['architecture']=='openvla':
        from backend.services.vla_openvla import Adapter
    else:
        from backend.services.vla_lerobot import Adapter
    return Adapter(c),None


def extra_statistics(data,stats,c):
    if c['architecture']=='openvla':
        from backend.services.vla_openvla import action_statistics
        stats={**stats,'openvla':action_statistics(data.rows)}
    return stats


def validate_rows(data,c):
    if c['architecture']=='openvla':
        from backend.services.vla_openvla import action_statistics
        action_statistics(data.rows)  # Validate gripper contract on holdout too; do not use holdout statistics.
    if c['architecture']=='pi0' and any(len(row['actions'])!=c['action_horizon'] for row in data.rows):
        raise ValueError('π0 requires full action chunks in every row; reduce horizon or provide complete chunks')


def preprocessor(c):
    """Instantiate only configs/tokenizers, never full policy weights during preflight."""
    import torch.nn as nn
    if c['architecture']=='openvla':
        from backend.services.vla_openvla import Adapter,metadata
        _,processor=metadata(c)
        obj=Adapter.__new__(Adapter);nn.Module.__init__(obj);obj.processor=processor
    else:
        from backend.services.vla_lerobot import Adapter,policy_config
        obj=Adapter.__new__(Adapter);nn.Module.__init__(obj);obj.kind=c['architecture'];obj.policy_config=policy_config(c)
        obj.tokenizer=None
        if obj.kind!='act':
            from transformers import AutoTokenizer
            pc=obj.policy_config
            obj.tokenizer=AutoTokenizer.from_pretrained(pc.vlm_model_name if obj.kind=='smolvla' else pc.text_tokenizer_name)
        obj.camera=c.get('policy_camera_key') or next(iter(obj.policy_config.image_features),'')
        if obj.camera not in obj.policy_config.image_features:raise ValueError(f'Invalid camera key {obj.camera}')
    return obj


def export_config(c,directory):
    directory=Path(directory);c={**c}
    if c['architecture']=='openvla':
        c['policy_processor_path']=str(directory/'processor')
        if c['tuning_method']=='full':c['model_path']=str(directory/'policy');c['policy_revision']=''
    else:
        c['model_path']=str(directory/'policy')
        if (directory/'tokenizer').is_dir():c['policy_tokenizer_path']=str(directory/'tokenizer')
        if (directory/'vlm_assets').is_dir():c['policy_vlm_assets']=str(directory/'vlm_assets')
    return c


def denormalize(model,values,stats):
    if hasattr(model,'denormalize'):return model.denormalize(values,stats)
    return values*values.new_tensor(stats['action']['std'])+values.new_tensor(stats['action']['mean'])
