"""Build reduced real policy fixtures for CPU integration tests. Downloads configs/processors only."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',choices=['smolvla','openvla'],required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    import torch
    torch.manual_seed(42)
    if args.policy=='smolvla':
        from transformers import AutoConfig,AutoProcessor
        from lerobot.configs import FeatureType,PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        source='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
        processor=AutoProcessor.from_pretrained(source);processor.save_pretrained(root/'vlm_assets')
        c=AutoConfig.from_pretrained(source)
        c.text_config.hidden_size=64;c.text_config.num_hidden_layers=2;c.text_config.num_attention_heads=4;c.text_config.num_key_value_heads=2;c.text_config.head_dim=16;c.text_config.intermediate_size=128
        c.vision_config.hidden_size=32;c.vision_config.num_hidden_layers=1;c.vision_config.num_attention_heads=4;c.vision_config.intermediate_size=64;c.vision_config.image_size=32;c.vision_config.patch_size=8
        c.save_pretrained(root/'vlm_assets')
        config=SmolVLAConfig(device='cpu',vlm_model_name=str(root/'vlm_assets'),load_vlm_weights=False,num_vlm_layers=2,num_expert_layers=2,expert_width_multiplier=1.,
            chunk_size=2,n_action_steps=2,max_state_dim=2,max_action_dim=2,num_steps=2,resize_imgs_with_padding=(32,32),
            input_features={'observation.images.camera':PolicyFeature(type=FeatureType.VISUAL,shape=(3,32,32)),'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(1,))},
            output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(2,))})
        model=SmolVLAPolicy(config);model.save_pretrained(root/'policy')
    else:
        from transformers import AutoConfig,AutoModelForVision2Seq,AutoProcessor,LlamaConfig
        source='openvla/openvla-7b'
        c=AutoConfig.from_pretrained(source,trust_remote_code=True)
        c.text_config=LlamaConfig(vocab_size=32064,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,pad_token_id=32000)
        c.timm_model_ids=['vit_tiny_patch16_224'];c.timm_override_act_layers=[None];c.image_sizes=[32];c.use_fused_vision_backbone=False;c.norm_stats={};c._attn_implementation='eager'
        model=AutoModelForVision2Seq.from_config(c,trust_remote_code=True);model.save_pretrained(root/'policy')
        processor=AutoProcessor.from_pretrained(source,trust_remote_code=True);image=processor.image_processor
        processor.image_processor=type(image)(use_fused_vision_backbone=False,input_sizes=[(3,32,32)],interpolations=['bicubic'],means=[image.means[0]],stds=[image.stds[0]])
        type(processor.image_processor).register_for_auto_class('AutoImageProcessor')
        type(processor).register_for_auto_class('AutoProcessor')
        processor.save_pretrained(root/'policy')
    print(f'{args.policy}: {sum(p.numel() for p in model.parameters()):,} random parameters; {root / "policy"}. Test fixture, not pretrained quality.')

if __name__=='__main__':main()
