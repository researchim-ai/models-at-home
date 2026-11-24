"""Lazy model imports shared by training and inference."""


def model_classes(kind):
    if kind in ('blueprint', 'homellm_blueprint'):
        from backend.services.blueprint_model import StudioBlueprintConfig, StudioBlueprintForCausalLM
        return StudioBlueprintConfig, StudioBlueprintForCausalLM
    if kind in ('home', 'homellm'):
        from homellm.models.home_model import HomeConfig, HomeForCausalLM
        return HomeConfig, HomeForCausalLM
    if kind in ('home_moe', 'homellm_moe'):
        from homellm.models.home_model_moe import HomeMoEConfig, HomeMoEForCausalLM
        return HomeMoEConfig, HomeMoEForCausalLM
    if kind in ('gpt2', 'gpt2_home'):
        from homellm.models.gpt2_model import GPT2HomeConfig, GPT2HomeForCausalLM
        return GPT2HomeConfig, GPT2HomeForCausalLM
    raise ValueError(f'Unknown custom model type: {kind}')
