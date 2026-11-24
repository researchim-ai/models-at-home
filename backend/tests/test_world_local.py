import pytest
torch=pytest.importorskip('torch')
from backend.services.world_local import encode_action
from backend.vendor.diamond.atari.denoiser import Denoiser,DenoiserConfig
from backend.vendor.diamond.atari.inner_model import InnerModelConfig
from backend.vendor.diamond.atari.diffusion_sampler import DiffusionSampler,DiffusionSamplerConfig


def test_csgo_control_contract_includes_neutral_mouse_categories():
    neutral=encode_action([])
    assert neutral.shape==(51,) and neutral.sum()==2
    assert neutral[13+11]==1 and neutral[36+7]==1
    active=encode_action(['w','attack'],60,-50)
    assert active[0]==1 and active[11]==1 and active.sum()==4
    assert encode_action(['w','s','a','d'])[:4].sum()==0


def test_embedded_architecture_denoises_and_keeps_context_unmodified():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=Denoiser(DenoiserConfig(InnerModelConfig(3,4,16,[1,1],[8,16],[False,False],4),.5,.3)).eval()
    obs=torch.rand(1,4,3,16,16)*2-1;act=torch.zeros(1,4,dtype=torch.long)
    initial=obs.clone()
    predicted,trajectory=DiffusionSampler(model,DiffusionSamplerConfig(3)).sample(obs,act)
    assert predicted.shape==(1,3,16,16) and torch.isfinite(predicted).all()
    assert torch.equal(obs,initial) and len(trajectory)==4
