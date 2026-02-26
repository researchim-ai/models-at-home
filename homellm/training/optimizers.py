from __future__ import annotations

import math
import re
from typing import Iterable

import torch
from torch.optim import Optimizer


class MagmaAdamW(Optimizer):
    """AdamW with momentum-aligned stochastic update masking (Magma-style).

    Notes:
    - First and second moments are always updated densely.
    - Parameter updates are scaled by `s_t * m_t`, where:
      - `m_t ~ Bernoulli(magma_prob)`
      - `s_t` is EMA of sigmoid(cos(momentum, grad) / magma_tau)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        magma_prob: float = 0.5,
        magma_tau: float = 2.0,
        magma_ema_beta: float = 0.9,
        magma_cosine_eps: float = 1e-12,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < magma_prob <= 1.0:
            raise ValueError(f"Invalid magma_prob: {magma_prob}")
        if magma_tau <= 0.0:
            raise ValueError(f"Invalid magma_tau: {magma_tau}")
        if not 0.0 <= magma_ema_beta < 1.0:
            raise ValueError(f"Invalid magma_ema_beta: {magma_ema_beta}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            magma_prob=magma_prob,
            magma_tau=magma_tau,
            magma_ema_beta=magma_ema_beta,
            magma_cosine_eps=magma_cosine_eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            magma_prob = group["magma_prob"]
            magma_tau = group["magma_tau"]
            magma_ema_beta = group["magma_ema_beta"]
            magma_cosine_eps = group["magma_cosine_eps"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("MagmaAdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["magma_s"] = torch.tensor(1.0, device=p.device, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                # Dense moment updates (core Magma requirement)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # AdamW bias correction
                bias_correction1 = 1.0 - beta1 ** step_t
                bias_correction2 = 1.0 - beta2 ** step_t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)

                # Momentum-gradient alignment score
                mu = exp_avg.float()
                g = grad.float()
                cos = torch.dot(mu.flatten(), g.flatten()) / (
                    torch.norm(mu) * torch.norm(g) + magma_cosine_eps
                )
                tilde_s = torch.sigmoid(cos / magma_tau).float()

                s_prev = state["magma_s"]
                s_t = magma_ema_beta * s_prev + (1.0 - magma_ema_beta) * tilde_s
                state["magma_s"] = s_t.detach()

                m_t = 1.0 if torch.rand((), device=p.device) < magma_prob else 0.0
                scale = float(s_t.item()) * m_t

                if scale == 0.0:
                    continue

                # Scale full base update (including decoupled weight decay)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd * scale)
                p.addcdiv_(exp_avg, denom, value=-step_size * scale)

        return loss


# Attribution:
# This MuonWithAuxAdam implementation is adapted from @Muon:
# - https://github.com/KellerJordan/Muon
# - https://github.com/KellerJordan/Muon/blob/master/muon.py
# In particular, the Newton-Schulz orthogonalization and Muon+AuxAdam step logic
# follow the reference `SingleDeviceMuonWithAuxAdam` / `muon_update` flow.
def _repo_zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    if grad.ndim < 2:
        raise ValueError("Repo Muon expects ndim >= 2")
    a, b, c = ns_coefficients
    x = grad.bfloat16()
    if grad.size(-2) > grad.size(-1):
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(ns_steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if grad.size(-2) > grad.size(-1):
        x = x.mT
    return x


class MuonWithAuxAdam(Optimizer):
    """Single-device MuonWithAuxAdam adapted from @Muon reference implementation."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        named_params: list[tuple[str, torch.nn.Parameter]] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        muon_lr: float | None = None,
        muon_weight_decay: float | None = None,
        adamw_lr: float | None = None,
        adamw_weight_decay: float | None = None,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        muon_eps: float = 1e-7,
        muon_ns_steps: int = 5,
        muon_adjust_lr_fn: str | None = None,  # kept for interface compatibility
        muon_hidden_patterns: tuple[str, ...] = (
            r"(^|\.)(layers|h|blocks)(\.|$)",
        ),
        muon_exclude_patterns: tuple[str, ...] = (
            r"(^|\.)(embed|embeddings|embed_tokens|tok_embeddings|token_embeddings|wte|wpe|word_embeddings|position_embeddings)(\.|$)",
            r"(^|\.)(lm_head|output|classifier|head)(\.|$)",
            r"(^|\.)(lora(_[AaBb])?|lora[AaBb]?|adapter|adapters|ia3|prompt|prefix)(\.|$)",
        ),
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-10,
    ) -> None:
        if muon_adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError(
                f"muon_adjust_lr_fn must be one of None|'original'|'match_rms_adamw', got: {muon_adjust_lr_fn}"
            )
        params_list = [p for p in params if p.requires_grad]
        if len(params_list) == 0:
            raise ValueError("MuonWithAuxAdam received empty params list")

        name_by_id: dict[int, str] = {}
        if named_params:
            for n, p in named_params:
                if p.requires_grad:
                    name_by_id[id(p)] = n

        exclude_regex = [re.compile(p) for p in muon_exclude_patterns]
        hidden_regex = [re.compile(p) for p in muon_hidden_patterns]

        total_2d = 0
        excluded_2d = 0
        nonhidden_2d = 0
        unnamed_2d = 0

        def should_use_muon(p: torch.nn.Parameter) -> bool:
            nonlocal total_2d, excluded_2d, nonhidden_2d, unnamed_2d
            if p.ndim != 2:
                return False
            total_2d += 1
            name = name_by_id.get(id(p), "")
            if not name:
                unnamed_2d += 1
                return False
            if hidden_regex and not any(rx.search(name) for rx in hidden_regex):
                nonhidden_2d += 1
                return False
            for rx in exclude_regex:
                if rx.search(name):
                    excluded_2d += 1
                    return False
            return True

        muon_params = [p for p in params_list if should_use_muon(p)]
        muon_ids = {id(p) for p in muon_params}
        adamw_params = [p for p in params_list if id(p) not in muon_ids]
        if len(muon_params) == 0:
            raise ValueError(
                "MuonWithAuxAdam requires at least one trainable 2D parameter for Muon."
            )

        muon_lr_resolved = float(lr if muon_lr is None else muon_lr)
        muon_wd_resolved = float(weight_decay if muon_weight_decay is None else muon_weight_decay)
        adamw_lr_resolved = float(lr if adamw_lr is None else adamw_lr)
        adamw_wd_resolved = float(
            weight_decay if adamw_weight_decay is None else adamw_weight_decay
        )
        if muon_lr_resolved <= 0.0:
            raise ValueError(f"muon_lr must be > 0, got: {muon_lr_resolved}")
        if adamw_lr_resolved <= 0.0:
            raise ValueError(f"adamw_lr must be > 0, got: {adamw_lr_resolved}")
        if muon_wd_resolved < 0.0:
            raise ValueError(f"muon_weight_decay must be >= 0, got: {muon_wd_resolved}")
        if adamw_wd_resolved < 0.0:
            raise ValueError(f"adamw_weight_decay must be >= 0, got: {adamw_wd_resolved}")

        param_groups: list[dict] = [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": muon_lr_resolved,
                "momentum": float(muon_momentum),
                "nesterov": bool(muon_nesterov),
                "weight_decay": muon_wd_resolved,
                "ns_coefficients": muon_ns_coefficients,
                "ns_steps": int(muon_ns_steps),
                "eps": float(muon_eps),
            }
        ]
        if len(adamw_params) > 0:
            param_groups.append(
                {
                    "params": adamw_params,
                    "use_muon": False,
                    "lr": adamw_lr_resolved,
                    "betas": adamw_betas,
                    "eps": float(adamw_eps),
                    "weight_decay": adamw_wd_resolved,
                }
            )
        super().__init__(param_groups, defaults={})

        self.muon_param_count = sum(p.numel() for p in muon_params)
        self.adamw_param_count = sum(p.numel() for p in adamw_params)
        self.total_2d_param_tensors = total_2d
        self.excluded_2d_param_tensors = excluded_2d
        self.nonhidden_2d_param_tensors = nonhidden_2d
        self.unnamed_2d_param_tensors = unnamed_2d
        self.muon_lr = muon_lr_resolved
        self.muon_weight_decay = muon_wd_resolved
        self.adamw_lr = adamw_lr_resolved
        self.adamw_weight_decay = adamw_wd_resolved
        self.muon_param_names_sample = [name_by_id.get(id(p), "<unnamed>") for p in muon_params[:8]]
        self.adamw_param_names_sample = [name_by_id.get(id(p), "<unnamed>") for p in adamw_params[:8]]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                lr = float(group["lr"])
                wd = float(group["weight_decay"])
                momentum = float(group["momentum"])
                nesterov = bool(group["nesterov"])
                ns_coefficients = tuple(group["ns_coefficients"])
                ns_steps = int(group["ns_steps"])
                eps = float(group["eps"])

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    momentum_buf = state["momentum_buffer"]
                    momentum_buf.lerp_(grad, 1 - momentum)
                    update = grad.lerp(momentum_buf, momentum) if nesterov else momentum_buf
                    if update.ndim == 4:
                        update = update.view(len(update), -1)
                    update = _repo_zeropower_via_newtonschulz(
                        update,
                        ns_coefficients=ns_coefficients,  # type: ignore[arg-type]
                        ns_steps=ns_steps,
                        eps=eps,
                    )
                    update *= math.sqrt(max(1.0, update.size(-2) / update.size(-1)))
                    p.mul_(1 - lr * wd)
                    p.add_(update.reshape_as(p), alpha=-lr)
            else:
                lr = float(group["lr"])
                wd = float(group["weight_decay"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    step_t = state["step"]

                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                    exp_avg_hat = exp_avg / (1 - beta1**step_t)
                    exp_avg_sq_hat = exp_avg_sq / (1 - beta2**step_t)
                    update = exp_avg_hat / (exp_avg_sq_hat.sqrt() + eps)
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)

        return loss


class HybridMuonAdamW(Optimizer):
    """Hybrid optimizer: Muon for 2D params, AdamW for others.

    Muon in PyTorch supports only 2D parameters. This wrapper enforces the
    recommended setup from the docs: hidden-layer matrices with Muon, all
    non-2D parameters (bias/embeddings/norm scales, etc.) with AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        named_params: list[tuple[str, torch.nn.Parameter]] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        muon_lr: float | None = None,
        muon_weight_decay: float | None = None,
        adamw_lr: float | None = None,
        adamw_weight_decay: float | None = None,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        muon_eps: float = 1e-7,
        muon_ns_steps: int = 5,
        muon_adjust_lr_fn: str | None = None,
        muon_hidden_patterns: tuple[str, ...] = (
            r"(^|\.)(layers|h|blocks)(\.|$)",
        ),
        muon_exclude_patterns: tuple[str, ...] = (
            r"(^|\.)(embed|embeddings|embed_tokens|tok_embeddings|token_embeddings|wte|wpe|word_embeddings|position_embeddings)(\.|$)",
            r"(^|\.)(lm_head|output|classifier|head)(\.|$)",
            r"(^|\.)(lora(_[AaBb])?|lora[AaBb]?|adapter|adapters|ia3|prompt|prefix)(\.|$)",
        ),
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        params_list = [p for p in params if p.requires_grad]
        if len(params_list) == 0:
            raise ValueError("HybridMuonAdamW received empty params list")

        # Build a name map to support robust exclusions (embeddings/lm_head).
        name_by_id: dict[int, str] = {}
        if named_params:
            for n, p in named_params:
                if p.requires_grad:
                    name_by_id[id(p)] = n

        exclude_regex = [re.compile(p) for p in muon_exclude_patterns]
        hidden_regex = [re.compile(p) for p in muon_hidden_patterns]

        total_2d = 0
        excluded_2d = 0
        nonhidden_2d = 0
        unnamed_2d = 0

        def should_use_muon(p: torch.nn.Parameter) -> bool:
            nonlocal total_2d, excluded_2d, nonhidden_2d, unnamed_2d
            if p.ndim != 2:
                return False
            total_2d += 1
            name = name_by_id.get(id(p), "")
            if not name:
                unnamed_2d += 1
                return False
            if hidden_regex and not any(rx.search(name) for rx in hidden_regex):
                nonhidden_2d += 1
                return False
            for rx in exclude_regex:
                if rx.search(name):
                    excluded_2d += 1
                    return False
            return True

        muon_params = [p for p in params_list if should_use_muon(p)]
        muon_ids = {id(p) for p in muon_params}
        adamw_params = [p for p in params_list if id(p) not in muon_ids]
        if len(muon_params) == 0:
            raise ValueError(
                "HybridMuonAdamW requires at least one trainable 2D parameter for Muon."
            )

        muon_lr_resolved = float(lr if muon_lr is None else muon_lr)
        muon_wd_resolved = float(weight_decay if muon_weight_decay is None else muon_weight_decay)
        adamw_lr_resolved = float(lr if adamw_lr is None else adamw_lr)
        adamw_wd_resolved = float(
            weight_decay if adamw_weight_decay is None else adamw_weight_decay
        )
        if muon_lr_resolved <= 0.0:
            raise ValueError(f"muon_lr must be > 0, got: {muon_lr_resolved}")
        if adamw_lr_resolved <= 0.0:
            raise ValueError(f"adamw_lr must be > 0, got: {adamw_lr_resolved}")
        if muon_wd_resolved < 0.0:
            raise ValueError(f"muon_weight_decay must be >= 0, got: {muon_wd_resolved}")
        if adamw_wd_resolved < 0.0:
            raise ValueError(f"adamw_weight_decay must be >= 0, got: {adamw_wd_resolved}")

        super().__init__(params_list, defaults={"lr": muon_lr_resolved})

        muon_cls = getattr(torch.optim, "Muon", None)
        if muon_cls is None:
            raise RuntimeError(
                f"torch.optim.Muon is unavailable in torch=={torch.__version__}. "
                "Install/upgrade to PyTorch 2.9+."
            )

        self.muon_optimizer = muon_cls(
            muon_params,
            lr=muon_lr_resolved,
            weight_decay=muon_wd_resolved,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_coefficients=muon_ns_coefficients,
            eps=muon_eps,
            ns_steps=muon_ns_steps,
            adjust_lr_fn=muon_adjust_lr_fn,
        )
        self.adamw_optimizer = (
            torch.optim.AdamW(
                adamw_params,
                lr=adamw_lr_resolved,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_wd_resolved,
            )
            if len(adamw_params) > 0
            else None
        )

        self.muon_param_count = sum(p.numel() for p in muon_params)
        self.adamw_param_count = sum(p.numel() for p in adamw_params)
        self.total_2d_param_tensors = total_2d
        self.excluded_2d_param_tensors = excluded_2d
        self.nonhidden_2d_param_tensors = nonhidden_2d
        self.unnamed_2d_param_tensors = unnamed_2d
        self.muon_lr = muon_lr_resolved
        self.muon_weight_decay = muon_wd_resolved
        self.adamw_lr = adamw_lr_resolved
        self.adamw_weight_decay = adamw_wd_resolved
        self.muon_param_names_sample = [name_by_id.get(id(p), "<unnamed>") for p in muon_params[:8]]
        self.adamw_param_names_sample = [name_by_id.get(id(p), "<unnamed>") for p in adamw_params[:8]]

        # Expose both groups so external schedulers update both optimizers.
        self.param_groups = list(self.muon_optimizer.param_groups)
        if self.adamw_optimizer is not None:
            self.param_groups.extend(self.adamw_optimizer.param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = self.muon_optimizer.step(closure)
            if self.adamw_optimizer is not None:
                self.adamw_optimizer.step()
            return loss

        loss = self.muon_optimizer.step()
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon_optimizer.state_dict(),
            "adamw": self.adamw_optimizer.state_dict() if self.adamw_optimizer is not None else None,
        }

    def load_state_dict(self, state_dict):
        self.muon_optimizer.load_state_dict(state_dict["muon"])
        if self.adamw_optimizer is not None and state_dict.get("adamw") is not None:
            self.adamw_optimizer.load_state_dict(state_dict["adamw"])


def parse_magma_target_patterns(raw: str | None) -> list[re.Pattern]:
    if not raw:
        return []
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return [re.compile(p) for p in parts]

