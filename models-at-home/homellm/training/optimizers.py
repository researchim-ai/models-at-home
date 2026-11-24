from __future__ import annotations

import math
import re
import time
from contextlib import nullcontext
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
    """MuonWithAuxAdam adapted from @Muon, extended for DeepSpeed ZeRO modes."""

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
        muon_ds_zero_stage: int = 0,
        muon_ds_offload_optimizer: str | None = None,
        muon_ds_offload_param: str | None = None,
        muon_ds_strict_mode: bool = True,
        muon_ds_profile_optimizer_step: bool = False,
        muon_ds_gather_bucket_numel: int = 50_000_000,
        muon_ds_fast_aux_adamw: bool = True,
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
        self.muon_ds_zero_stage = int(muon_ds_zero_stage or 0)
        self.muon_ds_offload_optimizer = str(muon_ds_offload_optimizer or "none").lower()
        self.muon_ds_offload_param = str(muon_ds_offload_param or "none").lower()
        self.muon_ds_strict_mode = bool(muon_ds_strict_mode)
        self.muon_ds_profile_optimizer_step = bool(muon_ds_profile_optimizer_step)
        self.muon_ds_gather_bucket_numel = int(max(1, muon_ds_gather_bucket_numel))
        self.muon_ds_fast_aux_adamw = bool(muon_ds_fast_aux_adamw)
        self._use_ds_gather = (
            self.muon_ds_zero_stage >= 2
            or self.muon_ds_offload_optimizer not in ("none", "")
            or self.muon_ds_offload_param not in ("none", "")
        )
        self._safe_get_full_grad = None
        self._safe_get_full_fp32_param = None
        self._safe_set_full_fp32_param = None
        self._GatheredParameters = None
        if self._use_ds_gather:
            try:
                from deepspeed.runtime.zero.partition_parameters import GatheredParameters
                from deepspeed.utils import (
                    safe_get_full_fp32_param,
                    safe_get_full_grad,
                    safe_set_full_fp32_param,
                )

                self._GatheredParameters = GatheredParameters
                self._safe_get_full_grad = safe_get_full_grad
                self._safe_get_full_fp32_param = safe_get_full_fp32_param
                self._safe_set_full_fp32_param = safe_set_full_fp32_param
            except Exception as exc:
                if self.muon_ds_strict_mode:
                    raise RuntimeError(
                        "MuonWithAuxAdam requires DeepSpeed ZeRO gather APIs for this DS mode."
                    ) from exc

        params_list = [p for p in params if p.requires_grad]
        if len(params_list) == 0:
            raise ValueError("MuonWithAuxAdam received empty params list")

        name_by_id: dict[int, str] = {}
        if named_params:
            for n, p in named_params:
                if p.requires_grad:
                    name_by_id[id(p)] = n
        self._param_name_by_id = name_by_id

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
        self._muon_params = muon_params
        self._adamw_params = adamw_params

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
        self._muon_group_cfg = {
            "lr": muon_lr_resolved,
            "weight_decay": muon_wd_resolved,
            "momentum": float(muon_momentum),
            "nesterov": bool(muon_nesterov),
            "ns_coefficients": muon_ns_coefficients,
            "ns_steps": int(muon_ns_steps),
            "eps": float(muon_eps),
        }
        self._adamw_group_cfg = {
            "lr": adamw_lr_resolved,
            "weight_decay": adamw_wd_resolved,
            "betas": adamw_betas,
            "eps": float(adamw_eps),
        }
        self.last_step_profile: dict[str, float | int] = {
            "muon_gather_ms": 0.0,
            "muon_ns_ms": 0.0,
            "muon_scatter_ms": 0.0,
            "muon_gathered_param_tensors": 0,
            "muon_gathered_param_bytes_est": 0,
        }

    def _param_name(self, p: torch.nn.Parameter) -> str:
        name = self._param_name_by_id.get(id(p), "")
        if name:
            return name
        return f"__unnamed_{id(p)}"

    def _serialize_tensor(self, t: torch.Tensor) -> torch.Tensor:
        # Keep dtype/device; checkpoint backend handles sharding/placement.
        return t.detach().clone()

    def state_dict(self):
        # Custom state_dict to avoid tensor-id mapping issues under ZeRO-3,
        # where DeepSpeed may replace optimizer.param_groups tensors.
        muon_state: dict[str, dict] = {}
        for p in self._muon_params:
            st = self.state.get(p, {})
            if not st:
                continue
            row: dict[str, object] = {}
            if "momentum_buffer" in st:
                row["momentum_buffer"] = self._serialize_tensor(st["momentum_buffer"])
            muon_state[self._param_name(p)] = row

        adamw_state: dict[str, dict] = {}
        for p in self._adamw_params:
            st = self.state.get(p, {})
            if not st:
                continue
            row = {}
            if "exp_avg" in st:
                row["exp_avg"] = self._serialize_tensor(st["exp_avg"])
            if "exp_avg_sq" in st:
                row["exp_avg_sq"] = self._serialize_tensor(st["exp_avg_sq"])
            if "step" in st:
                row["step"] = int(st["step"])
            adamw_state[self._param_name(p)] = row

        return {
            "version": 1,
            "muon_hyper": dict(self._muon_group_cfg),
            "adamw_hyper": dict(self._adamw_group_cfg),
            "muon_state_by_name": muon_state,
            "adamw_state_by_name": adamw_state,
        }

    def load_state_dict(self, state_dict):
        muon_state = state_dict.get("muon_state_by_name", {}) or {}
        adamw_state = state_dict.get("adamw_state_by_name", {}) or {}

        for p in self._muon_params:
            name = self._param_name(p)
            row = muon_state.get(name)
            if not isinstance(row, dict):
                continue
            st = self.state[p]
            mb = row.get("momentum_buffer")
            if isinstance(mb, torch.Tensor):
                st["momentum_buffer"] = mb.to(p.device, dtype=p.dtype)

        for p in self._adamw_params:
            name = self._param_name(p)
            row = adamw_state.get(name)
            if not isinstance(row, dict):
                continue
            st = self.state[p]
            exp_avg = row.get("exp_avg")
            exp_avg_sq = row.get("exp_avg_sq")
            step = row.get("step")
            if isinstance(exp_avg, torch.Tensor):
                st["exp_avg"] = exp_avg.to(p.device, dtype=p.dtype)
            if isinstance(exp_avg_sq, torch.Tensor):
                st["exp_avg_sq"] = exp_avg_sq.to(p.device, dtype=p.dtype)
            if step is not None:
                st["step"] = int(step)

    def _fail_or_warn(self, message: str) -> bool:
        if self.muon_ds_strict_mode:
            raise RuntimeError(message)
        return False

    def _maybe_gather_context(self, params: list[torch.nn.Parameter]):
        if not self._use_ds_gather:
            return nullcontext()
        if self._GatheredParameters is None:
            self._fail_or_warn("DeepSpeed GatheredParameters is unavailable for Muon step.")
            return nullcontext()
        # Some DeepSpeed setups (or world_size=1) keep regular tensors without all_gather().
        if not any(hasattr(p, "all_gather") for p in params):
            return nullcontext()
        return self._GatheredParameters(params, modifier_rank=None)

    def _get_full_grad_like_param(
        self, p: torch.nn.Parameter, expected_shape: torch.Size | None = None
    ) -> torch.Tensor | None:
        grad = None
        if self._safe_get_full_grad is not None:
            try:
                grad = self._safe_get_full_grad(p)
            except Exception:
                grad = None
        if grad is None:
            grad = p.grad
        if grad is None:
            return None
        target_shape = expected_shape if expected_shape is not None else p.shape
        if grad.shape != target_shape:
            target_numel = math.prod(target_shape) if len(target_shape) > 0 else 1
            if grad.numel() == target_numel:
                grad = grad.reshape(target_shape)
            else:
                self._fail_or_warn(
                    f"MuonWithAuxAdam expected full grad shape {tuple(target_shape)}, got {tuple(grad.shape)}"
                )
                return None
        return grad

    def _get_full_param(self, p: torch.nn.Parameter) -> torch.Tensor | None:
        full_param = None
        if self._safe_get_full_fp32_param is not None:
            try:
                full_param = self._safe_get_full_fp32_param(p)
            except Exception:
                full_param = None
        if full_param is None and self._use_ds_gather and self._GatheredParameters is not None:
            # Fallback path for ZeRO-3/offload when safe_get_full_fp32_param returns None.
            try:
                with self._GatheredParameters([p], modifier_rank=None):
                    full_param = p.detach().float().clone()
            except Exception:
                full_param = None
        if full_param is None:
            if self._use_ds_gather:
                self._fail_or_warn("Failed to get full param in DeepSpeed mode.")
                return None
            full_param = p.detach()
        return full_param

    def _iter_param_buckets(self, params: list[torch.nn.Parameter]):
        bucket: list[torch.nn.Parameter] = []
        cur = 0
        max_n = self.muon_ds_gather_bucket_numel
        for p in params:
            n = int(p.numel())
            if bucket and cur + n > max_n:
                yield bucket
                bucket = []
                cur = 0
            bucket.append(p)
            cur += n
        if bucket:
            yield bucket

    @torch.no_grad()
    def _step_distributed(self) -> None:
        cfg_m = self._muon_group_cfg
        cfg_a = self._adamw_group_cfg

        t_gather = 0.0
        t_ns = 0.0
        t_total_start = time.perf_counter()

        # Muon branch: gather in buckets to reduce ZeRO-3 gather/scatter overhead.
        for bucket in self._iter_param_buckets(self._muon_params):
            tg0 = time.perf_counter()
            with self._maybe_gather_context(bucket):
                t_gather += time.perf_counter() - tg0
                for p in bucket:
                    full_param = p.detach().float().clone() if self._use_ds_gather else self._get_full_param(p)
                    if full_param is None:
                        continue
                    grad = self._get_full_grad_like_param(p, expected_shape=full_param.shape)
                    if grad is None:
                        continue

                    state = self.state[p]
                    if "momentum_buffer" not in state or tuple(state["momentum_buffer"].shape) != tuple(
                        full_param.shape
                    ):
                        state["momentum_buffer"] = torch.zeros_like(full_param)

                    momentum_buf = state["momentum_buffer"]
                    work_device = full_param.device
                    if work_device.type == "cpu" and torch.cuda.is_available():
                        work_device = torch.device("cuda", torch.cuda.current_device())

                    grad_work = grad if grad.device == work_device else grad.to(work_device)
                    param_work = full_param if full_param.device == work_device else full_param.to(work_device)
                    if grad_work.ndim < 2 and param_work.ndim >= 2 and grad_work.numel() == param_work.numel():
                        grad_work = grad_work.reshape_as(param_work)

                    mom_work = momentum_buf if momentum_buf.device == work_device else momentum_buf.to(work_device)
                    mom_work.lerp_(grad_work, 1 - float(cfg_m["momentum"]))
                    update = (
                        grad_work.lerp(mom_work, float(cfg_m["momentum"]))
                        if bool(cfg_m["nesterov"])
                        else mom_work
                    )
                    if update.ndim == 4:
                        update = update.view(len(update), -1)

                    tns0 = time.perf_counter()
                    update = _repo_zeropower_via_newtonschulz(
                        update,
                        ns_coefficients=cfg_m["ns_coefficients"],  # type: ignore[arg-type]
                        ns_steps=int(cfg_m["ns_steps"]),
                        eps=float(cfg_m["eps"]),
                    )
                    t_ns += time.perf_counter() - tns0
                    update *= math.sqrt(max(1.0, update.size(-2) / update.size(-1)))

                    if mom_work.data_ptr() != momentum_buf.data_ptr():
                        momentum_buf.copy_(mom_work.to(momentum_buf.device))

                    param_work.mul_(1 - float(cfg_m["lr"]) * float(cfg_m["weight_decay"]))
                    param_work.add_(update.reshape_as(param_work), alpha=-float(cfg_m["lr"]))

                    if self._safe_set_full_fp32_param is not None:
                        self._safe_set_full_fp32_param(p, param_work.to(full_param.dtype))
                    else:
                        if param_work.device != p.device:
                            param_work = param_work.to(p.device)
                        p.copy_(param_work.reshape_as(p))

        # Aux AdamW branch: fast local-shard path for distributed modes.
        beta1, beta2 = cfg_a["betas"]  # type: ignore[assignment]
        if self._use_ds_gather and self.muon_ds_fast_aux_adamw:
            for p in self._adamw_params:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "exp_avg" not in state or tuple(state["exp_avg"].shape) != tuple(p.shape):
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
                update = exp_avg_hat / (exp_avg_sq_hat.sqrt() + float(cfg_a["eps"]))
                p.mul_(1 - float(cfg_a["lr"]) * float(cfg_a["weight_decay"]))
                p.add_(update, alpha=-float(cfg_a["lr"]))
        else:
            for bucket in self._iter_param_buckets(self._adamw_params):
                tg0 = time.perf_counter()
                with self._maybe_gather_context(bucket):
                    t_gather += time.perf_counter() - tg0
                    for p in bucket:
                        full_param = (
                            p.detach().float().clone() if self._use_ds_gather else self._get_full_param(p)
                        )
                        if full_param is None:
                            continue
                        grad = self._get_full_grad_like_param(p, expected_shape=full_param.shape)
                        if grad is None:
                            continue

                        state = self.state[p]
                        if "exp_avg" not in state or tuple(state["exp_avg"].shape) != tuple(full_param.shape):
                            state["exp_avg"] = torch.zeros_like(full_param)
                            state["exp_avg_sq"] = torch.zeros_like(full_param)
                            state["step"] = 0

                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        state["step"] += 1
                        step_t = state["step"]

                        work_device = full_param.device
                        if work_device.type == "cpu" and torch.cuda.is_available():
                            work_device = torch.device("cuda", torch.cuda.current_device())
                        grad_work = grad if grad.device == work_device else grad.to(work_device)
                        param_work = full_param if full_param.device == work_device else full_param.to(work_device)
                        exp_avg_work = exp_avg if exp_avg.device == work_device else exp_avg.to(work_device)
                        exp_avg_sq_work = (
                            exp_avg_sq if exp_avg_sq.device == work_device else exp_avg_sq.to(work_device)
                        )

                        exp_avg_work.lerp_(grad_work, 1 - beta1)
                        exp_avg_sq_work.lerp_(grad_work.square(), 1 - beta2)
                        exp_avg_hat = exp_avg_work / (1 - beta1**step_t)
                        exp_avg_sq_hat = exp_avg_sq_work / (1 - beta2**step_t)
                        update = exp_avg_hat / (exp_avg_sq_hat.sqrt() + float(cfg_a["eps"]))

                        param_work.mul_(1 - float(cfg_a["lr"]) * float(cfg_a["weight_decay"]))
                        param_work.add_(update, alpha=-float(cfg_a["lr"]))

                        if exp_avg_work.data_ptr() != exp_avg.data_ptr():
                            exp_avg.copy_(exp_avg_work.to(exp_avg.device))
                        if exp_avg_sq_work.data_ptr() != exp_avg_sq.data_ptr():
                            exp_avg_sq.copy_(exp_avg_sq_work.to(exp_avg_sq.device))

                        if self._safe_set_full_fp32_param is not None:
                            self._safe_set_full_fp32_param(p, param_work.to(full_param.dtype))
                        else:
                            if param_work.device != p.device:
                                param_work = param_work.to(p.device)
                            p.copy_(param_work.reshape_as(p))

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        self.last_step_profile = {
            "muon_gather_ms": float(t_gather * 1000.0),
            "muon_ns_ms": float(t_ns * 1000.0),
            "muon_scatter_ms": float(max(0.0, total_ms - (t_gather * 1000.0) - (t_ns * 1000.0))),
            "muon_gathered_param_tensors": int(len(self._muon_params)),
            "muon_gathered_param_bytes_est": int(
                sum(p.numel() * p.element_size() for p in self._muon_params) * max(1, world_size)
            ),
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._use_ds_gather:
            self._step_distributed()
            return loss

        t_step_start = time.perf_counter()
        t_gather = 0.0
        t_ns = 0.0

        for group in self.param_groups:
            if group.get("use_muon", False):
                lr = float(group["lr"])
                wd = float(group["weight_decay"])
                momentum = float(group["momentum"])
                nesterov = bool(group["nesterov"])
                ns_coefficients = tuple(group["ns_coefficients"])
                ns_steps = int(group["ns_steps"])
                eps = float(group["eps"])
                params = list(group["params"])
                if len(params) == 0:
                    continue

                t0 = time.perf_counter()
                with self._maybe_gather_context(params):
                    t_gather += time.perf_counter() - t0
                    for p in params:
                        full_param = self._get_full_param(p)
                        if full_param is None:
                            continue
                        grad = self._get_full_grad_like_param(p, expected_shape=full_param.shape)
                        if grad is None:
                            continue
                        state = self.state[p]
                        if "momentum_buffer" not in state or tuple(state["momentum_buffer"].shape) != tuple(full_param.shape):
                            state["momentum_buffer"] = torch.zeros_like(full_param)

                        momentum_buf = state["momentum_buffer"]
                        work_device = full_param.device
                        if work_device.type == "cpu" and torch.cuda.is_available():
                            work_device = torch.device("cuda", torch.cuda.current_device())

                        grad_work = grad if grad.device == work_device else grad.to(work_device)
                        param_work = full_param if full_param.device == work_device else full_param.to(work_device)
                        if grad_work.ndim < 2 and param_work.ndim >= 2 and grad_work.numel() == param_work.numel():
                            grad_work = grad_work.reshape_as(param_work)
                        mom_work = (
                            momentum_buf
                            if momentum_buf.device == work_device
                            else momentum_buf.to(work_device)
                        )
                        mom_work.lerp_(grad_work, 1 - momentum)
                        update = grad_work.lerp(mom_work, momentum) if nesterov else mom_work
                        if update.ndim == 4:
                            update = update.view(len(update), -1)

                        tns0 = time.perf_counter()
                        update = _repo_zeropower_via_newtonschulz(
                            update,
                            ns_coefficients=ns_coefficients,  # type: ignore[arg-type]
                            ns_steps=ns_steps,
                            eps=eps,
                        )
                        t_ns += time.perf_counter() - tns0
                        update *= math.sqrt(max(1.0, update.size(-2) / update.size(-1)))

                        if mom_work.data_ptr() != momentum_buf.data_ptr():
                            momentum_buf.copy_(mom_work.to(momentum_buf.device))

                        param_work.mul_(1 - lr * wd)
                        param_work.add_(update.reshape_as(param_work), alpha=-lr)

                        if self._safe_set_full_fp32_param is not None and self._use_ds_gather:
                            self._safe_set_full_fp32_param(p, param_work.to(full_param.dtype))
                        else:
                            if param_work.device != p.device:
                                param_work = param_work.to(p.device)
                            p.copy_(param_work.reshape_as(p))
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

        t_total = time.perf_counter() - t_step_start
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        gathered_param_bytes = int(
            sum(p.numel() * p.element_size() for p in self._muon_params) * max(1, world_size)
            if self._use_ds_gather
            else 0
        )
        self.last_step_profile = {
            "muon_gather_ms": float(t_gather * 1000.0),
            "muon_ns_ms": float(t_ns * 1000.0),
            "muon_scatter_ms": float(max(0.0, t_total - t_gather - t_ns) * 1000.0),
            "muon_gathered_param_tensors": int(len(self._muon_params) if self._use_ds_gather else 0),
            "muon_gathered_param_bytes_est": gathered_param_bytes,
        }
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

