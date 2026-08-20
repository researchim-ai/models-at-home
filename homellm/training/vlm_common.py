from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

logger = logging.getLogger(__name__)


def _get_gpu_stats() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                gpu_idx, used_mb, total_mb = [part.strip() for part in line.split(",")[:3]]
                out.append(
                    {
                        "id": int(gpu_idx),
                        "memory_used_gb": round(float(used_mb) / 1024, 2),
                        "memory_total_gb": round(float(total_mb) / 1024, 2),
                    }
                )
            return out
    except Exception:
        pass

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                out.append(
                    {
                        "id": i,
                        "memory_used_gb": round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                        "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    }
                )
            except Exception:
                continue
    return out


class MetricsLogger:
    """Writes a VLM-compatible metrics.json for Streamlit monitoring."""

    def __init__(self, log_path: Path, enabled: bool = True):
        self.log_path = Path(log_path)
        self.enabled = enabled
        self.start_timestamp = time.time()
        self.metrics: Dict[str, Any] = {
            "status": "initializing",
            "start_time": datetime.now().isoformat(),
            "current_step": 0,
            "total_steps": 0,
            "epoch": 0,
            "loss_history": [],
            "lr_history": [],
            "steps_history": [],
            "reward_history": [],
            "current_loss": 0.0,
            "current_lr": 0.0,
            "current_reward": 0.0,
            "samples_per_second": 0.0,
            "eta_seconds": 0,
            "elapsed_seconds": 0,
            "error": None,
            "checkpoints": [],
            "gpu_stats": [],
            "sample_outputs": [],
        }
        self._save()

    def _save(self) -> None:
        if not self.enabled:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.log_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.log_path)
        except Exception as exc:
            logger.warning("Failed to save metrics: %s", exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def update(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        self.metrics.update(kwargs)
        self.metrics["elapsed_seconds"] = time.time() - self.start_timestamp
        self._save()

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        *,
        reward: Optional[float] = None,
        samples_per_sec: float = 0.0,
        step_time: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        self.metrics["current_step"] = step
        self.metrics["current_loss"] = float(loss)
        self.metrics["current_lr"] = float(lr)
        self.metrics["samples_per_second"] = float(samples_per_sec)
        self.metrics["loss_history"] = self.metrics.get("loss_history", []) + [float(loss)]
        self.metrics["lr_history"] = self.metrics.get("lr_history", []) + [float(lr)]
        self.metrics["steps_history"] = self.metrics.get("steps_history", []) + [int(step)]
        if reward is not None:
            self.metrics["current_reward"] = float(reward)
            self.metrics["reward_history"] = self.metrics.get("reward_history", []) + [float(reward)]
        self.metrics["elapsed_seconds"] = time.time() - self.start_timestamp
        gpu_stats = _get_gpu_stats()
        self.metrics["gpu_stats"] = gpu_stats
        if gpu_stats:
            self.metrics["gpu_memory_used_mb"] = int(gpu_stats[0].get("memory_used_gb", 0.0) * 1024)
        if step > 0 and step_time > 0:
            remaining = max(0, int(self.metrics.get("total_steps", 0)) - step)
            self.metrics["eta_seconds"] = int(remaining * step_time)
        self._save()

    def log_checkpoint(self, path: str, *, step: Optional[int] = None, loss: Optional[float] = None) -> None:
        checkpoints = self.metrics.get("checkpoints", [])
        checkpoints.append(
            {
                "step": int(step if step is not None else self.metrics.get("current_step", 0)),
                "path": str(path),
                "loss": None if loss is None else float(loss),
                "saved_at": datetime.now().isoformat(),
            }
        )
        self.metrics["checkpoints"] = checkpoints
        self._save()

    def log_sample_output(self, prompt: str, response: str, reward: Optional[float] = None) -> None:
        samples = self.metrics.get("sample_outputs", [])
        samples.append(
            {
                "prompt": prompt[:1000],
                "response": response[:2000],
                "reward": reward,
                "ts": datetime.now().isoformat(),
            }
        )
        self.metrics["sample_outputs"] = samples[-20:]
        self._save()


def get_best_torch_dtype(device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def get_requested_torch_dtype(device: str, config: Dict[str, Any]) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    mixed_precision = str(config.get("mixed_precision", "") or "").lower()
    if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return get_best_torch_dtype(device)


def load_vlm_processor(model_name_or_path: str, config: Dict[str, Any]) -> Any:
    from transformers import AutoProcessor

    processor_kwargs = {"trust_remote_code": True}
    min_pixels = config.get("min_pixels")
    max_pixels = config.get("max_pixels")
    if min_pixels:
        processor_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels:
        processor_kwargs["max_pixels"] = int(max_pixels)
    return AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)


def ensure_cuda_available() -> None:
    """Fail fast with a clear message if CUDA is not usable.

    Without this, training loads the model on CPU and only crashes deep inside
    ``pin_memory`` with a cryptic driver error. A common cause is a torch build
    whose CUDA runtime is newer than the installed NVIDIA driver (e.g. a
    ``+cu130`` wheel on a CUDA 12.8 / 570.x driver), which makes
    ``torch.cuda.is_available()`` return ``False``.
    """
    if torch.cuda.is_available():
        return
    torch_ver = getattr(torch, "__version__", "unknown")
    cuda_ver = getattr(torch.version, "cuda", "unknown")
    raise RuntimeError(
        "CUDA недоступна (torch.cuda.is_available() == False). "
        f"Установлен torch {torch_ver} (CUDA runtime {cuda_ver}). "
        "Скорее всего сборка torch новее драйвера NVIDIA (например, +cu130 при драйвере 12.8/570.x). "
        "Нужен torch с cu128 под этот драйвер — пересоберите образ (Dockerfile уже пиннит torch 2.9.0+cu128)."
    )


def configure_sdpa_kernels(config: Dict[str, Any]) -> None:
    """Enable/disable PyTorch SDPA flash kernels, mirroring the LLM worker.

    PyTorch's scaled_dot_product_attention has FlashAttention-2 built in. Turning
    on the flash SDP backend is what actually makes "FlashAttention" fast — this
    is the same mechanism the LLM Studio uses.
    """
    if not torch.cuda.is_available():
        return
    use_flash_attention = bool(config.get("use_flash_attention"))
    try:
        if use_flash_attention:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        else:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        logger.info(
            "SDPA kernels: flash=%s mem_efficient=%s math=%s (use_flash_attention=%s)",
            getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: "N/A")(),
            getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: "N/A")(),
            getattr(torch.backends.cuda, "math_sdp_enabled", lambda: "N/A")(),
            use_flash_attention,
        )
    except Exception as exc:
        logger.warning("Could not configure CUDA SDPA kernels: %s", exc)


def resolve_attn_implementation(config: Dict[str, Any], device: str) -> str:
    """Pick the attention backend, matching the LLM Studio's exact rules.

    - FlashAttention-2 (the standalone ``flash_attn`` package) is used only when it
      is requested AND weights are fp16/bf16 AND the method is not QLoRA AND the
      package actually imports. This is identical to ``homellm/models/adapters.py``.
    - When FlashAttention is off we force eager.
    - Otherwise we use SDPA, whose flash backend (enabled via
      :func:`configure_sdpa_kernels`) IS FlashAttention-2 built into PyTorch.

    NOTE: The standalone ``flash_attn`` package may be broken in the image (ABI
    mismatch with the installed torch build). In that case both LLM and VLM fall
    back to SDPA flash kernels — real flash attention, just via PyTorch.
    """
    if device != "cuda":
        return "eager"

    use_flash_attention = bool(config.get("use_flash_attention"))
    tuning_method = (config.get("tuning_method") or "lora").lower()
    dtype = get_requested_torch_dtype(device, config)

    if not use_flash_attention:
        return "eager"

    want_flash_pkg = dtype in (torch.float16, torch.bfloat16) and tuning_method != "qlora"
    if want_flash_pkg:
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except Exception as exc:
            logger.warning(
                "flash_attn package requested but not importable (%s); "
                "using PyTorch SDPA flash kernels instead",
                exc,
            )
            return "sdpa"
    # QLoRA / non-half precision: SDPA flash backend (same as LLM leaving it unset).
    return "sdpa"


def load_vlm_model(model_name_or_path: str, config: Dict[str, Any], device: str) -> Any:
    try:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        model_cls = AutoModelForVision2Seq

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    model_kwargs["attn_implementation"] = resolve_attn_implementation(config, device)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    tuning_method = (config.get("tuning_method") or "lora").lower()
    if tuning_method == "qlora":
        from transformers import BitsAndBytesConfig

        compute_dtype = get_requested_torch_dtype(device, config)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype if compute_dtype in (torch.float16, torch.bfloat16) else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # Quantized weights must be placed on a concrete device at load time.
        if device == "cuda":
            model_kwargs["device_map"] = {"": local_rank}
    else:
        model_kwargs["dtype"] = get_requested_torch_dtype(device, config)
        # Single-process, single-GPU: pin to the visible device. Under DDP/accelerate
        # (world_size > 1) let the launcher place the model to avoid clashes.
        if device == "cuda" and world_size == 1:
            model_kwargs["device_map"] = {"": local_rank}

    model = model_cls.from_pretrained(model_name_or_path, **model_kwargs)
    if device == "cuda" and not getattr(model, "hf_device_map", None):
        model = model.to(device)
    # Gradient checkpointing is enabled centrally (prepare_model_for_kbit_training
    # for QLoRA, or the HF Trainer via TrainingArguments) to avoid double-enabling.
    if config.get("gradient_checkpointing", True) and hasattr(model, "config"):
        model.config.use_cache = False
    return model


def _set_requires_grad(module: Any, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def apply_vlm_freeze_policy(model: Any, config: Dict[str, Any]) -> None:
    freeze_vision = bool(config.get("freeze_vision_tower", False))
    freeze_projector = bool(config.get("freeze_projector", False))
    freeze_language = bool(config.get("freeze_language_model", False))

    vision_patterns = ("vision", "visual", "image_tower", "vision_tower", "vision_model")
    projector_patterns = ("projector", "multi_modal_projector", "mm_projector", "connector", "merger")
    language_patterns = ("language_model", "model.layers", "lm_head", "transformer", "embed_tokens")

    for name, module in model.named_modules():
        if freeze_vision and _match_any(name, vision_patterns):
            _set_requires_grad(module, False)
        if freeze_projector and _match_any(name, projector_patterns):
            _set_requires_grad(module, False)
        if freeze_language and _match_any(name, language_patterns):
            _set_requires_grad(module, False)


def apply_vlm_lora(model: Any, config: Dict[str, Any]) -> Any:
    tuning_method = (config.get("tuning_method") or "lora").lower()
    if tuning_method not in {"lora", "qlora"}:
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    use_gc = bool(config.get("gradient_checkpointing", True))
    if tuning_method == "qlora":
        # Casts norms to fp32 and enables input grads so 4-bit QLoRA actually
        # backprops. The HF Trainer enables the gradient-checkpointing hooks.
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    elif use_gc and hasattr(model, "enable_input_require_grads"):
        # LoRA on a frozen base + gradient checkpointing needs input grads enabled.
        model.enable_input_require_grads()

    target_modules: Any = config.get("lora_target_modules") or "all-linear"
    modules_to_save = config.get("lora_modules_to_save") or None
    peft_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        target_modules=target_modules,
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, peft_config)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def mark_training_completed(metrics_logger: MetricsLogger, output_dir: str) -> None:
    elapsed = time.time() - metrics_logger.start_timestamp
    metrics_logger.update(
        status="completed",
        training_duration=f"{int(elapsed // 60)} min",
        final_model_dir=str(output_dir),
    )


def resolve_resume_checkpoint(config: Dict[str, Any]) -> Optional[str]:
    resume_path = config.get("resume_from_checkpoint")
    if resume_path and Path(resume_path).exists():
        return str(Path(resume_path))
    return None
