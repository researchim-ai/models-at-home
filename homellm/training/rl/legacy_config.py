"""
Backward-compatible конфигурация для GRPO/RL обучения.

DEPRECATED: Используйте новую модульную систему из homellm.training.rl.config:
    from homellm.training.rl.config import GRPOConfig, DAPOConfig, SDPOConfig

Этот файл сохранён для обратной совместимости со старым кодом.
Он содержит универсальный GRPOConfig который работает со всеми алгоритмами.
"""
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any

# Импортируем из новой модульной системы
from .config.base import RLAlgorithm

# Re-export для обратной совместимости
__all__ = ["RLAlgorithm", "GRPOConfig"]


@dataclass
class GRPOConfig:
    """
    Универсальный конфиг для всех RL алгоритмов (backward compatible).
    
    DEPRECATED: Рекомендуется использовать специализированные конфиги:
        - GRPOConfig (из config.grpo)
        - DrGRPOConfig (из config.drgrpo)
        - DAPOConfig (из config.dapo)
        - SDPOConfig (из config.sdpo)
    
    Этот класс сохранён для обратной совместимости со старым кодом.
    Он автоматически выбирает правильные дефолты в зависимости от algorithm.
    """
    
    # Алгоритм
    algorithm: RLAlgorithm = RLAlgorithm.DAPO
    
    # ============================================================
    # РАЗМЕРЫ БАТЧА
    # ============================================================
    group_size: int = 8
    batch_size: int = 8
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    # ============================================================
    # ПАРАМЕТРЫ ГЕНЕРАЦИИ
    # ============================================================
    max_new_tokens: int = 1024
    max_prompt_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    
    # ============================================================
    # КЛИППИНГ (GRPO/DAPO)
    # ============================================================
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2
    
    # ============================================================
    # KL РЕГУЛЯРИЗАЦИЯ
    # ============================================================
    kl_weight: float = 0.0
    
    # ============================================================
    # ОПТИМИЗАТОР
    # ============================================================
    learning_rate: float = 5e-6
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ============================================================
    # ОБУЧЕНИЕ
    # ============================================================
    num_epochs: int = 1
    epochs_per_step: int = 1
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    max_prompts: Optional[int] = None
    
    # ============================================================
    # DR.GRPO
    # ============================================================
    use_std_normalization: bool = True
    fixed_length_normalizer: Optional[int] = None
    
    # ============================================================
    # DAPO
    # ============================================================
    dynamic_sampling: bool = False
    max_refill_rounds: int = 3
    token_level_loss: bool = False
    overlong_penalty: float = -1.0
    overlong_buffer: int = 0
    
    # ============================================================
    # SDPO
    # ============================================================
    use_self_distillation: bool = False
    sdpo_success_threshold: float = 0.5
    sdpo_alpha: float = 0.5
    sdpo_full_logit_distillation: bool = False
    sdpo_distillation_topk: Optional[int] = None
    sdpo_is_clip: float = 2.0
    sdpo_include_feedback: bool = True
    sdpo_max_reprompt_len: int = 4096
    sdpo_ema_rate: float = 0.0
    sdpo_loss_weight: float = 1.0
    sdpo_reprompt_template: str = """Here is the problem:
{question}

Here is a successful solution for reference:
{successful_solution}

Now solve this problem step by step."""
    sdpo_feedback_template: str = """
Previous attempt feedback:
{feedback}
"""
    
    # ============================================================
    # ОПТИМИЗАЦИИ ГЕНЕРАЦИИ
    # ============================================================
    use_prefix_grouper: bool = True
    rollout_batch_size: int = 1
    ds3_gather_for_generation: bool = True

    # ============================================================
    # ROLLOUT ENGINE
    # ============================================================
    use_rollout_engine: bool = False
    rollout_engine_backend: Literal["hf", "vllm"] = "hf"
    rollout_sync_interval: int = 10
    rollout_sync_trainable_only: bool = True
    rollout_offload_to_cpu: bool = False
    rollout_device: str = "auto"
    vllm_gpu_memory_utilization: float = 0.85
    vllm_device: str = "cuda:0"
    
    # ============================================================
    # LIGER KERNEL
    # ============================================================
    use_liger: bool = True
    liger_patch_model: bool = True
    liger_chunk_size: int = 4096
    liger_fused_grpo: bool = True
    liger_grpo_loss_type: str = "dapo"
    
    # ============================================================
    # ЛОГИРОВАНИЕ
    # ============================================================
    output_dir: str = "./output/grpo"
    save_steps: int = 100
    log_steps: int = 10
    export_on_checkpoint: bool = True
    merge_lora: bool = True
    use_wandb: bool = False
    wandb_project: str = "homellm-grpo"
    
    # ============================================================
    # LORA
    # ============================================================
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # ============================================================
    # КВАНТИЗАЦИЯ
    # ============================================================
    use_4bit: bool = False
    use_8bit: bool = False
    quantize_reference_model: bool = False
    use_flash_attention: bool = True
    
    # ============================================================
    # ФОРМАТ
    # ============================================================
    reasoning_format: str = "deepseek"
    user_system_prompt: str = ""
    
    # ============================================================
    # СИСТЕМА
    # ============================================================
    seed: int = 42
    mixed_precision: str = "bf16"
    fp16_pure: bool = False
    grad_checkpoint: bool = False
    ui_run_dir: Optional[str] = None
    
    def __post_init__(self):
        """Применяем настройки по умолчанию в зависимости от алгоритма."""
        if isinstance(self.algorithm, str):
            self.algorithm = RLAlgorithm(self.algorithm)
        
        if self.algorithm == RLAlgorithm.DRGRPO:
            self.use_std_normalization = False
            if self.fixed_length_normalizer is None:
                self.fixed_length_normalizer = self.max_new_tokens
            self.liger_grpo_loss_type = "dr_grpo"
                
        elif self.algorithm == RLAlgorithm.DAPO:
            self.clip_eps_high = max(self.clip_eps_high, 0.28)
            self.dynamic_sampling = True
            self.token_level_loss = True
            self.use_std_normalization = False
            if self.fixed_length_normalizer is None:
                self.fixed_length_normalizer = self.max_new_tokens
            self.liger_grpo_loss_type = "dapo"
        
        elif self.algorithm == RLAlgorithm.SDPO:
            self.use_std_normalization = False
            if self.fixed_length_normalizer is None:
                self.fixed_length_normalizer = self.max_new_tokens
            self.use_self_distillation = True
            self.dynamic_sampling = True
            # SDPO не использует fused loss (нужен доступ к logits)
            self.liger_fused_grpo = False
    
    @classmethod
    def from_preset(cls, preset: str) -> "GRPOConfig":
        """
        Создаёт конфигурацию из предустановки.
        
        Args:
            preset: Название предустановки
        """
        presets = {
            "grpo": cls(
                algorithm=RLAlgorithm.GRPO,
                use_std_normalization=True,
                clip_eps_high=0.2,
            ),
            "drgrpo": cls(
                algorithm=RLAlgorithm.DRGRPO,
                use_std_normalization=False,
                clip_eps_high=0.2,
            ),
            "dapo": cls(
                algorithm=RLAlgorithm.DAPO,
                use_std_normalization=False,
                clip_eps_high=0.28,
                dynamic_sampling=True,
                token_level_loss=True,
            ),
            "sdpo": cls(
                algorithm=RLAlgorithm.SDPO,
                use_std_normalization=False,
                use_self_distillation=True,
                dynamic_sampling=True,
                sdpo_alpha=0.5,
                sdpo_success_threshold=0.5,
                sdpo_full_logit_distillation=False,
                sdpo_loss_weight=1.0,
            ),
            "reasoning_small": cls(
                algorithm=RLAlgorithm.DRGRPO,
                group_size=8,
                batch_size=4,
                train_batch_size=2,
                max_new_tokens=512,
                learning_rate=5e-6,
                use_lora=True,
                use_4bit=True,
            ),
            "reasoning_large": cls(
                algorithm=RLAlgorithm.DAPO,
                group_size=16,
                batch_size=8,
                train_batch_size=4,
                max_new_tokens=2048,
                learning_rate=1e-6,
                use_lora=True,
                use_4bit=True,
                dynamic_sampling=True,
            ),
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        
        return presets[preset]
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфиг в словарь для логирования."""
        result = {
            "algorithm": self.algorithm.value,
            "group_size": self.group_size,
            "batch_size": self.batch_size,
            "train_batch_size": self.train_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "clip_eps_low": self.clip_eps_low,
            "clip_eps_high": self.clip_eps_high,
            "kl_weight": self.kl_weight,
            "learning_rate": self.learning_rate,
            "min_lr_ratio": self.min_lr_ratio,
            "max_steps": self.max_steps,
            "max_prompts": self.max_prompts,
            "mixed_precision": self.mixed_precision,
            "grad_checkpoint": self.grad_checkpoint,
            "save_steps": self.save_steps,
            "log_steps": self.log_steps,
            "export_on_checkpoint": self.export_on_checkpoint,
            "merge_lora": self.merge_lora,
            "use_std_normalization": self.use_std_normalization,
            "fixed_length_normalizer": self.fixed_length_normalizer,
            "dynamic_sampling": self.dynamic_sampling,
            "token_level_loss": self.token_level_loss,
            "use_lora": self.use_lora,
            "use_4bit": self.use_4bit,
        }
        
        # SDPO-специфичные параметры
        if self.algorithm == RLAlgorithm.SDPO or self.use_self_distillation:
            result.update({
                "use_self_distillation": self.use_self_distillation,
                "sdpo_success_threshold": self.sdpo_success_threshold,
                "sdpo_alpha": self.sdpo_alpha,
                "sdpo_full_logit_distillation": self.sdpo_full_logit_distillation,
                "sdpo_distillation_topk": self.sdpo_distillation_topk,
                "sdpo_is_clip": self.sdpo_is_clip,
                "sdpo_include_feedback": self.sdpo_include_feedback,
                "sdpo_loss_weight": self.sdpo_loss_weight,
                "sdpo_ema_rate": self.sdpo_ema_rate,
            })
        
        return result
