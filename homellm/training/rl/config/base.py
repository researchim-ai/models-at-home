"""
Базовая конфигурация для RL алгоритмов.

BaseRLConfig содержит общие параметры, которые используются всеми алгоритмами:
- Размеры батча и генерации
- Оптимизатор
- Rollout engine
- Liger optimizations
- LoRA/квантизация
- Логирование
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Literal, Dict, Any


class RLAlgorithm(str, Enum):
    """Поддерживаемые алгоритмы RL."""
    GRPO = "grpo"           # Стандартный GRPO с нормализацией по std
    DRGRPO = "drgrpo"       # Dr.GRPO - без деления на std, фиксированная нормализация
    DAPO = "dapo"           # DAPO - clip higher + dynamic sampling + token-level loss
    SDPO = "sdpo"           # SDPO - Self-Distilled Policy Optimization


@dataclass
class BaseRLConfig:
    """
    Базовая конфигурация для всех RL алгоритмов.
    
    Содержит параметры, общие для GRPO, Dr.GRPO, DAPO, SDPO и будущих алгоритмов.
    Алгоритм-специфичные параметры определяются в наследниках.
    
    Attributes:
        # Размеры батча
        group_size: Количество генераций на один промпт (G в статьях)
        batch_size: Размер батча промптов на один шаг rollout
        train_batch_size: Размер батча для обучения
        gradient_accumulation_steps: Накопление градиентов
        
        # Параметры генерации
        max_new_tokens: Максимальное количество генерируемых токенов
        max_prompt_length: Максимальная длина промпта
        temperature: Температура сэмплирования
        top_p: Top-p для nucleus sampling
        
        # Оптимизатор
        learning_rate: Скорость обучения
        weight_decay: Weight decay для AdamW
        max_grad_norm: Максимальная норма градиента
    """
    
    # ============================================================
    # РАЗМЕРЫ БАТЧА
    # ============================================================
    group_size: int = 8
    batch_size: int = 8  # промптов на шаг роллаута
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
    # ОПТИМИЗАТОР
    # ============================================================
    learning_rate: float = 5e-6
    min_lr_ratio: float = 0.1  # Минимальный LR как доля от base LR (для cosine)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ============================================================
    # ОБУЧЕНИЕ
    # ============================================================
    num_epochs: int = 1
    epochs_per_step: int = 1
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    max_prompts: Optional[int] = None  # Лимит по количеству промптов
    
    # ============================================================
    # ОПТИМИЗАЦИИ ГЕНЕРАЦИИ
    # ============================================================
    # Prefix Grouper: shared KV-cache для G completions одного промпта
    use_prefix_grouper: bool = True
    
    # Multi-prompt batching
    rollout_batch_size: int = 1
    
    # ZeRO-3 gather
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
    
    # ============================================================
    # ЛОГИРОВАНИЕ
    # ============================================================
    output_dir: str = "./output/rl"
    save_steps: int = 100
    log_steps: int = 10
    export_on_checkpoint: bool = True
    merge_lora: bool = True
    use_wandb: bool = False
    wandb_project: str = "homellm-rl"
    
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
    # ФОРМАТ И ПРОМПТЫ
    # ============================================================
    reasoning_format: str = "deepseek"  # "deepseek" (<think>), "simple" (<reasoning>)
    user_system_prompt: str = ""
    
    # ============================================================
    # СИСТЕМА
    # ============================================================
    seed: int = 42
    mixed_precision: str = "bf16"  # "no" | "fp16" | "bf16"
    fp16_pure: bool = False
    grad_checkpoint: bool = False
    ui_run_dir: Optional[str] = None
    
    @property
    def algorithm(self) -> RLAlgorithm:
        """Возвращает тип алгоритма. Должен быть переопределён в наследниках."""
        raise NotImplementedError("Subclasses must define algorithm property")
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфиг в словарь для логирования."""
        return {
            "algorithm": self.algorithm.value,
            "group_size": self.group_size,
            "batch_size": self.batch_size,
            "train_batch_size": self.train_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
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
            "use_lora": self.use_lora,
            "use_4bit": self.use_4bit,
            "use_liger": self.use_liger,
        }
