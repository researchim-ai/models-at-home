"""
Конфигурация для GRPO/RL обучения.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Literal
from pathlib import Path


class RLAlgorithm(str, Enum):
    """Поддерживаемые алгоритмы RL."""
    GRPO = "grpo"           # Стандартный GRPO с нормализацией по std
    DRGRPO = "drgrpo"       # Dr.GRPO - без деления на std, фиксированная нормализация
    DAPO = "dapo"           # DAPO - clip higher + dynamic sampling + token-level loss


@dataclass
class GRPOConfig:
    """
    Конфигурация для обучения GRPO/RL.
    
    Содержит все гиперпараметры, описанные в:
    - DeepSeek-R1 (оригинальный GRPO)
    - Dr.GRPO (Understanding R1-Zero)
    - DAPO (GRPO++)
    
    Attributes:
        algorithm: Алгоритм RL (grpo, drgrpo, dapo)
        
        # Размеры батча и генерации
        group_size: Количество генераций на один промпт (G в статьях)
        batch_size: Размер батча промптов на один шаг
        train_batch_size: Размер батча для обучения (после сбора опыта)
        gradient_accumulation_steps: Накопление градиентов
        
        # Параметры генерации
        max_new_tokens: Максимальное количество генерируемых токенов
        temperature: Температура сэмплирования
        top_p: Top-p для nucleus sampling
        
        # Клиппинг (PPO-style)
        clip_eps_low: Нижняя граница клиппинга (default: 0.2)
        clip_eps_high: Верхняя граница клиппинга (DAPO рекомендует 0.28)
        
        # KL регуляризация
        kl_weight: Вес KL-штрафа (в reasoning-RL обычно 0 или очень маленький)
        
        # Оптимизатор
        learning_rate: Скорость обучения
        weight_decay: Weight decay для AdamW
        max_grad_norm: Максимальная норма градиента для клиппинга
        
        # Обучение
        num_epochs: Количество эпох RL обучения
        epochs_per_step: Количество эпох оптимизации на один батч роллаутов
        warmup_steps: Количество шагов warmup
        
        # Dr.GRPO специфичные параметры
        use_std_normalization: Использовать деление на std (True для GRPO, False для DrGRPO)
        fixed_length_normalizer: Фиксированный делитель для нормализации loss (DrGRPO)
        
        # DAPO специфичные параметры
        dynamic_sampling: Фильтровать zero-gradient группы
        token_level_loss: Использовать token-level агрегацию (vs sample-level)
        overlong_penalty: Штраф за слишком длинные ответы
        overlong_buffer: Буферная зона для штрафа за длину
        
        # Логирование и сохранение
        output_dir: Директория для сохранения моделей
        save_steps: Сохранять модель каждые N шагов
        log_steps: Логировать каждые N шагов
        use_wandb: Использовать Weights & Biases
        wandb_project: Название проекта в W&B
        
        # Система
        mixed_precision: Mixed precision режим ("no" | "fp16" | "bf16"). Должен совпадать с UI.
        use_lora: Использовать LoRA адаптеры
        lora_r: Ранг LoRA
        lora_alpha: Alpha для LoRA
        use_4bit: 4-bit квантизация (bitsandbytes)
        use_8bit: 8-bit квантизация
        use_flash_attention: Использовать Flash Attention 2
    """
    
    # Алгоритм (DAPO рекомендуется)
    algorithm: RLAlgorithm = RLAlgorithm.DAPO
    
    # Размеры батча
    group_size: int = 8
    batch_size: int = 8  # промптов на шаг роллаута
    train_batch_size: int = 2  # для обучения - уменьшено до 2 чтобы избежать OOM при 1200 токенах
    gradient_accumulation_steps: int = 4
    
    # Параметры генерации
    max_new_tokens: int = 1024
    max_prompt_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    
    # Клиппинг
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.2  # Для DAPO: 0.28
    
    # KL регуляризация
    kl_weight: float = 0.0  # Для reasoning-RL обычно 0
    
    # Оптимизатор
    learning_rate: float = 5e-6
    # Нижний предел LR: lr = base_lr * min_lr_ratio в конце cosine (0.0 = до нуля)
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Обучение
    num_epochs: int = 1
    epochs_per_step: int = 1
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    # Лимит по данным (сколько промптов обработать всего). Если None — идём по датасету (с учётом num_epochs).
    # Это ближе к человеческому ожиданию "сколько примеров прошло".
    max_prompts: Optional[int] = None
    
    # Dr.GRPO
    use_std_normalization: bool = True  # False для DrGRPO
    fixed_length_normalizer: Optional[int] = None  # max_new_tokens для DrGRPO
    
    # DAPO
    dynamic_sampling: bool = False  # True для DAPO
    max_refill_rounds: int = 3  # Макс. попыток добора групп (0 = без добора, 8 = как в статье)
    token_level_loss: bool = False  # True для DAPO
    overlong_penalty: float = -1.0
    overlong_buffer: int = 0  # 4000 в DAPO для длинных контекстов
    
    # ============================================================
    # ОПТИМИЗАЦИИ ГЕНЕРАЦИИ
    # ============================================================
    
    # Prefix Grouper: shared KV-cache для G completions одного промпта
    # Даёт 2-3x ускорение генерации (не работает с ZeRO-3)
    use_prefix_grouper: bool = True
    
    # Multi-prompt batching: сколько промптов генерировать одним батчем
    # 1 = отключено (по умолчанию). 2-4 = хороший выбор для GPU с запасом памяти.
    # Не совместимо с Prefix Grouper (взаимоисключающие).
    rollout_batch_size: int = 1
    
    # ds3_gather_for_generation: сбор параметров ZeRO-3 перед генерацией
    # Даёт 10-100x ускорение для ZeRO-3 (автоматически включается при ZeRO-3)
    ds3_gather_for_generation: bool = True

    # ============================================================
    # ROLLOUT ENGINE (отдельная модель для генерации)
    # ============================================================
    # Если True — генерация (rollout) будет делаться отдельной моделью (HF / vLLM),
    # а training-модель (DDP/ZeRO/FSDP) будет использоваться только для teacher-forcing logprobs + backprop.
    use_rollout_engine: bool = False

    # Backend для rollout engine:
    # - "hf": отдельная HF-модель (быстро для DDP, совместимо с ZeRO-3 через редкий sync весов)
    # - "vllm": (экспериментально) vLLM rollout (лучший throughput, но требует отдельной интеграции)
    rollout_engine_backend: Literal["hf", "vllm"] = "hf"

    # Как часто синхронизировать веса training->rollout (в единицах rollout_step).
    # 1 = каждый rollout-step (самое on-policy), 2-10 = быстрее (меньше overhead), но слегка "stale".
    rollout_sync_interval: int = 1

    # Если True — синхронизируем только trainable параметры (например LoRA), а не весь base model.
    # Это критично для ZeRO-3/FSDP, т.к. полный state_dict дорогой.
    rollout_sync_trainable_only: bool = True

    # Управление памятью: можно держать rollout модель на CPU между генерациями, чтобы освобождать VRAM.
    rollout_offload_to_cpu: bool = False

    # Устройство для rollout-модели:
    # - "auto": local accelerator.device (cuda:local_rank)
    # - "cpu": всегда CPU (медленно, но стабильно)
    rollout_device: str = "auto"
    
    # vLLM GPU memory utilization (0.0-1.0)
    # Сколько % памяти GPU выделить под vLLM (KV-cache и модель)
    vllm_gpu_memory_utilization: float = 0.85
    
    # vLLM device: на каком устройстве запускать vLLM
    # - "cuda:0", "cuda:1", ... — конкретная GPU
    # - "cpu" — на CPU (медленно, но освобождает GPU для training)
    # vLLM загружается ТОЛЬКО на main process (rank 0), результаты broadcast'ятся
    vllm_device: str = "cuda:0"
    
    # ============================================================
    # LIGER KERNEL OPTIMIZATIONS
    # ============================================================
    # Liger Kernel — оптимизированные Triton kernels для LLM тренировки
    # Экономит память (не материализует logits) и ускоряет вычисления
    # https://github.com/linkedin/Liger-Kernel
    
    # Включить Liger оптимизации (CrossEntropy, RMSNorm, MLP и т.д.)
    use_liger: bool = True
    
    # Патчить модель Liger kernels (RMSNorm -> LigerRMSNorm, MLP -> LigerSwiGLU и т.д.)
    # Поддерживаемые архитектуры: Qwen2, Llama, Mistral, Gemma, Phi3
    liger_patch_model: bool = True
    
    # Размер chunk'а для chunked cross-entropy (уменьшить если OOM)
    liger_chunk_size: int = 4096
    
    # 🔥 LigerFusedLinearGRPOLoss — НЕ материализует logits (экономия до 80% памяти!)
    # Fused: hidden_states -> lm_head -> GRPO loss в одном kernel
    # ВАЖНО: требует output_hidden_states=True в forward pass
    liger_fused_grpo: bool = True
    
    # Тип loss для Liger GRPO (если liger_fused_grpo=True):
    # - "grpo": стандартный GRPO (sample-level нормализация)
    # - "dapo": DAPO с улучшенной нормализацией (рекомендуется)
    # - "dr_grpo": DrGRPO с фиксированной нормализацией
    # - "bnpo": Batch Normalized Per-token loss
    liger_grpo_loss_type: str = "dapo"  # DAPO рекомендуется как более стабильный
    
    # Логирование
    output_dir: str = "./output/grpo"
    save_steps: int = 100
    log_steps: int = 10
    # Если True — обновляем `final_model/` при каждом сохранении чекпоинта, чтобы модель можно было сразу использовать.
    export_on_checkpoint: bool = True
    use_wandb: bool = False
    wandb_project: str = "homellm-grpo"
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    # Если lora_target_modules пустой список, будет использован "all-linear" (автоматическое определение)
    
    # Квантизация
    use_4bit: bool = False
    use_8bit: bool = False
    quantize_reference_model: bool = False  # Квантизировать ли reference модель (по умолчанию False для точности KL)
    use_flash_attention: bool = True  # По умолчанию True; flash-attn опционален (в Docker ставится отдельно)
    
    # Формат reasoning
    reasoning_format: str = "deepseek"  # "deepseek" (<think>), "simple" (<reasoning>)
    
    # Seed
    seed: int = 42

    # Precision (должно приходить из UI/distributed_config)
    mixed_precision: str = "bf16"  # "no" | "fp16" | "bf16"
    # Если True и mixed_precision="fp16": "pure fp16" (веса fp16, без GradScaler).
    # Это снижает VRAM (сравнимо с bf16), но может быть менее стабильным, чем AMP fp16.
    fp16_pure: bool = False
    
    # Memory
    grad_checkpoint: bool = False

    # UI/monitoring plumbing (опционально)
    # Если обучение запущено из Streamlit, UI создаёт run_dir (RUNS_DIR/<run_id>).
    # Чтобы мониторинг работал "железно" даже при различиях путей (/app vs host),
    # можем дублировать metrics.jsonl/samples.jsonl в эту директорию.
    ui_run_dir: Optional[str] = None
    
    def __post_init__(self):
        """Применяем настройки по умолчанию в зависимости от алгоритма."""
        if isinstance(self.algorithm, str):
            self.algorithm = RLAlgorithm(self.algorithm)
        
        if self.algorithm == RLAlgorithm.DRGRPO:
            # Dr.GRPO: без деления на std, фиксированная нормализация
            self.use_std_normalization = False
            if self.fixed_length_normalizer is None:
                self.fixed_length_normalizer = self.max_new_tokens
                
        elif self.algorithm == RLAlgorithm.DAPO:
            # DAPO: clip higher, dynamic sampling, token-level loss
            self.clip_eps_high = max(self.clip_eps_high, 0.28)
            self.dynamic_sampling = True
            self.token_level_loss = True
            self.use_std_normalization = False
            if self.fixed_length_normalizer is None:
                self.fixed_length_normalizer = self.max_new_tokens
    
    @classmethod
    def from_preset(cls, preset: str) -> "GRPOConfig":
        """
        Создаёт конфигурацию из предустановки.
        
        Args:
            preset: Название предустановки:
                - "grpo": Стандартный GRPO
                - "drgrpo": Dr.GRPO с фиксами
                - "dapo": Полный DAPO с clip higher и dynamic sampling
                - "reasoning_small": Для маленьких моделей (0.5-3B)
                - "reasoning_large": Для больших моделей (7B+)
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
    
    def to_dict(self) -> dict:
        """Конвертирует конфиг в словарь для логирования."""
        return {
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
            "use_std_normalization": self.use_std_normalization,
            "fixed_length_normalizer": self.fixed_length_normalizer,
            "dynamic_sampling": self.dynamic_sampling,
            "token_level_loss": self.token_level_loss,
            "use_lora": self.use_lora,
            "use_4bit": self.use_4bit,
        }
