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
        use_lora: Использовать LoRA адаптеры
        lora_r: Ранг LoRA
        lora_alpha: Alpha для LoRA
        use_4bit: 4-bit квантизация (bitsandbytes)
        use_8bit: 8-bit квантизация
        use_flash_attention: Использовать Flash Attention 2
    """
    
    # Алгоритм
    algorithm: RLAlgorithm = RLAlgorithm.GRPO
    
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
    token_level_loss: bool = False  # True для DAPO
    overlong_penalty: float = -1.0
    overlong_buffer: int = 0  # 4000 в DAPO для длинных контекстов
    
    # Логирование
    output_dir: str = "./output/grpo"
    save_steps: int = 100
    log_steps: int = 10
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
    use_flash_attention: bool = True  # По умолчанию True, flash-attn в requirements.txt
    
    # Формат reasoning
    reasoning_format: str = "deepseek"  # "deepseek" (<think>), "simple" (<reasoning>)
    
    # Seed
    seed: int = 42

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
            "max_steps": self.max_steps,
            "max_prompts": self.max_prompts,
            "use_std_normalization": self.use_std_normalization,
            "fixed_length_normalizer": self.fixed_length_normalizer,
            "dynamic_sampling": self.dynamic_sampling,
            "token_level_loss": self.token_level_loss,
            "use_lora": self.use_lora,
            "use_4bit": self.use_4bit,
        }
