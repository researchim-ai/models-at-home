#!/usr/bin/env python
"""
Скрипт для обучения reasoning на GSM8K с GRPO.

Пример использования:
    # Базовое обучение
    python -m homellm.training.rl.train_rl --model Qwen/Qwen2.5-0.5B-Instruct

    # С кастомными параметрами
    python -m homellm.training.rl.train_rl \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --algorithm drgrpo \
        --batch_size 4 \
        --group_size 8 \
        --max_samples 1000 \
        --output_dir ./output/grpo

    # С W&B логированием
    python -m homellm.training.rl.train_rl \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --use_wandb \
        --wandb_project my-grpo-experiments
"""
import argparse
import logging
from pathlib import Path
from datetime import datetime

from .legacy_config import GRPOConfig, RLAlgorithm
from .trainer import GRPOTrainer
from .data.gsm8k import load_gsm8k
from .rewards.base import CombinedReward, UniversalRuleReward
from .rewards.math import GSM8KReward
from .rewards.format import FormatReward, ReasoningQualityReward

logger = logging.getLogger(__name__)


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Обучение LLM reasoning на GSM8K с GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Модель
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Название модели HuggingFace или путь",
    )
    
    # Алгоритм
    parser.add_argument(
        "--algorithm",
        type=str,
        default="grpo",
        choices=["grpo", "drgrpo", "dapo", "sdpo"],
        help="Алгоритм RL",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["grpo", "drgrpo", "dapo", "sdpo", "reasoning_small", "reasoning_large"],
        help="Использовать предустановленную конфигурацию",
    )
    
    # Датасет
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Максимальное количество примеров (None = весь датасет)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split датасета",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="Путь к JSONL файлу вместо GSM8K (поля: prompt, answer)",
    )
    
    # Batch размеры
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Количество промптов на шаг",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Количество генераций на промпт",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help="Batch size для обучения",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Шагов накопления градиента",
    )
    
    # Генерация
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Максимум токенов в ответе",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Температура сэмплирования",
    )
    
    # Обучение
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Количество эпох",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Максимум шагов (None = по эпохам)",
    )
    
    # GRPO параметры
    parser.add_argument(
        "--clip_eps",
        type=float,
        default=0.2,
        help="Epsilon для PPO клиппинга",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.0,
        help="Вес KL штрафа (0 для reasoning)",
    )
    
    # LoRA
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Использовать LoRA",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Не использовать LoRA",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    
    # Квантизация
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Использовать 4-bit квантизацию",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Использовать 8-bit квантизацию",
    )
    
    # Логирование и сохранение
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Директория для сохранения (auto если не указано)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Сохранять каждые N шагов",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Логировать каждые N шагов",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Использовать W&B",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="homellm-grpo",
        help="Название проекта W&B",
    )
    
    # Формат
    parser.add_argument(
        "--reasoning_format",
        type=str,
        default="deepseek",
        choices=["deepseek", "simple", "russian"],
        help="Формат reasoning тегов",
    )
    
    # Разное
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # JSON конфигурация (для запуска из Streamlit UI)
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="JSON строка с полной конфигурацией (перезаписывает другие параметры)",
    )
    
    return parser.parse_args()


def main():
    """Основная функция."""
    import json
    
    args = parse_args()
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("GRPO Training на GSM8K")
    logger.info("=" * 60)
    
    # Если передана JSON конфигурация (из Streamlit UI)
    ui_config = None
    reward_rules = None
    
    if args.config_json:
        logger.info("Загрузка конфигурации из JSON...")
        ui_config = json.loads(args.config_json)
        
        # === Unsloth Backend ===
        training_backend = ui_config.get("training_backend", "models-at-home")
        if training_backend == "unsloth":
            logger.info("🦥 Using Unsloth backend for GRPO training")
            try:
                from homellm.training.unsloth_grpo import run_unsloth_grpo, is_unsloth_available
                
                if is_unsloth_available():
                    # Создаём простой metrics logger
                    from pathlib import Path
                    import time
                    
                    ui_run_dir = ui_config.get("ui_run_dir")
                    if ui_run_dir:
                        metrics_path = Path(ui_run_dir) / "metrics.json"
                    else:
                        metrics_path = Path(ui_config.get("output_dir", "out/grpo")) / "metrics.json"
                    
                    class SimpleMetricsLogger:
                        def __init__(self, path):
                            self.path = Path(path)
                            self.start_ts = time.time()
                            self.metrics = {
                                "status": "initializing", 
                                "start_time": datetime.now().isoformat(),
                                "elapsed_seconds": 0.0,
                                "eta_seconds": 0.0,
                                "steps_history": [],
                                "loss_history": [],
                                "lr_history": [],
                                "reward_history": [],
                                "kl_history": [],
                            }
                            self._save()
                        
                        def _save(self):
                            self.path.parent.mkdir(parents=True, exist_ok=True)
                            with open(self.path, "w") as f:
                                json.dump(self.metrics, f, indent=2)
                        
                        def update(self, **kwargs):
                            self.metrics.update(kwargs)
                            self._save()
                        
                        def log_step(self, step, loss, lr, samples_per_sec=0, reward=None, kl=None):
                            self.metrics["current_step"] = step
                            self.metrics["current_loss"] = loss
                            self.metrics["current_lr"] = lr
                            self.metrics["samples_per_second"] = samples_per_sec
                            elapsed = max(0.0, time.time() - self.start_ts)
                            self.metrics["elapsed_seconds"] = elapsed
                            total_steps = self.metrics.get("total_steps")
                            if samples_per_sec and total_steps:
                                try:
                                    remaining_steps = max(0.0, float(total_steps) - float(step))
                                    self.metrics["eta_seconds"] = remaining_steps / float(samples_per_sec)
                                except Exception:
                                    pass
                            
                            # Добавляем в историю
                            self.metrics["steps_history"].append(step)
                            self.metrics["loss_history"].append(loss)
                            self.metrics["lr_history"].append(lr)
                            
                            # GRPO специфичные метрики
                            if reward is not None:
                                self.metrics["current_reward"] = reward
                                self.metrics["reward_history"].append(reward)
                            if kl is not None:
                                self.metrics["current_kl"] = kl
                                self.metrics["kl_history"].append(kl)
                            
                            self._save()
                        
                        def log_checkpoint(self, path):
                            if "checkpoints" not in self.metrics:
                                self.metrics["checkpoints"] = []
                            self.metrics["checkpoints"].append({"path": path, "step": self.metrics.get("current_step", 0)})
                            self._save()
                    
                    metrics_logger = SimpleMetricsLogger(metrics_path)
                    run_unsloth_grpo(ui_config, metrics_logger)
                    return  # Успешно завершено
                else:
                    logger.warning("⚠️ Unsloth not available, falling back to models-at-home backend")
            except ImportError as e:
                logger.warning(f"⚠️ Could not import unsloth_grpo: {e}. Falling back to models-at-home backend.")
        
        # Извлекаем reward правила
        reward_rules = ui_config.get("grpo_reward_rules", [])
        if reward_rules:
            logger.info(f"Загружено {len(reward_rules)} reward правил из UI")
    
    # Создаём конфигурацию
    if args.preset:
        logger.info(f"Используем preset: {args.preset}")
        config = GRPOConfig.from_preset(args.preset)
    else:
        algorithm = args.algorithm
        if ui_config:
            algorithm = ui_config.get("grpo_algorithm", algorithm)
        config = GRPOConfig(
            algorithm=RLAlgorithm(algorithm),
        )
    
    # Переопределяем параметры из UI конфига
    # ВАЖНО: Все параметры должны быть явно переданы из UI, без fallback на args
    # Это гарантирует что мы точно знаем откуда берутся значения
    if ui_config:
        # GRPO параметры (обязательные из UI)
        if "grpo_prompt_batch_size" not in ui_config:
            raise ValueError("❌ Не задан grpo_prompt_batch_size (prompts/step) из UI.")
        config.batch_size = ui_config["grpo_prompt_batch_size"]

        config.group_size = ui_config["grpo_group_size"]
        if config.group_size < 8:
            raise ValueError("❌ group_size должен быть >= 8 для стабильного GRPO.")

        config.train_batch_size = ui_config["grpo_train_batch_size"]
        config.gradient_accumulation_steps = ui_config["gradient_accumulation"]
        config.max_new_tokens = ui_config["grpo_max_new_tokens"]
        config.temperature = ui_config["grpo_temperature"]
        config.learning_rate = ui_config["grpo_learning_rate"]
        config.min_lr_ratio = float(ui_config.get("grpo_min_lr_ratio", getattr(config, "min_lr_ratio", 0.0)))
        # Лимиты обучения:
        # - max_prompts: "сколько примеров пройти" (понятно пользователю)
        # - max_steps: legacy лимит по optimizer steps (если кто-то ещё передаёт старый ключ)
        config.max_prompts = ui_config.get("grpo_max_prompts", None)
        config.max_steps = ui_config.get("grpo_max_optim_steps", ui_config.get("grpo_max_steps", None))
        config.clip_eps_low = ui_config["grpo_clip_eps_low"]
        config.clip_eps_high = ui_config.get("grpo_clip_eps_high", config.clip_eps_low)
        config.kl_weight = ui_config["grpo_kl_weight"]
        config.epochs_per_step = ui_config.get("grpo_epochs_per_step", 1)
        config.reasoning_format = ui_config.get("grpo_reasoning_format", config.reasoning_format)
        # Precision должен приходить из UI (render_distributed_config -> full_config -> config_json)
        config.mixed_precision = (ui_config.get("mixed_precision") or config.mixed_precision)
        config.fp16_pure = bool(ui_config.get("fp16_pure", getattr(config, "fp16_pure", False)))
        config.use_flash_attention = bool(ui_config.get("use_flash_attention", getattr(config, "use_flash_attention", True)))
        # Memory: gradient checkpointing должен приходить из UI
        config.grad_checkpoint = bool(ui_config.get("grad_checkpoint", False))

        # Liger Kernel оптимизации — берём из общих настроек Precision & Memory
        config.use_liger = bool(ui_config.get("use_liger", getattr(config, "use_liger", True)))
        config.liger_patch_model = config.use_liger  # Всегда патчим если Liger включён
        config.liger_chunk_size = 4096  # Оптимальный размер
        
        # 🔥 Liger Fused GRPO Loss — автоматически если use_liger и liger_fused_ce включены
        liger_fused = bool(ui_config.get("liger_fused_ce", True))  # Fused Loss из общих настроек
        config.liger_fused_grpo = config.use_liger and liger_fused
        
        # Loss type автоматически из алгоритма (grpo→grpo, dapo→dapo, drgrpo→dr_grpo)
        config.liger_grpo_loss_type = ui_config.get("grpo_liger_loss_type", getattr(config, "liger_grpo_loss_type", "dapo"))

        # DAPO-специфичные параметры (UI может переопределить дефолты из __post_init__)
        if "grpo_dynamic_sampling" in ui_config:
            config.dynamic_sampling = bool(ui_config["grpo_dynamic_sampling"])
        if "grpo_max_refill_rounds" in ui_config:
            config.max_refill_rounds = int(ui_config["grpo_max_refill_rounds"])
        if "grpo_token_level_loss" in ui_config:
            config.token_level_loss = bool(ui_config["grpo_token_level_loss"])
        
        # 🎓 SDPO-специфичные параметры
        if "sdpo_success_threshold" in ui_config:
            config.sdpo_success_threshold = float(ui_config["sdpo_success_threshold"])
        if "sdpo_alpha" in ui_config:
            config.sdpo_alpha = float(ui_config["sdpo_alpha"])
        if "sdpo_loss_weight" in ui_config:
            config.sdpo_loss_weight = float(ui_config["sdpo_loss_weight"])
        # 🔥 SDPO Top-K Distillation и EMA (из verl)
        if "sdpo_distillation_topk" in ui_config:
            topk = ui_config["sdpo_distillation_topk"]
            config.sdpo_distillation_topk = int(topk) if topk is not None else None
        if "sdpo_full_logit_distillation" in ui_config:
            config.sdpo_full_logit_distillation = bool(ui_config["sdpo_full_logit_distillation"])
        if "sdpo_ema_rate" in ui_config:
            config.sdpo_ema_rate = float(ui_config["sdpo_ema_rate"])

        # Rollout engine (отдельная модель для генерации)
        config.use_rollout_engine = bool(ui_config.get("grpo_use_rollout_engine", getattr(config, "use_rollout_engine", False)))
        config.rollout_engine_backend = ui_config.get("grpo_rollout_backend", getattr(config, "rollout_engine_backend", "hf"))
        config.rollout_sync_interval = int(ui_config.get("grpo_rollout_sync_interval", getattr(config, "rollout_sync_interval", 1)))
        config.rollout_sync_trainable_only = bool(ui_config.get("grpo_rollout_trainable_only", getattr(config, "rollout_sync_trainable_only", True)))
        config.rollout_offload_to_cpu = bool(ui_config.get("grpo_rollout_offload_to_cpu", getattr(config, "rollout_offload_to_cpu", False)))
        config.vllm_gpu_memory_utilization = float(ui_config.get("grpo_vllm_gpu_memory", getattr(config, "vllm_gpu_memory_utilization", 0.85)))
        config.vllm_device = ui_config.get("grpo_vllm_device", getattr(config, "vllm_device", "main_gpu"))

        # Сохранение/логирование (из UI output_config)
        config.save_steps = int(ui_config.get("save_every", config.save_steps))
        config.log_steps = int(ui_config.get("log_every", config.log_steps))
        config.export_on_checkpoint = bool(ui_config.get("export_on_checkpoint", config.export_on_checkpoint))
        config.merge_lora = bool(ui_config.get("merge_lora", True))  # Merge LoRA при сохранении final_model

        # Путь до run_dir, который создал UI (для "железного" мониторинга).
        # Если задан, trainer будет дублировать metrics/samples в эту директорию.
        config.ui_run_dir = ui_config.get("ui_run_dir", ui_config.get("run_dir", None))
        
        # LoRA параметры (обязательные из UI, если use_lora=True)
        config.use_lora = ui_config.get("use_lora", config.use_lora)
        if config.use_lora:
            # ВАЖНО: Если use_lora=True, все LoRA параметры должны быть явно указаны в UI
            if "lora_r" not in ui_config or ui_config["lora_r"] is None:
                raise ValueError(
                    "❌ use_lora=True но lora_r не указан в UI конфиге! "
                    "Укажите lora_r в render_grpo_sidebar_config() или в model_config."
                )
            config.lora_r = ui_config["lora_r"]
            
            if "lora_alpha" not in ui_config or ui_config["lora_alpha"] is None:
                raise ValueError(
                    "❌ use_lora=True но lora_alpha не указан в UI конфиге! "
                    "Укажите lora_alpha в render_grpo_sidebar_config() или в model_config."
                )
            config.lora_alpha = ui_config["lora_alpha"]
            
            # lora_dropout и lora_target_modules могут быть None (используются дефолты из GRPOConfig)
            config.lora_dropout = ui_config.get("lora_dropout", config.lora_dropout)
            config.lora_target_modules = ui_config.get("lora_target_modules", config.lora_target_modules)
        
        # Квантизация (обязательные из UI)
        config.use_4bit = ui_config.get("use_4bit", False)
        config.use_8bit = ui_config.get("use_8bit", False)
        config.quantize_reference_model = ui_config.get("quantize_reference_model", config.quantize_reference_model)
        
        # Output директория
        config.output_dir = ui_config.get("output_dir", f"./output/grpo_gsm8k/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        config.batch_size = args.batch_size
        config.group_size = args.group_size
        config.train_batch_size = args.train_batch_size
        config.gradient_accumulation_steps = args.gradient_accumulation
        config.max_new_tokens = args.max_new_tokens
        config.temperature = args.temperature
        config.learning_rate = args.learning_rate
        config.num_epochs = args.num_epochs
        config.max_steps = args.max_steps
        config.clip_eps_low = args.clip_eps
        config.clip_eps_high = args.clip_eps if args.algorithm != "dapo" else 0.28
        config.kl_weight = args.kl_weight
        config.use_lora = args.use_lora and not args.no_lora
        config.lora_r = args.lora_r
        config.use_4bit = args.use_4bit
        config.use_8bit = args.use_8bit
        config.reasoning_format = args.reasoning_format
        
        # Output директория
        if args.output_dir:
            config.output_dir = args.output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.output_dir = f"./output/grpo_gsm8k/{timestamp}"
    
    config.save_steps = args.save_steps
    config.log_steps = args.log_steps
    config.use_wandb = args.use_wandb
    config.wandb_project = args.wandb_project
    config.seed = args.seed
    
    # Определяем модель
    model_name = args.model
    if ui_config:
        model_name = ui_config.get("base_model_path") or ui_config.get("model_name", model_name)
    
    logger.info(f"Модель: {model_name}")
    logger.info(f"Алгоритм: {config.algorithm.value}")
    logger.info(f"Output: {config.output_dir}")
    
    # Загружаем датасет
    dataset_file = args.dataset_file
    max_samples = args.max_samples
    dataset_language = "en"  # По умолчанию английский
    
    if ui_config:
        dataset_source = ui_config.get("grpo_dataset_source", "")
        dataset_key = ui_config.get("grpo_dataset_key", "gsm8k_en")
        dataset_language = ui_config.get("grpo_dataset_language", "en")
        
        if "GSM8K" in dataset_source or dataset_key in ("gsm8k_en", "gsm8k_ru"):
            dataset_file = None  # Используем GSM8K из HuggingFace
            max_samples = ui_config.get("grpo_max_samples", max_samples)
            # Определяем язык по ключу датасета
            if dataset_key == "gsm8k_ru":
                dataset_language = "ru"
        else:
            dataset_file = ui_config.get("grpo_dataset_path") or ui_config.get("data_path")
    
    # Определяем ключ датасета
    dataset_key = None
    if ui_config:
        dataset_key = ui_config.get("grpo_dataset_key")
    
    if dataset_file:
        logger.info(f"Загрузка датасета из файла: {dataset_file}")
        from .data.base import RLDataset, RLSample
        from .data.gsm8k import extract_gsm8k_final_answer
        
        # Получаем настройки маппинга полей из UI конфига
        field_mapping = {}
        prompt_template = "{{prompt}}"
        user_system_prompt = ""
        
        if ui_config:
            field_mapping = ui_config.get("grpo_field_mapping", {})
            prompt_template = ui_config.get("grpo_prompt_template", "{{prompt}}")
            user_system_prompt = ui_config.get("grpo_system_prompt", "")
        
        # Поля для чтения из датасета
        prompt_field = field_mapping.get("prompt_field", "question")
        reference_field = field_mapping.get("reference_field", "answer")
        metadata_fields = field_mapping.get("metadata_fields", [])
        
        # Fallback поля для совместимости
        prompt_fallbacks = ["question", "prompt", "input", "instruction", "problem", "query", "text"]
        reference_fallbacks = ["answer", "response", "output", "solution", "target", "completion"]
        
        logger.info(f"Маппинг полей: prompt={prompt_field}, reference={reference_field}")
        if prompt_template != "{{prompt}}":
            logger.info(f"Шаблон промпта: {prompt_template[:100]}...")
        if user_system_prompt:
            logger.info(f"System prompt: {user_system_prompt[:100]}...")
        
        samples = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    
                    # Извлекаем промпт с fallback
                    raw_prompt = data.get(prompt_field)
                    if raw_prompt is None:
                        for fb in prompt_fallbacks:
                            if fb in data:
                                raw_prompt = data[fb]
                                break
                    raw_prompt = str(raw_prompt) if raw_prompt else ""
                    
                    # Извлекаем референсный ответ с fallback
                    raw_answer = data.get(reference_field)
                    if raw_answer is None:
                        for fb in reference_fallbacks:
                            if fb in data:
                                raw_answer = data[fb]
                                break
                    raw_answer = str(raw_answer) if raw_answer else ""
                    
                    # Извлекаем финальный ответ (для GSM8K-стиля с ####)
                    ref_answer = raw_answer
                    if isinstance(raw_answer, str) and "####" in raw_answer:
                        ref_answer = extract_gsm8k_final_answer(raw_answer)
                    
                    # Применяем шаблон промпта
                    formatted_prompt = prompt_template
                    formatted_prompt = formatted_prompt.replace("{{prompt}}", raw_prompt)
                    formatted_prompt = formatted_prompt.replace("{{reference}}", ref_answer)
                    
                    # Собираем metadata из указанных полей
                    sample_metadata = {
                        "full_answer": raw_answer,
                        "raw_prompt": raw_prompt,
                    }
                    
                    # Добавляем дополнительные поля в metadata
                    for mf in metadata_fields:
                        if mf in data:
                            sample_metadata[mf] = data[mf]
                            # Также подставляем в шаблон
                            formatted_prompt = formatted_prompt.replace(f"{{{{metadata.{mf}}}}}", str(data[mf]))
                    
                    # Добавляем все поля из исходных данных в metadata для доступа в reward
                    for key, value in data.items():
                        if key not in sample_metadata:
                            sample_metadata[key] = value
                        formatted_prompt = formatted_prompt.replace(f"{{{{metadata.{key}}}}}", str(value))
                    
                    samples.append(RLSample(
                        prompt=formatted_prompt,
                        reference_answer=ref_answer,
                        metadata=sample_metadata,
                    ))
        
        if max_samples:
            samples = samples[:max_samples]
        train_dataset = RLDataset(samples)
        
        # Сохраняем user_system_prompt в конфиг для использования в trainer
        if user_system_prompt:
            config.user_system_prompt = user_system_prompt
    else:
        # Загружаем датасет из HuggingFace
        dataset_names = {
            "gsm8k_en": "GSM8K (English)",
            "gsm8k_ru": "GSM8K-RU (d0rj/gsm8k-ru)",
            "math_ru": "MATH-RU (d0rj/competition_math_ru)",
            "mgsm_ru": "MGSM (juletxara/mgsm)",
        }
        ds_name = dataset_names.get(dataset_key, f"GSM8K ({dataset_language})")
        logger.info(f"Загрузка датасета: {ds_name}...")
        
        train_dataset = load_gsm8k(
            split=args.split,
            max_samples=max_samples,
            reasoning_format=config.reasoning_format,
            language=dataset_language,
            dataset_key=dataset_key,
        )
    logger.info(f"Загружено {len(train_dataset)} примеров")
    if len(train_dataset) <= 0:
        raise ValueError(
            "❌ Датасет пустой (0 примеров). "
            "Проверьте, что выбран reasoning-датасет (GSM8K/GSM8K-RU/MATH-RU), "
            "и что в JSONL есть поля question/prompt и answer/response."
        )

    # Сохраняем размер датасета в output_dir, чтобы UI мог показывать прогресс по данным
    try:
        import json as _json
        from pathlib import Path as _Path
        _cfg_path = _Path(config.output_dir) / "dataset_info.json"
        with open(_cfg_path, "w", encoding="utf-8") as f:
            _json.dump({"dataset_size": int(len(train_dataset))}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    
    # Создаём reward функцию
    if reward_rules:
        # Используем универсальные правила из UI
        logger.info(f"Создание UniversalRuleReward из {len(reward_rules)} правил")
        for rule in reward_rules:
            logger.info(f"  - {rule.get('name')}: weight={rule.get('weight')}")
        reward_fn = UniversalRuleReward.from_config(reward_rules)
    else:
        # Стандартная конфигурация для GSM8K
        logger.info("Используем стандартную reward функцию для GSM8K")
        reward_fn = CombinedReward([
            FormatReward(format_reward=0.2, weight=1.0),
            ReasoningQualityReward(max_reward=0.2, weight=0.5),
            GSM8KReward(correct_reward=1.0, close_reward=0.3, weight=2.0),
        ])
    
    # Создаём trainer
    trainer = GRPOTrainer(
        model_name=model_name,
        config=config,
        reward_fn=reward_fn,
    )
    
    # Запускаем обучение
    logger.info("Начинаем обучение...")
    
    # 🔥 Логируем system prompt чтобы было видно что используется
    user_sys_prompt = getattr(config, 'user_system_prompt', None)
    if user_sys_prompt:
        logger.info(f"📝 System prompt (из UI): {user_sys_prompt[:200]}...")
    else:
        logger.info(f"📝 System prompt: (используется default из reasoning_format={config.reasoning_format})")
    
    trainer.train(train_dataset)
    
    logger.info("=" * 60)
    logger.info("Обучение завершено!")
    logger.info(f"Модель сохранена в: {config.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
