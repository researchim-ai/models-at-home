"""
GRPOTrainer - основной класс для обучения GRPO/RL.

Реализует полный цикл обучения:
1. Генерация rollout'ов (completions)
2. Вычисление rewards и advantages
3. Обновление политики (модели)
4. Логирование и сохранение чекпоинтов
"""
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Union
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_cosine_schedule_with_warmup,
)

from .config import GRPOConfig, RLAlgorithm
from .experience import Experience, ReplayBuffer, join_experience_batch
from .loss import GRPOLoss, compute_advantages, compute_entropy
from .rollout import (
    generate_rollouts,
    rollout_to_experiences,
    build_reasoning_prompt,
    compute_log_probs,
)
from .rewards.base import RewardFunction, CombinedReward
from .rewards.math import GSM8KReward
from .rewards.format import FormatReward, ReasoningQualityReward
from .data.base import RLDataset, RLSample

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Настройка логирования."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="[%(asctime)s] [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int):
    """Устанавливает seed для воспроизводимости."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GRPOTrainer:
    """
    Trainer для обучения LLM с GRPO.
    
    Поддерживает:
    - Стандартный GRPO
    - Dr.GRPO (без std нормализации)
    - DAPO (clip higher, dynamic sampling)
    - LoRA для эффективного обучения
    - Multi-GPU через accelerate
    - W&B логирование
    
    Example:
        >>> from homellm.training.rl import GRPOConfig, GRPOTrainer
        >>> from homellm.training.rl.data import load_gsm8k
        >>> 
        >>> config = GRPOConfig.from_preset("reasoning_small")
        >>> dataset = load_gsm8k(split="train", max_samples=1000)
        >>> 
        >>> trainer = GRPOTrainer(
        ...     model_name="Qwen/Qwen2.5-0.5B-Instruct",
        ...     config=config,
        ... )
        >>> trainer.train(dataset)
    """
    
    def __init__(
        self,
        model_name: str,
        config: Optional[GRPOConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        reward_fn: Optional[RewardFunction] = None,
        device: Optional[torch.device] = None,
        use_accelerate: bool = True,
    ):
        """
        Args:
            model_name: Название модели или путь
            config: Конфигурация GRPO
            tokenizer: Токенизатор (опционально, загрузится автоматически)
            reward_fn: Функция reward (опционально, будет создана по умолчанию)
            device: Устройство (опционально)
            use_accelerate: Использовать accelerate для multi-GPU
        """
        self.model_name = model_name
        self.config = config or GRPOConfig()
        self.use_accelerate = use_accelerate
        
        # Устанавливаем seed
        set_seed(self.config.seed)
        
        # Определяем устройство
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Загружаем модель и токенизатор
        self.tokenizer = tokenizer
        self.model = None
        self.reference_model = None
        
        # Reward функция
        if reward_fn is None:
            # По умолчанию: комбинация format + correctness
            self.reward_fn = CombinedReward([
                FormatReward(weight=1.0),
                ReasoningQualityReward(weight=0.5),
                GSM8KReward(weight=2.0),
            ])
        else:
            self.reward_fn = reward_fn
        
        # Компоненты обучения (инициализируются в setup())
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.replay_buffer = None
        self.accelerator = None
        
        # Метрики
        self.global_step = 0
        self.total_rollouts = 0
        self.best_mean_reward = float("-inf")
        
        # W&B
        self.wandb_run = None
    
    def setup(self):
        """Инициализирует все компоненты для обучения."""
        logger.info("Инициализация GRPOTrainer...")
        
        # Accelerate
        if self.use_accelerate:
            try:
                from accelerate import Accelerator
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    mixed_precision="bf16" if torch.cuda.is_available() else "no",
                )
                self.device = self.accelerator.device
                self.is_main_process = self.accelerator.is_main_process
            except ImportError:
                logger.warning("accelerate не установлен, используем single GPU")
                self.is_main_process = True
        else:
            self.is_main_process = True
        
        # Токенизатор
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Загружаем модель
        self._load_model()
        
        # Loss функция
        self.loss_fn = GRPOLoss(config=self.config)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # W&B
        if self.config.use_wandb and self.is_main_process:
            self._setup_wandb()
        
        logger.info(f"GRPOTrainer инициализирован на {self.device}")
        logger.info(f"Алгоритм: {self.config.algorithm.value}")
    
    def _load_model(self):
        """Загружает модель с опциональной квантизацией и LoRA."""
        logger.info(f"Загрузка модели {self.model_name}...")
        
        # Конфигурация квантизации
        quantization_config = None
        if self.config.use_4bit or self.config.use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                
                if self.config.use_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
            except ImportError:
                logger.warning("bitsandbytes не установлен, квантизация отключена")
        
        # Загрузка модели
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if quantization_config else None,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        
        # LoRA
        if self.config.use_lora:
            self._apply_lora()
        
        # Референсная модель (для KL)
        if self.config.kl_weight > 0:
            logger.info("Загрузка референсной модели для KL...")
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        
        # Перемещаем на устройство (если не device_map)
        if not quantization_config:
            self.model = self.model.to(self.device)
            if self.reference_model:
                self.reference_model = self.reference_model.to(self.device)
        
        # Подсчёт параметров
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Параметры модели: {total_params:,} всего, {trainable_params:,} обучаемых")
    
    def _apply_lora(self):
        """Применяет LoRA адаптеры к модели."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Подготовка для квантизированной модели
            if self.config.use_4bit or self.config.use_8bit:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=True,
                )
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        except ImportError:
            logger.warning("peft не установлен, LoRA отключено")
            self.config.use_lora = False
    
    def _setup_wandb(self):
        """Настраивает Weights & Biases логирование."""
        try:
            import wandb
            
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                config=self.config.to_dict(),
                name=f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            logger.info(f"W&B инициализирован: {wandb.run.name}")
            
        except ImportError:
            logger.warning("wandb не установлен")
            self.config.use_wandb = False
    
    def _setup_optimizer(self, num_training_steps: int):
        """Настраивает оптимизатор и scheduler."""
        # Оптимизатор (только для обучаемых параметров)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        try:
            from bitsandbytes.optim import AdamW8bit
            self.optimizer = AdamW8bit(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        except ImportError:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        
        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Accelerate prepare
        if self.accelerator:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )
    
    def train(
        self,
        dataset: RLDataset,
        eval_dataset: Optional[RLDataset] = None,
    ):
        """
        Основной цикл обучения GRPO.
        
        Args:
            dataset: Тренировочный датасет
            eval_dataset: Валидационный датасет (опционально)
        """
        self.setup()
        
        # Вычисляем количество шагов
        num_prompts = len(dataset)
        steps_per_epoch = math.ceil(num_prompts / self.config.batch_size)
        total_steps = steps_per_epoch * self.config.num_epochs
        
        if self.config.max_steps:
            total_steps = min(total_steps, self.config.max_steps)
        
        logger.info(f"Начало обучения: {num_prompts} промптов, ~{total_steps} шагов")
        
        # Настройка оптимизатора
        self._setup_optimizer(total_steps)
        
        # Создаём output директорию
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # DataLoader для промптов
        prompt_loader = DataLoader(
            list(range(len(dataset))),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        # Основной цикл
        for epoch in range(self.config.num_epochs):
            epoch_metrics = self._train_epoch(
                dataset=dataset,
                prompt_loader=prompt_loader,
                epoch=epoch,
                eval_dataset=eval_dataset,
            )
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} завершена")
            logger.info(f"  Mean reward: {epoch_metrics.get('mean_reward', 0):.4f}")
            
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                logger.info("Достигнут max_steps, останавливаем обучение")
                break
        
        # Финальное сохранение
        if self.is_main_process:
            self._save_checkpoint(output_dir / "final", is_final=True)
        
        logger.info("Обучение завершено!")
        
        if self.wandb_run:
            self.wandb_run.finish()
    
    def _train_epoch(
        self,
        dataset: RLDataset,
        prompt_loader: DataLoader,
        epoch: int,
        eval_dataset: Optional[RLDataset] = None,
    ) -> Dict[str, float]:
        """Один epoch обучения."""
        epoch_rewards = []
        epoch_losses = []
        
        pbar = tqdm(
            prompt_loader,
            desc=f"Epoch {epoch + 1}",
            disable=not self.is_main_process,
        )
        
        for batch_idx, prompt_indices in enumerate(pbar):
            # Получаем промпты и ответы
            batch_samples = [dataset[i] for i in prompt_indices]
            prompts = [
                build_reasoning_prompt(
                    s.prompt,
                    self.tokenizer,
                    self.config.reasoning_format,
                )
                for s in batch_samples
            ]
            reference_answers = [s.reference_answer for s in batch_samples]
            
            # Генерация rollout'ов
            self.replay_buffer.clear()
            batch_rewards = self._generate_and_collect(
                prompts=prompts,
                reference_answers=reference_answers,
            )
            epoch_rewards.extend(batch_rewards)
            
            # Обучение на собранном опыте
            train_metrics = self._train_on_buffer()
            epoch_losses.append(train_metrics.get("loss", 0))
            
            # Логирование
            if self.global_step % self.config.log_steps == 0:
                self._log_metrics({
                    "step": self.global_step,
                    "epoch": epoch,
                    "batch_reward_mean": sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0,
                    **train_metrics,
                })
            
            # Обновляем progress bar
            pbar.set_postfix({
                "reward": f"{sum(batch_rewards) / max(len(batch_rewards), 1):.3f}",
                "loss": f"{train_metrics.get('loss', 0):.4f}",
            })
            
            # Сохранение чекпоинта
            if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                if self.is_main_process:
                    self._save_checkpoint(
                        Path(self.config.output_dir) / f"step_{self.global_step}"
                    )
            
            # Проверяем max_steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        # Валидация
        if eval_dataset and self.is_main_process:
            eval_metrics = self._evaluate(eval_dataset)
            logger.info(f"Validation: {eval_metrics}")
            self._log_metrics({"val/" + k: v for k, v in eval_metrics.items()})
        
        return {
            "mean_reward": sum(epoch_rewards) / max(len(epoch_rewards), 1),
            "mean_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
        }
    
    def _generate_and_collect(
        self,
        prompts: List[str],
        reference_answers: List[str],
    ) -> List[float]:
        """
        Генерирует rollout'ы и собирает опыт в буфер.
        
        Returns:
            Список всех rewards
        """
        self.model.eval()
        all_rewards = []
        
        # Обёртка для reward функции
        def reward_wrapper(completion, reference_answer, reasoning_format, is_truncated):
            return self.reward_fn(
                completion=completion,
                reference_answer=reference_answer,
                reasoning_format=reasoning_format,
                is_truncated=is_truncated,
            )
        
        # Генерируем rollout'ы
        rollouts = generate_rollouts(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            reference_answers=reference_answers,
            reward_fn=reward_wrapper,
            config=self.config,
            reference_model=self.reference_model,
            device=self.device,
        )
        
        # Конвертируем в Experience и добавляем в буфер
        for rollout in rollouts:
            experiences = rollout_to_experiences(
                rollout=rollout,
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.config,
                reference_model=self.reference_model,
                device=self.device,
            )
            
            # Dynamic sampling: фильтруем zero-gradient группы
            filter_zero = self.config.dynamic_sampling
            added = self.replay_buffer.append_group(
                experiences,
                prompt_id=rollout.metadata.get("prompt_idx", 0),
                filter_zero_gradient=filter_zero,
            )
            
            if added:
                all_rewards.extend(rollout.rewards.tolist())
            
            self.total_rollouts += len(rollout.completions)
        
        return all_rewards
    
    def _train_on_buffer(self) -> Dict[str, float]:
        """
        Обучение на собранном опыте в буфере.
        
        Returns:
            Метрики обучения
        """
        self.model.train()
        
        if len(self.replay_buffer) == 0:
            return {"loss": 0}
        
        # DataLoader для experience
        exp_loader = DataLoader(
            self.replay_buffer.items,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )
        
        epoch_losses = []
        epoch_kls = []
        epoch_grad_norms = []
        
        for _ in range(self.config.epochs_per_step):
            for exp_batch in exp_loader:
                exp_batch = exp_batch.to(self.device)
                
                # Forward pass для текущей политики
                with torch.cuda.amp.autocast(enabled=self.accelerator is not None):
                    log_probs = compute_log_probs(
                        self.model,
                        exp_batch.sequences,
                        exp_batch.attention_mask,
                    )
                    
                    loss, metrics = self.loss_fn(
                        log_probs=log_probs,
                        experience=exp_batch,
                    )
                
                if not loss.isfinite():
                    logger.warning(f"Loss не finite: {loss.item()}, пропускаем batch")
                    continue
                
                # Backward
                if self.accelerator:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                
                # Gradient clipping
                grad_norm = clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Собираем метрики
                epoch_losses.append(loss.item())
                epoch_kls.append(metrics.get("kl_mean", 0))
                epoch_grad_norms.append(
                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                )
                
                self.global_step += 1
        
        return {
            "loss": sum(epoch_losses) / max(len(epoch_losses), 1),
            "kl": sum(epoch_kls) / max(len(epoch_kls), 1),
            "grad_norm": sum(epoch_grad_norms) / max(len(epoch_grad_norms), 1),
        }
    
    @torch.no_grad()
    def _evaluate(
        self,
        dataset: RLDataset,
        max_samples: int = 100,
    ) -> Dict[str, float]:
        """Оценка на валидационном датасете."""
        self.model.eval()
        
        # Берём подвыборку
        indices = list(range(min(len(dataset), max_samples)))
        samples = [dataset[i] for i in indices]
        
        correct = 0
        total = 0
        rewards = []
        
        for sample in samples:
            prompt = build_reasoning_prompt(
                sample.prompt,
                self.tokenizer,
                self.config.reasoning_format,
            )
            
            # Генерируем один ответ (greedy)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_prompt_length,
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  # Greedy для eval
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            completion = self.tokenizer.decode(
                outputs[0, inputs["input_ids"].size(1):],
                skip_special_tokens=True,
            )
            
            reward = self.reward_fn(
                completion=completion,
                reference_answer=sample.reference_answer,
                reasoning_format=self.config.reasoning_format,
            )
            rewards.append(reward)
            
            if reward >= 0.5:  # Threshold для "правильного" ответа
                correct += 1
            total += 1
        
        return {
            "accuracy": correct / max(total, 1),
            "mean_reward": sum(rewards) / max(len(rewards), 1),
            "samples": total,
        }
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Логирует метрики."""
        if self.config.use_wandb and self.wandb_run:
            import wandb
            wandb.log(metrics, step=self.global_step)
        
        # Также логируем в консоль (основные метрики)
        if self.is_main_process:
            log_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
                if k in ["step", "loss", "kl", "batch_reward_mean"]
            ])
            if log_str:
                logger.info(f"Step {self.global_step}: {log_str}")
            
            # Записываем в JSONL для мониторинга из UI
            metrics_file = Path(self.config.output_dir) / "metrics.jsonl"
            try:
                import json
                with open(metrics_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": self.global_step,
                        "reward": metrics.get("batch_reward_mean", 0),
                        "loss": metrics.get("loss", 0),
                        "kl": metrics.get("kl", 0),
                        "timestamp": datetime.now().isoformat(),
                    }) + "\n")
            except Exception:
                pass
    
    def _save_checkpoint(self, path: Path, is_final: bool = False):
        """Сохраняет чекпоинт."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем модель (с LoRA адаптерами если используются)
        if self.config.use_lora:
            self.model.save_pretrained(path)
        else:
            if self.accelerator:
                unwrapped = self.accelerator.unwrap_model(self.model)
                unwrapped.save_pretrained(path)
            else:
                self.model.save_pretrained(path)
        
        # Сохраняем токенизатор
        self.tokenizer.save_pretrained(path)
        
        # Сохраняем конфиг
        import json
        with open(path / "grpo_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Чекпоинт сохранён: {path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Генерирует ответ для одного промпта.
        
        Args:
            prompt: Вопрос/задача
            max_new_tokens: Максимальное количество токенов
            temperature: Температура сэмплирования
            do_sample: Использовать сэмплирование
            
        Returns:
            Сгенерированный ответ
        """
        self.model.eval()
        
        formatted_prompt = build_reasoning_prompt(
            prompt,
            self.tokenizer,
            self.config.reasoning_format,
        )
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        completion = self.tokenizer.decode(
            outputs[0, inputs["input_ids"].size(1):],
            skip_special_tokens=True,
        )
        
        return completion
