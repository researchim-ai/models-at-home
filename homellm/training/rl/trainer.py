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
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Union, Tuple
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

from .legacy_config import GRPOConfig, RLAlgorithm
from .experience import Experience, ReplayBuffer, join_experience_batch
from .loss import GRPOLoss, SDPOLoss, LigerFusedGRPOLoss, compute_advantages, compute_entropy, create_loss_function
from .rollout import (
    generate_rollouts,
    generate_rollouts_vllm,
    rollout_to_experiences,
    build_reasoning_prompt,
    compute_log_probs,
)
from .rollout_engine import HFRolloutEngine, VLLMRolloutEngine, VLLMSubprocessEngine
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
        
        # Устройство будет определено из accelerator в setup()
        # Не определяем device здесь, чтобы accelerator мог правильно настроить multi-GPU
        self._device = device  # Сохраняем для fallback если accelerate не используется
        
        # Загружаем модель и токенизатор
        self.tokenizer = tokenizer
        self.model = None
        self.reference_model = None

        # Rollout engine (отдельная модель для генерации)
        self.rollout_engine: Optional[HFRolloutEngine] = None
        self._rollout_last_sync_step: int = -10**9
        
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
        # Отдельный счётчик для rollout-батчей (prompts/step). Нужен для понятного мониторинга.
        self.rollout_step = 0
        # Уникальные ID для групп (нельзя использовать dataset index при dynamic sampling с добором)
        self._group_uid = 0
        # Кумулятивные счётчики для мониторинга "сколько примеров реально прошло"
        self.cum_prompts_generated = 0
        self.cum_prompts_used = 0
        self.cum_completions_generated = 0
        self.cum_experiences_tuned = 0

        # Прочие метрики/статусы
        self.total_rollouts = 0
        self.best_mean_reward = float("-inf")
        
        # W&B
        self.wandb_run = None

    # ---------------------------------------------------------------------
    # Rollout engine helpers
    # ---------------------------------------------------------------------
    def _get_train_module_for_sync(self) -> PreTrainedModel:
        """
        Возвращает "реальную" модель для доступа к named_parameters/state_dict.
        Для DDP/DeepSpeed обёрток используем accelerator.unwrap_model.
        """
        if self.accelerator is not None:
            try:
                return self.accelerator.unwrap_model(self.model)  # type: ignore[arg-type]
            except Exception:
                pass
        # fallback
        return getattr(self.model, "module", self.model)

    def _sync_rollout_engine_weights(self, *, force: bool = False) -> None:
        """
        Синхронизирует веса training->rollout.
        По умолчанию: только trainable параметры (LoRA), чтобы работало быстро и с ZeRO-3.
        """
        if not getattr(self.config, "use_rollout_engine", False):
            return
        if self.rollout_engine is None:
            return

        interval = max(int(getattr(self.config, "rollout_sync_interval", 1)), 1)
        if (not force) and (self.rollout_step - self._rollout_last_sync_step) < interval:
            return

        backend = getattr(self.config, "rollout_engine_backend", "hf")

        trainable_only = bool(getattr(self.config, "rollout_sync_trainable_only", True))
        train_mod = self._get_train_module_for_sync()

        state_dict = None

        # Distributed broadcast (rank0 -> all). Для single-GPU: просто локально.
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_dist else 0

        # vLLM backend: синхронизируем LoRA адаптер через save_pretrained (rank0) и broadcast path
        if backend == "vllm":
            if not bool(getattr(self.config, "use_lora", False)):
                raise RuntimeError("vLLM rollout backend сейчас поддержан только для LoRA (use_lora=True).")
            if not trainable_only:
                raise RuntimeError("vLLM rollout backend: full weight sync не поддержан. Используйте trainable-only (LoRA).")

            adapter_path = None
            adapter_name = None
            adapter_int_id = None

            if rank == 0:
                # Собираем trainable params (LoRA) на rank0 при ZeRO-3 и сохраняем адаптер.
                try:
                    from peft.utils import get_peft_model_state_dict
                except Exception as e:
                    raise RuntimeError("peft нужен для vLLM LoRA sync. Установите peft.") from e

                peft_model = train_mod
                if not hasattr(peft_model, "peft_config"):
                    raise RuntimeError("Модель не выглядит как PEFT/LoRA модель, но use_lora=True. Проверьте загрузку LoRA.")

                out_dir = Path(self.config.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                adapter_dir = out_dir / "rollout_engine" / "vllm_adapters" / f"step_{int(self.rollout_step)}"
                adapter_dir.mkdir(parents=True, exist_ok=True)

                # Gather trainable params for ZeRO-3
                params = [(n, p) for n, p in peft_model.named_parameters() if getattr(p, "requires_grad", False)]
                if getattr(self, "is_deepspeed_zero3", False) and params:
                    from deepspeed.runtime.zero.partition_parameters import GatheredParameters
                    with GatheredParameters([p for _, p in params], modifier_rank=0):
                        lora_sd = get_peft_model_state_dict(peft_model)
                else:
                    lora_sd = get_peft_model_state_dict(peft_model)

                # save_pretrained with provided state_dict avoids reading partitioned weights
                peft_model.save_pretrained(str(adapter_dir), state_dict=lora_sd, safe_serialization=True)
                
                # Проверяем что адаптер сохранился
                adapter_config_path = adapter_dir / "adapter_config.json"
                if adapter_config_path.exists():
                    logger.info(f"🧩 LoRA adapter saved to {adapter_dir}")
                    # Логируем содержимое директории
                    saved_files = list(adapter_dir.iterdir())
                    logger.info(f"🧩 Saved files: {[f.name for f in saved_files]}")
                else:
                    logger.error(f"🧩 ERROR: adapter_config.json not found after save_pretrained!")

                adapter_path = str(adapter_dir)
                adapter_name = "rollout_lora"
                # ВАЖНО: используем ФИКСИРОВАННЫЙ id=1 чтобы vLLM перезаписывал адаптер
                # Уникальные id накапливаются в памяти vLLM и вызывают CUDA OOM
                adapter_int_id = 1

            if is_dist:
                obj_list = [(adapter_path, adapter_name, adapter_int_id)]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                adapter_path, adapter_name, adapter_int_id = obj_list[0]

            if adapter_path is None:
                raise RuntimeError("vLLM adapter sync failed: adapter_path is None")

            # Apply LoRA adapter to vLLM engine (поддерживаем оба типа)
            if isinstance(self.rollout_engine, (VLLMRolloutEngine, VLLMSubprocessEngine)):
                self.rollout_engine.set_lora_adapter(
                    lora_path=adapter_path,
                    lora_name=adapter_name,
                    lora_int_id=int(adapter_int_id or 1),
                )
            else:
                raise RuntimeError(f"rollout_engine backend mismatch (expected VLLMRolloutEngine or VLLMSubprocessEngine, got {type(self.rollout_engine).__name__})")

            if self.is_main_process:
                logger.info(f"🧩 RolloutEngine(vLLM) sync: adapter={adapter_path}")

            self._rollout_last_sync_step = int(self.rollout_step)
            return

        # HF backend: state_dict sync
        if rank == 0:
            if trainable_only:
                params = [(n, p) for n, p in train_mod.named_parameters() if getattr(p, "requires_grad", False)]
                if not params:
                    logger.warning("RolloutEngine sync: нет trainable параметров, fallback -> full state_dict")
                    state_dict = {k: v.detach().cpu() for k, v in train_mod.state_dict().items()}
                else:
                    if getattr(self, "is_deepspeed_zero3", False):
                        try:
                            from deepspeed.runtime.zero.partition_parameters import GatheredParameters
                            with GatheredParameters([p for _, p in params], modifier_rank=0):
                                state_dict = {n: p.detach().cpu() for n, p in params}
                        except Exception as e:
                            logger.warning(f"RolloutEngine sync (ZeRO-3 trainable-only) failed: {e}. Fallback -> full state_dict")
                            state_dict = {k: v.detach().cpu() for k, v in train_mod.state_dict().items()}
                    else:
                        state_dict = {n: p.detach().cpu() for n, p in params}
            else:
                if self.accelerator is not None:
                    try:
                        state_dict = {k: v.detach().cpu() for k, v in self.accelerator.get_state_dict(self.model).items()}
                    except Exception:
                        state_dict = {k: v.detach().cpu() for k, v in train_mod.state_dict().items()}
                else:
                    state_dict = {k: v.detach().cpu() for k, v in train_mod.state_dict().items()}

        if is_dist:
            obj_list = [state_dict]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            state_dict = obj_list[0]

        if state_dict is None:
            raise RuntimeError("RolloutEngine sync failed: state_dict is None")

        # Применяем state_dict в rollout engine (HF)
        strict = not trainable_only
        stats = self.rollout_engine.apply_state_dict(state_dict, strict=strict)

        if self.is_main_process:
            mode = "trainable-only" if trainable_only else "full"
            logger.info(
                f"🧩 RolloutEngine sync: mode={mode}, keys={stats.synced_keys}, "
                f"~numel={stats.total_numel:,}, interval={interval}"
            )

        self._rollout_last_sync_step = int(self.rollout_step)

    def _create_loss_function(self) -> None:
        """
        Создаёт loss функцию в зависимости от алгоритма.
        
        Поддерживаемые алгоритмы:
        - SDPO: SDPOLoss (GRPO + self-distillation)
        - GRPO/DrGRPO/DAPO с Liger: LigerFusedGRPOLoss
        - GRPO/DrGRPO/DAPO без Liger: GRPOLoss
        
        ВАЖНО: Вызывается ПОСЛЕ accelerator.prepare() потому что для Liger Fused Loss
        нужна unwrapped модель для доступа к lm_head.weight.
        """
        # Получаем unwrapped модель для Liger (нужен доступ к lm_head.weight)
        if self.accelerator:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model
        
        # Проверяем тип алгоритма
        is_sdpo = (
            getattr(self.config, 'algorithm', None) == RLAlgorithm.SDPO or
            getattr(self.config, 'use_self_distillation', False)
        )
        
        if is_sdpo:
            # SDPO: используем SDPOLoss
            self.loss_fn = SDPOLoss(config=self.config)
            logger.info("🎓 SDPOLoss активирован!")
            logger.info("   - GRPO loss + Self-Distillation")
            logger.info(f"   - success_threshold: {getattr(self.config, 'sdpo_success_threshold', 0.5)}")
            logger.info(f"   - alpha (KL type): {getattr(self.config, 'sdpo_alpha', 0.5)}")
            
            # ============================================================
            # 🔥 ОПТИМИЗАЦИЯ: Teacher Module Setup (из verl)
            # ============================================================
            self.sdpo_ema_rate = getattr(self.config, 'sdpo_ema_rate', 0.0)
            
            # Определяем стратегию teacher:
            # 1. LoRA + EMA > 0: Teacher = Student с EMA LoRA весами (через context manager)
            # 2. LoRA без EMA: Teacher = Student (detached)  
            # 3. Full FT + reference: Teacher = Reference Model
            # 4. Full FT без reference: Teacher = Student (detached)
            
            use_lora = getattr(self.config, 'use_lora', False)
            
            if use_lora:
                # При LoRA: НЕ используем reference_model как teacher!
                # Reference model не имеет LoRA адаптеров — структуры несовместимы
                self.teacher_module = None  # Будет использоваться student через context manager
                
                if self.sdpo_ema_rate > 0:
                    logger.info("   🔥 Teacher = Student + EMA LoRA (через context manager)")
                else:
                    logger.info("   ℹ️ Teacher = Student (detached, EMA отключен)")
            else:
                # Full Fine-tuning: можем использовать reference_model
                self.teacher_module = self.reference_model  # 🔥 Шаринг памяти!
                
                if self.teacher_module is not None:
                    logger.info("   🔥 Teacher = Reference Model (шаринг памяти!)")
                else:
                    logger.info("   ⚠️ Нет reference модели, teacher = student (detached)")
            
            # Top-K Distillation параметры
            self.sdpo_distillation_topk = getattr(self.config, 'sdpo_distillation_topk', None)
            self.sdpo_full_logit_distillation = getattr(self.config, 'sdpo_full_logit_distillation', False)
            
            if self.sdpo_distillation_topk is not None:
                logger.info(f"   🔥 Top-K Distillation: k={self.sdpo_distillation_topk}")
                logger.info(f"      Экономия памяти: ~99.97% vs full vocab!")
            
            if self.sdpo_ema_rate > 0:
                logger.info(f"   📈 EMA Teacher: rate={self.sdpo_ema_rate}")
                
                # 🔥 Инициализируем EMA для LoRA (если используется)
                if getattr(self.config, 'use_lora', False):
                    self._init_ema_for_lora()
                    logger.info("   🔥 EMA LoRA режим активирован!")
            
            # Инициализируем хранилище успешных траекторий
            self._successful_trajectories: Dict[int, List[str]] = {}
            return
        
        # GRPO/DrGRPO/DAPO с Liger Fused Loss
        if self.use_liger_fused_loss:
            try:
                from .liger_utils import is_liger_available, get_liger_fused_linear_grpo
                
                if is_liger_available() and get_liger_fused_linear_grpo() is not None:
                    self.loss_fn = LigerFusedGRPOLoss(
                        model=unwrapped_model,
                        config=self.config,
                    )
                    logger.info("🦁 LigerFusedGRPOLoss активирован!")
                    logger.info("   ⚡ Logits НЕ материализуются — экономия памяти!")
                    return
                else:
                    logger.warning("⚠️ Liger недоступен, используем стандартный GRPOLoss")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось создать LigerFusedGRPOLoss: {e}")
                logger.warning("   Используем стандартный GRPOLoss")
        
        # Fallback на стандартный GRPOLoss
        self.loss_fn = GRPOLoss(config=self.config)
        logger.info("📊 Используется стандартный GRPOLoss")

    # =========================================================================
    # 🎓 SDPO: Teacher Model и EMA Update (из verl)
    # =========================================================================
    
    def _init_ema_for_lora(self) -> None:
        """
        Инициализирует EMA state dict для LoRA весов.
        
        При использовании LoRA мы храним EMA копию только LoRA адаптеров (~50-200 MB),
        а не всей модели (~ГБ). Это позволяет использовать EMA Teacher с минимальным
        overhead по памяти.
        """
        if not getattr(self.config, 'use_lora', False):
            return
        
        if not hasattr(self, 'sdpo_ema_rate') or self.sdpo_ema_rate <= 0:
            return
        
        # Unwrap модели
        if self.accelerator:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        
        # Проверяем что это PEFT модель
        try:
            from peft import PeftModel
            if not isinstance(model, PeftModel):
                logger.warning("⚠️ use_lora=True, но модель не PeftModel. EMA LoRA пропущен.")
                return
        except ImportError:
            logger.warning("⚠️ peft не установлен. EMA LoRA пропущен.")
            return
        
        # Инициализируем EMA state dict для LoRA весов
        self._ema_lora_state_dict: Dict[str, torch.Tensor] = {}
        self._original_lora_state_dict: Dict[str, torch.Tensor] = {}  # Для восстановления
        
        lora_param_count = 0
        lora_memory_bytes = 0
        
        for name, param in model.named_parameters():
            # LoRA параметры имеют 'lora_' в имени и требуют градиентов
            if 'lora_' in name.lower() and param.requires_grad:
                # Клонируем текущие веса как начальное значение EMA
                self._ema_lora_state_dict[name] = param.data.clone().detach()
                lora_param_count += param.numel()
                lora_memory_bytes += param.numel() * param.element_size()
        
        if lora_param_count > 0:
            lora_memory_mb = lora_memory_bytes / (1024 ** 2)
            logger.info(f"✅ EMA LoRA инициализирован:")
            logger.info(f"   - LoRA параметров: {lora_param_count:,}")
            logger.info(f"   - EMA память: ~{lora_memory_mb:.1f} MB")
            logger.info(f"   - EMA rate: {self.sdpo_ema_rate}")
        else:
            logger.warning("⚠️ Не найдено LoRA параметров для EMA!")
            self._ema_lora_state_dict = {}
    
    def _update_teacher_ema(self) -> None:
        """
        🔥 ОПТИМИЗАЦИЯ: EMA Update для Teacher модели (из verl).
        
        Teacher = EMA(Student) — медленно обновляемая копия student модели.
        Это даёт более стабильный target для distillation.
        
        Формула: teacher = (1 - ema_rate) * teacher + ema_rate * student
        
        Для LoRA: обновляем только LoRA адаптеры (экономия памяти!)
        Для Full Fine-tuning: обновляем все параметры teacher модели
        """
        if not hasattr(self, 'sdpo_ema_rate') or self.sdpo_ema_rate <= 0:
            return
        
        ema_rate = self.sdpo_ema_rate
        
        # Unwrap модели
        if self.accelerator:
            student_model = self.accelerator.unwrap_model(self.model)
        else:
            student_model = self.model
        
        # ============================================================
        # РЕЖИМ 1: LoRA — обновляем только EMA LoRA весов
        # ============================================================
        if getattr(self.config, 'use_lora', False) and hasattr(self, '_ema_lora_state_dict'):
            if not self._ema_lora_state_dict:
                return  # EMA не инициализирован
            
            with torch.no_grad():
                for name, param in student_model.named_parameters():
                    if name in self._ema_lora_state_dict:
                        # EMA update: ema = (1 - rate) * ema + rate * current
                        ema_tensor = self._ema_lora_state_dict[name]
                        student_data = param.data.to(device=ema_tensor.device, dtype=ema_tensor.dtype)
                        ema_tensor.mul_(1.0 - ema_rate).add_(student_data, alpha=ema_rate)
            
            logger.debug(f"🎓 SDPO EMA LoRA обновлён (rate={ema_rate}, params={len(self._ema_lora_state_dict)})")
            return
        
        # ============================================================
        # РЕЖИМ 2: Full Fine-tuning — обновляем teacher_module
        # ============================================================
        if not hasattr(self, 'teacher_module') or self.teacher_module is None:
            return
        
        # Проверяем что teacher != student (иначе EMA бессмысленен)
        if self.teacher_module is self.model:
            return
        
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_module.parameters(),
                student_model.parameters()
            ):
                # EMA update: teacher = (1 - ema) * teacher + ema * student
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - ema_rate).add_(student_data, alpha=ema_rate)
        
        logger.debug(f"🎓 SDPO EMA Teacher обновлён (rate={ema_rate})")
    
    def _apply_ema_lora_weights(self) -> None:
        """
        Применяет EMA LoRA веса к модели (сохраняя оригинальные для восстановления).
        
        Используется перед forward pass teacher для SDPO.
        После forward pass нужно вызвать _restore_lora_weights().
        """
        if not hasattr(self, '_ema_lora_state_dict') or not self._ema_lora_state_dict:
            return
        
        # Unwrap модели
        if self.accelerator:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        
        self._original_lora_state_dict = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._ema_lora_state_dict:
                    # Сохраняем оригинальные веса
                    self._original_lora_state_dict[name] = param.data.clone()
                    # Применяем EMA веса
                    param.data.copy_(self._ema_lora_state_dict[name])
    
    def _restore_lora_weights(self) -> None:
        """
        Восстанавливает оригинальные LoRA веса после forward pass teacher.
        """
        if not hasattr(self, '_original_lora_state_dict') or not self._original_lora_state_dict:
            return
        
        # Unwrap модели
        if self.accelerator:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._original_lora_state_dict:
                    param.data.copy_(self._original_lora_state_dict[name])
        
        # Очищаем временное хранилище
        self._original_lora_state_dict = {}
    
    @contextmanager
    def _with_ema_lora_weights(self):
        """
        Context manager для временного применения EMA LoRA весов.
        
        Использование:
            with self._with_ema_lora_weights():
                output = model(input_ids)  # Использует EMA веса
            # Здесь веса восстановлены
        """
        use_ema_lora = (
            getattr(self.config, 'use_lora', False) and
            hasattr(self, '_ema_lora_state_dict') and
            bool(self._ema_lora_state_dict) and
            getattr(self, 'sdpo_ema_rate', 0) > 0
        )
        
        if use_ema_lora:
            self._apply_ema_lora_weights()
        
        try:
            yield
        finally:
            if use_ema_lora:
                self._restore_lora_weights()
    
    # =========================================================================
    # 🎓 SDPO: Reprompting методы
    # =========================================================================
    
    def _create_reprompted_input(
        self,
        original_prompt: str,
        successful_solution: str,
        feedback: Optional[str] = None,
    ) -> str:
        """
        Создаёт reprompted контекст для teacher (SDPO).
        
        Формат:
            Here is the problem:
            {original_question}
            
            Here is a successful solution for reference:
            {successful_solution}
            
            Now solve this problem step by step.
        
        Args:
            original_prompt: Исходный промпт с вопросом
            successful_solution: Успешное решение (completion)
            feedback: Опциональный feedback (ошибки и т.д.)
            
        Returns:
            Reprompted строка для teacher
        """
        # Получаем шаблоны из конфига
        reprompt_template = getattr(
            self.config, 
            'sdpo_reprompt_template',
            """Here is the problem:
{question}

Here is a successful solution for reference:
{successful_solution}

Now solve this problem step by step."""
        )
        
        feedback_template = getattr(
            self.config,
            'sdpo_feedback_template',
            """
Previous attempt feedback:
{feedback}
"""
        )
        
        # Извлекаем вопрос из original_prompt
        # Обычно prompt содержит system message + user question
        # Пытаемся извлечь только вопрос
        question = original_prompt
        
        # Пытаемся найти вопрос в chat template формате
        if "User:" in original_prompt:
            parts = original_prompt.split("User:")
            if len(parts) > 1:
                question = parts[-1].split("Assistant:")[0].strip()
        elif "<|user|>" in original_prompt:
            parts = original_prompt.split("<|user|>")
            if len(parts) > 1:
                question = parts[-1].split("<|assistant|>")[0].strip()
        elif "[INST]" in original_prompt:
            parts = original_prompt.split("[INST]")
            if len(parts) > 1:
                question = parts[-1].split("[/INST]")[0].strip()
        
        # Создаём reprompted текст
        reprompted = reprompt_template.format(
            question=question,
            successful_solution=successful_solution,
        )
        
        # Добавляем feedback если есть
        include_feedback = getattr(self.config, 'sdpo_include_feedback', True)
        if include_feedback and feedback:
            feedback_text = feedback_template.format(feedback=feedback)
            reprompted = reprompted + feedback_text
        
        # Обрезаем если слишком длинный
        max_len = getattr(self.config, 'sdpo_max_reprompt_len', 4096)
        if len(reprompted) > max_len:
            reprompted = reprompted[:max_len]
        
        return reprompted
    
    def _get_teacher_log_probs(
        self,
        exp_batch: "Experience",
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        🔥 ОПТИМИЗАЦИЯ: Вычисляет teacher_log_probs для SDPO (из verl).
        
        Ключевые оптимизации:
        1. Использует teacher_module (шаринг с reference_model) — экономия памяти!
        2. Top-K Distillation — вместо vocab=152k используем только k=50-100 токенов
        3. Chunked processing — по одному reprompt за раз
        
        Для каждого сэмпла в batch:
        1. Ищем успешную траекторию для его prompt_id
        2. Создаём reprompted контекст (prompt + successful_solution)
        3. Делаем forward pass через teacher_module
        4. Получаем log_probs (и top-k если нужно)
        
        Args:
            exp_batch: Batch Experience объектов
            device: Устройство для вычислений
            
        Returns:
            (teacher_log_probs, distillation_mask, student_topk, teacher_topk) или (None, None, None, None)
        """
        if not hasattr(self, '_successful_trajectories'):
            return None, None, None, None
        
        batch_size = exp_batch.sequences.size(0)
        seq_len = exp_batch.action_log_probs.size(1)
        
        # ============================================================
        # 🔥 ОПТИМИЗАЦИЯ: Выбор Teacher Model
        # ============================================================
        # Для LoRA + EMA: используем student модель с EMA весами
        # Для Full Fine-tuning: используем teacher_module (reference model)
        use_ema_lora = (
            getattr(self.config, 'use_lora', False) and
            hasattr(self, '_ema_lora_state_dict') and
            bool(self._ema_lora_state_dict) and
            getattr(self, 'sdpo_ema_rate', 0) > 0
        )
        
        if use_ema_lora:
            # LoRA + EMA: используем student с EMA весами
            if self.accelerator:
                teacher_model = self.accelerator.unwrap_model(self.model)
            else:
                teacher_model = self.model
            logger.debug("🎓 Teacher: Student + EMA LoRA веса")
        else:
            # Full Fine-tuning или LoRA без EMA: используем teacher_module
            teacher_model = getattr(self, 'teacher_module', None)
            if teacher_model is None:
                # Fallback: используем student model
                if self.accelerator:
                    teacher_model = self.accelerator.unwrap_model(self.model)
                else:
                    teacher_model = self.model
        
        # Top-K параметры
        use_topk = getattr(self, 'sdpo_full_logit_distillation', False) and \
                   getattr(self, 'sdpo_distillation_topk', None) is not None
        topk = getattr(self, 'sdpo_distillation_topk', 50) if use_topk else None
        
        # Проверяем какие сэмплы имеют успешные траектории
        has_teacher = torch.zeros(batch_size, dtype=torch.bool, device=device)
        reprompted_inputs = []
        sample_to_reprompt_idx = {}
        
        # Получаем prompt_ids из batch если есть
        prompt_ids = getattr(exp_batch, 'prompt_ids', None)
        if prompt_ids is None:
            prompt_ids = [None] * batch_size
        
        for idx in range(batch_size):
            pid = prompt_ids[idx] if prompt_ids is not None else None
            if pid is None:
                continue
                
            pid_int = int(pid) if torch.is_tensor(pid) else pid
            
            if pid_int in self._successful_trajectories and self._successful_trajectories[pid_int]:
                import random
                trajectory = random.choice(self._successful_trajectories[pid_int])
                
                reprompted = self._create_reprompted_input(
                    original_prompt=trajectory['prompt'],
                    successful_solution=trajectory['completion'],
                )
                
                has_teacher[idx] = True
                sample_to_reprompt_idx[idx] = len(reprompted_inputs)
                reprompted_inputs.append(reprompted)
        
        if not reprompted_inputs:
            return None, None, None, None
        
        # 🔥 Результаты
        teacher_log_probs = torch.zeros(batch_size, seq_len, device=device)
        
        # Top-K tensors (только если используем full_logit_distillation)
        student_topk_log_probs = None
        teacher_topk_log_probs = None
        if use_topk and topk is not None:
            student_topk_log_probs = torch.zeros(batch_size, seq_len, topk, device=device)
            teacher_topk_log_probs = torch.zeros(batch_size, seq_len, topk, device=device)
        
        # 🔥 Forward pass через Teacher (с EMA LoRA весами если нужно)
        with torch.no_grad():
            # Используем context manager для EMA LoRA весов
            with self._with_ema_lora_weights():
                for idx in range(batch_size):
                    if not has_teacher[idx]:
                        continue
                    
                    reprompt_idx = sample_to_reprompt_idx[idx]
                    reprompt_text = reprompted_inputs[reprompt_idx]
                    
                    # Токенизируем ОДИН reprompt
                    reprompt_encoding = self.tokenizer(
                        reprompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=getattr(self.config, 'sdpo_max_reprompt_len', 4096),
                    ).to(device)
                    
                    # 🔥 Forward pass через TEACHER MODEL
                    # При LoRA + EMA: student модель с EMA весами
                    # При Full FT: reference модель
                    output = teacher_model(
                        input_ids=reprompt_encoding['input_ids'],
                        attention_mask=reprompt_encoding['attention_mask'],
                        use_cache=False,
                    )
                    
                    logits = output.logits[0]  # [reprompt_seq, vocab]
                    reprompt_seq_len = logits.size(0)
                    completion_tokens = exp_batch.sequences[idx, 1:seq_len+1]
                    
                    for t in range(min(seq_len, reprompt_seq_len - 1)):
                        pos = reprompt_seq_len - seq_len - 1 + t if reprompt_seq_len > seq_len else t
                        if pos >= 0 and pos < reprompt_seq_len - 1:
                            token_id = completion_tokens[t].item()
                            if token_id < logits.size(-1):
                                log_probs_pos = F.log_softmax(logits[pos], dim=-1)
                                teacher_log_probs[idx, t] = log_probs_pos[token_id]
                                
                                # 🔥 TOP-K DISTILLATION
                                if use_topk and teacher_topk_log_probs is not None:
                                    topk_vals, topk_idxs = torch.topk(log_probs_pos, topk)
                                    teacher_topk_log_probs[idx, t] = topk_vals
                    
                    del output, logits, reprompt_encoding
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # 🔥 TOP-K для Student (нужно если используем full_logit_distillation)
        # Вычисляем top-k student log_probs для тех же позиций
        if use_topk and student_topk_log_probs is not None:
            # Unwrap student model
            if self.accelerator:
                student_model = self.accelerator.unwrap_model(self.model)
            else:
                student_model = self.model
            
            with torch.no_grad():
                # Forward pass student на оригинальных sequences
                student_output = student_model(
                    input_ids=exp_batch.sequences,
                    attention_mask=exp_batch.attention_mask,
                    use_cache=False,
                )
                student_logits = student_output.logits[:, :-1]  # [batch, seq, vocab]
                
                for idx in range(batch_size):
                    if has_teacher[idx]:
                        for t in range(seq_len):
                            log_probs_t = F.log_softmax(student_logits[idx, t], dim=-1)
                            topk_vals, _ = torch.topk(log_probs_t, topk)
                            student_topk_log_probs[idx, t] = topk_vals
                
                del student_output, student_logits
        
        distillation_mask = has_teacher.float()
        
        logger.debug(
            f"🎓 SDPO: {has_teacher.sum().item()}/{batch_size} сэмплов с teacher, "
            f"top-k={topk if use_topk else 'off'}"
        )
        
        return teacher_log_probs, distillation_mask, student_topk_log_probs, teacher_topk_log_probs

    def _setup_rollout_engine(self) -> None:
        """
        Инициализация отдельной модели для генерации (rollout engine).
        """
        if not getattr(self.config, "use_rollout_engine", False):
            return

        backend = getattr(self.config, "rollout_engine_backend", "hf")

        mp = (getattr(self.config, "mixed_precision", None) or "bf16").lower()
        if mp == "bf16":
            dtype = torch.bfloat16
        elif mp == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        offload = bool(getattr(self.config, "rollout_offload_to_cpu", False))
        rollout_device = getattr(self.config, "rollout_device", "auto")
        if rollout_device == "cpu":
            device = torch.device("cpu")
            offload = True
        else:
            device = self.device

        if backend == "hf":
            self.rollout_engine = HFRolloutEngine(
                base_model_path=self.model_name,
                device=device,
                torch_dtype=dtype,
                use_flash_attention=bool(getattr(self.config, "use_flash_attention", True)),
                trust_remote_code=True,
                offload_to_cpu=offload,
            )
            self.rollout_engine.ensure_loaded()
            self._sync_rollout_engine_weights(force=True)
        elif backend == "vllm":
            # Проверяем количество процессов
            num_processes = getattr(self.accelerator, "num_processes", 1)
            
            # vLLM + Multi-GPU DDP не поддерживается (каждый процесс генерирует свои данные)
            if num_processes > 1:
                logger.warning(
                    f"⚠️ vLLM + Multi-GPU DDP ({num_processes} процессов) не поддерживается. "
                    f"Используем HF backend с Prefix Grouper для генерации."
                )
                self.rollout_engine = None
                return
            
            # vLLM устройство из конфига
            vllm_device_str = getattr(self.config, "vllm_device", "cuda:0")
            
            # vLLM на CPU не поддерживается
            if vllm_device_str == "cpu":
                raise RuntimeError(
                    "vLLM не поддерживает CPU. Выберите GPU (cuda:X) или используйте HF backend."
                )
            
            # Извлекаем номер GPU для vLLM (физический индекс из UI)
            if vllm_device_str.startswith("cuda:"):
                vllm_physical_gpu = int(vllm_device_str.split(":")[1])
            else:
                vllm_physical_gpu = 0
            
            # ВАЖНО: Ремапим физический индекс GPU в индекс внутри процесса
            # CUDA_VISIBLE_DEVICES=0,1 означает cuda:0=physical0, cuda:1=physical1
            # CUDA_VISIBLE_DEVICES=1,0 означает cuda:0=physical1, cuda:1=physical0
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            available_gpus = torch.cuda.device_count()
            
            if cuda_visible:
                visible_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
                if vllm_physical_gpu in visible_gpus:
                    # Находим индекс внутри процесса
                    vllm_gpu_id = visible_gpus.index(vllm_physical_gpu)
                    logger.info(f"🔄 vLLM GPU: physical {vllm_physical_gpu} → process cuda:{vllm_gpu_id}")
                else:
                    logger.error(
                        f"❌ vLLM GPU (physical {vllm_physical_gpu}) не в CUDA_VISIBLE_DEVICES={cuda_visible}! "
                        f"Используем cuda:0."
                    )
                    vllm_gpu_id = 0
            else:
                # Нет ограничений, используем физический индекс напрямую
                vllm_gpu_id = vllm_physical_gpu
            
            # Проверяем что GPU существует внутри процесса
            if vllm_gpu_id >= available_gpus:
                logger.error(
                    f"❌ vLLM GPU cuda:{vllm_gpu_id} не существует! "
                    f"Доступно только {available_gpus} GPU.\n"
                    f"CUDA_VISIBLE_DEVICES={cuda_visible}\n"
                    f"Используем cuda:0 для vLLM."
                )
                vllm_gpu_id = 0
            
            # Определяем текущую GPU training модели
            training_device = self.device
            if hasattr(training_device, 'index') and training_device.index is not None:
                training_gpu_id = training_device.index
            elif str(training_device).startswith("cuda:"):
                training_gpu_id = int(str(training_device).split(":")[1])
            elif str(training_device) == "cuda":
                training_gpu_id = torch.cuda.current_device()
            else:
                training_gpu_id = 0
            
            # GPU memory utilization из конфига
            vllm_gpu_util = float(getattr(self.config, "vllm_gpu_memory_utilization", 0.85))
            
            # Проверяем, на той же GPU или на другой
            same_gpu = (vllm_gpu_id == training_gpu_id)
            
            # max_model_len: prompt + response
            max_len = int(getattr(self.config, "max_prompt_length", 512)) + int(getattr(self.config, "max_new_tokens", 1024))
            
            # Получаем lora_r для vLLM max_lora_rank
            lora_rank_for_vllm = self.config.lora_r if self.config.use_lora else 16
            
            if same_gpu:
                # На той же GPU — используем VLLMRolloutEngine напрямую
                if vllm_gpu_util > 0.5:
                    logger.warning(
                        f"⚠️ vLLM на той же GPU что training (cuda:{training_gpu_id}). "
                        f"gpu_memory_utilization={vllm_gpu_util:.0%} может быть слишком высоким! "
                        f"Рекомендуется 30-50%."
                    )
                logger.info(f"🧩 vLLM: загружаем на cuda:{vllm_gpu_id} (та же GPU что training, memory={vllm_gpu_util:.0%})")
                
                self.rollout_engine = VLLMRolloutEngine(
                    base_model_path=self.model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    max_model_len=max_len,
                    gpu_memory_utilization=vllm_gpu_util,
                    max_lora_rank=lora_rank_for_vllm,  # Для vLLM max_lora_rank!
                )
                self.rollout_engine.ensure_loaded()
            else:
                # На ОТДЕЛЬНОЙ GPU — используем VLLMSubprocessEngine!
                # Это запускает vLLM в отдельном процессе с правильным CUDA_VISIBLE_DEVICES
                logger.info(f"🧩 vLLM: запуск на cuda:{vllm_gpu_id} через SUBPROCESS (отдельная GPU, memory={vllm_gpu_util:.0%})")
                logger.info(f"   Training на cuda:{training_gpu_id}, vLLM на cuda:{vllm_physical_gpu}")
                
                # Используем ФИЗИЧЕСКИЙ индекс GPU для subprocess
                # (внутри subprocess будет CUDA_VISIBLE_DEVICES={vllm_physical_gpu})
                self.rollout_engine = VLLMSubprocessEngine(
                    base_model_path=self.model_name,
                    torch_dtype=dtype,
                    gpu_id=vllm_physical_gpu,  # Физический индекс GPU!
                    max_model_len=max_len,
                    gpu_memory_utilization=vllm_gpu_util,
                    enable_lora=True,
                    max_lora_rank=lora_rank_for_vllm,  # Для vLLM max_lora_rank!
                    output_dir=getattr(self.config, "output_dir", None),
                )
                self.rollout_engine.ensure_loaded()
            
            # Сохраняем GPU IDs
            self._vllm_gpu_id = vllm_gpu_id
            self._training_gpu_id = training_gpu_id
            self._use_vllm_subprocess = not same_gpu
            
            # Первичный sync LoRA адаптера
            self._sync_rollout_engine_weights(force=True)
        else:
            raise NotImplementedError(f"Unknown rollout_engine_backend='{backend}'")

    def _next_group_uids(self, n: int) -> List[int]:
        """Возвращает n уникальных group_id для группировки Experience."""
        start = self._group_uid
        self._group_uid += int(n)
        return list(range(start, start + int(n)))
    
    def setup(self):
        """Инициализирует все компоненты для обучения."""
        logger.info("Инициализация GRPOTrainer...")
        
        # Логируем информацию о GPU ДО создания accelerator
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"🖥️  Обнаружено GPU: {num_gpus} устройств")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.info(f"  - GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.info("🖥️  GPU не обнаружены, используется CPU")
        
        # Accelerate - создаем ПЕРЕД загрузкой модели (как в pretrain/SFT)
        if self.use_accelerate:
            try:
                from accelerate import Accelerator
                
                # Mixed precision должен соответствовать выбору пользователя в UI.
                mixed_precision = (getattr(self.config, "mixed_precision", None) or "bf16").lower()
                if mixed_precision not in ("no", "fp16", "bf16"):
                    logger.warning(f"Неизвестный mixed_precision='{mixed_precision}', fallback -> bf16")
                    mixed_precision = "bf16"
                if mixed_precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                    logger.warning("bf16 выбран в UI, но GPU не поддерживает bf16. Fallback -> fp16")
                    mixed_precision = "fp16"

                # "Pure fp16" (веса fp16, без GradScaler): для accelerate нужно mixed_precision='no',
                # иначе он включит GradScaler и упадёт при fp16 градиентах.
                accel_mp = mixed_precision
                if mixed_precision == "fp16" and bool(getattr(self.config, "fp16_pure", False)):
                    accel_mp = "no"
                    logger.info("🧪 FP16 Pure режим: Accelerator(mixed_precision='no'), веса модели будут torch.float16")
                
                logger.info(f"🚀 Инициализация Accelerator...")
                logger.info(f"  - gradient_accumulation_steps: {self.config.gradient_accumulation_steps}")
                logger.info(f"  - mixed_precision (UI): {mixed_precision}")
                logger.info(f"  - mixed_precision (accelerate): {accel_mp}")
                
                # Проверяем нужен ли find_unused_parameters для DDP
                # lm_head/embed_tokens связаны через tie_word_embeddings и могут не получать градиенты
                target_modules = getattr(self.config, "lora_target_modules", None) or []
                needs_find_unused = any(m in target_modules for m in ["lm_head", "embed_tokens"])
                
                ddp_kwargs = None
                if needs_find_unused:
                    from accelerate import DistributedDataParallelKwargs
                    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
                    logger.info(f"  - find_unused_parameters: True (lm_head/embed_tokens в target_modules)")
                
                self.accelerator = Accelerator(
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    mixed_precision=accel_mp,
                    kwargs_handlers=[ddp_kwargs] if ddp_kwargs else None,
                )
                
                # Устройство берем из accelerator (поддерживает multi-GPU)
                self.device = self.accelerator.device
                self.is_main_process = self.accelerator.is_main_process
                
                # Логируем информацию о распределении
                if self.accelerator.num_processes > 1:
                    logger.info(f"✅ Multi-GPU режим: {self.accelerator.num_processes} процессов")
                    logger.info(f"  - Текущий процесс: {self.accelerator.process_index} / {self.accelerator.num_processes - 1}")
                    logger.info(f"  - Main process: {self.is_main_process}")
                else:
                    logger.info(f"✅ Single GPU режим")
                
                logger.info(f"📱 Устройство: {self.device}")
                
                # Логируем и настраиваем DeepSpeed конфигурацию
                self._log_and_setup_deepspeed_config()
                
            except ImportError:
                logger.warning("⚠️  accelerate не установлен, используем single GPU")
                self.accelerator = None
                self.device = self._device if self._device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.is_main_process = True
        else:
            logger.info("ℹ️  Accelerate отключен (use_accelerate=False)")
            self.accelerator = None
            self.device = self._device if self._device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.is_main_process = True
            logger.info(f"📱 Устройство: {self.device}")
        
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
        
        # Loss функция — создаётся ПОСЛЕ загрузки модели (нужна для Liger Fused Loss)
        # Будет создана в setup() после accelerator.prepare()
        self.loss_fn = None
        self.use_liger_fused_loss = getattr(self.config, 'liger_fused_grpo', False) and getattr(self.config, 'use_liger', False)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # W&B
        if self.config.use_wandb and self.is_main_process:
            self._setup_wandb()
        
        logger.info(f"GRPOTrainer инициализирован на {self.device}")
        logger.info(f"Алгоритм: {self.config.algorithm.value}")
        if self.config.dynamic_sampling:
            logger.info(f"  🎯 Dynamic sampling: ON (max_refill_rounds={self.config.max_refill_rounds})")
            # Предупреждение для multi-GPU
            if self.accelerator.num_processes > 1:
                logger.warning(
                    f"  ⚠️ Dynamic sampling + Multi-GPU ({self.accelerator.num_processes} процессов) "
                    f"может вызывать рассинхронизацию! Добавлены барьеры синхронизации."
                )
        else:
            logger.info(f"  🎯 Dynamic sampling: OFF (быстрее)")
        if self.config.token_level_loss:
            logger.info(f"  📊 Token-level loss: ON")
        else:
            logger.info(f"  📊 Sample-level loss: ON")
    
    def _log_and_setup_deepspeed_config(self):
        """Логирует и настраивает DeepSpeed конфигурацию."""
        if self.accelerator is None:
            return
        
        ds_plugin = getattr(self.accelerator.state, 'deepspeed_plugin', None)
        if ds_plugin is None:
            logger.info("📋 DeepSpeed: не используется (DDP/FSDP режим)")
            return
        
        logger.info("=" * 60)
        logger.info("📋 КОНФИГУРАЦИЯ DEEPSPEED:")
        logger.info("=" * 60)
        
        # Основные параметры
        zero_stage = getattr(ds_plugin, 'zero_stage', 'N/A')
        logger.info(f"  - ZeRO Stage: {zero_stage}")
        
        # Offload настройки
        offload_optimizer = getattr(ds_plugin, 'offload_optimizer_device', None)
        offload_param = getattr(ds_plugin, 'offload_param_device', None)
        logger.info(f"  - Offload Optimizer: {offload_optimizer or 'none'}")
        logger.info(f"  - Offload Param: {offload_param or 'none'}")
        
        # Полный конфиг
        ds_config = getattr(ds_plugin, 'deepspeed_config', {})
        if ds_config:
            logger.info("  - Полный DeepSpeed конфиг:")
            for key, value in ds_config.items():
                if isinstance(value, dict):
                    logger.info(f"    {key}:")
                    for k, v in value.items():
                        logger.info(f"      {k}: {v}")
                else:
                    logger.info(f"    {key}: {value}")
        
        # ВАЖНО: Устанавливаем train_micro_batch_size_per_gpu для DeepSpeed
        # DeepSpeed требует это значение при accelerator.prepare() без dataloader
        # Для GRPO используем train_batch_size из конфига
        micro_batch_size = getattr(self.config, 'train_batch_size', None)
        if micro_batch_size is None:
            micro_batch_size = getattr(self.config, 'batch_size', 1) or 1
        micro_batch_size = max(1, int(micro_batch_size))
        
        logger.info(f"  - Устанавливаем train_micro_batch_size_per_gpu: {micro_batch_size}")
        logger.info(f"  - gradient_accumulation_steps: {self.config.gradient_accumulation_steps}")
        
        # Устанавливаем значения в DeepSpeed конфиг
        try:
            from accelerate.state import AcceleratorState
            state = AcceleratorState()
            if hasattr(state, 'deepspeed_plugin') and state.deepspeed_plugin is not None:
                ds_cfg = state.deepspeed_plugin.deepspeed_config
                if ds_cfg is not None:
                    # Устанавливаем batch sizes
                    ds_cfg['train_micro_batch_size_per_gpu'] = micro_batch_size
                    ds_cfg['gradient_accumulation_steps'] = self.config.gradient_accumulation_steps
                    # train_batch_size = micro_batch_size * gradient_accumulation * num_gpus
                    num_gpus = self.accelerator.num_processes
                    ds_cfg['train_batch_size'] = micro_batch_size * self.config.gradient_accumulation_steps * num_gpus
                    logger.info(f"  ✅ DeepSpeed batch sizes установлены:")
                    logger.info(f"    - train_micro_batch_size_per_gpu: {ds_cfg['train_micro_batch_size_per_gpu']}")
                    logger.info(f"    - gradient_accumulation_steps: {ds_cfg['gradient_accumulation_steps']}")
                    logger.info(f"    - train_batch_size: {ds_cfg['train_batch_size']}")
        except Exception as e:
            logger.warning(f"⚠️  Не удалось установить DeepSpeed batch sizes: {e}")
        
        logger.info("=" * 60)
    
    def _load_model(self):
        """Загружает модель с опциональной квантизацией и LoRA."""
        logger.info(f"Загрузка модели {self.model_name}...")
        
        # Логируем конфигурацию
        logger.info(f"📋 Конфигурация:")
        logger.info(f"  - use_4bit: {self.config.use_4bit}")
        logger.info(f"  - use_8bit: {self.config.use_8bit}")
        logger.info(f"  - use_lora: {self.config.use_lora}")
        if self.config.use_lora:
            logger.info(f"  - lora_r: {self.config.lora_r}")
            logger.info(f"  - lora_alpha: {self.config.lora_alpha}")
            logger.info(f"  - lora_target_modules: {self.config.lora_target_modules}")
        
        # Определяем используется ли DeepSpeed ZeRO-3 в начале функции
        # При ZeRO-3 параметры sharded между процессами и нельзя делать .to(device)
        # Сохраняем как атрибут класса для использования в других методах
        self.is_deepspeed_zero3 = False
        if self.accelerator is not None:
            ds_plugin = getattr(self.accelerator.state, 'deepspeed_plugin', None)
            if ds_plugin is not None:
                zero_stage = getattr(ds_plugin, 'zero_stage', 0)
                self.is_deepspeed_zero3 = zero_stage == 3
                logger.info(f"🔧 DeepSpeed ZeRO stage: {zero_stage}")
                if self.is_deepspeed_zero3:
                    logger.info("⚡ ZeRO-3 режим: параметры будут sharded, пропускаем .to(device)")
        is_deepspeed_zero3 = self.is_deepspeed_zero3  # локальная переменная для совместимости
        
        # Проверяем использование памяти до загрузки
        memory_before = 0.0
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"💾 Память CUDA до загрузки модели: {memory_before:.1f} MB")
        
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
                    logger.info("✅ Создан BitsAndBytesConfig для 4-bit квантизации")
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    logger.info("✅ Создан BitsAndBytesConfig для 8-bit квантизации")
            except ImportError:
                logger.warning("❌ bitsandbytes не установлен, квантизация отключена")
                quantization_config = None
        else:
            logger.info("ℹ️  Квантизация отключена (use_4bit=False, use_8bit=False)")
        
        # Загрузка модели
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if quantization_config else None,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            # ВАЖНО:
            # - bf16: можно грузить веса в bf16 (нет GradScaler).
            # - fp16: по умолчанию это AMP fp16 (fp32 master-веса + GradScaler) => веса оставляем fp32.
            #   Для экономии VRAM можно включить "pure fp16" (веса fp16, без GradScaler) через config.fp16_pure.
            mp = (getattr(self.config, "mixed_precision", None) or "bf16").lower()
            if mp == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_kwargs["dtype"] = torch.bfloat16
            elif mp == "fp16":
                if bool(getattr(self.config, "fp16_pure", False)):
                    model_kwargs["dtype"] = torch.float16
                else:
                    # AMP fp16: оставляем fp32 веса (GradScaler требует fp32 master weights)
                    pass
            elif mp == "no":
                # Оставляем fp32 (дефолт HF)
                pass
            else:
                pass
        
        # Проверяем наличие flash_attn перед использованием
        # ВАЖНО: Flash Attention может конфликтовать с квантизацией в некоторых случаях
        # Для квантизированных моделей лучше использовать стандартный attention
        if self.config.use_flash_attention and not quantization_config:
            try:
                import flash_attn
                mp = (getattr(self.config, "mixed_precision", None) or "bf16").lower()
                if mp == "no":
                    logger.info("Flash Attention 2 отключен: mixed_precision='no' (fp32 не поддерживается flash-attn)")
                else:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Используется Flash Attention 2")
            except ImportError:
                logger.warning(
                    "Flash Attention 2 запрошен, но пакет flash_attn не установлен. "
                    "Используется стандартная реализация attention. "
                    "Для установки: pip install flash-attn"
                )
                # Не устанавливаем attn_implementation, используется дефолтная
        elif self.config.use_flash_attention and quantization_config:
            logger.info(
                "Flash Attention отключен для квантизированной модели "
                "(может конфликтовать с bitsandbytes). Используется стандартный attention."
            )
        
        # Логируем параметры загрузки
        logger.info(f"📦 Параметры загрузки модели:")
        logger.info(f"  - quantization_config: {'✅ Применяется' if quantization_config else '❌ Не применяется'}")
        logger.info(f"  - device_map: {model_kwargs.get('device_map', 'None')}")
        if quantization_config:
            logger.info(f"  - Тип квантизации: {'4-bit' if self.config.use_4bit else '8-bit'}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Диагностика dtype модели (помогает понять, почему fp16 может потреблять больше памяти чем bf16)
        try:
            first_param = next(self.model.parameters(), None)
            if first_param is not None:
                logger.info(f"🔎 DType весов модели (пример): {first_param.dtype}")
        except Exception:
            pass

        # Gradient checkpointing (управляется из UI)
        if getattr(self.config, "grad_checkpoint", False) and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()
                logger.info("✅ Gradient checkpointing включен (из UI)")
            except Exception as e:
                logger.warning(f"Не удалось включить gradient checkpointing: {e}")
        
        # Liger Kernel патчинг модели (оптимизированные Triton kernels)
        if getattr(self.config, "use_liger", True) and getattr(self.config, "liger_patch_model", True):
            try:
                from homellm.training.rl.liger_utils import apply_liger_patch_to_model, is_liger_available
                if is_liger_available():
                    # Патчим RMSNorm, RoPE, MLP — но НЕ CrossEntropy
                    # Для GRPO мы используем свой chunked cross-entropy в rollout.py
                    patched = apply_liger_patch_to_model(
                        self.model,
                        patch_rms_norm=True,
                        patch_rope=True,
                        patch_mlp=True,
                        patch_fused_linear_ce=False,  # Используем свой loss
                    )
                    if patched:
                        logger.info("✅ Liger Kernel патчи применены (RMSNorm, RoPE, MLP)")
                    else:
                        logger.info("ℹ️ Liger: модель не поддерживается для патчинга, используем стандартные kernels")
                else:
                    logger.info("ℹ️ Liger Kernel не установлен, используем стандартные kernels")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось применить Liger патчи: {e}")
        
        # Проверяем использование памяти после загрузки модели
        if torch.cuda.is_available():
            memory_after_load = torch.cuda.memory_allocated() / (1024 ** 2)
            logger.info(f"💾 Память CUDA после загрузки модели: {memory_after_load:.1f} MB (+{memory_after_load - memory_before:.1f} MB)")
        
        # Проверяем что модель действительно квантизирована
        if quantization_config:
            is_quantized = False
            try:
                # Проверяем наличие квантизированных параметров
                for name, param in self.model.named_parameters():
                    if hasattr(param, 'quant_state') or str(param.dtype) == 'torch.uint8':
                        is_quantized = True
                        break
                    # Для bitsandbytes квантизированные параметры могут иметь специальные атрибуты
                    if hasattr(param, 'data') and hasattr(param.data, 'quant_state'):
                        is_quantized = True
                        break
                
                if is_quantized:
                    logger.info("✅ Модель успешно квантизирована (найдены квантизированные параметры)")
                else:
                    logger.warning("⚠️  Модель может быть не квантизирована! Проверьте BitsAndBytesConfig.")
            except Exception as e:
                logger.debug(f"Не удалось проверить квантизацию: {e}")
        
        # LoRA
        # ВАЖНО: Если use_lora=True, все параметры должны быть явно указаны (без fallback)
        if self.config.use_lora:
            if self.config.lora_r is None:
                raise ValueError(
                    "❌ use_lora=True но lora_r=None! "
                    "lora_r должен быть явно указан в конфигурации. "
                    "Проверьте что render_grpo_sidebar_config() возвращает lora_r."
                )
            if self.config.lora_alpha is None:
                raise ValueError(
                    "❌ use_lora=True но lora_alpha=None! "
                    "lora_alpha должен быть явно указан в конфигурации. "
                    "Проверьте что render_grpo_sidebar_config() возвращает lora_alpha."
                )
            self._apply_lora()
        else:
            # Если LoRA не используется, включаем градиенты для всех параметров
            # (для full fine-tuning)
            # ВАЖНО: При квантизации без LoRA параметры заморожены!
            if quantization_config:
                raise RuntimeError(
                    "❌ Квантизация (4bit/8bit) без LoRA не поддерживается! "
                    "При квантизации все параметры заморожены. "
                    "Включите use_lora=True в конфигурации."
                )
            
            logger.info("LoRA отключен, включаем градиенты для всех параметров (full fine-tuning)...")
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Референсная модель (для KL)
        # ВАЖНО: Reference модель используется только для forward pass (без градиентов)
        # Квантизация не обязательна, но может экономить память
        # По умолчанию НЕ квантизируем для более точного KL divergence
        if self.config.kl_weight > 0:
            logger.info("Загрузка референсной модели для KL...")
            
            # Создаём отдельные model_kwargs для reference модели
            ref_model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if (self.config.quantize_reference_model and quantization_config) else None,
            }
            
            # Квантизация reference модели опциональна
            if self.config.quantize_reference_model and quantization_config:
                ref_model_kwargs["quantization_config"] = quantization_config
                logger.info("⚠️ Референсная модель квантизирована (экономия памяти, но может быть менее точный KL)")
            else:
                # Не квантизируем reference модель для точности KL
                # Используем тот же dtype что и основная модель (или bfloat16 по умолчанию)
                if not quantization_config:
                    # Reference модель без градиентов: можно грузить в mp dtype для экономии памяти.
                    mp = (getattr(self.config, "mixed_precision", None) or "bf16").lower()
                    if mp == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        ref_model_kwargs["dtype"] = torch.bfloat16
                    elif mp == "fp16" and torch.cuda.is_available():
                        ref_model_kwargs["dtype"] = torch.float16
                    else:
                        # fp32
                        pass
                logger.info("✅ Референсная модель НЕ квантизирована (точный KL divergence)")
            
            # Flash Attention для reference модели (если не квантизирована)
            if self.config.use_flash_attention and not (self.config.quantize_reference_model and quantization_config):
                try:
                    import flash_attn
                    ref_model_kwargs["attn_implementation"] = "flash_attention_2"
                except ImportError:
                    pass
            
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **ref_model_kwargs,
            )
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
            
            # Перемещаем reference модель на устройство если не device_map и не ZeRO-3
            if not (self.config.quantize_reference_model and quantization_config) and not is_deepspeed_zero3:
                self.reference_model = self.reference_model.to(self.device)
        else:
            logger.info("KL weight = 0, референсная модель не загружается (экономия памяти)")
        
        # Перемещаем основную модель на устройство (если не device_map и не ZeRO-3)
        # При ZeRO-3 DeepSpeed сам управляет размещением параметров
        # Reference модель уже перемещена выше
        if not quantization_config and not is_deepspeed_zero3:
            self.model = self.model.to(self.device)
        
        # ВАЖНО: После перемещения на устройство или применения LoRA,
        # убеждаемся что trainable параметры всё ещё требуют градиентов
        # Для ZeRO-3 пропускаем эту проверку - DeepSpeed сам управляет градиентами
        if not is_deepspeed_zero3:
            if self.config.use_lora:
                # Для LoRA проверяем что LoRA параметры требуют градиентов
                # PEFT должен это делать автоматически, но проверим
                try:
                    from peft import PeftModel
                    if isinstance(self.model, PeftModel):
                        # PEFT модель - проверяем что есть trainable параметры
                        pass  # PEFT должен автоматически настроить requires_grad
                except:
                    pass
            else:
                # Для full fine-tuning убеждаемся что все параметры требуют градиентов
                for param in self.model.parameters():
                    if not param.requires_grad:
                        logger.warning(f"Параметр не требует градиентов, включаем: {param.shape}")
                        param.requires_grad = True
        else:
            logger.info("⚡ ZeRO-3: пропускаем ручную настройку requires_grad")
        
        # Подсчёт параметров и проверка
        # При ZeRO-3 параметры sharded, нужен специальный подход
        if is_deepspeed_zero3:
            # Для ZeRO-3: используем num_parameters() если доступен, или конфиг модели
            try:
                # DeepSpeed модели могут иметь метод для полного подсчёта
                if hasattr(self.model, 'num_parameters'):
                    total_params = self.model.num_parameters()
                    trainable_params = self.model.num_parameters(only_trainable=True)
                else:
                    # Альтернатива: подсчёт из конфигурации модели (HuggingFace)
                    from transformers import AutoConfig
                    model_config = AutoConfig.from_pretrained(self.model_name)
                    # Примерная оценка для трансформеров: 
                    # vocab_size * hidden_size + num_layers * (4 * hidden_size^2 + ...)
                    hidden = getattr(model_config, 'hidden_size', 768)
                    layers = getattr(model_config, 'num_hidden_layers', 12)
                    vocab = getattr(model_config, 'vocab_size', 32000)
                    # Грубая оценка: embedding + transformer layers
                    total_params = vocab * hidden + layers * 12 * hidden * hidden
                    trainable_params = total_params  # full fine-tuning = все trainable
                    logger.info(f"⚡ ZeRO-3: приблизительная оценка параметров из конфига")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось оценить параметры для ZeRO-3: {e}")
                # Fallback: используем sharded размеры * world_size
                world_size = self.accelerator.num_processes if self.accelerator else 1
                total_params = sum(p.numel() for p in self.model.parameters()) * world_size
                trainable_params = total_params  # при full fine-tuning все trainable
                logger.info(f"⚡ ZeRO-3: оценка параметров = sharded * world_size ({world_size})")
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Оценка использования памяти
        if torch.cuda.is_available():
            try:
                # Примерная оценка памяти модели
                # Для квантизированных моделей: ~0.5 bytes/param (4-bit)
                # Для fp16: 2 bytes/param, для fp32: 4 bytes/param
                if quantization_config:
                    if self.config.use_4bit:
                        bytes_per_param = 0.5  # 4-bit = 0.5 bytes
                        quant_type = "4-bit"
                    else:
                        bytes_per_param = 1.0  # 8-bit = 1 byte
                        quant_type = "8-bit"
                    model_memory_mb = (total_params * bytes_per_param) / (1024 ** 2)
                else:
                    # Предполагаем bfloat16/fp16
                    try:
                        first_param = next(self.model.parameters(), None)
                        dt = getattr(first_param, "dtype", None)
                        if dt == torch.float32:
                            bytes_per_param = 4.0
                            quant_type = "fp32"
                        elif dt == torch.bfloat16:
                            bytes_per_param = 2.0
                            quant_type = "bf16"
                        elif dt == torch.float16:
                            bytes_per_param = 2.0
                            quant_type = "fp16"
                        else:
                            bytes_per_param = 2.0
                            quant_type = "fp16/bf16"
                    except Exception:
                        bytes_per_param = 2.0
                        quant_type = "fp16/bf16"
                    model_memory_mb = (total_params * bytes_per_param) / (1024 ** 2)
                
                logger.info(
                    f"Параметры модели: {total_params:,} всего, {trainable_params:,} обучаемых "
                    f"({100*trainable_params/total_params:.2f}%)"
                )
                logger.info(
                    f"💾 Примерное использование памяти модели: ~{model_memory_mb:.1f} MB ({quant_type})"
                )
                
                # Для LoRA добавляем оценку памяти адаптеров
                if self.config.use_lora:
                    # LoRA адаптеры: r * (input_dim + output_dim) * 2 (A и B матрицы) * 2 bytes (fp16)
                    # Примерная оценка: r * 2 * avg_dim * 2 bytes
                    # Для r=16, avg_dim=1024: ~16 * 2 * 1024 * 2 = 64KB на модуль
                    # Но это очень грубая оценка, реально зависит от архитектуры
                    lora_memory_mb = (trainable_params * 2.0) / (1024 ** 2)  # fp16 для адаптеров
                    logger.info(f"💾 Память LoRA адаптеров: ~{lora_memory_mb:.1f} MB")
                    
            except Exception as e:
                logger.debug(f"Не удалось оценить память: {e}")
        else:
            logger.info(f"Параметры модели: {total_params:,} всего, {trainable_params:,} обучаемых")
        
        # КРИТИЧЕСКАЯ ПРОВЕРКА: должны быть trainable параметры
        # Для ZeRO-3 пропускаем эту проверку - параметры sharded и требуют специальной обработки
        if trainable_params == 0 and not is_deepspeed_zero3:
            raise RuntimeError(
                "❌ Нет trainable параметров в модели! "
                "Проверьте конфигурацию: use_lora, use_4bit, use_8bit. "
                "Для full fine-tuning нужен use_lora=False без квантизации."
            )
        
        # Дополнительная проверка: тестовый forward pass должен требовать градиентов
        # ВАЖНО: при flash_attention_2 и mixed_precision fp16/bf16 делаем forward под autocast,
        # иначе FlashAttention может ругаться на fp32 dtype.
        # Для ZeRO-3 пропускаем тестовый forward - параметры ещё не материализованы
        if is_deepspeed_zero3:
            logger.info("⚡ ZeRO-3: пропускаем тестовый forward pass (параметры sharded)")
            self.model.train()
            return  # Выходим из _load_model для ZeRO-3
        
        self.model.train()  # Убеждаемся что в train режиме
        test_input = torch.randint(0, 1000, (1, 10), device=self.device)
        test_mask = torch.ones_like(test_input)
        mp = (getattr(self.config, "mixed_precision", None) or "bf16").lower()
        use_autocast = torch.cuda.is_available() and mp in ("bf16", "fp16")
        if use_autocast:
            amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
            autocast_ctx = torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
        if self.accelerator is not None:
            try:
                logger.info(
                    "🔎 AMP/Precision: "
                    f"mixed_precision={mp}, "
                    f"autocast={'on' if use_autocast else 'off'}, "
                    f"autocast_dtype={('bf16' if mp=='bf16' else 'fp16') if use_autocast else 'n/a'}, "
                    f"grad_scaler={'on' if getattr(self.accelerator, 'scaler', None) is not None else 'off'}"
                )
            except Exception:
                pass
        with torch.enable_grad():
            with autocast_ctx:
                test_output = self.model(input_ids=test_input, attention_mask=test_mask, use_cache=False)
        if not test_output.logits.requires_grad:
            logger.warning("⚠️ Тестовый forward pass не требует градиентов! Это может быть проблемой.")
            # Попробуем принудительно включить градиенты для всех параметров
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("Принудительно включены градиенты для всех параметров")
        
        # Финальная проверка использования памяти
        if torch.cuda.is_available():
            memory_final = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            logger.info("=" * 60)
            logger.info("📊 ИТОГОВОЕ ИСПОЛЬЗОВАНИЕ ПАМЯТИ ПОСЛЕ ЗАГРУЗКИ МОДЕЛИ:")
            logger.info(f"  - Выделено (allocated): {memory_final:.1f} MB")
            logger.info(f"  - Зарезервировано (reserved): {memory_reserved:.1f} MB")
            logger.info(f"  - Всего использовано с начала: +{memory_final - memory_before:.1f} MB")
            logger.info("=" * 60)
    
    def _apply_lora(self):
        """Применяет LoRA адаптеры к модели."""
        logger.info("🔧 Применение LoRA адаптеров...")
        
        # Проверяем использование памяти до применения LoRA
        memory_before_lora = 0.0
        if torch.cuda.is_available():
            memory_before_lora = torch.cuda.memory_allocated() / (1024 ** 2)
        
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # ВАЖНО:
            # `prepare_model_for_kbit_training` нужно ТОЛЬКО для QLoRA (4/8bit).
            # Для обычной LoRA на fp16/bf16 оно может кастить LayerNorm в fp32 => FlashAttention падает.
            if self.config.use_4bit or self.config.use_8bit:
                logger.info("📦 Подготовка модели для QLoRA (prepare_model_for_kbit_training)...")
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=bool(getattr(self.config, "grad_checkpoint", False)),
                )
            else:
                # Обычная LoRA: gradient checkpointing — только если включено в UI
                if bool(getattr(self.config, "grad_checkpoint", False)) and hasattr(self.model, "gradient_checkpointing_enable"):
                    try:
                        self.model.gradient_checkpointing_enable()
                        logger.info("✅ Gradient checkpointing включен для LoRA (из UI)")
                    except Exception as e:
                        logger.warning(f"Не удалось включить gradient checkpointing для LoRA: {e}")
            
            # Проверяем что модель действительно квантизирована (если запрашивалось)
            if self.config.use_4bit or self.config.use_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    # Проверяем что есть квантизированные параметры
                    quantized_params = sum(
                        1 for p in self.model.parameters() 
                        if hasattr(p, 'quant_state') or str(p.dtype) == 'torch.uint8'
                    )
                    if quantized_params > 0:
                        logger.info(f"✅ Модель квантизирована: найдено {quantized_params} квантизированных параметров")
                    else:
                        logger.warning("⚠️ Модель может быть не квантизирована! Проверьте BitsAndBytesConfig.")
                except:
                    pass
            
            # Используем "all-linear" для автоматического определения модулей (как в re-grpo)
            # Это более надежно чем ручной список, особенно для разных архитектур
            if isinstance(self.config.lora_target_modules, list) and len(self.config.lora_target_modules) > 0:
                target_modules = self.config.lora_target_modules
                logger.info(f"📋 Используем target_modules из конфига: {target_modules}")
            else:
                # Fallback на "all-linear" если список пустой или не указан
                target_modules = "all-linear"
                logger.info("📋 Используем target_modules='all-linear' для автоматического определения модулей")
            
            # ВАЖНО: Валидация LoRA параметров перед созданием конфигурации
            # Все параметры должны быть явно указаны (без fallback)
            lora_r = self.config.lora_r
            lora_alpha = self.config.lora_alpha
            lora_dropout = self.config.lora_dropout
            
            # Строгая валидация: если параметры None - это ошибка конфигурации
            if lora_r is None:
                raise ValueError(
                    "❌ lora_r = None! "
                    "lora_r должен быть явно указан в конфигурации. "
                    "Проверьте что render_grpo_sidebar_config() возвращает lora_r."
                )
            
            if lora_alpha is None:
                raise ValueError(
                    "❌ lora_alpha = None! "
                    "lora_alpha должен быть явно указан в конфигурации. "
                    "Проверьте что render_grpo_sidebar_config() возвращает lora_alpha."
                )
            
            # lora_dropout может использовать дефолт из GRPOConfig (0.1)
            if lora_dropout is None:
                lora_dropout = self.config.lora_dropout  # Используем дефолт из dataclass
            
            # Валидация типов и значений
            if not isinstance(lora_r, int) or lora_r <= 0:
                raise ValueError(
                    f"❌ Невалидный lora_r: {lora_r} (тип: {type(lora_r)}). "
                    f"Должно быть положительное целое число. "
                    f"Проверьте что в UI передается число, а не None или строка."
                )
            
            if not isinstance(lora_alpha, (int, float)) or lora_alpha <= 0:
                raise ValueError(
                    f"❌ Невалидный lora_alpha: {lora_alpha} (тип: {type(lora_alpha)}). "
                    f"Должно быть положительное число. "
                    f"Проверьте что в UI передается число, а не None или строка."
                )
            
            logger.info(f"🔧 Создание LoRA конфигурации:")
            logger.info(f"  - r (rank): {lora_r}")
            logger.info(f"  - alpha: {lora_alpha}")
            logger.info(f"  - dropout: {lora_dropout}")
            logger.info(f"  - target_modules: {target_modules}")
            
            # Это включает градиенты для входных эмбеддингов, без чего градиенты не протекают через LoRA!
            logger.info("🔧 Включение градиентов для входов модели (enable_input_require_grads)...")
            self.model.enable_input_require_grads()
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            logger.info("📦 Применение LoRA адаптеров к модели...")
            self.model = get_peft_model(self.model, lora_config)
            logger.info("✅ LoRA адаптеры применены!")

            # Для FlashAttention важно, чтобы hidden_states были fp16/bf16.
            # После LoRA некоторые модули/веса могут оказаться в fp32 и промоутить dtype в forward.
            # Для НЕ-квантизированных моделей приводим LoRA-параметры к AMP dtype (bf16/fp16).
            if not (self.config.use_4bit or self.config.use_8bit):
                mp = (getattr(self.config, "mixed_precision", None) or "bf16").lower()
                if mp in ("bf16", "fp16") and torch.cuda.is_available():
                    target_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
                    try:
                        casted = 0
                        for name, p in self.model.named_parameters():
                            if "lora" in name.lower() and getattr(p, "dtype", None) != target_dtype:
                                p.data = p.data.to(target_dtype)
                                casted += 1
                        if casted > 0:
                            logger.info(f"✅ Привели {casted} LoRA параметров к dtype={target_dtype} (для совместимости с flash-attn)")
                    except Exception as e:
                        logger.warning(f"Не удалось привести LoRA параметры к AMP dtype: {e}")
            
            # Проверяем использование памяти после применения LoRA
            if torch.cuda.is_available():
                memory_after_lora = torch.cuda.memory_allocated() / (1024 ** 2)
                logger.info(f"💾 Память CUDA после применения LoRA: {memory_after_lora:.1f} MB (+{memory_after_lora - memory_before_lora:.1f} MB)")
            
            # PEFT автоматически выводит информацию о trainable параметрах
            logger.info("📊 Информация о trainable параметрах (от PEFT):")
            self.model.print_trainable_parameters()
            
            # Дополнительная проверка: убеждаемся что только LoRA параметры trainable
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            frozen_params = total_params - trainable_params
            trainable_percent = 100 * trainable_params / total_params if total_params > 0 else 0
            
            logger.info(f"📊 Детальная проверка параметров:")
            logger.info(f"  - Всего параметров: {total_params:,}")
            logger.info(f"  - Trainable (LoRA): {trainable_params:,} ({trainable_percent:.2f}%)")
            logger.info(f"  - Frozen (базовая модель): {frozen_params:,} ({100 - trainable_percent:.2f}%)")
            
            # Проверяем что только LoRA параметры требуют градиентов
            non_lora_trainable = 0
            lora_trainable = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'lora' in name.lower():
                        lora_trainable += param.numel()
                    else:
                        non_lora_trainable += param.numel()
            
            if non_lora_trainable > 0:
                logger.warning(
                    f"⚠️  Найдено {non_lora_trainable:,} trainable параметров БЕЗ 'lora' в названии! "
                    f"Это может означать что LoRA не применился правильно."
                )
            else:
                logger.info(f"✅ Все trainable параметры - это LoRA адаптеры ({lora_trainable:,} параметров)")
            
            if trainable_percent > 5.0:
                logger.warning(
                    f"⚠️  Слишком много trainable параметров ({trainable_percent:.2f}%)! "
                    f"Возможно LoRA не применился правильно. Ожидается < 1% для LoRA."
                )
            elif trainable_percent < 0.1:
                logger.warning(
                    f"⚠️  Слишком мало trainable параметров ({trainable_percent:.2f}%)! "
                    f"Возможно LoRA не применился правильно."
                )
            else:
                logger.info(f"✅ Процент trainable параметров в норме ({trainable_percent:.2f}%)")
            
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
        logger.info(f"🔧 Настройка оптимизатора:")
        
        # При ZeRO-3 параметры sharded - нужен специальный подсчёт
        if getattr(self, 'is_deepspeed_zero3', False):
            # Для ZeRO-3: все параметры trainable при full fine-tuning
            # Используем оценку из конфига модели
            try:
                from transformers import AutoConfig
                model_config = AutoConfig.from_pretrained(self.model_name)
                hidden = getattr(model_config, 'hidden_size', 768)
                layers = getattr(model_config, 'num_hidden_layers', 12)
                vocab = getattr(model_config, 'vocab_size', 32000)
                total_params = vocab * hidden + layers * 12 * hidden * hidden
                num_trainable = total_params  # full fine-tuning
                logger.info(f"  ⚡ ZeRO-3: приблизительная оценка параметров из конфига")
                logger.info(f"  - Trainable параметров: ~{num_trainable:,} (оценка)")
            except Exception as e:
                logger.warning(f"  ⚠️ Не удалось оценить параметры для ZeRO-3: {e}")
                num_trainable = 1  # placeholder для избежания division by zero
                total_params = 1
            trainable_params = list(self.model.parameters())  # все параметры для ZeRO-3
        else:
            # Стандартный подсчёт для не-ZeRO-3
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            num_trainable = sum(p.numel() for p in trainable_params)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            if total_params > 0:
                logger.info(f"  - Trainable параметров: {num_trainable:,} / {total_params:,} ({100*num_trainable/total_params:.2f}%)")
            else:
                logger.info(f"  - Trainable параметров: {num_trainable:,}")
            logger.info(f"  - Групп параметров: {len(trainable_params)}")
        
        # Проверяем что есть параметры для оптимизатора
        # Для ZeRO-3 пропускаем эту проверку - DeepSpeed сам управляет параметрами
        if len(trainable_params) == 0 and not getattr(self, 'is_deepspeed_zero3', False):
            raise RuntimeError("❌ Нет trainable параметров для оптимизатора! Проверьте LoRA конфигурацию.")
        
        # Определяем используется ли DeepSpeed (для выбора оптимизатора)
        uses_deepspeed = (
            self.accelerator is not None and 
            getattr(self.accelerator.state, 'deepspeed_plugin', None) is not None
        )
        
        # При DeepSpeed используем стандартный AdamW - bitsandbytes может конфликтовать с CPU offload
        if uses_deepspeed:
            logger.info("⚡ DeepSpeed режим: используется стандартный AdamW (совместимо с ZeRO offload)")
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            try:
                from bitsandbytes.optim import AdamW8bit
                logger.info("✅ Используется AdamW8bit (8-bit оптимизатор для экономии памяти)")
                self.optimizer = AdamW8bit(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            except ImportError:
                logger.info("ℹ️  Используется стандартный AdamW (bitsandbytes не установлен)")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
        
        # Оцениваем память оптимизатора
        # AdamW хранит: градиенты (fp16), momentum (fp16), variance (fp16) = 3x trainable_params
        optimizer_memory_mb = (num_trainable * 3 * 2) / (1024 ** 2)  # 3 состояния * 2 bytes (fp16)
        logger.info(f"💾 Примерная память оптимизатора: ~{optimizer_memory_mb:.1f} MB")
        
        # Проверяем что оптимизатор действительно использует только trainable параметры
        # Для ZeRO-3 пропускаем - параметры sharded
        if not getattr(self, 'is_deepspeed_zero3', False):
            optimizer_param_count = sum(p.numel() for group in self.optimizer.param_groups for p in group['params'])
            if optimizer_param_count != num_trainable:
                logger.warning(
                    f"⚠️  Несоответствие: оптимизатор использует {optimizer_param_count:,} параметров, "
                    f"а trainable параметров {num_trainable:,}"
                )
            else:
                logger.info(f"✅ Оптимизатор использует только trainable параметры ({optimizer_param_count:,})")
        else:
            logger.info(f"⚡ ZeRO-3: пропускаем проверку параметров оптимизатора (sharded)")
        
        # Scheduler
        # ВАЖНО: scheduler.step() вызывается на optimizer-step, поэтому num_training_steps должен быть в optim-шагах.
        min_lr_ratio = float(getattr(self.config, "min_lr_ratio", 0.0) or 0.0)
        if min_lr_ratio > 0:
            from torch.optim.lr_scheduler import LambdaLR
            warmup = int(self.config.warmup_steps or 0)
            total = max(int(num_training_steps), 1)

            def lr_lambda(step: int):
                # warmup: 0 -> 1
                if warmup > 0 and step < warmup:
                    return float(step) / float(max(1, warmup))
                # cosine with floor
                denom = max(1, total - warmup)
                progress = float(step - warmup) / float(denom)
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        else:
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
            try:
                def _strip_fp32_convert(m):
                    if m is None:
                        return
                    fwd = getattr(m, "forward", None)
                    if fwd is not None and hasattr(fwd, "model_forward"):
                        m.forward = fwd.model_forward  # type: ignore[attr-defined]

                # accelerate может навесить ConvertOutputsToFp32 на разных уровнях обёрток
                _strip_fp32_convert(self.model)
                _strip_fp32_convert(getattr(self.model, "module", None))
                base = self.accelerator.unwrap_model(self.model)
                _strip_fp32_convert(base)
                _strip_fp32_convert(getattr(base, "module", None))
            except Exception as e:
                logger.warning(f"Не удалось отключить accelerate convert_to_fp32: {e}")

        # 🦁 Создаём Loss функцию ПОСЛЕ prepare() — нужна unwrapped модель для Liger Fused Loss
        self._create_loss_function()

        # Rollout engine (отдельная модель для генерации) инициализируем ПОСЛЕ prepare(),
        # чтобы training модель уже была в финальной обёртке (DDP/DeepSpeed).
        self._setup_rollout_engine()
    
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
        # Оценка rollout-шагов (для логов/шедулера). В multi-gpu глобально за шаг проходит batch_size * num_processes.
        world = int(self.accelerator.num_processes) if self.accelerator is not None else 1
        denom = max(int(self.config.batch_size) * max(world, 1), 1)
        steps_per_epoch = math.ceil(num_prompts / denom)
        total_steps_uncapped = steps_per_epoch * self.config.num_epochs
        
        # Лимит "по данным": сколько промптов реально хотим пройти (понятная семантика).
        planned_prompts = int(num_prompts) * int(self.config.num_epochs)
        if getattr(self.config, "max_prompts", None):
            try:
                planned_prompts = min(planned_prompts, int(self.config.max_prompts))
            except Exception:
                pass
        rollout_total_steps = math.ceil(planned_prompts / denom) if planned_prompts > 0 else 0
        
        if self.config.max_steps:
            rollout_total_steps = rollout_total_steps  # max_steps — это лимит optim_step, не rollout_step
        
        # Для UI/ETA: фиксируем плановые шаги (не "max_steps", а реальный план на датасет/лимит).
        self.planned_total_steps = int(rollout_total_steps) if rollout_total_steps else 0
        self.planned_total_steps_uncapped = int(total_steps_uncapped) if total_steps_uncapped else 0

        # Для scheduler: оцениваем число optimizer steps.
        # 1 rollout (на ОДИН процесс) даёт примерно batch_size * group_size опытов.
        # exp_loader drop_last=True => число микробатчей = floor(exps / train_batch_size)
        est_exps = int(self.config.batch_size) * int(self.config.group_size)
        est_micro_batches = max(1, est_exps // max(1, int(self.config.train_batch_size)))
        est_optim_steps_per_rollout = math.ceil(est_micro_batches / max(1, int(self.config.gradient_accumulation_steps)))
        est_optim_steps_per_rollout *= max(1, int(self.config.epochs_per_step))

        planned_optim_steps = int(rollout_total_steps) * int(est_optim_steps_per_rollout)
        if self.config.max_steps:
            # max_steps — явный лимит optim_step из UI
            planned_optim_steps = min(int(planned_optim_steps), int(self.config.max_steps))
        self.planned_optim_total_steps = max(int(planned_optim_steps), 1)
        
        logger.info(
            f"Начало обучения: {num_prompts} промптов, ~{int(rollout_total_steps)} rollout-шагов, "
            f"~{int(self.planned_optim_total_steps)} optim-шагов"
        )
        
        # Настройка оптимизатора
        self._setup_optimizer(self.planned_optim_total_steps)
        
        # Создаём output директорию
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ВАЖНО: Записываем начальный heartbeat в metrics.jsonl
        # Это сигнализирует UI что обучение стартовало
        if self.is_main_process:
            initial_metrics = {
                "step": 0,
                "status": "training_started",
                "epoch": 0,
                "total_prompts": num_prompts,
                "planned_optim_steps": self.planned_optim_total_steps,
            }
            self._log_metrics(initial_metrics, jsonl_only=True)
            logger.info("📝 Записан начальный heartbeat в metrics.jsonl")
        
        # DataLoader для промптов
        prompt_loader = DataLoader(
            list(range(len(dataset))),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        # ВАЖНО (как в re-grpo accelerate): при multi-gpu делим промпты между процессами
        if self.accelerator is not None:
            prompt_loader = self.accelerator.prepare(prompt_loader)
        
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
        
        # Финальное сохранение (все ранки должны войти для синхронизации)
        self._save_checkpoint(output_dir / "final", is_final=True)
        
        logger.info("Обучение завершено!")
        
        # Записываем финальный статус в metrics.jsonl для UI
        if self.is_main_process:
            world = int(self.accelerator.num_processes) if self.accelerator is not None else 1
            prompts_processed = int(self.rollout_step) * int(self.config.batch_size) * max(world, 1)
            final_metrics = {
                "step": self.global_step,
                "rollout_step": self.rollout_step,
                "status": "completed",
                "total_prompts_processed": prompts_processed,
            }
            self._log_metrics(final_metrics, jsonl_only=True)
            logger.info("📝 Записан финальный статус 'completed' в metrics.jsonl")
        
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
                    system_prompt=getattr(self.config, 'user_system_prompt', None),
                )
                for s in batch_samples
            ]
            reference_answers = [s.reference_answer for s in batch_samples]
            metadata_list = [s.metadata if hasattr(s, 'metadata') else {} for s in batch_samples]
            # group_id должен быть уникальным для каждой группы (особенно при dynamic sampling с добором)
            desired_groups = len(batch_samples)
            group_ids = self._next_group_uids(desired_groups)
            
            # Генерация rollout'ов
            logger.info(f"🎲 Batch {batch_idx}: начинаем генерацию {len(prompts)} промптов...")
            self.replay_buffer.clear()
            batch_rewards = self._generate_and_collect(
                prompts=prompts,
                reference_answers=reference_answers,
                prompt_ids=group_ids,
                metadata_list=metadata_list,
            )
            logger.info(f"✅ Batch {batch_idx}: генерация завершена, rewards={len(batch_rewards)}")
            refill_rounds = 0
            # DAPO dynamic sampling: добор групп до нужного размера (НЕ уменьшаем batch автоматически)
            if self.config.dynamic_sampling and self.config.max_refill_rounds > 0:
                import random
                max_refill_rounds = self.config.max_refill_rounds  # из UI/config (по умолчанию 3)
                while self.replay_buffer.get_stats().get("num_groups", 0) < desired_groups and refill_rounds < max_refill_rounds:
                    missing = desired_groups - int(self.replay_buffer.get_stats().get("num_groups", 0))
                    if missing <= 0:
                        break
                    # добираем новые промпты (с replacement допустимо, но group_id уникальный)
                    extra_indices = [random.randrange(0, len(dataset)) for _ in range(missing)]
                    extra_samples = [dataset[i] for i in extra_indices]
                    extra_prompts = [
                        build_reasoning_prompt(
                            s.prompt, 
                            self.tokenizer, 
                            self.config.reasoning_format,
                            system_prompt=getattr(self.config, 'user_system_prompt', None),
                        )
                        for s in extra_samples
                    ]
                    extra_refs = [s.reference_answer for s in extra_samples]
                    extra_metadata = [s.metadata if hasattr(s, 'metadata') else {} for s in extra_samples]
                    extra_group_ids = self._next_group_uids(len(extra_samples))
                    extra_rewards = self._generate_and_collect(
                        prompts=extra_prompts,
                        reference_answers=extra_refs,
                        prompt_ids=extra_group_ids,
                        metadata_list=extra_metadata,
                    )
                    batch_rewards.extend(extra_rewards)
                    refill_rounds += 1

                if self.replay_buffer.get_stats().get("num_groups", 0) < desired_groups:
                    logger.warning(
                        f"⚠️ dynamic_sampling: не удалось добрать группы до {desired_groups}. "
                        f"Получилось {self.replay_buffer.get_stats().get('num_groups', 0)} после {refill_rounds} доборов. "
                        f"Возможная причина: модель даёт одинаковый reward на большинстве промптов."
                    )
                
                # 🔥 Barrier после refills чтобы все GPU закончили генерацию
                if self.accelerator.num_processes > 1:
                    self.accelerator.wait_for_everyone()

            epoch_rewards.extend(batch_rewards)
            
            # Обучение на собранном опыте
            buffer_size = len(self.replay_buffer)
            
            # ============================================================
            # 🔥 КРИТИЧНО: Синхронизация между GPU перед training!
            # ============================================================
            # При multi-GPU dynamic_sampling может дать разные buffer_size на разных GPU.
            # Без barrier один GPU ждёт другой в DDP forward → NCCL timeout!
            if self.accelerator.num_processes > 1:
                # Собираем buffer_size со всех GPU
                buffer_tensor = torch.tensor([buffer_size], device=self.device)
                all_buffers = self.accelerator.gather(buffer_tensor)
                min_buffer = int(all_buffers.min().item())
                max_buffer = int(all_buffers.max().item())
                
                # Если хотя бы один GPU имеет пустой буфер — все пропускают training
                if min_buffer == 0:
                    if buffer_size > 0:
                        logger.warning(
                            f"⚠️ Multi-GPU sync: пропускаем training (другой GPU имеет пустой буфер). "
                            f"Local buffer: {buffer_size}, all buffers: {all_buffers.tolist()}"
                        )
                    buffer_size = 0  # Force skip
                
                # Barrier для синхронизации перед training
                self.accelerator.wait_for_everyone()
            
            if buffer_size == 0:
                logger.warning(
                    f"⚠️ Буфер пуст на шаге {self.global_step}! "
                    f"Проверьте dynamic_sampling и reward функцию."
                )
                train_metrics = {"loss": 0.0, "kl": 0.0, "grad_norm": 0.0}
            else:
                train_metrics = self._train_on_buffer()
            
            epoch_losses.append(train_metrics.get("loss", 0))
            
            # ---- Мониторинг ----
            # ВАЖНО: UI должен получать прогресс сразу, иначе он "зависает" на STARTING.
            # Поэтому пишем heartbeat метрики КАЖДЫЙ rollout, а в консоль/W&B — по log_steps.
            batch_reward_mean = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
            group_size = max(int(self.config.group_size), 1)
            prompts_generated = int(len(batch_rewards) // group_size) if group_size > 0 else 0
            num_groups_used = int(self.replay_buffer.get_stats().get("num_groups", 0))
            completions_generated = int(len(batch_rewards))
            experiences_tuned = int(len(self.replay_buffer))
            filtered_groups = max(0, prompts_generated - num_groups_used)

            # Кумулятивные счётчики (на каждый rollout, чтобы UI показывал "по факту")
            self.cum_prompts_generated += prompts_generated
            self.cum_prompts_used += num_groups_used
            self.cum_completions_generated += completions_generated
            self.cum_experiences_tuned += experiences_tuned

            heartbeat = {
                "step": self.global_step,
                "epoch": epoch,
                "batch_reward_mean": batch_reward_mean,
                "buffer_size": buffer_size,
                "rollouts_count": len(batch_rewards),
                "prompts_generated": prompts_generated,
                "prompts_used": num_groups_used,
                "filtered_groups": filtered_groups,
                "completions_generated": completions_generated,
                "experiences_tuned": experiences_tuned,
                "refill_rounds": refill_rounds,
                "cum_prompts_generated": int(self.cum_prompts_generated),
                "cum_prompts_used": int(self.cum_prompts_used),
                "cum_completions_generated": int(self.cum_completions_generated),
                "cum_experiences_tuned": int(self.cum_experiences_tuned),
                **train_metrics,
            }

            # Пишем метрики каждый rollout (для UI), а консоль/W&B — только по log_steps.
            should_log = (self.global_step % max(int(self.config.log_steps), 1) == 0)
            self._log_metrics(heartbeat, jsonl_only=(not should_log))
            
            # Обновляем progress bar
            pbar.set_postfix({
                "reward": f"{sum(batch_rewards) / max(len(batch_rewards), 1):.3f}",
                "loss": f"{train_metrics.get('loss', 0):.4f}",
            })
            
            # Сохранение чекпоинта
            if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                # ВАЖНО: в distributed режиме сохранение должно вызываться ВСЕМИ процессами,
                # иначе возможны рассинхронизации/таймауты на collectives.
                self._save_checkpoint(Path(self.config.output_dir) / f"step_{self.global_step}")

            # Rollout-step завершён (1 batch промптов -> сбор rollout -> train on buffer)
            self.rollout_step += 1
            
            # 🔥 Barrier в конце batch для синхронизации multi-GPU
            if self.accelerator.num_processes > 1:
                self.accelerator.wait_for_everyone()
            
            # Проверяем max_steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

            # Проверяем лимит по данным (сколько промптов обработать)
            if getattr(self.config, "max_prompts", None):
                try:
                    world = int(self.accelerator.num_processes) if self.accelerator is not None else 1
                    prompts_seen = int(self.rollout_step) * int(self.config.batch_size) * max(world, 1)
                    if prompts_seen >= int(self.config.max_prompts):
                        logger.info(
                            f"Достигнут max_prompts={int(self.config.max_prompts)} "
                            f"(оценка prompts_seen={prompts_seen}), останавливаем обучение"
                        )
                        break
                except Exception:
                    # если что-то не так — не ломаем обучение
                    pass
        
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
        prompt_ids: Optional[List[int]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        """
        Генерирует rollout'ы и собирает опыт в буфер.
        
        Args:
            prompts: Список промптов
            reference_answers: Эталонные ответы
            prompt_ids: ID для группировки
            metadata_list: Metadata для каждого промпта (для reward функций)
        
        Returns:
            Список всех rewards
        """
        # ВАЖНО: При ZeRO-3 все процессы должны войти в generate() синхронно
        # Иначе будет deadlock при сборке параметров
        if getattr(self, 'is_deepspeed_zero3', False) and self.accelerator is not None:
            logger.info("⚡ ZeRO-3: синхронизация процессов перед генерацией...")
            self.accelerator.wait_for_everyone()
            logger.info("⚡ ZeRO-3: синхронизация завершена, начинаем генерацию")
        
        self.model.eval()
        all_rewards = []
        
        # Обёртка для reward функции
        def reward_wrapper(completion, reference_answer, reasoning_format, is_truncated, metadata=None):
            return self.reward_fn(
                completion=completion,
                reference_answer=reference_answer,
                reasoning_format=reasoning_format,
                is_truncated=is_truncated,
                metadata=metadata or {},
            )
        
        # Генерируем rollout'ы.
        # ВАЖНО: Для ZeRO-3/FSDP generation внутри training engine может быть на порядки медленнее.
        # Если включён rollout_engine — генерируем отдельной моделью, а training модель
        # используем только для teacher-forcing logprobs + backprop.
        use_rollout_engine = bool(getattr(self.config, "use_rollout_engine", False))
        backend = getattr(self.config, "rollout_engine_backend", "hf")
        
        # Проверяем доступность rollout engine
        # При vLLM + multi-GPU он загружен только на main process
        rollout_engine_available = use_rollout_engine and self.rollout_engine is not None
        
        if rollout_engine_available:
            # Синхронизируем веса training -> rollout (обычно trainable-only, т.е. LoRA)
            self._sync_rollout_engine_weights(force=False)
            
            if backend == "hf":
                self.rollout_engine.ensure_on_device()
                rollouts = generate_rollouts(
                    model=self.rollout_engine.model,  # type: ignore[arg-type]
                    tokenizer=self.tokenizer,
                    prompts=prompts,
                    reference_answers=reference_answers,
                    reward_fn=reward_wrapper,
                    config=self.config,
                    accelerator=None,          # rollout модель не DeepSpeed/DDP wrapper
                    reference_model=None,      # ref logprobs считаем на training стороне, если нужно
                    device=self.rollout_engine.device,
                    prompt_ids=prompt_ids,
                    metadata_list=metadata_list,
                )
                self.rollout_engine.maybe_offload()
            elif backend == "vllm":
                # Поддерживаем оба типа: VLLMRolloutEngine и VLLMSubprocessEngine
                if not isinstance(self.rollout_engine, (VLLMRolloutEngine, VLLMSubprocessEngine)):
                    raise RuntimeError("rollout_engine backend mismatch (expected VLLMRolloutEngine or VLLMSubprocessEngine)")
                
                # VLLMSubprocessEngine работает через IPC — не нужно переключать GPU
                # VLLMRolloutEngine на той же GPU — тоже не нужно переключать
                rollouts = generate_rollouts_vllm(
                    vllm_engine=self.rollout_engine,
                    tokenizer=self.tokenizer,
                    prompts=prompts,
                    reference_answers=reference_answers,
                    reward_fn=reward_wrapper,
                    config=self.config,
                    prompt_ids=prompt_ids,
                    metadata_list=metadata_list,
                )
            else:
                raise NotImplementedError(f"Unknown rollout_engine_backend='{backend}'")
        else:
            # ВАЖНО: Передаем accelerator для unwrap модели (DDP не поддерживает generate напрямую)
            rollouts = generate_rollouts(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=prompts,
                reference_answers=reference_answers,
                reward_fn=reward_wrapper,
                config=self.config,
                accelerator=self.accelerator,
                reference_model=self.reference_model,
                device=self.device,
                prompt_ids=prompt_ids,
                metadata_list=metadata_list,
            )
        
        # ВАЖНО: Синхронизация после генерации для ZeRO-3
        if getattr(self, 'is_deepspeed_zero3', False) and self.accelerator is not None:
            logger.debug("⚡ ZeRO-3: синхронизация после генерации")
            self.accelerator.wait_for_everyone()
        
        # Конвертируем в Experience и добавляем в буфер
        # ВАЖНО: Обрабатываем и сразу удаляем, чтобы не копить память
        num_rollouts = len(rollouts)
        for i in range(num_rollouts):
            # Извлекаем и удаляем из списка, чтобы освободить ссылку
            rollout = rollouts.pop(0)
            
            # ВАЖНО: Всегда добавляем rewards в статистику, даже если группа отфильтрована
            rollout_rewards = rollout.rewards.tolist()
            all_rewards.extend(rollout_rewards)
            
            # Отладочное логирование (только для первых нескольких rollout'ов)
            if i < 2:
                logger.debug(
                    f"Rollout {rollout.metadata.get('prompt_idx', 0)}: "
                    f"rewards={[f'{r:.3f}' for r in rollout_rewards]}, "
                    f"mean={sum(rollout_rewards)/len(rollout_rewards):.3f}, "
                    f"completions_len={[len(c) for c in rollout.completions[:2]]}"
                )
            
            # Логируем семплы для мониторинга (периодически)
            if self.global_step % max(self.config.log_steps, 1) == 0 and rollout.metadata.get("prompt_idx", 0) == 0:
                self._log_sample(rollout)
            
            experiences = rollout_to_experiences(
                rollout=rollout,
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.config,
                reference_model=self.reference_model,
                device=self.device,
                accelerator=self.accelerator,
            )
            
            # Сохраняем метаданные перед удалением rollout
            prompt_idx = rollout.metadata.get("prompt_id", rollout.metadata.get("prompt_idx", 0))
            rollout_completions_len = len(rollout.completions)
            
            # 🎓 SDPO: сохраняем успешные траектории для self-distillation
            if hasattr(self, '_successful_trajectories') and isinstance(self.loss_fn, SDPOLoss):
                sdpo_threshold = getattr(self.config, 'sdpo_success_threshold', 0.5)
                for comp_idx, (reward, completion) in enumerate(zip(rollout_rewards, rollout.completions)):
                    if reward >= sdpo_threshold:
                        # Сохраняем пару (prompt, completion) для reprompting
                        if prompt_idx not in self._successful_trajectories:
                            self._successful_trajectories[prompt_idx] = []
                        
                        trajectory_data = {
                            'prompt': rollout.prompt,  # Исходный prompt (текст)
                            'completion': completion,   # Успешный completion
                            'reward': reward,
                        }
                        
                        # Храним только последние N успешных (чтобы не раздувать память)
                        if len(self._successful_trajectories[prompt_idx]) < 5:
                            self._successful_trajectories[prompt_idx].append(trajectory_data)
                        else:
                            # Заменяем случайный старый
                            import random
                            replace_idx = random.randint(0, 4)
                            self._successful_trajectories[prompt_idx][replace_idx] = trajectory_data
            
            # Явно удаляем rollout после использования
            del rollout
            
            # ВАЖНО: Перемещаем опыты на CPU для экономии VRAM (как в re-grpo)
            # Это критично для Multi-GPU и больших буферов
            cpu_device = torch.device("cpu")
            experiences_cpu = [exp.to(cpu_device) for exp in experiences]
            
            # Dynamic sampling: фильтруем zero-gradient группы
            filter_zero = self.config.dynamic_sampling
            added = self.replay_buffer.append_group(
                experiences_cpu,
                prompt_id=prompt_idx,
                filter_zero_gradient=filter_zero,
            )
            
            # Освобождаем GPU память после перемещения на CPU
            del experiences
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if not added and filter_zero:
                logger.debug(
                    f"Группа {prompt_idx} отфильтрована "
                    f"(zero-gradient, rewards={rollout_rewards})"
                )
            
            self.total_rollouts += rollout_completions_len
        
        # Логируем статистику по rewards
        if all_rewards:
            logger.debug(
                f"Batch rewards: mean={sum(all_rewards)/len(all_rewards):.4f}, "
                f"min={min(all_rewards):.4f}, max={max(all_rewards):.4f}, "
                f"count={len(all_rewards)}"
            )
        else:
            logger.warning("⚠️ Нет rewards в batch! Проверьте reward функцию и генерацию.")
        
        return all_rewards
    
    def _train_on_buffer(self) -> Dict[str, float]:
        """
        Обучение на собранном опыте в буфере.
        
        Returns:
            Метрики обучения
        """
        self.model.train()
        
        buffer_size = len(self.replay_buffer)
        if buffer_size == 0:
            logger.warning(
                "⚠️ Буфер пуст, пропускаем обучение. "
                "Возможные причины: все группы отфильтрованы (dynamic_sampling) или нет опыта."
            )
            return {"loss": 0.0, "kl": 0.0, "grad_norm": 0.0}
        
        logger.debug(f"Обучение на буфере: {buffer_size} опытов")
        
        # DataLoader для experience
        exp_loader = DataLoader(
            self.replay_buffer.items,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )
        
        # ============================================================
        # 🔥 КРИТИЧНО: Синхронизация количества батчей между GPU!
        # ============================================================
        # При multi-GPU каждый GPU может иметь разное количество элементов в буфере.
        # Это приводит к разному количеству итераций → DDP deadlock!
        local_num_batches = len(exp_loader)
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            # Собираем количество батчей со всех GPU
            num_batches_tensor = torch.tensor([local_num_batches], device=self.device)
            all_num_batches = self.accelerator.gather(num_batches_tensor)
            min_batches = int(all_num_batches.min().item())
            
            if min_batches != local_num_batches:
                logger.info(
                    f"🔄 Multi-GPU sync: ограничиваем итерации до {min_batches} "
                    f"(local={local_num_batches}, all={all_num_batches.tolist()})"
                )
            local_num_batches = min_batches
        
        epoch_losses = []
        epoch_kls = []
        epoch_grad_norms = []
        
        from contextlib import nullcontext

        for epoch_idx in range(self.config.epochs_per_step):
            for batch_idx, exp_batch in enumerate(exp_loader):
                # 🔥 Прерываем если достигли min_batches (для синхронизации multi-GPU)
                if batch_idx >= local_num_batches:
                    break
                exp_batch = exp_batch.to(self.device)
                accumulate_ctx = (
                    self.accelerator.accumulate(self.model)
                    if self.accelerator is not None
                    else nullcontext()
                )
                
                # ВАЖНО: Логирование размеров батча для диагностики OOM
                batch_size = exp_batch.sequences.size(0)
                max_seq_len = exp_batch.sequences.size(1)
                total_tokens = batch_size * max_seq_len
                
                if batch_idx == 0 and epoch_idx == 0:
                    # ВАЖНО: Для DDP модели нужно unwrap для доступа к config
                    if self.accelerator is not None:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        vocab_size = unwrapped_model.config.vocab_size
                    else:
                        vocab_size = self.model.config.vocab_size
                    
                    estimated_logits_memory = total_tokens * vocab_size * 2 / (1024**3)  # GB (fp16)
                    logger.info(
                        f"📊 Размеры батча для обучения: "
                        f"batch_size={batch_size}, max_seq_len={max_seq_len}, "
                        f"total_tokens={total_tokens:,}, "
                        f"примерная память для logits: ~{estimated_logits_memory:.2f} GB"
                    )
                
                # Защита от очевидного OOM: оцениваем минимум под logits + разумный overhead и сверяем со свободной памятью.
                # Это НЕ "авто-настройка" — просто ранняя, понятная ошибка с рекомендациями.
                if torch.cuda.is_available():
                    try:
                        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
                        free_gb = free_bytes / (1024**3)
                        # logits fp16/bf16 + временные буферы + активации => очень грубо 2.2x
                        # (для Qwen с большим vocab и длинной seq это ближе к реальности).
                        required_gb = estimated_logits_memory * 2.2
                        # Оставляем немного воздуха под allocator/фрагментацию
                        if required_gb > free_gb * 0.9:
                            raise RuntimeError(
                                "❌ Недостаточно VRAM для шага обучения GRPO.\n"
                                f"  - train_batch_size={batch_size}\n"
                                f"  - max_seq_len={max_seq_len}\n"
                                f"  - оценка logits(fp16/bf16)≈{estimated_logits_memory:.2f} GB\n"
                                f"  - оценка пика (с overhead)≈{required_gb:.2f} GB\n"
                                f"  - свободно сейчас≈{free_gb:.2f} GB (из {total_bytes/(1024**3):.2f} GB)\n\n"
                                "Что делать (без авто-подстроек):\n"
                                "  - Уменьшите **Train Batch Size** (рекомендация: 1–4)\n"
                                "  - Уменьшите **Max new tokens**\n"
                                "  - Включите **LoRA/QLoRA** вместо full fine-tuning\n"
                                "  - При необходимости включите/увеличьте gradient checkpointing (если добавите в UI)\n"
                            )
                    except Exception:
                        # Если mem_get_info недоступен/падает — не блокируем обучение
                        pass
                
                # Forward pass для текущей политики
                # Используем новый API для autocast (исправляем deprecated warning)
                # ВАЖНО: autocast должен быть включен только на CUDA
                mp = (getattr(self.config, "mixed_precision", None) or "bf16").lower()
                use_autocast = (self.accelerator is not None and torch.cuda.is_available() and mp != "no")
                
                if use_autocast:
                    amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
                    autocast_context = torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype)
                else:
                    from contextlib import nullcontext
                    autocast_context = nullcontext()
                
                # Убеждаемся что модель в train режиме
                if not self.model.training:
                    self.model.train()
                
                # ВАЖНО: Освобождаем память перед forward pass (на случай накопления)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with accumulate_ctx:
                    with autocast_context:
                        # Очистка кэша перед тяжелой операцией вычисления логитов
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # 🦁 Разные пути для Liger Fused Loss и стандартного loss
                        if self.use_liger_fused_loss and isinstance(self.loss_fn, LigerFusedGRPOLoss):
                            # LIGER FUSED PATH: hidden_states -> fused loss (НЕ материализуем logits!)
                            # Forward pass с output_hidden_states=True
                            outputs = self.model(
                                input_ids=exp_batch.sequences,
                                attention_mask=exp_batch.attention_mask,
                                output_hidden_states=True,
                                use_cache=False,
                            )
                            
                            # Получаем последний hidden state
                            hidden_states = outputs.hidden_states[-1]
                            
                            # Вычисляем loss через Liger Fused Loss
                            loss, metrics = self.loss_fn.forward_with_experience(
                                hidden_states=hidden_states,
                                experience=exp_batch,
                            )
                            
                            # Освобождаем память
                            del outputs, hidden_states
                        else:
                            # STANDARD PATH: logits -> log_probs -> loss
                            log_probs = compute_log_probs(
                                self.model,
                                exp_batch.sequences,
                                exp_batch.attention_mask,
                                accelerator=self.accelerator,
                            )
                            
                            # 🎓 SDPO: получаем teacher_log_probs через reprompting
                            teacher_log_probs = None
                            distillation_mask = None
                            student_topk_log_probs = None
                            teacher_topk_log_probs = None
                            
                            if isinstance(self.loss_fn, SDPOLoss) and hasattr(self, '_successful_trajectories'):
                                # 🔥 ОПТИМИЗАЦИЯ: Top-K Distillation + Teacher Module (из verl)
                                teacher_log_probs, distillation_mask, student_topk_log_probs, teacher_topk_log_probs = \
                                    self._get_teacher_log_probs(
                                        exp_batch=exp_batch,
                                        device=exp_batch.sequences.device,
                                    )
                            
                            # Вызываем loss функцию
                            if isinstance(self.loss_fn, SDPOLoss):
                                loss, metrics = self.loss_fn(
                                    log_probs=log_probs,
                                    experience=exp_batch,
                                    teacher_log_probs=teacher_log_probs,
                                    distillation_mask=distillation_mask,
                                    student_topk_log_probs=student_topk_log_probs,  # 🔥 Top-K
                                    teacher_topk_log_probs=teacher_topk_log_probs,  # 🔥 Top-K
                                )
                            else:
                                loss, metrics = self.loss_fn(
                                    log_probs=log_probs,
                                    experience=exp_batch,
                                )
                            
                            # Освобождаем память
                            del log_probs
                            if teacher_log_probs is not None:
                                del teacher_log_probs
                            if student_topk_log_probs is not None:
                                del student_topk_log_probs, teacher_topk_log_probs
                
                    # ВАЖНО: Освобождаем промежуточные активации после forward pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # КРИТИЧЕСКИЕ ПРОВЕРКИ перед backward
                    if not loss.isfinite():
                        logger.warning(f"Loss не finite: {loss.item()}, пропускаем batch")
                        continue
                    
                    if not loss.requires_grad:
                        # Детальная диагностика
                        trainable_count = sum(1 for p in self.model.parameters() if p.requires_grad)
                        total_count = sum(1 for _ in self.model.parameters())
                        
                        # Проверяем что происходит с forward pass
                        test_seq = exp_batch.sequences[:1, :5]
                        test_mask = exp_batch.attention_mask[:1, :5]
                        with torch.enable_grad():
                            test_output = self.model(input_ids=test_seq, attention_mask=test_mask)
                            test_logits_grad = test_output.logits.requires_grad
                        
                        raise RuntimeError(
                            f"❌ Loss не требует градиентов!\n"
                            f"  - loss.requires_grad: {loss.requires_grad}\n"
                            f"  - loss.dtype: {loss.dtype}\n"
                            f"  - Модель training: {self.model.training}\n"
                            f"  - Trainable параметры: {trainable_count}/{total_count}\n"
                            f"  - Test logits requires_grad: {test_logits_grad}\n"
                            f"  - use_lora: {self.config.use_lora}\n"
                            f"  - use_4bit: {self.config.use_4bit}\n"
                            f"  - use_8bit: {self.config.use_8bit}\n"
                            f"  - use_autocast: {use_autocast}\n"
                        )
                    
                    # Сохраняем loss для метрик ПЕРЕД backward
                    loss_value = loss.item()
                    
                    # Backward
                    if self.accelerator is not None:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()
                    
                    # ВАЖНО: Освобождаем loss после backward для экономии памяти
                    del loss
                    if torch.cuda.is_available() and batch_idx % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    # Optimizer step делаем ТОЛЬКО когда накопили нужное число micro-steps.
                    do_step = True
                    if self.accelerator is not None:
                        do_step = bool(self.accelerator.sync_gradients)
                    
                    if do_step:
                        # Gradient clipping
                        if self.accelerator is not None:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm,
                            )
                        else:
                            grad_norm = clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm,
                            )
                        
                        # DEBUG: проверяем градиенты LoRA до optimizer.step()
                        if self.is_main_process and self.config.use_lora:
                            lora_grads = []
                            lora_total_numel = 0
                            for name, p in self.model.named_parameters():
                                if p.grad is not None and 'lora' in name.lower():
                                    grad_norm_p = p.grad.norm().item()
                                    lora_grads.append((name, grad_norm_p, p.numel()))
                                    lora_total_numel += p.numel()
                            
                            if lora_grads:
                                avg_lora_grad = sum(g for _, g, _ in lora_grads) / len(lora_grads)
                                max_lora_grad = max(g for _, g, _ in lora_grads)
                                # Показываем: матриц, общее число параметров, avg/max grad norm
                                logger.info(
                                    f"🔍 LoRA grads: {len(lora_grads)} matrices, "
                                    f"{lora_total_numel:,} params, "
                                    f"avg={avg_lora_grad:.6f}, max={max_lora_grad:.6f}"
                                )
                            else:
                                logger.warning(f"⚠️ Нет градиентов для LoRA параметров!")
                        
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        # 🔥 SDPO: EMA Update для Teacher модели (из verl)
                        if isinstance(self.loss_fn, SDPOLoss):
                            self._update_teacher_ema()
                        
                        self.global_step += 1
                    else:
                        grad_norm = 0.0
                    
                    # Собираем метрики
                    epoch_losses.append(loss_value)
                    epoch_kls.append(metrics.get("kl_mean", 0))
                    epoch_grad_norms.append(
                        grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    )
            
            # 🔥 Barrier после каждой epoch для синхронизации multi-GPU
            if self.accelerator is not None and self.accelerator.num_processes > 1:
                self.accelerator.wait_for_everyone()
        
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
                system_prompt=getattr(self.config, 'user_system_prompt', None),
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
                metadata=sample.metadata if hasattr(sample, 'metadata') else {},
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
    
    def _log_metrics(self, metrics: Dict[str, Any], *, jsonl_only: bool = False):
        """Логирует метрики.

        Важно для UI: `metrics.jsonl` должен обновляться регулярно, иначе мониторинг "зависает" на STARTING.
        Поэтому можно писать JSONL даже часто (каждый rollout), а консоль/W&B — реже.
        """
        # В distributed режиме пишем метрики только с main процесса, иначе jsonl будет перемешан.
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        if (not jsonl_only) and self.config.use_wandb and self.wandb_run:
            import wandb
            wandb.log(metrics, step=self.global_step)
        
        # Записываем в JSONL для мониторинга из UI (всегда на main process)
        metrics_file = Path(self.config.output_dir) / "metrics.jsonl"
        ui_metrics_file = None
        try:
            if getattr(self.config, "ui_run_dir", None):
                ui_metrics_file = Path(str(self.config.ui_run_dir)) / "metrics.jsonl"
        except Exception:
            ui_metrics_file = None
        try:
            import json
            from datetime import datetime
            log_entry = {
                    # optim_step: шаг оптимизатора (растёт внутри _train_on_buffer)
                    "step": self.global_step,
                    "optim_step": self.global_step,
                    # rollout_step: сколько батчей промптов (prompts/step) уже обработано
                    "rollout_step": getattr(self, "rollout_step", 0),
                    # current_step для UI: по умолчанию прогресс в GRPO считаем по rollout_step (покрытие датасета)
                    "current_step": int(getattr(self, "rollout_step", 0)),
                    # total_steps для UI/ETA: план на обучение (на датасет/лимиты), а не только max_steps.
                    "total_steps": int(getattr(self, "planned_total_steps", 0)) or None,
                    # planned_total_steps: "план на эпоху" без лимитов по max_prompts/max_steps (полезно для сравнения)
                    "planned_total_steps": int(getattr(self, "planned_total_steps_uncapped", 0)) or None,
                    "reward": metrics.get("batch_reward_mean", metrics.get("reward", 0)),
                    "loss": metrics.get("loss", 0),
                    "kl": metrics.get("kl", 0),
                    "grad_norm": metrics.get("grad_norm", 0),
                    "epoch": metrics.get("epoch", 0),
                    "learning_rate": self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate,
                    "timestamp": datetime.now().isoformat(),
                    # Для прогресса по датасету/скорости:
                    # batch_size здесь = prompts/step на ОДИН процесс; в UI умножаем на num_gpus (из config.json)
                    "prompt_batch_size": int(self.config.batch_size),
                    "group_size": int(self.config.group_size),
                    "train_batch_size": int(self.config.train_batch_size),
                    "epochs_per_step": int(self.config.epochs_per_step),
            }
            # Добавляем все остальные метрики (числа и строки типа status)
            for k, v in metrics.items():
                if k not in log_entry:
                    if isinstance(v, (int, float, str)):
                        log_entry[k] = v
            
            with open(metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            # Дублируем в run_dir UI (если задан), чтобы мониторинг работал независимо от путей output_dir.
            if ui_metrics_file is not None:
                ui_metrics_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ui_metrics_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.debug(f"Не удалось записать метрики в JSONL: {e}")

        # Также логируем в консоль (основные метрики) — опционально
        if (not jsonl_only) and self.is_main_process:
            log_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
                if k in ["step", "loss", "kl", "batch_reward_mean", "buffer_size", "rollouts_count"]
            ])
            if log_str:
                logger.info(f"Step {self.global_step}: {log_str}")
    
    def _log_sample(self, rollout):
        """Логирует семпл (промпт и ответы) для мониторинга в UI."""
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        try:
            import json
            from pathlib import Path
            
            # Сохраняем в output_dir/samples.jsonl (UI будет читать из run_dir)
            samples_file = Path(self.config.output_dir) / "samples.jsonl"
            ui_samples_file = None
            try:
                if getattr(self.config, "ui_run_dir", None):
                    ui_samples_file = Path(str(self.config.ui_run_dir)) / "samples.jsonl"
            except Exception:
                ui_samples_file = None
            
            # Формируем полные тексты (промпт + completion) для отображения
            full_texts = []
            for completion in rollout.completions:
                full_text = rollout.prompt + completion
                full_texts.append(full_text)
            
            sample_entry = {
                "step": self.global_step,
                "prompt": rollout.prompt,
                "reference_answer": rollout.metadata.get("reference_answer", ""),
                "completions": rollout.completions,
                "full_texts": full_texts,  # Промпт + completion для отображения
                "rewards": rollout.rewards.tolist(),
                "timestamp": datetime.now().isoformat(),
            }
            
            with open(samples_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")
            # Дублируем в run_dir UI (если задан)
            if ui_samples_file is not None:
                ui_samples_file.parent.mkdir(parents=True, exist_ok=True)
                with open(ui_samples_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Не удалось записать семпл: {e}")
    
    def _save_checkpoint(self, path: Path, is_final: bool = False):
        """Сохраняет чекпоинт."""
        # DDP-safe сохранение:
        # 1) все ранки синхронизируются до сохранения
        # 2) сохраняет только main process
        # 3) сохранение атомарное: пишем в tmp-dir и делаем rename
        # 4) все ранки синхронизируются после сохранения
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(path.name + "_tmp")

            # чистим старый tmp (если остался от падения) — только на main
            if self.accelerator is None or self.is_main_process:
                if tmp_path.exists():
                    import shutil
                    shutil.rmtree(tmp_path, ignore_errors=True)
            # создаём tmp-dir на всех процессах
            tmp_path.mkdir(parents=True, exist_ok=True)

            if self.accelerator is None:
                # Single-process: сохраняем state модели в HF формате.
                self.model.save_pretrained(tmp_path)
            else:
                # Distributed (DDP/FSDP/DeepSpeed): чекпоинт для resume.
                self.accelerator.save_state(tmp_path)

            # Сохраняем несшардированные артефакты только на main
            if self.accelerator is None or self.is_main_process:
                self.tokenizer.save_pretrained(tmp_path)
                import json
                with open(tmp_path / "grpo_config.json", "w", encoding="utf-8") as f:
                    json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

            # Все дождались записи файлов, затем main делает финализацию
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

            if self.accelerator is None or self.is_main_process:
                if path.exists():
                    import shutil
                    shutil.rmtree(path, ignore_errors=True)
                tmp_path.rename(path)
                logger.info(f"Чекпоинт сохранён: {path}")

            # Обновляем "usable" модель для инференса (перезаписываем final_model), если включено.
            if bool(getattr(self.config, "export_on_checkpoint", False)):
                final_dir = Path(self.config.output_dir) / "final_model"
                final_tmp = final_dir.with_name(final_dir.name + "_tmp")

                # Определяем, нужно ли мерджить LoRA (по умолчанию True для удобства inference)
                merge_lora = bool(getattr(self.config, "merge_lora", True))
                use_lora = bool(getattr(self.config, "use_lora", False))
                
                # ВАЖНО: синхронизация ДО сохранения, чтобы все процессы были на одной точке
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

                # чистим tmp на main
                if self.accelerator is None or self.is_main_process:
                    if final_tmp.exists():
                        import shutil
                        shutil.rmtree(final_tmp, ignore_errors=True)
                    final_tmp.mkdir(parents=True, exist_ok=True)
                
                if self.accelerator is None:
                    # === Single-process ===
                    save_model = self.model
                    
                    if merge_lora:
                        try:
                            from peft import PeftModel
                            if isinstance(save_model, PeftModel):
                                logger.info("🔄 Merging LoRA adapters into base model for final_model...")
                                save_model = save_model.merge_and_unload()
                                logger.info("✅ LoRA adapters merged successfully")
                        except ImportError:
                            pass
                        except Exception as e:
                            logger.warning(f"⚠️ Could not merge LoRA: {e}. Saving as-is.")
                    
                    save_model.save_pretrained(final_tmp, safe_serialization=True)
                
                elif merge_lora and use_lora:
                    # === Distributed + LoRA + merge ===
                    # Только main process сохраняет (без NCCL коллективных операций)
                    # Другие процессы просто ждут
                    if self.is_main_process:
                        try:
                            from peft import PeftModel
                            
                            # Unwrap модель для доступа к PEFT
                            unwrapped = self.model
                            while hasattr(unwrapped, "module"):
                                unwrapped = unwrapped.module
                            
                            if isinstance(unwrapped, PeftModel):
                                logger.info("🔄 Merging LoRA adapters for distributed final_model...")
                                merged_model = unwrapped.merge_and_unload()
                                merged_model.save_pretrained(final_tmp, safe_serialization=True)
                                logger.info("✅ LoRA adapters merged and saved")
                            else:
                                # Не PEFT модель - сохраняем как есть
                                unwrapped.save_pretrained(final_tmp, safe_serialization=True)
                        except ImportError:
                            logger.warning("⚠️ PEFT not available, saving model as-is")
                            unwrapped = self.model
                            while hasattr(unwrapped, "module"):
                                unwrapped = unwrapped.module
                            unwrapped.save_pretrained(final_tmp, safe_serialization=True)
                        except Exception as e:
                            logger.warning(f"⚠️ Could not merge LoRA: {e}. Saving as-is.")
                            unwrapped = self.model
                            while hasattr(unwrapped, "module"):
                                unwrapped = unwrapped.module
                            unwrapped.save_pretrained(final_tmp, safe_serialization=True)
                else:
                    # === Distributed без merge (или без LoRA) ===
                    # Используем accelerate.save_model для корректной работы с FSDP/ZeRO
                    self.accelerator.save_model(self.model, final_tmp, safe_serialization=True)

                # Синхронизация после сохранения модели
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

                if self.accelerator is None or self.is_main_process:
                    self.tokenizer.save_pretrained(final_tmp)
                
                # Финальная синхронизация перед переименованием
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()

                if self.accelerator is None or self.is_main_process:
                    if final_dir.exists():
                        import shutil
                        shutil.rmtree(final_dir, ignore_errors=True)
                    final_tmp.rename(final_dir)
                    logger.info(f"final_model обновлён: {final_dir}")
        finally:
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
    
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
            system_prompt=getattr(self.config, 'user_system_prompt', None),
        )
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_prompt_length,
        ).to(self.device)
        
        # ВАЖНО: Если модель обернута в DDP, используем unwrapped модель для generate()
        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        elif hasattr(self.model, 'module'):
            unwrapped_model = self.model.module
        else:
            unwrapped_model = self.model
        
        with torch.no_grad():
            outputs = unwrapped_model.generate(
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
