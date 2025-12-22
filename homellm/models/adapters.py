"""
homellm.models.adapters
-----------------------
Универсальные адаптеры для загрузки и работы с разными типами моделей.

Поддерживает:
- Home модели (HomeForCausalLM)
- Любые HuggingFace модели (AutoModelForCausalLM)
- LoRA/QLoRA тюнинг через PEFT
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from homellm.models.home_model import HomeConfig, HomeForCausalLM

logger = logging.getLogger(__name__)

# PEFT опционально (для LoRA/QLoRA)
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
        PeftModel,
    )
    from transformers import BitsAndBytesConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    PeftModel = None
    logger.warning("PEFT not available. LoRA/QLoRA features will be disabled.")


def detect_model_type(model_path: Path) -> str:
    """
    Определяет тип модели по config.json.
    
    Returns:
        "home" | "hf" | "unknown"
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        return "unknown"
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        model_type = config.get("model_type", "")
        architectures = config.get("architectures", [])
        
        if model_type == "homellm" or "HomeForCausalLM" in architectures:
            return "home"
        
        # Любая другая модель с model_type - это HF модель
        if model_type:
            return "hf"
        
        # Если есть architectures - тоже HF
        if architectures:
            return "hf"
        
        return "unknown"
    except Exception as e:
        logger.warning(f"Failed to detect model type: {e}")
        return "unknown"


def resolve_adapter(config: Dict[str, Any]) -> "ModelAdapter":
    """
    Определяет и возвращает подходящий адаптер на основе конфига.
    
    Логика:
    1. Если указан base_model_path - определяем тип по config.json
    2. Если указан model_id (для pretrain) - пробуем определить тип
    3. По умолчанию - HomeAdapter
    """
    base_model_path = config.get("base_model_path")
    model_id = config.get("model_id")  # Для pretrain from scratch
    
    if base_model_path:
        model_path = Path(base_model_path)
        model_type = detect_model_type(model_path)
        
        if model_type == "home":
            return HomeAdapter()
        elif model_type == "hf":
            return HFAdapter()
        else:
            logger.warning(f"Could not detect model type for {base_model_path}, using HomeAdapter")
            return HomeAdapter()
    
    if model_id:
        # Для pretrain from scratch - пробуем определить по model_id
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            if cfg.model_type == "homellm":
                return HomeAdapter()
            else:
                return HFAdapter()
        except Exception:
            # Если не получилось - используем Home по умолчанию
            return HomeAdapter()
    
    # По умолчанию - Home
    return HomeAdapter()


class ModelAdapter:
    """Базовый интерфейс для адаптеров моделей."""
    
    def load_tokenizer(
        self,
        source: str | Path,
        trust_remote_code: bool = True,
    ) -> PreTrainedTokenizer:
        """
        Загружает токенизатор из источника.
        
        Args:
            source: путь к модели или model_id
            trust_remote_code: разрешить выполнение кода из модели
        
        Returns:
            Загруженный токенизатор
        """
        raise NotImplementedError
    
    def prepare_tokenizer(self, tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        """
        Подготавливает токенизатор для обучения.
        
        ВАЖНО: используем pad_token = eos_token вместо добавления нового токена.
        Это сохраняет совместимость vocab с базовыми моделями.
        """
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token = eos_token ({tokenizer.eos_token})")
            else:
                # Fallback: добавляем pad только если нет eos
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                logger.warning("Added new pad_token (no eos_token found)")
        return tokenizer
    
    def load_for_training(
        self,
        base_model_path: Optional[str | Path],
        stage: str,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        trust_remote_code: bool = True,
    ) -> Tuple[PreTrainedModel, Any]:  # (model, model_config)
        """
        Загружает модель для обучения.
        
        Args:
            base_model_path: путь к базовой модели (None для pretrain from scratch)
            stage: "pretrain" | "continual_pretrain" | "sft"
            tokenizer: загруженный токенизатор
            config: конфиг обучения
            trust_remote_code: разрешить выполнение кода
        
        Returns:
            (model, model_config)
        """
        raise NotImplementedError
    
    def init_from_scratch(
        self,
        model_id_or_config: str | Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        trust_remote_code: bool = True,
    ) -> Tuple[PreTrainedModel, Any]:
        """
        Инициализирует модель с нуля (для pretrain).
        
        Args:
            model_id_or_config: model_id или словарь конфига
            tokenizer: токенизатор
            config: конфиг обучения
            trust_remote_code: разрешить выполнение кода
        
        Returns:
            (model, model_config)
        """
        raise NotImplementedError
    
    def prepare_for_training(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
    ) -> PreTrainedModel:
        """
        Подготавливает модель для обучения.
        
        Выполняет:
        - resize_token_embeddings если vocab изменился
        - tie_weights если нужно
        - use_cache = False
        - gradient_checkpointing если включен
        - LoRA/QLoRA если указано в config
        """
        tuning_method = config.get("tuning_method", "full")
        
        # QLoRA: подготовка модели для 4-bit (если модель уже загружена в 4-bit)
        if tuning_method == "qlora" and PEFT_AVAILABLE:
            # Модель должна быть уже загружена в 4-bit через quantization_config в load_for_training()
            # Проверяем несколькими способами, что модель действительно в 4-bit
            is_4bit = (
                getattr(model, "is_loaded_in_4bit", False) or
                getattr(model, "is_quantized", False) or
                getattr(model.config, "quantization_config", None) is not None
            )
            
            if is_4bit:
                logger.info("Preparing 4-bit model for QLoRA training...")
                model = prepare_model_for_kbit_training(model)
                logger.info("Model prepared for 4-bit training")
            else:
                logger.warning(
                    "QLoRA requires 4-bit quantization, but model was not loaded in 4-bit. "
                    "This may cause issues. Ensure quantization_config is applied during model loading. "
                    f"Model attributes: is_loaded_in_4bit={getattr(model, 'is_loaded_in_4bit', 'N/A')}, "
                    f"is_quantized={getattr(model, 'is_quantized', 'N/A')}, "
                    f"quantization_config={getattr(model.config, 'quantization_config', 'N/A')}"
                )
        
        # Resize embeddings если vocab изменился (до LoRA)
        if hasattr(model.config, 'vocab_size') and model.config.vocab_size != len(tokenizer):
            logger.info(f"Resizing token embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
            model.config.vocab_size = len(tokenizer)
            if hasattr(model, "tie_weights"):
                model.tie_weights()
        
        # LoRA/QLoRA: применяем PEFT
        if tuning_method in ("lora", "qlora") and PEFT_AVAILABLE:
            target_modules = config.get("lora_target_modules")
            if not target_modules:
                # Автодетект target_modules по типу модели
                target_modules = self._detect_target_modules(model)
            
            lora_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                target_modules=target_modules,
                lora_dropout=config.get("lora_dropout", 0.1),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, lora_config)
            logger.info(f"Applied LoRA with r={lora_config.r}, target_modules={target_modules}")
        
        # Отключаем cache для обучения
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
            logger.info("Model use_cache set to False for training")
        
        # Gradient checkpointing
        if config.get("grad_checkpoint", False):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        
        return model
    
    def _detect_target_modules(self, model: PreTrainedModel) -> list[str]:
        """
        Автоматически определяет target_modules для LoRA на основе типа модели.
        
        Returns:
            Список имён модулей для LoRA
        """
        model_type = getattr(model.config, "model_type", "").lower()
        
        # Стандартные паттерны для популярных архитектур
        if model_type in ("llama", "mistral", "mixtral", "qwen"):
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif model_type in ("gpt2", "gpt_neox", "gptj"):
            return ["c_attn", "c_proj", "c_fc"]
        elif model_type == "opt":
            return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        elif model_type == "bloom":
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif model_type == "homellm":
            # Для Home модели
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # Fallback: ищем все linear слои
            logger.warning(f"Unknown model_type {model_type}, using fallback target_modules detection")
            target_modules = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and "embed" not in name.lower():
                    # Берем последнюю часть имени (например "q_proj" из "model.layers.0.q_proj")
                    target_modules.append(name.split(".")[-1])
            
            # Убираем дубликаты и возвращаем уникальные имена
            return list(set(target_modules))[:8]  # Ограничиваем до 8 модулей
    
    def save_final(
        self,
        accelerator,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: Path,
    ):
        """
        Сохраняет финальную модель в HF формате.
        
        ВАЖНО: Если модель использует LoRA/QLoRA, мерджим адаптер в базу,
        чтобы чат мог загрузить модель как обычную.
        
        Args:
            accelerator: Accelerator instance
            model: модель (может быть wrapped)
            tokenizer: токенизатор
            output_dir: директория для сохранения
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Unwrap модель если нужно
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Если это PEFT-модель (LoRA/QLoRA) — мерджим адаптер в базу
        if PEFT_AVAILABLE and PeftModel is not None:
            try:
                if isinstance(unwrapped_model, PeftModel):
                    logger.info("Merging LoRA adapter into base model for final save...")
                    unwrapped_model = unwrapped_model.merge_and_unload()
                    logger.info("LoRA adapter merged successfully")
            except Exception as e:
                logger.warning(f"LoRA merge failed, saving as-is: {e}")
        
        # Сохраняем в HF формате
        unwrapped_model.save_pretrained(
            str(output_dir),
            safe_serialization=True,
            save_function=accelerator.save,
        )
        
        # Сохраняем токенизатор
        tokenizer.save_pretrained(str(output_dir))
        
        logger.info(f"Model and tokenizer saved to {output_dir}")


class HomeAdapter(ModelAdapter):
    """Адаптер для Home моделей."""
    
    def load_tokenizer(
        self,
        source: str | Path,
        trust_remote_code: bool = True,
    ) -> PreTrainedTokenizer:
        """Загружает токенизатор для Home модели."""
        source = Path(source) if isinstance(source, str) else source
        
        # Пробуем загрузить из модели, иначе fallback на gpt2
        if source.exists() and (source / "tokenizer.json").exists():
            try:
                return AutoTokenizer.from_pretrained(str(source), trust_remote_code=trust_remote_code)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from {source}: {e}")
        
        # Fallback
        return AutoTokenizer.from_pretrained("gpt2", trust_remote_code=trust_remote_code)
    
    def load_for_training(
        self,
        base_model_path: Optional[str | Path],
        stage: str,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        trust_remote_code: bool = True,
    ) -> Tuple[PreTrainedModel, Any]:
        """Загружает Home модель для обучения."""
        # QLoRA не поддерживается для Home моделей (bitsandbytes не применим)
        tuning_method = config.get("tuning_method", "full")
        if tuning_method == "qlora":
            logger.warning("QLoRA is not supported for Home models. Falling back to LoRA.")
            config["tuning_method"] = "lora"  # Понижаем до LoRA
        
        base_model_path = Path(base_model_path) if base_model_path else None
        
        # Проверяем, является ли это accelerate checkpoint (для resume)
        # ВАЖНО: pytorch_model.bin.index.json может быть и у обычных шардированных HF-сейвов,
        # поэтому проверяем наличие accelerator_state.json - это точный признак accelerate checkpoint
        def is_accelerate_checkpoint(p: Path) -> bool:
            """Проверяет, является ли путь accelerate checkpoint'ом."""
            return (p / "accelerator_state.json").exists()
        
        is_checkpoint = False
        if base_model_path:
            is_checkpoint = is_accelerate_checkpoint(base_model_path)
        
        if is_checkpoint and stage == "continual_pretrain":
            # Для resume - загружаем конфиг, модель создадим позже
            if (base_model_path / "config.json").exists():
                model_config = HomeConfig.from_pretrained(str(base_model_path))
            else:
                parent_config = base_model_path.parent / "run_config.json"
                if parent_config.exists():
                    with open(parent_config) as f:
                        run_cfg = json.load(f)
                    model_config = HomeConfig(
                        vocab_size=len(tokenizer),
                        hidden_size=run_cfg.get("hidden_size", 512),
                        num_hidden_layers=run_cfg.get("num_layers", 8),
                        num_attention_heads=run_cfg.get("n_heads", 8),
                        max_position_embeddings=run_cfg.get("seq_len", 512),
                    )
                else:
                    raise ValueError(f"Cannot find config.json in {base_model_path}")
            
            model = HomeForCausalLM(model_config)
            logger.info(f"Home model initialized for resume from accelerate checkpoint")
            logger.warning(
                "Model initialized with random weights. "
                "Weights will be loaded from checkpoint via accelerator.load_state(). "
                "If resume fails, training will continue with random weights (this is likely incorrect)."
            )
            return model, model_config
        
        elif base_model_path:
            # Загружаем из final_model
            if not (base_model_path / "config.json").exists():
                raise ValueError(
                    f"Base model path {base_model_path} does not contain config.json. "
                    f"For continual_pretrain, please use a final_model directory."
                )
            
            try:
                logger.info(f"Loading Home model from {base_model_path} using from_pretrained...")
                model = HomeForCausalLM.from_pretrained(
                    str(base_model_path),
                    torch_dtype=torch.float32,
                    trust_remote_code=trust_remote_code,
                )
                model_config = model.config
                logger.info(f"✅ Successfully loaded Home model from {base_model_path}")
                return model, model_config
            except Exception as e:
                logger.error(f"Failed to load Home model using from_pretrained: {e}")
                # Fallback на ручную загрузку
                logger.warning("Falling back to manual weight loading...")
                model_config = HomeConfig.from_pretrained(str(base_model_path))
                model = HomeForCausalLM(model_config)
                
                from safetensors.torch import load_file
                if (base_model_path / "model.safetensors").exists():
                    state_dict = load_file(str(base_model_path / "model.safetensors"))
                    model.load_state_dict(state_dict, strict=False)
                elif (base_model_path / "pytorch_model.bin").exists():
                    state_dict = torch.load(base_model_path / "pytorch_model.bin", map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                else:
                    raise ValueError(f"No weights found in {base_model_path}")
                
                logger.info(f"Loaded Home model from {base_model_path} (fallback method)")
                return model, model_config
        
        else:
            # Pretrain from scratch
            return self.init_from_scratch(None, tokenizer, config, trust_remote_code)
    
    def init_from_scratch(
        self,
        model_id_or_config: str | Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        trust_remote_code: bool = True,
    ) -> Tuple[PreTrainedModel, Any]:
        """Инициализирует Home модель с нуля."""
        model_config = HomeConfig(
            vocab_size=len(tokenizer),
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["n_heads"],
            max_position_embeddings=config["seq_len"],
            dropout=config.get("dropout", 0.0),
        )
        model = HomeForCausalLM(model_config)
        return model, model_config


class HFAdapter(ModelAdapter):
    """Адаптер для HuggingFace моделей."""
    
    def load_tokenizer(
        self,
        source: str | Path,
        trust_remote_code: bool = True,
    ) -> PreTrainedTokenizer:
        """Загружает токенизатор для HF модели."""
        try:
            return AutoTokenizer.from_pretrained(
                str(source),
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {source}: {e}")
            # Fallback на gpt2
            return AutoTokenizer.from_pretrained("gpt2", trust_remote_code=trust_remote_code)
    
    def load_for_training(
        self,
        base_model_path: Optional[str | Path],
        stage: str,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        trust_remote_code: bool = True,
    ) -> Tuple[PreTrainedModel, Any]:
        """Загружает HF модель для обучения."""
        # Определяем параметры для QLoRA и mixed precision
        tuning_method = config.get("tuning_method", "full")
        mp = config.get("mixed_precision", "no")
        
        # Определяем torch_dtype
        if mp == "bf16":
            torch_dtype = torch.bfloat16
        elif mp == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        quantization_config = None
        device_map = None
        
        if tuning_method == "qlora":
            if not PEFT_AVAILABLE:
                raise ValueError("QLoRA выбран, но peft/bitsandbytes не установлены.")
            
            # QLoRA требует 4-bit quantization при загрузке
            # Для multi-GPU через accelerate нужен device_map
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                device_map = {"": local_rank}
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype if torch_dtype != torch.float32 else torch.float16,
            )
            
            logger.info(f"QLoRA: loading model in 4-bit with compute_dtype={quantization_config.bnb_4bit_compute_dtype}")
        
        if base_model_path:
            base_model_path = Path(base_model_path)
            
            if not (base_model_path / "config.json").exists():
                raise ValueError(
                    f"Base model path {base_model_path} does not contain config.json. "
                    f"For HF models, please use a model directory with config.json."
                )
            
            try:
                logger.info(f"Loading HF model from {base_model_path} using from_pretrained...")
                model = AutoModelForCausalLM.from_pretrained(
                    str(base_model_path),
                    torch_dtype=torch_dtype if tuning_method != "qlora" else None,  # dtype в qlora контролируется bnb
                    trust_remote_code=trust_remote_code,
                    quantization_config=quantization_config,
                    device_map=device_map,
                )
                model_config = model.config
                logger.info(f"✅ Successfully loaded HF model from {base_model_path}")
                return model, model_config
            except Exception as e:
                logger.error(f"Failed to load HF model: {e}")
                raise
        
        else:
            # Pretrain from scratch для HF моделей
            return self.init_from_scratch(None, tokenizer, config, trust_remote_code)
    
    def init_from_scratch(
        self,
        model_id_or_config: str | Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        trust_remote_code: bool = True,
    ) -> Tuple[PreTrainedModel, Any]:
        """
        Инициализирует HF модель с нуля.
        
        Для HF моделей это возможно только если указан model_id,
        из которого можно загрузить конфиг.
        
        ВАЖНО: QLoRA не поддерживается для pretrain from scratch (нужна базовая модель).
        """
        model_id = config.get("model_id")
        if not model_id:
            raise ValueError(
                "For HF models, pretrain from scratch requires 'model_id' in config. "
                "Please specify a HuggingFace model ID (e.g., 'gpt2', 'microsoft/DialoGPT-small')."
            )
        
        tuning_method = config.get("tuning_method", "full")
        if tuning_method == "qlora":
            raise ValueError(
                "QLoRA is not supported for pretrain from scratch. "
                "Please use a base model for QLoRA training, or use LoRA/full fine-tuning."
            )
        
        try:
            logger.info(f"Initializing HF model from scratch using {model_id}...")
            model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            
            # Обновляем vocab_size если нужно
            if model_config.vocab_size != len(tokenizer):
                logger.info(f"Updating vocab_size: {model_config.vocab_size} -> {len(tokenizer)}")
                model_config.vocab_size = len(tokenizer)
            
            # Определяем torch_dtype
            mp = config.get("mixed_precision", "no")
            if mp in ("fp16", "bf16"):
                torch_dtype = torch.float16 if mp == "fp16" else torch.bfloat16
            else:
                torch_dtype = torch.float32
            
            # Создаём модель из конфига
            model = AutoModelForCausalLM.from_config(
                model_config,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
            )
            
            logger.info(f"✅ Initialized HF model from {model_id}")
            return model, model_config
        
        except Exception as e:
            logger.error(f"Failed to initialize HF model from {model_id}: {e}")
            raise ValueError(
                f"Cannot initialize HF model from {model_id}. "
                f"Some models require trust_remote_code=True or cannot be initialized from config. "
                f"Error: {e}"
            )

