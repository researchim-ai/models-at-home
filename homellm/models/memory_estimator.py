"""
homellm.models.memory_estimator
--------------------------------
Универсальная система расчета памяти для разных архитектур моделей.

Поддерживает точный расчет:
- Параметров модели (weights)
- Градиентов (grads)
- Состояния оптимизатора (optimizer state)
- Активаций (activations)
- Буфера CUDA allocator

Архитектуры:
- HomeModel (SwiGLU, RMSNorm, RoPE)
- Llama (SwiGLU, RMSNorm, RoPE, pre-norm)
- Mistral (SwiGLU, RMSNorm, RoPE, sliding window attention)
- GPT-2 (GELU, LayerNorm, learned pos embeddings)
- Blueprint (custom architectures)
"""
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ArchitectureProfile:
    """Профиль архитектуры для расчета памяти."""
    name: str
    # Параметры на слой
    attn_proj_count: int  # Количество проекций в attention (q,k,v,out = 4)
    attn_has_bias: bool  # Есть ли bias в attention проекциях
    mlp_type: str  # "swiglu" | "gelu" | "relu"
    mlp_has_bias: bool  # Есть ли bias в MLP
    norm_type: str  # "rmsnorm" | "layernorm"
    norm_count_per_layer: int  # Количество норм на слой (обычно 2: attn_norm + ffn_norm)
    has_final_norm: bool  # Есть ли финальная нормализация
    lm_head_tied: bool  # Связан ли lm_head с embed_tokens (weight tying)
    has_pos_embeddings: bool  # Есть ли обучаемые позиционные эмбеддинги (не RoPE)
    
    def calculate_parameters(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        intermediate_size: int,
        max_position_embeddings: int = 4096,
    ) -> int:
        """
        Точный расчет количества параметров для данной архитектуры.
        
        Returns:
            Общее количество параметров
        """
        h = int(hidden_size)
        l = int(num_layers)
        v = int(vocab_size)
        i = int(intermediate_size)
        s = int(max_position_embeddings)
        
        # Embedding
        embed_params = v * h
        
        # Позиционные эмбеддинги (если есть, не RoPE)
        pos_embed_params = s * h if self.has_pos_embeddings else 0
        
        # Attention на слой
        attn_bias = h if self.attn_has_bias else 0
        attn_params_per_layer = self.attn_proj_count * (h * h + attn_bias)
        
        # MLP на слой
        if self.mlp_type == "swiglu":
            # SwiGLU: w1(H->I), w2(I->H), w3(H->I)
            mlp_bias = (i + h + i) if self.mlp_has_bias else 0
            mlp_params_per_layer = 3 * h * i + mlp_bias
        elif self.mlp_type in ("gelu", "relu"):
            # Стандартный FFN: gate(H->I), up(H->I) для некоторых, или просто fc1(H->I), fc2(I->H)
            # Для GPT-2: fc1(H->4H), fc2(4H->H)
            mlp_bias = (i + h) if self.mlp_has_bias else 0
            mlp_params_per_layer = 2 * h * i + mlp_bias
        else:
            # Fallback: стандартный FFN
            mlp_bias = (i + h) if self.mlp_has_bias else 0
            mlp_params_per_layer = 2 * h * i + mlp_bias
        
        # Нормализации на слой
        if self.norm_type == "rmsnorm":
            # RMSNorm: только weight (scale), нет bias
            norm_params_per_layer = self.norm_count_per_layer * h
        else:  # layernorm
            # LayerNorm: weight + bias
            norm_params_per_layer = self.norm_count_per_layer * (h + h)
        
        # Финальная нормализация
        if self.has_final_norm:
            if self.norm_type == "rmsnorm":
                final_norm_params = h
            else:
                final_norm_params = h + h  # weight + bias
        else:
            final_norm_params = 0
        
        # LM Head
        if self.lm_head_tied:
            # Weight tying: параметры не удваиваются
            lm_head_params = 0
        else:
            lm_head_params = v * h
        
        total = (
            embed_params +
            pos_embed_params +
            l * (attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer) +
            final_norm_params +
            lm_head_params
        )
        
        return int(total)
    
    def calculate_activations(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        intermediate_size: int,
        num_heads: int,
        vocab_size: int,
        grad_checkpoint: bool = False,
        attention_mode: str = "flash",  # "flash" | "math"
        dtype_bytes: int = 2,  # 2 для bf16/fp16, 4 для fp32
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Расчет активаций для forward+backward pass.
        
        Args:
            attention_mode: "flash" (SDPA, не материализует S×S) или "math" (материализует)
            dtype_bytes: размер dtype в байтах (2 для bf16/fp16, 4 для fp32)
        
        Returns:
            (total_act_bytes, detail_dict)
        """
        b = int(batch_size)
        s = int(seq_len)
        h = int(hidden_size)
        l = int(num_layers)
        i = int(intermediate_size)
        nh = max(1, int(num_heads))
        v = int(vocab_size)
        
        # Базовые тензоры (всегда нужны)
        embed_act = b * s * h  # embeddings
        logits_act = b * s * v  # logits для loss
        
        # Активации на слой
        if grad_checkpoint:
            # Gradient checkpointing: сохраняем только input для одного слоя
            # В backward пересчитываем промежуточные активации
            per_layer_saved = b * s * h  # только input
        else:
            # Без checkpointing: сохраняем все промежуточные активации для backward
            # Attention: q, k, v, attn_out (нужны для backward)
            attn_act_elems = b * s * (3 * h + h)  # q, k, v, out
            
            # MLP активации
            # В реальности PyTorch сохраняет только необходимые для backward активации
            # Для упрощения считаем основные промежуточные результаты
            if self.mlp_type == "swiglu":
                # SwiGLU: w1_out (H->I), w3_out (H->I), gate = swish(w1*w3) (I), w2_out (I->H)
                # Для backward нужны: w1_out, w3_out, gate, w2_out, mlp_out
                # Более реалистичный расчет: основные промежуточные результаты
                mlp_act_elems = b * s * (i + i + i + h + h)  # w1, w3, gate, w2, mlp_out
            else:
                # Стандартный FFN: fc1_out, fc2_out
                mlp_act_elems = b * s * (i + h)
            
            # Residuals: после attn и после mlp (нужны для backward)
            residual_act_elems = b * s * (2 * h)  # после attn, после mlp
            
            # Нормы: обычно не сохраняются отдельно (пересчитываются), но для точности можно учесть
            # В реальности они пересчитываются, но для консервативной оценки можно учесть
            per_layer_saved = attn_act_elems + mlp_act_elems + residual_act_elems
        
        # Attention S×S матрицы (только для math mode)
        if attention_mode == "math":
            # scores (B, heads, S, S) в fp32 для softmax
            # probs (B, heads, S, S) в fp32 после softmax
            attn_ss_elems = b * nh * s * s
            attn_ss_bytes = int(attn_ss_elems * 8)  # scores fp32 (4) + probs fp32 (4)
        else:
            # Flash attention: не материализует S×S
            attn_ss_bytes = 0
        
        # Итоговый расчет
        if grad_checkpoint:
            # Только один слой активаций + embed + logits
            base_act_bytes = int((embed_act + per_layer_saved + logits_act) * dtype_bytes)
            # Attention S×S только для одного слоя (если math)
            attn_ss_total = attn_ss_bytes if attention_mode == "math" else 0
            total_act_bytes = base_act_bytes + attn_ss_total
        else:
            # Все слои одновременно
            base_act_bytes = int((embed_act + l * per_layer_saved + logits_act) * dtype_bytes)
            attn_ss_total = int(l * attn_ss_bytes) if attention_mode == "math" else 0
            total_act_bytes = base_act_bytes + attn_ss_total
        
        detail = {
            "embed_act_gb": round((embed_act * dtype_bytes) / (1024**3), 3),
            "logits_act_gb": round((logits_act * dtype_bytes) / (1024**3), 3),
            "per_layer_act_gb": round((per_layer_saved * dtype_bytes) / (1024**3), 3),
            "attn_ss_gb": round(attn_ss_total / (1024**3), 3) if attn_ss_total > 0 else 0,
            "grad_checkpoint": grad_checkpoint,
            "attention_mode": attention_mode,
        }
        
        return int(total_act_bytes), detail


# Архитектурные профили
ARCHITECTURE_PROFILES = {
    "home": ArchitectureProfile(
        name="HomeModel",
        attn_proj_count=4,  # q, k, v, out
        attn_has_bias=False,
        mlp_type="swiglu",
        mlp_has_bias=False,
        norm_type="rmsnorm",
        norm_count_per_layer=2,  # attn_norm, ffn_norm
        has_final_norm=True,
        lm_head_tied=True,
        has_pos_embeddings=False,  # RoPE
    ),
    "llama": ArchitectureProfile(
        name="Llama",
        attn_proj_count=4,  # q, k, v, out
        attn_has_bias=False,
        mlp_type="swiglu",
        mlp_has_bias=False,
        norm_type="rmsnorm",
        norm_count_per_layer=2,  # attn_norm, ffn_norm
        has_final_norm=True,
        lm_head_tied=False,  # Llama не использует weight tying
        has_pos_embeddings=False,  # RoPE
    ),
    "mistral": ArchitectureProfile(
        name="Mistral",
        attn_proj_count=4,  # q, k, v, out
        attn_has_bias=False,
        mlp_type="swiglu",
        mlp_has_bias=False,
        norm_type="rmsnorm",
        norm_count_per_layer=2,
        has_final_norm=True,
        lm_head_tied=False,
        has_pos_embeddings=False,  # RoPE
    ),
    "gpt2": ArchitectureProfile(
        name="GPT-2",
        attn_proj_count=3,  # c_attn (qkv fused), c_proj (out)
        attn_has_bias=True,
        mlp_type="gelu",
        mlp_has_bias=True,
        norm_type="layernorm",
        norm_count_per_layer=2,  # ln_1, ln_2
        has_final_norm=True,
        lm_head_tied=True,
        has_pos_embeddings=True,  # Learned positional embeddings
    ),
    "gpt_neox": ArchitectureProfile(
        name="GPT-NeoX",
        attn_proj_count=4,  # q, k, v, out
        attn_has_bias=False,
        mlp_type="gelu",
        mlp_has_bias=False,
        norm_type="layernorm",
        norm_count_per_layer=2,
        has_final_norm=True,
        lm_head_tied=False,
        has_pos_embeddings=True,  # Learned positional embeddings
    ),
}


def get_architecture_profile(
    model_type: str,
    arch_preset: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> ArchitectureProfile:
    """
    Получает профиль архитектуры по типу модели.
    
    Args:
        model_type: "home" | "hf" | "blueprint"
        arch_preset: "llama" | "mistral" | "gpt2" | None (из UI)
        model_config: словарь config.json модели (для определения архитектуры HF моделей)
    
    Returns:
        ArchitectureProfile
    """
    if model_type == "home":
        return ARCHITECTURE_PROFILES["home"]
    
    if model_type == "blueprint":
        # Для blueprint используем HomeModel как базовый (можно улучшить позже)
        return ARCHITECTURE_PROFILES["home"]
    
    # Для HF моделей определяем архитектуру
    if model_type == "hf":
        # Сначала пробуем arch_preset из UI
        if arch_preset:
            arch_lower = arch_preset.lower()
            if arch_lower in ARCHITECTURE_PROFILES:
                return ARCHITECTURE_PROFILES[arch_lower]
        
        # Потом пробуем определить по model_config
        if model_config:
            hf_model_type = model_config.get("model_type", "").lower()
            if hf_model_type in ARCHITECTURE_PROFILES:
                return ARCHITECTURE_PROFILES[hf_model_type]
            # Маппинг некоторых типов
            if hf_model_type in ("gpt_neox", "pythia"):
                return ARCHITECTURE_PROFILES["gpt_neox"]
            if hf_model_type in ("llama", "llama2", "llama3"):
                return ARCHITECTURE_PROFILES["llama"]
            if hf_model_type == "mistral":
                return ARCHITECTURE_PROFILES["mistral"]
            if hf_model_type in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"):
                return ARCHITECTURE_PROFILES["gpt2"]
    
    # По умолчанию используем HomeModel
    return ARCHITECTURE_PROFILES["home"]


def estimate_grpo_memory_footprint(
    config: Dict[str, Any],
    distributed_mode: str = "default",
    num_gpus: int = 1,
) -> Dict[str, Any]:
    """
    Специализированный расчет памяти для GRPO (RL) обучения.
    Учитывает специфику: генерация (KV cache) -> обучение (Forward/Backward).
    """
    try:
        # 1. Извлекаем параметры модели
        hidden_size = int(config["hidden_size"])
        num_layers = int(config["num_layers"])
        n_heads = int(config.get("n_heads", 0) or 0)
        vocab_size = int(config.get("vocab_size", 50257))
        intermediate_size = int(config.get("intermediate_size") or (hidden_size * 4))
        
        # 2. Параметры GRPO
        group_size = int(config.get("grpo_group_size", 8))
        train_batch_size = int(config.get("grpo_train_batch_size", 2)) # Используем train_batch_size
        max_prompt_length = int(config.get("max_prompt_length", 512))
        max_new_tokens = int(config.get("grpo_max_new_tokens", 1024))
        total_seq_len = max_prompt_length + max_new_tokens
        
        use_lora = bool(config.get("use_lora", False))
        use_4bit = bool(config.get("use_4bit", False))
        use_8bit = bool(config.get("use_8bit", False))
        kl_weight = float(config.get("grpo_kl_weight", 0.0))
        quantize_ref = bool(config.get("quantize_reference_model", False))
        grad_checkpoint = True # Для GRPO/LoRA почти всегда включаем
        
        # Профиль
        model_type = config.get("model_type", "home")
        arch_preset = config.get("arch_preset")
        profile = get_architecture_profile(model_type, arch_preset)
        
        # 3. Расчет параметров (в FP16 эквиваленте для оценки)
        total_params = profile.calculate_parameters(
            vocab_size, hidden_size, num_layers, intermediate_size, total_seq_len
        )
        
        # --- ФАЗА 1: СТАТИЧЕСКАЯ ПАМЯТЬ (Модели) ---
        
        # Базовая модель
        if use_4bit:
            model_bytes = total_params * 0.5  # 4 bit = 0.5 byte
        elif use_8bit:
            model_bytes = total_params * 1.0  # 8 bit = 1 byte
        else:
            model_bytes = total_params * 2.0  # bf16/fp16 = 2 bytes
            
        # LoRA адаптеры (примерно 1-5% от весов, берем 2%)
        lora_bytes = (total_params * 2.0) * 0.02 if use_lora else 0
        
        # Reference модель (если нужна)
        ref_model_bytes = 0
        if kl_weight > 0:
            if quantize_ref:
                 # Если квантуем референс (обычно 4/8 bit)
                 ref_model_bytes = total_params * (0.5 if use_4bit else 1.0)
            else:
                 # Обычно fp16
                 ref_model_bytes = total_params * 2.0
        
        # Оптимизатор и Градиенты
        # При LoRA обучаем только адаптеры (~2% параметров)
        trainable_params = total_params if not use_lora else (total_params * 0.02)
        
        grads_bytes = trainable_params * 2.0 # Градиенты в fp16/bf16
        
        # Optimizer States (AdamW = 2 states per param)
        # Если используем 8-bit optimizer (по умолчанию в скрипте для LoRA), то 2 байта на state
        # Иначе 4 (fp32) или 2 (fp16)
        # Считаем консервативно: 8 байт на параметр (2 состояния по 4 байта) или 2 байта (8-bit opt)
        # В нашем скрипте trainer.py используется bnb.optim.AdamW8bit если доступно
        optimizer_bytes = trainable_params * 2.0 # 8-bit optimizer assumption for LoRA
        if not use_lora:
            optimizer_bytes = trainable_params * 8.0 # Full finetune (fp32 optimizer states)
            
        static_mem_bytes = model_bytes + lora_bytes + ref_model_bytes + grads_bytes + optimizer_bytes
        
        # --- ФАЗА 2: ГЕНЕРАЦИЯ (KV Cache) ---
        # KV Cache: 2 * layers * heads * head_dim * seq_len * batch
        # batch = group_size (параллельная генерация)
        head_dim = hidden_size // n_heads if n_heads > 0 else 64
        kv_cache_per_token = 2 * num_layers * n_heads * head_dim * 2 # 2 bytes (fp16)
        
        # KV Cache растет до total_seq_len
        kv_cache_bytes = kv_cache_per_token * total_seq_len * group_size
        
        # Activations при генерации (минимальны, 1 токен)
        gen_act_bytes = group_size * hidden_size * 2 * num_layers # Очень грубо, но мало
        
        peak_gen_bytes = static_mem_bytes + kv_cache_bytes + gen_act_bytes
        
        # --- ФАЗА 3: ОБУЧЕНИЕ (Forward/Backward) ---
        # Здесь batch = train_batch_size (микро-батч)
        # Полная длина последовательности (prompt + completion)
        
        # Активации (с grad checkpointing)
        # Используем существующую функцию
        train_act_bytes, _ = profile.calculate_activations(
            batch_size=train_batch_size,
            seq_len=total_seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            num_heads=n_heads,
            vocab_size=vocab_size,
            grad_checkpoint=grad_checkpoint,
            attention_mode="flash", # GRPO обычно с FlashAttn
            dtype_bytes=2
        )
        
        peak_train_bytes = static_mem_bytes + train_act_bytes
        
        # Выбираем максимум
        peak_bytes = max(peak_gen_bytes, peak_train_bytes)
        
        # --- ФАЗА 4: ПЕРЕВОД В GB И БУФЕР ---
        
        static_gb = static_mem_bytes / (1024**3)
        peak_gb = peak_bytes / (1024**3)
        kv_gb = kv_cache_bytes / (1024**3)
        train_act_gb = train_act_bytes / (1024**3)
        
        # Buffer (overhead)
        fixed_overhead = 4.0 # Базовый оверхед PyTorch/CUDA
        variable_coeff = 0.1 # Фрагментация
        
        buffer_gb = fixed_overhead + (peak_gb * variable_coeff)
        
        total_gb = peak_gb + buffer_gb
        
        detail = {
            "mode": "GRPO (RL)",
            "weights_gb": round(static_gb, 2),
            "kv_cache_gb": round(kv_gb, 2),
            "train_act_gb": round(train_act_gb, 2),
            "peak_phase": "Generation" if peak_gen_bytes > peak_train_bytes else "Training",
            "train_batch": train_batch_size,
            "group_size": group_size,
            "seq_len": total_seq_len
        }
        
        notes = (
            f"Оценка для GRPO. Пик потребления: {detail['peak_phase']}. "
            f"KV Cache: {kv_gb:.1f}GB (G={group_size}). "
            f"Train Act: {train_act_gb:.1f}GB (B={train_batch_size})."
        )
        
        return {
            "method": "estimate_grpo",
            "total_gb": round(total_gb, 2),
            "model_gb": round(static_gb, 2),
            "act_gb": round(max(kv_gb, train_act_gb), 2), # Показываем наибольший динамический компонент
            "buf_gb": round(buffer_gb, 2),
            "params": total_params,
            "detail": detail,
            "notes": notes,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "method": "error",
            "total_gb": 0,
            "notes": f"Ошибка расчета GRPO: {e}"
        }


def estimate_memory_footprint(
    config: Dict[str, Any],
    batch_size: int,
    distributed_mode: str = "default",
    num_gpus: int = 1,
) -> Dict[str, Any]:
    """
    Универсальная функция расчета памяти для любой архитектуры.
    
    Args:
        config: конфиг модели и обучения
        batch_size: размер батча на устройство
        distributed_mode: режим распараллеливания
        num_gpus: количество GPU
    
    Returns:
        Словарь с оценкой памяти
    """
    # Если это GRPO режим, переключаемся на спец функцию
    if config.get("stage") == "grpo":
        return estimate_grpo_memory_footprint(config, distributed_mode, num_gpus)

    try:
        # Извлекаем параметры модели
        hidden_size = int(config["hidden_size"])
        num_layers = int(config["num_layers"])
        n_heads = int(config.get("n_heads", 0) or 0)
        seq_len = int(config["seq_len"])
        vocab_size = int(config.get("vocab_size", 50257))
        intermediate_size = int(config.get("intermediate_size") or (hidden_size * 4))
        max_position_embeddings = int(config.get("max_position_embeddings", seq_len))
        
        # Параметры обучения
        mp = (config.get("mixed_precision") or "no").lower()
        opt = (config.get("optimizer") or "adamw").lower()
        grad_checkpoint = bool(config.get("grad_checkpoint", False))
        attention_mode = str(config.get("attention_estimate_mode", "flash")).lower()
        
        # Определяем архитектуру
        model_type = config.get("model_type", "home")
        arch_preset = config.get("arch_preset")
        # Пробуем загрузить model_config для HF моделей (если есть base_model_path)
        model_config_dict = None
        if model_type == "hf" and config.get("base_model_path"):
            try:
                import json
                from pathlib import Path
                base_path = Path(config["base_model_path"])
                config_path = base_path / "config.json" if base_path.is_dir() else base_path.parent / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        model_config_dict = json.load(f)
            except Exception:
                pass  # Игнорируем ошибки загрузки config
        profile = get_architecture_profile(model_type, arch_preset, model_config_dict)
        
        # ---- 1) Расчет параметров ----
        total_params = profile.calculate_parameters(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
        )
        
        # ---- 2) Static memory: weights/grads/optimizer ----
        param_bytes = 2 if mp in ("fp16", "bf16") else 4
        grad_bytes = param_bytes  # Градиенты обычно в том же dtype
        
        # Optimizer state
        opt_state_dtype = (config.get("optimizer_state_dtype") or ("fp32" if opt in ("adam", "adamw") else mp)).lower()
        opt_state_bytes = 2 if opt_state_dtype in ("fp16", "bf16") else 4
        
        weights_bytes = total_params * param_bytes
        grads_bytes = total_params * grad_bytes
        
        # Optimizer state зависит от типа
        if opt in ("adam", "adamw"):
            # exp_avg + exp_avg_sq
            optim_bytes = total_params * (2 * opt_state_bytes)
        elif opt == "sgd":
            momentum = float(config.get("momentum") or 0.0)
            optim_bytes = total_params * (opt_state_bytes if momentum > 0 else 0)
        else:
            # Другие оптимизаторы - по умолчанию 2 state
            optim_bytes = total_params * (2 * opt_state_bytes)
        
        # Master weights для mixed precision (bf16/fp16)
        # При использовании mixed precision некоторые оптимизаторы (особенно AdamW)
        # хранят master weights в fp32 для стабильности обновлений
        # Это добавляет еще total_params * 4 bytes
        master_weights_bytes = 0
        if mp in ("fp16", "bf16") and opt in ("adam", "adamw"):
            # AdamW с mixed precision часто использует master weights в fp32
            master_weights_bytes = total_params * 4  # fp32 master weights
        
        # ---- 3) Distributed sharding ----
        shard = max(1, int(num_gpus or 1))
        if "deepspeed_zero3" in distributed_mode or distributed_mode == "fsdp":
            weights_bytes /= shard
            grads_bytes /= shard
            optim_bytes /= shard
        elif "deepspeed_zero2" in distributed_mode:
            # weights реплицируются; grads+optim шардируются
            grads_bytes /= shard
            optim_bytes /= shard
        
        static_mem_bytes = int(weights_bytes + grads_bytes + optim_bytes + master_weights_bytes)
        
        # ---- 4) Activations ----
        act_bytes, act_detail = profile.calculate_activations(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            num_heads=n_heads,
            vocab_size=vocab_size,
            grad_checkpoint=grad_checkpoint,
            attention_mode=attention_mode,
            dtype_bytes=param_bytes,
        )
        
        # ---- 5) Buffer / allocator slack ----
        # CUDA allocator резервирует больше памяти, чем фактически используется
        # Также есть overhead от промежуточных вычислений, временных буферов, и т.д.
        static_gb = static_mem_bytes / (1024**3)
        act_gb = act_bytes / (1024**3)
        
        # Реалистичный buffer: CUDA allocator резервирует память блоками
        # Также есть overhead от промежуточных вычислений, временных буферов, 
        # фрагментации памяти, и т.д.
        # ВАЖНО: На практике реальное использование памяти может быть в 1.5-2.5 раза больше
        # теоретического расчета из-за:
        # - Фрагментации памяти CUDA allocator
        # - Промежуточных буферов для вычислений (matmul, conv, и т.д.)
        # - Временных тензоров в autograd
        # - Overhead от mixed precision (мастер-копии весов в fp32)
        # - Буферов для коммуникации в distributed training
        
        # Более реалистичная оценка: buffer = фиксированный overhead + пропорциональная часть
        # Анализ реальных замеров показывает:
        # - Есть фиксированный overhead от CUDA allocator (~8-9 GB)
        # - Пропорциональная часть зависит от размера модели (~0.2x от base)
        # Это объясняет, почему маленькие модели имеют относительно больший buffer
        
        total_base = static_gb + act_gb
        
        # Формула на основе реальных замеров:
        # buffer = fixed_overhead + variable_coeff * total_base
        # Где fixed_overhead ~ 8-9 GB (минимальный overhead CUDA allocator)
        # И variable_coeff ~ 0.2 (пропорциональная часть)
        
        # Базовые коэффициенты (калиброваны на реальных данных)
        fixed_overhead_gb = 8.5  # Фиксированный overhead CUDA allocator
        variable_coeff = 0.2     # Пропорциональная часть от total_base
        
        # Корректировки для разных размеров:
        # Для очень маленьких моделей fixed overhead может быть больше
        if total_base < 1.0:
            fixed_overhead_gb = 9.0
            variable_coeff = 0.25
        elif total_base < 3.0:
            fixed_overhead_gb = 8.5
            variable_coeff = 0.2
        elif total_base < 10.0:
            fixed_overhead_gb = 8.0
            variable_coeff = 0.18
        else:
            # Для больших моделей fixed overhead относительно меньше
            fixed_overhead_gb = 7.0
            variable_coeff = 0.15
        
        buffer_gb = fixed_overhead_gb + variable_coeff * total_base
        
        # Дополнительный overhead для distributed training (DDP синхронизация)
        ddp_overhead_gb = 0.0
        # DDP overhead нужен для любого multi-GPU режима (default, multi_gpu, DDP)
        if num_gpus > 1 and distributed_mode in ("default", "multi_gpu", "ddp"):
            # DDP создает дополнительные буферы для all-reduce операций
            # Градиенты копируются в буферы для синхронизации между GPU
            # Также может быть overhead от коммуникационных буферов
            # Учитываем также небольшие дополнительные буферы для синхронизации
            ddp_overhead_gb = max(1.3, 0.20 * (static_gb + act_gb))
            buffer_gb += ddp_overhead_gb
        
        total_gb = static_gb + act_gb + buffer_gb
        
        detail = {
            "architecture": profile.name,
            "weights_gb": round(weights_bytes / (1024**3), 3),
            "grads_gb": round(grads_bytes / (1024**3), 3),
            "optim_state_gb": round(optim_bytes / (1024**3), 3),
            "master_weights_gb": round(master_weights_bytes / (1024**3), 3) if master_weights_bytes > 0 else 0,
            **act_detail,
            "buffer_rule": f"{fixed_overhead_gb:.1f}GB (fixed) + {variable_coeff:.2f}*base" + (f" + DDP overhead ({ddp_overhead_gb:.2f}GB)" if ddp_overhead_gb > 0 else ""),
        }
        
        notes = (
            f"Консервативная оценка памяти для {profile.name} архитектуры. "
            "Реальное использование может варьироваться в зависимости от "
            "CUDA allocator, фрагментации памяти, и системных процессов. "
            "Для точного замера используйте профилирование на реальной GPU."
        )
        
        return {
            "method": "estimate",
            "total_gb": round(total_gb, 2),
            "model_gb": round(static_gb, 2),
            "act_gb": round(act_gb, 2),
            "buf_gb": round(buffer_gb, 2),
            "params": total_params,
            "detail": detail,
            "notes": notes,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "method": "estimate",
            "total_gb": 0,
            "model_gb": 0,
            "act_gb": 0,
            "buf_gb": 0,
            "params": 0,
            "notes": f"Ошибка расчета: {e}",
        }
