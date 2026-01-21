"""
vLLM backend для inference в чате.

Преимущества vLLM:
- Значительно быстрее генерация (continuous batching, PagedAttention)
- Поддержка LoRA hot-swap
- Эффективное использование памяти

Недостатки:
- Требует больше VRAM при загрузке (компиляция CUDA графов)
- Не поддерживает все модели (нужна совместимость с vLLM)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def is_vllm_available() -> bool:
    """Проверяет доступность vLLM."""
    try:
        import vllm
        return True
    except ImportError:
        return False


class VLLMChatBackend:
    """vLLM backend для inference в чате."""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
    ):
        """
        Инициализация vLLM backend.
        
        Args:
            model_path: Путь к модели или HuggingFace model ID
            dtype: Тип данных ("float16", "bfloat16", "float32")
            gpu_memory_utilization: Доля VRAM для использования (0.0-1.0)
            max_model_len: Максимальная длина контекста (None = авто)
            trust_remote_code: Разрешить выполнение кода из модели
            enable_lora: Включить поддержку LoRA адаптеров
            max_lora_rank: Максимальный ранг LoRA (должен быть >= lora_r модели)
        """
        from vllm import LLM
        
        self.model_path = model_path
        self.dtype = dtype
        self._lora_path: Optional[str] = None
        
        # Конфигурация vLLM
        llm_kwargs = {
            "model": model_path,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "tensor_parallel_size": 1,  # Single GPU
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": True,  # Отключаем CUDA graphs для совместимости
        }
        
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        
        if enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = 1
            llm_kwargs["max_lora_rank"] = max_lora_rank
        
        logger.info(f"🚀 Загрузка модели в vLLM: {model_path}")
        logger.info(f"   dtype={dtype}, gpu_util={gpu_memory_utilization}, enable_lora={enable_lora}")
        
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        
        logger.info("✅ vLLM модель загружена")
    
    def set_lora(self, lora_path: Optional[str] = None):
        """
        Устанавливает LoRA адаптер для генерации.
        
        Args:
            lora_path: Путь к LoRA адаптеру (None = отключить LoRA)
        """
        self._lora_path = lora_path
        if lora_path:
            logger.info(f"🔧 Установлен LoRA адаптер: {lora_path}")
        else:
            logger.info("🔧 LoRA адаптер отключён")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        """
        Генерация текста.
        
        Args:
            prompt: Входной текст
            max_tokens: Максимальное количество токенов
            temperature: Температура сэмплирования
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling (-1 = отключено)
            stop: Список стоп-строк
            stream: Потоковая генерация (не поддерживается пока)
            
        Returns:
            Сгенерированный текст
        """
        from vllm import SamplingParams
        
        # Фильтруем None значения
        params_dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if top_k > 0:
            params_dict["top_k"] = top_k
        
        if stop:
            params_dict["stop"] = stop
        
        sampling_params = SamplingParams(**params_dict)
        
        # Генерация с или без LoRA
        generate_kwargs = {}
        if self._lora_path:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError:
                from vllm.lora import LoRARequest
            
            lora_request = LoRARequest("chat_lora", 1, self._lora_path)
            generate_kwargs["lora_request"] = lora_request
        
        outputs = self.llm.generate([prompt], sampling_params, **generate_kwargs)
        
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        return ""
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Батчевая генерация текста.
        
        Args:
            prompts: Список входных текстов
            max_tokens: Максимальное количество токенов
            temperature: Температура сэмплирования
            top_p: Top-p sampling
            
        Returns:
            Список сгенерированных текстов
        """
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        generate_kwargs = {}
        if self._lora_path:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError:
                from vllm.lora import LoRARequest
            
            lora_request = LoRARequest("chat_lora", 1, self._lora_path)
            generate_kwargs["lora_request"] = lora_request
        
        outputs = self.llm.generate(prompts, sampling_params, **generate_kwargs)
        
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text)
            else:
                results.append("")
        
        return results
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Применяет chat template токенизатора.
        
        Args:
            messages: Список сообщений [{"role": "user", "content": "..."}, ...]
            add_generation_prompt: Добавить промпт для генерации
            
        Returns:
            Отформатированный текст
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            # Fallback: простая конкатенация
            result = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    result += f"System: {content}\n\n"
                elif role == "user":
                    result += f"User: {content}\n\n"
                elif role == "assistant":
                    result += f"Assistant: {content}\n\n"
            if add_generation_prompt:
                result += "Assistant: "
            return result
    
    @property
    def has_chat_template(self) -> bool:
        """Проверяет наличие chat template у токенизатора."""
        return hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None


class TransformersChatBackend:
    """Transformers backend для inference (fallback если vLLM недоступен)."""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        """
        Инициализация Transformers backend.
        
        Args:
            model: Загруженная модель (PreTrainedModel)
            tokenizer: Токенизатор
            device: Устройство для inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        """Генерация текста через transformers."""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        # Декодируем только новые токены
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Применяет chat template токенизатора."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            # Fallback
            result = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    result += f"System: {content}\n\n"
                elif role == "user":
                    result += f"User: {content}\n\n"
                elif role == "assistant":
                    result += f"Assistant: {content}\n\n"
            if add_generation_prompt:
                result += "Assistant: "
            return result
    
    @property
    def has_chat_template(self) -> bool:
        """Проверяет наличие chat template у токенизатора."""
        return hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None
