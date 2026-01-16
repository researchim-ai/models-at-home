"""
Rollout engine (отдельная модель для генерации):

- training engine (DDP/ZeRO/FSDP) отвечает за teacher-forcing logprobs + backprop
- rollout engine отвечает только за autoregressive generate()

Важное для ZeRO-3/FSDP:
autoregressive generation внутри sharded training engine может быть на порядки медленнее.

Два режима работы:
1. LoRA fine-tuning: синхронизируем только LoRA адаптеры (быстро, ~MB)
2. Full fine-tuning: синхронизируем все веса (дорого, ~GB, делаем периодически)
"""

from __future__ import annotations

import json
import logging
import os
import select
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

logger = logging.getLogger(__name__)




def _is_zero3_model(model: Any, accelerator: Any) -> bool:
    """Проверяет, обёрнута ли модель в DeepSpeed ZeRO-3."""
    if accelerator is None:
        return False
    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if ds_plugin is None:
        return False
    return getattr(ds_plugin, "zero_stage", 0) == 3


def _gather_full_state_dict_zero3(
    model: PreTrainedModel,
    accelerator: Any,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Собирает полный state_dict для ZeRO-3 sharded модели.
    Возвращает state_dict только на rank 0, на остальных — None.
    """
    try:
        import deepspeed
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
    except ImportError:
        logger.warning("DeepSpeed не установлен, не могу собрать ZeRO-3 веса")
        return None

    # Для ZeRO-3: собираем все параметры на rank 0
    params = list(model.parameters())
    
    with GatheredParameters(params, modifier_rank=0):
        if accelerator.is_main_process:
            # На rank 0 теперь все параметры доступны полностью
            state_dict = model.state_dict()
            # Копируем на CPU чтобы освободить GPU
            state_dict = {k: v.cpu().clone() for k, v in state_dict.items()}
            return state_dict
    
    return None


def _gather_lora_state_dict(
    model: PreTrainedModel,
    accelerator: Any,
) -> Tuple[Dict[str, torch.Tensor], bool]:
    """
    Извлекает только LoRA параметры из модели.
    Возвращает (state_dict, is_lora_model).
    Для ZeRO-3 — собирает только trainable параметры.
    """
    is_peft = getattr(model, "peft_type", None) is not None or hasattr(model, "get_base_model")
    
    if not is_peft:
        return {}, False
    
    # Получаем trainable параметры (это LoRA адаптеры)
    trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    
    if _is_zero3_model(model, accelerator):
        try:
            from deepspeed.runtime.zero.partition_parameters import GatheredParameters
            
            params_to_gather = list(trainable_params.values())
            with GatheredParameters(params_to_gather, modifier_rank=0):
                if accelerator.is_main_process:
                    # Копируем trainable параметры
                    lora_state = {k: v.cpu().clone() for k, v in trainable_params.items()}
                    return lora_state, True
            return {}, True  # Не main process
        except ImportError:
            pass
    
    # Для не-ZeRO-3: просто копируем
    lora_state = {k: v.cpu().clone() for k, v in trainable_params.items()}
    return lora_state, True


@dataclass
class RolloutSyncStats:
    synced_keys: int
    synced_tensors: int
    total_numel: int


class HFRolloutEngine:
    """
    Отдельная HF-модель для генерации.

    Поддерживает два режима синхронизации:
    1. LoRA-only (trainable_only=True): синхронизируем только LoRA адаптеры — быстро
    2. Full weights (trainable_only=False): синхронизируем все веса — медленно, но работает для full fine-tuning
    
    При ZeRO-3/FSDP веса собираются через GatheredParameters на rank 0, 
    затем broadcast на остальные ранки.
    """

    def __init__(
        self,
        base_model_path: str,
        device: torch.device,
        torch_dtype: torch.dtype,
        use_flash_attention: bool = True,
        trust_remote_code: bool = True,
        offload_to_cpu: bool = False,
    ) -> None:
        self.base_model_path = str(base_model_path)
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_flash_attention = bool(use_flash_attention)
        self.trust_remote_code = bool(trust_remote_code)
        self.offload_to_cpu = bool(offload_to_cpu)

        self.model: Optional[PreTrainedModel] = None
        self._sync_count = 0

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.use_flash_attention:
            # HF >= 4.36: attn_implementation supports flash_attention_2 for compatible models
            model_kwargs["attn_implementation"] = "flash_attention_2"

        logger.info(f"🧩 RolloutEngine(HF): loading model {self.base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(self.base_model_path, **model_kwargs)
        model.eval()
        model.requires_grad_(False)

        if self.offload_to_cpu:
            model.to(torch.device("cpu"))
        else:
            model.to(self.device)

        self.model = model
        logger.info("🧩 RolloutEngine(HF): model loaded")

    def ensure_on_device(self) -> None:
        self.ensure_loaded()
        assert self.model is not None
        if self.offload_to_cpu:
            # move to target device just-in-time for generation
            self.model.to(self.device)

    def maybe_offload(self) -> None:
        if not self.offload_to_cpu:
            return
        if self.model is None:
            return
        self.model.to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def apply_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        strict: bool,
    ) -> RolloutSyncStats:
        self.ensure_loaded()
        assert self.model is not None

        missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"RolloutEngine state_dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}")

        total_numel = 0
        for v in state_dict.values():
            try:
                total_numel += int(v.numel())
            except Exception:
                pass

        return RolloutSyncStats(
            synced_keys=len(state_dict),
            synced_tensors=len(state_dict),
            total_numel=total_numel,
        )

    def sync_weights(
        self,
        training_model: PreTrainedModel,
        accelerator: Any,
        trainable_only: bool = True,
    ) -> Optional[RolloutSyncStats]:
        """
        Синхронизирует веса из training model в rollout engine.
        
        Args:
            training_model: Модель, которая тренируется (может быть обёрнута в ZeRO-3/DDP)
            accelerator: Accelerator объект
            trainable_only: True = только LoRA (быстро), False = все веса (медленно)
        
        Returns:
            RolloutSyncStats или None если синхронизация не нужна на этом ранке.
        """
        self.ensure_loaded()
        is_zero3 = _is_zero3_model(training_model, accelerator)
        
        if trainable_only:
            # LoRA-only синхронизация
            lora_state, is_lora = _gather_lora_state_dict(training_model, accelerator)
            
            if not is_lora:
                logger.warning(
                    "⚠️ trainable_only=True, но модель не использует LoRA. "
                    "Синхронизация пропущена. Для full fine-tuning установите trainable_only=False."
                )
                return None
            
            if accelerator.is_main_process and lora_state:
                # Broadcast LoRA state dict на все ранки
                # Для простоты: сохраняем в shared storage и загружаем
                stats = self.apply_state_dict(lora_state, strict=False)
                self._sync_count += 1
                logger.info(f"🔄 RolloutEngine: LoRA sync #{self._sync_count}, {stats.total_numel:,} params")
                return stats
            elif not accelerator.is_main_process:
                # На не-main процессах ждём и загружаем
                accelerator.wait_for_everyone()
                return None
        else:
            # Full weights синхронизация
            if is_zero3:
                state_dict = _gather_full_state_dict_zero3(training_model, accelerator)
            else:
                # DDP или single GPU — просто берём state_dict
                unwrapped = accelerator.unwrap_model(training_model) if accelerator else training_model
                state_dict = {k: v.cpu().clone() for k, v in unwrapped.state_dict().items()}
            
            if accelerator.is_main_process and state_dict:
                stats = self.apply_state_dict(state_dict, strict=True)
                self._sync_count += 1
                logger.info(f"🔄 RolloutEngine: Full sync #{self._sync_count}, {stats.total_numel:,} params")
                return stats
            elif not accelerator.is_main_process:
                accelerator.wait_for_everyone()
                return None
        
        return None
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        Генерация через HF model.generate().
        Автоматически управляет offload если включен.
        """
        self.ensure_on_device()
        assert self.model is not None
        
        # Переносим входы на устройство модели
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        
        self.maybe_offload()
        return outputs


class VLLMSubprocessEngine:
    """
    vLLM через subprocess.Popen — позволяет запускать на ОТДЕЛЬНОЙ GPU!
    
    Запускает vLLM в НАСТОЯЩЕМ отдельном процессе с CUDA_VISIBLE_DEVICES 
    установленным ДО запуска Python интерпретатора.
    
    Коммуникация через stdin/stdout с JSON lines.
    """
    
    def __init__(
        self,
        base_model_path: str,
        torch_dtype: torch.dtype,
        gpu_id: int = 0,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        enable_lora: bool = True,
        max_lora_rank: int = 64,  # vLLM max_lora_rank - должен быть >= lora_r
        output_dir: Optional[str] = None,
    ) -> None:
        self.base_model_path = str(base_model_path)
        self.torch_dtype = torch_dtype
        self.gpu_id = int(gpu_id)
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.enable_lora = bool(enable_lora)
        self.max_lora_rank = int(max_lora_rank)
        self.output_dir = output_dir
        
        self._process = None
        self._lora_adapter_path: Optional[str] = None
        self._sync_count = 0
    
    def ensure_loaded(self) -> None:
        """Запускает subprocess с vLLM если ещё не запущен."""
        if self._process is not None and self._process.poll() is None:
            return
        
        logger.info(f"🧩 VLLMSubprocessEngine: запуск на физической GPU {self.gpu_id}")
        logger.info(f"🧩 VLLMSubprocessEngine: model={self.base_model_path}, memory={self.gpu_memory_utilization:.0%}")
        
        # Путь к worker скрипту
        worker_script = Path(__file__).parent / "vllm_worker.py"
        if not worker_script.exists():
            raise RuntimeError(f"vLLM worker script not found: {worker_script}")
        
        # Создаём environment с CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # Очищаем переменные которые могут мешать
        env.pop("CUDA_DEVICE_ORDER", None)
        
        logger.info(f"🧩 VLLMSubprocessEngine: запуск с CUDA_VISIBLE_DEVICES={self.gpu_id}")
        
        # Запускаем subprocess
        self._process = subprocess.Popen(
            [sys.executable, str(worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # stderr идёт в консоль для отладки
            env=env,
            bufsize=1,  # line buffered
            text=True,
        )
        logger.info(f"🧩 VLLMSubprocessEngine: subprocess started (PID={self._process.pid})")
        
        # Отправляем конфигурацию
        dtype_str = "bfloat16" if self.torch_dtype == torch.bfloat16 else (
            "float16" if self.torch_dtype == torch.float16 else "float32"
        )
        config = {
            "model_path": self.base_model_path,
            "dtype": dtype_str,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enable_lora": self.enable_lora,
            "max_lora_rank": self.max_lora_rank,
        }
        self._send(config)
        
        # Ждём готовности
        logger.info(f"🧩 VLLMSubprocessEngine: ожидаем загрузку модели...")
        try:
            response = self._recv(timeout=300)  # 5 минут на загрузку
            if response.get("status") == "error":
                raise RuntimeError(f"vLLM worker failed: {response.get('error')}")
            logger.info(f"🧩 VLLMSubprocessEngine: ✅ ready on physical GPU {self.gpu_id}")
        except Exception as e:
            logger.error(f"🧩 VLLMSubprocessEngine: failed to start: {e}")
            self.shutdown()
            raise RuntimeError(f"vLLM subprocess failed to start: {e}")
    
    def _send(self, data: dict) -> None:
        """Отправляет JSON в subprocess."""
        line = json.dumps(data) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()
    
    def _recv(self, timeout: float = 60) -> dict:
        """Получает JSON из subprocess."""
        # Ждём данные с таймаутом
        ready, _, _ = select.select([self._process.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError(f"vLLM worker timeout after {timeout}s")
        
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("vLLM worker closed connection")
        
        return json.loads(line.strip())
    
    def shutdown(self) -> None:
        """Останавливает subprocess."""
        if self._process is not None:
            try:
                self._send({"cmd": "shutdown"})
            except:
                pass
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except:
                self._process.kill()
            self._process = None
        logger.info("🧩 VLLMSubprocessEngine: shutdown")
    
    def __del__(self):
        self.shutdown()
    
    def set_lora_adapter(self, *, lora_path: Optional[str], lora_name: Optional[str] = None, lora_int_id: int = 1) -> None:
        """Устанавливает LoRA адаптер."""
        self.ensure_loaded()
        # ВАЖНО: передаём lora_int_id чтобы vLLM перезагрузил адаптер
        self._send({
            "cmd": "set_lora", 
            "lora_path": lora_path,
            "lora_name": lora_name or "rollout_lora",
            "lora_int_id": int(lora_int_id),
        })
        response = self._recv(timeout=60)
        if response.get("status") == "error":
            raise RuntimeError(f"set_lora failed: {response.get('error')}")
        self._lora_adapter_path = lora_path
        logger.info(f"🧩 VLLMSubprocessEngine: LoRA set to {lora_path} (id={lora_int_id})")
    
    def make_sampling_params(
        self,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Создаёт параметры сэмплирования (как dict для subprocess)."""
        params = {
            "n": int(n),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if stop_token_ids is not None:
            params["stop_token_ids"] = list(stop_token_ids)
        return params
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Any,
    ) -> List[Any]:
        """Генерирует completions через subprocess."""
        self.ensure_loaded()
        
        # Конвертируем SamplingParams в dict
        if isinstance(sampling_params, dict):
            params_dict = sampling_params
        elif hasattr(sampling_params, "__dict__"):
            params_dict = {
                "n": getattr(sampling_params, "n", 1),
                "max_tokens": getattr(sampling_params, "max_tokens", 1024),
                "temperature": getattr(sampling_params, "temperature", 0.7),
                "top_p": getattr(sampling_params, "top_p", 0.9),
                "stop_token_ids": getattr(sampling_params, "stop_token_ids", None),
            }
        else:
            params_dict = {}
        
        self._send({
            "cmd": "generate",
            "prompts": prompts,
            "sampling_params": params_dict,
        })
        
        response = self._recv(timeout=600)  # 10 минут на генерацию
        if response.get("status") == "error":
            raise RuntimeError(f"generate failed: {response.get('error')}")
        
        # Конвертируем обратно в объекты похожие на vLLM outputs
        outputs = response.get("outputs", [])
        return [_VLLMOutput(o) for o in outputs]
    
    def sync_weights(
        self,
        training_model: PreTrainedModel,
        accelerator: Any,
        trainable_only: bool = True,
    ) -> Optional[RolloutSyncStats]:
        """Синхронизирует веса — сохраняем LoRA и обновляем в subprocess."""
        is_peft = getattr(training_model, "peft_type", None) is not None or hasattr(training_model, "get_base_model")
        
        if not trainable_only or not is_peft:
            logger.warning("⚠️ VLLMSubprocessEngine поддерживает только LoRA sync")
            return None
        
        self.ensure_loaded()
        
        # Сохраняем LoRA адаптер
        if self.output_dir:
            adapter_dir = Path(self.output_dir) / "rollout_engine" / "vllm_adapters" / f"step_{self._sync_count}"
        else:
            adapter_dir = Path(tempfile.mkdtemp()) / f"vllm_lora_{self._sync_count}"
        
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            training_model.save_pretrained(str(adapter_dir), safe_serialization=True)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения LoRA: {e}")
            return None
        
        self.set_lora_adapter(lora_path=str(adapter_dir))
        self._sync_count += 1
        
        return RolloutSyncStats(
            time_sync=0.0,
            time_save=0.0,
            params_synced=0,
            bytes_synced=0,
        )


class _VLLMOutput:
    """Простой wrapper для результатов генерации из subprocess."""
    def __init__(self, data: dict):
        self.prompt = data.get("prompt", "")
        self.outputs = [_VLLMCompletionOutput(o) for o in data.get("outputs", [])]


class _VLLMCompletionOutput:
    """Wrapper для одного completion."""
    def __init__(self, data: dict):
        self.text = data.get("text", "")
        self.token_ids = data.get("token_ids", [])
        self.finish_reason = data.get("finish_reason", None)


class VLLMRolloutEngine:
    """
    vLLM rollout engine — высокопроизводительная генерация (НА ТОЙ ЖЕ GPU).
    
    ВАЖНО: Этот класс работает только на той же GPU что training!
    Для использования vLLM на ОТДЕЛЬНОЙ GPU используйте VLLMSubprocessEngine.

    vLLM оптимизирован для inference throughput (continuous batching, PagedAttention).
    
    Поддерживает два режима синхронизации:
    
    1. **LoRA fine-tuning** (trainable_only=True):
       - Быстрая синхронизация через LoRARequest
       - Адаптер сохраняется на диск, vLLM подгружает его "на лету"
       - ~секунды на синхронизацию
    
    2. **Full fine-tuning** (trainable_only=False):
       - vLLM не поддерживает горячую замену всех весов
       - Поэтому: сохраняем checkpoint → перезагружаем vLLM
       - ~5-15 секунд на синхронизацию (для 1.5B модели)
       - Рекомендуется увеличить rollout_sync_interval до 10-50
       - Всё равно быстрее чем генерация через ZeRO-3!
    """

    def __init__(
        self,
        base_model_path: str,
        torch_dtype: torch.dtype,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        enable_lora: bool = True,
        max_lora_rank: int = 64,  # vLLM max_lora_rank - должен быть >= lora_r
        output_dir: Optional[str] = None,
    ) -> None:
        self.base_model_path = str(base_model_path)
        self.torch_dtype = torch_dtype
        self.trust_remote_code = bool(trust_remote_code)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.enable_lora = bool(enable_lora)
        self.max_lora_rank = int(max_lora_rank)
        self.output_dir = output_dir

        self.llm = None
        self._sampling_params_cls = None
        self._lora_request_cls = None
        self._current_lora_request = None
        self._lora_adapter_path: Optional[str] = None
        self._sync_count = 0

    def ensure_loaded(self) -> None:
        if self.llm is not None:
            return
        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            raise RuntimeError(
                "vLLM не установлен или не может быть импортирован. "
                "Установите `pip install vllm` (требуется CUDA) или используйте backend='hf'."
            ) from e

        # LoRARequest API может быть в разных путях в зависимости от версии vLLM
        lora_request_cls = None
        try:
            from vllm.lora.request import LoRARequest  # type: ignore
            lora_request_cls = LoRARequest
        except Exception:
            try:
                from vllm.lora import LoRARequest  # type: ignore
                lora_request_cls = LoRARequest
            except Exception:
                lora_request_cls = None

        self._sampling_params_cls = SamplingParams
        self._lora_request_cls = lora_request_cls

        dtype_str = "bfloat16" if self.torch_dtype == torch.bfloat16 else ("float16" if self.torch_dtype == torch.float16 else "float32")
        logger.info(
            f"🧩 RolloutEngine(vLLM): loading model {self.base_model_path} "
            f"(tp={self.tensor_parallel_size}, dtype={dtype_str}, max_model_len={self.max_model_len}, enable_lora={self.enable_lora})"
        )

        # В vLLM выбор GPU делается через CUDA_VISIBLE_DEVICES на процесс.
        llm_kwargs = {
            "model": self.base_model_path,
            "trust_remote_code": self.trust_remote_code,
            "dtype": dtype_str,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        if self.enable_lora:
            llm_kwargs["enable_lora"] = True
            # vLLM требует max_loras и max_lora_rank если enable_lora=True
            llm_kwargs["max_loras"] = 1
            llm_kwargs["max_lora_rank"] = self.max_lora_rank
        
        self.llm = LLM(**llm_kwargs)
        logger.info("🧩 RolloutEngine(vLLM): model loaded")

    def set_lora_adapter(self, *, lora_path: Optional[str], lora_name: Optional[str] = None, lora_int_id: int = 1) -> None:
        """
        Устанавливает LoRA адаптер для генерации.
        Если lora_path=None — отключает LoRA.
        """
        self.ensure_loaded()
        if self._lora_request_cls is None:
            if lora_path is None:
                self._current_lora_request = None
                return
            raise RuntimeError("vLLM LoRARequest недоступен в вашей версии vLLM.")

        if lora_path is None:
            self._current_lora_request = None
            self._lora_adapter_path = None
            return

        name = lora_name or "rollout_lora"
        self._current_lora_request = self._lora_request_cls(name, int(lora_int_id), str(lora_path))
        self._lora_adapter_path = lora_path
        logger.info(f"🧩 RolloutEngine(vLLM): LoRA adapter set: {lora_path}")

    def sync_weights(
        self,
        training_model: PreTrainedModel,
        accelerator: Any,
        trainable_only: bool = True,
    ) -> Optional[RolloutSyncStats]:
        """
        Синхронизирует веса из training model в vLLM.
        
        Два режима:
        1. trainable_only=True (LoRA): быстрая синхронизация через LoRARequest
        2. trainable_only=False (full): перезагрузка всей модели (дорого, но работает)
        
        Args:
            training_model: Модель (PEFT или обычная)
            accelerator: Accelerator объект
            trainable_only: True = только LoRA, False = все веса (перезагрузка vLLM)
        """
        is_peft = getattr(training_model, "peft_type", None) is not None or hasattr(training_model, "get_base_model")
        
        if trainable_only:
            # LoRA sync — быстрый путь через LoRARequest
            if not is_peft:
                logger.warning(
                    "⚠️ trainable_only=True, но модель не PEFT. "
                    "Для full fine-tuning установите trainable_only=False."
                )
                return None
            
            self.ensure_loaded()
            lora_state, _ = _gather_lora_state_dict(training_model, accelerator)
            
            if accelerator.is_main_process and lora_state:
                self._sync_count += 1
                adapter_save_path = self._get_adapter_save_path()
                
                try:
                    training_model.save_pretrained(adapter_save_path)
                    logger.info(f"🔄 RolloutEngine(vLLM): LoRA adapter saved to {adapter_save_path}")
                except Exception as e:
                    logger.error(f"❌ Не удалось сохранить LoRA адаптер: {e}")
                    return None
                
                self.set_lora_adapter(lora_path=adapter_save_path, lora_name=f"step_{self._sync_count}")
                
                total_numel = sum(v.numel() for v in lora_state.values())
                logger.info(f"🔄 RolloutEngine(vLLM): LoRA sync #{self._sync_count}, {total_numel:,} params")
                
                return RolloutSyncStats(
                    synced_keys=len(lora_state),
                    synced_tensors=len(lora_state),
                    total_numel=total_numel,
                )
            
            if not accelerator.is_main_process:
                accelerator.wait_for_everyone()
            return None
        
        else:
            # Full weights sync — перезагружаем vLLM с новыми весами
            # Это дорого (~5-15 сек), но всё равно быстрее чем генерация через ZeRO-3
            return self._sync_full_weights(training_model, accelerator)
    
    def _sync_full_weights(
        self,
        training_model: PreTrainedModel,
        accelerator: Any,
    ) -> Optional[RolloutSyncStats]:
        """
        Полная синхронизация весов: сохраняем checkpoint, перезагружаем vLLM.
        """
        import time
        
        is_zero3 = _is_zero3_model(training_model, accelerator)
        self._sync_count += 1
        
        # Путь для сохранения checkpoint'а
        checkpoint_path = self._get_full_checkpoint_path()
        
        # 1. Собираем и сохраняем веса на rank 0
        if accelerator.is_main_process:
            logger.info(f"🔄 RolloutEngine(vLLM): Full sync #{self._sync_count} — сохраняем checkpoint...")
            start_time = time.time()
            
            if is_zero3:
                # ZeRO-3: собираем sharded веса
                state_dict = _gather_full_state_dict_zero3(training_model, accelerator)
                if state_dict:
                    # Сохраняем в формате HF
                    unwrapped = accelerator.unwrap_model(training_model)
                    unwrapped.save_pretrained(
                        checkpoint_path,
                        state_dict=state_dict,
                        safe_serialization=True,
                    )
            else:
                # DDP или single GPU
                unwrapped = accelerator.unwrap_model(training_model)
                unwrapped.save_pretrained(checkpoint_path, safe_serialization=True)
            
            save_time = time.time() - start_time
            logger.info(f"🔄 RolloutEngine(vLLM): checkpoint saved in {save_time:.1f}s")
        
        # 2. Барьер — все ранки ждут пока checkpoint сохранится
        accelerator.wait_for_everyone()
        
        # 3. Перезагружаем vLLM на всех ранках с новым checkpoint'ом
        if accelerator.is_main_process:
            logger.info(f"🔄 RolloutEngine(vLLM): перезагружаем vLLM с {checkpoint_path}...")
            start_time = time.time()
        
        # Закрываем старый vLLM
        old_llm = self.llm
        self.llm = None
        del old_llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Меняем путь к модели на checkpoint
        self.base_model_path = str(checkpoint_path)
        # Для full fine-tuning отключаем LoRA в vLLM
        self.enable_lora = False
        self._current_lora_request = None
        
        # Загружаем заново
        self.ensure_loaded()
        
        if accelerator.is_main_process:
            reload_time = time.time() - start_time
            logger.info(f"🔄 RolloutEngine(vLLM): vLLM reloaded in {reload_time:.1f}s")
        
        # Барьер после перезагрузки
        accelerator.wait_for_everyone()
        
        # Оценка количества параметров
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(checkpoint_path)
            estimated_params = getattr(cfg, "num_parameters", None)
            if estimated_params is None:
                # Грубая оценка
                estimated_params = cfg.hidden_size * cfg.num_hidden_layers * 12
        except Exception:
            estimated_params = 0
        
        return RolloutSyncStats(
            synced_keys=1,  # один checkpoint
            synced_tensors=1,
            total_numel=estimated_params,
        )
    
    def _get_full_checkpoint_path(self) -> str:
        """Путь для сохранения полного checkpoint'а."""
        if self.output_dir:
            ckpt_dir = Path(self.output_dir) / "rollout_engine" / "vllm_checkpoints" / f"step_{self._sync_count}"
        else:
            ckpt_dir = Path(tempfile.gettempdir()) / "vllm_full_checkpoints" / f"step_{self._sync_count}"
        
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return str(ckpt_dir)
    
    def _get_adapter_save_path(self) -> str:
        """Возвращает путь для сохранения LoRA адаптера."""
        if self.output_dir:
            adapter_dir = Path(self.output_dir) / "rollout_engine" / "vllm_adapters" / f"step_{self._sync_count}"
        else:
            adapter_dir = Path(tempfile.gettempdir()) / "vllm_lora_adapters" / f"step_{self._sync_count}"
        
        adapter_dir.mkdir(parents=True, exist_ok=True)
        return str(adapter_dir)

    def make_sampling_params(
        self,
        *,
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
    ):
        self.ensure_loaded()
        assert self._sampling_params_cls is not None
        kwargs = {
            "n": int(n),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if stop_token_ids is not None:
            kwargs["stop_token_ids"] = list(stop_token_ids)
        return self._sampling_params_cls(**kwargs)

    def generate(self, prompts: List[str], sampling_params) -> List:
        """
        Генерация через vLLM.
        
        Args:
            prompts: Список промптов (строки)
            sampling_params: vLLM SamplingParams
            
        Returns:
            list[RequestOutput] (vLLM API)
        """
        self.ensure_loaded()
        assert self.llm is not None
        # lora_request применяется ко всем запросам батча
        return self.llm.generate(prompts, sampling_params, lora_request=self._current_lora_request)

