#!/usr/bin/env python3
"""
Standalone vLLM worker script.
Запускается как отдельный процесс с CUDA_VISIBLE_DEVICES установленным ДО старта Python.
Коммуникация через stdin/stdout (JSON lines).

ВАЖНО: stdout используется ТОЛЬКО для JSON протокола!
Все логи идут в stderr.
"""
import json
import sys
import os

# Сохраняем оригинальный stdout для JSON протокола
_json_out = sys.stdout

# Перенаправляем stdout в stderr чтобы vLLM логи не попадали в JSON канал
sys.stdout = sys.stderr


def send_json(data: dict) -> None:
    """Отправляет JSON в оригинальный stdout (протокол)."""
    _json_out.write(json.dumps(data) + "\n")
    _json_out.flush()


def main():
    # Логируем environment
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"🧩 vLLM Worker: CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
    
    # Импортируем torch и проверяем GPU
    import torch
    print(f"🧩 vLLM Worker: torch.cuda.device_count()={torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        print(f"🧩 vLLM Worker: cuda:0 = {torch.cuda.get_device_name(0)}", flush=True)
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        print(f"🧩 vLLM Worker: GPU memory: {free_mem/1e9:.1f}GB free / {total_mem/1e9:.1f}GB total", flush=True)
    
    # Импортируем vLLM
    from vllm import LLM, SamplingParams
    try:
        from vllm.lora.request import LoRARequest
    except ImportError:
        from vllm.lora import LoRARequest
    
    print(f"🧩 vLLM Worker: vLLM imported", flush=True)
    
    # Читаем конфигурацию из первой строки stdin
    # stdin остался оригинальным, читаем из него
    import sys as _sys
    config_line = _sys.__stdin__.readline()
    config = json.loads(config_line)
    
    model_path = config["model_path"]
    dtype_str = config["dtype"]
    max_model_len = config["max_model_len"]
    gpu_memory_utilization = config["gpu_memory_utilization"]
    enable_lora = config["enable_lora"]
    max_lora_rank = config.get("max_lora_rank", 64)  # Дефолт 64, чтобы поддерживать rank до 64
    
    print(f"🧩 vLLM Worker: loading model {model_path}", flush=True)
    print(f"🧩 vLLM Worker: enable_lora={enable_lora}, max_lora_rank={max_lora_rank}", flush=True)
    
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "dtype": dtype_str,
        "tensor_parallel_size": 1,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": True,  # Отключаем CUDA graphs
    }
    if enable_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_loras"] = 1
        llm_kwargs["max_lora_rank"] = max_lora_rank  # КРИТИЧНО! Должен быть >= lora_r
    
    try:
        llm = LLM(**llm_kwargs)
        print(f"🧩 vLLM Worker: model loaded!", flush=True)
        # Отправляем ready signal
        send_json({"status": "ready"})
    except Exception as e:
        print(f"🧩 vLLM Worker: failed to load: {e}", flush=True)
        send_json({"status": "error", "error": str(e)})
        _sys.exit(1)
    
    current_lora_request = None
    
    # Основной цикл обработки запросов
    for line in _sys.__stdin__:
        try:
            request = json.loads(line.strip())
            cmd = request.get("cmd")
            
            if cmd == "shutdown":
                print(f"🧩 vLLM Worker: shutting down", flush=True)
                break
            
            elif cmd == "set_lora":
                lora_path = request.get("lora_path")
                lora_name = request.get("lora_name", "rollout_lora")
                lora_int_id = request.get("lora_int_id", 1)
                
                if lora_path:
                    # Проверяем что адаптер существует
                    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
                    adapter_model_path = os.path.join(lora_path, "adapter_model.safetensors")
                    adapter_model_bin_path = os.path.join(lora_path, "adapter_model.bin")
                    
                    if not os.path.exists(adapter_config_path):
                        print(f"🧩 vLLM Worker: ERROR - adapter_config.json not found at {lora_path}", flush=True)
                        # Показываем что есть в директории
                        if os.path.isdir(lora_path):
                            files = os.listdir(lora_path)
                            print(f"🧩 vLLM Worker: Files in {lora_path}: {files}", flush=True)
                        send_json({"status": "error", "error": f"adapter_config.json not found at {lora_path}"})
                        continue
                    
                    # Проверяем наличие весов
                    has_weights = os.path.exists(adapter_model_path) or os.path.exists(adapter_model_bin_path)
                    if not has_weights:
                        print(f"🧩 vLLM Worker: WARNING - no adapter weights found (safetensors or bin)", flush=True)
                    
                    # Читаем конфиг адаптера для диагностики
                    try:
                        with open(adapter_config_path, 'r') as f:
                            adapter_cfg = json.load(f)
                        print(f"🧩 vLLM Worker: adapter_config: r={adapter_cfg.get('r')}, alpha={adapter_cfg.get('lora_alpha')}, modules={adapter_cfg.get('target_modules', [])[:3]}...", flush=True)
                    except Exception as e:
                        print(f"🧩 vLLM Worker: couldn't read adapter_config: {e}", flush=True)
                    
                    # ВАЖНО: используем уникальный lora_int_id чтобы vLLM перезагрузил адаптер
                    current_lora_request = LoRARequest(str(lora_name), int(lora_int_id), str(lora_path))
                    print(f"🧩 vLLM Worker: LoRA set to {lora_path} (name={lora_name}, id={lora_int_id})", flush=True)
                else:
                    current_lora_request = None
                    print(f"🧩 vLLM Worker: LoRA disabled", flush=True)
                send_json({"status": "ok"})
            
            elif cmd == "generate":
                prompts = request.get("prompts", [])
                sampling_params_dict = request.get("sampling_params", {})
                
                # Фильтруем None
                filtered_params = {k: v for k, v in sampling_params_dict.items() if v is not None}
                print(f"🧩 vLLM Worker: generating {len(prompts)} prompts", flush=True)
                
                sampling_params = SamplingParams(**filtered_params)
                outputs = llm.generate(prompts, sampling_params, lora_request=current_lora_request)
                
                # Сериализуем результаты
                results = []
                for output in outputs:
                    result = {
                        "prompt": output.prompt,
                        "outputs": [
                            {
                                "text": o.text,
                                "token_ids": list(o.token_ids) if o.token_ids else [],
                                "finish_reason": str(o.finish_reason) if o.finish_reason else None,
                            }
                            for o in output.outputs
                        ]
                    }
                    results.append(result)
                
                print(f"🧩 vLLM Worker: generated {len(results)} outputs", flush=True)
                send_json({"status": "ok", "outputs": results})
            
            else:
                send_json({"status": "error", "error": f"unknown cmd: {cmd}"})
        
        except Exception as e:
            print(f"🧩 vLLM Worker: error: {e}", flush=True)
            send_json({"status": "error", "error": str(e)})


if __name__ == "__main__":
    main()
