"""Chat endpoints: load model, generate text with streaming."""
from __future__ import annotations

import threading
import json
from queue import Empty
from typing import Literal
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

router = APIRouter()
_model_lock = threading.Lock()

_loaded_backend = None
_loaded_model_path = None
_loaded_backend_name = None


class LoadRequest(BaseModel):
    model_path: str
    backend: Literal["transformers", "vllm", "llama.cpp"] = "transformers"


class GenerateRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    messages: list[dict]
    model_path: str
    backend: Literal["transformers", "vllm", "llama.cpp"] = "transformers"
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    top_k: int = Field(default=50, ge=0)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    stream: bool = True


@router.post("/load")
def load_model(req: LoadRequest):
    """Load a model for chat inference."""
    with _model_lock:
        try:
            _ensure_chat_backend(req)
            return {"success": True}
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc


@router.post("/unload")
def unload_model():
    """Unload the currently loaded model."""
    with _model_lock:
        global _loaded_backend, _loaded_model_path, _loaded_backend_name
        _loaded_backend = None
        _loaded_model_path = None
        _loaded_backend_name = None

        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        return {"success": True}


@router.get("/backends")
def available_backends():
    backends = ["transformers"]
    try:
        from homellm.app.vllm_chat import is_vllm_available
        if is_vllm_available():
            backends.append("vllm")
    except Exception:
        # vllm может конфликтовать с transformers (aimv2 etc.) — не валим эндпойнт
        pass
    try:
        from homellm.app.llama_cpp_chat import is_llama_cpp_available
        if is_llama_cpp_available():
            backends.append("llama.cpp")
    except Exception:
        pass
    return {"backends": backends}


@router.post("/generate")
def generate(req: GenerateRequest):
    """Generate a response. Supports SSE streaming."""
    if req.stream:
        return StreamingResponse(
            _stream_generate(req),
            media_type="text/event-stream",
        )
    else:
        text = _generate_full(req)
        return {"text": text}


def _stream_generate(req: GenerateRequest):
    """SSE generator for streaming tokens."""
    with _model_lock:
        try:
            if req.backend in ("vllm", "llama.cpp"):
                backend = _ensure_chat_backend(req)
                prompt = _apply_backend_chat_template(backend, req.messages)
                text = backend.generate(
                    prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                )
                if text:
                    yield f"data: {json.dumps({'text': text})}\n\n"
                yield "data: [DONE]\n\n"
                return

            import torch
            from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
            from threading import Thread, Event

            loaded = _ensure_chat_backend(req)
            model, tokenizer = loaded["model"], loaded["tokenizer"]
            chat_text = _tokenizer_prompt(tokenizer, req.messages)
            inputs = tokenizer(chat_text, return_tensors="pt", return_token_type_ids=False).to(model.device)
            _validate_context(model, inputs, req.max_tokens)

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=2)
            gen_kwargs = {**inputs, **_generation_settings(req), "streamer": streamer}

            cancelled = Event()
            errors = []
            class Cancelled(StoppingCriteria):
                def __call__(self, *args, **kwargs):
                    return cancelled.is_set()
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([Cancelled()])
            def generate_tokens():
                try:
                    with torch.inference_mode():
                        model.generate(**gen_kwargs)
                except Exception as exc:
                    errors.append(exc)
                    streamer.on_finalized_text("", stream_end=True)
            thread = Thread(target=generate_tokens, daemon=True)
            thread.start()
            try:
                while True:
                    try:
                        text = next(streamer)
                    except Empty:
                        if not thread.is_alive():
                            break
                        continue
                    except StopIteration:
                        break
                    if text:
                        yield f"data: {json.dumps({'text': text})}\n\n"
                if errors:
                    raise errors[0]
            finally:
                cancelled.set()
                thread.join()
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"


def _generate_full(req: GenerateRequest) -> str:
    """Non-streaming generation."""
    with _model_lock:
        try:
            if req.backend in ("vllm", "llama.cpp"):
                backend = _ensure_chat_backend(req)
                prompt = _apply_backend_chat_template(backend, req.messages)
                return backend.generate(
                    prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                )

            loaded = _ensure_chat_backend(req)
            model, tokenizer = loaded["model"], loaded["tokenizer"]
            chat_text = _tokenizer_prompt(tokenizer, req.messages)
            inputs = tokenizer(chat_text, return_tensors="pt", return_token_type_ids=False).to(model.device)
            _validate_context(model, inputs, req.max_tokens)

            outputs = model.generate(**inputs, **_generation_settings(req))

            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return text

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))


def _ensure_chat_backend(req: GenerateRequest | LoadRequest):
    """Return the requested loaded backend, loading it lazily if needed."""
    global _loaded_backend, _loaded_model_path, _loaded_backend_name

    if (
        _loaded_backend is not None
        and _loaded_model_path == req.model_path
        and _loaded_backend_name == req.backend
    ):
        return _loaded_backend

    if req.backend == "vllm":
        from homellm.app.vllm_chat import VLLMChatBackend
        _loaded_backend = VLLMChatBackend(req.model_path)
    elif req.backend == "llama.cpp":
        from homellm.app.llama_cpp_chat import LlamaCppChatBackend
        _loaded_backend = LlamaCppChatBackend(req.model_path)
    elif req.backend == "transformers":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(req.model_path, trust_remote_code=True)
        dtype = torch.float32 if not torch.cuda.is_available() else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        model_class = AutoModelForCausalLM
        home_model = False
        config_file = Path(req.model_path) / "config.json"
        if config_file.is_file():
            kind = json.loads(config_file.read_text()).get('model_type')
            if kind in ('homellm', 'homellm_moe', 'gpt2_home', 'homellm_blueprint'):
                from backend.services.custom_models import model_classes
                _, model_class = model_classes(kind)
                home_model = True
        model = model_class.from_pretrained(req.model_path, torch_dtype=dtype,
                                           device_map="auto", trust_remote_code=True)
        if home_model:
            if kind in ('homellm_moe', 'gpt2_home', 'homellm_blueprint'):
                # These legacy implementations advertise DynamicCache support
                # but cannot consume an empty modern Transformers cache.
                model.generation_config.use_cache = False
            for key in ("eos_token_id", "pad_token_id", "bos_token_id"):
                value = getattr(tokenizer, key, None)
                setattr(model.generation_config, key, value)
        _loaded_backend = {"model": model, "tokenizer": tokenizer}
    else:
        raise ValueError(f"Unsupported chat backend: {req.backend}")

    _loaded_model_path = req.model_path
    _loaded_backend_name = req.backend
    return _loaded_backend


def _apply_backend_chat_template(backend, messages: list[dict]) -> str:
    if hasattr(backend, "apply_chat_template"):
        return backend.apply_chat_template(messages, add_generation_prompt=True)

    chunks = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        chunks.append(f"{role}: {content}")
    chunks.append("assistant:")
    return "\n".join(chunks)


def _tokenizer_prompt(tokenizer, messages: list[dict]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _apply_backend_chat_template(object(), messages)


def _generation_settings(req: GenerateRequest) -> dict:
    settings = {"max_new_tokens": req.max_tokens, "do_sample": req.temperature > 0}
    if req.temperature > 0:
        settings.update(temperature=req.temperature, top_p=req.top_p, top_k=req.top_k)
    return settings


def _validate_context(model, inputs, max_tokens: int) -> None:
    config = getattr(model, 'config', None)
    if getattr(config, 'model_type', '') not in ('homellm', 'homellm_moe', 'gpt2_home', 'homellm_blueprint'):
        return
    capacity = getattr(config, 'max_position_embeddings', None)
    if isinstance(capacity, int) and inputs['input_ids'].shape[-1] + max_tokens > capacity:
        raise HTTPException(400, f'Prompt plus max_tokens exceeds the model context length ({capacity})')
