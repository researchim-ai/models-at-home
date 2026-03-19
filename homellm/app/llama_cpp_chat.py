"""llama.cpp-backed chat backend for local agent inference."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def is_llama_cpp_available() -> bool:
    """Return True when llama-cpp-python is installed."""
    try:
        import llama_cpp  # noqa: F401

        return True
    except ImportError:
        return False


class LlamaCppChatBackend:
    """Thin wrapper around llama.cpp for prompt-completion style inference."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        from llama_cpp import Llama

        self.model_path = model_path
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_batch = int(n_batch)
        self.n_threads = int(n_threads or max(1, (os.cpu_count() or 4) // 2))

        logger.info(
            "Loading llama.cpp model path=%s n_ctx=%s n_gpu_layers=%s n_batch=%s n_threads=%s",
            model_path,
            self.n_ctx,
            self.n_gpu_layers,
            self.n_batch,
            self.n_threads,
        )

        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            n_threads=self.n_threads,
            verbose=verbose,
            **kwargs,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = 40,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        """Generate a completion for the provided prompt."""
        params: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "echo": False,
        }
        if top_p is not None:
            params["top_p"] = float(top_p)
        if top_k is not None:
            params["top_k"] = int(top_k)
        if stop:
            params["stop"] = stop
        if stream:
            chunks = self.llm.create_completion(stream=True, **params)
            pieces: List[str] = []
            for chunk in chunks:
                for choice in chunk.get("choices", []):
                    text = choice.get("text")
                    if text:
                        pieces.append(text)
            return "".join(pieces)

        result = self.llm.create_completion(**params)
        choices = result.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("text", ""))

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Fallback formatter for chat-style prompts."""
        chunks: List[str] = []
        for message in messages:
            role = message.get("role", "user").strip().lower()
            content = message.get("content", "")
            if role == "system":
                chunks.append(f"<|system|>\n{content}")
            elif role == "assistant":
                chunks.append(f"<|assistant|>\n{content}")
            else:
                chunks.append(f"<|user|>\n{content}")
        if add_generation_prompt:
            chunks.append("<|assistant|>\n")
        return "\n".join(chunks)

    @property
    def has_chat_template(self) -> bool:
        """Expose a similar interface to the other chat backends."""
        return False
