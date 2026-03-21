"""Agent loop and prompt contract for Agent Studio."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .agent_tools import execute_tool, get_tool_specs

AGENT_PROMPT_SECTIONS: Dict[str, str] = {
    "role": """Ты внутренний агент Models at Home Studio.
Твоя задача — помогать пользователю готовить, запускать и сопровождать обучение моделей внутри студии.""",
    "capabilities": """Твои возможности:
- анализировать локальные trainable-модели, датасеты и готовые training presets;
- запускать text/VLM training и ТОНКО НАСТРАИВАТЬ любые гиперпараметры (learning_rate, batch_size, lora_r, epochs и т.д.);
- проверять статус run, читать config, metrics и логи;
- подсказывать, какой preset или конфиг лучше подходит под задачу пользователя;
- выполнять bash-команды внутри контейнера через run_system_command (например: nvidia-smi, ls, free -h).""",
    "rules": """Правила:
- не выдумывай состояние файлов, моделей, датасетов и процессов, если это можно проверить tool'ом;
- используй run_system_command для проверки системного железа и нагрузки (GPU, RAM), если пользователь просит;
- если пользователь просит запустить обучение, но не указал конкретную модель или датасет — выбери наиболее разумные/подходящие локальные файлы сам по умолчанию и сразу запускай процесс, не задавай лишних уточняющих вопросов;
- перед запуском обучения быстро проверяй наличие датасета и модели;
- строго используй только те названия параметров (ключи словаря), которые описаны в arguments tool'а (например: используй 'data_path', а не 'dataset_path'; 'epochs', а не 'num_epochs');
- не предлагай GGUF-файлы llama.cpp как базовые модели для обучения: они используются только для inference;
- не вызывай больше 2 tools за один шаг;
- если запускаешь run, обязательно сообщай пользователю, что именно стартуешь и почему;
- при успешном запуске обучения ОБЯЗАТЕЛЬНО дай пользователю ссылку `[Перейти к мониторингу (LLM Студия)](/)` или `[Перейти к мониторингу (VLM Студия)](/VLM_Studio)` и укажи Run ID, чтобы он мог перейти по ссылке;
- не говори, что обучение успешно запущено, если tool вернул ошибку или run умер сразу после старта;
- если запущен run, предлагай пользователю смотреть плашку активного процесса, конфиг, логи и графики на странице;""",
    "output_contract": """Ты ОБЯЗАН отвечать строго одним JSON-объектом без markdown и без пояснений вокруг.

Формат:
{
  "assistant_message": "текст для пользователя",
  "tool_calls": [
    {"tool": "tool_name", "arguments": {"key": "value"}}
  ],
  "final": false
}

Требования:
- всегда включай assistant_message, tool_calls, final;
- assistant_message должен быть строкой;
- tool_calls должен быть массивом;
- final=true только если уже готов финальный ответ на текущий ход;
- если вызываешь tools, не пиши заранее длинный финальный ответ.""",
}

SYSTEM_PROMPT = "\n\n".join(AGENT_PROMPT_SECTIONS.values())


def get_agent_prompt_sections() -> Dict[str, str]:
    return AGENT_PROMPT_SECTIONS


@dataclass
class AgentStep:
    step: int
    prompt: str
    raw_response: str
    parsed: Dict[str, Any]
    tool_results: List[Dict[str, Any]]


def _serialize_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> str:
    text = text.strip()
    
    # Check for markdown codeblocks first
    if "```json" in text:
        parts = text.split("```json")
        for part in parts[1:]:
            if "```" in part:
                candidate = part.split("```")[0].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    return candidate
                    
    if text.startswith("{") and text.endswith("}"):
        return text

    fenced = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(0)
    raise ValueError("Model did not return a JSON object")


def _parse_agent_response(raw: str) -> Dict[str, Any]:
    parsed = json.loads(_extract_json_object(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Agent response is not a JSON object")
    parsed.setdefault("assistant_message", "")
    parsed.setdefault("tool_calls", [])
    parsed.setdefault("final", False)
    if not isinstance(parsed["assistant_message"], str):
        parsed["assistant_message"] = str(parsed["assistant_message"])
    if not isinstance(parsed["tool_calls"], list):
        parsed["tool_calls"] = []
    parsed["final"] = bool(parsed["final"])
    return parsed


def _format_conversation(messages: List[Dict[str, str]]) -> str:
    blocks: List[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        blocks.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(blocks)


def _format_tool_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "[]"
    return _serialize_json(history)


def build_agent_messages(
    conversation: List[Dict[str, str]],
    tool_history: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    system_content = "\n\n".join(
        [
            SYSTEM_PROMPT,
            "Доступные tools:",
            _serialize_json(get_tool_specs()),
        ]
    )
    
    messages = [{"role": "system", "content": system_content}]
    for msg in conversation:
        if msg.get("role") == "system":
            continue
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
    if tool_history:
        messages.append({
            "role": "user",
            "content": f"История tool usage в этом ходе:\n{_format_tool_history(tool_history)}\n\nСформируй следующий JSON-ответ сейчас."
        })
    else:
        if messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "Сформируй следующий JSON-ответ сейчас."})
        else:
            messages[-1]["content"] = messages[-1]["content"] + "\n\nСформируй следующий JSON-ответ сейчас."
            
    return messages


def run_agent_turn(
    backend: Any,
    conversation: List[Dict[str, str]],
    *,
    max_steps: int = 6,
    max_tokens: int = 700,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
    stream_callback: Any = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run a multi-step agent loop and return final text plus execution trace."""
    tool_history: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        messages = build_agent_messages(conversation=conversation, tool_history=tool_history)
        prompt_text = "\n\n".join([msg.get("content", "") for msg in messages])
        
        # Fallback for backends that don't support chat_completion
        if not hasattr(backend, "chat_completion"):
            raw = backend.generate(
                prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=["</tool_result>", "\nUSER:\n", "\nSYSTEM:\n"],
            )
        else:
            if hasattr(backend, "chat_completion_stream") and stream_callback:
                raw_chunks = []
                stream_callback("\n\n") # add spacing between steps in stream
                for chunk in backend.chat_completion_stream(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=["</tool_result>"],
                ):
                    if chunk is not None:
                        raw_chunks.append(chunk)
                        stream_callback(chunk)
                raw = "".join(raw_chunks)
            else:
                raw = backend.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=["</tool_result>"],
                )
            
        parsed: Dict[str, Any]
        tool_results: List[Dict[str, Any]] = []
        try:
            parsed = _parse_agent_response(raw)
        except Exception as exc:
            parsed = {
                "assistant_message": (
                    "Не удалось надёжно распарсить ответ модели как JSON. "
                    "Нужно либо уменьшить температуру, либо переформулировать запрос."
                ),
                "tool_calls": [],
                "final": True,
                "parse_error": str(exc),
            }
            trace.append(
                AgentStep(
                    step=step,
                    prompt=prompt_text,
                    raw_response=raw,
                    parsed=parsed,
                    tool_results=[],
                ).__dict__
            )
            return parsed["assistant_message"], trace

        tool_calls = parsed.get("tool_calls", [])[:2]
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("tool", "")).strip()
            arguments = tool_call.get("arguments") or {}
            
            if stream_callback:
                stream_callback(f"\n\n*(Вызываю инструмент: `{tool_name}`...)*\n\n")
                
            try:
                result = execute_tool(tool_name, arguments)
                if stream_callback:
                    stream_callback(f"*(Инструмент `{tool_name}` успешно выполнен)*\n\n")
            except Exception as exc:
                result = {"error": str(exc), "tool": tool_name, "arguments": arguments}
                if stream_callback:
                    stream_callback(f"*(Ошибка при вызове `{tool_name}`)*\n\n")
                    
            tool_record = {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            }
            tool_results.append(tool_record)
            tool_history.append(tool_record)

        trace.append(
            AgentStep(
                step=step,
                prompt=prompt_text,
                raw_response=raw,
                parsed=parsed,
                tool_results=tool_results,
            ).__dict__
        )

        if parsed.get("final") and parsed.get("assistant_message"):
            return parsed["assistant_message"], trace

        if not tool_results and parsed.get("assistant_message"):
            return parsed["assistant_message"], trace

    fallback = (
        "Я собрал промежуточный контекст, но упёрся в лимит шагов. "
        "Сузь задачу или попроси меня сначала собрать только план/конфиг."
    )
    return fallback, trace
