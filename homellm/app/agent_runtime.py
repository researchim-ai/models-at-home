"""Agent loop and prompt contract for Agent Studio."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .agent_tools import execute_tool, get_tool_specs

SYSTEM_PROMPT = """Ты внутренний агент Models at Home Studio.

Твоя задача:
- помогать пользователю готовить и запускать обучение LLM/VLM;
- использовать доступные tools для проверки моделей, датасетов и запуска run;
- предлагать реалистичные конфиги, а не абстрактные советы;
- не выдумывать состояние файлов, моделей и датасетов, если можно проверить tool'ом.

Правила поведения:
- Если для ответа нужна фактическая информация о локальном проекте, сначала вызывай tool.
- Перед запуском обучения проверяй, что есть датасет, базовая модель и разумный output_dir.
- Не вызывай больше 2 tools за один шаг.
- Если пользователь просит план, сначала собери контекст tool'ами и только потом формируй план.
- Если запускаешь обучение, коротко объясни, что именно стартуешь и почему такой preset.
- Если данных недостаточно, попроси уточнение, а не придумывай.

Ты ОБЯЗАН отвечать строго одним JSON-объектом без markdown и без пояснений вокруг.

Формат ответа:
{
  "assistant_message": "текст для пользователя",
  "tool_calls": [
    {"tool": "tool_name", "arguments": {"key": "value"}}
  ],
  "final": false
}

Требования к JSON:
- Всегда включай ключи assistant_message, tool_calls, final.
- assistant_message должен быть строкой.
- tool_calls должен быть массивом.
- final должен быть true только если уже готов финальный ответ пользователю на этот ход.
- Если вызываешь tools, не пиши длинный финальный ответ заранее.
"""


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


def build_agent_prompt(
    conversation: List[Dict[str, str]],
    tool_history: List[Dict[str, Any]],
) -> str:
    return "\n\n".join(
        [
            SYSTEM_PROMPT,
            "Доступные tools:",
            _serialize_json(get_tool_specs()),
            "История диалога:",
            _format_conversation(conversation),
            "История tool usage в этом ходе:",
            _format_tool_history(tool_history),
            "Сформируй следующий JSON-ответ сейчас.",
        ]
    )


def run_agent_turn(
    backend: Any,
    conversation: List[Dict[str, str]],
    *,
    max_steps: int = 6,
    max_tokens: int = 700,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run a multi-step agent loop and return final text plus execution trace."""
    tool_history: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        prompt = build_agent_prompt(conversation=conversation, tool_history=tool_history)
        raw = backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=["</tool_result>", "\nUSER:\n", "\nSYSTEM:\n"],
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
                    prompt=prompt,
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
            try:
                result = execute_tool(tool_name, arguments)
            except Exception as exc:
                result = {"error": str(exc), "tool": tool_name, "arguments": arguments}
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
                prompt=prompt,
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
