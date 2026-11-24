"""Agent Studio endpoints: llama-server management, agent chat, sessions."""
from __future__ import annotations

import threading
import json
import re
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.paths import get_project_root, safe_id, atomic_json

router = APIRouter()
_model_lock = threading.Lock()

PROJECT_ROOT = get_project_root(Path(__file__).parent.parent.parent / "models-at-home")
SESSIONS_DIR = PROJECT_ROOT / ".runs" / "agent_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

_server_backend = None
_agent_temperature = 0.2
_server_model_path: str | None = None
_server_ctx: int = 32768

LLAMA_CHAT_ENDPOINT = "http://127.0.0.1:8787/v1/chat/completions"
AGENT_DEBUG_LOG = SESSIONS_DIR / "agent_debug.log"


def _log_agent_debug(finish_reason: str | None, content: str, reasoning: str) -> None:
    """Append a compact record of each model turn so parse failures are
    diagnosable from a real run (finish_reason='length' => truncation, etc.)."""
    try:
        entry = (
            f"\n===== {datetime.now().isoformat()} =====\n"
            f"finish_reason={finish_reason}\n"
            f"content_len={len(content)} reasoning_len={len(reasoning)}\n"
            f"--- content ---\n{content[:4000]}\n"
            f"--- reasoning (head) ---\n{reasoning[:1500]}\n"
        )
        with open(AGENT_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception:
        pass


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks (they can contain braces that break the
    naive first-{/last-} JSON extraction in agent_runtime), including a trailing
    unterminated <think> left by truncated reasoning."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    idx = cleaned.lower().rfind("<think>")
    if idx != -1 and "</think>" not in cleaned[idx:].lower():
        cleaned = cleaned[:idx]
    return cleaned.strip()


def _extract_balanced_json(text: str) -> str:
    """Return the first brace-balanced JSON object (string-aware). If it never
    closes (truncated by max_tokens) return the tail so recovery can still run."""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _looks_like_json(text: str) -> bool:
    return bool(text) and text.lstrip().startswith("{")


def _clean_agent_json(content: str) -> str:
    """Turn a raw model turn into just its JSON payload (reasoning removed)."""
    stripped = _strip_reasoning(content)
    return _extract_balanced_json(stripped) or stripped


class _AgentBackend:
    """Keeps the model's reasoning ENABLED (it materially improves tool planning)
    but sanitizes the raw response before homellm's strict JSON parser sees it.

    Root cause of 'Не удалось распарсить ответ модели как JSON': on the step that
    summarizes a large tool result, the model's <think> reasoning quotes run
    configs containing `{ }`, and agent_runtime extracts from the first `{` (inside
    the reasoning) to the last `}`, producing invalid JSON. We therefore:
      * pull the reasoning channel out separately,
      * strip any inline <think> blocks,
      * isolate a brace-balanced JSON object,
    so the parser always receives clean JSON while the model still reasons freely.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def chat_completion(self, messages, max_tokens=512, temperature=0.2,
                        top_p=0.95, top_k=40, stop=None, stream=False):
        payload: dict = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p or 0.95),
            "top_k": int(top_k or 40),
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                LLAMA_CHAT_ENDPOINT,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1800) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            choice = (result.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content") if isinstance(msg.get("content"), str) else ""
            reasoning = msg.get("reasoning_content") if isinstance(msg.get("reasoning_content"), str) else ""
            _log_agent_debug(choice.get("finish_reason"), content, reasoning)

            cleaned = _clean_agent_json(content)
            # If the model routed its JSON into the reasoning channel (or content
            # was empty), recover it from there.
            if not _looks_like_json(cleaned) and reasoning:
                alt = _clean_agent_json(reasoning)
                if _looks_like_json(alt):
                    cleaned = alt
            return cleaned
        except Exception as exc:
            _log_agent_debug("exception", str(exc), "")
            # Fall back to the backend's own implementation, still cleaned.
            raw = self._inner.chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, stop=stop,
            )
            return _clean_agent_json(raw)


def _recommended_output_tokens(n_ctx: int) -> int:
    """Reasonable agent response budget based on active context size.

    Mirrors homellm/app/pages/06_Agent_Studio.py. The agent must be able to emit
    a full JSON object (including reasoning + tool results summary); a small budget
    causes thinking models to truncate the JSON and the parse fails.
    """
    if n_ctx <= 32768:
        return 4096
    if n_ctx <= 131072:
        return 8192
    return 16384


def _unescape_json_string(value: str) -> str:
    for src, dst in (("\\n", "\n"), ("\\t", "\t"), ('\\"', '"'), ("\\\\", "\\")):
        value = value.replace(src, dst)
    return value.strip()


def _salvage_agent_text(raw: str) -> str:
    """Best-effort extraction of the model's actual answer when the strict JSON
    parser inside homellm failed.

    Thinking models (e.g. Qwen3) emit `<think>...</think>` and can produce JSON
    that is truncated or has stray braces inside reasoning, which trips the strict
    parser. Rather than showing a generic error, we recover the `assistant_message`
    (or, failing that, the plain prose) so the user still gets a useful reply.
    """
    if not raw:
        return ""

    # Strip reasoning blocks (closed and unclosed).
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    start = cleaned.find("{")
    if start != -1:
        end = cleaned.rfind("}")
        # Prefer a balanced slice for a strict parse; otherwise use the tail so we
        # can still regex out fields from JSON truncated by max_tokens.
        candidate = cleaned[start:end + 1] if end > start else cleaned[start:]
        if end > start:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    msg = obj.get("assistant_message")
                    if isinstance(msg, str) and msg.strip():
                        return msg.strip()
            except Exception:
                pass
        # Regex fallback: pull assistant_message even from broken/truncated JSON.
        m = re.search(r'"assistant_message"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate)
        if m and m.group(1).strip():
            return _unescape_json_string(m.group(1))
        m = re.search(r'"assistant_message"\s*:\s*"((?:[^"\\]|\\.)*)', candidate)
        if m and m.group(1).strip():
            return _unescape_json_string(m.group(1))

    # No usable JSON — return the cleaned prose if it isn't JSON scaffolding.
    if cleaned and "assistant_message" not in cleaned and "tool_calls" not in cleaned:
        return cleaned
    return ""


class ServerStartRequest(BaseModel):
    model_repo: str = "unsloth/Qwen3.5-9B-GGUF"
    quant: str = "9B-UD-Q4_K_XL"
    ctx_size: int = 262144
    gpu_layers: int = -1
    temperature: float = 0.7


class ChatRequest(BaseModel):
    session_id: str
    message: str


@router.post("/start-server")
def start_server(req: ServerStartRequest):
    """Start llama-server with specified model."""
    with _model_lock:
        global _server_backend, _agent_temperature, _server_model_path, _server_ctx
        try:
            from homellm.app.llama_server_backend import LlamaServerBackend
            from homellm.app.hf_gguf import download_gguf_to_models_dir

            models_dir = PROJECT_ROOT / "models"
            model_path = download_gguf_to_models_dir(
                models_dir=models_dir,
                repo_id=req.model_repo,
                quant=req.quant,
            )

            if _server_backend:
                try:
                    _server_backend.stop()
                except Exception:
                    pass

            _server_backend = LlamaServerBackend(
                model_path=str(model_path),
                n_ctx=req.ctx_size,
                n_gpu_layers=req.gpu_layers,
            )
            _agent_temperature = req.temperature
            _server_model_path = str(model_path)
            _server_ctx = int(req.ctx_size)
            return {"success": True, "model_path": str(model_path)}
        except Exception as e:
            raise HTTPException(500, str(e))


@router.post("/stop-server")
def stop_server():
    """Stop llama-server."""
    with _model_lock:
        global _server_backend, _server_model_path
        if _server_backend:
            try:
                _server_backend.stop()
            except Exception:
                pass
            _server_backend = None
            _server_model_path = None
        return {"success": True}


@router.get("/server-status")
def server_status():
    """Check llama-server status."""
    if _server_backend is None:
        return {"running": False}
    try:
        req = urllib.request.Request("http://127.0.0.1:8787/health")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            running = resp.status == 200
        return {"running": running, "model": _server_model_path}
    except Exception:
        return {"running": False}


def _fallback_title(message: str) -> str:
    """Derive a readable title straight from the user's first message."""
    text = " ".join((message or "").split())
    if not text:
        return "Новый чат"
    return text[:40].rstrip() + "…" if len(text) > 40 else text


def _clean_title(raw: str) -> str:
    """Strip reasoning/quotes/noise from an LLM-generated title."""
    if not raw:
        return ""
    t = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    t = re.sub(r"<think>.*$", "", t, flags=re.DOTALL)
    t = t.strip()
    if not t:
        return ""
    t = t.splitlines()[0].strip()
    t = t.strip('"').strip("'").strip("«»").strip()
    # Drop common prefixes the model may add.
    t = re.sub(r"^(заголовок|title|тема|chat)\s*[:\-–]\s*", "", t, flags=re.IGNORECASE).strip()
    if len(t) > 50:
        t = t[:50].rstrip() + "…"
    return t


def _generate_session_title(first_message: str) -> str:
    """Ask the running model for a short chat title; fall back to the message."""
    fallback = _fallback_title(first_message)
    if _server_backend is None:
        return fallback
    try:
        raw = _server_backend.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Придумай очень короткий заголовок (максимум 5 слов) для диалога "
                        "на языке пользователя. Ответь ТОЛЬКО заголовком — без кавычек, без "
                        "размышлений, без префиксов вроде 'Заголовок:'."
                    ),
                },
                {"role": "user", "content": (first_message or "")[:2000]},
            ],
            max_tokens=512,
            temperature=0.3,
            top_p=0.95,
            top_k=40,
        )
    except Exception:
        return fallback
    return _clean_title(raw) or fallback


def _is_default_name(name: str) -> bool:
    return not name or name.strip().lower().startswith(("session ", "новый чат"))


def _normalize_session(data: dict, file_id: str) -> dict:
    """Coerce a stored session (which may use the legacy Streamlit shape with
    `session_id`/`updated_at` and no `name`) into the canonical contract the UI
    expects: `{id, name, messages:[{role, content:str}], created_at}`.

    The `id` is anchored to the file stem so delete/chat routes resolve the same
    file regardless of the fields stored inside.
    """
    sid = str(data.get("id") or data.get("session_id") or file_id)
    name = data.get("name") or f"Session {sid[:6]}"
    messages = []
    for m in data.get("messages") or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else json.dumps(content, ensure_ascii=False)
        messages.append({"role": str(m.get("role", "assistant")), "content": content})
    return {
        "id": file_id,
        "name": name,
        "messages": messages,
        "created_at": data.get("created_at") or data.get("updated_at"),
    }


@router.get("/sessions")
def list_sessions():
    """List all agent sessions, newest/most-recently-active first.

    Sorted by file mtime descending so that freshly created or freshly used
    chats appear leftmost in the UI.
    """
    items = []
    for f in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            mtime = f.stat().st_mtime
        except Exception:
            continue
        items.append((mtime, _normalize_session(data, f.stem)))
    items.sort(key=lambda x: x[0], reverse=True)
    return {"sessions": [s for _, s in items]}


@router.post("/sessions")
def create_session():
    """Create a new agent session."""
    session_id = uuid.uuid4().hex[:12]
    session = {
        "id": session_id,
        "name": f"Session {session_id[:6]}",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    atomic_json(safe_id(SESSIONS_DIR, session_id).with_suffix(".json"), session)
    return session


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete an agent session."""
    session_file = safe_id(SESSIONS_DIR, session_id).with_suffix(".json")
    with _model_lock:
        session_file.unlink(missing_ok=True)
    return {"success": True}


@router.get("/tools")
def agent_tools():
    """Return the real tool specs/groups the agent can call."""
    try:
        from homellm.app.agent_tools import get_tool_groups, get_tool_specs

        return {"groups": get_tool_groups(), "tools": get_tool_specs()}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/capabilities")
def agent_capabilities():
    """Return the agent training capabilities overview."""
    try:
        from homellm.app.agent_tools import list_training_capabilities

        return {"capabilities": list_training_capabilities()}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/runs")
def agent_runs(status: str = "all"):
    """List agent-initiated (and other) training runs with live status."""
    try:
        from homellm.app.agent_tools import list_runs

        return list_runs(status=status)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/runs/{run_id}/stop")
def agent_stop_run(run_id: str):
    """Stop a run started by the agent."""
    try:
        from homellm.app.agent_tools import stop_run

        return stop_run(run_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/chat")
def agent_chat(req: ChatRequest):
    """Chat with the agent. Returns SSE stream."""
    if _server_backend is None:
        raise HTTPException(400, "Agent server is not running")

    session_file = safe_id(SESSIONS_DIR, req.session_id).with_suffix(".json")
    if not session_file.exists():
        raise HTTPException(404, "Session not found")

    return StreamingResponse(
        _run_agent_stream(req, session_file),
        media_type="text/event-stream",
    )


def _run_agent_stream(req: ChatRequest, session_file: Path):
    """Run agent turn with streaming response."""
    with _model_lock:
        try:
            session = _normalize_session(json.loads(session_file.read_text()), session_file.stem)
            session["messages"].append({"role": "user", "content": req.message})
            atomic_json(session_file, session)
            from homellm.app.agent_runtime import run_agent_turn

            text, trace = run_agent_turn(
                backend=_AgentBackend(_server_backend),
                conversation=session["messages"],
                max_steps=6,
                max_tokens=_recommended_output_tokens(_server_ctx),
                temperature=_agent_temperature,
                top_p=0.95,
                top_k=40,
            )

            # If the strict parser inside homellm failed, salvage the model's real
            # answer from the last raw response instead of returning the stub error.
            parse_failed = any(
                isinstance(s, dict)
                and isinstance(s.get("parsed"), dict)
                and s["parsed"].get("parse_error")
                for s in (trace or [])
            )
            if parse_failed and trace:
                salvaged = _salvage_agent_text(str(trace[-1].get("raw_response", "")))
                if salvaged:
                    text = salvaged

            session["messages"].append({"role": "assistant", "content": text})

            # Auto-name the chat after the first user turn using the model itself,
            # so the session tab is understandable instead of "Session ab12cd".
            user_messages = [m for m in session["messages"] if m.get("role") == "user"]
            if len(user_messages) == 1 and _is_default_name(session.get("name", "")):
                title = _generate_session_title(user_messages[0].get("content", ""))
                session["name"] = title

            atomic_json(session_file, session)

            for item in trace or []:
                tool_name = item.get("tool") or item.get("name")
                if tool_name:
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': item.get('arguments', {})})}\n\n"
                for tool_result in item.get("tool_results", []) or []:
                    tool_name = tool_result.get("tool") or tool_result.get("name")
                    if tool_name:
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': tool_result.get('arguments', {})})}\n\n"

            for i in range(0, len(text), 20):
                chunk = text[i:i+20]
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"


        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield "data: [DONE]\n\n"
