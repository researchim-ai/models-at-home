"""Prebuilt llama.cpp server backend for Agent Studio."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / ".runs"
LLAMA_SERVER_DIR = RUNS_DIR / "agent_llama_server"
LLAMA_SERVER_BIN_DIR = LLAMA_SERVER_DIR / "bin"
LLAMA_SERVER_LOG = LLAMA_SERVER_DIR / "server.log"
LLAMA_SERVER_PID = LLAMA_SERVER_DIR / "server.pid"
LLAMA_SERVER_VARIANT = LLAMA_SERVER_DIR / ".variant"
LLAMA_HOST = "127.0.0.1"
LLAMA_PORT = 8787
GITHUB_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"


def is_llama_server_supported() -> bool:
    return platform.system().lower() in {"linux", "darwin", "windows"}


def _http_json(url: str, *, timeout: float = 20.0, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "models-at-home-agent/0.1"})
    with urllib.request.urlopen(req, timeout=120) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
        return
    if archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)
        return
    raise ValueError(f"Unsupported archive type: {archive_path.name}")


def _server_bin_name() -> str:
    return "llama-server.exe" if platform.system().lower() == "windows" else "llama-server"


def _find_server_bin_in(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    direct = root / _server_bin_name()
    if direct.exists():
        return direct
    for nested in root.rglob(_server_bin_name()):
        if nested.is_file():
            return nested
    return None


def _find_server_bin() -> Optional[Path]:
    return _find_server_bin_in(LLAMA_SERVER_BIN_DIR)


def _detect_variant_candidates(backend_name: Optional[str] = None) -> List[str]:
    explicit_variant = os.environ.get("MAH_LLAMA_SERVER_VARIANT", "").strip()
    if explicit_variant:
        return [explicit_variant]

    backend_name = _resolve_backend_name(backend_name)
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        return ["macos-arm64" if "arm" in machine else "macos-x64"]
    if system == "windows":
        return ["win-cpu-arm64" if "arm" in machine else "win-cpu-x64"]

    if backend_name == "vulkan":
        return ["ubuntu-vulkan-x64", "ubuntu-x64"]
    if backend_name in {"hip", "rocm"}:
        return ["ubuntu-rocm-7.2-x64", "ubuntu-x64"]

    # On Linux, CPU is the universal fallback.
    return ["ubuntu-x64"]


def _get_latest_release() -> Dict[str, Any]:
    return _http_json(
        GITHUB_RELEASE_API,
        timeout=30.0,
        headers={"User-Agent": "models-at-home-agent/0.1", "Accept": "application/json"},
    )


def _match_asset(assets: List[Dict[str, Any]], variant: str) -> Optional[Dict[str, Any]]:
    for asset in assets:
        name = str(asset.get("name", ""))
        if f"-bin-{variant}." in name:
            return asset
    return None


def ensure_binary(backend_name: Optional[str] = None) -> Path:
    resolved_backend = _resolve_backend_name(backend_name)
    existing = _find_server_bin()
    desired_variants = _detect_variant_candidates(resolved_backend)
    current_variant = ""
    if LLAMA_SERVER_VARIANT.exists():
        try:
            current_variant = LLAMA_SERVER_VARIANT.read_text(encoding="utf-8").strip()
        except Exception:
            current_variant = ""
    if existing and current_variant == desired_variants[0]:
        return existing

    release = _get_latest_release()
    assets = release.get("assets", []) or []
    variants = list(desired_variants)
    if "ubuntu-x64" not in variants and platform.system().lower() == "linux":
        variants.append("ubuntu-x64")

    last_error: Optional[str] = None
    for variant in variants:
        asset = _match_asset(assets, variant)
        if not asset:
            last_error = f"llama.cpp release asset not found for variant {variant}"
            continue
        archive_name = str(asset["name"])
        archive_url = str(asset["browser_download_url"])
        archive_path = LLAMA_SERVER_DIR / archive_name
        try:
            if LLAMA_SERVER_BIN_DIR.exists():
                shutil.rmtree(LLAMA_SERVER_BIN_DIR, ignore_errors=True)
            LLAMA_SERVER_BIN_DIR.mkdir(parents=True, exist_ok=True)
            _download_file(archive_url, archive_path)
            _extract_archive(archive_path, LLAMA_SERVER_BIN_DIR)
            archive_path.unlink(missing_ok=True)
            server_bin = _find_server_bin()
            if not server_bin:
                raise FileNotFoundError("llama-server not found after extraction")
            if platform.system().lower() != "windows":
                server_bin.chmod(0o755)
            LLAMA_SERVER_VARIANT.write_text(variant, encoding="utf-8")
            return server_bin
        except Exception as exc:
            last_error = str(exc)
            try:
                archive_path.unlink(missing_ok=True)
            except Exception:
                pass

    raise RuntimeError(f"Не удалось установить llama-server: {last_error or 'unknown error'}")


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _kill_existing_server() -> None:
    if LLAMA_SERVER_PID.exists():
        try:
            pid = int(LLAMA_SERVER_PID.read_text(encoding="utf-8").strip())
            os.kill(pid, 15)
            time.sleep(1.0)
        except Exception:
            pass
        try:
            LLAMA_SERVER_PID.unlink(missing_ok=True)
        except Exception:
            pass


def _ctx_batch_args(ctx_size: int) -> List[str]:
    if ctx_size >= 131072:
        return ["--batch-size", "4096", "--ubatch-size", "1024"]
    if ctx_size >= 65536:
        return ["--batch-size", "4096"]
    if ctx_size >= 32768:
        return ["--batch-size", "2048"]
    return []


def _installed_backend_name() -> str:
    return (os.environ.get("LLAMA_CPP_BACKEND", "vulkan") or "vulkan").strip().lower()


def _resolve_backend_name(backend_name: Optional[str] = None) -> str:
    requested = str(backend_name or _installed_backend_name()).strip().lower()
    if requested in {"", "auto"}:
        if platform.system().lower() == "linux":
            return "vulkan"
        return "cpu"
    return requested


def _gpu_env_overrides(backend_name: str, visible_devices: Optional[List[int]]) -> Dict[str, str]:
    if not visible_devices:
        return {}

    device_value = ",".join(str(int(device_id)) for device_id in visible_devices)
    backend_name = str(backend_name or "cpu").lower()

    if backend_name == "cuda":
        return {"CUDA_VISIBLE_DEVICES": device_value}
    if backend_name in {"hip", "rocm"}:
        return {
            "HIP_VISIBLE_DEVICES": device_value,
            "ROCR_VISIBLE_DEVICES": device_value,
        }
    if backend_name == "vulkan":
        return {"GGML_VK_VISIBLE_DEVICES": device_value}
    return {}


def _device_args(backend_name: str, visible_devices: List[int]) -> List[str]:
    if not visible_devices:
        return []

    backend_name = str(backend_name or "cpu").lower()
    if backend_name == "vulkan":
        names = [f"Vulkan{device_id}" for device_id in visible_devices]
    elif backend_name == "cuda":
        names = [f"CUDA{device_id}" for device_id in visible_devices]
    else:
        return []

    args = ["--device", ",".join(names)]
    args.extend(["--split-mode", "none" if len(visible_devices) == 1 else "layer"])
    return args


class LlamaServerBackend:
    """Use a standalone llama-server process rather than in-process bindings."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        visible_devices: Optional[List[int]] = None,
        backend_name: Optional[str] = None,
        verbose: bool = False,
        **_: Any,
    ) -> None:
        self.model_path = str(model_path)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_batch = int(n_batch)
        self.n_threads = int(n_threads or max(1, (os.cpu_count() or 4) // 2))
        self.visible_devices = [int(device_id) for device_id in (visible_devices or [])]
        self.backend_name = _resolve_backend_name(backend_name)
        self.verbose = verbose
        self.server_bin = ensure_binary(self.backend_name)
        self._start_server()

    def _start_server(self) -> None:
        LLAMA_SERVER_DIR.mkdir(parents=True, exist_ok=True)
        _kill_existing_server()

        if _port_open(LLAMA_HOST, LLAMA_PORT):
            logger.info("Port %s already in use; continuing to health check", LLAMA_PORT)

        cmd = [
            str(self.server_bin),
            "--model",
            self.model_path,
            "--host",
            LLAMA_HOST,
            "--port",
            str(LLAMA_PORT),
            "--jinja",
            "--n-gpu-layers",
            str(self.n_gpu_layers),
            "--ctx-size",
            str(self.n_ctx),
            "--threads",
            str(self.n_threads),
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--parallel",
            "1",
            "--kv-unified",
            "--cont-batching",
        ]
        cmd.extend(_ctx_batch_args(self.n_ctx))
        if self.n_gpu_layers != 0:
            cmd.append("--kv-offload")
            cmd.extend(_device_args(self.backend_name, self.visible_devices))
        if platform.system().lower() != "windows":
            cmd.append("--mlock")
        if self.n_gpu_layers != 0:
            cmd.extend(["--flash-attn", "on"])

        env = os.environ.copy()
        env.update(_gpu_env_overrides(self.backend_name, self.visible_devices if self.n_gpu_layers != 0 else []))

        logger.info("Starting llama-server: %s", " ".join(cmd))
        with open(LLAMA_SERVER_LOG, "a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
                env=env,
            )

        LLAMA_SERVER_PID.write_text(str(process.pid), encoding="utf-8")
        self._wait_ready()

    def _wait_ready(self, timeout_secs: int = 180) -> None:
        deadline = time.time() + timeout_secs
        last_error = ""
        while time.time() < deadline:
            try:
                health = _http_json(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health", timeout=3.0)
                if health.get("status") == "ok":
                    return
                if health.get("status") == "loading model":
                    time.sleep(1.5)
                    continue
            except Exception as exc:
                last_error = str(exc)
            time.sleep(1.5)

        tail = ""
        if LLAMA_SERVER_LOG.exists():
            tail = "\n".join(LLAMA_SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise RuntimeError(
            "llama-server не поднялся вовремя. "
            f"Последняя ошибка health: {last_error or 'unknown'}. "
            f"Последний лог:\n{tail}"
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
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "n_predict": int(max_tokens),
            "temperature": float(temperature),
            "stream": bool(stream),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://{LLAMA_HOST}:{LLAMA_PORT}/completion",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server completion error: HTTP {exc.code} {body}") from exc

        result = json.loads(raw)
        if isinstance(result, dict):
            if "content" in result:
                return str(result["content"])
            if "response" in result:
                return str(result["response"])
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    if "text" in first:
                        return str(first["text"])
                    message = first.get("message")
                    if isinstance(message, dict) and "content" in message:
                        return str(message["content"])
        return ""

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = 40,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": bool(stream),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if stop:
            payload["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server chat error: HTTP {exc.code} {body}") from exc

        result = json.loads(raw)
        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                message = first.get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
        return ""

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
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
        return False

    def stop(self) -> None:
        _kill_existing_server()
