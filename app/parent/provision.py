"""Provision local coder models with user consent.

Behavior:
- Checks if (backend, model) is available locally using selector probes.
- If missing and network is not allowed, returns a consent message.
- If network allowed (NERION_ALLOW_NETWORK=1), performs best-effort provisioning:
  - ollama: `ollama pull <model>`
  - llama_cpp: requires LLAMA_CPP_MODEL_URL -> downloads to LLAMA_CPP_MODEL_PATH
  - vllm/exllamav2: print instructions (serve or set directories)
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import os
import time


def _run(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 1800) -> tuple[int, str]:
    try:
        import subprocess
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)
        out = (p.stdout or "") + ("\n" + (p.stderr or ""))
        return p.returncode, out
    except Exception as e:
        return 1, f"spawn-error: {e}"


def _http_ok(url: str, timeout: int = 3) -> bool:
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout)
        return bool(getattr(r, 'ok', False))
    except Exception:
        return False


def _spawn_vllm_server(repo: str, host: str = "127.0.0.1", port: int = 8000) -> tuple[bool, str]:
    # Launch vLLM OpenAI server in background; write PID under .nerion
    try:
        import subprocess
        import sys
        work = Path.cwd()
        meta = work / ".nerion"
        meta.mkdir(parents=True, exist_ok=True)
        pidfile = meta / f"vllm_{port}.pid"
        if pidfile.exists():
            try:
                pid = int(pidfile.read_text().strip())
                # Assume it's fine if endpoint responds
                if _http_ok(f"http://{host}:{port}/v1/models"):
                    return True, f"vLLM server already running at http://{host}:{port} (pid {pid})"
            except Exception:
                pass
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", repo, "--host", host, "--port", str(port)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pidfile.write_text(str(proc.pid))
        # Poll readiness
        base = f"http://{host}:{port}"
        for _ in range(50):
            if _http_ok(base + "/v1/models"):
                return True, base
            time.sleep(0.2)
        return False, "vLLM server did not become ready"
    except Exception as e:
        return False, f"vLLM spawn-error: {e}"


def _ensure_exl_dir_from_catalog(model: str) -> Optional[str]:
    try:
        from app.parent.model_catalog import resolve  # type: ignore
        src = resolve("exllamav2", model)
        repo = src.get("repo") if isinstance(src, dict) else None
        if not repo:
            return None
        dest = Path.home() / f".cache/nerion/models/exl/{model}"
        if dest.exists():
            return str(dest)
        rc, out = _run(["git", "lfs", "clone", f"https://huggingface.co/{repo}", str(dest)])
        if rc == 0 and dest.exists():
            return str(dest)
        return None
    except Exception:
        return None


def check_available(backend: str, model: str) -> bool:
    try:
        from app.parent.selector import probe_ollama, probe_vllm, probe_llama_cpp, probe_exllamav2
        if backend == "ollama":
            return model in set(probe_ollama())
        if backend == "vllm":
            return model in set(probe_vllm())
        if backend == "llama_cpp":
            return len(probe_llama_cpp()) > 0
        if backend == "exllamav2":
            return len(probe_exllamav2()) > 0
    except Exception:
        return False
    return False


def ensure_available(backend: str, model: str) -> tuple[bool, str]:
    """Ensure the requested backend/model is available.

    Returns (ok, message). When ok=False and network is required, message contains
    a human-readable consent prompt describing the action to be taken.
    """
    if check_available(backend, model):
        return True, f"{backend}:{model} ready"

    allow_net = (os.getenv("NERION_ALLOW_NETWORK") or "").strip().lower() in {"1","true","yes","on"}
    if not allow_net:
        if backend == "ollama":
            return False, (
                f"I can pull '{model}' for Ollama locally. Grant network to run: 'ollama pull {model}'.\n"
                f"Set NERION_ALLOW_NETWORK=1 and rerun, or execute the pull manually."
            )
        if backend == "llama_cpp":
            url = os.getenv("LLAMA_CPP_MODEL_URL") or "<set LLAMA_CPP_MODEL_URL>"
            path = os.getenv("LLAMA_CPP_MODEL_PATH") or "<set LLAMA_CPP_MODEL_PATH>"
            return False, (
                f"I can download a GGUF for llama.cpp. Grant network to fetch from {url} to {path}.\n"
                f"Set NERION_ALLOW_NETWORK=1 and provide LLAMA_CPP_MODEL_URL and LLAMA_CPP_MODEL_PATH."
            )
        if backend == "vllm":
            return False, (
                "Start your vLLM server locally with the desired model and set NERION_CODER_BASE_URL."
            )
        if backend == "exllamav2":
            return False, (
                "Provide EXLLAMA_MODEL_DIR pointing to a local model directory for exllamav2."
            )
        return False, "Model not available and backend-specific instructions required."

    # Network allowed: best-effort provisioning
    if backend == "ollama":
        rc, out = _run(["ollama", "pull", model])
        ok = (rc == 0)
        return ok, (out if ok else f"ollama pull failed: {out}")
    if backend == "llama_cpp":
        url = os.getenv("LLAMA_CPP_MODEL_URL")
        path = os.getenv("LLAMA_CPP_MODEL_PATH")
        if not url:
            # Try to resolve from catalog
            try:
                from app.parent.model_catalog import resolve  # type: ignore
                src = resolve("llama_cpp", model)
            except Exception:
                src = None
            if src and isinstance(src.get("url"), str):
                url = src["url"]
                if not path:
                    filename = src.get("filename") or os.path.basename(url)
                    path = os.path.expanduser(f"~/.cache/nerion/models/llama.cpp/{filename}")
                    os.environ.setdefault("LLAMA_CPP_MODEL_PATH", path)
        if not url or not path:
            return False, "Set LLAMA_CPP_MODEL_URL and LLAMA_CPP_MODEL_PATH (or use catalog) to download GGUF"
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import requests  # type: ignore
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True, f"Downloaded GGUF to {dest}"
        except Exception as e:
            return False, f"download-error: {e}"
    if backend == "vllm":
        # Autostart if requested
        autostart = (os.getenv("NERION_AUTOSTART_VLLM") or "").strip().lower() in {"1","true","yes","on"}
        repo = None
        try:
            from app.parent.model_catalog import resolve  # type: ignore
            src = resolve("vllm", model)
            repo = src.get("repo") if isinstance(src, dict) else None
        except Exception:
            repo = None
        base_url = os.getenv("NERION_CODER_BASE_URL") or "http://127.0.0.1:8000"
        if autostart and repo:
            # If not ready, spawn
            if not _http_ok(base_url + "/v1/models"):
                ok, msg = _spawn_vllm_server(repo, host=base_url.split("://",1)[-1].split(":")[0], port=int(base_url.rsplit(":",1)[-1]))
                if ok and msg.startswith("http"):
                    os.environ.setdefault("NERION_CODER_BASE_URL", msg)
                    return True, f"vLLM started at {msg}"
                elif ok:
                    return False, msg
                else:
                    return False, f"vLLM autostart failed: {msg}"
        # Otherwise, instruct user
        cmd = f"python -m vllm.entrypoints.openai.api_server --model {repo or '<hf_repo>'} --host 127.0.0.1 --port 8000"
        return False, ("Start a local vLLM server for this model, e.g.:\n" + cmd + "\nThen set NERION_CODER_BASE_URL to http://127.0.0.1:8000")
    if backend == "exllamav2":
        # Autoprep model dir if requested
        autostart = (os.getenv("NERION_AUTOSTART_EXL") or "").strip().lower() in {"1","true","yes","on"}
        if autostart and not os.getenv("EXLLAMA_MODEL_DIR"):
            path = _ensure_exl_dir_from_catalog(model)
            if path:
                os.environ.setdefault("EXLLAMA_MODEL_DIR", path)
                return True, f"EXL model prepared at {path}"
        # Otherwise instruct
        try:
            from app.parent.model_catalog import resolve  # type: ignore
            src = resolve("exllamav2", model)
            repo = src.get("repo") if isinstance(src, dict) else None
        except Exception:
            repo = None
        msg = "Provide EXLLAMA_MODEL_DIR pointing to a local model directory."
        if repo:
            msg += f"\nExample: git lfs clone https://huggingface.co/{repo} ~/.cache/nerion/models/exl/{model}"
        return False, msg
    return False, "Unsupported backend"
