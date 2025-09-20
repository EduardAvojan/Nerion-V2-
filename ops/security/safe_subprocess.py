# ops/security/safe_subprocess.py
from __future__ import annotations
import shutil
import subprocess
import os
from typing import Mapping, Optional

class SubprocessPolicyError(RuntimeError): ...
class ExecutableNotFound(RuntimeError): ...

def safe_run(
    argv: list[str],
    *,
    cwd: Optional[str] = None,
    timeout: int = 60,
    env: Optional[Mapping[str, str]] = None,
    inherit_env: bool = False,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Strict subprocess runner:
    - Requires argv list (no shell strings), forbids shell=True.
    - Enforces executable resolution via PATH.
    - Applies a timeout (default 60s).
    - Uses empty env unless explicitly passed.
    - When inherit_env=True, the child process environment will be os.environ merged with env (if provided).
      This allows inheriting the parent environment safely while overriding or adding variables.
    """
    if not isinstance(argv, list) or not argv:
        raise SubprocessPolicyError("safe_run requires argv: list[str] with at least one element")
    exe = argv[0]
    if shutil.which(exe) is None:
        raise ExecutableNotFound(f"Executable not found on PATH: {exe}")

    # NEVER expose shell=True; callers cannot override this
    return subprocess.run(
        argv,
        cwd=cwd,
        timeout=timeout,
        env=(
            {**os.environ, **dict(env)} if inherit_env and env is not None
            else (os.environ.copy() if inherit_env else dict(env or {}))
        ),
        check=check,
        capture_output=capture_output,
    )