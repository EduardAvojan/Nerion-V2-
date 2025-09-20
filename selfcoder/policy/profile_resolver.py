from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


_DEFAULT_PATH = Path("config/profiles.yaml")


@dataclass
class ProfileDecision:
    name: str
    env: Dict[str, str]
    why: str


def _load_profiles(path: Path = _DEFAULT_PATH) -> Dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("profiles", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _to_env_from_profile(p: Dict[str, Any]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not isinstance(p, dict):
        return env
    # Policy profile (NERION_POLICY)
    pol = str(p.get("policy") or "").strip()
    if pol:
        env["NERION_POLICY"] = pol
    # Reviewer gates
    rv = p.get("reviewer") or {}
    if isinstance(rv, dict):
        if rv.get("strict") is True:
            env["NERION_REVIEW_STRICT"] = "1"
        style_max = rv.get("style_max")
        if style_max is not None:
            env["NERION_REVIEW_STYLE_MAX"] = str(style_max)
        ex = rv.get("externals") or {}
        if isinstance(ex, dict):
            if ex.get("ruff") is True:
                env["NERION_REVIEW_RUFF"] = "1"
            if ex.get("pydocstyle") is True:
                env["NERION_REVIEW_PYDOCSTYLE"] = "1"
            if ex.get("mypy") is True:
                env["NERION_REVIEW_MYPY"] = "1"
    # Runner knobs
    rn = p.get("runner") or {}
    if isinstance(rn, dict):
        if rn.get("inprocess_pytest") is True:
            env["NERION_BENCH_USE_LIBPYTEST"] = "1"
        if rn.get("coverage_assist") is True:
            env["NERION_BENCH_COVERAGE_ASSIST"] = "1"
    # Timeouts (bench pytest)
    to = p.get("timeouts") or {}
    if isinstance(to, dict):
        if to.get("pytest") is not None:
            env["NERION_BENCH_PYTEST_TIMEOUT"] = str(int(to.get("pytest")))
    # LLM model pinning (optional)
    llm = p.get("llm") or {}
    if isinstance(llm, dict):
        chat = llm.get("chat") or {}
        if isinstance(chat, dict):
            m = str(chat.get("model") or "").strip()
            if m:
                env["NERION_LLM_MODEL"] = m
        coder = llm.get("coder") or {}
        if isinstance(coder, dict):
            be = str(coder.get("backend") or "").strip()
            if be:
                env["NERION_CODER_BACKEND"] = be
            m = str(coder.get("model") or "").strip()
            if m:
                env["NERION_CODER_MODEL"] = m
            base = str(coder.get("base_url") or "").strip()
            if base:
                env["NERION_CODER_BASE_URL"] = base
    # Network gate (optional): map net.allow -> NERION_ALLOW_NETWORK=1 when true
    net = p.get("net") or {}
    if isinstance(net, dict) and bool(net.get("allow", False)):
        env.setdefault("NERION_ALLOW_NETWORK", "1")
    return env


def decide(task_type: str, *, preview: Optional[Dict[str, Any]] = None, signals: Optional[Dict[str, Any]] = None, profiles_path: Path = _DEFAULT_PATH) -> ProfileDecision:
    task = (task_type or "").strip().lower()
    profs = _load_profiles(profiles_path)
    sig = signals or {}
    # Sticky per-task override
    try:
        prefs = _lp()
        sticky = (prefs.get('profile_overrides') or {}).get(task)
        if sticky and sticky in profs:
            p = profs.get(sticky, {})
            return ProfileDecision(name=sticky, env=_to_env_from_profile(p), why="sticky_override")
    except Exception:
        pass
    # Deterministic rules
    if task in {"bench", "bench_repair", "repair", "bench-repair"} and "bench-recommended" in profs:
        p = profs.get("bench-recommended", {})
        return ProfileDecision(name="bench-recommended", env=_to_env_from_profile(p), why="task=bench_repair")
    # Security critical/high → safe
    sec_findings = int(sig.get("security_findings", 0) or 0)
    if sec_findings > 0 and "safe" in profs:
        p = profs.get("safe", {})
        return ProfileDecision(name="safe", env=_to_env_from_profile(p), why="security_findings>0")
    # Rename with large import graph → balanced
    if (sig.get("has_rename") and int(sig.get("import_graph_breadth", 0)) >= 50) and "balanced" in profs:
        p = profs.get("balanced", {})
        return ProfileDecision(name="balanced", env=_to_env_from_profile(p), why="rename+large_graph")
    # AST-only + small diff + no security → fast
    kinds_ast_only = bool(sig.get("kinds_ast_only", False))
    files_count = int(sig.get("files_count", 0) or 0)
    delta_bytes = int(sig.get("delta_bytes", 0) or 0)
    if kinds_ast_only and sec_findings == 0 and files_count <= 2 and abs(delta_bytes) < 800 and "fast" in profs:
        p = profs.get("fast", {})
        return ProfileDecision(name="fast", env=_to_env_from_profile(p), why="ast_small_change")
    # Network intent → balanced or safe (domains scoped unavailable)
    if bool(sig.get("network_intent", False)) and "balanced" in profs:
        p = profs.get("balanced", {})
        why = "network_intent"
        if not bool(sig.get("domains_scoped", True)) and "safe" in profs:
            p = profs.get("safe", {})
            why = "network_unscoped"
        return ProfileDecision(name=p.get("policy", "balanced"), env=_to_env_from_profile(p), why=why)
    # Score-based fallback (fast vs safe)
    low_risk = 1.0 if sec_findings == 0 else 0.0
    small_change = 1.0 if (files_count <= 2 and abs(delta_bytes) < 800) else 0.0
    offline = 1.0 if not bool(sig.get("network_intent", False)) else 0.0
    passing = 1.0 if bool(sig.get("tests_passing", False)) else 0.0
    cross_file = 1.0 if files_count > 1 else 0.0
    user_safe_bias = 1.0 if bool(sig.get("user_safety_bias", False)) else 0.0
    score_fast = low_risk + small_change + offline + passing
    score_safe = (sec_findings > 0) * 2.0 + cross_file + user_safe_bias
    margin = score_fast - score_safe
    if margin > 0.5 and "fast" in profs:
        p = profs.get("fast", {})
        return ProfileDecision(name="fast", env=_to_env_from_profile(p), why=f"score_fast={score_fast:.2f} > safe={score_safe:.2f}")
    if margin < -0.5 and "safe" in profs:
        p = profs.get("safe", {})
        return ProfileDecision(name="safe", env=_to_env_from_profile(p), why=f"score_safe={score_safe:.2f} > fast={score_fast:.2f}")
    # Bandit fallback: prefer historically better profile on small margin
    try:
        prefs = _lp()
        stats = ((prefs.get('profile_success') or {}).get(task) or {})
        best = None
        best_rate = -1.0
        for name in ('fast', 'balanced', 'safe'):
            s = stats.get(name) or {}
            ok = float(s.get('ok') or 0.0)
            tot = float(s.get('total') or 0.0)
            rate = (ok / tot) if tot > 0 else -1.0
            if rate > best_rate:
                best_rate = rate
                best = name
        if best and best in profs:
            p = profs.get(best, {})
            return ProfileDecision(name=best, env=_to_env_from_profile(p), why=f"bandit_best={best}:{best_rate:.2f}")
    except Exception:
        pass
    # Default balanced
    if "balanced" in profs:
        p = profs.get("balanced", {})
        return ProfileDecision(name="balanced", env=_to_env_from_profile(p), why="margin_small_or_default")
    return ProfileDecision(name="", env={}, why="no-profiles")

try:
    from selfcoder.learning.continuous import load_prefs as _lp, save_prefs as _sp
except Exception:
    def _lp():
        return {}
    def _sp(p):
        Path('out/learning').mkdir(parents=True, exist_ok=True)
        (Path('out/learning')/ 'prefs.json').write_text(str(p))

def apply_env(decision: ProfileDecision, *, override_existing: bool = False) -> None:
    for k, v in (decision.env or {}).items():
        if override_existing or (os.getenv(k) is None or str(os.getenv(k)).strip() == ""):
            os.environ[k] = str(v)


class _EnvScope:
    def __init__(self, env: Dict[str, str]) -> None:
        self.env = env
        self._prev: Dict[str, Optional[str]] = {}

    def __enter__(self):
        for k, v in (self.env or {}).items():
            self._prev[k] = os.getenv(k)
            # Guardrail: do not relax existing settings; only set when unset/empty
            prev = self._prev[k]
            if prev is None or str(prev).strip() == "":
                os.environ[k] = str(v)
        return self

    def __exit__(self, exc_type, exc, tb):
        for k, prev in self._prev.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def apply_env_scoped(decision: ProfileDecision) -> _EnvScope:
    return _EnvScope(decision.env or {})
