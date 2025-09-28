"""Continuous learning utilities.

Reviews recent experience logs and artifacts to produce a compact summary and
update a local prefs file that can bias future runs. This is intentionally
simple and offline.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
import time
import io
import tempfile
from typing import Dict, Any, List, Tuple, DefaultDict, Optional
from collections import defaultdict


PREFS_PATH = Path('out/learning/prefs.json')
GLOBAL_PREFS_PATH = Path.home() / '.nerion' / 'prefs_global.json'
LOG_PATH = Path('out/experience/log.jsonl')

# Schema
_SCHEMA_VERSION = 3

try:
    from .abseq import MSPRT as _MSPRT
    from .guardrails import breached as _guard_breached
except Exception:  # pragma: no cover
    _MSPRT = None  # type: ignore
    def _guard_breached(_m):  # type: ignore
        return False


def _shrink_to_global(
    intent_rates: Dict[str, float],
    intent_n: Dict[str, float],
    global_rates: Dict[str, float],
    global_n: Dict[str, float],
    ess_cap: float,
) -> Dict[str, float]:
    """Empirical-Bayes shrinkage of per-intent rates toward global rates.

    Uses a weighted average where weight = n_intent / (n_intent + n_global),
    with n_intent capped by `ess_cap` to avoid over-trusting noisy histories.
    """
    out: Dict[str, float] = {}
    for tool, r_i in (intent_rates or {}).items():
        try:
            n_i = float(intent_n.get(tool, 0.0))
        except Exception:
            n_i = 0.0
        try:
            n_g = float(global_n.get(tool, 0.0))
        except Exception:
            n_g = 0.0
        try:
            n_i = min(max(0.0, n_i), float(ess_cap))
        except Exception:
            n_i = n_i
        denom = n_i + (n_g if n_g > 0.0 else 1.0)
        w = (n_i / denom) if denom > 0 else 0.0
        r_g = float(global_rates.get(tool, r_i))
        out[tool] = float(w * float(r_i) + (1.0 - w) * r_g)
    return out


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def load_global_prefs() -> Dict[str, Any]:
    if GLOBAL_PREFS_PATH.exists():
        return _read_json(GLOBAL_PREFS_PATH)
    return {}


def load_prefs(*, merge_global: bool = True) -> Dict[str, Any]:
    """Load repo-local prefs and optionally merge in global prefs as fallbacks.

    Local values take precedence; global fills missing keys.
    """
    local = _read_json(PREFS_PATH) if PREFS_PATH.exists() else {}
    if not merge_global:
        return local
    glob = load_global_prefs()
    if not glob:
        return local
    # Scope-aware merge: only merge learning maps when scope matches; always allow personalization
    out: Dict[str, Any] = json.loads(json.dumps(local)) or {}
    # Determine current scope and whether global prefs match current scope
    def _current_scope() -> Dict[str, str]:
        return {
            'user': str(os.getenv('USER', 'unknown')),
            'workspace': str(os.getenv('NERION_SCOPE_WS', 'default')),
            'project': str(os.getenv('NERION_SCOPE_PROJECT', 'default')),
        }
    def _scope_matches(s: Dict[str, Any] | None) -> bool:
        try:
            cur = _current_scope()
            if not isinstance(s, dict):
                return False
            for k in ('user', 'workspace', 'project'):
                if str(s.get(k, '')) != str(cur.get(k, '')):
                    return False
            return True
        except Exception:
            return False
    merge_learning = _scope_matches(((glob.get('stats') or {}).get('scope') if isinstance(glob.get('stats'), dict) else None))
    def _merge_map(key: str):
        g = glob.get(key) or {}
        if isinstance(g, dict):
            out.setdefault(key, {})
            for k, v in g.items():
                out[key].setdefault(k, v)
    def _merge_nested_map(key: str):
        g = glob.get(key) or {}
        if isinstance(g, dict):
            out.setdefault(key, {})
            for intent, m in g.items():
                if not isinstance(m, dict):
                    continue
                out[key].setdefault(intent, {})
                for tool, val in m.items():
                    out[key][intent].setdefault(tool, val)
    # Merge learning-related maps only when scope matches
    if merge_learning:
        for k in ("tool_success_rate", "tool_sample_weight", "tool_success_weight"):
            _merge_map(k)
        for k in ("tool_success_rate_by_intent", "tool_sample_weight_by_intent", "tool_success_weight_by_intent"):
            _merge_nested_map(k)
    # Merge personalization and experiments metadata as fallbacks
    if isinstance(glob.get('personalization'), dict):
        out.setdefault('personalization', {})
        for k, v in (glob['personalization'] or {}).items():
            out['personalization'].setdefault(k, v)
    if merge_learning and isinstance(glob.get('experiments'), dict):
        out.setdefault('experiments', {})
        for name, armdata in (glob['experiments'] or {}).items():
            out['experiments'].setdefault(name, armdata)
    return out


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Crash-safe JSON write using a temp file + fsync + atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.prefs.', dir=str(path.parent))
    try:
        with io.open(fd, 'w', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True))
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                # Best-effort on platforms without fsync
                pass
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _validate_prefs(p: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal shape validation for prefs (no external deps)."""
    if not isinstance(p, dict):
        raise ValueError('prefs must be an object')
    # Forward-compat guard
    try:
        if int(p.get('schema_version') or 0) > _SCHEMA_VERSION:
            raise ValueError('prefs schema too new')
    except Exception:
        pass
    # Map fields that should be dicts
    for k in (
        'tool_success_rate', 'tool_sample_weight', 'tool_success_weight',
        'tool_success_rate_global', 'tool_success_weight_global',
        'personalization', 'experiments',
    ):
        if k in p and not isinstance(p[k], dict):
            raise ValueError(f'{k} must be a map')
    # Nested maps by intent
    for k in (
        'tool_success_rate_by_intent', 'tool_sample_weight_by_intent', 'tool_success_weight_by_intent',
    ):
        if k in p and isinstance(p[k], dict):
            for intent, m in (p[k] or {}).items():
                if not isinstance(m, dict):
                    raise ValueError(f'{k}[{intent}] must be a map')
    return p


def save_prefs(p: Dict[str, Any]) -> None:
    _atomic_write_json(PREFS_PATH, _validate_prefs(p))


def save_global_prefs(p: Dict[str, Any]) -> None:
    GLOBAL_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_PREFS_PATH.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding='utf-8')


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        return int(v) if v is not None and str(v).strip() != '' else int(default)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return float(v) if v is not None and str(v).strip() != '' else float(default)
    except Exception:
        return float(default)


def review_outcomes(max_lines: int = 2000) -> Dict[str, Any]:
    """Aggregate recent outcomes and update prefs with simple biases.

    - Counts successes vs failures
    - Tracks tool names seen in action_taken steps (if present)
    - Writes a bias map like {'tool_success_rate': {...}} to prefs
    """
    # Window/decay knobs
    max_lines = _env_int('NERION_LEARN_WINDOW_N', max_lines)
    half_life_days = _env_float('NERION_LEARN_DECAY_HALF_LIFE_DAYS', 0.0)
    _min_samples = _env_float('NERION_LEARN_MIN_SAMPLES', 3.0)
    now = time.time()

    successes = 0
    failures = 0
    latencies: List[float] = []
    escalations = 0
    # Accumulate weighted successes/total per tool (global)
    by_tool: Dict[str, Tuple[float, float]] = {}
    # Accumulate weighted successes/total per (intent, tool)
    by_intent: DefaultDict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)

    if LOG_PATH.exists():
        lines = LOG_PATH.read_text(encoding='utf-8', errors='ignore').splitlines()[-max_lines:]
    else:
        lines = []

    # A/B experiment accumulator
    experiments: Dict[str, Dict[str, Any]] = {}

    for ln in lines:
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        ok = bool(rec.get('outcome_success'))
        if ok:
            successes += 1
        else:
            failures += 1
        # Weight by decay if enabled (applies to both successes and failures)
        try:
            ts = float(rec.get('ts') or now)
        except Exception:
            ts = now
        weight = 1.0
        if half_life_days and half_life_days > 0:
            try:
                age = max(0.0, now - ts)
                half_life_s = float(half_life_days) * 86400.0
                weight = 0.5 ** (age / half_life_s)
            except Exception:
                weight = 1.0

        steps = ((rec.get('action_taken') or {}).get('steps') or [])
        # Escalation detection (ask_user/escalate/human_review)
        try:
            for st in steps:
                tname = (st or {}).get('tool')
                if str(tname) in {'ask_user', 'escalate', 'human_review'}:
                    escalations += 1
                    break
        except Exception:
            pass
        # Infer intent from Parent decision if available
        intent = None
        try:
            dec = rec.get('parent_decision') or {}
            intent = dec.get('intent') or None
        except Exception:
            intent = None
        # Per-step credit weights: last step gets full credit; preceding steps get partial.
        prev_w = 0.25
        try:
            prev_w = float((os.environ.get('NERION_LEARN_PREV_STEP_WEIGHT') or '0.25').strip())
        except Exception:
            prev_w = 0.25
        n_steps = len(steps)
        credits: List[float] = []
        if n_steps > 0:
            credits = [prev_w] * max(0, n_steps - 1) + [1.0]
        # Optional cost weighting: reduce credit for long-duration steps
        try:
            use_cost = (os.environ.get('NERION_LEARN_COST_WEIGHT') or '').strip().lower() in {'1','true','yes','on'}
        except Exception:
            use_cost = False
        if use_cost and n_steps > 0:
            durs = []
            for st in steps:
                try:
                    durs.append(float((st or {}).get('duration_ms') or 0.0))
                except Exception:
                    durs.append(0.0)
            mean_d = (sum(durs)/max(1,len(durs))) if durs else 0.0
            if mean_d > 0:
                for i in range(n_steps):
                    try:
                        dn = durs[i] / mean_d if mean_d else 0.0
                        factor = 1.0 / (1.0 + dn)
                        credits[i] *= factor
                    except Exception:
                        continue
        # Apply credited weights into aggregates
        for idx, st in enumerate(steps):
            tool = (st or {}).get('tool')
            if not tool:
                continue
            credit = credits[idx] if idx < len(credits) else prev_w
            ok_add = (weight * credit) if ok else 0.0
            tot_add = (weight * credit)
            ok_sum, tot_sum = by_tool.get(tool, (0.0, 0.0))
            ok_sum += ok_add
            tot_sum += tot_add
            by_tool[tool] = (ok_sum, tot_sum)
            if intent:
                it = by_intent.get(intent) or {}
                i_ok, i_tot = it.get(tool, (0.0, 0.0))
                i_ok += ok_add
                i_tot += tot_add
                it[tool] = (i_ok, i_tot)
                by_intent[intent] = it

        # A/B experiment aggregation
        try:
            exp = rec.get('experiment') or {}
            ename = (exp.get('name') or '').strip()
            earm = (exp.get('arm') or '').strip()
            if ename and earm:
                experiments.setdefault(ename, {}).setdefault(earm, {"ok":0.0,"total":0.0,"lat_sum":0.0,"lat_n":0})
                ex = experiments[ename][earm]
                ex["ok"] += (1.0 if ok else 0.0)
                ex["total"] += 1.0
                try:
                    # Record experiment latency and global latency
                    lat_v = rec.get('latency_ms')
                    lat = float(lat_v or 0.0)
                    if lat > 0:
                        ex["lat_sum"] += lat
                        ex["lat_n"] += 1
                        latencies.append(lat)
                except Exception:
                    pass
        except Exception:
            pass

    # Update prefs
    prefs = load_prefs()
    # Compute smoothed rates with Beta(1,1) prior and min sample threshold
    bias: Dict[str, float] = {}
    samples: Dict[str, float] = {}
    succ_w: Dict[str, float] = {}
    for tool, (ok_sum, tot_sum) in by_tool.items():
        # Compute smoothed rate; include even if below min_samples to avoid dropping fresh tools
        rate = (ok_sum + 1.0) / (tot_sum + 2.0)
        # Clamp for safety
        if rate < 0.0:
            rate = 0.0
        if rate > 1.0:
            rate = 1.0
        bias[tool] = float(rate)
        samples[tool] = float(tot_sum)
        succ_w[tool] = float(ok_sum)
    # Replace the map with fresh rates to avoid stale values across runs/tests
    prefs['tool_success_rate'] = dict(bias)
    # Back-compat alias for readability when multiple namespaces exist
    prefs['tool_success_rate_global'] = dict(bias)
    prefs['tool_sample_weight'] = samples
    prefs['tool_success_weight'] = succ_w
    prefs['tool_success_weight_global'] = dict(succ_w)
    # Include scope to prevent cross-user/workspace/project bleed-through
    prefs['stats'] = {
        'successes': successes,
        'failures': failures,
        'updated_at': int(now),
        'scope': {
            'user': str(os.getenv('USER', 'unknown')),
            'workspace': str(os.getenv('NERION_SCOPE_WS', 'default')),
            'project': str(os.getenv('NERION_SCOPE_PROJECT', 'default')),
        },
    }
    # Schema versioning for backward compatibility
    try:
        prev_ver = int(prefs.get('schema_version') or 0)
    except Exception:
        prev_ver = 0
    prefs['schema_version'] = _SCHEMA_VERSION
    if prev_ver != _SCHEMA_VERSION:
        prefs['last_migrated'] = int(now)

    # Per-intent aggregates
    bias_by_intent: Dict[str, Dict[str, float]] = {}
    samples_by_intent: Dict[str, Dict[str, float]] = {}
    succ_by_intent: Dict[str, Dict[str, float]] = {}
    for intent, tool_map in by_intent.items():
        ib: Dict[str, float] = {}
        isamp: Dict[str, float] = {}
        isucc: Dict[str, float] = {}
        for tool, (ok_sum, tot_sum) in tool_map.items():
            rate = (ok_sum + 1.0) / (tot_sum + 2.0)
            if rate < 0.0:
                rate = 0.0
            if rate > 1.0:
                rate = 1.0
            ib[tool] = float(rate)
            isamp[tool] = float(tot_sum)
            isucc[tool] = float(ok_sum)
        if ib:
            bias_by_intent[intent] = ib
        if isamp:
            samples_by_intent[intent] = isamp
        if isucc:
            succ_by_intent[intent] = isucc

    # Optionally shrink per-intent rates toward global (stabilize low-N intents)
    if bias_by_intent:
        try:
            ess_cap = float((os.environ.get('NERION_LEARN_EFF_N_MAX') or '1000').strip())
        except Exception:
            ess_cap = 1000.0
        # Use just-computed global maps as priors
        global_rates = prefs.get('tool_success_rate') or prefs.get('tool_success_rate_global') or {}
        global_n = prefs.get('tool_sample_weight') or {}
        shrunk: Dict[str, Dict[str, float]] = {}
        for intent, rates in (bias_by_intent or {}).items():
            n_int = (samples_by_intent or {}).get(intent, {})
            shrunk[intent] = _shrink_to_global(rates, n_int, global_rates, global_n, ess_cap)
        prefs['tool_success_rate_by_intent'] = shrunk
    if samples_by_intent:
        prefs['tool_sample_weight_by_intent'] = samples_by_intent
    if succ_by_intent:
        prefs['tool_success_weight_by_intent'] = succ_by_intent
    # Experiments summary (if any)
    if experiments:
        exp_out: Dict[str, Any] = {}
        exp_meta: Dict[str, Any] = {}
        for name, arms in experiments.items():
            arm_stats: Dict[str, Any] = {}
            for arm, st in arms.items():
                okc = float(st.get('ok') or 0.0)
                tot = float(st.get('total') or 0.0)
                lat_sum = float(st.get('lat_sum') or 0.0)
                lat_n = int(st.get('lat_n') or 0)
                sr = (okc / tot) if tot > 0 else None
                avg_lat = (lat_sum / lat_n) if lat_n > 0 else None
                arm_stats[arm] = {"success_rate": sr, "n": int(tot), "avg_latency_ms": avg_lat,
                                   "successes": int(round(okc)), "failures": int(round(max(0.0, tot - okc)))}
            # Decision via MSPRT over first two arms (sorted for determinism)
            decision = None
            if _MSPRT is not None and len(arm_stats) >= 2:
                ordered = sorted(arm_stats.keys())
                A, B = ordered[0], ordered[1]
                a = arm_stats[A]
                b = arm_stats[B]
                s_a = int(a.get('successes') or 0)
                f_a = int(a.get('failures') or 0)
                s_b = int(b.get('successes') or 0)
                f_b = int(b.get('failures') or 0)
                try:
                    d = _MSPRT().decide(s_a, f_a, s_b, f_b)
                    decision = {"arms": [A, B], "decision": d}
                except Exception:
                    decision = None
            exp_out[name] = arm_stats
            if decision:
                exp_meta[name] = decision
        if exp_meta:
            prefs['experiments_meta'] = exp_meta
        prefs['experiments'] = exp_out

    # Guardrails metrics (global, last window)
    total = successes + failures
    try:
        err_rate = (failures / float(total)) if total > 0 else 0.0
    except Exception:
        err_rate = 0.0
    try:
        lat_p95 = None
        if latencies:
            latencies.sort()
            k = max(0, min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1)))))
            lat_p95 = float(latencies[k])
    except Exception:
        lat_p95 = None
    metrics = {
        'error_rate': float(err_rate),
        'latency_p95_ms': float(lat_p95) if lat_p95 is not None else 0.0,
        'escalation_rate': float((escalations / float(total)) if total > 0 else 0.0),
        'n': int(total),
    }
    try:
        breached = bool(_guard_breached(metrics))
    except Exception:
        breached = False
    prefs['guardrails'] = {'metrics': metrics, 'breached': breached}

    save_prefs(prefs)
    return {
        'successes': successes,
        'failures': failures,
        'tool_success_rate': bias,
        'tool_success_rate_by_intent': bias_by_intent,
    }
