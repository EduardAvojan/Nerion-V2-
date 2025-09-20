from __future__ import annotations

import os
from typing import Dict, Iterable, Optional


def _p95(values: Iterable[float]) -> Optional[float]:
    arr = [float(v) for v in values if isinstance(v, (int, float))]
    if not arr:
        return None
    arr.sort()
    k = max(0, min(len(arr) - 1, int(round(0.95 * (len(arr) - 1)))))
    return float(arr[k])


def compute_metrics(records: Iterable[Dict]) -> Dict[str, float]:
    n = 0
    failures = 0
    latencies = []
    escal = 0
    for rec in records:
        try:
            n += 1
            ok = bool(rec.get('outcome_success'))
            if not ok:
                failures += 1
            lat = rec.get('latency_ms')
            if isinstance(lat, (int, float)) and float(lat) > 0:
                latencies.append(float(lat))
            # Detect escalation via action steps (ask_user, escalate, human_review)
            steps = ((rec.get('action_taken') or {}).get('steps') or [])
            for st in steps:
                tool = (st or {}).get('tool')
                if str(tool) in {'ask_user', 'escalate', 'human_review'}:
                    escal += 1
                    break
        except Exception:
            continue
    err_rate = (failures / float(n)) if n > 0 else 0.0
    p95 = _p95(latencies)
    esc_rate = (escal / float(n)) if n > 0 else 0.0
    out = {
        'error_rate': float(err_rate),
        'latency_p95_ms': float(p95) if p95 is not None else 0.0,
        'escalation_rate': float(esc_rate),
        'n': int(n),
    }
    return out


def breached(metrics: Dict[str, float]) -> bool:
    try:
        if metrics.get('error_rate', 0.0) > float(os.getenv('NERION_GUARDRAIL_ERR', '0.10')):
            return True
        if metrics.get('latency_p95_ms', 0.0) > float(os.getenv('NERION_GUARDRAIL_P95', '8000')):
            return True
        if metrics.get('escalation_rate', 0.0) > float(os.getenv('NERION_GUARDRAIL_ESC', '0.15')):
            return True
    except Exception:
        return False
    return False

