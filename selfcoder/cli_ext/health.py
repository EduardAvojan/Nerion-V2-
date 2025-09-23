from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ops.telemetry import load_operator_snapshot


def _load_jsonl_tail(path: Path, n: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
        out: List[Dict[str, Any]] = []
        for ln in reversed(lines):
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
            if len(out) >= n:
                break
        return list(reversed(out))
    except Exception:
        return []


def cmd_dashboard(args: argparse.Namespace) -> int:
    N = int(getattr(args, 'last', 100) or 100)
    rep: Dict[str, Any] = {}
    # Experience
    rows = _load_jsonl_tail(Path('out/experience/log.jsonl'), N)
    succ = sum(1 for r in rows if r.get('outcome_success'))
    fail = sum(1 for r in rows if not r.get('outcome_success'))
    rep['experience'] = {'records': len(rows), 'successes': succ, 'failures': fail}
    # Realized epsilon (approx): fraction where chosen tool != current top tool by learned rates
    realized_eps = None
    try:
        from selfcoder.learning.continuous import load_prefs as _lp
        prefs = _lp()
        rates_g = (prefs.get('tool_success_rate') or {})
        rates_by_int = (prefs.get('tool_success_rate_by_intent') or {})
        def _top_tool(intent: str | None) -> str | None:
            src = None
            if intent and isinstance(rates_by_int, dict) and rates_by_int.get(intent):
                src = rates_by_int.get(intent) or {}
            else:
                src = rates_g
            if not isinstance(src, dict) or not src:
                return None
            try:
                return sorted(((k, float(v)) for k, v in src.items()), key=lambda kv: kv[1], reverse=True)[0][0]
            except Exception:
                return None
        total, off = 0, 0
        for r in rows:
            steps = ((r.get('action_taken') or {}).get('steps') or [])
            if not steps:
                continue
            intent = None
            try:
                dec = r.get('parent_decision') or {}
                intent = dec.get('intent') or None
            except Exception:
                intent = None
            chosen = (steps[0] or {}).get('tool')
            best = _top_tool(intent)
            if chosen and best:
                total += 1
                if str(chosen) != str(best):
                    off += 1
        realized_eps = (off / float(total)) if total > 0 else 0.0
    except Exception:
        realized_eps = None
    rep['realized_epsilon'] = None if realized_eps is None else round(float(realized_eps), 3)
    # Effective sample size after decay
    try:
        from selfcoder.learning.continuous import load_prefs as _lp2
        prefs2 = _lp2()
        samp = prefs2.get('tool_sample_weight') or {}
        ess_total = 0.0
        if isinstance(samp, dict):
            for v in samp.values():
                try:
                    ess_total += float(v)
                except Exception:
                    continue
        rep['ess_total'] = int(ess_total)
    except Exception:
        rep['ess_total'] = None
    # Intent drift (KL divergence between first/second half)
    try:
        def _intent_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for r in items:
                try:
                    dec = r.get('parent_decision') or {}
                    intent = (dec.get('intent') or 'unknown')
                    out[intent] = out.get(intent, 0) + 1
                except Exception:
                    continue
            return out
        half = max(1, len(rows)//2)
        a = _intent_counts(rows[:half])
        b = _intent_counts(rows[half:])
        def _kl(p: Dict[str, int], q: Dict[str, int]) -> float:
            import math
            total_p = max(1, sum(p.values()))
            total_q = max(1, sum(q.values()))
            keys = set(p.keys()) | set(q.keys())
            s = 0.0
            for k in keys:
                pi = (p.get(k, 0) / total_p)
                qi = (q.get(k, 0) / total_q)
                if pi > 0 and qi > 0:
                    s += pi * math.log(pi/qi)
            return s
        rep['intent_drift_kl'] = round(_kl(a, b), 3)
    except Exception:
        rep['intent_drift_kl'] = None
    # STT metrics
    stt_rows = _load_jsonl_tail(Path('out/voice/latency.jsonl'), N)
    from collections import defaultdict
    by = defaultdict(list)
    for r in stt_rows:
        key = (str(r.get('backend') or 'unknown'), str(r.get('model') or ''))
        try:
            by[key].append(int(r.get('duration_ms') or 0))
        except Exception:
            continue
    stt_groups = []
    for (backend, model), vals in by.items():
        vals = [v for v in vals if isinstance(v, int) and v >= 0]
        if not vals:
            continue
        stt_groups.append({'backend': backend, 'model': model, 'count': len(vals), 'avg_ms': int(sum(vals)/max(1,len(vals)))})
    rep['stt'] = {'samples': len(stt_rows), 'groups': stt_groups}
    # Coverage baseline vs current (if any)
    try:
        from selfcoder import coverage_utils as covu
        baseline = covu.load_baseline()
        cur = None
        cov_json = Path('coverage.json')
        if cov_json.exists():
            try:
                cur = json.loads(cov_json.read_text(encoding='utf-8'))
            except Exception:
                cur = None
        if cur:
            pct, delta = covu.compare_to_baseline(cur, baseline)
            rep['coverage'] = {'current_pct': round(pct, 2), 'delta_vs_baseline': round(delta, 2)}
        else:
            rep['coverage'] = {'current_pct': None, 'delta_vs_baseline': None}
    except Exception:
        rep['coverage'] = {'current_pct': None, 'delta_vs_baseline': None}
    # Top tools by learned success
    try:
        from selfcoder.learning.continuous import load_prefs as _lp
        prefs = _lp()
        rates = (prefs.get('tool_success_rate') or {})
        top = []
        if isinstance(rates, dict) and rates:
            top = sorted(((k, float(v)) for k, v in rates.items()), key=lambda kv: kv[1], reverse=True)[:5]
        rep['top_tools'] = [{'tool': k, 'rate': round(v, 2)} for k, v in top]
        # Per-intent brief: top-2 per intent
        pi = (prefs.get('tool_success_rate_by_intent') or {})
        per_intent = {}
        if isinstance(pi, dict):
            for intent, m in list(pi.items())[:8]:
                try:
                    ranked = sorted(((k, float(v)) for k, v in (m or {}).items()), key=lambda kv: kv[1], reverse=True)[:2]
                except Exception:
                    ranked = []
                per_intent[intent] = [{'tool': k, 'rate': round(v,2)} for k, v in ranked]
        rep['per_intent'] = per_intent
        # Experiments summary
        rep['experiments'] = prefs.get('experiments') or {}
    except Exception:
        rep['top_tools'] = []
        rep['per_intent'] = {}
        rep['experiments'] = {}

    try:
        rep['telemetry'] = load_operator_snapshot()
    except Exception:
        rep['telemetry'] = None

    if getattr(args, 'json', False):
        print(json.dumps(rep, indent=2))
    else:
        print('[health] dashboard')
        print(f"  experience: {rep['experience']['records']} records, {rep['experience']['successes']} OK / {rep['experience']['failures']} FAIL")
        print(f"  stt: {rep['stt']['samples']} samples, groups={len(rep['stt']['groups'])}")
        print(f"  learning: ESS={rep.get('ess_total')} realized_epsilon={rep.get('realized_epsilon')} drift_KL={rep.get('intent_drift_kl')}")
        cov = rep['coverage']
        print(f"  coverage: current={cov['current_pct']}% delta={cov['delta_vs_baseline']}")
        telemetry = rep.get('telemetry') or {}
        knowledge = telemetry.get('knowledge_graph') if isinstance(telemetry, dict) else None
        hotspots = []
        if isinstance(knowledge, dict):
            hotspots = knowledge.get('hotspots') or []
        if hotspots:
            top = hotspots[0]
            component = top.get('component') or 'unknown'
            risk = top.get('risk_score')
            if isinstance(risk, (int, float)):
                print(f"  hotspot: {component} (risk {risk:.1f})")
            else:
                print(f"  hotspot: {component}")
        tele = rep.get('telemetry') or {}
        window = (tele.get('window') or {})
        ratio = tele.get('prompt_completion_ratio') or {}
        print(
            "  telemetry: "
            + (
                f"{tele.get('counts_total', 0)} events/{window.get('hours', 0)}h prompts={ratio.get('prompts', 0)} completions={ratio.get('completions', 0)}"
                if tele
                else "no samples"
            )
        )
        providers = tele.get('providers') if tele else None
        if providers:
            top = providers[0]
            latency = top.get('avg_latency_ms')
            latency_str = f"{int(latency)}ms" if isinstance(latency, (int, float)) else '—'
            cost = top.get('cost_usd')
            cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else '—'
            err = top.get('error_rate')
            err_str = f"{err:.3f}" if isinstance(err, (int, float)) else '—'
            print(f"    top provider: {top.get('provider')} latency={latency_str} cost={cost_str} error_rate={err_str}")
        apply = tele.get('apply_metrics') if tele else None
        if apply and isinstance(apply, dict) and apply.get('total'):
            rate = apply.get('rate')
            rate_str = f"{rate * 100:.1f}%" if isinstance(rate, (int, float)) else '—'
            print(
                "    apply window: "
                + f"success {apply.get('success', 0)}/{apply.get('total', 0)} (rate {rate_str}), "
                + f"rolled_back {apply.get('rolled_back', 0)}"
            )
        policy_gates = tele.get('policy_gates') if tele else None
        if isinstance(policy_gates, dict) and policy_gates:
            summary = ', '.join(f"{k}:{v}" for k, v in sorted(policy_gates.items()))
            print(f"    policy gates: {summary}")
        governor_counts = tele.get('governor_decisions') if tele else None
        if isinstance(governor_counts, dict) and governor_counts:
            summary = ', '.join(f"{k}:{v}" for k, v in sorted(governor_counts.items()))
            print(f"    governor decisions: {summary}")
        latest = (tele.get('latest_reflection') or {}) if tele else {}
        anomalies = latest.get('anomalies') or []
        if anomalies:
            detail = anomalies[0].get('detail') or anomalies[0].get('kind')
            print(f"    anomalies: {len(anomalies)} (e.g. {detail})")
        # Table for top tools (if available)
        tops = rep.get('top_tools') or []
        if tops:
            print('  top tools (learned):')
            # Simple fixed-width table
            print('   Tool'.ljust(28) + 'Rate')
            print('   ' + '-'*26 + ' ' + '-'*6)
            for t in tops:
                name = str(t.get('tool') or '')[:24]
                rate = f"{float(t.get('rate') or 0.0):.2f}"
                print(f"   {name.ljust(26)} {rate.rjust(6)}")
        # Per-intent
        pi = rep.get('per_intent') or {}
        if pi:
            print('  per-intent top tools:')
            for intent, rows in pi.items():
                left = ', '.join(f"{r['tool']}:{r['rate']:.2f}" for r in rows)
                print(f"   {intent}: {left}")
        # Experiments
        exps = rep.get('experiments') or {}
        if exps:
            print('  experiments:')
            for name, arms in exps.items():
                arms = arms or {}
                parts = []
                for arm, st in arms.items():
                    sr = st.get('success_rate')
                    n = st.get('n')
                    av = st.get('avg_latency_ms')
                    parts.append(f"{arm}: sr={None if sr is None else round(sr,2)} n={n} lat={None if av is None else int(av)}ms")
                print('   ' + name + ' — ' + '; '.join(parts))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('health', help='health dashboards and summaries')
    sp = p.add_subparsers(dest='health_cmd', required=True)
    d = sp.add_parser('dashboard', help='print a quick terminal dashboard summary')
    d.add_argument('--json', action='store_true')
    d.add_argument('--last', type=int, default=100)
    d.set_defaults(func=cmd_dashboard)

    def _cmd_html(args: argparse.Namespace) -> int:
        out = Path(getattr(args, 'out', 'out/health/index.html'))
        out.parent.mkdir(parents=True, exist_ok=True)
    # Reuse dashboard data and write a tiny HTML
        # Capture JSON by calling cmd_dashboard
        data = {
            'experience': {'records': 0, 'successes': 0, 'failures': 0},
            'stt': {'samples': 0, 'groups': []},
            'coverage': {'current_pct': None, 'delta_vs_baseline': None}
        }
        try:
            # Inline call to build fresh data
            N = int(getattr(args, 'last', 100) or 100)
            # Use same helpers
            def _tail(path: Path, n: int):
                return _load_jsonl_tail(path, n)
            rows = _tail(Path('out/experience/log.jsonl'), N)
            succ = sum(1 for r in rows if r.get('outcome_success'))
            fail = sum(1 for r in rows if not r.get('outcome_success'))
            data['experience'] = {'records': len(rows), 'successes': succ, 'failures': fail}
            stt_rows = _tail(Path('out/voice/latency.jsonl'), N)
            from collections import defaultdict
            by = defaultdict(list)
            for r in stt_rows:
                key = (str(r.get('backend') or 'unknown'), str(r.get('model') or ''))
                try:
                    by[key].append(int(r.get('duration_ms') or 0))
                except Exception:
                    continue
            groups = []
            for (backend, model), vals in by.items():
                vals = [v for v in vals if isinstance(v, int) and v >= 0]
                if not vals:
                    continue
                groups.append({'backend': backend, 'model': model, 'count': len(vals), 'avg_ms': int(sum(vals)/max(1,len(vals)))})
            data['stt'] = {'samples': len(stt_rows), 'groups': groups}
            try:
                from selfcoder import coverage_utils as covu
                baseline = covu.load_baseline()
                cur = None
                cov_json = Path('coverage.json')
                if cov_json.exists():
                    cur = json.loads(cov_json.read_text(encoding='utf-8'))
                if cur:
                    pct, delta = covu.compare_to_baseline(cur, baseline)
                    data['coverage'] = {'current_pct': round(pct, 2), 'delta_vs_baseline': round(delta, 2)}
            except Exception:
                pass
        except Exception:
            pass
        # Compute simple bar widths
        ok = int(data['experience']['successes'])
        fail = int(data['experience']['failures'])
        total = max(1, ok+fail)
        ok_w = int(100 * ok / total)
        fail_w = 100 - ok_w
        bars = ''.join(
            f"<li>{g['backend']} {g['model']} <div style='display:inline-block;background:#4a90e2;height:10px;width:{min(100, g['avg_ms']//2)}px;margin-left:8px'></div> {g['avg_ms']}ms avg ({g['count']} samples)</li>"
            for g in data['stt']['groups']
        )
        refresh = int(getattr(args, 'refresh', 10) or 10)
        # Prepare top tools table
        try:
            from selfcoder.learning.continuous import load_prefs as _lp
            prefs = _lp()
            rates = (prefs.get('tool_success_rate') or {})
            top = []
            if isinstance(rates, dict) and rates:
                top = sorted(((k, float(v)) for k, v in rates.items()), key=lambda kv: kv[1], reverse=True)[:5]
            top_rows = [{'tool': k, 'rate': round(v, 2)} for k, v in top]
        except Exception:
            top_rows = []
        rows = ''.join(f"<tr><td>{t['tool']}</td><td style='text-align:right'>{t['rate']}</td></tr>" for t in top_rows)
        # Per-intent HTML (build directly from prefs)
        try:
            from selfcoder.learning.continuous import load_prefs as _lp2
            _prefs = _lp2()
            _pi_src = (_prefs.get('tool_success_rate_by_intent') or {})
            pi_rows = ''
            for intent, m in (_pi_src or {}).items():
                try:
                    ranked = sorted(((k, float(v)) for k, v in (m or {}).items()), key=lambda kv: kv[1], reverse=True)[:2]
                except Exception:
                    ranked = []
                cells = ', '.join(f"{k}:{v}" for k, v in ranked)
                pi_rows += f"<tr><td>{intent}</td><td>{cells}</td></tr>"
            ex_src = (_prefs.get('experiments') or {})
            ex_rows = ''
            for name, arms in (ex_src or {}).items():
                parts = []
                for arm, st in (arms or {}).items():
                    sr = st.get('success_rate')
                    n = st.get('n')
                    av = st.get('avg_latency_ms')
                    parts.append(f"{arm}: sr={sr:.2f if isinstance(sr,(int,float)) else 'NA'} n={n} lat={av if av is not None else 'NA'}")
                ex_rows += f"<tr><td>{name}</td><td>{'; '.join(parts)}</td></tr>"
        except Exception:
            pi_rows = ''
            ex_rows = ''
        html = f"""
<!doctype html>
<html><head><meta charset='utf-8'><meta http-equiv='refresh' content='{refresh}'><title>Nerion Health</title>
<style>body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:2rem;}}h1{{margin-bottom:.5rem}}.card{{border:1px solid #ddd;padding:1rem;border-radius:8px;margin:.5rem 0}}code{{background:#f5f5f5;padding:.15rem .35rem;border-radius:4px}}.prog{{display:flex;width:100%;height:12px;border-radius:6px;background:#eee;overflow:hidden}}.ok{{background:#2ecc71}}.fail{{background:#e74c3c}}</style>
</head><body>
<h1>Nerion Health Dashboard</h1>
<div class='card'><h3>Experience</h3>
<div>Records: <b>{data['experience']['records']}</b> — ✅ {data['experience']['successes']} / ❌ {data['experience']['failures']}</div>
<div class='prog'><div class='ok' style='width:{ok_w}%'></div><div class='fail' style='width:{fail_w}%'></div></div>
</div>
<div class='card'><h3>STT Latency</h3>
<div>Samples: <b>{data['stt']['samples']}</b></div>
<ul>
{bars}
</ul></div>
<div class='card'><h3>Coverage</h3>
<div>Current: <b>{data['coverage']['current_pct']}</b>% — Δ vs baseline: <b>{data['coverage']['delta_vs_baseline']}</b></div>
</div>
<div class='card'><h3>Top Tools (learned)</h3>
<table style='width:100%;border-collapse:collapse'>
<tr><th style='text-align:left'>Tool</th><th style='text-align:right'>Rate</th></tr>
{rows}
</table>
</div>
<div class='card'><h3>Per-Intent Top Tools</h3>
<table style='width:100%;border-collapse:collapse'>
<tr><th style='text-align:left'>Intent</th><th style='text-align:left'>Top</th></tr>
{pi_rows}
</table>
</div>
<div class='card'><h3>Experiments</h3>
<table style='width:100%;border-collapse:collapse'>
<tr><th style='text-align:left'>Name</th><th style='text-align:left'>Arms</th></tr>
{ex_rows}
</table>
</div>
<p>Generated by <code>nerion health html</code></p>
</body></html>
"""
        out.write_text(html, encoding='utf-8')
        print(f"[health] wrote: {out}")
        return 0

    h = sp.add_parser('html', help='write an HTML health dashboard')
    h.add_argument('--out', default='out/health/index.html')
    h.add_argument('--last', type=int, default=100)
    h.add_argument('--refresh', type=int, default=10, help='seconds to auto-refresh the page')
    h.set_defaults(func=_cmd_html)
