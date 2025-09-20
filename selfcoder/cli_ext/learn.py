from __future__ import annotations

import argparse
import json
from selfcoder.learning.continuous import review_outcomes, load_prefs, load_global_prefs
from pathlib import Path


def cmd_review(_args: argparse.Namespace) -> int:
    out = review_outcomes()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_show(_args: argparse.Namespace) -> int:
    prefs = load_prefs()
    print(json.dumps(prefs, ensure_ascii=False, indent=2))
    rates = (prefs.get('tool_success_rate') or {})
    if isinstance(rates, dict) and rates:
        print("\n[top tools by success rate]")
        for k, v in sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:8]:
            print(f" - {k}: {float(v):.2f}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser('learn', help='continuous learning tools')
    sp = p.add_subparsers(dest='learn_cmd', required=True)

    rv = sp.add_parser('review', help='review recent outcomes and update prefs')
    rv.set_defaults(func=cmd_review)

    sh = sp.add_parser('show', help='show current learned prefs')
    sh.add_argument('--intent', help='filter by intent name')
    sh.add_argument('--explain', action='store_true', help='print why bias would be chosen (samples, CI, delta)')
    sh.add_argument('--global-only', action='store_true', help='show global prefs only (~/.nerion/prefs_global.json)')
    sh.add_argument('--no-merge', action='store_true', help='do not merge global prefs into local view')
    def _cmd_show(args: argparse.Namespace) -> int:
        if getattr(args, 'global_only', False):
            prefs = load_global_prefs()
        else:
            prefs = load_prefs(merge_global=not bool(getattr(args, 'no_merge', False)))
        intent = getattr(args, 'intent', None)
        explain = bool(getattr(args, 'explain', False))
        if not intent:
            print(json.dumps(prefs, ensure_ascii=False, indent=2))
            rates = (prefs.get('tool_success_rate') or {})
            if isinstance(rates, dict) and rates:
                print("\n[top tools by success rate]")
                for k, v in sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:8]:
                    print(f" - {k}: {float(v):.2f}")
            return 0
        # Per-intent view
        by_int = (prefs.get('tool_success_rate_by_intent') or {})
        rates = by_int.get(intent) or {}
        if not rates:
            print(f"[learn] no per-intent rates for {intent}")
            return 0
        print(json.dumps({intent: rates}, ensure_ascii=False, indent=2))
        if explain:
            samples = ((prefs.get('tool_sample_weight_by_intent') or {}).get(intent) or {})
            import math as _m
            def _wilson(p: float, n: float, z: float = 1.96):
                if (n or 0.0) <= 0:
                    return (0.0, 1.0)
                z2 = z*z
                denom = 1.0 + z2/n
                center = (p + z2/(2.0*n)) / denom
                margin = z * (_m.sqrt((p*(1.0-p)/n) + (z2/(4.0*n*n))) / denom)
                return (max(0.0, center - margin), min(1.0, center + margin))
            ranked = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
            print("\n[explain]")
            for k, v in ranked[:5]:
                n = float(samples.get(k, 0.0))
                lo, hi = _wilson(float(v), n)
                print(f" - {k}: p={v:.2f} n={int(n)} CI95%=[{lo:.2f},{hi:.2f}]")
            if len(ranked) >= 2:
                (k1, p1), (k2, p2) = ranked[0], ranked[1]
                print(f" delta={p1-p2:.2f} top=({k1},{p1:.2f}) next=({k2},{p2:.2f})")
        return 0
    sh.set_defaults(func=_cmd_show)

    # Tag outcomes as useful/off (appends a tag record referencing last event)
    def _cmd_tag(args: argparse.Namespace) -> int:
        log = Path('out/experience/log.jsonl')
        if not log.exists():
            print('[learn] no experience log yet')
            return 1
        last = None
        for ln in reversed(log.read_text(encoding='utf-8', errors='ignore').splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
                last = rec
                break
            except Exception:
                continue
        if not last:
            print('[learn] no records found')
            return 1
        tag = 'useful' if getattr(args, 'useful', False) else ('off' if getattr(args, 'off', False) else None)
        if not tag:
            print('Usage: nerion learn tag --useful | --off')
            return 1
        Path('out/learning').mkdir(parents=True, exist_ok=True)
        tag_path = Path('out/learning/tags.jsonl')
        event = {'event_id': last.get('event_id'), 'tag': tag}
        with tag_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')
        print(json.dumps(event))
        return 0

    tg = sp.add_parser('tag', help='tag the last outcome as useful/off')
    g = tg.add_mutually_exclusive_group(required=True)
    g.add_argument('--useful', action='store_true')
    g.add_argument('--off', action='store_true')
    tg.set_defaults(func=_cmd_tag)

    # Personalization (persist tone/voice defaults)
    def _cmd_set(args: argparse.Namespace) -> int:
        prefs = load_prefs()
        if getattr(args, 'tone', None):
            prefs.setdefault('personalization', {})['tone'] = args.tone
        if getattr(args, 'voice_profile', None):
            prefs.setdefault('personalization', {})['voice_profile'] = args.voice_profile
        Path('out/learning').mkdir(parents=True, exist_ok=True)
        Path('out/learning/prefs.json').write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(prefs, indent=2))
        return 0

    st = sp.add_parser('set', help='set personalization defaults (tone/voice)')
    st.add_argument('--tone', choices=['concise', 'detailed'])
    st.add_argument('--voice-profile')
    st.set_defaults(func=_cmd_set)

    # Reset prefs (clear file)
    def _cmd_reset(_args: argparse.Namespace) -> int:
        from pathlib import Path
        p = Path('out/learning/prefs.json')
        try:
            if p.exists():
                p.unlink()
            print('[learn] prefs reset')
            return 0
        except Exception as e:
            print(f"[learn] reset failed: {e}")
            return 1

    rs = sp.add_parser('reset', help='reset learned preferences (delete prefs.json or clear an intent)')
    rs.add_argument('--intent', help='only clear per-intent learned maps for this intent')
    def _cmd_reset2(args: argparse.Namespace) -> int:
        intent = getattr(args, 'intent', None)
        if intent:
            try:
                prefs = load_prefs()
                for k in (
                    'tool_success_rate_by_intent',
                    'tool_sample_weight_by_intent',
                    'tool_success_weight_by_intent',
                ):
                    m = prefs.get(k)
                    if isinstance(m, dict):
                        m.pop(intent, None)
                        prefs[k] = m
                from selfcoder.learning.continuous import save_prefs as _sp
                _sp(prefs)
                print(f"[learn] cleared intent: {intent}")
                return 0
            except Exception as e:
                print(f"[learn] intent reset failed: {e}")
                return 1
        return _cmd_reset(args)
    rs.set_defaults(func=_cmd_reset2)

    # --- A/B experiment helpers ---
    AB_PATH = Path('out/learning/ab.json')

    def _cmd_ab_start(args: argparse.Namespace) -> int:
        spec = {
            'name': getattr(args, 'name', 'eval1'),
            'split': float(getattr(args, 'split', 0.5)),
            'assign_by': getattr(args, 'assign_by', 'query'),
            'arms': [s.strip() for s in str(getattr(args, 'arms', 'baseline,bandit+credit')).split(',') if s.strip()],
        }
        AB_PATH.parent.mkdir(parents=True, exist_ok=True)
        AB_PATH.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(spec, indent=2))
        return 0

    def _cmd_ab_stop(_args: argparse.Namespace) -> int:
        try:
            if AB_PATH.exists():
                AB_PATH.unlink()
            print('[learn.ab] stopped')
            return 0
        except Exception as e:
            print(f"[learn.ab] stop failed: {e}")
            return 1

    def _cmd_ab_status(args: argparse.Namespace) -> int:
        # Optionally refresh learning summary before reporting
        if getattr(args, 'refresh', False):
            try:
                review_outcomes()
            except Exception:
                pass
        status = {'active': False}
        if AB_PATH.exists():
            try:
                data = json.loads(AB_PATH.read_text(encoding='utf-8'))
                data['active'] = True
                status.update({'config': data, 'active': True})
            except Exception:
                status.update({'error': 'invalid ab.json'})
        # Load decisions and guardrails from prefs
        try:
            prefs = load_prefs()
            decisions = prefs.get('experiments_meta') or {}
            guard = prefs.get('guardrails') or {}
            status['decisions'] = decisions
            status['guardrails'] = guard
        except Exception:
            pass
        print(json.dumps(status, indent=2))
        return 0

    ab = sp.add_parser('ab', help='A/B evaluation controls')
    absp = ab.add_subparsers(dest='ab_cmd', required=True)
    ab_start = absp.add_parser('start', help='Start an experiment with local assignment')
    ab_start.add_argument('--name', required=True)
    ab_start.add_argument('--split', type=float, default=0.5)
    ab_start.add_argument('--assign-by', choices=['query', 'session'], default='query')
    ab_start.add_argument('--arms', default='baseline,bandit+credit', help='comma-separated arm names')
    ab_start.set_defaults(func=_cmd_ab_start)

    ab_stop = absp.add_parser('stop', help='Stop the current experiment')
    ab_stop.set_defaults(func=_cmd_ab_stop)

    ab_status = absp.add_parser('status', help='Show current experiment status (config, decisions, guardrails)')
    ab_status.add_argument('--refresh', action='store_true', help='refresh learning summary before reporting')
    ab_status.set_defaults(func=_cmd_ab_status)

    # Export markdown report
    def _cmd_export(args: argparse.Namespace) -> int:
        out = Path(getattr(args, 'out', 'out/learning/report.md'))
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            from selfcoder.learning.report import make_report
        except Exception as e:
            print(f"[learn] export unavailable: {e}")
            return 1
        try:
            md = make_report(window=str(getattr(args, 'window', '30d') or '30d'))
            out.write_text(md, encoding='utf-8')
            print(f"[learn] wrote {out}")
            return 0
        except Exception as e:
            print(f"[learn] export failed: {e}")
            return 1

    ex = sp.add_parser('export', help='export a learning report (markdown)')
    ex.add_argument('--window', default='30d')
    ex.add_argument('--out', default='out/learning/report.md')
    ex.set_defaults(func=_cmd_export)

    # Replay summary from rotated logs
    def _cmd_replay(args: argparse.Namespace) -> int:
        since = str(getattr(args, 'since', '30d') or '30d').strip()
        try:
            from selfcoder.learning.report import summarize_from_logs
            rep = summarize_from_logs(window=since)
            print(json.dumps(rep, indent=2))
            return 0
        except Exception as e:
            print(f"[learn] replay failed: {e}")
            return 1

    rp = sp.add_parser('replay', help='replay window from rotated logs and print a summary (no writes)')
    rp.add_argument('--since', default='30d', help='time window, e.g., 7d, 24h, 90d')
    rp.set_defaults(func=_cmd_replay)
