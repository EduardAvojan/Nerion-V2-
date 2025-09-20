from __future__ import annotations

import argparse
import json
import shutil
from typing import Any, Dict, List, Tuple
from core.ui.progress import progress
from core.ui.messages import fmt as _fmt_msg
from core.ui.messages import Result as _MsgRes
from pathlib import Path


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def _check_import(mod: str) -> Tuple[bool, str]:
    try:
        __import__(mod)
        return True, f"import {mod}"
    except Exception as e:
        return False, f"import {mod} failed: {e}"


def _check_microphones() -> Tuple[bool, Dict[str, Any]]:
    try:
        import speech_recognition as sr  # type: ignore
    except Exception as e:
        return False, {
            "component": "microphone",
            "ok": False,
            "detail": f"SpeechRecognition not available: {e}",
            "remedy": [
                "pip install SpeechRecognition",
                "Ensure Python has access to audio devices",
                "On macOS: System Settings → Privacy & Security → Microphone → allow Terminal/VS Code",
            ],
        }
    try:
        names = sr.Microphone.list_microphone_names() or []
    except Exception as e:
        return False, {
            "component": "microphone",
            "ok": False,
            "detail": f"Could not list microphones: {e}",
            "remedy": [
                "Check OS microphone permissions",
                "Ensure an input device is connected and not exclusively in use",
            ],
        }
    ok = len(names) > 0
    return ok, {
        "component": "microphone",
        "ok": ok,
        "detail": f"{len(names)} device(s) detected",
        "devices": names[:20],
        "remedy": ([] if ok else [
            "Connect a microphone or select the correct input device",
            "On macOS: grant microphone permission and restart the app",
        ]),
    }


def _check_tts() -> Tuple[bool, Dict[str, Any]]:
    # pyttsx3 engine
    ok_engine = False
    engine_err = None
    try:
        import pyttsx3  # type: ignore
        try:
            eng = pyttsx3.init()
            # best-effort quick property set
            eng.setProperty('rate', int(180))
            ok_engine = True
        except Exception as e:
            engine_err = str(e)
    except Exception as e:
        engine_err = f"pyttsx3 import failed: {e}"

    # macOS 'say' availability
    say_path = shutil.which("say")
    say_ok = bool(say_path)

    ok = ok_engine or say_ok
    detail = []
    if ok_engine:
        detail.append("pyttsx3: OK")
    else:
        detail.append(f"pyttsx3: missing/failed ({engine_err})")
    detail.append(f"say: {'OK' if say_ok else 'missing'}")

    remedy: List[str] = []
    if not ok_engine:
        remedy.append("pip install pyttsx3")
        remedy.append("If on Linux, install a speech backend (e.g., espeak-ng)")
    if not say_ok:
        remedy.append("On macOS, the built-in 'say' command should exist; if not, check PATH or use pyttsx3")

    return ok, {
        "component": "tts",
        "ok": ok,
        "detail": "; ".join(detail),
        "remedy": remedy,
    }


def _check_offline_stt() -> Tuple[bool, Dict[str, Any]]:
    # Optional offline STT: pocketsphinx
    try:
        ok = True
        detail = "pocketsphinx present"
        remedy: List[str] = []
    except Exception as e:
        ok = False
        detail = f"pocketsphinx missing: {e}"
        remedy = [
            "pip install pocketsphinx",
            "Set NERION_STT_OFFLINE=1 to prefer offline recognition",
        ]
    return ok, {
        "component": "offline_stt",
        "ok": ok,
        "detail": detail,
        "remedy": remedy,
    }


def _doctor_report() -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []

    # Imports
    ok_sr, sr_msg = _check_import("speech_recognition")
    items.append({"component": "SpeechRecognition", "ok": ok_sr, "detail": sr_msg, "remedy": ["pip install SpeechRecognition"] if not ok_sr else []})

    # Microphones
    _, m = _check_microphones()
    items.append(m)

    # TTS backends
    _, tts = _check_tts()
    items.append(tts)

    # Offline STT (optional)
    _, off = _check_offline_stt()
    items.append(off)

    # Static gate (optional: may be slow depending on tools)
    try:
        from selfcoder.security.extlinters import run_on_dir as _run_ext, summarize as _summ
        root = Path('.')
        with progress("doctor: static gate"):
            findings = _run_ext(root)
            summ = _summ(findings)
        ok = summ.get('score', 0) == 0
        items.append({
            'component': 'static_gate',
            'ok': ok,
            'detail': f"risk_score={summ.get('score',0)} counts={summ.get('counts',{})}",
            'remedy': [
                "Enable/Install ruff, mypy, bandit, semgrep (optional) for full coverage",
                "Run 'nerion plan --json-grammar' for strict plan validation",
            ],
        })
    except Exception as e:
        items.append({
            'component': 'static_gate',
            'ok': False,
            'detail': f"static gate unavailable: {e}",
            'remedy': ["pip install ruff mypy bandit semgrep (optional)"]
        })

    # Summarize
    # Overall status excludes optional components (offline_stt, static_gate)
    ok_all = all(
        x.get("ok")
        for x in items
        if x.get("component") not in {"offline_stt", "static_gate"}
    )
    return {"ok": ok_all, "items": items}


def _print_human(report: Dict[str, Any]) -> None:
    print("[doctor] Nerion voice & audio report")
    for it in report.get("items", []):
        mark = _mark(bool(it.get("ok")))
        name = it.get("component", "?")
        detail = it.get("detail", "")
        print(f" {mark} {name}: {detail}")
        remedies = it.get("remedy") or []
        if (not it.get("ok")) and remedies:
            for r in remedies[:4]:
                print(f"    → {r}")
    print(f"[doctor] Overall: {'OK' if report.get('ok') else 'Needs attention'}")


def _cmd_doctor(args: argparse.Namespace) -> int:
    rep = _doctor_report()
    if getattr(args, "json", False):
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        _print_human(rep)
        print(_fmt_msg('doctor', 'summary', _MsgRes.OK if rep.get('ok') else _MsgRes.FAIL, 'voice/io/linters'))
    return 0 if rep.get("ok") else 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("doctor", help="check mic, TTS/STT backends, and give remedies")
    p.add_argument("--json", action="store_true", help="output JSON instead of human text")
    p.set_defaults(func=_cmd_doctor)
