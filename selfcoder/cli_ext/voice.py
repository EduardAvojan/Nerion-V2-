from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml
from ops.security import fs_guard

# ---- helpers ---------------------------------------------------------------

def _settings_file_path() -> Path:
    """Return app/settings.yaml guarded to stay inside the repo root."""
    root = Path(__file__).resolve().parents[2]
    target = (root / "app" / "settings.yaml").resolve()
    # Enforce repo jail: this raises RepoPathViolation if the path escapes
    return fs_guard.ensure_in_repo(root, str(target))


def _read_yaml(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        return {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    tmp.replace(path)


# ---- command implementations ---------------------------------------------

def cmd_voice_diagnose(args: argparse.Namespace) -> int:
    """Live voice diagnostics: prints device info (if available), VAD/STT hints.
    This command is resilient: if the underlying modules don't expose
    diagnostics yet, it will print helpful guidance instead of failing.
    """
    # Defer imports so CLI stays fast and optional deps don't break global import
    try:
        from voice.stt import recognizer as _rec
    except Exception as e:
        print("[voice] recognizer unavailable; to enable, grant access to voice/stt/recognizer.py and add a diagnose entrypoint.")
        print(f"[voice] import error: {e}")
        return 0

    # Try to call a best-effort diagnostic hook if present
    duration = getattr(args, "duration", 10)
    device = getattr(args, "device", None)
    sensitivity = getattr(args, "vad_sensitivity", None)
    min_speech_ms = getattr(args, "min_speech_ms", None)
    silence_tail_ms = getattr(args, "silence_tail_ms", None)

    # Print banner so users know what to expect
    print("[voice] starting diagnostics…")
    print(f"  duration: {duration}s  device: {device!r}")
    if sensitivity is not None:
        print(f"  vad.sensitivity: {sensitivity}")
    if min_speech_ms is not None or silence_tail_ms is not None:
        print(f"  vad.min_speech_ms: {min_speech_ms}  vad.silence_tail_ms: {silence_tail_ms}")

    # Preferred: recognizer exposes `run_diagnostics(**kwargs)`
    try:
        if hasattr(_rec, "run_diagnostics") and callable(_rec.run_diagnostics):
            return int(not bool(_rec.run_diagnostics(
                duration=duration,
                device=device,
                vad_sensitivity=sensitivity,
                min_speech_ms=min_speech_ms,
                silence_tail_ms=silence_tail_ms,
                color=not getattr(args, "no_color", False),
            )))
    except Exception as e:
        print(f"[voice] run_diagnostics raised: {e}")

    # Fallback: recognizer exposes `list_input_devices()` and maybe a basic meter
    try:
        if hasattr(_rec, "list_input_devices"):
            devices = _rec.list_input_devices()
            print("[voice] input devices:")
            for d in devices or []:
                print(f"  - {d}")
    except Exception as e:
        print(f"[voice] device listing failed: {e}")

    # Basic amplitude meter (fallback) using PyAudio
    try:
        import pyaudio  # type: ignore
        import struct
        import math
        pa = pyaudio.PyAudio()
        try:
            rate = 16000
            chunk_ms = 100
            frames_per_buffer = int(rate * (chunk_ms / 1000.0))
            kwargs = dict(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=frames_per_buffer)
            if device:
                # Best-effort match by name substring
                try:
                    for i in range(pa.get_device_count()):
                        info = pa.get_device_info_by_index(i)
                        if device.lower() in str(info.get('name','')).lower() and int(info.get('maxInputChannels') or 0) > 0:
                            kwargs['input_device_index'] = i
                            break
                except Exception:
                    pass
            stream = pa.open(**kwargs)
            print('[voice] amplitude meter: speak into the mic…')
            steps = max(1, int(duration * (1000 / chunk_ms)))
            for _ in range(steps):
                data = stream.read(frames_per_buffer, exception_on_overflow=False)
                # Convert to RMS
                try:
                    count = len(data) // 2
                    shorts = struct.unpack('<' + 'h' * count, data[: count * 2])
                    rms = math.sqrt(sum(s * s for s in shorts) / max(1, count))
                    # Normalize bar to 0..40 approx
                    lvl = int(min(40, (rms / 1000.0) * 40))
                except Exception:
                    lvl = 0
                bar = '#' * lvl
                print(f"[ {bar:<40} ] {lvl:02d}")
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        finally:
            try:
                pa.terminate()
            except Exception:
                pass
    except Exception as e:
        print(f"[voice] PyAudio meter unavailable: {e}")

    # Final message to guide next steps
    print("[voice] diagnostics hook not implemented in recognizer; add `run_diagnostics()` to enable live levels/VAD/partials.")
    return 0


def cmd_voice_set(args: argparse.Namespace) -> int:
    """Persist voice settings to app/settings.yaml."""
    path = _settings_file_path()
    cfg = _read_yaml(path)
    voice = cfg.get("voice")
    if not isinstance(voice, dict):
        voice = {}
    # top-level
    if getattr(args, "device", None) is not None:
        voice["device"] = args.device
    if getattr(args, "backend", None) is not None:
        voice["tts_backend"] = args.backend
    if getattr(args, "mode", None) is not None:
        voice["mode"] = args.mode
    if getattr(args, "barge_in", None) is not None:
        voice["barge_in"] = bool(args.barge_in)

    if hasattr(args, "always_speak"):
        if args.always_speak is True:
            voice["always_speak"] = True
        elif args.always_speak is False:
            voice["always_speak"] = False

    # nested VAD
    vad = voice.get("vad")
    if not isinstance(vad, dict):
        vad = {}
    if getattr(args, "vad_sensitivity", None) is not None:
        s = int(args.vad_sensitivity)
        # Clamp to [0, 10]
        s = 0 if s < 0 else (10 if s > 10 else s)
        vad["sensitivity"] = s
    if getattr(args, "min_speech_ms", None) is not None:
        ms = int(args.min_speech_ms)
        vad["min_speech_ms"] = ms if ms > 0 else 0
    if getattr(args, "silence_tail_ms", None) is not None:
        tail = int(args.silence_tail_ms)
        vad["silence_tail_ms"] = tail if tail > 0 else 0
    if vad:
        voice["vad"] = vad

    cfg["voice"] = voice
    _write_yaml(path, cfg)
    # friendly echo of what was set
    print("[voice.set] wrote app/settings.yaml")
    subset = {k: voice.get(k) for k in ("device", "tts_backend", "mode", "barge_in", "always_speak")}
    if "vad" in voice:
        subset["vad"] = voice["vad"]
    print(json.dumps(subset, indent=2))
    return 0


def cmd_voice_show(args: argparse.Namespace) -> int:
    """Show current voice settings from app/settings.yaml."""
    path = _settings_file_path()
    cfg = _read_yaml(path)
    voice = cfg.get("voice", {})
    print(json.dumps(voice, indent=2))
    return 0


# ---- registration ---------------------------------------------------------

def register(sub: argparse._SubParsersAction) -> None:
    """Register the `voice` command tree under the given subparsers action."""
    p_voice = sub.add_parser("voice", help="voice utilities (diagnostics, etc.)")
    p_voice_sub = p_voice.add_subparsers(dest="voice_cmd", required=True)

    p_voice_diag = p_voice_sub.add_parser("diagnose", help="live mic/VAD/STT diagnostics")
    p_voice_diag.add_argument("--duration", type=int, default=10, help="seconds to run diagnostics")
    p_voice_diag.add_argument("--device", help="input device id/name (optional)")
    p_voice_diag.add_argument("--vad-sensitivity", type=int, help="VAD sensitivity (0–10)")
    p_voice_diag.add_argument("--min-speech-ms", type=int, help="minimum speech duration to count as start (ms)")
    p_voice_diag.add_argument("--silence-tail-ms", type=int, help="silence required to end speech (ms)")
    p_voice_diag.add_argument("--no-color", action="store_true", help="disable ANSI colors in output")
    p_voice_diag.set_defaults(func=cmd_voice_diagnose)

    p_voice_set = p_voice_sub.add_parser("set", help="persist voice settings to app/settings.yaml")
    p_voice_set.add_argument("--device", help="input device name/substring (e.g. 'Studio Display Microphone')")
    p_voice_set.add_argument("--backend", choices=["auto", "say", "pyttsx3", "piper", "coqui"], help="TTS backend preference")
    p_voice_set.add_argument("--mode", choices=["ptt", "vad"], help="listening mode")
    grp = p_voice_set.add_mutually_exclusive_group()
    grp.add_argument("--barge-in", dest="barge_in", action="store_true", help="enable acoustic barge-in (VAD mode)")
    grp.add_argument("--no-barge-in", dest="barge_in", action="store_false", help="disable acoustic barge-in")
    grp2 = p_voice_set.add_mutually_exclusive_group()
    grp2.add_argument("--always-speak", dest="always_speak", action="store_true", help="start with speech enabled")
    grp2.add_argument("--no-always-speak", dest="always_speak", action="store_false", help="start with speech disabled")
    p_voice_set.add_argument("--vad-sensitivity", type=int, help="0–10 (higher = less sensitive)")
    p_voice_set.add_argument("--min-speech-ms", type=int, help="VAD: ms above threshold to start speech")
    p_voice_set.add_argument("--silence-tail-ms", type=int, help="VAD: ms below threshold to end speech")
    p_voice_set.set_defaults(func=cmd_voice_set)

    p_voice_show = p_voice_sub.add_parser("show", help="print current voice settings")
    p_voice_show.set_defaults(func=cmd_voice_show)

    # List input devices
    def cmd_voice_devices(_args: argparse.Namespace) -> int:
        try:
            import pyaudio  # type: ignore
            pa = pyaudio.PyAudio()
            try:
                print('[voice] input devices:')
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    name = str(info.get('name') or '')
                    ch = int(info.get('maxInputChannels') or 0)
                    if ch > 0:
                        print(f"  [{i}] {name} (channels={ch})")
            finally:
                try:
                    pa.terminate()
                except Exception:
                    pass
        except Exception as e:
            print(f"[voice] device enumeration unavailable: {e}")
        return 0

    p_voice_dev = p_voice_sub.add_parser('devices', help='list available input devices')
    p_voice_dev.set_defaults(func=cmd_voice_devices)

    # quick toggles for speech default (persisted)
    p_voice_on = p_voice_sub.add_parser("on", help="persist: speech enabled by default")
    p_voice_on.set_defaults(func=lambda a: cmd_voice_set(argparse.Namespace(device=None, backend=None, mode=None, barge_in=None, vad_sensitivity=None, min_speech_ms=None, silence_tail_ms=None, always_speak=True)))

    p_voice_off = p_voice_sub.add_parser("off", help="persist: speech disabled by default")
    p_voice_off.set_defaults(func=lambda a: cmd_voice_set(argparse.Namespace(device=None, backend=None, mode=None, barge_in=None, vad_sensitivity=None, min_speech_ms=None, silence_tail_ms=None, always_speak=False)))

    # STT metrics summary from latency log
    def _cmd_metrics(args: argparse.Namespace) -> int:
        import json as _json
        from statistics import mean
        N = int(getattr(args, 'last', 100) or 100)
        path = Path('out/voice/latency.jsonl')
        if not path.exists():
            print('[voice.metrics] no latency log found')
            return 0
        rows = []
        try:
            lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
            for ln in reversed(lines):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(_json.loads(ln))
                except Exception:
                    continue
                if len(rows) >= N:
                    break
        except Exception:
            rows = []
        if not rows:
            print('[voice.metrics] no records')
            return 0
        # Group by (backend, model)
        from collections import defaultdict
        groups = defaultdict(list)
        for r in rows:
            key = (str(r.get('backend') or 'unknown'), str(r.get('model') or ''))
            try:
                groups[key].append(int(r.get('duration_ms') or 0))
            except Exception:
                continue
        out = []
        for (backend, model), vals in groups.items():
            vals = [v for v in vals if isinstance(v, int) and v >= 0]
            if not vals:
                continue
            p95 = sorted(vals)[max(0, int(len(vals) * 0.95) - 1)]
            out.append({
                'backend': backend,
                'model': model,
                'count': len(vals),
                'avg_ms': int(mean(vals)),
                'p95_ms': int(p95),
            })
        print(_json.dumps({'samples': len(rows), 'groups': out}, indent=2))
        return 0

    m = p_voice_sub.add_parser('metrics', help='summarize STT latency over last N turns')
    m.add_argument('--last', type=int, default=100, help='number of recent entries to include (default 100)')
    m.set_defaults(func=_cmd_metrics)
