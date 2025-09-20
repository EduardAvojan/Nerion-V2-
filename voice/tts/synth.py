import logging
from typing import Optional

try:
    import pyttsx3  # type: ignore
except Exception:  # pragma: no cover
    pyttsx3 = None  # type: ignore

_logger = logging.getLogger(__name__)
_engine: Optional["pyttsx3.Engine"] = None


def _ensure_engine() -> Optional["pyttsx3.Engine"]:
    global _engine
    if _engine is not None:
        return _engine
    if pyttsx3 is None:
        return None
    try:
        _engine = pyttsx3.init()
    except Exception as e:  # pragma: no cover
        _logger.warning("pyttsx3 init failed in synth: %s", e)
        _engine = None
    return _engine


def speak(text: str, rate: float = 1.0):
    """Basic speak fallback. Projects that own TTS may override this."""
    eng = _ensure_engine()
    if eng is None:
        print(f"[TTS] {text}")
        return
    try:
        # Map rate multiplier to pyttsx3 integer rate if possible
        try:
            base_rate = eng.getProperty("rate") or 200
            eng.setProperty("rate", int(base_rate * max(0.1, float(rate))))
        except Exception:
            pass
        eng.say(text)
        eng.runAndWait()
    except Exception as e:  # pragma: no cover
        _logger.warning("TTS speak failed: %s", e)
        print(f"[TTS] {text}")


def cancel():
    """Attempt to stop any ongoing speech promptly (safe if no engine)."""
    eng = _engine or _ensure_engine()
    if eng is None:
        return
    try:
        # Primary stop
        eng.stop()
        try:
            # Some engines require ending the event loop to take effect immediately
            eng.endLoop()
        except Exception:
            pass
        # If still speaking, try a brief retry loop
        for _ in range(4):  # ~200ms total
            try:
                if not eng.isBusy():
                    break
            except Exception:
                break
            import time as _t
            _t.sleep(0.05)
            try:
                eng.stop()
            except Exception:
                pass
    except Exception:  # pragma: no cover
        pass


def is_busy() -> bool:
    """Return True if the TTS engine is currently speaking (best-effort)."""
    eng = _engine or _ensure_engine()
    if eng is None:
        return False
    try:
        return bool(eng.isBusy())
    except Exception:
        return False