from __future__ import annotations

import json
import os
from pathlib import Path


def test_latency_log_written_for_streaming(monkeypatch, tmp_path):
    import app.chat.voice_io as vio

    # Redirect out/voice to a temp dir by monkeypatching os.path.join locally used in voice_io
    # We'll just ensure the directory exists and assert file grows after call.
    out_dir = Path("out/voice")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "latency.jsonl"
    if log_path.exists():
        before = log_path.read_text(encoding="utf-8")
    else:
        before = ""

    # Stub streaming transcriber and mic frames
    monkeypatch.setattr(vio, "transcribe_streaming", lambda *a, **k: "ok", raising=False)
    monkeypatch.setattr(vio, "_mic_frames_ptt", lambda watcher, device_hint: iter([b"\x00\x00" * 4000]), raising=False)

    # Invoke the internal function; watcher/device not needed for stub
    _ = vio._ptt_stream_transcribe(None, None)

    assert log_path.exists()
    after = log_path.read_text(encoding="utf-8")
    assert len(after) > len(before)
    # Last line should be valid JSON with duration_ms and backend keys
    last = [ln for ln in after.splitlines() if ln.strip()][-1]
    rec = json.loads(last)
    assert isinstance(rec.get("duration_ms"), int)
    assert "backend" in rec

