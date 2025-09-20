Voice Stack, Offline STT, and Hot‑Reload
=======================================

Offline STT Toggle
------------------
- Set `NERION_STT_OFFLINE=1` to prefer offline speech recognition via PocketSphinx.
- The pipeline in `app/chat/voice_io.py` will attempt `recognize_sphinx()` first and
  fall back to the network recognizer if offline decoding fails.
- Recommended extras: install `pocketsphinx` for best offline results.

TTS Reset for Hot‑Reload
------------------------
- `app/chat/tts_router.py` exposes `reset()` to tear down state safely:
  - stops current speech,
  - terminates any macOS `say` process,
  - clears the pyttsx3 engine and worker thread handles.
- Call `reset()` when hot‑reloading the chat loop or re‑initializing TTS settings
  to avoid orphan threads or stale audio handles.

When to Call Reset()
--------------------
- Before reloading modules that import or initialize TTS.
- After changing TTS backend/rate/voice at runtime.
- On device errors (AUHAL/PaMacCore), cancel and `reset()` to recover.

Concurrency Notes
-----------------
- TTS worker initialization is guarded by a lock; repeated init is safe.
- Chat state is injected into `voice_io` via `set_voice_state`; avoid hot‑reloading
  while threads are mid‑turn if possible.

Voice Backends (Offline)
------------------------
Nerion supports multiple local STT/TTS backends. The built‑ins work out of the box
(`pyttsx3` TTS, PocketSphinx STT). You can optionally enable stronger local models:

- TTS: Piper (ONNX) or Coqui TTS (CLI)
- STT: whisper.cpp (GGML/GGUF) or Whisper (PyTorch)

Quick setups (examples)
1) Piper TTS (offline)

```bash
# Install Piper CLI (choose your platform) and download a local voice model.
# Example env (adjust the model path to your file):
export NERION_TTS_BACKEND=piper
export PIPER_MODEL_PATH=/models/piper/en_US-amy-medium.onnx
```

2) Coqui TTS (CLI, offline)

```bash
# Install the coqui TTS CLI locally, download an offline model file.
export NERION_TTS_BACKEND=coqui
export COQUI_MODEL_PATH=/models/coqui/tts_model.pth
# Optional: if your CLI isn't named `tts`:
export COQUI_TTS_CMD=/path/to/tts
```

3) whisper.cpp STT (offline)

```bash
# Install a whisper.cpp Python binding and download a local GGML/GGUF model.
export NERION_STT_BACKEND=whisper.cpp
export WHISPER_CPP_MODEL_PATH=/models/whisper/ggml-base.en.bin
```

4) Whisper (PyTorch, offline if models are local)

```bash
export NERION_STT_BACKEND=whisper
export NERION_STT_MODEL=small  # tiny|base|small|medium|large
```

Selecting backends at runtime
-----------------------------
- Persist settings:
  - `nerion voice set --backend pyttsx3|say|piper|coqui`
- Temporary session:
  - `export NERION_TTS_BACKEND=pyttsx3|say|piper|coqui`
  - `export NERION_STT_BACKEND=whisper|whisper.cpp|vosk|sphinx|auto`

Notes
-----
- All backends operate fully offline with locally installed binaries/models.
- Piper/Coqui synthesize to a temporary WAV and use a native player (`afplay`/`aplay`/`ffplay`).
- If a requested backend/tool is missing, Nerion falls back to a working default
  (e.g., `say` on macOS or `pyttsx3` elsewhere; Sphinx for strict‑offline STT).
