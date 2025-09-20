# Local Models (Offline)

Nerion supports multiple offline backends for coder models. Choose one via `NERION_CODER_BACKEND` and set the matching environment variables.

Backends
- ollama (default): uses `langchain_ollama.ChatOllama`.
  - Vars: `NERION_CODER_MODEL` (e.g., `deepseek-coder-v2`), `NERION_CODER_BASE_URL` (e.g., `http://localhost:11434`).
- llama_cpp: runs GGUF locally via `llama_cpp`.
  - Vars: `NERION_CODER_BACKEND=llama_cpp`, `LLAMA_CPP_MODEL_PATH=/path/to/model.gguf`.
  - Catalog auto-fill: `nerion models ensure --auto` uses `config/model_catalog.yaml` to set `LLAMA_CPP_MODEL_URL` and a default download path under `~/.cache/nerion/models/llama.cpp/`, then downloads with your consent (`NERION_ALLOW_NETWORK=1`).
- vllm: OpenAI-compatible HTTP endpoint.
  - Vars: `NERION_CODER_BACKEND=vllm`, `NERION_CODER_BASE_URL=http://localhost:8000`.
  - Autostart (opt-in): set `NERION_AUTOSTART_VLLM=1` and Nerion will try to spawn a local vLLM server using the catalog repo when needed.
- exllamav2 (experimental): optional bindings; CPU/GPU local.
  - Vars: `NERION_CODER_BACKEND=exllamav2`, `EXLLAMA_MODEL_DIR=/path/to/model/dir`.
  - Auto-prepare (opt-in): set `NERION_AUTOSTART_EXL=1` and Nerion will try to clone a compatible repo from the catalog into `~/.cache/nerion/models/exl/<model>` if missing.
  - Note: Python bindings and APIs vary across releases; Nerion attempts a best‑effort integration and will print a clear warning if the bindings are unavailable. Consider vLLM or Ollama as the most robust paths.

Models
- Coder families (quantized supported): `deepseek-coder-v2`, `qwen2.5-coder`, `starcoder2`, `codellama`.
- Set `NERION_CODER_MODEL` (you can also prefix as `ollama:deepseek-coder-v2`, etc.).

CLI Bench (latency)
```bash
nerion models bench \
  --backends ollama llama_cpp vllm exllamav2 \
  --models deepseek-coder-v2 qwen2.5-coder starcoder2 codellama
```
Prints a simple table: `backend\tmodel\tlatency_s`. Unavailable combos show `n/a`.

Auto-select and provision
- Auto-select: set `NERION_CODER_AUTO=1` and run `nerion plan --llm` — Nerion picks a local model automatically (task-aware where possible).
- Provision with consent: `nerion models ensure --auto` prompts or performs pulls/downloads (Ollama: pull; llama.cpp: GGUF via catalog). For vLLM/exllamav2, it prints exact commands/paths to run locally.

Planner Integration
- `nerion plan --llm` uses the configured backend.
- `--json-grammar` forces strict JSON planning with schema validation.
- `NERION_LLM_STRICT=1` surfaces planner errors instead of silently falling back.

Plan Cache
- Plans are normalized and cached at `.nerion/plan_cache.json`, keyed by repo fingerprint, target, and instruction.

Troubleshooting
- Empty outputs: verify backend env vars and that the service (e.g., Ollama) is running.
- Strict JSON errors: remove `--json-grammar` or set a compatible model; otherwise fix backend config.
