from __future__ import annotations

import argparse
import os
import time
from typing import List
from pathlib import Path
import sys


def _bench_once(model: str, backend: str) -> float | None:
    try:
        from app.parent.coder import Coder
    except Exception:
        return None
    coder = Coder(model=model, backend=backend)
    t0 = time.time()
    out = coder.complete("Reply with OK only.")
    if not out:
        return None
    dt = time.time() - t0
    return dt


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("models", help="local model utilities")
    sp = p.add_subparsers(dest="models_cmd", required=True)

    b = sp.add_parser("bench", help="run a tiny latency bench for available backends/models")
    b.add_argument("--backends", nargs="*", default=["ollama", "llama_cpp", "vllm", "exllamav2"])
    b.add_argument("--language", choices=["py","ts","js"], help="Filter model list by language preference")
    b.add_argument("--models", nargs="*", default=[
        "deepseek-coder-v2",
        "qwen2.5-coder",
        "starcoder2",
        "codellama",
    ])

    def _run_bench(args: argparse.Namespace) -> int:
        rows: List[str] = []
        rows.append("backend\tmodel\tlatency_s")
        models = args.models
        if args.language == 'ts':
            models = ["qwen2.5-coder", "deepseek-coder-v2", "starcoder2", "codellama"]
        elif args.language == 'js':
            models = ["qwen2.5-coder", "deepseek-coder-v2", "starcoder2", "codellama"]
        for be in args.backends:
            for m in models:
                dt = _bench_once(m, be)
                rows.append(f"{be}\t{m}\t{dt if dt is not None else 'n/a'}")
        print("\n".join(rows))
        return 0

    b.set_defaults(func=_run_bench)

    e = sp.add_parser("ensure", help="auto-select and provision a local model if missing")
    e.add_argument("--auto", action="store_true", help="auto-select best model based on availability")
    e.add_argument("--backend", help="backend override")
    e.add_argument("--model", help="model override")

    def _run_ensure(args: argparse.Namespace) -> int:
        try:
            if args.auto or (not args.backend and not args.model):
                from app.parent.selector import auto_select_model
                choice = auto_select_model()
                if not choice:
                    print("No local backends detected; configure one first.")
                    return 2
                be, m, base = choice
            else:
                be, m = (args.backend or "ollama"), (args.model or "deepseek-coder-v2")
            # Resolve llama.cpp catalog defaults if missing
            if be == "llama_cpp" and not (os.getenv("LLAMA_CPP_MODEL_URL") and os.getenv("LLAMA_CPP_MODEL_PATH")):
                try:
                    from app.parent.model_catalog import resolve
                    src = resolve("llama_cpp", m)
                    if src and isinstance(src.get("url"), str):
                        import os as _os
                        import pathlib as _pl
                        _os.environ.setdefault("LLAMA_CPP_MODEL_URL", src["url"]) 
                        fn = src.get("filename") or src["url"].rsplit("/", 1)[-1]
                        default_path = str((_pl.Path.home() / f".cache/nerion/models/llama.cpp/{fn}").resolve())
                        _os.environ.setdefault("LLAMA_CPP_MODEL_PATH", default_path)
                except Exception:
                    pass
            from app.parent.provision import ensure_available
            ok, msg = ensure_available(be, m)
            print(msg)
            return 0 if ok else 3
        except Exception as e:
            print(f"ensure error: {e}")
            return 4

    e.set_defaults(func=_run_ensure)

    r = sp.add_parser("router", help="explain router decision for a task")
    r.add_argument("--task", choices=["chat","code"], default="code")
    r.add_argument("-i", "--instruction")
    r.add_argument("-f", "--file")

    def _run_router(args: argparse.Namespace) -> int:
        try:
            from selfcoder.llm_router import apply_router_env
            be, m, base = apply_router_env(instruction=getattr(args, 'instruction', None), file=getattr(args, 'file', None), task=getattr(args, 'task', 'code'))
            import json as _json
            print(_json.dumps({"backend": be, "model": m, "base": base}, ensure_ascii=False))
            return 0
        except Exception as e:
            print(f"router explain error: {e}")
            return 2

    r.set_defaults(func=_run_router)

    # ---------------- A/B evaluation harness -----------------
    ab = sp.add_parser(
        "ab",
        help="run plan+apply across multiple coder models on small tasks and rank results",
    )
    ab.add_argument("--backend", default="ollama", help="coder backend (default: ollama)")
    ab.add_argument(
        "--candidates",
        nargs="*",
        default=["deepseek-coder-v2", "qwen2.5-coder", "starcoder2", "codellama"],
        help="model names to evaluate",
    )
    ab.add_argument(
        "--tasks",
        nargs="*",
        default=["py", "ts"],
        choices=["py", "ts", "js"],
        help="task set to run (default: py ts)",
    )
    ab.add_argument("--retries", type=int, default=0, help="retries per task/model on failure")
    ab.add_argument("--json", action="store_true", help="emit JSON summary")

    def _mk_py_project(root: Path) -> tuple[Path, Path]:
        src = root / "foo.py"
        src.write_text(
            """
def add(a, b):
    return a + b
""".strip()
            + "\n",
            encoding="utf-8",
        )
        tst = root / "tests"
        tst.mkdir(parents=True, exist_ok=True)
        (tst / "test_doc.py").write_text(
            """
import ast, pathlib
p = pathlib.Path(__file__).parent.parent / 'foo.py'
mod = ast.parse(p.read_text(encoding='utf-8'))
assert ast.get_docstring(mod) is not None, 'module docstring missing'
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return src, tst

    def _mk_ts_project(root: Path) -> Path:
        (root / "tsconfig.json").write_text(
            '{"compilerOptions":{"target":"ES2019","strict":true}}\n',
            encoding="utf-8",
        )
        src = root / "index.ts"
        src.write_text(
            """
export function add(a, b) {
  return a + b;
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return src

    def _run_pytest(root: Path, timeout: int = 120) -> bool:
        try:
            from ops.security.safe_subprocess import safe_run as _run
        except Exception:
            import subprocess

            def _run(argv, **kwargs):
                return subprocess.run(argv, **{k: v for k, v in kwargs.items() if k in ("cwd", "timeout", "check", "capture_output")})
        res = _run([sys.executable, "-m", "pytest", "-q"], cwd=root, timeout=timeout, check=False, capture_output=True)
        return res.returncode == 0

    def _run_tsc(root: Path, timeout: int = 60) -> tuple[bool, bool]:
        import shutil as _sh
        tsc = _sh.which("tsc")
        if not tsc:
            return (False, True)  # unavailable, treat as neutral pass
        try:
            import subprocess
            p = subprocess.run([tsc, "--noEmit"], cwd=root, text=True, capture_output=True, timeout=timeout)
            return (p.returncode == 0, True)
        except Exception:
            return (False, True)

    def _ab_run_one(model: str, backend: str, task: str) -> tuple[bool, float, str]:
        import time
        import tempfile
        import shutil
        t0 = time.time()
        tmp = Path(tempfile.mkdtemp(prefix=f"nerion_ab_{task}_"))
        note = ""
        try:
            os.environ["NERION_CODER_BACKEND"] = backend
            os.environ["NERION_CODER_MODEL"] = model
            os.environ.setdefault("NERION_LLM_STRICT", "1")
            # Build minimal project
            if task == "py":
                src, _tests = _mk_py_project(tmp)
                from selfcoder.planner.llm_planner import plan_with_llm
                plan = plan_with_llm("Insert a concise one-sentence module docstring at the top explaining this module.", str(src))
                actions = plan.get("actions") or []
                from selfcoder.orchestrator import run_actions_on_file as _apply
                _apply(src, actions, dry_run=False)
                ok = _run_pytest(tmp)
                if not ok:
                    note = "pytest failed"
                return (ok, time.time() - t0, note)
            elif task == "ts":
                src = _mk_ts_project(tmp)
                from selfcoder.planner.llm_planner import plan_with_llm
                plan = plan_with_llm("Add explicit TypeScript types for all parameters and return types so 'tsc --noEmit' passes.", str(src))
                actions = plan.get("actions") or []
                from selfcoder.orchestrator import run_actions_on_file as _apply
                _apply(src, actions, dry_run=False)
                ok, present = _run_tsc(tmp)
                if not present:
                    # Fallback heuristic: check for type annotations in output
                    got = src.read_text(encoding="utf-8")
                    ok = ":" in got and "export function" in got
                    note = "tsc not found; heuristic check" if ok else "tsc not found; heuristic failed"
                elif not ok:
                    note = "tsc reported errors"
                return (ok, time.time() - t0, note)
            elif task == "js":
                # JS: simple modernization (use const/arrow) — verify file changed
                src = tmp / "app.js"
                src.write_text("function add(a,b){return a+b}\n", encoding="utf-8")
                from selfcoder.planner.llm_planner import plan_with_llm
                plan = plan_with_llm("Modernize to use const and arrow function syntax.", str(src))
                actions = plan.get("actions") or []
                before = src.read_text(encoding="utf-8")
                from selfcoder.orchestrator import run_actions_on_file as _apply
                _apply(src, actions, dry_run=False)
                after = src.read_text(encoding="utf-8")
                ok = after != before and ("const" in after or "=>" in after)
                if not ok:
                    note = "modernization not detected"
                return (ok, time.time() - t0, note)
        except Exception as e:
            note = f"error: {e}"
            return (False, time.time() - t0, note)
        finally:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass

    def _ab_eval(args: argparse.Namespace) -> int:
        import time
        import json
        results = []
        _summary = {}
        for model in args.candidates:
            ok_count = 0
            total = 0
            total_time = 0.0
            per_task = []
            for task in args.tasks:
                total += 1
                ok, dt, note = _ab_run_one(model, args.backend, task)
                retries = int(args.retries or 0)
                r = 0
                while (not ok) and r < retries:
                    ok, dt2, note = _ab_run_one(model, args.backend, task)
                    dt += dt2
                    r += 1
                ok_count += 1 if ok else 0
                total_time += dt
                per_task.append({"task": task, "ok": ok, "time_s": round(dt, 3), "note": note})
            results.append({
                "model": model,
                "ok": ok_count,
                "total": total,
                "time_s": round(total_time, 3),
                "tasks": per_task,
            })
        # Rank: success desc, time asc
        results.sort(key=lambda x: (-int(x["ok"]), float(x["time_s"])) )
        if args.json:
            print(json.dumps({"results": results}, indent=2))
        else:
            print("model\tok/total\ttime_s\tnotes")
            for r in results:
                notes = "; ".join([f"{t['task']}:{'OK' if t['ok'] else 'FAIL'}{('('+t['note']+')') if t['note'] else ''}" for t in r["tasks"]])
                print(f"{r['model']}\t{r['ok']}/{r['total']}\t{r['time_s']}\t{notes}")
        # Log JSONL for tuning
        try:
            if (os.getenv('NERION_ROUTER_LOG') or '').strip().lower() in {'1','true','yes','on'}:
                import pathlib
                p = pathlib.Path('.nerion') / 'ab_results.jsonl'
                p.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps({"ts": int(time.time()), "results": results})
                with p.open('a', encoding='utf-8') as f:
                    f.write(line + "\n")
        except Exception:
            pass
        return 0

    ab.set_defaults(func=_ab_eval)
