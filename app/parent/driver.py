

from __future__ import annotations
from typing import Optional, Dict, List
import json
import os

from .prompt import build_master_prompt
from .schemas import ParentDecision
from .tools_manifest import ToolsManifest
from app.chat.providers import get_registry, ProviderNotConfigured, ProviderError
try:
    from selfcoder.learning.continuous import load_prefs as _load_prefs
    from selfcoder.learning.continuous import save_prefs as _save_prefs
except Exception:  # pragma: no cover
    _load_prefs = None
    _save_prefs = None


class ParentLLM:
    """Abstract interface for the DeepSeek Parent.
    Implement `.complete(messages)` to call your provider and return a string (JSON).
    """
    def complete(self, messages: List[Dict[str, str]]) -> str:  # pragma: no cover - provider specific
        raise NotImplementedError


class _RegistryParentLLM(ParentLLM):
    """Provider-registry-backed ParentLLM used in Nerion V2."""

    def __init__(self, *, role: str = "planner", temperature: float = 0.1) -> None:
        self._registry = get_registry()
        self.role = role
        self.temperature = float(temperature)

    def complete(self, messages: List[Dict[str, str]]) -> str:  # type: ignore[override]
        try:
            resp = self._registry.generate(
                role=self.role,
                messages=messages,
                temperature=self.temperature,
            )
        except ProviderNotConfigured as exc:
            raise RuntimeError(f"Planner provider not configured: {exc}") from exc
        except ProviderError as exc:
            raise RuntimeError(f"Planner provider error: {exc}") from exc
        return resp.text or '{"intent":"error","plan":[],"final_response":null,"confidence":0.0,"requires_network":false}'


class ParentDriver:
    def __init__(self, llm: ParentLLM, tools: ToolsManifest, coder_llm: Optional[ParentLLM] = None):
        self.llm = llm
        self.tools = tools
        self.coder_llm = coder_llm

    # --- routing helpers -------------------------------------------------
    @staticmethod
    def _looks_like_code_task(user_query: str) -> bool:
        try:
            q = (user_query or "").lower()
            keywords = [
                "code", "coding", "debug", "debugging", "refactor", "rename", "function", "class", "module", "docstring",
                "ast", "apply diff", "unified diff", "patch", "fix", "bug", "traceback", "pytest",
                "test", "coverage", "bench", "repair", "self-code", "self code", "self-improve", "self improve",
            ]
            if any(k in q for k in keywords):
                return True
            if ".py" in q:
                return True
            if any(tok in q for tok in ["def ", "class ", "import "]):
                return True
        except Exception:
            pass
        return False

    def _enforce_tool_usage(self, decision: ParentDecision, user_query: str) -> ParentDecision:
        """Architectural validation: prevent LLM from choosing 'respond' when tools are available."""
        # Skip enforcement if decision already uses tools or asks user
        has_tool_call = any(step.action == "tool_call" for step in (decision.plan or []))
        has_ask_user = any(step.action == "ask_user" for step in (decision.plan or []))

        if has_tool_call or has_ask_user:
            return decision

        # Analyze query to determine what tools should be used
        q = (user_query or "").lower()
        tool_name = None
        args = {}

        # File operations
        if any(word in q for word in ["file", "files"]):
            if any(word in q for word in ["recent", "modified", "changed", "updated", "latest"]):
                tool_name = "list_recent_files"
                args = {"limit": 10}
            elif any(word in q for word in ["find", "search", "locate", "where"]):
                tool_name = "find_files"
                args = {"pattern": "*"}

        # Web operations
        elif any(word in q for word in ["search", "google", "look up", "find online"]):
            tool_name = "web_search"
            args = {"query": user_query, "max_results": 5}

        # If we identified an applicable tool, enforce its usage
        if tool_name and any(t.name == tool_name for t in (self.tools.tools or [])):
            from .schemas import Step
            new_step = Step(
                action="tool_call",
                tool=tool_name,
                args=args,
                summary=f"Using {tool_name}",
            )
            return ParentDecision(
                intent=decision.intent or "tool_usage",
                plan=[new_step],
                final_response=None,
                confidence=decision.confidence,
                requires_network=(tool_name in ["web_search", "read_url"]),
                notes=f"Enforced: {tool_name}",
            )

        return decision

    def _get_coder_parent_llm(self) -> Optional[ParentLLM]:
        if self.coder_llm is not None:
            return self.coder_llm
        # Lazy build a ParentLLM wrapper around the Coder V2 adapter
        try:
            from app.parent.coder_v2 import DeepSeekCoderV2
        except Exception:
            return None

        class _CoderV2ParentLLM(ParentLLM):
            def __init__(self) -> None:
                self._coder = DeepSeekCoderV2()
            def complete(self, messages: List[Dict[str, str]]) -> str:  # type: ignore[override]
                system = ""
                user_parts: List[str] = []
                for m in messages or []:
                    role = (m.get('role') or '').lower()
                    if role == 'system':
                        system = (m.get('content') or '')
                    else:
                        user_parts.append(m.get('content') or '')
                prompt = "\n\n".join([p for p in user_parts if p])
                out = self._coder.complete_json(prompt, system=system)
                return out or '{"intent":"error","plan":[],"final_response":null,"confidence":0.0,"requires_network":false,"notes":"coder llm unavailable"}'

        try:
            self.coder_llm = _CoderV2ParentLLM()
            return self.coder_llm
        except Exception:
            return None

    def plan_and_route(
        self,
        user_query: str,
        *,
        context_snippet: Optional[str] = None,
        extra_policies: Optional[str] = None,
    ) -> ParentDecision:
        """Build prompt → call Parent → parse ParentDecision (with robust fallback)."""
        # Optional bias: include local tool success rates in prompt policies
        bias_block = None
        learned_weights_for_prompt: Optional[Dict[str, float]] = None
        try:
            if _load_prefs:
                prefs = _load_prefs()
                # Prefer per-intent rates when we can infer an intent; else fall back to global
                intent_key = None
                try:
                    # Allow an explicit override for tests or power users
                    _env_hint = (__import__('os').environ.get('NERION_INTENT_HINT') or '').strip()
                    if _env_hint:
                        intent_key = _env_hint
                    else:
                        # Best-effort offline intent hint from triage rules
                        from app.chat.intents import detect_intent as _detect
                        spec = _detect(user_query)
                        intent_key = getattr(spec, 'name', None)
                except Exception:
                    intent_key = None

                # Load candidate maps
                rates_global = (prefs.get('tool_success_rate') or prefs.get('tool_success_rate_global') or {})
                samples_global = (prefs.get('tool_sample_weight') or {})
                succ_global = (prefs.get('tool_success_weight') or prefs.get('tool_success_weight_global') or {})
                rates_by_int = (prefs.get('tool_success_rate_by_intent') or {})
                samples_by_int = (prefs.get('tool_sample_weight_by_intent') or {})
                succ_by_int = (prefs.get('tool_success_weight_by_intent') or {})

                if intent_key and isinstance(rates_by_int, dict) and rates_by_int.get(intent_key):
                    rates = dict(rates_by_int.get(intent_key) or {})
                    samples = dict(samples_by_int.get(intent_key) or {})
                    succ = dict(succ_by_int.get(intent_key) or {})
                else:
                    rates = dict(rates_global or {})
                    samples = dict(samples_global or {})
                    succ = dict(succ_global or {})

                def _wilson(p: float, n: float, z: float = 1.96) -> tuple[float, float]:
                    try:
                        if n <= 0:
                            return (0.0, 1.0)
                        import math as _m
                        z2 = z * z
                        denom = 1.0 + z2 / n
                        center = (p + z2 / (2.0 * n)) / denom
                        margin = z * ((_m.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom)
                        lo = max(0.0, center - margin)
                        hi = min(1.0, center + margin)
                        return (float(lo), float(hi))
                    except Exception:
                        return (0.0, 1.0)

                if isinstance(rates, dict) and rates:
                    # Filter by min samples (weighted) and format top-K
                    try:
                        min_s = float((__import__('os').environ.get('NERION_LEARN_MIN_SAMPLES') or '3').strip())
                    except Exception:
                        min_s = 3.0
                    try:
                        top_k = int((__import__('os').environ.get('NERION_LEARN_TOP_K') or '6').strip())
                    except Exception:
                        top_k = 6
                    # Policy-specific adjustments
                    try:
                        from selfcoder.config import get_policy as _get_policy
                        _pol = _get_policy()
                    except Exception:
                        _pol = 'balanced'
                    # Cap list to keep prompt lean
                    top_k = max(1, min(int(top_k), 6))
                    if _pol == 'fast':
                        # Keep prompt lighter in fast mode
                        top_k = max(1, min(top_k, 3))
                    if _pol == 'safe':
                        # Require more samples for stability
                        min_s = max(min_s, 5.0)

                    # Compute learned weights according to strategy (greedy/ucb/thompson)
                    try:
                        strategy = (__import__('os').environ.get('NERION_BANDIT_STRATEGY') or 'greedy').strip().lower()
                    except Exception:
                        strategy = 'greedy'
                    weights: Dict[str, float] = {}
                    # Deterministic sampling if requested
                    try:
                        seed_raw = (__import__('os').environ.get('NERION_BANDIT_SEED') or '').strip()
                        if seed_raw:
                            import random as _rnd
                            _rnd.seed(int(seed_raw))
                    except Exception:
                        pass
                    # Precompute totals for UCB
                    try:
                        T = sum(float(samples.get(k, 0.0)) for k in rates.keys())
                    except Exception:
                        T = 0.0
                    try:
                        import math as _m
                        import random as _rnd
                    except Exception:
                        _m = None  # type: ignore
                        _rnd = None  # type: ignore
                    for k, v in rates.items():
                        n = float(samples.get(k, 0.0) or 0.0)
                        if strategy == 'ucb' and _m is not None and n > 0.0 and T > 0.0:
                            c = float((__import__('os').environ.get('NERION_BANDIT_UCB_C') or '2.0').strip() or 2.0)
                            bonus = (c * (_m.log(T + 1.0) / n)) ** 0.5
                            weights[k] = float(v) + float(bonus)
                        elif strategy == 'thompson' and _rnd is not None:
                            # Use Beta posterior sampling from (success+1, fail+1)
                            s = float(succ.get(k, v * max(n, 0.0)))
                            f = max(0.0, n - s)
                            a = max(1.0, s + 1.0)
                            b = max(1.0, f + 1.0)
                            try:
                                w = _rnd.betavariate(a, b)
                            except Exception:
                                w = float(v)
                            weights[k] = float(w)
                        else:
                            # greedy
                            weights[k] = float(v)

                    # Filter by min samples for the bias block and construct top list
                    items = []
                    for k, w in weights.items():
                        s = float(samples.get(k, 0.0)) if isinstance(samples, dict) else 0.0
                        if s >= min_s:
                            items.append((k, float(w)))
                    if not items:
                        items = [(k, float(weights.get(k, rates.get(k, 0.0)))) for k in rates.keys()]
                    top = sorted(items, key=lambda kv: kv[1], reverse=True)[:top_k]

                    # Confidence/uncertainty gates (per-intent aware)
                    try:
                        conf_delta = float((__import__('os').environ.get('NERION_LEARN_CONFIDENCE_DELTA') or '0.10').strip())
                    except Exception:
                        conf_delta = 0.10
                    try:
                        min_intent_samples = float((__import__('os').environ.get('NERION_LEARN_MIN_SAMPLES_INTENT') or '3').strip())
                    except Exception:
                        min_intent_samples = 3.0
                    try:
                        z = float((__import__('os').environ.get('NERION_LEARN_CI_Z') or '1.96').strip())
                    except Exception:
                        z = 1.96
                    use_wilson = False
                    try:
                        use_wilson = ( (__import__('os').environ.get('NERION_LEARN_WILSON') or '').strip().lower() in {'1','true','yes','on'} )
                    except Exception:
                        use_wilson = False

                    # Policy interplay adjustments (safe vs fast)
                    try:
                        from selfcoder.config import get_policy as _get_policy
                        pol = _get_policy()
                    except Exception:
                        pol = 'balanced'
                    if pol == 'safe':
                        min_intent_samples = max(min_intent_samples, 8.0)
                        conf_delta = max(conf_delta, 0.15)
                    elif pol == 'fast':
                        conf_delta = min(conf_delta, 0.05)

                    # Gather CI and apply softening/fallback when not confident
                    confident = True
                    if intent_key and isinstance(samples, dict) and len(top) >= 2:
                        k1, w1 = top[0]
                        k2, w2 = top[1]
                        n1 = float(samples.get(k1, 0.0))
                        n2 = float(samples.get(k2, 0.0))
                        p1 = float(rates.get(k1, 0.0))
                        p2 = float(rates.get(k2, 0.0))
                        # Confidence criterion: sufficient samples AND delta above threshold
                        if (n1 < min_intent_samples or n2 < min_intent_samples) or ((p1 - p2) < conf_delta):
                            confident = False
                        # Optional Wilson overlap check
                        if confident and use_wilson:
                            try:
                                import math as _m
                                def _wilson(p, n, zv):
                                    z2 = zv*zv
                                    denom = 1.0 + z2/n
                                    center = (p + z2/(2.0*n)) / denom
                                    margin = zv * (_m.sqrt((p*(1.0-p)/n) + (z2/(4.0*n*n))) / denom)
                                    return (max(0.0, center - margin), min(1.0, center + margin))
                                lo1, hi1 = _wilson(p1, n1, z)
                                lo2, hi2 = _wilson(p2, n2, z)
                                # Overlap if intervals intersect significantly
                                if not (lo1 > hi2 or lo2 > hi1):
                                    confident = False
                            except Exception:
                                pass

                    if not confident:
                        # Fall back to global rates if available; else keep only top-1
                        if isinstance(rates_global, dict) and rates_global:
                            items_g = []
                            for k, v in rates_global.items():
                                s = float(samples_global.get(k, 0.0)) if isinstance(samples_global, dict) else 0.0
                                if s >= min_s:
                                    items_g.append((k, float(v)))
                            if not items_g:
                                items_g = [(k, float(v)) for k, v in rates_global.items()]
                            top = sorted(items_g, key=lambda kv: kv[1], reverse=True)[:max(1, min(3, top_k))]
                        else:
                            top = top[:1]
                    # In safe mode, only bias strongly when separation is obvious
                    if _pol == 'safe' and len(top) >= 2:
                        try:
                            min_delta = float((__import__('os').environ.get('NERION_LEARN_MIN_DELTA') or '0.10').strip())
                        except Exception:
                            min_delta = 0.10
                        if (top[0][1] - top[1][1]) < min_delta:
                            top = top[:1]
                    if top:
                        # Optional Top-K hysteresis to prevent flapping
                        try:
                            hyst_on = (os.environ.get('NERION_LEARN_HYSTERESIS_ON') or '1').strip().lower() in {'1','true','yes','on'}
                        except Exception:
                            hyst_on = True
                        if hyst_on and intent_key and isinstance(prefs, dict):
                            try:
                                min_delta = float((__import__('os').environ.get('NERION_LEARN_MIN_IMPROVEMENT_DELTA') or '0.02').strip())
                            except Exception:
                                min_delta = 0.02
                            try:
                                hysteresis_m = int((__import__('os').environ.get('NERION_LEARN_HYSTERESIS_M') or '3').strip())
                            except Exception:
                                hysteresis_m = 3
                            top_tool = top[0][0]
                            prev_map = (prefs.get('preferred_tool_by_intent') or {}) if isinstance(prefs.get('preferred_tool_by_intent'), dict) else {}
                            prev_tool = prev_map.get(intent_key)
                            # Initialize maps
                            if prev_tool is None:
                                prev_map[intent_key] = top_tool
                                prefs['preferred_tool_by_intent'] = prev_map
                            else:
                                if top_tool != prev_tool:
                                    p_top = float(rates.get(top_tool, 0.0))
                                    p_prev = float(rates.get(prev_tool, rates_global.get(prev_tool, 0.0) if isinstance(rates_global, dict) else 0.0))
                                    streaks_all = prefs.get('hysteresis_streak_by_intent') if isinstance(prefs.get('hysteresis_streak_by_intent'), dict) else {}
                                    if not isinstance(streaks_all, dict):
                                        streaks_all = {}
                                    streaks = streaks_all.get(intent_key) if isinstance(streaks_all.get(intent_key), dict) else {}
                                    if (p_top - p_prev) >= float(min_delta):
                                        c = int(streaks.get(top_tool, 0)) + 1
                                        streaks[top_tool] = c
                                        if c < int(hysteresis_m):
                                            # Keep previous tool at the head until challenger proves itself
                                            prev_w = float(weights.get(prev_tool, rates.get(prev_tool, 0.0)))
                                            top = [(prev_tool, prev_w)] + [kv for kv in top if kv[0] != prev_tool]
                                        else:
                                            # Adopt new top; reset streaks
                                            prev_map[intent_key] = top_tool
                                            streaks = {top_tool: 0}
                                            prefs['preferred_tool_by_intent'] = prev_map
                                        streaks_all[intent_key] = streaks
                                        prefs['hysteresis_streak_by_intent'] = streaks_all
                                        if _save_prefs:
                                            try:
                                                _save_prefs(prefs)
                                            except Exception:
                                                pass
                                    else:
                                        # Below improvement threshold: keep previous and reset challenger streak
                                        prev_w = float(weights.get(prev_tool, rates.get(prev_tool, 0.0)))
                                        top = [(prev_tool, prev_w)] + [kv for kv in top if kv[0] != prev_tool]
                                        streaks_all = prefs.get('hysteresis_streak_by_intent') if isinstance(prefs.get('hysteresis_streak_by_intent'), dict) else {}
                                        if not isinstance(streaks_all, dict):
                                            streaks_all = {}
                                        streaks = streaks_all.get(intent_key) if isinstance(streaks_all.get(intent_key), dict) else {}
                                        if isinstance(streaks, dict):
                                            streaks[top_tool] = 0
                                            streaks_all[intent_key] = streaks
                                            prefs['hysteresis_streak_by_intent'] = streaks_all
                                            if _save_prefs:
                                                try:
                                                    _save_prefs(prefs)
                                                except Exception:
                                                    pass
                        parts = [f"{k}:{rates.get(k, 0.0):.2f}" for k, _w in top]
                        header = f"LEARNED BIASES (intent: {intent_key})" if intent_key else "LEARNED BIASES"
                        if not confident and intent_key:
                            header += " — softened (low confidence)"
                        bias_block = (
                            header + ":\n"
                            + "When tool choice is ambiguous, prefer tools with higher local success rates (listed below).\n"
                            + "Success rates: " + ", ".join(parts)
                        )
                        # Provide weights to the prompt manifest rendering
                        learned_weights_for_prompt = {k: float(weights.get(k, rates.get(k, 0.0))) for k, _ in top}
        except Exception:
            bias_block = None
            learned_weights_for_prompt = None

        pol = extra_policies
        if bias_block:
            pol = (pol + "\n" + bias_block) if pol else bias_block

        # Epsilon-greedy policy for exploration
        try:
            import os as _os
            eps = float((_os.getenv('NERION_BANDIT_EPSILON') or '0.05').strip())
        except Exception:
            eps = 0.05
        # Lightweight context hook for future contextual bandits (no behavior change today)
        try:
            from selfcoder.config import get_policy as _get_policy_name
            policy_name = _get_policy_name()
        except Exception:
            policy_name = 'balanced'
        intent_for_ctx = locals().get('intent_key', None)
        ctx = {
            'intent': (intent_for_ctx or 'unknown'),
            'query_len': len(user_query or ''),
            'policy': policy_name,
        }
        try:
            payload = build_master_prompt(
                user_query=user_query,
                tools=self.tools,
                context_snippet=context_snippet,
                context=ctx,
                extra_policies=pol,
                learned_weights=(learned_weights_for_prompt if learned_weights_for_prompt is not None else ((prefs.get('tool_success_rate') if locals().get('prefs') else None) if 'prefs' in locals() else None)),
                epsilon=eps,
            )
        except TypeError:
            # Back-compat with older build_master_prompt signature (tests may monkeypatch)
            payload = build_master_prompt(
                user_query=user_query,
                tools=self.tools,
                context_snippet=context_snippet,
                extra_policies=pol,
            )
        # Choose LLM: coder for code-like tasks, else default parent
        mode = (os.environ.get('NERION_PARENT_USE_CODER') or 'auto').strip().lower()
        use_coder = False
        if mode in {'always', 'on', 'true', '1'}:
            use_coder = True
        elif mode not in {'off', 'false', 'no', '0'}:
            use_coder = self._looks_like_code_task(user_query)
        parent_llm = self._get_coder_parent_llm() if use_coder else self.llm
        if parent_llm is None:
            parent_llm = self.llm
        raw = parent_llm.complete(payload["messages"])  # provider returns JSON string

        # Robust parse → ParentDecision; fallback to a clarify plan on error
        try:
            data = json.loads(raw)
            decision = ParentDecision(**data)

            # ARCHITECTURAL FIX: Validate that LLM didn't cop out when tools are available
            # This prevents the LLM from choosing "respond" to bypass using available tools
            decision = self._enforce_tool_usage(decision, user_query)

            return decision
        except Exception as e:
            # Minimal safe fallback that asks the user for clarification
            return ParentDecision(
                intent="clarify",
                plan=[{"action": "ask_user", "tool": None, "args": {}, "summary": "clarify request"}],
                final_response=None,
                confidence=0.0,
                requires_network=False,
                notes=f"Invalid ParentDecision JSON: {e}",
            )
