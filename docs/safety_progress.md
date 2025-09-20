# Nerion Safety & Security Roadmap

## ✅ Current Capabilities (Baseline Security Layer)

- **Static code scanner** (`selfcoder.security.scanner`)
  - Detects dangerous patterns (e.g., `eval(...)`) in Python source.
  - Returns structured `Finding` objects with rule ID, severity, evidence.

- **Security gate** (`selfcoder.security.gate.assess_plan`)
  - Runs pre-flight before applying a plan.
  - Blocks if any `critical` findings are present.
  - Integrated into the orchestrator.

- **CLI Scan Command**
  - `nerion scan FILES...` to scan files manually.
  - `--fail-on` severity threshold for CI/CD integration.
  - Human-readable and JSON report output.

- **Rollback Integration**
  - Every apply path uses `snapshot` + `restore` to prevent partial/insecure changes.
  - Healthcheck runs after apply; failures trigger rollback.

- **Tests**
  - Unit tests for scanner & CLI fail-on policies.

- **Network Gate** (`ops.security.net_gate`)
  - Offline by default (local-first).
  - Requires explicit user grant for any outbound HTTP/API calls.
  - Supports session-based grants (sliding 10‑minute window with idle auto‑revoke).
  - Optional persistent preference: "Always for this task type" remembered across sessions.
  - Enforces domain scoping and provides audit log at `out/security_audit/net_gate.log`.
  - Integrated into `search_api`, `web_render`, and chat orchestration layer.

---

## 🔸 Known Gaps (Not Yet Implemented)

- More detection rules:
  - `exec(...)`
  - Dangerous subprocess usage
  - `yaml.load` without `SafeLoader`
  - `pickle.load` / `pickle.loads`
  - Hard-coded secrets / API keys
  - `os.system` / shell injection patterns
  - Bare `except:` blocks

- Repo-wide scanning:
  - `nerion scan --repo` to scan all allowed source files.

- Rule configuration:
  - Enable/disable specific rules per repo.
  - Adjustable severities.

- Inline allowlist annotations:
  - Allow exceptions with `# nerion: allow=RULE_ID`.
  - Logged for audit trail.

- Export formats:
  - SARIF (Static Analysis Results Interchange Format) for GitHub Code Scanning.
  - HTML dashboard reports.

- Auto-fixes:
  - `--autofix` mode to rewrite known unsafe patterns into safe alternatives.

- Network UX Enhancements:
  - Richer status chips (showing online/offline and remaining time).
  - CLI commands: `nerion net status|allow|off` for explicit control.
  - Finer-grained domain policies.

---

## 📌 Next Steps (Future Work)

1. **Rule Expansion**
   - Add exec/subprocess/pickle/yaml/secret detection.
   - Assign severities per rule.

2. **Repo-Wide Scan Mode**
   - Support scanning the entire repo or specific subdirs.

3. **Rule Config File**
   - `.nerionsec.json` or `.nerionsec.yaml` to control rules/severity.

4. **Allowlist System**
   - Inline comments + central allowlist file.

5. **Output Enhancements**
   - SARIF, HTML reports, richer JSON with remediation suggestions.

6. **Autofix Mode**
   - Safe in-place rewrites for common issues.

7. **Network Controls**
   - Add CLI `nerion net` commands.
   - Display time‑remaining in status chip consistently.
   - Extend domain/task‑type policy configuration.

---

**Status:** Baseline complete (includes static scanner, security gate, rollback, tests, and network gate).  
**Blocking Issues:** None — safe to proceed with Evolution Steps 7–12.
