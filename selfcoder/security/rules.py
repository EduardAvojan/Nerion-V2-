import ast
import re
from pathlib import Path
from typing import List
from . import Finding
from selfcoder.config import allow_network
AWS_KEY_RE = re.compile(r'AKIA[0-9A-Z]{16}')
GENERIC_TOKEN_RE = re.compile(r'(?i)(api[_-]?key|secret|token)\s*[:=]\s*["\']?[A-Za-z0-9_\-]{16,}["\']?')
SLACK_TOKEN_RE = re.compile(r"xox[bapo]-[0-9A-Za-z-]{10,}")
GITHUB_TOKEN_RE = re.compile(r"gh[pousr]_[A-Za-z0-9]{36}")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----")
JWT_TOKEN_RE = re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")
class DangerVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, repo_root: Path) -> None:
        self.filename = filename
        self.repo_root = repo_root
        self.findings: List[Finding] = []
    def _add(self, rule_id: str, severity: str, node: ast.AST, message: str, evidence: str):
        line = getattr(node, "lineno", 1)
        self.findings.append(Finding(rule_id=rule_id, severity=severity, message=message,
                                     filename=self.filename, line=line, evidence=evidence))
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile"}:
            self._add("AST-EXEC-001", "critical", node, f"Use of {node.func.id}()", node.func.id)
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            self._add("AST-EXEC-002", "high", node, "Use of __import__()", "__import__")
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "importlib":
                # Allow the resources API which is a common safe pattern
                if node.func.attr != "resources":
                    self._add("AST-EXEC-003", "high", node, f"Dynamic import via importlib.{node.func.attr}", f"importlib.{node.func.attr}")
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
            shell_kw = False
            for kw in node.keywords or []:
                if getattr(kw, "arg", None) == "shell" and getattr(kw.value, "value", None) is True:
                    shell_kw = True
                    self._add("AST-PROC-001", "high", node, "subprocess.* called with shell=True", "shell=True")
                    break
            attr = node.func.attr
            if attr in {"Popen", "call", "run", "check_output"}:
                if not shell_kw:
                    self._add("AST-PROC-010", "medium", node, f"subprocess.{attr} invocation", f"subprocess.{attr}")
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "yaml" and node.func.attr == "load":
            safe = any((getattr(kw, "arg", None) == "Loader" and isinstance(kw.value, ast.Attribute) and getattr(kw.value, "attr", "") == "SafeLoader") for kw in (node.keywords or []))
            if not safe:
                self._add("AST-DESER-001", "high", node, "yaml.load without SafeLoader", "yaml.load")

        # os.system shell execution
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "system":
            self._add("AST-EXEC-004", "high", node, "os.system shell execution", "os.system")

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "asyncio" and node.func.attr in {"create_subprocess_exec", "create_subprocess_shell"}:
            severity = "high" if node.func.attr == "create_subprocess_shell" else "medium"
            self._add("AST-PROC-011", severity, node, f"asyncio.{node.func.attr} invocation", f"asyncio.{node.func.attr}")

        # shutil potentially destructive operations
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "shutil" and node.func.attr in {"rmtree", "move"}:
            self._add("AST-FS-010", "medium", node, f"Potentially destructive file op shutil.{node.func.attr}", f"shutil.{node.func.attr}")

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr in {"remove", "unlink", "rmdir", "removedirs", "replace"}:
            self._add("AST-FS-020", "medium", node, f"Potentially destructive file op os.{node.func.attr}", f"os.{node.func.attr}")

        if isinstance(node.func, ast.Attribute) and node.func.attr in {"unlink", "rmdir"}:
            base = node.func.value
            base_label = None
            if isinstance(base, ast.Name):
                base_label = base.id
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                base_label = base.value.id
            if base_label in {"pathlib", "Path", "path"}:
                self._add("AST-FS-021", "medium", node, f"Path.{node.func.attr} invoked", f"Path.{node.func.attr}")

        # requests with TLS verification disabled
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "requests":
            for kw in node.keywords or []:
                if getattr(kw, "arg", None) == "verify":
                    if isinstance(getattr(kw, "value", None), ast.Constant) and getattr(kw.value, "value", None) is False:
                        self._add("AST-NET-002", "medium", node, "requests call with verify=False", "verify=False")
                        break

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in {"pickle", "marshal"} and node.func.attr == "loads":
                self._add("AST-DESER-002", "high", node, f"Unsafe deserialization via {node.func.value.id}.loads", f"{node.func.value.id}.loads")
        # Generic network use via requests.*
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "requests":
            # If TLS is explicitly disabled, that is handled above (verify=False)
            # Only flag general network usage when network is disallowed, else downgrade/skip
            try:
                if not allow_network():
                    self._add("AST-NET-001", "medium", node, f"Network call via requests.{node.func.attr}", f"requests.{node.func.attr}")
                else:
                    # In network-allowed mode, do not penalize general requests usage
                    pass
            except Exception:
                # On config error, err on the safer side and record as low severity
                self._add("AST-NET-001", "low", node, f"Network call via requests.{node.func.attr}", f"requests.{node.func.attr}")
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                mode = node.args[1].value
            elif node.keywords:
                for kw in node.keywords:
                    if getattr(kw, "arg", "") == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        mode = kw.value.value
            if mode and any(m in mode for m in ("w", "a", "x")):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    p = Path(node.args[0].value)
                    try:
                        rp = p if p.is_absolute() else (self.repo_root / p).resolve()
                        if not str(rp).startswith(str(self.repo_root.resolve())):
                            self._add("AST-FS-001", "high", node, f"Write outside repo: {p}", str(p))
                    except Exception:
                        self._add("AST-FS-002", "medium", node, "Unresolvable write path in open()", str(getattr(node.args[0], "value", "")))
        self.generic_visit(node)
def regex_findings(source: str, filename: str) -> List[Finding]:
    findings: List[Finding] = []
    for m in AWS_KEY_RE.finditer(source):
        findings.append(Finding("REG-SECRET-001", "high", "AWS style access key detected", filename, source.count("\n", 0, m.start())+1, m.group(0)))
    for m in GENERIC_TOKEN_RE.finditer(source):
        findings.append(Finding("REG-SECRET-002", "medium", "Possible API key/secret assignment", filename, source.count("\n", 0, m.start())+1, m.group(0)))
    for m in SLACK_TOKEN_RE.finditer(source):
        findings.append(Finding("REG-SECRET-005", "high", "Possible Slack token detected", filename, source.count("\n", 0, m.start())+1, m.group(0)))
    for m in GITHUB_TOKEN_RE.finditer(source):
        findings.append(Finding("REG-SECRET-006", "high", "Possible GitHub token detected", filename, source.count("\n", 0, m.start())+1, m.group(0)))
    for m in PRIVATE_KEY_RE.finditer(source):
        findings.append(Finding("REG-SECRET-003", "critical", "Possible private key material detected", filename, source.count("\n", 0, m.start())+1, m.group(0)))
    for m in JWT_TOKEN_RE.finditer(source):
        findings.append(Finding("REG-SECRET-004", "medium", "JWT-like token detected", filename, source.count("\n", 0, m.start())+1, m.group(0)))
    return findings
