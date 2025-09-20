# ops/telemetry/logger.py
import logging
import re
from typing import Iterable

# Heuristics to mask common secret patterns (kept simple on purpose).
_SENSITIVE_PATTERNS: Iterable[re.Pattern[str]] = [
    re.compile(r'(api[_-]?key|token|secret|password)\s*[:=]\s*([^\s,;]+)', re.I),
    re.compile(r'AKIA[0-9A-Z]{16}'),  # AWS Access Key ID
]

class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        for pat in _SENSITIVE_PATTERNS:
            # Preserve the key if present, mask the value
            def _mask(m: re.Match[str]) -> str:
                if m.lastindex and m.lastindex >= 2:
                    return f"{m.group(1)}=***"
                return "***"
            msg = pat.sub(_mask, msg)
        record.msg, record.args = msg, ()
        return True

_logger = logging.getLogger("nerion")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.addFilter(_RedactFilter())
    h.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s'))
    _logger.addHandler(h)

def log(msg: str, level: str = "INFO"):
    level = (level or "INFO").upper()
    if level == "DEBUG":
        _logger.debug(msg)
    elif level in ("WARN", "WARNING"):
        _logger.warning(msg)
    elif level in ("ERR", "ERROR"):
        _logger.error(msg)
    else:
        _logger.info(msg)