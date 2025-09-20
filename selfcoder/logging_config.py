from __future__ import annotations
import logging
import os
from typing import Optional

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

def _parse_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    val = val.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return default


def setup_logging(
    default_level: Optional[str] = None,
    *,
    level: Optional[str | int] = None,
    stream=None,
) -> logging.Logger:
    """
    Initialize stdlib logging in a robust, idempotent way and return the project logger ("selfcoder").

    Priority of log level: explicit `level` arg > env SELFCODER_LOG_LEVEL > `default_level` > INFO.

    Env variables supported:
      - SELFCODER_LOG_LEVEL: e.g., "DEBUG" | "INFO" | int
      - SELFCODER_LOG_FORMAT: "simple" (default) or "rich" (adds time, module, line)
      - SELFCODER_LOG_FILE: path to also write logs (rotating not included)
      - SELFCODER_LOG_COLOR: "true/false" (ANSI colors for console). Default: true if TTY.
      - SELFCODER_LOG_FORCE: "true/false". If true, reset handlers each call.

    Parameters:
        default_level: Default log level name if no env or `level` is given.
        level: Log level as a string (e.g., "DEBUG") or int (e.g., 10). Overrides env/default.
        stream: Stream to write console logs to (e.g., sys.stdout). If None, logging uses default.

    Returns:
        logging.Logger: Configured logger for the "selfcoder" namespace.
    """
    # Resolve level
    if level is not None:
        if isinstance(level, str):
            level_name = level.upper()
            log_level = _LEVELS.get(level_name, logging.INFO)
        else:
            log_level = int(level)
            level_name = str(level)
    else:
        env_level = os.getenv("SELFCODER_LOG_LEVEL") or default_level or "INFO"
        level_name = str(env_level).upper()
        # allow numeric in env
        try:
            log_level = int(level_name)
        except ValueError:
            log_level = _LEVELS.get(level_name, logging.INFO)

    fmt_mode = (os.getenv("SELFCODER_LOG_FORMAT") or "simple").strip().lower()
    want_color = _parse_bool(os.getenv("SELFCODER_LOG_COLOR"), default=True)
    log_file = os.getenv("SELFCODER_LOG_FILE")
    force_reset = _parse_bool(os.getenv("SELFCODER_LOG_FORCE"), default=False)

    # Build format strings
    if fmt_mode == "rich":
        base_fmt = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
        date_fmt = "%H:%M:%S"
    else:
        base_fmt = "%(levelname)s %(name)s: %(message)s"
        date_fmt = None

    # Optional ANSI color for levelname in console
    class _ColorFormatter(logging.Formatter):
        LEVEL_COLORS = {
            logging.DEBUG: "\x1b[36m",    # cyan
            logging.INFO: "\x1b[32m",     # green
            logging.WARNING: "\x1b[33m",  # yellow
            logging.ERROR: "\x1b[31m",    # red
            logging.CRITICAL: "\x1b[35m", # magenta
        }
        RESET = "\x1b[0m"
        def format(self, record: logging.LogRecord) -> str:
            original = record.levelname
            try:
                if want_color:
                    color = self.LEVEL_COLORS.get(record.levelno, "")
                    if color:
                        record.levelname = f"{color}{original}{self.RESET}"
                return super().format(record)
            finally:
                record.levelname = original

    console_formatter: logging.Formatter
    if want_color:
        console_formatter = _ColorFormatter(base_fmt, datefmt=date_fmt)
    else:
        console_formatter = logging.Formatter(base_fmt, datefmt=date_fmt)

    file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s:%(lineno)d: %(message)s", "%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger("selfcoder")
    logger.setLevel(log_level)
    logger.propagate = False  # keep our handlers local

    # When a specific stream is requested, always route to it (replace handlers).
    if stream is not None:
        for h in list(logger.handlers):
            logger.removeHandler(h)
        ch = logging.StreamHandler(stream)
        ch.setLevel(log_level)
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)
        # Optional file handler
        if log_file:
            try:
                fh = logging.FileHandler(log_file)
                fh.setLevel(log_level)
                fh.setFormatter(file_formatter)  # plain (no color)
                logger.addHandler(fh)
            except Exception as e:
                logger.warning("Failed to set file handler for '%s': %s", log_file, e)
    else:
        # Idempotent handler setup; optionally reset if requested.
        if force_reset:
            for h in list(logger.handlers):
                logger.removeHandler(h)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(console_formatter)
            logger.addHandler(ch)
            if log_file:
                try:
                    fh = logging.FileHandler(log_file)
                    fh.setLevel(log_level)
                    fh.setFormatter(file_formatter)
                    logger.addHandler(fh)
                except Exception as e:
                    logger.warning("Failed to set file handler for '%s': %s", log_file, e)

    logger.debug("Logging initialized at %s (mode=%s, color=%s, file=%s)", level_name, fmt_mode, want_color, bool(log_file))
    return logger


def set_log_level(new_level: str | int) -> None:
    """Dynamically change the log level for the selfcoder logger and its handlers."""
    logger = logging.getLogger("selfcoder")
    if isinstance(new_level, str):
        lvl = _LEVELS.get(new_level.upper(), logging.INFO)
    else:
        lvl = int(new_level)
    logger.setLevel(lvl)
    for h in logger.handlers:
        h.setLevel(lvl)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the root selfcoder logger or a child logger (e.g., get_logger('ast'))."""
    base = logging.getLogger("selfcoder")
    return base if not name else base.getChild(name)