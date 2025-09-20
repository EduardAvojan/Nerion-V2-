import sys
import signal
from contextlib import contextmanager

_cancelled = False
def _sigint_handler(signum, frame):
    global _cancelled
    _cancelled = True

def cancelled() -> bool:
    return _cancelled

@contextmanager
def progress(task: str):
    """
    Prints a start/stop line for long-running tasks.
    Use core.ui.progress.cancelled() in loops to abort cleanly.
    """
    global _cancelled
    _cancelled = False
    prev = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)
    try:
        print(f"{task} …", file=sys.stderr)
        yield
    finally:
        status = "cancelled" if _cancelled else "done"
        print(f"{task} → {status}", file=sys.stderr)
        signal.signal(signal.SIGINT, prev)
