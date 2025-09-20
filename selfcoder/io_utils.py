"""
Stable facade for common I/O helpers used by Selfcoder internals.

This module re-exports safe runtime utilities implemented under `selfcoder.io`
so callers do not depend on application-specific modules.
"""

from selfcoder.io.runtime import (  # noqa: F401
    run,
    backup_file,
    write_file,
    run_lint,
    run_tests,
    git_commit,
)
