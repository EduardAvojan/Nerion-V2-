from __future__ import annotations

import os

DEFAULT_BUILD = "2025-08-11 control-fuzzy v3"

BUILD_TAG = os.getenv("NERION_BUILD_TAG", DEFAULT_BUILD)

