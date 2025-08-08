"""Compatibility layer for the legacy ``layer_2_governance`` package."""

Historically, many parts of the codebase imported monitoring, security and core
functionality from ``src.layer_2_governance``.  The real implementations now
live in the top‑level ``src.monitoring``, ``src.security`` and ``src.core``
packages.  This module provides shims that map the old import paths to the new
locations by inserting the appropriate modules into ``sys.modules``.
"""

import sys
from pathlib import Path

# Ensure the project root is on the path so absolute imports keep working.
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import the real modules that should appear under ``layer_2_governance``.
from src import core, monitoring, security, utils

# Expose them under the legacy package namespace for backwards compatibility.
sys.modules["src.layer_2_governance.monitoring"] = monitoring
sys.modules["src.layer_2_governance.security"] = security
sys.modules["src.layer_2_governance.core"] = core
sys.modules["src.layer_2_governance.utils"] = utils

__all__ = ["monitoring", "security", "core", "utils"]
