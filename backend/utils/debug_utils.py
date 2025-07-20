"""backend.utils.debug_utils
--------------------------------
Miscellaneous helpers that aid debugging and observability without relying on
an external debugger.
"""

from __future__ import annotations

import threading
from typing import List, Dict, Any


def get_thread_info() -> List[Dict[str, Any]]:  # noqa: D401
    """Return a structured listing of all active threads."""
    thread_data: List[Dict[str, Any]] = []
    for t in threading.enumerate():
        thread_data.append(
            {
                "name": t.name,
                "ident": t.ident,
                "daemon": t.daemon,
                "alive": t.is_alive(),
            }
        )
    return thread_data 