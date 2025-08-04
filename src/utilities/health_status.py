"""
Health Status Utilities
======================

Provides health status enums and system uptime functionality for KIMERA SWM.

Author: KIMERA Development Team
Version: 1.0.0
"""

import time
from datetime import datetime
from enum import Enum


class HealthStatus(Enum):
    """System health status enumeration"""

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    INITIALIZING = "initializing"
    SHUTTING_DOWN = "shutting_down"


# Track system start time
_system_start_time = time.time()


def get_system_uptime() -> float:
    """
    Get system uptime in seconds

    Returns:
        float: System uptime in seconds since start
    """
    return time.time() - _system_start_time
