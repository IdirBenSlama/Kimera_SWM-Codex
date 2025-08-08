"""
Configuration Compatibility Layer
=================================

Provides backward compatibility for various configuration patterns
used throughout the KIMERA codebase.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Import the robust config as primary
try:
    from .robust_config import get_api_settings as _robust_get_api_settings
    from .robust_config import get_setting, reload_settings

    ROBUST_CONFIG_AVAILABLE = True
except ImportError:
    ROBUST_CONFIG_AVAILABLE = False

# Fallback to original config if available
try:
    from .config import get_api_settings as _original_get_api_settings

    ORIGINAL_CONFIG_AVAILABLE = True
except ImportError:
    ORIGINAL_CONFIG_AVAILABLE = False


def get_api_settings() -> Any:
    """Get API settings with maximum compatibility"""

    # Try robust config first
    if ROBUST_CONFIG_AVAILABLE:
        try:
            return _robust_get_api_settings()
        except Exception as e:
            logger.debug(f"Robust config failed: {e}")

    # Fall back to original config
    if ORIGINAL_CONFIG_AVAILABLE:
        try:
            return _original_get_api_settings()
        except Exception as e:
            logger.debug(f"Original config failed: {e}")

    # Final fallback - create minimal settings
    logger.warning("All config systems failed, using emergency fallback")
class EmergencySettings:
    """Auto-generated class."""
    pass
        environment = "emergency"
        debug = True
        gpu_enabled = False

    return EmergencySettings()


# Legacy aliases for backward compatibility
get_settings = get_api_settings
api_settings = get_api_settings


# Common patterns used in engines
def safe_get_api_settings():
    """Ultra-safe version that never raises"""
    try:
        return get_api_settings()
    except:
class UltraSafeSettings:
    """Auto-generated class."""
    pass
            environment = "safe"

        return UltraSafeSettings()
