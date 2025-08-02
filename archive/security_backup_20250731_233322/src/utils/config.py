"""backend.utils.config
--------------------------------
Light-weight compatibility layer that exposes convenient helpers for accessing
configuration without importing the (heavy) full settings module everywhere.

Historically, several API modules imported :pyfunc:`get_api_settings` and a
``Config`` class from ``backend.utils.config``. During the recent refactor we
moved the canonical configuration definitions to
``backend.config.settings``.  To avoid chasing down and rewriting all legacy
imports we provide this thin wrapper which delegates to the new implementation.

The wrapper keeps our promise of *zero-debugging* by guaranteeing that modules
continue to work unchanged while surfacing any configuration loading issues via
structured logging.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from src.config.settings import KimeraSettings, get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
Config = KimeraSettings  # Backwards-compatibility alias


@lru_cache(maxsize=1)
def _cached_settings() -> KimeraSettings:  # noqa: D401
    """Load and cache the global Kimera settings instance.

    Using ``functools.lru_cache`` ensures the settings object is initialised
    only once during the lifecycle which prevents duplicate environment reads
    and expensive validations.
    """
    try:
        settings = get_settings()
        logger.debug("Kimera settings loaded (env=%s)", settings.environment)
        return settings
    except Exception as exc:  # pragma: no cover â€“ critical startup failure
        logger.exception("Failed to load Kimera settings: %s", exc)
        raise


def get_api_settings() -> KimeraSettings:  # noqa: D401
    """Return the cached global :class:`KimeraSettings` instance.

    The name is preserved for legacy import paths.
    """
    return _cached_settings()


# ---------------------------------------------------------------------------
# Convenience helpers â€“ these are optional and may be extended later
# ---------------------------------------------------------------------------

def reload_settings() -> KimeraSettings:  # noqa: D401
    """Force reload the settings from scratch and update the cache."""
    _cached_settings.cache_clear()
    return _cached_settings()


def get_setting(path: str, default: Any | None = None) -> Any:  # noqa: D401
    """Retrieve a dotted-path attribute from the global settings.

    Examples
    --------
    >>> host = get_setting("server.host")
    >>> db_url = get_setting("database.url")
    """
    parts = path.split(".")
    obj: Any = get_api_settings()
    try:
        for part in parts:
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        logger.warning("Requested unknown config path: %s", path)
        return default 

def get_config():
    """
    Get the current configuration settings.
    This is a compatibility wrapper for the new settings system.
    
    Returns:
        KimeraSettings: The current configuration
    """
    from src.config.settings import get_settings
    return get_settings()
