"""KIMERA SWM Graph Database Package"""

This package provides Neo4j integration for the KIMERA system, enabling
graph-based storage and querying of semantic relationships, causal chains,
and understanding structures.

Key modules:
- session: Neo4j driver factory and connection management
- models: High-level CRUD operations for graph entities

Usage:
    try:
    from graph.session import get_session
except ImportError:
    # Create placeholders for graph.session
        def get_session(*args, **kwargs): return None
    try:
    from graph.models import create_geoid, get_geoid
except ImportError:
    # Create placeholders for graph.models
        def create_geoid(*args, **kwargs): return None
    def get_geoid(*args, **kwargs): return None
"""

from .models import create_geoid, create_scar, get_geoid, get_scar
from .session import driver_liveness_check, get_driver, get_session

__all__ = [
    "get_driver",
    "get_session",
    "driver_liveness_check",
    "create_geoid",
    "get_geoid",
    "create_scar",
    "get_scar",
]
