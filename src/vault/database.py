"""
DEPRECATED: This file has been archived as part of database consolidation.
Use src.config.database_config instead.

Archived on: 2025-08-04T16:10:34.061505
Original location: src/vault/database.py
"""

# Import the new unified config
from src.config.database_config import *

# Import GeoidDB for backward compatibility
try:
    from ..core.geoid import GeoidDB
except ImportError:
    # Create placeholder if GeoidDB doesn't exist
class GeoidDB:
    """Auto-generated class."""
    pass
        """Placeholder GeoidDB for backward compatibility"""

        def __init__(self):
            pass


import warnings

warnings.warn(
    f"{__file__} is deprecated. Use src.config.database_config instead.",
    DeprecationWarning,
    stacklevel=2,
)
