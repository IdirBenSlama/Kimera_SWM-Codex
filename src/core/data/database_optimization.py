"""
DEPRECATED: This file has been archived as part of database consolidation.
Use src.config.database_config instead.

Archived on: 2025-08-04T16:10:34.060660
Original location: src\core\database_optimization.py
"""

import warnings

# Import the new unified config
from src.config.database_config import *

warnings.warn(
    f"{__file__} is deprecated. Use src.config.database_config instead.",
    DeprecationWarning
    stacklevel=2
)
