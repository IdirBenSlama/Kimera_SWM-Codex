"""
Kimera Backend Module
====================
Sets up global environment for the entire backend system.
"""

import os
import sys
from pathlib import Path

# Set critical environment variables early in the import process
# This ensures all modules use the correct database credentials
os.environ["DATABASE_URL"] = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm",
)
os.environ["KIMERA_DATABASE_URL"] = os.environ.get(
    "KIMERA_DATABASE_URL",
    "postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm",
)

# Additional PostgreSQL environment variables for completeness
os.environ["POSTGRES_USER"] = os.environ.get("POSTGRES_USER", "kimera")
os.environ["POSTGRES_PASSWORD"] = os.environ.get(
    "POSTGRES_PASSWORD", "kimera_secure_pass_2025"
)
os.environ["POSTGRES_DB"] = os.environ.get("POSTGRES_DB", "kimera_swm")
os.environ["POSTGRES_HOST"] = os.environ.get("POSTGRES_HOST", "localhost")
os.environ["POSTGRES_PORT"] = os.environ.get("POSTGRES_PORT", "5432")

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

__version__ = "0.1.0"
__author__ = "Kimera SWM Team"
