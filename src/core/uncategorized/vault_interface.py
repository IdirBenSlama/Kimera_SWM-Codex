"""
Vault Interface - Read-Only Data Access for the Cognitive Core
=============================================================

Provides a high-level, abstracted interface for cognitive components to
access necessary data from the Vault without directly managing database
sessions or CRUD operations.

This promotes a clean architecture by decoupling the cognitive engines
from the underlying database implementation.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from ..vault import crud
from ..vault.database import SessionLocal


@dataclass
class GeoidMetrics:
    """Auto-generated class."""
    pass
    """
    A structured representation of the metrics for a single Geoid
    used for constitutional analysis by the Heart.
    """

    geoid_id: str
    exists: bool = False
    stability: float = 0.5
    connectivity: int = 0
    scar_count: int = 0
class VaultInterface:
    """Auto-generated class."""
    pass
    """
    A read-only interface to the Kimera data vault.
    """

    def __init__(self):
        # Initialize without creating session to avoid circular dependency
        self.db_session = None

    def _ensure_session(self):
        """Lazily create the database session when needed."""
        if self.db_session is None:
            if SessionLocal is not None:
                self.db_session = SessionLocal()
            else:
                raise RuntimeError("Database not initialized. SessionLocal is None.")

    def __del__(self):
        # Ensure the session is closed when the object is destroyed.
        if hasattr(self, "db_session") and self.db_session is not None:
            self.db_session.close()

    @lru_cache(maxsize=1024)
    def get_geoid_metrics(self, geoid_id: str) -> GeoidMetrics:
        """
        Retrieves a comprehensive set of metrics for a given Geoid.

        This method is cached to reduce database load for frequently
        accessed Geoids within a single cognitive operation.

        Args:
            geoid_id: The ID of the Geoid to analyze.

        Returns:
            A GeoidMetrics object containing the Geoid's vital stats.
        """
        self._ensure_session()
        geoid = crud.get_geoid_by_id(self.db_session, geoid_id)

        if not geoid:
            return GeoidMetrics(geoid_id=geoid_id, exists=False)

        stability = crud.get_geoid_stability(self.db_session, geoid_id)
        connectivity = crud.get_geoid_connectivity(self.db_session, geoid_id)
        scar_count = crud.get_scar_count_for_geoid(self.db_session, geoid_id)

        return GeoidMetrics(
            geoid_id=geoid_id
            exists=True
            stability=stability
            connectivity=connectivity
            scar_count=scar_count
        )


# Global instance for easy access from cognitive components.
# This is created at import time but sessions are created lazily
_vault_interface_instance: Optional[VaultInterface] = None


def get_vault_interface() -> VaultInterface:
    """Get the global vault interface instance (lazy initialization)."""
    global _vault_interface_instance
    if _vault_interface_instance is None:
        _vault_interface_instance = VaultInterface()
    return _vault_interface_instance


# For backward compatibility, expose as vault_interface
vault_interface = get_vault_interface()
