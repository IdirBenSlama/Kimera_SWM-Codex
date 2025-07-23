"""
FastAPI Dependency Providers for Kimera Subsystems

This module implements the dependency injection system for the Kimera API.
It replaces the functionality of the old `KimeraSystem` service locator.

Each provider function is responsible for creating and caching a single
subsystem instance, allowing FastAPI's `Depends` system to manage the
lifecycle of each component. This makes dependencies explicit, improves
testability, and decouples our API layer from the core system.
"""

import logging
from typing import Optional
from functools import lru_cache

from fastapi import HTTPException

# Import the concrete classes we need to provide
from src.vault.vault_manager import VaultManager
from src.engines.contradiction_engine import ContradictionEngine
from src.engines.foundational_thermodynamic_engine import FoundationalThermodynamicEngine
from src.utils.gpu_foundation import GPUFoundation

logger = logging.getLogger(__name__)

# --- Provider Functions with Caching (Singleton Behavior) ---

@lru_cache(maxsize=None)
def get_gpu_foundation() -> Optional[GPUFoundation]:
    """
    Dependency provider for the GPUFoundation.
    Caches the instance for application-wide reuse.
    Returns None if GPU is unavailable.
    """
    try:
        gpu_found = GPUFoundation()
        logger.info("GPU detected via DI – operations will use %s", gpu_found.get_device())
        return gpu_found
    except (RuntimeError, ImportError, AttributeError) as exc:
        logger.warning("GPU unavailable or initialisation failed via DI (%s).", exc)
        return None

@lru_cache(maxsize=None)
def get_vault_manager() -> VaultManager:
    """
    Dependency provider for the VaultManager.
    Caches the instance for application-wide reuse.
    """
    logger.info("Initializing VaultManager via DI.")
    try:
        return VaultManager()
    except Exception as e:
        logger.critical("Failed to initialize VaultManager: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail="VaultManager is unavailable.")

@lru_cache(maxsize=None)
def get_contradiction_engine() -> ContradictionEngine:
    """
    Dependency provider for the ContradictionEngine.
    Caches the instance for application-wide reuse.
    """
    logger.info("Initializing ContradictionEngine via DI.")
    try:
        return ContradictionEngine(tension_threshold=0.4)
    except Exception as e:
        logger.critical("Failed to initialize ContradictionEngine: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail="ContradictionEngine is unavailable.")

@lru_cache(maxsize=None)
def get_thermodynamic_engine() -> FoundationalThermodynamicEngine:
    """
    Dependency provider for the FoundationalThermodynamicEngine.
    Caches the instance for application-wide reuse.
    """
    logger.info("Initializing FoundationalThermodynamicEngine via DI.")
    try:
        return FoundationalThermodynamicEngine()
    except Exception as e:
        logger.critical("Failed to initialize FoundationalThermodynamicEngine: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail="FoundationalThermodynamicEngine is unavailable.")

# Note: The embedding model is handled differently and seems to be managed
# within `embedding_utils` itself. We will address that separately if needed. 