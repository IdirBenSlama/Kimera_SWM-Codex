"""
API Dependencies for Dependency Injection
=========================================

Provides FastAPI dependencies for consistent access to core components.
Implements aerospace-grade singleton patterns with proper error handling.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException

logger = logging.getLogger(__name__)

# Cached instances
_kimera_system = None
_vault_manager = None
_contradiction_engine = None
_translator_hub = None
_governance_engine = None


def get_kimera_system():
    """Get KimeraSystem instance with caching."""
    global _kimera_system
    if _kimera_system is None:
        try:
            from src.core.kimera_system import get_kimera_system as _get_system

            _kimera_system = _get_system()
        except Exception as e:
            logger.error(f"Failed to get KimeraSystem: {e}")
            raise HTTPException(status_code=500, detail="KimeraSystem not available")
    return _kimera_system


def get_vault_manager():
    """Get VaultManager instance with caching."""
    global _vault_manager
    if _vault_manager is None:
        try:
            from src.vault.vault_manager import VaultManager

            _vault_manager = VaultManager()
        except Exception as e:
            logger.error(f"Failed to get VaultManager: {e}")
            raise HTTPException(status_code=500, detail="VaultManager not available")
    return _vault_manager


def get_contradiction_engine():
    """Get ContradictionEngine instance with caching."""
    global _contradiction_engine
    if _contradiction_engine is None:
        try:
            from src.engines.contradiction_engine import ContradictionEngine

            _contradiction_engine = ContradictionEngine()
        except Exception as e:
            logger.error(f"Failed to get ContradictionEngine: {e}")
            raise HTTPException(
                status_code=500, detail="ContradictionEngine not available"
            )
    return _contradiction_engine


def get_translator_hub():
    """Get TranslatorHub instance with caching."""
    global _translator_hub
    if _translator_hub is None:
        try:
            from src.engines.universal_translator_hub import (
                create_universal_translator_hub,
            )

            _translator_hub = create_universal_translator_hub()
        except Exception as e:
            logger.error(f"Failed to get TranslatorHub: {e}")
            raise HTTPException(status_code=500, detail="TranslatorHub not available")
    return _translator_hub


def get_governance_engine():
    """Get GovernanceEngine instance with caching."""
    global _governance_engine
    if _governance_engine is None:
        try:
            from src.governance import GovernanceEngine, create_default_policies

            _governance_engine = GovernanceEngine()

            # Load default policies
            policies = create_default_policies()
            for policy in policies:
                _governance_engine.register_policy(policy)
                _governance_engine.activate_policy(policy.id)

        except Exception as e:
            logger.error(f"Failed to get GovernanceEngine: {e}")
            raise HTTPException(
                status_code=500, detail="GovernanceEngine not available"
            )
    return _governance_engine


# Optional dependencies that may not be available
def get_performance_monitor():
    """Get PerformanceMonitor if available."""
    try:
        from src.monitoring.performance_monitor import get_performance_monitor

        return get_performance_monitor()
    except Exception as e:
        logger.error(f"Error in dependencies.py: {e}", exc_info=True)
        raise  # Re-raise for proper error handling
        return None


def get_metrics_collector():
    """Get MetricsCollector if available."""
    try:
        from src.monitoring.metrics_collector import get_metrics_collector

        return get_metrics_collector()
    except Exception as e:
        logger.error(f"Error in dependencies.py: {e}", exc_info=True)
        raise  # Re-raise for proper error handling
        return None


def get_alert_manager():
    """Get AlertManager if available."""
    try:
        from src.monitoring.alert_manager import AlertManager

        return AlertManager()
    except Exception as e:
        logger.error(f"Error in dependencies.py: {e}", exc_info=True)
        raise  # Re-raise for proper error handling
        return None


# Database session dependency
def get_db():
    """Get database session."""
    try:
        from src.vault.database import SessionLocal

        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise HTTPException(status_code=500, detail="Database not available")
