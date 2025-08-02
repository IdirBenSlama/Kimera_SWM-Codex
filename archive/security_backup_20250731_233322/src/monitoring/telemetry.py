"""Prometheus / OpenTelemetry exporter for Kimera SWM.

Exposes runtime metrics such as active geoids, cycle count, and vault pressure
at `/telemetry/metrics` in Prometheus text format.
"""
from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any
import logging

log = logging.getLogger(__name__)

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

# Prometheus registry ---------------------------------------------------------
REGISTRY = CollectorRegistry()

# Gauges
ACTIVE_GEOIDS = Gauge("kimera_active_geoids", "Number of active geoids", registry=REGISTRY)
CYCLE_COUNT = Gauge("kimera_cycle_count", "Total cognitive cycles processed", registry=REGISTRY)
VAULT_PRESSURE = Gauge("kimera_vault_pressure", "Vault memory pressure", registry=REGISTRY)


# Helper ---------------------------------------------------------------------

def update_metrics(system: Dict[str, Any]) -> None:  # noqa: D401
    """Update gauge metrics from `kimera_system`."""
    try:
        ACTIVE_GEOIDS.set(len(system.get("active_geoids", {})))
        CYCLE_COUNT.set(system.get("system_state", {}).get("cycle_count", 0))
        # Example vault pressure: scars / cache size (mocked if absent)
        vp = system.get("system_state", {}).get("vault_pressure", 0.0)
        VAULT_PRESSURE.set(vp)
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to update telemetry metrics: %s", exc)


def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics for monitoring."""
    try:
        # Import here to avoid circular during app import
        from ..api.main import kimera_system  # type: ignore
        
        metrics = {
            "active_geoids": len(kimera_system.get("active_geoids", {})),
            "cycle_count": kimera_system.get("system_state", {}).get("cycle_count", 0),
            "vault_pressure": kimera_system.get("system_state", {}).get("vault_pressure", 0.0),
            "timestamp": kimera_system.get("system_state", {}).get("last_update", "unknown")
        }
        return metrics
    except Exception as exc:
        log.warning("Failed to get system metrics: %s", exc)
        return {
            "active_geoids": 0,
            "cycle_count": 0,
            "vault_pressure": 0.0,
            "timestamp": "error"
        }


# Commented out to avoid conflict with monitoring_routes.py
# @router.get("/metrics")
# async def get_telemetry_metrics():
#     return get_metrics_data() 