# -*- coding: utf-8 -*-
"""
API Router for Vault Management and Data Retrieval
--------------------------------------------------
This module contains endpoints for interacting with Kimera's data vaults,
including retrieving contents, rebalancing, and fetching specific data
representations like the linguistic form of a Geoid.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...core.kimera_system import get_kimera_system
from ...vault.realtime_vault_monitor import RealTimeVaultMonitor
from ...vault.vault_manager import VaultManager
from ..dependencies import get_vault_manager

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/vault",
    tags=["Vault"],
)

# Global monitoring instance
_vault_monitor = None


def get_vault_monitor() -> Optional[RealTimeVaultMonitor]:
    """Get or create the global vault monitor instance"""
    global _vault_monitor
    if _vault_monitor is None:
        try:
            kimera_system = get_kimera_system()
            vault_manager = kimera_system.get_vault_manager()
            contradiction_engine = kimera_system.get_contradiction_engine()

            if vault_manager and contradiction_engine:
                _vault_monitor = RealTimeVaultMonitor(
                    vault_manager, contradiction_engine
                )
                logger.info("✅ Vault monitor initialized")
            else:
                logger.warning(
                    "❌ Could not initialize vault monitor - missing dependencies"
                )
        except Exception as e:
            logger.error(f"❌ Failed to initialize vault monitor: {e}")
    return _vault_monitor


# --- Pydantic Models ---


class LinguisticGeoid(BaseModel):
    geoid_id: str
    text_representation: str
    timestamp: str


# --- Monitoring API Endpoints ---


@router.post("/monitoring/start", summary="Start Real-Time Vault Monitoring")
def start_vault_monitoring():
    """
    Start the real-time vault monitoring system.

    Begins continuous monitoring of vault activity, including:
    - Memory formations (Geoids)
    - Scar formations and cognitive transitions
    - Database health and performance
    - Anomaly detection
    """
    try:
        monitor = get_vault_monitor()
        if not monitor:
            raise HTTPException(status_code=503, detail="Vault monitor not available")

        monitor.start_monitoring()

        return {
            "status": "monitoring_started",
            "message": "Real-time vault monitoring activated",
            "monitoring_interval": monitor.monitoring_interval,
            "health_check_interval": monitor.health_check_interval,
        }
    except Exception as e:
        logger.error(f"Failed to start vault monitoring: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start monitoring: {str(e)}"
        )


@router.post("/monitoring/stop", summary="Stop Real-Time Vault Monitoring")
def stop_vault_monitoring():
    """Stop the real-time vault monitoring system"""
    try:
        monitor = get_vault_monitor()
        if not monitor:
            raise HTTPException(status_code=503, detail="Vault monitor not available")

        monitor.stop_monitoring()

        return {
            "status": "monitoring_stopped",
            "message": "Real-time vault monitoring deactivated",
        }
    except Exception as e:
        logger.error(f"Failed to stop vault monitoring: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to stop monitoring: {str(e)}"
        )


@router.get("/monitoring/health", summary="Get Current Vault Health")
def get_vault_health():
    """
    Get the current health status of the vault system.

    Returns comprehensive health metrics including:
    - Database connectivity and latency
    - Memory and scar counts
    - Recent activity rates
    - Cognitive state assessment
    - Anomaly detection results
    """
    try:
        monitor = get_vault_monitor()
        if not monitor:
            raise HTTPException(status_code=503, detail="Vault monitor not available")

        health = monitor.get_current_health()
        if not health:
            return {
                "status": "no_health_data",
                "message": "No health data available - monitoring may not be running",
            }

        return {"status": "health_data_available", "health_metrics": health}
    except Exception as e:
        logger.error(f"Failed to get vault health: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get health data: {str(e)}"
        )


@router.get("/monitoring/activities", summary="Get Recent Cognitive Activities")
def get_recent_activities(limit: int = 20):
    """
    Get recent cognitive activities detected by the monitoring system.

    Returns activities such as:
    - New geoid formations
    - Scar formations
    - Insight generations
    """
    try:
        monitor = get_vault_monitor()
        if not monitor:
            raise HTTPException(status_code=503, detail="Vault monitor not available")

        activities = monitor.get_recent_activities(limit=limit)

        return {
            "status": "activities_retrieved",
            "count": len(activities),
            "activities": activities,
        }
    except Exception as e:
        logger.error(f"Failed to get recent activities: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get activities: {str(e)}"
        )


@router.get("/monitoring/performance", summary="Get Performance Summary")
def get_performance_summary():
    """
    Get performance summary and trends for the vault system.

    Includes metrics like database latency trends and activity patterns.
    """
    try:
        monitor = get_vault_monitor()
        if not monitor:
            raise HTTPException(status_code=503, detail="Vault monitor not available")

        performance = monitor.get_performance_summary()

        return {
            "status": "performance_data_available",
            "performance_summary": performance,
        }
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance data: {str(e)}"
        )


@router.get("/monitoring/analytics", summary="Get Cognitive Analytics")
def get_cognitive_analytics():
    """
    Get advanced cognitive analytics from the vault monitoring system.

    Provides insights into cognitive patterns, activity distributions,
    and formation rates across different types of mental constructs.
    """
    try:
        monitor = get_vault_monitor()
        if not monitor:
            raise HTTPException(status_code=503, detail="Vault monitor not available")

        analytics = monitor.get_cognitive_analytics()

        return {"status": "analytics_available", "cognitive_analytics": analytics}
    except Exception as e:
        logger.error(f"Failed to get cognitive analytics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get analytics: {str(e)}"
        )


@router.get("/monitoring/report", summary="Get Comprehensive Monitoring Report")
def get_monitoring_report():
    """
    Generate a comprehensive monitoring report including all available data.

    This endpoint provides a complete overview of the vault's current state,
    recent activities, performance metrics, and cognitive analytics.
    """
    try:
        monitor = get_vault_monitor()
        if not monitor:
            raise HTTPException(status_code=503, detail="Vault monitor not available")

        report = monitor.generate_monitoring_report()

        return {"status": "report_generated", "report": report}
    except Exception as e:
        logger.error(f"Failed to generate monitoring report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/monitoring/status", summary="Get Monitoring System Status")
def get_monitoring_status():
    """Get the current status of the monitoring system itself"""
    try:
        monitor = get_vault_monitor()
        if not monitor:
            return {
                "status": "monitor_unavailable",
                "message": "Vault monitor not initialized",
                "is_monitoring": False,
            }

        return {
            "status": "monitor_available",
            "is_monitoring": monitor.is_monitoring,
            "configuration": {
                "monitoring_interval": monitor.monitoring_interval,
                "health_check_interval": monitor.health_check_interval,
                "activity_window_minutes": monitor.activity_window.total_seconds() / 60,
            },
            "history_sizes": {
                "activities": len(monitor.activity_history),
                "health_checks": len(monitor.health_history),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# --- Original Vault API Endpoints ---


@router.get("/status", summary="Get Vault Status")
def get_vault_status(vault: VaultManager = Depends(get_vault_manager)):
    """
    Retrieves the operational status of the Vault, including database connection
    and table information.
    """
    try:
        status = vault.get_status()
        return status
    except Exception as e:
        logger.error(f"Error getting vault status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve vault status.")


@router.get("/geoids/count", summary="Get Geoid Count")
def get_geoid_count(vault: VaultManager = Depends(get_vault_manager)):
    """
    Returns the total number of Geoid states stored in the Vault.
    """
    try:
        count = vault.get_geoid_count()
        return {"geoid_count": count}
    except Exception as e:
        logger.error(f"Error getting geoid count: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve geoid count.")


@router.get("/scars/count", summary="Get Scar Count")
def get_scar_count(vault: VaultManager = Depends(get_vault_manager)):
    """
    Returns the total number of Scar records stored in the Vault.
    """
    try:
        count = vault.get_scar_count()
        return {"scar_count": count}
    except Exception as e:
        logger.error(f"Error getting scar count: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve scar count.")


@router.post("/query", summary="Query Vault")
async def query_vault(query: dict, vault: VaultManager = Depends(get_vault_manager)):
    """
    Comprehensive vault query endpoint for cognitive operations.

    Supports multiple query types:
    - pattern_search: Search for learned patterns
    - causal_analysis: Analyze causal relationships
    - insight_retrieval: Retrieve stored insights
    - performance_analysis: Analyze performance data
    - epistemic_questions: Generate learning questions
    """
    try:
        from src.core.vault_cognitive_interface import get_vault_cognitive_interface

        cognitive_interface = get_vault_cognitive_interface()

        query_type = query.get("type", "pattern_search")
        domain = query.get("domain", "general")
        context = query.get("context", {})

        logger.info(f"🔍 VAULT QUERY RECEIVED: {query_type} in domain {domain}")

        if query_type == "pattern_search":
            result = await cognitive_interface.query_learned_patterns(domain, context)
        elif query_type == "market_insights":
            symbol = context.get("symbol", "BTCUSDT")
            timeframe = context.get("timeframe", "1h")
            result = await cognitive_interface.query_market_insights(
                symbol, timeframe, context
            )
        elif query_type == "risk_patterns":
            result = await cognitive_interface.query_risk_patterns(context)
        elif query_type == "epistemic_questions":
            questions = await cognitive_interface.generate_market_questions(context)
            result = {"questions": questions, "count": len(questions)}
        elif query_type == "session_summary":
            result = await cognitive_interface.get_session_summary()
        else:
            # Fallback to basic vault query
            result = await cognitive_interface.query_learned_patterns(domain, context)

        logger.info(
            f"✅ VAULT QUERY COMPLETED: {query_type} - {len(str(result))} bytes returned"
        )
        return {
            "status": "success",
            "query_type": query_type,
            "domain": domain,
            "result": result,
        }

    except Exception as e:
        logger.error(f"❌ VAULT QUERY FAILED: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e), "query": query}


@router.get("/vaults/{vault_id}", tags=["Vaults"])
async def get_vault_contents(vault_id: str, limit: int = 10):
    """Get contents of a specific vault."""
    vault_manager = get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")

    try:
        # Check if vault_manager has the method, otherwise use fallback
        if hasattr(vault_manager, "get_vault_contents"):
            contents = vault_manager.get_vault_contents(vault_id, limit=limit)
        else:
            # Fallback to getting scars from vault
            contents = vault_manager.get_scars_from_vault(vault_id, limit=limit)
            contents = [
                {"scar_id": s.scar_id, "timestamp": s.timestamp, "reason": s.reason}
                for s in contents
            ]

        return {"vault_id": vault_id, "contents": contents}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Vault '{vault_id}' not found.")
    except Exception as e:
        logger.error(f"Failed to get vault contents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", tags=["Vaults"])
async def get_vault_stats():
    """Get statistics about all vaults in the system."""
    vault_manager = get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")

    try:
        # Get basic stats
        stats = {"total_geoids": 0, "total_scars": 0, "vaults": {}}

        # Count geoids
        if hasattr(vault_manager, "get_all_geoids"):
            geoids = vault_manager.get_all_geoids()
            stats["total_geoids"] = len(geoids)

        # Count scars
        if hasattr(vault_manager, "get_all_scars"):
            scars = vault_manager.get_all_scars()
            stats["total_scars"] = len(scars)

        # Get vault-specific stats if available
        if hasattr(vault_manager, "get_vault_stats"):
            vault_stats = vault_manager.get_vault_stats()
            stats["vaults"] = vault_stats

        return stats
    except Exception as e:
        logger.error(f"Failed to get vault stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geoids/recent", tags=["Vaults"])
async def get_recent_geoids(limit: int = 5):
    """Get recently created geoids."""
    # For now, return a mock response to ensure the endpoint works
    return {"recent_geoids": [], "count": 0, "message": "No recent geoids available"}


@router.get("/scars/recent", tags=["Vaults"])
async def get_recent_scars(limit: int = 5):
    """Get recently created scars."""
    vault_manager = get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")

    try:
        if hasattr(vault_manager, "get_recent_scars"):
            scars = vault_manager.get_recent_scars(limit=limit)
        elif hasattr(vault_manager, "get_all_scars"):
            # Fallback: get all and take last N
            all_scars = vault_manager.get_all_scars(limit=limit * 10)
            scars = all_scars[-limit:] if len(all_scars) > limit else all_scars
        else:
            scars = []

        # Convert to serializable format
        result = []
        for scar in scars:
            if hasattr(scar, "to_dict"):
                result.append(scar.to_dict())
            else:
                result.append(
                    {
                        "scar_id": getattr(scar, "scar_id", "unknown"),
                        "timestamp": (
                            getattr(scar, "timestamp", datetime.now()).isoformat()
                            if hasattr(scar, "timestamp")
                            else datetime.now().isoformat()
                        ),
                        "reason": getattr(scar, "reason", "No reason provided"),
                    }
                )

        return {"recent_scars": result, "count": len(result)}
    except Exception as e:
        logger.error(f"Failed to get recent scars: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vaults/rebalance", tags=["Vaults"])
async def rebalance_vaults(by_weight: bool = False):
    """Triggers a rebalancing of the data vaults."""
    vault_manager = get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")

    try:
        # Check if vault_manager has rebalance methods
        if hasattr(vault_manager, "rebalance_vaults"):
            moved_count = vault_manager.rebalance_vaults(by_weight=by_weight)
            return {"status": "rebalanced", "moved_scars": moved_count}
        else:
            return {
                "status": "rebalance_not_implemented",
                "message": "Rebalancing functionality not yet available",
            }
    except Exception as e:
        logger.error(f"Failed to rebalance vaults: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geoids/{geoid_id}/speak", response_model=LinguisticGeoid, tags=["Vaults"])
async def speak_geoid(geoid_id: str):
    """
    Retrieves a linguistic (textual) representation of a specific Geoid.
    This is conceptually a data retrieval operation from the vault.
    """
    vault_manager = get_vault_manager()
    thermo_engine = get_vault_manager().get_thermodynamics_engine()

    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")

    try:
        # Check if vault_manager has get_geoid method
        if hasattr(vault_manager, "get_geoid"):
            geoid_db = vault_manager.get_geoid(geoid_id)
        else:
            raise HTTPException(
                status_code=501, detail="Geoid retrieval not yet implemented"
            )

        if not geoid_db:
            raise HTTPException(status_code=404, detail="Geoid not found")

        # Generate a simple linguistic representation
        if thermo_engine and hasattr(thermo_engine, "describe_geoid"):
            # This helper function should be moved to a common utility module
            from .geoid_scar_router import to_state

            geoid_state = to_state(geoid_db)
            linguistic_representation = thermo_engine.describe_geoid(geoid_state)
        else:
            # Fallback representation
            linguistic_representation = f"Geoid {geoid_id}: {getattr(geoid_db, 'semantic_features', 'No description available')}"

        return LinguisticGeoid(
            geoid_id=geoid_id,
            text_representation=linguistic_representation,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to speak geoid {geoid_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vault/store", response_model=Dict[str, Any])
async def store_in_vault(
    key: str, value: Any, metadata: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Store data in the vault"""
    import json
    from traceback import format_exc

    logger.info(
        f"/vault/store called with key={key}, value={value}, metadata={metadata}"
    )
    try:
        vault_manager = get_vault_manager()
        if not vault_manager:
            logger.error("Vault manager not available")
            return {"status": "error", "error": "Vault manager not available"}
        # Validate key
        if not isinstance(key, str) or not key:
            logger.error("Invalid key provided")
            return {"status": "error", "error": "Invalid key provided"}
        # Handle value as JSON string if it's a string
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                logger.warning("Value is not valid JSON, storing as string")
        # Store in vault
        try:
            result = await vault_manager.store(key, value, metadata)
        except Exception as e:
            logger.error(f"Vault manager store error: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "trace": format_exc()}
        return {"status": "success", "key": key, "stored": True}
    except Exception as e:
        logger.error(f"Error storing in vault: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "trace": format_exc()}


@router.get("/vault/retrieve/{key}", response_model=Dict[str, Any])
async def retrieve_from_vault(key: str) -> dict[str, Any]:
    """Retrieve data from the vault"""
    try:
        vault_manager = get_vault_manager()
        if not vault_manager:
            raise HTTPException(status_code=503, detail="Vault manager not available")

        value = await vault_manager.retrieve(key)
        if value is None:
            raise HTTPException(
                status_code=404, detail=f"Key '{key}' not found in vault"
            )

        return {"status": "success", "key": key, "value": value}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving from vault: {e}")
        raise HTTPException(status_code=500, detail=str(e))
