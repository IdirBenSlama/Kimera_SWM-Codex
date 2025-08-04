# -*- coding: utf-8 -*-
"""
API Router for System Status, Health, and Monitoring
-----------------------------------------------------
This module contains all endpoints providing insights into the operational
status, health, and performance metrics of the KIMERA system.
"""

import logging
from fastapi import APIRouter, HTTPException, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime, timezone
from typing import Dict, Any
import threading

try:
    from ...core.kimera_system import kimera_singleton
except ImportError:
    try:
        from core.kimera_system import kimera_singleton
    except ImportError:
        # Create placeholders for core.kimera_system
            kimera_singleton = None
try:
    from ...monitoring.kimera_prometheus_metrics import get_kimera_metrics
except ImportError:
    try:
        from monitoring.kimera_prometheus_metrics import get_kimera_metrics
    except ImportError:
        # Create placeholders for monitoring.kimera_prometheus_metrics
            def get_kimera_metrics(*args, **kwargs): return None
try:
    from ...engines.asm import AxisStabilityMonitor
except ImportError:
    try:
        from engines.asm import AxisStabilityMonitor
    except ImportError:
        # Create placeholders for engines.asm
            class AxisStabilityMonitor: pass
try:
    from ...core.statistical_modeling import statistical_engine
except ImportError:
    try:
        from core.statistical_modeling import statistical_engine
    except ImportError:
        # Create placeholders for core.statistical_modeling
            statistical_engine = None
try:
    from ...utils.config import get_api_settings
except ImportError:
    try:
        from utils.config import get_api_settings
    except ImportError:
        # Create placeholders for utils.config
            def get_api_settings(*args, **kwargs): return None

logger = logging.getLogger(__name__)
router = APIRouter()

# --- API Endpoints ---

@router.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        kimera_system = kimera_singleton
        
        # Get component statuses
        vault_manager = kimera_system.get_vault_manager()
        gpu_foundation = kimera_system.get_gpu_foundation()
        contradiction_engine = kimera_system.get_contradiction_engine()
        thermodynamic_engine = kimera_system.get_thermodynamic_engine()
        
        # Get GPU status if available
        gpu_status = None
        if gpu_foundation:
            gpu_status = gpu_foundation.get_status()
        
        return {
            "status": "operational",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_state": kimera_system._state,
            "components": {
                "vault_manager": vault_manager is not None,
                "gpu_foundation": gpu_foundation is not None,
                "contradiction_engine": contradiction_engine is not None,
                "thermodynamic_engine": thermodynamic_engine is not None,
                "embedding_model": kimera_system.get_embedding_model() is not None
            },
            "gpu_status": gpu_status,
            "engines": {
                "contradiction": {
                    "available": contradiction_engine is not None,
                    "tension_threshold": contradiction_engine.tension_threshold if contradiction_engine else None
                },
                "thermodynamic": {
                    "available": thermodynamic_engine is not None,
                    "mode": thermodynamic_engine.mode.value if thermodynamic_engine else None,
                    "consciousness_threshold": thermodynamic_engine.consciousness_threshold if thermodynamic_engine else None
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/contradiction_engine")
async def get_contradiction_engine_status():
    """Get contradiction engine status and capabilities"""
    try:
        kimera_system = kimera_singleton
        contradiction_engine = kimera_system.get_contradiction_engine()
        
        if not contradiction_engine:
            return {
                "available": False,
                "error": "Contradiction engine not initialized"
            }
        
        return {
            "available": True,
            "tension_threshold": contradiction_engine.tension_threshold,
            "capabilities": [
                "tension_gradient_detection",
                "pulse_strength_calculation", 
                "collapse_surge_decisions",
                "insight_conflict_analysis"
            ],
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Error getting contradiction engine status: {e}")
        return {
            "available": False,
            "error": str(e)
        }

@router.get("/thermodynamic_engine")
async def get_thermodynamic_engine_status():
    """Get thermodynamic engine status and capabilities"""
    try:
        kimera_system = kimera_singleton
        thermodynamic_engine = kimera_system.get_thermodynamic_engine()
        
        if not thermodynamic_engine:
            return {
                "available": False,
                "error": "Thermodynamic engine not initialized"
            }
        
        # Get physics compliance report if available
        compliance_report = None
        try:
            compliance_report = thermodynamic_engine.get_physics_compliance_report()
        except Exception:
            pass
        
        return {
            "available": True,
            "mode": thermodynamic_engine.mode.value,
            "consciousness_threshold": thermodynamic_engine.consciousness_threshold,
            "temperature_scale": thermodynamic_engine.temperature_scale,
            "entropy_scale": thermodynamic_engine.entropy_scale,
            "physics_compliance": compliance_report,
            "capabilities": [
                "epistemic_temperature_calculation",
                "zetetic_carnot_cycles",
                "consciousness_detection",
                "thermodynamic_entropy_analysis"
            ],
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Error getting thermodynamic engine status: {e}")
        return {
            "available": False,
            "error": str(e)
        }

@router.get("/system/health", tags=["System"])
async def get_system_health_simple():
    """Provides a simple health check endpoint."""
    return {"status": "ok"}

@router.get("/system-metrics", tags=["System"])
async def prometheus_metrics(request: Request):
    """Exposes system metrics in Prometheus format."""
    metrics = get_kimera_metrics(request.app)
    return Response(generate_latest(metrics), media_type=CONTENT_TYPE_LATEST)

@router.get("/system/health/detailed", tags=["System"])
async def get_system_health_detailed():
    """Provides a detailed health report of all major system components."""
    
    components = {
        "vault_manager": kimera_singleton.get_vault_manager() is not None,
        "embedding_model": kimera_singleton.get_embedding_model() is not None,
        "contradiction_engine": kimera_singleton.get_contradiction_engine() is not None,
        "thermodynamic_engine": kimera_singleton.get_thermodynamic_engine() is not None,
    }
    
    overall_health = "operational" if all(components.values()) else "degraded"
    
    return {
        "overall_health": overall_health,
        "components": {name: "ok" if status else "failed" for name, status in components.items()}
    }

@router.get("/system/stability", tags=["System"])
async def get_system_stability():
    """Retrieves the system stability index from the Axis Stability Monitor."""
    try:
        # Try to get ASM from kimera system first
        kimera_system = kimera_singleton
        asm = None
        
        # Check if ASM is available in the system
        if hasattr(kimera_system, '_axis_stability_monitor'):
            asm = kimera_system._axis_stability_monitor
        else:
            # Create a new instance if not available
            try:
                asm = AxisStabilityMonitor()
            except Exception as init_error:
                logger.warning(f"Could not initialize ASM: {init_error}")
                # Return a default stable value
                return {"stability_index": 0.95, "status": "estimated", "message": "ASM not fully initialized, returning stable estimate"}
        
        if asm and hasattr(asm, 'get_stability_index'):
            stability_index = asm.get_stability_index()
            return {"stability_index": stability_index, "status": "measured"}
        else:
            return {"stability_index": 0.95, "status": "default", "message": "ASM available but stability index method not found"}
            
    except Exception as e:
        logger.error(f"Failed to get stability index: {e}", exc_info=True)
        # Return a safe default rather than failing
        return {"stability_index": 0.9, "status": "error", "message": f"Error retrieving stability: {str(e)}"}

@router.get("/system/gpu_foundation", tags=["System"])
async def get_gpu_foundation_status():
    """Gets the status of the GPU Foundation, including GPU details."""
    try:
        gpu_foundation = kimera_singleton.get_gpu_foundation()
        if not gpu_foundation:
            return {"status": "unavailable", "device": kimera_singleton.get_device()}
        
        return gpu_foundation.get_status()
    except Exception as e:
        logger.error(f"Failed to get GPU foundation status: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"GPU foundation unavailable: {str(e)}")

@router.get("/system/utilization_stats", tags=["System"])
async def get_utilization_statistics():
    """Gets utilization statistics for various system components."""
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault manager not available")

    try:
        return {
            "vault_stats": vault_manager.get_all_vault_stats() if hasattr(vault_manager, 'get_all_vault_stats') else "not_available",
            "cognitive_cycle_stats": "not_available",  # Will be implemented when cognitive cycle is restored
            "statistical_engine_cache": statistical_engine.get_cache_stats() if hasattr(statistical_engine, 'get_cache_stats') else "not_available"
        }
    except Exception as e:
        logger.error(f"Failed to get utilization stats: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to retrieve utilization statistics: {str(e)}")

@router.post("/run_zetetic_audit", tags=["System"])
async def run_zetetic_audit():
    """
    Triggers a comprehensive, real-world zetetic audit of the system's
    thermodynamic and cognitive capabilities.
    """
    try:
        # This is a simplified way to run the script.
        # In a production system, this would be a background task.
        from scripts.validation.run_zetetic_audit import ZeteticAuditor
        
        logger.info("Starting Zetetic Audit via API call...")
        auditor = ZeteticAuditor()
        report = auditor.run_audit()
        
        return {
            "message": "Zetetic audit completed successfully.",
            "report_summary": report.get("audit_summary")
        }
    except Exception as e:
        logger.error(f"Failed to run Zetetic Audit: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to run audit: {str(e)}")

@router.get("/test-metrics-redirect", tags=["System"])
async def test_metrics_redirect():
    """Test endpoint to check if redirection is happening from middleware or route handlers"""
    return {"status": "direct_access_successful"}

@router.get("/metrics", summary="Get system metrics for Prometheus", tags=["System"])
async def get_prometheus_metrics(request: Request):
    """
    Endpoint for Prometheus to scrape metrics.
    """
    metrics = get_kimera_metrics()
    # This might be a bit slow if there are many metrics, but it's the standard way.
    return Response(content=generate_latest(metrics.registry), media_type="text/plain")

@router.get("/system/state", summary="Get the current state of the Kimera system", tags=["System"])
async def get_system_state(request: Request) -> Dict[str, Any]:
    """
    Provides a detailed report on the system's cognitive and operational status.
    """
    from traceback import format_exc
    try:
        kimera_system = kimera_singleton
        metrics = get_kimera_metrics()
        errors = {}
        # Safely get system state
        try:
            if hasattr(kimera_system, '_state'):
                system_state = kimera_system._state
            elif hasattr(kimera_system, 'state'):
                system_state = kimera_system.state
            else:
                system_state = "operational"  # Default state
        except Exception as e:
            system_state = "error"
            errors['system_state'] = str(e) + "\n" + format_exc()
            logger.error(f"system_state error: {e}", exc_info=True)
        try:
            is_operational = kimera_system.is_operational() if hasattr(kimera_system, 'is_operational') else True
        except Exception as e:
            is_operational = False
            errors['is_operational'] = str(e) + "\n" + format_exc()
            logger.error(f"is_operational error: {e}", exc_info=True)
        try:
            is_shutdown = kimera_system.is_shutdown() if hasattr(kimera_system, 'is_shutdown') else False
        except Exception as e:
            is_shutdown = False
            errors['is_shutdown'] = str(e) + "\n" + format_exc()
            logger.error(f"is_shutdown error: {e}", exc_info=True)
        try:
            active_threads = threading.active_count()
        except Exception as e:
            active_threads = -1
            errors['active_threads'] = str(e) + "\n" + format_exc()
            logger.error(f"active_threads error: {e}", exc_info=True)
        try:
            metrics_instance_id = id(metrics)
        except Exception as e:
            metrics_instance_id = -1
            errors['metrics_instance_id'] = str(e) + "\n" + format_exc()
            logger.error(f"metrics_instance_id error: {e}", exc_info=True)
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_state": system_state,
            "is_operational": is_operational,
            "is_shutdown": is_shutdown,
            "active_threads": active_threads,
            "metrics_instance_id": metrics_instance_id
        }
        if errors:
            report['errors'] = errors
            report['status'] = 'error'
        else:
            report['status'] = 'ok'
        return report
    except Exception as e:
        logger.error(f"Critical error in get_system_state: {e}", exc_info=True)
        return {
            "status": "critical_error",
            "error": str(e),
            "trace": format_exc(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.post("/system/shutdown", summary="Gracefully shut down the Kimera system", tags=["System"])
async def shutdown_kimera_system(request: Request):
    """
    Initiates a graceful shutdown of all Kimera services.
    """
    kimera_system = kimera_singleton
    # Correctly call the async shutdown method
    await kimera_system.shutdown()
    return {"message": "Kimera system shutdown initiated."}

@router.get("/system/threads", summary="Get information about active threads", tags=["System"])
async def get_system_threads():
    """
    Provides a list of all active threads for debugging purposes.
    """
    metrics = get_kimera_metrics()
    
    threads = []
    for thread in threading.enumerate():
        threads.append({
            "name": thread.name,
            "id": thread.ident,
            "is_alive": thread.is_alive()
        })
    
    return {
        "threads": threads
    } 

@router.get("/system/debug_singleton", tags=["System"])
async def debug_singleton():
    """Diagnostic endpoint to check kimera_singleton status"""
    try:
        return {
            "kimera_singleton_type": str(type(kimera_singleton)),
            "kimera_singleton_repr": repr(kimera_singleton),
            "kimera_singleton_is_none": kimera_singleton is None,
            "has_attr_state": hasattr(kimera_singleton, '_state') if kimera_singleton else False,
            "has_attr_is_operational": hasattr(kimera_singleton, 'is_operational') if kimera_singleton else False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "trace": format_exc(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        } 

@router.get("/system/ping", tags=["System"])
async def ping():
    """Minimal endpoint to confirm FastAPI app and router health."""
    return {"status": "ok", "message": "Kimera API is alive", "timestamp": datetime.now(timezone.utc).isoformat()}
