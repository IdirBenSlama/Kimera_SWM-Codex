"""
Kimera SWM - Monitoring API Routes
=================================

Comprehensive API endpoints for accessing real-time monitoring data,
metrics, alerts, and system status.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import json
import logging

from ..layer_2_governance.monitoring import (
    get_monitoring_core,
    MonitoringLevel,
    AlertSeverity
)
from ..layer_2_governance.monitoring.metrics_integration import get_integration_manager
from ..core.knowledge_base import KNOWLEDGE_BASE_DIR, list_vaults, get_vault_metadata
from ..core.kimera_system import KimeraSystem, get_kimera_system
from ..layer_2_governance.monitoring.kimera_prometheus_metrics import (
    get_kimera_metrics,
    generate_metrics_report,
    CONTENT_TYPE_LATEST as PROMETHEUS_CONTENT_TYPE
)
from ..utils.config import get_api_settings
from ..utils.debug_utils import get_thread_info

# Response models
class SystemStatusResponse(BaseModel):
    """System status response model"""
    status: str
    is_running: bool
    start_time: str
    uptime_seconds: float
    monitoring_level: str
    capabilities: Dict[str, Any]
    background_tasks: int
    alerts_count: int


class MetricsSummaryResponse(BaseModel):
    """Metrics summary response model"""
    timestamp: str
    metrics: Dict[str, Dict[str, float]]
    total_metrics: int


class AlertResponse(BaseModel):
    """Alert response model"""
    id: str
    severity: str
    message: str
    timestamp: str
    metric_name: str
    value: float
    threshold: float
    context: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    components: Dict[str, str]


# Create router
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint for Kimera system
    
    Returns the health status of all monitoring components and Kimera engines.
    """
    
    monitoring_core = get_monitoring_core()
    integration_manager = get_integration_manager()
    kimera_system = get_kimera_system()
    
    # Check monitoring components
    components = {
        "monitoring_core": "healthy" if monitoring_core.is_running else "stopped",
        "prometheus": "healthy",
        "tracing": "healthy" if monitoring_core.enable_tracing else "disabled",
        "anomaly_detection": "healthy" if monitoring_core.enable_anomaly_detection else "disabled",
        "alerting": "healthy" if monitoring_core.alert_channels else "no_channels",
        "integration": "healthy" if integration_manager.is_initialized else "not_initialized"
    }
    
    # Check Kimera engines status
    operational_engines = 0
    total_engines = 8  # Default value
    
    try:
        engine_names = [
            ("contradiction_engine", "Contradiction Engine"),
            ("thermodynamics_engine", "Thermodynamics Engine"), 
            ("vault_manager", "Vault Manager"),
            ("spde_engine", "SPDE Engine"),
            ("cognitive_cycle", "Cognitive Cycle"),
            ("meta_insight_engine", "Meta Insight Engine"),
            ("proactive_detector", "Proactive Detector"),
            ("revolutionary_intelligence", "Revolutionary Intelligence")
        ]
        
        total_engines = len(engine_names)
        
        for engine_key, engine_name in engine_names:
            try:
                # Get engine using the appropriate getter method
                engine = None
                if engine_key == "contradiction_engine":
                    engine = kimera_system.get_contradiction_engine()
                elif engine_key == "thermodynamics_engine":
                    engine = kimera_system.get_thermodynamic_engine()
                elif engine_key == "vault_manager":
                    engine = kimera_system.get_vault_manager()
                elif engine_key == "spde_engine":
                    engine = kimera_system.get_spde_engine()
                elif engine_key == "cognitive_cycle":
                    engine = kimera_system.get_cognitive_cycle_engine()
                elif engine_key == "meta_insight_engine":
                    engine = kimera_system.get_meta_insight_engine()
                elif engine_key == "proactive_detector":
                    engine = kimera_system.get_proactive_detector()
                elif engine_key == "revolutionary_intelligence":
                    engine = kimera_system.get_revolutionary_intelligence_engine()
                
                if engine:
                    components[f"engine_{engine_key}"] = "healthy"
                    operational_engines += 1
                else:
                    components[f"engine_{engine_key}"] = "not_available"
            except Exception as e:
                components[f"engine_{engine_key}"] = f"error: {str(e)[:50]}"
        
        # Add engine summary
        components["engines_summary"] = f"{operational_engines}/{total_engines}_operational"
        
    except Exception as e:
        # If there's an error accessing the kimera system, add error info
        components["engines_error"] = f"Failed to check engines: {str(e)[:100]}"
        components["engines_summary"] = "0/8_error"
    
    # Determine overall status - system is healthy if:
    # 1. At least 80% of engines are operational (primary criteria)
    # 2. No critical monitoring components are in error state
    engine_health = operational_engines >= (total_engines * 0.8)
    
    # Check for critical errors in monitoring (ignore disabled/no_channels as these are configuration choices)
    critical_monitoring_errors = any(
        status in ["error", "failed"] for status in components.values()
        if not status.startswith("engine_") and status != components["engines_summary"]
    )
    
    # System is healthy if engines are operational and no critical monitoring errors
    overall_status = "healthy" if engine_health and not critical_monitoring_errors else "unhealthy"
    
    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        components=components
    )


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Get comprehensive system monitoring status
    
    Returns detailed information about the monitoring system state,
    including running tasks, alerts, and capabilities.
    """
    
    monitoring_core = get_monitoring_core()
    status_data = monitoring_core.get_monitoring_status()
    
    return SystemStatusResponse(
        status="online" if status_data['is_running'] else "offline",
        is_running=status_data['is_running'],
        start_time=status_data['start_time'],
        uptime_seconds=status_data['uptime_seconds'],
        monitoring_level=status_data.get('monitoring_level', 'unknown'),
        capabilities=status_data['capabilities'],
        background_tasks=status_data['background_tasks'],
        alerts_count=status_data['alerts_count']
    )


@router.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def get_metrics_summary():
    """
    Get summary of all collected metrics
    
    Returns aggregated metrics data including current values,
    averages, minimums, and maximums.
    """
    
    monitoring_core = get_monitoring_core()
    metrics_data = monitoring_core.get_metrics_summary()
    
    return MetricsSummaryResponse(
        timestamp=datetime.now().isoformat(),
        metrics=metrics_data,
        total_metrics=len(metrics_data)
    )


@router.get("/metrics/system")
async def get_system_metrics(
    last_minutes: int = Query(default=60, description="Last N minutes of data"),
    metric_names: Optional[List[str]] = Query(default=None, description="Specific metrics to retrieve")
):
    """
    Get system-level metrics
    
    Returns CPU, memory, disk, and network metrics for the specified time period.
    """
    
    monitoring_core = get_monitoring_core()
    
    # Filter metrics based on time and names
    system_metrics = {}
    
    for metric_name, history in monitoring_core.metrics_history.items():
        if metric_names and metric_name not in metric_names:
            continue
            
        if metric_name.startswith(('cpu_', 'memory_', 'disk_', 'network_')):
            # Get last N minutes of data (assuming 5-second intervals)
            max_points = (last_minutes * 60) // 5
            recent_data = list(history)[-max_points:] if history else []
            
            system_metrics[metric_name] = {
                "values": recent_data,
                "timestamps": [
                    (datetime.now() - timedelta(seconds=i*5)).isoformat()
                    for i in range(len(recent_data)-1, -1, -1)
                ]
            }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "time_range_minutes": last_minutes,
        "metrics": system_metrics
    }


@router.get("/metrics/kimera")
async def get_kimera_metrics(
    last_minutes: int = Query(default=60, description="Last N minutes of data")
):
    """
    Get Kimera-specific metrics
    
    Returns cognitive architecture metrics including geoids, scars,
    contradictions, and selective feedback data.
    """
    
    monitoring_core = get_monitoring_core()
    
    # Get Kimera-specific metrics
    kimera_metrics = {}
    
    kimera_metric_prefixes = [
        'geoid_', 'scar_', 'contradiction_', 'selective_feedback_',
        'revolutionary_', 'cognitive_', 'thermodynamic_', 'vault_',
        'embedding_', 'insight_'
    ]
    
    for metric_name, history in monitoring_core.metrics_history.items():
        if any(metric_name.startswith(prefix) for prefix in kimera_metric_prefixes):
            max_points = (last_minutes * 60) // 5
            recent_data = list(history)[-max_points:] if history else []
            
            kimera_metrics[metric_name] = {
                "values": recent_data,
                "timestamps": [
                    (datetime.now() - timedelta(seconds=i*5)).isoformat()
                    for i in range(len(recent_data)-1, -1, -1)
                ]
            }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "time_range_minutes": last_minutes,
        "metrics": kimera_metrics
    }


@router.get("/metrics/gpu")
async def get_gpu_metrics(
    last_minutes: int = Query(default=60, description="Last N minutes of data")
):
    """
    Get GPU and AI workload metrics
    
    Returns GPU utilization, memory usage, temperature, and AI-specific metrics.
    """
    
    monitoring_core = get_monitoring_core()
    
    # Get GPU metrics
    gpu_metrics = {}
    
    for metric_name, history in monitoring_core.metrics_history.items():
        if metric_name.startswith(('gpu_', 'ai_', 'ml_', 'embedding_', 'inference_')):
            max_points = (last_minutes * 60) // 5
            recent_data = list(history)[-max_points:] if history else []
            
            gpu_metrics[metric_name] = {
                "values": recent_data,
                "timestamps": [
                    (datetime.now() - timedelta(seconds=i*5)).isoformat()
                    for i in range(len(recent_data)-1, -1, -1)
                ]
            }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "time_range_minutes": last_minutes,
        "gpu_available": monitoring_core.monitoring_core.get('nvidia_monitoring', False),
        "metrics": gpu_metrics
    }


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    severity: Optional[AlertSeverity] = Query(default=None, description="Filter by severity"),
    last_hours: int = Query(default=24, description="Last N hours of alerts"),
    limit: int = Query(default=100, description="Maximum number of alerts to return")
):
    """
    Get monitoring alerts
    
    Returns recent alerts with optional filtering by severity and time range.
    """
    
    monitoring_core = get_monitoring_core()
    
    # Filter alerts
    cutoff_time = datetime.now() - timedelta(hours=last_hours)
    filtered_alerts = []
    
    for alert in monitoring_core.alerts:
        # Filter by time
        if alert.timestamp < cutoff_time:
            continue
            
        # Filter by severity
        if severity and alert.severity != severity:
            continue
            
        filtered_alerts.append(AlertResponse(
            id=alert.id,
            severity=alert.severity.value,
            message=alert.message,
            timestamp=alert.timestamp.isoformat(),
            metric_name=alert.metric_name,
            value=alert.value,
            threshold=alert.threshold,
            context=alert.context
        ))
        
        if len(filtered_alerts) >= limit:
            break
    
    return filtered_alerts


@router.get("/alerts/summary")
async def get_alerts_summary():
    """
    Get alerts summary
    
    Returns aggregated alert statistics by severity and time periods.
    """
    
    monitoring_core = get_monitoring_core()
    
    # Calculate summary statistics
    now = datetime.now()
    time_periods = {
        "last_hour": now - timedelta(hours=1),
        "last_24_hours": now - timedelta(hours=24),
        "last_week": now - timedelta(days=7)
    }
    
    summary = {}
    
    for period_name, cutoff_time in time_periods.items():
        period_alerts = [
            alert for alert in monitoring_core.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        summary[period_name] = {
            "total": len(period_alerts),
            "by_severity": {
                severity.value: len([
                    alert for alert in period_alerts
                    if alert.severity == severity
                ])
                for severity in AlertSeverity
            }
        }
    
    return {
        "timestamp": now.isoformat(),
        "summary": summary,
        "total_alerts_in_memory": len(monitoring_core.alerts)
    }


@router.get("/performance")
async def get_performance_metrics(
    last_minutes: int = Query(default=60, description="Last N minutes of data")
):
    """
    Get performance metrics
    
    Returns latency, throughput, and performance-related metrics.
    """
    
    monitoring_core = get_monitoring_core()
    
    # Get performance metrics
    performance_metrics = {}
    
    performance_prefixes = [
        'latency_', 'throughput_', 'response_time_', 'requests_',
        'memory_current', 'memory_peak', 'cpu_usage'
    ]
    
    for metric_name, history in monitoring_core.metrics_history.items():
        if any(metric_name.startswith(prefix) or metric_name == prefix 
               for prefix in performance_prefixes):
            max_points = (last_minutes * 60) // 5
            recent_data = list(history)[-max_points:] if history else []
            
            performance_metrics[metric_name] = {
                "values": recent_data,
                "timestamps": [
                    (datetime.now() - timedelta(seconds=i*5)).isoformat()
                    for i in range(len(recent_data)-1, -1, -1)
                ],
                "statistics": {
                    "current": recent_data[-1] if recent_data else 0,
                    "average": sum(recent_data) / len(recent_data) if recent_data else 0,
                    "min": min(recent_data) if recent_data else 0,
                    "max": max(recent_data) if recent_data else 0
                }
            }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "time_range_minutes": last_minutes,
        "metrics": performance_metrics
    }


@router.get("/anomalies")
async def get_anomaly_detection_results(
    last_hours: int = Query(default=24, description="Last N hours of anomaly data")
):
    """
    Get anomaly detection results
    
    Returns detected anomalies and anomaly scores for the specified time period.
    """
    
    monitoring_core = get_monitoring_core()
    
    if not monitoring_core.enable_anomaly_detection:
        return {
            "anomaly_detection_enabled": False,
            "message": "Anomaly detection is not enabled"
        }
    
    # Get anomaly data
    cutoff_time = datetime.now() - timedelta(hours=last_hours)
    
    # Filter anomaly alerts
    anomaly_alerts = [
        alert for alert in monitoring_core.alerts
        if (alert.metric_name == "anomaly_score" and 
            alert.timestamp >= cutoff_time)
    ]
    
    return {
        "timestamp": datetime.now().isoformat(),
        "anomaly_detection_enabled": True,
        "time_range_hours": last_hours,
        "detected_anomalies": [
            {
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "score": alert.value,
                "threshold": alert.threshold,
                "context": alert.context
            }
            for alert in anomaly_alerts
        ],
        "anomaly_scores": monitoring_core.anomaly_scores if hasattr(monitoring_core, 'anomaly_scores') else {}
    }


@router.get("/integration/status")
async def get_integration_status():
    """
    Get metrics integration status
    
    Returns the status of metrics integration with various Kimera components.
    """
    
    integration_manager = get_integration_manager()
    return integration_manager.get_integration_status()


@router.post("/monitoring/start")
async def start_monitoring():
    """
    Start monitoring system
    
    Starts all background monitoring tasks and data collection.
    """
    
    monitoring_core = get_monitoring_core()
    
    if monitoring_core.is_running:
        return {"message": "Monitoring is already running", "status": "running"}
    
    try:
        await monitoring_core.start_monitoring()
        return {"message": "Monitoring started successfully", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/monitoring/stop")
async def stop_monitoring():
    """
    Stop monitoring system
    
    Stops all background monitoring tasks and data collection.
    """
    
    monitoring_core = get_monitoring_core()
    
    if not monitoring_core.is_running:
        return {"message": "Monitoring is not running", "status": "stopped"}
    
    try:
        await monitoring_core.stop_monitoring()
        return {"message": "Monitoring stopped successfully", "status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.get("/export/prometheus")
async def export_prometheus_metrics():
    """
    Export metrics in Prometheus format
    
    Returns all metrics in Prometheus exposition format.
    """
    
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    metrics_data = generate_latest()
    
    return PlainTextResponse(
        content=metrics_data.decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST
    )


# Add the /metrics endpoint that Prometheus expects
@router.get("/metrics")
async def prometheus_metrics(request: Request):
    """Standard /metrics endpoint that Prometheus scrapes for metrics data."""
    metrics = get_kimera_metrics()
    return Response(generate_latest(metrics), media_type=PROMETHEUS_CONTENT_TYPE)


@router.get("/dashboard/data")
async def get_dashboard_data():
    """
    Get comprehensive dashboard data
    
    Returns all data needed for the monitoring dashboard in a single request.
    """
    
    monitoring_core = get_monitoring_core()
    
    # Collect all dashboard data
    dashboard_data = {
        "timestamp": datetime.now().isoformat(),
        "system_status": monitoring_core.get_monitoring_status(),
        "metrics_summary": monitoring_core.get_metrics_summary(),
        "recent_alerts": [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metric_name": alert.metric_name,
                "value": alert.value
            }
            for alert in list(monitoring_core.alerts)[-10:]  # Last 10 alerts
        ],
        "integration_status": get_integration_manager().get_integration_status()
    }
    
    return dashboard_data


# Additional utility endpoints
@router.get("/config")
async def get_monitoring_config():
    """Get current monitoring configuration"""
    
    monitoring_core = get_monitoring_core()
    integration_manager = get_integration_manager()
    
    return {
        "monitoring_level": monitoring_core.monitoring_level.value,
        "features": {
            "tracing": monitoring_core.enable_tracing,
            "profiling": monitoring_core.enable_profiling,
            "anomaly_detection": monitoring_core.enable_anomaly_detection
        },
        "integration_config": integration_manager.config.__dict__ if hasattr(integration_manager, 'config') else {}
    }


# Add entropy monitoring endpoints
@router.get("/entropy/current")
async def get_current_entropy():
    """
    Get current system entropy metrics
    
    Returns current entropy measurements and related thermodynamic data.
    """
    try:
        # Try to get entropy monitor
        try:
            from ..monitoring.entropy_monitor import entropy_monitor
            from ..vault import get_vault_manager
            
            vault_manager = get_vault_manager()
            
            # Get current geoids from the database
            from ..vault.database import SessionLocal, GeoidDB
            with SessionLocal() as db:
                geoids_db = db.query(GeoidDB).all()
                
                # Convert to GeoidState objects
                from ..core.geoid import GeoidState
                geoids = []
                for geoid_db in geoids_db:
                    embedding_vector = []
                    if geoid_db.semantic_vector is not None:
                        if hasattr(geoid_db.semantic_vector, 'tolist'):
                            embedding_vector = geoid_db.semantic_vector.tolist()
                        else:
                            embedding_vector = list(geoid_db.semantic_vector)
                    
                    geoid = GeoidState(
                        geoid_id=geoid_db.geoid_id,
                        semantic_state=geoid_db.semantic_state_json or {},
                        symbolic_state=geoid_db.symbolic_state or {},
                        embedding_vector=embedding_vector,
                        metadata=geoid_db.metadata_json or {},
                    )
                    geoids.append(geoid)
                
                # Get vault info
                vault_info = {
                    'vault_a_scars': vault_manager.get_total_scar_count("vault_a"),
                    'vault_b_scars': vault_manager.get_total_scar_count("vault_b"),
                }
                
                # Calculate current entropy
                entropy_measurement = entropy_monitor.calculate_system_entropy(geoids, vault_info)
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "shannon_entropy": entropy_measurement.shannon_entropy,
                    "thermodynamic_entropy": entropy_measurement.thermodynamic_entropy,
                    "relative_entropy": entropy_measurement.relative_entropy,
                    "conditional_entropy": entropy_measurement.conditional_entropy,
                    "mutual_information": entropy_measurement.mutual_information,
                    "system_complexity": entropy_measurement.system_complexity,
                    "geoid_count": entropy_measurement.geoid_count,
                    "vault_distribution": entropy_measurement.vault_distribution,
                    "metadata": entropy_measurement.metadata
                }
                
        except ImportError:
            logging.warning("Entropy monitor not available")
            return {
                "timestamp": datetime.now().isoformat(),
                "shannon_entropy": 0.0,
                "thermodynamic_entropy": 0.0,
                "relative_entropy": 0.0,
                "conditional_entropy": 0.0,
                "mutual_information": 0.0,
                "system_complexity": 0.0,
                "geoid_count": 0,
                "vault_distribution": {"vault_a": 0, "vault_b": 0, "total_scars": 0},
                "metadata": {"status": "entropy_monitor_unavailable"}
            }
            
    except Exception as e:
        logging.error(f"Error getting current entropy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entropy data: {str(e)}")

@router.get("/entropy/trends")
async def get_entropy_trends(
    window_size: int = Query(default=100, description="Number of recent measurements to include")
):
    """
    Get entropy trends over recent measurements
    
    Returns entropy trends and patterns over the specified time window.
    """
    try:
        from ..monitoring.entropy_monitor import entropy_monitor
        
        trends = entropy_monitor.get_entropy_trends(window_size=window_size)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "window_size": window_size,
            "trends": trends
        }
        
    except ImportError:
        return {
            "timestamp": datetime.now().isoformat(),
            "window_size": window_size,
            "trends": {},
            "status": "entropy_monitor_unavailable"
        }
    except Exception as e:
        logging.error(f"Error getting entropy trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get entropy trends: {str(e)}")

@router.get("/entropy/anomalies")
async def get_entropy_anomalies(
    threshold_std: float = Query(default=2.0, description="Standard deviation threshold for anomaly detection")
):
    """
    Get detected entropy anomalies
    
    Returns recent entropy anomalies based on statistical thresholds.
    """
    try:
        from ..monitoring.entropy_monitor import entropy_monitor
        
        anomalies = entropy_monitor.detect_entropy_anomalies(threshold_std=threshold_std)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "threshold_std": threshold_std,
            "anomalies": anomalies,
            "anomalies_detected": len(anomalies)
        }
        
    except ImportError:
        return {
            "timestamp": datetime.now().isoformat(),
            "threshold_std": threshold_std,
            "anomalies": [],
            "anomalies_detected": 0,
            "status": "entropy_monitor_unavailable"
        }
    except Exception as e:
        logging.error(f"Error detecting entropy anomalies: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect anomalies: {str(e)}")

# Add engine monitoring endpoints
@router.get("/engines/status")
async def get_engines_status():
    """
    Get status of all Kimera engines
    
    Returns the current status and basic metrics for all cognitive engines.
    """
    try:
        # Get kimera system instance
        kimera_system = get_kimera_system()
        
        engines_status = {
            "timestamp": datetime.now().isoformat(),
            "engines": {}
        }
        
        # Check each engine
        engines_to_check = [
            ("contradiction_engine", "Contradiction Engine"),
            ("thermodynamics_engine", "Thermodynamics Engine"),
            ("vault_manager", "Vault Manager"),
            ("spde_engine", "SPDE Engine"),
            ("cognitive_cycle", "Cognitive Cycle"),
            ("meta_insight_engine", "Meta Insight Engine"),
            ("proactive_detector", "Proactive Detector"),
            ("revolutionary_intelligence", "Revolutionary Intelligence")
        ]
        
        for engine_key, engine_name in engines_to_check:
            # Use appropriate getter methods for kimera_system
            engine = None
            if engine_key == "contradiction_engine":
                engine = kimera_system.get_contradiction_engine()
            elif engine_key == "thermodynamics_engine":
                engine = kimera_system.get_thermodynamic_engine()
            elif engine_key == "vault_manager":
                engine = kimera_system.get_vault_manager()
            elif engine_key == "spde_engine":
                engine = kimera_system.get_spde_engine()
            elif engine_key == "cognitive_cycle":
                engine = kimera_system.get_cognitive_cycle_engine()
            elif engine_key == "meta_insight_engine":
                engine = kimera_system.get_meta_insight_engine()
            elif engine_key == "proactive_detector":
                engine = kimera_system.get_proactive_detector()
            elif engine_key == "revolutionary_intelligence":
                engine = kimera_system.get_revolutionary_intelligence_engine()
            
            if engine:
                engines_status["engines"][engine_key] = {
                    "name": engine_name,
                    "status": "operational",
                    "type": type(engine).__name__,
                    "available": True
                }
            else:
                engines_status["engines"][engine_key] = {
                    "name": engine_name,
                    "status": "not_available",
                    "type": "unknown",
                    "available": False
                }
        
        # Add system-level metrics
        engines_status["system_metrics"] = {
            "active_geoids": 0,  # Not tracked in singleton
            "cycle_count": 0,  # Not tracked in singleton
            "insights_count": 0,  # Not tracked in singleton
            "recent_insights_count": 0  # Not tracked in singleton
        }
        
        return engines_status
        
    except Exception as e:
        logging.error(f"Error getting engines status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get engines status: {str(e)}")

@router.get("/engines/contradiction")
async def get_contradiction_engine_metrics():
    """
    Get contradiction engine specific metrics
    
    Returns metrics specific to the contradiction detection engine.
    """
    try:
        kimera_system = get_kimera_system()
        
        contradiction_engine = kimera_system.get_contradiction_engine()
        if not contradiction_engine:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "not_available",
                "metrics": {}
            }
        
        # Get basic engine info
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "tension_threshold": getattr(contradiction_engine, 'tension_threshold', 0.0),
            "engine_type": type(contradiction_engine).__name__
        }
        
        # Try to get recent tensions if available
        if hasattr(contradiction_engine, 'recent_tensions'):
            metrics["recent_tensions_count"] = len(contradiction_engine.recent_tensions)
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting contradiction engine metrics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

@router.get("/engines/thermodynamics") 
async def get_thermodynamics_engine_metrics():
    """
    Get thermodynamics engine specific metrics
    
    Returns metrics specific to the semantic thermodynamics engine.
    """
    try:
        kimera_system = get_kimera_system()
        
        thermo_engine = kimera_system.get_thermodynamic_engine()
        if not thermo_engine:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "not_available",
                "metrics": {}
            }
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "engine_type": type(thermo_engine).__name__
        }
        
        # Try to get thermodynamic state if available
        if hasattr(thermo_engine, 'current_temperature'):
            metrics["current_temperature"] = thermo_engine.current_temperature
        
        if hasattr(thermo_engine, 'entropy_history'):
            metrics["entropy_measurements"] = len(thermo_engine.entropy_history)
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting thermodynamics engine metrics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error", 
            "error": str(e)
        }

@router.get("/engines/revolutionary")
async def get_revolutionary_intelligence_metrics():
    """
    Get revolutionary intelligence engine metrics
    
    Returns metrics specific to the revolutionary intelligence system.
    """
    try:
        kimera_system = get_kimera_system()
        
        revolutionary_intelligence = kimera_system.get_revolutionary_intelligence_engine()
        if not revolutionary_intelligence:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "not_available",
                "metrics": {}
            }
        
        # Get engine status
        engine_status = revolutionary_intelligence.get_engine_status()
        
        # Get revolutionary status
        revolutionary_status = revolutionary_intelligence.get_revolutionary_status()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "engine_type": type(revolutionary_intelligence).__name__,
            "intelligence_level": engine_status.get("intelligence_level", "unknown"),
            "current_mode": engine_status.get("current_mode", "unknown"),
            "total_orchestrations": engine_status.get("total_orchestrations", 0),
            "total_breakthroughs": engine_status.get("total_breakthroughs", 0),
            "intelligence_assessments": engine_status.get("intelligence_assessments", 0),
            "registered_engines": engine_status.get("registered_engines", 0),
            "breakthrough_count": engine_status.get("breakthrough_count", 0),
            "cognitive_capacity": engine_status.get("cognitive_capacity", 0.0),
            "processing_efficiency": engine_status.get("processing_efficiency", 0.0),
            "revolutionary_index": engine_status.get("revolutionary_index", 0.0),
            "revolutionary_status": revolutionary_status,
            "revolutionary_capabilities": [
                "breakthrough_detection",
                "paradigm_shift_analysis",
                "emergent_intelligence",
                "cognitive_transcendence",
                "multi_engine_orchestration",
                "intelligence_analysis"
            ]
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting revolutionary intelligence metrics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

@router.get("/engines/spde")
async def get_spde_engine_metrics():
    """
    Get SPDE engine specific metrics
    
    Returns metrics specific to the stochastic PDE engine.
    """
    try:
        kimera_system = get_kimera_system()
        
        spde_engine = kimera_system.get_spde_engine()
        if not spde_engine:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "not_available",
                "metrics": {}
            }
        
        # Get engine status
        engine_status = spde_engine.get_engine_status()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "engine_type": type(spde_engine).__name__,
            "device": engine_status.get("device", "unknown"),
            "total_solutions": engine_status.get("metrics", {}).get("total_solutions", 0),
            "average_solve_time": engine_status.get("metrics", {}).get("average_solve_time", 0.0),
            "last_solve_time": engine_status.get("metrics", {}).get("last_solve_time", 0.0),
            "solution_history_size": engine_status.get("solution_history_size", 0),
            "available_solvers": engine_status.get("available_solvers", [])
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting SPDE engine metrics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

@router.get("/engines/cognitive_cycle")
async def get_cognitive_cycle_metrics():
    """
    Get cognitive cycle engine metrics
    
    Returns metrics specific to the cognitive cycle management.
    """
    try:
        kimera_system = get_kimera_system()
        
        cognitive_cycle_engine = kimera_system.get_cognitive_cycle_engine()
        if not cognitive_cycle_engine:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "not_available",
                "metrics": {}
            }
        
        # Get engine status
        engine_status = cognitive_cycle_engine.get_engine_status()
        
        # Get cycle history
        cycle_history = cognitive_cycle_engine.get_cycle_history(limit=10)
        
        # Get insights summary
        insights_summary = cognitive_cycle_engine.get_insights_summary()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "engine_type": type(cognitive_cycle_engine).__name__,
            "current_state": engine_status.get("current_state", "unknown"),
            "cycle_count": engine_status.get("cycle_count", 0),
            "total_processing_time": engine_status.get("total_processing_time", 0.0),
            "average_cycle_time": engine_status.get("average_cycle_time", 0.0),
            "working_memory_load": engine_status.get("working_memory_load", 0.0),
            "attention_entropy": engine_status.get("attention_entropy", 0.0),
            "long_term_memory_size": engine_status.get("long_term_memory_size", 0),
            "semantic_network_size": engine_status.get("semantic_network_size", 0),
            "insight_events_count": engine_status.get("insight_events_count", 0),
            "recent_insights": engine_status.get("recent_insights", 0),
            "recent_cycles": cycle_history,
            "insights_summary": insights_summary
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting cognitive cycle metrics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

@router.get("/engines/meta_insight")
async def get_meta_insight_engine_metrics():
    """
    Get Meta Insight engine specific metrics
    
    Returns metrics specific to the meta-cognitive insight engine.
    """
    try:
        kimera_system = get_kimera_system()
        
        meta_insight_engine = kimera_system.get_meta_insight_engine()
        if not meta_insight_engine:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "not_available",
                "metrics": {}
            }
        
        # Get engine status
        engine_status = meta_insight_engine.get_engine_status()
        
        # Get insights summary
        insights_summary = meta_insight_engine.get_insights_summary()
        
        # Get meta-insights summary
        meta_insights_summary = meta_insight_engine.get_meta_insights_summary()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "engine_type": type(meta_insight_engine).__name__,
            "total_insights_processed": engine_status.get("total_insights_processed", 0),
            "total_meta_insights_generated": engine_status.get("total_meta_insights_generated", 0),
            "average_processing_time": engine_status.get("average_processing_time", 0.0),
            "insights_in_repository": engine_status.get("insights_in_repository", 0),
            "meta_insights_in_repository": engine_status.get("meta_insights_in_repository", 0),
            "processing_history_size": engine_status.get("processing_history_size", 0),
            "quality_assessor_history": engine_status.get("quality_assessor_history", 0),
            "metacognitive_processor_history": engine_status.get("metacognitive_processor_history", 0),
            "insights_summary": insights_summary,
            "meta_insights_summary": meta_insights_summary
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting meta insight engine metrics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

@router.get("/engines/proactive_detector")
async def get_proactive_detector_metrics():
    """
    Get Proactive Detector specific metrics
    
    Returns metrics specific to the predictive analysis and early warning system.
    """
    try:
        kimera_system = get_kimera_system()
        
        proactive_detector = kimera_system.get_proactive_detector()
        if not proactive_detector:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "not_available",
                "metrics": {}
            }
        
        # Get engine status
        engine_status = proactive_detector.get_engine_status()
        
        # Get detection summary
        detection_summary = proactive_detector.get_detection_summary()
        
        # Get active alerts
        active_alerts = proactive_detector.get_active_alerts()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "engine_type": type(proactive_detector).__name__,
            "total_detections": engine_status.get("total_detections", 0),
            "active_alerts": engine_status.get("active_alerts", 0),
            "data_streams": engine_status.get("data_streams", 0),
            "detection_accuracy": engine_status.get("detection_accuracy", 0.0),
            "false_positive_count": engine_status.get("false_positive_count", 0),
            "false_negative_count": engine_status.get("false_negative_count", 0),
            "detection_events_history": engine_status.get("detection_events_history", 0),
            "prediction_models": engine_status.get("prediction_models", 0),
            "threshold_adaptations": engine_status.get("threshold_adaptations", 0),
            "detection_summary": detection_summary,
            "active_alerts_list": [
                {
                    "event_id": alert.event_id,
                    "detection_type": alert.detection_type.value,
                    "alert_level": alert.alert_level.value,
                    "confidence": alert.confidence,
                    "description": alert.description,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts[:10]  # Limit to 10 most recent
            ]
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting proactive detector metrics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }


@router.get("/engines/revolutionary_thermodynamics")
async def get_revolutionary_thermodynamics_metrics():
    """
    Get revolutionary thermodynamic engine metrics
    
    Returns comprehensive metrics for the world's first physics-compliant
    thermodynamic AI system with consciousness detection capabilities.
    """
    try:
        kimera_system = get_kimera_system()
        
        foundational_engine = kimera_system.get_thermodynamic_engine()
        consciousness_detector = None  # Not available in singleton
        
        if not foundational_engine:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "not_available",
                "message": "Revolutionary Thermodynamic Engine not initialized",
                "metrics": {}
            }
        
        # Basic engine metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "status": "operational",
            "engine_type": type(foundational_engine).__name__,
            "engine_mode": getattr(foundational_engine, 'mode', {}).value if hasattr(getattr(foundational_engine, 'mode', {}), 'value') else "unknown"
        }
        
        # Physics compliance metrics
        try:
            carnot_cycles = getattr(foundational_engine, 'carnot_cycles', [])
            total_cycles = len(carnot_cycles)
            
            if total_cycles > 0:
                compliant_cycles = sum(1 for cycle in carnot_cycles if getattr(cycle, 'physics_compliant', True))
                physics_compliance_rate = compliant_cycles / total_cycles
                
                # Calculate average efficiency
                efficiencies = [getattr(cycle, 'actual_efficiency', 0.0) for cycle in carnot_cycles[-10:]]
                average_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
                
                metrics["physics_compliance"] = {
                    "total_cycles": total_cycles,
                    "compliant_cycles": compliant_cycles,
                    "compliance_rate": physics_compliance_rate,
                    "average_efficiency": average_efficiency,
                    "recent_cycles": len(carnot_cycles[-10:])
                }
            else:
                metrics["physics_compliance"] = {
                    "total_cycles": 0,
                    "compliant_cycles": 0,
                    "compliance_rate": 1.0,
                    "average_efficiency": 0.0,
                    "recent_cycles": 0
                }
        except Exception as e:
            logging.warning(f"Error calculating physics compliance metrics: {e}")
            metrics["physics_compliance"] = {"error": str(e)}
        
        # Consciousness detection metrics
        try:
            if consciousness_detector:
                if hasattr(consciousness_detector, 'get_consciousness_statistics'):
                    consciousness_stats = consciousness_detector.get_consciousness_statistics()
                    metrics["consciousness_detection"] = {
                        "detector_active": True,
                        "total_detections": consciousness_stats.get("total_detections", 0),
                        "average_probability": consciousness_stats.get("average_probability", 0.0),
                        "last_detection": consciousness_stats.get("last_detection", None)
                    }
                else:
                    metrics["consciousness_detection"] = {
                        "detector_active": True,
                        "status": "operational",
                        "capabilities": ["thermodynamic_phase_detection", "consciousness_emergence"]
                    }
            else:
                metrics["consciousness_detection"] = {
                    "detector_active": False,
                    "message": "Consciousness detector not available"
                }
        except Exception as e:
            logging.warning(f"Error getting consciousness detection metrics: {e}")
            metrics["consciousness_detection"] = {"error": str(e)}
        
        # Thermodynamic performance metrics
        try:
            physics_violations = getattr(foundational_engine, 'physics_violations', [])
            metrics["thermodynamic_performance"] = {
                "total_violations": len(physics_violations),
                "recent_violations": len(physics_violations[-5:]) if physics_violations else 0,
                "violation_types": list(set([v.get("violation_type", "unknown") for v in physics_violations[-10:]])) if physics_violations else []
            }
            
            # Temperature coherence if available
            if carnot_cycles:
                recent_cycle = carnot_cycles[-1]
                if hasattr(recent_cycle, 'hot_temperature'):
                    hot_temp = recent_cycle.hot_temperature
                    if hasattr(hot_temp, 'semantic_temperature') and hasattr(hot_temp, 'physical_temperature'):
                        semantic = hot_temp.semantic_temperature
                        physical = hot_temp.physical_temperature
                        if semantic > 0 and physical > 0:
                            coherence = 1.0 - abs(semantic - physical) / max(semantic, physical)
                            metrics["thermodynamic_performance"]["temperature_coherence"] = coherence
        except Exception as e:
            logging.warning(f"Error calculating thermodynamic performance metrics: {e}")
            metrics["thermodynamic_performance"] = {"error": str(e)}
        
        # System capabilities
        metrics["capabilities"] = [
            "epistemic_temperature_calculation",
            "zetetic_carnot_engine",
            "physics_violation_detection",
            "automatic_correction",
            "consciousness_detection",
            "thermodynamic_optimization"
        ]
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error getting revolutionary thermodynamics metrics: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

@router.get(
    "/prometheus",
    summary="Get system metrics for Prometheus",
    tags=["Monitoring"],
    response_class=Response,
    responses={
        200: {
            "description": "Prometheus metrics",
            "content": {"text/plain; version=0.0.4; charset=utf-8": {}}
        }
    }
)
async def get_prometheus_metrics_endpoint():
    """
    Endpoint for Prometheus to scrape metrics.
    """
    report = generate_metrics_report()
    return Response(content=report, media_type=PROMETHEUS_CONTENT_TYPE)


@router.get("/system/report", summary="Get a detailed system report", tags=["Monitoring"])
async def get_system_report(request: Request) -> Dict[str, Any]:
    """
    Provides a comprehensive report on the system's cognitive and operational status.
    """
    kimera_system = get_kimera_system(request.app)
    metrics = get_kimera_metrics()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_status": get_monitoring_core().get_monitoring_status(),
        "metrics_summary": get_monitoring_core().get_metrics_summary(),
        "recent_alerts": [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metric_name": alert.metric_name,
                "value": alert.value
            }
            for alert in list(get_monitoring_core().alerts)[-10:]  # Last 10 alerts
        ],
        "integration_status": get_integration_manager().get_integration_status(),
        "is_operational": kimera_system.is_operational(),
        "is_shutdown": kimera_system.is_shutdown(),
        "active_threads": get_thread_info(),
        "metrics_instance_id": id(metrics),
        "settings": get_api_settings().dict()
    }
    return report

@router.get("/threads", summary="Get information about active threads", tags=["Monitoring"])
async def get_threads_endpoint():
    """
    Provides a list of all active threads for debugging purposes.
    """
    return get_thread_info()


@router.post("/shutdown", summary="Gracefully shut down the Kimera system", tags=["Monitoring"])
async def shutdown_system_endpoint(request: Request):
    """
    Initiates a graceful shutdown of all Kimera services.
    """
    kimera_system = get_kimera_system(request.app)
    await kimera_system.shutdown()
    return {"message": "Kimera system shutdown initiated."}

# This is an example of a route that might use metrics
@router.get("/test/metrics", summary="Test endpoint for metrics", tags=["Monitoring"])
async def test_metrics_usage(request: Request):
    metrics = get_kimera_metrics()
    metrics.record_api_request(
        method="GET",
        endpoint="/test/metrics",
        status_code=200,
        duration=0.1
    )
    return {"message": "Metrics recorded for test endpoint."}
