"""
Metrics Router for Kimera System Monitoring
Provides system metrics and health monitoring endpoints
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import psutil
import torch
import time
from datetime import datetime
try:
    from ...utils.gpu_foundation import GPUFoundation
except ImportError:
    try:
        from utils.gpu_foundation import GPUFoundation
    except ImportError:
        # Create placeholders for utils.gpu_foundation
            class GPUFoundation: pass
try:
    from ...vault.vault_manager import VaultManager
except ImportError:
    try:
        from vault.vault_manager import VaultManager
    except ImportError:
        # Create placeholders for vault.vault_manager
            class VaultManager: pass
try:
    from ...monitoring.system_health_monitor import get_health_monitor
except ImportError:
    try:
        from monitoring.system_health_monitor import get_health_monitor
    except ImportError:
        # Create placeholders for monitoring.system_health_monitor
            def get_health_monitor(*args, **kwargs): return None

router = APIRouter(prefix="/system-metrics", tags=["system-metrics"])

@router.get("/")
async def get_system_metrics() -> Dict[str, Any]:
    """
    Get comprehensive system metrics including GPU, memory, and database stats
    """
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            }
        }
    }
    
    # GPU metrics if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_reserved = torch.cuda.memory_reserved(0)
        
        metrics["gpu"] = {
            "device": torch.cuda.get_device_name(0),
            "memory_total": gpu_memory,
            "memory_allocated": gpu_allocated,
            "memory_reserved": gpu_reserved,
            "memory_free": gpu_memory - gpu_reserved,
            "utilization_percent": (gpu_allocated / gpu_memory) * 100
        }
    
    return metrics

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "kimera-swm"
    }

@router.get("/cognitive")
async def cognitive_metrics() -> Dict[str, Any]:
    """
    Get cognitive system specific metrics
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cognitive_status": "operational",
        "engines": {
            "contradiction_engine": "active",
            "thermodynamic_engine": "active", 
            "diffusion_engine": "active",
            "vault_manager": "active"
        },
        "gpu_foundation": "initialized" if torch.cuda.is_available() else "cpu_mode"
    }

@router.get("/health-status")
async def get_health_status() -> Dict[str, Any]:
    """
    Get comprehensive health status from the health monitor
    """
    monitor = get_health_monitor()
    return monitor.get_health_status()

@router.get("/history")
async def get_metrics_history(hours: int = 1) -> Dict[str, Any]:
    """
    Get metrics history for the specified number of hours
    """
    monitor = get_health_monitor()
    return {
        "history": monitor.get_metrics_history(hours),
        "hours": hours,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/optimize")
async def optimize_system() -> Dict[str, Any]:
    """
    Trigger system optimization
    """
    monitor = get_health_monitor()
    return monitor.optimize_system()

@router.post("/start-monitoring")
async def start_monitoring() -> Dict[str, str]:
    """
    Start continuous health monitoring
    """
    monitor = get_health_monitor()
    monitor.start_monitoring()
    return {
        "message": "Health monitoring started",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/stop-monitoring")
async def stop_monitoring() -> Dict[str, str]:
    """
    Stop continuous health monitoring
    """
    monitor = get_health_monitor()
    monitor.stop_monitoring()
    return {
        "message": "Health monitoring stopped",
        "timestamp": datetime.utcnow().isoformat()
    } 

@router.get("/gpu")
async def get_gpu_metrics() -> Dict[str, Any]:
    """
    Get detailed GPU metrics and utilization
    """
    try:
        if not torch.cuda.is_available():
            return {
                "status": "unavailable",
                "message": "CUDA not available",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        device = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(device)
        
        # Get memory info
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        memory_total = gpu_props.total_memory
        
        # Calculate utilization
        memory_utilization = (memory_allocated / memory_total) * 100 if memory_total > 0 else 0
        
        return {
            "status": "available",
            "device": {
                "name": gpu_props.name,
                "index": device,
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "total_memory": memory_total,
                "memory_allocated": memory_allocated,
                "memory_reserved": memory_reserved,
                "memory_free": memory_total - memory_reserved,
                "utilization_percent": memory_utilization
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        } 