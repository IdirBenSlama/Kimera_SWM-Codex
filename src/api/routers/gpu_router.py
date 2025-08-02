"""
KIMERA SWM - GPU API Router
===========================

FastAPI router for GPU acceleration system management and monitoring.
Provides endpoints for GPU status, performance metrics, and operations.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime, timezone

# GPU system imports
from src.core.gpu.gpu_manager import get_gpu_manager, is_gpu_available
from src.core.gpu.gpu_integration import get_gpu_integration_system, submit_gpu_task, GPUWorkloadType

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/status", tags=["GPU"], summary="Get GPU system status")
async def get_gpu_status() -> Dict[str, Any]:
    """Get comprehensive GPU system status and capabilities"""
    try:
        gpu_manager = get_gpu_manager()
        system_status = gpu_manager.get_system_status()
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu_system": system_status,
            "gpu_available": is_gpu_available()
        }
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        raise HTTPException(status_code=500, detail=f"GPU status error: {str(e)}")

@router.get("/performance", tags=["GPU"], summary="Get GPU performance metrics")
async def get_gpu_performance() -> Dict[str, Any]:
    """Get current GPU performance metrics and statistics"""
    try:
        integration_system = get_gpu_integration_system()
        performance = integration_system.get_performance_summary()
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": performance
        }
    except Exception as e:
        logger.error(f"Failed to get GPU performance: {e}")
        raise HTTPException(status_code=500, detail=f"GPU performance error: {str(e)}")

@router.post("/optimize", tags=["GPU"], summary="Optimize GPU performance")
async def optimize_gpu_performance() -> Dict[str, Any]:
    """Trigger GPU performance optimization"""
    try:
        integration_system = get_gpu_integration_system()
        optimization_result = await integration_system.optimize_performance()
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "optimization": optimization_result
        }
    except Exception as e:
        logger.error(f"Failed to optimize GPU: {e}")
        raise HTTPException(status_code=500, detail=f"GPU optimization error: {str(e)}")

@router.post("/clear-cache", tags=["GPU"], summary="Clear GPU memory cache")
async def clear_gpu_cache() -> Dict[str, Any]:
    """Clear GPU memory cache to free up memory"""
    try:
        gpu_manager = get_gpu_manager()
        gpu_manager.clear_cache()
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "GPU cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear GPU cache: {e}")
        raise HTTPException(status_code=500, detail=f"GPU cache clear error: {str(e)}")

@router.get("/devices", tags=["GPU"], summary="List available GPU devices")
async def list_gpu_devices() -> Dict[str, Any]:
    """List all available GPU devices with detailed information"""
    try:
        gpu_manager = get_gpu_manager()
        devices = []
        
        for device in gpu_manager.devices:
            devices.append(device.to_dict())
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_count": len(devices),
            "devices": devices,
            "current_device": gpu_manager.get_device_info()
        }
    except Exception as e:
        logger.error(f"Failed to list GPU devices: {e}")
        raise HTTPException(status_code=500, detail=f"GPU device listing error: {str(e)}")

@router.post("/switch-device/{device_id}", tags=["GPU"], summary="Switch to different GPU device")
async def switch_gpu_device(device_id: int) -> Dict[str, Any]:
    """Switch to a different GPU device"""
    try:
        gpu_manager = get_gpu_manager()
        success = gpu_manager.switch_device(device_id)
        
        if success:
            return {
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Switched to GPU device {device_id}",
                "current_device": gpu_manager.get_device_info()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to switch to device {device_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch GPU device: {e}")
        raise HTTPException(status_code=500, detail=f"GPU device switch error: {str(e)}")

@router.post("/submit-task", tags=["GPU"], summary="Submit GPU processing task")
async def submit_gpu_processing_task(
    workload_type: str,
    data: Dict[str, Any],
    priority: int = 5
) -> Dict[str, Any]:
    """Submit a task for GPU processing"""
    try:
        # Validate workload type
        try:
            gpu_workload_type = GPUWorkloadType(workload_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid workload type: {workload_type}")
        
        # Submit task
        task_id = await submit_gpu_task(gpu_workload_type, data, priority)
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "workload_type": workload_type,
            "priority": priority
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit GPU task: {e}")
        raise HTTPException(status_code=500, detail=f"GPU task submission error: {str(e)}")

@router.get("/workload-types", tags=["GPU"], summary="Get available GPU workload types")
async def get_gpu_workload_types() -> Dict[str, Any]:
    """Get list of available GPU workload types"""
    try:
        workload_types = [wt.value for wt in GPUWorkloadType]
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workload_types": workload_types,
            "descriptions": {
                "geoid_processing": "Process batches of geoids with GPU acceleration",
                "thermodynamic_evolution": "Evolve thermodynamic ensembles on GPU",
                "semantic_enhancement": "Enhance semantic representations using GPU",
                "cognitive_field": "Process cognitive field dynamics on GPU",
                "cryptographic": "GPU-accelerated cryptographic operations",
                "mixed_workload": "Mixed GPU workload optimization"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get workload types: {e}")
        raise HTTPException(status_code=500, detail=f"Workload types error: {str(e)}")

@router.get("/benchmarks", tags=["GPU"], summary="Run GPU performance benchmarks")
async def run_gpu_benchmarks() -> Dict[str, Any]:
    """Run comprehensive GPU performance benchmarks"""
    try:
        if not is_gpu_available():
            raise HTTPException(status_code=503, detail="GPU not available for benchmarking")
        
        # Import PyTorch for benchmarking
        import torch
        import time
        
        results = {}
        
        # Matrix multiplication benchmark
        sizes = [500, 1000, 2000]
        for size in sizes:
            # GPU benchmark
            a_gpu = torch.randn(size, size, device='cuda')
            b_gpu = torch.randn(size, size, device='cuda')
            
            # Warmup
            torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(5):
                result = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / 5
            
            # CPU comparison
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            
            start_time = time.time()
            result_cpu = torch.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            gflops = (2 * size**3) / gpu_time / 1e9
            
            results[f'matrix_{size}x{size}'] = {
                'gpu_time_ms': gpu_time * 1000,
                'cpu_time_ms': cpu_time * 1000,
                'speedup': speedup,
                'gflops': gflops
            }
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmarks": results,
            "gpu_info": get_gpu_manager().get_device_info()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run GPU benchmarks: {e}")
        raise HTTPException(status_code=500, detail=f"GPU benchmark error: {str(e)}")

@router.get("/health", tags=["GPU"], summary="GPU system health check")
async def gpu_health_check() -> Dict[str, Any]:
    """Perform comprehensive GPU system health check"""
    try:
        health_status = {
            "gpu_available": is_gpu_available(),
            "gpu_manager_status": "unknown",
            "integration_system_status": "unknown",
            "memory_available": 0,
            "temperature": 0,
            "utilization": 0,
            "errors": []
        }
        
        # Check GPU Manager
        try:
            gpu_manager = get_gpu_manager()
            system_status = gpu_manager.get_system_status()
            health_status["gpu_manager_status"] = system_status["status"]
            
            if system_status.get("current_device"):
                device_info = system_status["current_device"]
                health_status["memory_available"] = device_info.get("available_memory_gb", 0)
                health_status["temperature"] = device_info.get("temperature_celsius", 0)
                health_status["utilization"] = device_info.get("utilization_percent", 0)
                
        except Exception as e:
            health_status["errors"].append(f"GPU Manager error: {str(e)}")
        
        # Check Integration System
        try:
            integration_system = get_gpu_integration_system()
            performance = integration_system.get_performance_summary()
            health_status["integration_system_status"] = "operational"
            
        except Exception as e:
            health_status["errors"].append(f"Integration System error: {str(e)}")
            health_status["integration_system_status"] = "error"
        
        # Determine overall health
        if health_status["gpu_available"] and not health_status["errors"]:
            overall_health = "healthy"
        elif health_status["gpu_available"] and len(health_status["errors"]) <= 1:
            overall_health = "warning"
        else:
            overall_health = "critical"
        
        return {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": overall_health,
            "details": health_status
        }
        
    except Exception as e:
        logger.error(f"Failed GPU health check: {e}")
        raise HTTPException(status_code=500, detail=f"GPU health check error: {str(e)}")

# Add router description
router.tags = ["GPU Acceleration"]
router.prefix = "/gpu"
