"""
KIMERA SWM - GPU INTEGRATION SYSTEM
===================================

Comprehensive GPU integration and orchestration system that coordinates
all GPU-accelerated components in Kimera SWM. Provides unified GPU
resource management, performance optimization, and monitoring.

Features:
- Unified GPU resource management
- GPU-accelerated engine coordination
- Performance monitoring and optimization
- Automatic load balancing
- Memory management and optimization
- Real-time GPU metrics and alerts
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

# Core Kimera imports
from src.core.data_structures.geoid_state import GeoidState
from src.core.gpu.gpu_manager import get_gpu_manager, is_gpu_available
from src.engines.gpu.gpu_geoid_processor import get_gpu_geoid_processor
from src.engines.gpu.gpu_thermodynamic_engine import get_gpu_thermodynamic_engine

logger = logging.getLogger(__name__)

class GPUWorkloadType(Enum):
    """Types of GPU workloads"""
    GEOID_PROCESSING = "geoid_processing"
    THERMODYNAMIC_EVOLUTION = "thermodynamic_evolution"
    SEMANTIC_ENHANCEMENT = "semantic_enhancement"
    COGNITIVE_FIELD = "cognitive_field"
    CRYPTOGRAPHIC = "cryptographic"
    MIXED_WORKLOAD = "mixed_workload"

@dataclass
class GPUTask:
    """GPU computation task"""
    task_id: str
    workload_type: GPUWorkloadType
    priority: int  # 1-10, 10 is highest
    data: Any
    callback: Optional[Callable] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

@dataclass
class GPUPerformanceMetrics:
    """Comprehensive GPU performance metrics"""
    gpu_utilization: float
    memory_utilization: float
    temperature: float
    power_usage: float
    compute_throughput: float
    memory_bandwidth: float
    tasks_completed: int
    average_task_time: float
    queue_length: int
    error_rate: float
    timestamp: float = field(default_factory=time.time)

class GPUIntegrationSystem:
    """Unified GPU integration and orchestration system"""
    
    def __init__(self, max_concurrent_tasks: int = 8, monitoring_interval: float = 1.0):
        """Initialize GPU integration system
        
        Args:
            max_concurrent_tasks: Maximum concurrent GPU tasks
            monitoring_interval: GPU monitoring interval in seconds
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.monitoring_interval = monitoring_interval
        
        # Initialize GPU manager and engines
        self.gpu_manager = get_gpu_manager()
        self.gpu_available = is_gpu_available()
        
        # GPU engines
        self.geoid_processor = None
        self.thermodynamic_engine = None
        
        # Task management
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, GPUTask] = {}
        self.completed_tasks: List[GPUTask] = []
        self.task_executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Performance monitoring
        self.performance_history: List[GPUPerformanceMetrics] = []
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Statistics
        self.stats = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_gpu_time': 0.0,
            'total_cpu_fallback': 0,
            'peak_gpu_utilization': 0.0,
            'peak_memory_usage': 0.0,
            'average_throughput': 0.0
        }
        
        # Load balancing
        self.workload_weights = {
            GPUWorkloadType.GEOID_PROCESSING: 1.0,
            GPUWorkloadType.THERMODYNAMIC_EVOLUTION: 2.0,
            GPUWorkloadType.SEMANTIC_ENHANCEMENT: 0.8,
            GPUWorkloadType.COGNITIVE_FIELD: 1.5,
            GPUWorkloadType.CRYPTOGRAPHIC: 0.5,
            GPUWorkloadType.MIXED_WORKLOAD: 1.2
        }
        
        # Initialize engines
        self._initialize_gpu_engines()
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info(f"🚀 GPU Integration System initialized")
        logger.info(f"   GPU Available: {self.gpu_available}")
        logger.info(f"   Max Concurrent Tasks: {self.max_concurrent_tasks}")
        logger.info(f"   Monitoring Interval: {self.monitoring_interval}s")
    
    def _initialize_gpu_engines(self) -> None:
        """Initialize GPU-accelerated engines"""
        try:
            if self.gpu_available:
                # Initialize geoid processor
                self.geoid_processor = get_gpu_geoid_processor()
                logger.info("✅ GPU Geoid Processor initialized")
                
                # Initialize thermodynamic engine
                self.thermodynamic_engine = get_gpu_thermodynamic_engine()
                logger.info("✅ GPU Thermodynamic Engine initialized")
                
                # Initialize other GPU engines as needed
                self._initialize_additional_engines()
                
            else:
                logger.info("📱 GPU not available - CPU fallback mode")
                
        except Exception as e:
            logger.error(f"❌ GPU engine initialization failed: {e}")
            self.gpu_available = False
    
    def _initialize_additional_engines(self) -> None:
        """Initialize additional GPU engines"""
        try:
            # Import and initialize GPU cryptographic engine if available
            from engines.gpu_cryptographic_engine import GPUCryptographicEngine
            self.crypto_engine = GPUCryptographicEngine()
            logger.info("✅ GPU Cryptographic Engine initialized")
            
        except ImportError:
            logger.info("ℹ️ GPU Cryptographic Engine not available")
        except Exception as e:
            logger.warning(f"⚠️ GPU Cryptographic Engine initialization failed: {e}")
    
    def _start_monitoring(self) -> None:
        """Start GPU performance monitoring"""
        if not self.gpu_available:
            return
        
        async def monitoring_loop():
            while self.monitoring_active:
                try:
                    metrics = await self._collect_gpu_metrics()
                    self.performance_history.append(metrics)
                    
                    # Keep only recent history (last hour at 1-second intervals)
                    if len(self.performance_history) > 3600:
                        self.performance_history = self.performance_history[-3600:]
                    
                    # Update peak statistics
                    self.stats['peak_gpu_utilization'] = max(
                        self.stats['peak_gpu_utilization'], 
                        metrics.gpu_utilization
                    )
                    self.stats['peak_memory_usage'] = max(
                        self.stats['peak_memory_usage'],
                        metrics.memory_utilization
                    )
                    
                    await asyncio.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"❌ GPU monitoring error: {e}")
                    await asyncio.sleep(self.monitoring_interval * 2)
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(monitoring_loop())
        logger.info("📊 GPU monitoring started")
    
    async def _collect_gpu_metrics(self) -> GPUPerformanceMetrics:
        """Collect comprehensive GPU performance metrics"""
        if not self.gpu_available:
            return GPUPerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        try:
            # Get GPU utilization
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_manager.current_device.device_id)
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            memory_util = utilization.memory
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power usage
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            
            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_percent = (memory_info.used / memory_info.total) * 100
            
            # Task queue metrics
            queue_length = self.task_queue.qsize()
            
            # Calculate throughput and error rate
            completed_count = len(self.completed_tasks)
            error_count = sum(1 for task in self.completed_tasks if not hasattr(task, 'success') or not task.success)
            error_rate = error_count / completed_count if completed_count > 0 else 0
            
            avg_task_time = 0
            if self.completed_tasks:
                valid_times = [t.execution_time for t in self.completed_tasks if t.execution_time]
                avg_task_time = sum(valid_times) / len(valid_times) if valid_times else 0
            
            # Compute throughput (tasks per second)
            if avg_task_time > 0:
                throughput = 1.0 / avg_task_time
            else:
                throughput = 0
            
            return GPUPerformanceMetrics(
                gpu_utilization=gpu_util,
                memory_utilization=memory_used_percent,
                temperature=temp,
                power_usage=power,
                compute_throughput=throughput,
                memory_bandwidth=0,  # Would need more complex calculation
                tasks_completed=completed_count,
                average_task_time=avg_task_time,
                queue_length=queue_length,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to collect GPU metrics: {e}")
            return GPUPerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    async def submit_task(self, workload_type: GPUWorkloadType, data: Any,
                         priority: int = 5, callback: Optional[Callable] = None) -> str:
        """Submit a task for GPU processing
        
        Args:
            workload_type: Type of GPU workload
            data: Task data
            priority: Task priority (1-10, 10 is highest)
            callback: Optional callback function
            
        Returns:
            Task ID
        """
        task_id = f"{workload_type.value}_{int(time.time() * 1000000)}"
        
        task = GPUTask(
            task_id=task_id,
            workload_type=workload_type,
            priority=priority,
            data=data,
            callback=callback
        )
        
        # Add to queue with priority (lower number = higher priority in asyncio.PriorityQueue)
        await self.task_queue.put((10 - priority, task))
        
        self.stats['total_tasks_submitted'] += 1
        
        logger.debug(f"📋 Submitted {workload_type.value} task: {task_id}")
        
        # Start task processing if not running
        asyncio.create_task(self._process_tasks())
        
        return task_id
    
    async def _process_tasks(self) -> None:
        """Process tasks from the queue"""
        while not self.task_queue.empty() and len(self.active_tasks) < self.max_concurrent_tasks:
            try:
                priority, task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
                
                if task.task_id not in self.active_tasks:
                    self.active_tasks[task.task_id] = task
                    task.started_at = time.time()
                    
                    # Process task based on workload type
                    asyncio.create_task(self._execute_task(task))
                    
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"❌ Task processing error: {e}")
    
    async def _execute_task(self, task: GPUTask) -> None:
        """Execute a GPU task"""
        try:
            result = None
            
            if task.workload_type == GPUWorkloadType.GEOID_PROCESSING:
                result = await self._process_geoid_task(task)
            elif task.workload_type == GPUWorkloadType.THERMODYNAMIC_EVOLUTION:
                result = await self._process_thermodynamic_task(task)
            elif task.workload_type == GPUWorkloadType.SEMANTIC_ENHANCEMENT:
                result = await self._process_semantic_task(task)
            elif task.workload_type == GPUWorkloadType.CRYPTOGRAPHIC:
                result = await self._process_cryptographic_task(task)
            else:
                logger.warning(f"⚠️ Unknown workload type: {task.workload_type}")
                result = {"error": "Unknown workload type"}
            
            # Mark task as completed
            task.completed_at = time.time()
            task.success = result.get('success', True) if isinstance(result, dict) else True
            
            # Update statistics
            self.stats['total_tasks_completed'] += 1
            if task.execution_time:
                self.stats['total_gpu_time'] += task.execution_time
            
            # Execute callback if provided
            if task.callback:
                try:
                    await task.callback(task, result)
                except Exception as e:
                    logger.error(f"❌ Task callback failed: {e}")
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            
            # Keep only recent completed tasks
            if len(self.completed_tasks) > 1000:
                self.completed_tasks = self.completed_tasks[-1000:]
            
            logger.debug(f"✅ Completed task {task.task_id} in {task.execution_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Task execution failed {task.task_id}: {e}")
            task.completed_at = time.time()
            task.success = False
            
        finally:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _process_geoid_task(self, task: GPUTask) -> Dict[str, Any]:
        """Process geoid-related GPU task"""
        if not self.geoid_processor:
            return {"error": "Geoid processor not available"}
        
        data = task.data
        operation = data.get('operation', 'semantic_enhancement')
        geoids = data.get('geoids', [])
        parameters = data.get('parameters', {})
        
        results = await self.geoid_processor.process_geoid_batch(geoids, operation, parameters)
        
        return {
            'success': True,
            'results': results,
            'processed_count': len(results)
        }
    
    async def _process_thermodynamic_task(self, task: GPUTask) -> Dict[str, Any]:
        """Process thermodynamic evolution GPU task"""
        if not self.thermodynamic_engine:
            return {"error": "Thermodynamic engine not available"}
        
        data = task.data
        ensemble = data.get('ensemble')
        parameters = data.get('parameters')
        
        evolved_geoids, evolution_data = await self.thermodynamic_engine.evolve_ensemble(ensemble, parameters)
        
        return {
            'success': True,
            'evolved_geoids': evolved_geoids,
            'evolution_data': evolution_data
        }
    
    async def _process_semantic_task(self, task: GPUTask) -> Dict[str, Any]:
        """Process semantic enhancement GPU task"""
        # Delegate to geoid processor for semantic tasks
        return await self._process_geoid_task(task)
    
    async def _process_cryptographic_task(self, task: GPUTask) -> Dict[str, Any]:
        """Process cryptographic GPU task"""
        if not hasattr(self, 'crypto_engine') or not self.crypto_engine:
            return {"error": "Cryptographic engine not available"}
        
        # This would implement specific cryptographic operations
        # For now, return placeholder
        return {
            'success': True,
            'message': 'Cryptographic operation completed'
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        recent_metrics = self.performance_history[-10:] if self.performance_history else []
        
        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        avg_memory_util = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        avg_temp = sum(m.temperature for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        return {
            'gpu_status': {
                'available': self.gpu_available,
                'current_device': self.gpu_manager.get_device_info(),
                'average_utilization': avg_gpu_util,
                'average_memory_utilization': avg_memory_util,
                'average_temperature': avg_temp
            },
            'task_statistics': {
                'total_submitted': self.stats['total_tasks_submitted'],
                'total_completed': self.stats['total_tasks_completed'],
                'active_tasks': len(self.active_tasks),
                'queue_length': self.task_queue.qsize(),
                'completion_rate': (
                    self.stats['total_tasks_completed'] / self.stats['total_tasks_submitted']
                    if self.stats['total_tasks_submitted'] > 0 else 0
                )
            },
            'performance_stats': {
                'total_gpu_time': self.stats['total_gpu_time'],
                'peak_gpu_utilization': self.stats['peak_gpu_utilization'],
                'peak_memory_usage': self.stats['peak_memory_usage'],
                'average_task_time': (
                    self.stats['total_gpu_time'] / self.stats['total_tasks_completed']
                    if self.stats['total_tasks_completed'] > 0 else 0
                )
            },
            'engines_status': {
                'geoid_processor': self.geoid_processor is not None,
                'thermodynamic_engine': self.thermodynamic_engine is not None,
                'cryptographic_engine': hasattr(self, 'crypto_engine') and self.crypto_engine is not None
            }
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimize GPU performance based on current workload"""
        if not self.gpu_available:
            return {"message": "GPU not available for optimization"}
        
        optimization_actions = []
        
        # Analyze recent performance
        recent_metrics = self.performance_history[-60:] if self.performance_history else []
        
        if recent_metrics:
            avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
            avg_memory_util = sum(m.memory_utilization for m in recent_metrics) / len(recent_metrics)
            avg_temp = sum(m.temperature for m in recent_metrics) / len(recent_metrics)
            
            # GPU utilization optimization
            if avg_gpu_util < 50:
                optimization_actions.append("Increase batch sizes for better GPU utilization")
            elif avg_gpu_util > 95:
                optimization_actions.append("Reduce batch sizes to prevent GPU overload")
            
            # Memory optimization
            if avg_memory_util > 90:
                optimization_actions.append("Clear GPU cache and reduce memory usage")
                self.gpu_manager.clear_cache()
            
            # Temperature management
            if avg_temp > 80:
                optimization_actions.append("Reduce processing intensity due to high temperature")
            
        # Queue optimization
        if self.task_queue.qsize() > self.max_concurrent_tasks * 2:
            optimization_actions.append("Increase concurrent task limit or add more workers")
        
        return {
            'optimization_performed': True,
            'actions_taken': optimization_actions,
            'current_performance': {
                'gpu_utilization': avg_gpu_util if recent_metrics else 0,
                'memory_utilization': avg_memory_util if recent_metrics else 0,
                'temperature': avg_temp if recent_metrics else 0
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown GPU integration system gracefully"""
        logger.info("🔄 Shutting down GPU Integration System...")
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Wait for active tasks to complete (with timeout)
        timeout = 30  # 30 seconds timeout
        start_time = time.time()
        
        while self.active_tasks and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        # Force shutdown remaining tasks
        if self.active_tasks:
            logger.warning(f"⚠️ Force shutting down {len(self.active_tasks)} remaining tasks")
        
        # Shutdown engines
        if self.geoid_processor:
            await self.geoid_processor.shutdown()
        
        # Clear GPU cache
        if self.gpu_available:
            self.gpu_manager.clear_cache()
        
        # Final statistics
        summary = self.get_performance_summary()
        logger.info(f"📊 Final GPU Stats: {summary['task_statistics']['total_completed']} tasks completed")
        logger.info(f"   Total GPU Time: {summary['performance_stats']['total_gpu_time']:.1f}s")
        logger.info(f"   Peak GPU Utilization: {summary['performance_stats']['peak_gpu_utilization']:.1f}%")
        
        logger.info("✅ GPU Integration System shutdown complete")


# Global GPU integration system
_gpu_integration_system = None

def get_gpu_integration_system() -> GPUIntegrationSystem:
    """Get the global GPU integration system instance"""
    global _gpu_integration_system
    if _gpu_integration_system is None:
        _gpu_integration_system = GPUIntegrationSystem()
    return _gpu_integration_system

async def submit_gpu_task(workload_type: GPUWorkloadType, data: Any, priority: int = 5) -> str:
    """Submit a task to the GPU integration system"""
    system = get_gpu_integration_system()
    return await system.submit_task(workload_type, data, priority)

def get_gpu_performance() -> Dict[str, Any]:
    """Get current GPU performance summary"""
    system = get_gpu_integration_system()
    return system.get_performance_summary() 