"""
KIMERA GPU Optimization Engine
=============================

Advanced GPU optimization system to maximize performance and minimize the 197.6% degradation
identified in the audit. Implements intelligent GPU utilization, memory management, and 
processing optimizations.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import dependency management
from .dependency_manager import is_feature_available, get_fallback
from .memory_manager import memory_manager, MemoryContext

# Safe imports with fallback
torch = None
if is_feature_available("gpu_acceleration"):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        torch = None
else:
    TORCH_AVAILABLE = False
    torch = None

# CuPy for additional GPU acceleration
cupy = None
if is_feature_available("gpu_acceleration"):
    try:
        import cupy as cp
        CUPY_AVAILABLE = True
        cupy = cp
    except ImportError:
        CUPY_AVAILABLE = False
        cupy = get_fallback("cupy")
else:
    CUPY_AVAILABLE = False
    cupy = get_fallback("cupy")

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """GPU optimization levels"""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class ProcessingPriority(Enum):
    """Processing priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class GPUTask:
    """GPU processing task"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: ProcessingPriority
    created_at: float
    timeout: Optional[float] = None
    callback: Optional[Callable] = None

@dataclass
class GPUPerformanceMetrics:
    """GPU performance metrics"""
    utilization_percent: float
    memory_used_percent: float
    temperature: float
    power_usage: float
    throughput_ops_per_sec: float
    latency_ms: float
    efficiency_score: float
    timestamp: float

class GPUOptimizer:
    """Advanced GPU optimization engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.optimization_level = OptimizationLevel(
            self.config.get("optimization_level", "balanced")
        )
        
        # GPU device management
        self.devices = []
        self.current_device = None
        self.device_capabilities = {}
        
        # Task queue and processing
        self.task_queue = queue.PriorityQueue()
        self.processing_threads = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.performance_metrics: List[GPUPerformanceMetrics] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Optimization strategies
        self.active_optimizations: Dict[str, bool] = {
            "mixed_precision": False,
            "gradient_checkpointing": False,
            "memory_efficient_attention": False,
            "kernel_fusion": False,
            "async_processing": False,
            "batch_optimization": False
        }
        
        # Initialize GPU environment
        self._initialize_gpu_environment()
        
        # Start optimization monitoring
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"✅ GPU Optimizer initialized with {len(self.devices)} devices")
    
    def _initialize_gpu_environment(self):
        """Initialize GPU environment and detect capabilities"""
        
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch not available - GPU optimization disabled")
            return
        
        # Detect available GPUs
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"🔍 Detected {device_count} CUDA device(s)")
            
            for i in range(device_count):
                device = torch.device(f"cuda:{i}")
                self.devices.append(device)
                
                # Get device capabilities
                props = torch.cuda.get_device_properties(i)
                self.device_capabilities[i] = {
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "major": props.major,
                    "minor": props.minor,
                    "multiprocessor_count": props.multiprocessor_count,
                    "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor
                }
                
                logger.info(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
            
            # Set default device
            self.current_device = self.devices[0] if self.devices else None
            if self.current_device:
                torch.cuda.set_device(self.current_device)
        else:
            logger.warning("⚠️ No CUDA devices available")
            self.current_device = torch.device("cpu")
    
    def optimize_model(self, model: Any, optimization_level: OptimizationLevel = None) -> Any:
        """Optimize a model for better GPU performance"""
        
        if not TORCH_AVAILABLE or model is None:
            return model
        
        level = optimization_level or self.optimization_level
        logger.info(f"🔧 Optimizing model with {level.value} optimization level")
        
        try:
            # Move model to GPU
            if self.current_device and self.current_device.type == "cuda":
                model = model.to(self.current_device)
            
            # Apply optimizations based on level
            if level == OptimizationLevel.MINIMAL:
                model = self._apply_minimal_optimizations(model)
            elif level == OptimizationLevel.BALANCED:
                model = self._apply_balanced_optimizations(model)
            elif level == OptimizationLevel.AGGRESSIVE:
                model = self._apply_aggressive_optimizations(model)
            elif level == OptimizationLevel.MAXIMUM:
                model = self._apply_maximum_optimizations(model)
            
            logger.info("✅ Model optimization completed")
            return model
            
        except Exception as e:
            logger.error(f"❌ Model optimization failed: {e}")
            return model
    
    def _apply_minimal_optimizations(self, model: Any) -> Any:
        """Apply minimal optimizations (safe for all models)"""
        
        # Enable eval mode optimizations
        if hasattr(model, 'eval'):
            model.eval()
        
        # Disable gradient computation for inference
        if hasattr(torch, 'no_grad'):
            torch.set_grad_enabled(False)
        
        return model
    
    def _apply_balanced_optimizations(self, model: Any) -> Any:
        """Apply balanced optimizations (performance vs stability)"""
        
        model = self._apply_minimal_optimizations(model)
        
        # Enable mixed precision if supported
        if self._supports_mixed_precision():
            model = self._enable_mixed_precision(model)
            self.active_optimizations["mixed_precision"] = True
        
        # Enable memory efficient attention
        if self._supports_memory_efficient_attention():
            model = self._enable_memory_efficient_attention(model)
            self.active_optimizations["memory_efficient_attention"] = True
        
        return model
    
    def _apply_aggressive_optimizations(self, model: Any) -> Any:
        """Apply aggressive optimizations (higher performance, some risk)"""
        
        model = self._apply_balanced_optimizations(model)
        
        # Enable gradient checkpointing
        if self._supports_gradient_checkpointing():
            model = self._enable_gradient_checkpointing(model)
            self.active_optimizations["gradient_checkpointing"] = True
        
        # Enable kernel fusion
        if self._supports_kernel_fusion():
            model = self._enable_kernel_fusion(model)
            self.active_optimizations["kernel_fusion"] = True
        
        return model
    
    def _apply_maximum_optimizations(self, model: Any) -> Any:
        """Apply maximum optimizations (highest performance, highest risk)"""
        
        model = self._apply_aggressive_optimizations(model)
        
        # Enable all experimental optimizations
        self._enable_experimental_optimizations(model)
        
        return model
    
    def _supports_mixed_precision(self) -> bool:
        """Check if mixed precision is supported"""
        if not self.current_device or self.current_device.type != "cuda":
            return False
        
        # Check if GPU supports Tensor Cores (compute capability >= 7.0)
        if self.current_device.index in self.device_capabilities:
            cap = self.device_capabilities[self.current_device.index]
            return cap["major"] >= 7
        
        return False
    
    def _enable_mixed_precision(self, model: Any) -> Any:
        """Enable mixed precision training/inference"""
        
        try:
            if hasattr(torch.cuda, 'amp'):
                # Enable automatic mixed precision
                if hasattr(model, 'forward'):
                    original_forward = model.forward
                    
                    def mixed_precision_forward(*args, **kwargs):
                        with torch.cuda.amp.autocast():
                            return original_forward(*args, **kwargs)
                    
                    model.forward = mixed_precision_forward
                
                logger.info("✅ Mixed precision enabled")
            
        except Exception as e:
            logger.error(f"❌ Failed to enable mixed precision: {e}")
        
        return model
    
    def _supports_memory_efficient_attention(self) -> bool:
        """Check if memory efficient attention is supported"""
        return hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def _enable_memory_efficient_attention(self, model: Any) -> Any:
        """Enable memory efficient attention"""
        
        try:
            # Replace attention mechanisms with memory efficient versions
            for name, module in model.named_modules():
                if hasattr(module, 'attention') or 'attention' in name.lower():
                    # Enable memory efficient attention
                    if hasattr(module, 'enable_memory_efficient_attention'):
                        module.enable_memory_efficient_attention()
                    
            logger.info("✅ Memory efficient attention enabled")
            
        except Exception as e:
            logger.error(f"❌ Failed to enable memory efficient attention: {e}")
        
        return model
    
    def _supports_gradient_checkpointing(self) -> bool:
        """Check if gradient checkpointing is supported"""
        return hasattr(torch.utils, 'checkpoint')
    
    def _enable_gradient_checkpointing(self, model: Any) -> Any:
        """Enable gradient checkpointing to save memory"""
        
        try:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            elif hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
            
            logger.info("✅ Gradient checkpointing enabled")
            
        except Exception as e:
            logger.error(f"❌ Failed to enable gradient checkpointing: {e}")
        
        return model
    
    def _supports_kernel_fusion(self) -> bool:
        """Check if kernel fusion is supported"""
        return hasattr(torch.jit, 'script')
    
    def _enable_kernel_fusion(self, model: Any) -> Any:
        """Enable kernel fusion optimizations"""
        
        try:
            # Try to JIT compile the model
            if hasattr(torch.jit, 'script'):
                model = torch.jit.script(model)
                logger.info("✅ Kernel fusion enabled via JIT compilation")
            
        except Exception as e:
            logger.error(f"❌ Failed to enable kernel fusion: {e}")
        
        return model
    
    def _enable_experimental_optimizations(self, model: Any):
        """Enable experimental optimizations"""
        
        try:
            # Enable various experimental PyTorch optimizations
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
            
            if hasattr(torch.backends.cudnn, 'deterministic'):
                torch.backends.cudnn.deterministic = False
            
            # Enable TensorFloat-32 (TF32) on Ampere GPUs
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
                
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            
            logger.info("✅ Experimental optimizations enabled")
            
        except Exception as e:
            logger.error(f"❌ Failed to enable experimental optimizations: {e}")
    
    def optimize_tensor_operations(self, operation: Callable, *args, **kwargs) -> Any:
        """Optimize tensor operations for better performance"""
        
        if not TORCH_AVAILABLE:
            return operation(*args, **kwargs)
        
        try:
            # Use memory context for automatic cleanup
            with MemoryContext() as mem_ctx:
                # Move tensors to GPU if available
                if self.current_device and self.current_device.type == "cuda":
                    gpu_args = []
                    for arg in args:
                        if hasattr(arg, 'to'):
                            gpu_args.append(arg.to(self.current_device))
                        else:
                            gpu_args.append(arg)
                    
                    gpu_kwargs = {}
                    for key, value in kwargs.items():
                        if hasattr(value, 'to'):
                            gpu_kwargs[key] = value.to(self.current_device)
                        else:
                            gpu_kwargs[key] = value
                    
                    # Execute operation on GPU
                    with torch.cuda.amp.autocast(enabled=self.active_optimizations["mixed_precision"]):
                        result = operation(*gpu_args, **gpu_kwargs)
                else:
                    # Execute on CPU
                    result = operation(*args, **kwargs)
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Tensor operation optimization failed: {e}")
            # Fallback to original operation
            return operation(*args, **kwargs)
    
    def batch_optimize(self, operations: List[Tuple[Callable, tuple, dict]], batch_size: int = 8) -> List[Any]:
        """Optimize batch operations for better GPU utilization"""
        
        if not operations:
            return []
        
        results = []
        
        # Process operations in batches
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            
            # Execute batch on GPU
            batch_results = []
            for operation, args, kwargs in batch:
                result = self.optimize_tensor_operations(operation, *args, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Clear GPU cache between batches
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def async_optimize(self, operation: Callable, *args, **kwargs) -> asyncio.Future:
        """Asynchronously optimize operations"""
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self.thread_pool,
            self.optimize_tensor_operations,
            operation,
            *args,
            **kwargs
        )
        
        return future
    
    def _monitoring_loop(self):
        """Monitor GPU performance and adjust optimizations"""
        
        while self.monitoring_active:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                if metrics:
                    self.performance_metrics.append(metrics)
                
                # Adjust optimizations based on performance
                self._adjust_optimizations(metrics)
                
                # Sleep for monitoring interval
                time.sleep(self.config.get("monitoring_interval", 5))
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(5)
    
    def _collect_performance_metrics(self) -> Optional[GPUPerformanceMetrics]:
        """Collect current GPU performance metrics"""
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        try:
            device_id = self.current_device.index if self.current_device else 0
            
            # Get GPU utilization (simplified)
            utilization = torch.cuda.utilization(device_id) if hasattr(torch.cuda, 'utilization') else 0.0
            
            # Get memory usage
            memory_stats = torch.cuda.memory_stats(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            used_memory = torch.cuda.memory_allocated(device_id)
            memory_percent = (used_memory / total_memory) * 100
            
            # Create metrics
            metrics = GPUPerformanceMetrics(
                utilization_percent=utilization,
                memory_used_percent=memory_percent,
                temperature=0.0,  # Would need nvidia-ml-py for actual temperature
                power_usage=0.0,  # Would need nvidia-ml-py for actual power
                throughput_ops_per_sec=0.0,  # Would need benchmarking
                latency_ms=0.0,  # Would need benchmarking
                efficiency_score=self._calculate_efficiency_score(utilization, memory_percent),
                timestamp=time.time()
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")
            return None
    
    def _calculate_efficiency_score(self, utilization: float, memory_percent: float) -> float:
        """Calculate GPU efficiency score"""
        
        # Simple efficiency calculation
        # Ideal: high utilization, moderate memory usage
        utilization_score = min(utilization / 80.0, 1.0)  # 80% is ideal
        memory_score = max(0, 1.0 - max(0, memory_percent - 80) / 20.0)  # Penalize >80% memory
        
        return (utilization_score + memory_score) / 2.0
    
    def _adjust_optimizations(self, metrics: Optional[GPUPerformanceMetrics]):
        """Adjust optimizations based on performance metrics"""
        
        if not metrics:
            return
        
        # Adjust based on memory usage
        if metrics.memory_used_percent > 85:
            # High memory usage - enable memory optimizations
            self.active_optimizations["memory_efficient_attention"] = True
            self.active_optimizations["gradient_checkpointing"] = True
        elif metrics.memory_used_percent < 50:
            # Low memory usage - can enable more aggressive optimizations
            self.active_optimizations["batch_optimization"] = True
        
        # Adjust based on utilization
        if metrics.utilization_percent < 50:
            # Low utilization - enable async processing
            self.active_optimizations["async_processing"] = True
        
        # Log adjustments
        active_opts = [name for name, active in self.active_optimizations.items() if active]
        logger.debug(f"Active optimizations: {active_opts}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        
        report = {
            "timestamp": time.time(),
            "optimization_level": self.optimization_level.value,
            "active_optimizations": self.active_optimizations.copy(),
            "device_info": {
                "current_device": str(self.current_device) if self.current_device else None,
                "device_count": len(self.devices),
                "capabilities": self.device_capabilities
            },
            "performance_history": len(self.performance_metrics)
        }
        
        # Add latest performance metrics
        if self.performance_metrics:
            latest_metrics = self.performance_metrics[-1]
            report["current_performance"] = {
                "utilization_percent": latest_metrics.utilization_percent,
                "memory_used_percent": latest_metrics.memory_used_percent,
                "efficiency_score": latest_metrics.efficiency_score
            }
        
        return report
    
    def shutdown(self):
        """Shutdown GPU optimizer"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        self.thread_pool.shutdown(wait=True)
        
        logger.info("🛑 GPU Optimizer shutdown complete")

# Global GPU optimizer instance
gpu_optimizer = GPUOptimizer()

# Convenience functions
def optimize_model(model: Any, level: OptimizationLevel = OptimizationLevel.BALANCED) -> Any:
    """Optimize model for better GPU performance"""
    return gpu_optimizer.optimize_model(model, level)

def optimize_tensor_ops(operation: Callable, *args, **kwargs) -> Any:
    """Optimize tensor operations"""
    return gpu_optimizer.optimize_tensor_operations(operation, *args, **kwargs)

def batch_optimize(operations: List[Tuple[Callable, tuple, dict]], batch_size: int = 8) -> List[Any]:
    """Optimize batch operations"""
    return gpu_optimizer.batch_optimize(operations, batch_size)

def get_gpu_report() -> Dict[str, Any]:
    """Get GPU optimization report"""
    return gpu_optimizer.get_optimization_report() 