"""
KIMERA Memory Management System
==============================

Advanced memory management for KIMERA's GPU-accelerated components.
Handles memory leaks, optimizes allocation, and provides monitoring.
"""

import gc
try:
    from utils.memory_optimizer import memory_optimizer
except ImportError:
    # Create placeholders for utils.memory_optimizer
        memory_optimizer = None

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import weakref

# Import dependency management
from .dependency_manager import is_feature_available, get_fallback

# Safe imports with fallback
torch = None
if is_feature_available("gpu_acceleration"):
    try:
        import torch
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        torch = None
else:
    TORCH_AVAILABLE = False
    torch = None

# System monitoring
psutil = None
if is_feature_available("monitoring"):
    try:
        import psutil
        PSUTIL_AVAILABLE = True
    except ImportError:
        PSUTIL_AVAILABLE = False
        psutil = get_fallback("psutil")
else:
    PSUTIL_AVAILABLE = False
    psutil = get_fallback("psutil")

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Memory types for tracking"""
    CPU = "cpu"
    GPU = "gpu"
    SHARED = "shared"

class MemoryPriority(Enum):
    """Memory allocation priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class MemoryStats:
    """Memory statistics snapshot"""
    total_memory: int
    used_memory: int
    available_memory: int
    cached_memory: int
    percentage_used: float
    memory_type: MemoryType
    timestamp: float

@dataclass
class MemoryLeak:
    """Memory leak detection info"""
    object_type: str
    size_mb: float
    location: str
    first_seen: float
    last_seen: float
    growth_rate: float

class MemoryManager:
    """Advanced memory management for KIMERA system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.memory_stats: Dict[MemoryType, List[MemoryStats]] = {
            MemoryType.CPU: [],
            MemoryType.GPU: [],
            MemoryType.SHARED: []
        }
        self.allocated_objects: Dict[str, weakref.ref] = {}
        self.memory_pools: Dict[str, List[Any]] = {}
        self.cleanup_callbacks: List[Callable] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.leak_detector = MemoryLeakDetector()
        
        # Memory thresholds
        self.memory_thresholds = {
            "cpu_critical": self.config.get("cpu_memory_critical", 0.9),
            "cpu_warning": self.config.get("cpu_memory_warning", 0.8),
            "gpu_critical": self.config.get("gpu_memory_critical", 0.9),
            "gpu_warning": self.config.get("gpu_memory_warning", 0.8)
        }
        
        # Initialize monitoring
        self.start_monitoring()
        
        logger.info("✅ KIMERA Memory Manager initialized")
    
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("📊 Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("🛑 Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect memory statistics
                self._collect_memory_stats()
                
                # Check for memory leaks
                self.leak_detector.check_for_leaks()
                
                # Trigger cleanup if needed
                self._check_memory_thresholds()
                
                # Sleep for monitoring interval
                time.sleep(self.config.get("monitoring_interval", 10))
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(5)  # Shorter sleep on error
    
    def _collect_memory_stats(self):
        """Collect current memory statistics"""
        current_time = time.time()
        
        # CPU Memory Stats
        if psutil:
            try:
                cpu_memory = psutil.virtual_memory()
                cpu_stats = MemoryStats(
                    total_memory=cpu_memory.total,
                    used_memory=cpu_memory.used,
                    available_memory=cpu_memory.available,
                    cached_memory=getattr(cpu_memory, 'cached', 0),
                    percentage_used=cpu_memory.percent,
                    memory_type=MemoryType.CPU,
                    timestamp=current_time
                )
                self.memory_stats[MemoryType.CPU].append(cpu_stats)
            except Exception as e:
                logger.error(f"Failed to collect CPU memory stats: {e}")
        
        # GPU Memory Stats
        if TORCH_AVAILABLE and torch and torch.cuda.is_available():
            try:
                for device_id in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.memory_stats(device_id)
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory
                    allocated_memory = torch.cuda.memory_allocated(device_id)
                    cached_memory = torch.cuda.memory_reserved(device_id)
                    
                    gpu_stats = MemoryStats(
                        total_memory=total_memory,
                        used_memory=allocated_memory,
                        available_memory=total_memory - allocated_memory,
                        cached_memory=cached_memory,
                        percentage_used=(allocated_memory / total_memory) * 100,
                        memory_type=MemoryType.GPU,
                        timestamp=current_time
                    )
                    self.memory_stats[MemoryType.GPU].append(gpu_stats)
            except Exception as e:
                logger.error(f"Failed to collect GPU memory stats: {e}")
        
        # Trim old stats (keep last 100 entries)
        for memory_type in self.memory_stats:
            if len(self.memory_stats[memory_type]) > 100:
                self.memory_stats[memory_type] = self.memory_stats[memory_type][-100:]
    
    def _check_memory_thresholds(self):
        """Check if memory usage exceeds thresholds and trigger cleanup"""
        
        # Check CPU memory
        if self.memory_stats[MemoryType.CPU]:
            latest_cpu = self.memory_stats[MemoryType.CPU][-1]
            cpu_usage = latest_cpu.percentage_used / 100
            
            if cpu_usage > self.memory_thresholds["cpu_critical"]:
                logger.warning(f"🚨 CPU memory critical: {cpu_usage:.1%}")
                self.emergency_cleanup()
            elif cpu_usage > self.memory_thresholds["cpu_warning"]:
                logger.warning(f"⚠️ CPU memory warning: {cpu_usage:.1%}")
                self.cleanup_unused_objects()
        
        # Check GPU memory
        if self.memory_stats[MemoryType.GPU]:
            latest_gpu = self.memory_stats[MemoryType.GPU][-1]
            gpu_usage = latest_gpu.percentage_used / 100
            
            if gpu_usage > self.memory_thresholds["gpu_critical"]:
                logger.warning(f"🚨 GPU memory critical: {gpu_usage:.1%}")
                self.emergency_gpu_cleanup()
            elif gpu_usage > self.memory_thresholds["gpu_warning"]:
                logger.warning(f"⚠️ GPU memory warning: {gpu_usage:.1%}")
                self.cleanup_gpu_cache()
    
    def allocate_tensor(self, shape: tuple, dtype=None, device=None, priority: MemoryPriority = MemoryPriority.NORMAL) -> Any:
        """Safely allocate tensor with memory management"""
        if not TORCH_AVAILABLE:
            logger.warning("Torch not available - returning None")
            return None
        
        try:
            # Determine device
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Check available memory before allocation
            if device.type == "cuda":
                available_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
                required_memory = self._estimate_tensor_memory(shape, dtype or torch.float32)
                
                if required_memory > available_memory:
                    logger.warning(f"Insufficient GPU memory. Required: {required_memory / 1024**2:.1f}MB, Available: {available_memory / 1024**2:.1f}MB")
                    # Try cleanup and retry
                    self.cleanup_gpu_cache()
                    available_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
                    
                    if required_memory > available_memory:
                        logger.error("Still insufficient memory after cleanup")
                        return None
            
            # Allocate tensor
            tensor = torch.zeros(shape, dtype=dtype, device=device)
            
            # Track allocation
            tensor_id = id(tensor)
            self.allocated_objects[str(tensor_id)] = weakref.ref(tensor)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to allocate tensor: {e}")
            return None
    
    def _estimate_tensor_memory(self, shape: tuple, dtype) -> int:
        """Estimate memory usage of tensor"""
        if not TORCH_AVAILABLE:
            return 0
        
        element_count = 1
        for dim in shape:
            element_count *= dim
        
        # Get bytes per element for dtype
        dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.int32: 4,
            torch.int64: 8,
            torch.bool: 1
        }
        
        bytes_per_element = dtype_sizes.get(dtype, 4)
        return element_count * bytes_per_element
    
    def cleanup_unused_objects(self):
        """Clean up unused objects and run garbage collection"""
        logger.info("🧹 Starting cleanup of unused objects")
        
        # Clean up dead references
        dead_refs = []
        for obj_id, ref in self.allocated_objects.items():
            if ref() is None:
                dead_refs.append(obj_id)
        
        for obj_id in dead_refs:
            del self.allocated_objects[obj_id]
        
        # Run garbage collection
        collected = gc.collect()
        logger.info(f"   Collected {collected} objects")
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback: {e}")
    
    def cleanup_gpu_cache(self):
        """Clean up GPU cache and unused tensors"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        logger.info("🧹 Cleaning up GPU cache")
        
        try:
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory stats after cleanup
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            logger.info(f"   GPU Memory - Allocated: {allocated / 1024**2:.1f}MB, Cached: {cached / 1024**2:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error during GPU cleanup: {e}")
    
    def emergency_cleanup(self):
        """Emergency cleanup when memory is critically low"""
        logger.warning("🚨 Emergency memory cleanup initiated")
        
        # 1. Clear all memory pools
        self.memory_pools.clear()
        
        # 2. Force garbage collection (multiple times)
        for _ in range(3):
            collected = gc.collect()
            logger.info(f"   Emergency GC collected {collected} objects")
        
        # 3. Clear GPU cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 4. Run all cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in emergency cleanup callback: {e}")
        
        logger.warning("🚨 Emergency cleanup completed")
    
    def emergency_gpu_cleanup(self):
        """Emergency GPU cleanup when GPU memory is critically low"""
        logger.warning("🚨 Emergency GPU cleanup initiated")
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        try:
            # Clear all GPU tensors that we're tracking
            gpu_objects = 0
            for obj_id, ref in list(self.allocated_objects.items()):
                obj = ref()
                if obj is not None and hasattr(obj, 'device') and obj.device.type == 'cuda':
                    del obj
                    gpu_objects += 1
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            
            logger.warning(f"🚨 Emergency GPU cleanup completed - cleared {gpu_objects} GPU objects")
            
        except Exception as e:
            logger.error(f"Error during emergency GPU cleanup: {e}")
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback"""
        self.cleanup_callbacks.append(callback)
    
    def get_context(self, pool_name: str = None) -> 'MemoryContext':
        """Get a memory context for resource management"""
        return MemoryContext(pool_name)
    
    def create_memory_pool(self, pool_name: str, max_size: int = 100):
        """Create a memory pool for object reuse"""
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = []
            logger.info(f"Created memory pool '{pool_name}' with max size {max_size}")
    
    def get_from_pool(self, pool_name: str) -> Any:
        """Get object from memory pool"""
        if pool_name in self.memory_pools and self.memory_pools[pool_name]:
            return self.memory_pools[pool_name].pop()
        return None
    
    def return_to_pool(self, pool_name: str, obj: Any):
        """Return object to memory pool"""
        if pool_name in self.memory_pools:
            max_size = self.config.get(f"pool_{pool_name}_max_size", 100)
            if len(self.memory_pools[pool_name]) < max_size:
                self.memory_pools[pool_name].append(obj)
            else:
                # Pool is full, let object be garbage collected
                del obj
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        report = {
            "timestamp": time.time(),
            "monitoring_active": self.monitoring_active,
            "tracked_objects": len(self.allocated_objects),
            "memory_pools": {name: len(pool) for name, pool in self.memory_pools.items()},
            "cleanup_callbacks": len(self.cleanup_callbacks)
        }
        
        # Add CPU memory stats
        if self.memory_stats[MemoryType.CPU]:
            latest_cpu = self.memory_stats[MemoryType.CPU][-1]
            report["cpu_memory"] = {
                "total_gb": latest_cpu.total_memory / 1024**3,
                "used_gb": latest_cpu.used_memory / 1024**3,
                "available_gb": latest_cpu.available_memory / 1024**3,
                "percentage_used": latest_cpu.percentage_used
            }
        
        # Add GPU memory stats
        if self.memory_stats[MemoryType.GPU]:
            latest_gpu = self.memory_stats[MemoryType.GPU][-1]
            report["gpu_memory"] = {
                "total_gb": latest_gpu.total_memory / 1024**3,
                "used_gb": latest_gpu.used_memory / 1024**3,
                "available_gb": latest_gpu.available_memory / 1024**3,
                "percentage_used": latest_gpu.percentage_used
            }
        
        # Add leak detection info
        report["memory_leaks"] = self.leak_detector.get_leak_summary()
        
        return report

class MemoryLeakDetector:
    """Detects memory leaks in the system"""
    
    def __init__(self):
        self.object_counts: Dict[str, int] = {}
        self.growth_history: Dict[str, List[tuple]] = {}
        self.detected_leaks: List[MemoryLeak] = []
        self.last_check = time.time()
    
    def check_for_leaks(self):
        """Check for potential memory leaks"""
        current_time = time.time()
        
        # Get current object counts
        current_counts = self._get_object_counts()
        
        # Compare with previous counts
        for obj_type, count in current_counts.items():
            if obj_type in self.object_counts:
                growth = count - self.object_counts[obj_type]
                
                # Track growth history
                if obj_type not in self.growth_history:
                    self.growth_history[obj_type] = []
                
                self.growth_history[obj_type].append((current_time, growth))
                
                # Keep only recent history (last 10 minutes)
                cutoff_time = current_time - 600
                self.growth_history[obj_type] = [
                    (time, growth) for time, growth in self.growth_history[obj_type]
                    if time > cutoff_time
                ]
                
                # Check for leak pattern
                if self._is_leak_pattern(obj_type):
                    self._record_leak(obj_type, count, current_time)
        
        # Update counts
        self.object_counts = current_counts
        self.last_check = current_time
    
    def _get_object_counts(self) -> Dict[str, int]:
        """Get current object counts by type"""
        counts = {}
        
        # Exclude common system types that are expected to grow
        excluded_types = {
            'frame', 'traceback', 'code', 'module', 'method', 'builtin_function_or_method',
            'wrapper_descriptor', 'getset_descriptor', 'method_descriptor'
        }
        
        # Count objects in gc
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            if obj_type not in excluded_types:
                counts[obj_type] = counts.get(obj_type, 0) + 1
        
        return counts
    
    def _is_leak_pattern(self, obj_type: str) -> bool:
        """Check if object type shows leak pattern"""
        if obj_type not in self.growth_history:
            return False
        
        history = self.growth_history[obj_type]
        if len(history) < 5:  # Need at least 5 data points
            return False
        
        # Check if object count is consistently growing
        positive_growth = sum(1 for _, growth in history[-5:] if growth > 0)
        total_growth = sum(growth for _, growth in history[-5:])
        
        # Enhanced leak detection: 80% positive growth AND significant total growth
        significant_growth = total_growth > 100  # More than 100 objects created
        return positive_growth >= 4 and significant_growth
    
    def _record_leak(self, obj_type: str, count: int, current_time: float):
        """Record a detected memory leak"""
        
        # Check if we already know about this leak
        existing_leak = None
        for leak in self.detected_leaks:
            if leak.object_type == obj_type:
                existing_leak = leak
                break
        
        if existing_leak:
            # Update existing leak
            existing_leak.last_seen = current_time
            existing_leak.size_mb = count * 0.001  # Rough estimate
        else:
            # Create new leak record
            leak = MemoryLeak(
                object_type=obj_type,
                size_mb=count * 0.001,  # Rough estimate
                location="Unknown",
                first_seen=current_time,
                last_seen=current_time,
                growth_rate=0.0
            )
            self.detected_leaks.append(leak)
            
            logger.warning(f"🚨 Memory leak detected: {obj_type} (count: {count})")
    
    def get_leak_summary(self) -> List[Dict[str, Any]]:
        """Get summary of detected leaks"""
        return [
            {
                "object_type": leak.object_type,
                "size_mb": leak.size_mb,
                "duration_minutes": (leak.last_seen - leak.first_seen) / 60,
                "growth_rate": leak.growth_rate
            }
            for leak in self.detected_leaks
        ]

# Global memory manager instance
memory_manager = MemoryManager()

# Context manager for memory management
class MemoryContext:
    """Optimized context manager for automatic memory cleanup"""
    
    def __init__(self, pool_name: str = None):
        self.pool_name = pool_name
        self.allocated_objects = []
        self.start_time = time.time()
        self.has_gpu_objects = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Fast cleanup - only if we have objects
        if self.allocated_objects:
            for obj in self.allocated_objects:
                try:
                    if hasattr(obj, 'device') and obj.device.type == 'cuda':
                        self.has_gpu_objects = True
                    del obj
                except Exception:
                    pass  # Silent cleanup for performance
            
            self.allocated_objects.clear()
            
            # Only run expensive operations if context was long-running or had GPU objects
            duration = time.time() - self.start_time
            if duration > 0.1 or self.has_gpu_objects:  # 100ms threshold
                gc.collect()
                
                # GPU cleanup only if we had GPU objects
                if self.has_gpu_objects and TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def allocate_tensor(self, shape: tuple, **kwargs) -> Any:
        """Allocate tensor that will be automatically cleaned up"""
        tensor = memory_manager.allocate_tensor(shape, **kwargs)
        if tensor is not None:
            self.allocated_objects.append(tensor)
            if hasattr(tensor, 'device') and tensor.device.type == 'cuda':
                self.has_gpu_objects = True
        return tensor

# Convenience functions
def cleanup_memory():
    """Clean up unused objects and GPU cache"""
    memory_manager.cleanup_unused_objects()
    memory_manager.cleanup_gpu_cache()

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics"""
    return memory_manager.get_memory_report()

def emergency_cleanup():
    """Emergency memory cleanup"""
    memory_manager.emergency_cleanup()

def register_cleanup_callback(callback: Callable):
    """Register a cleanup callback"""
    memory_manager.register_cleanup_callback(callback) 