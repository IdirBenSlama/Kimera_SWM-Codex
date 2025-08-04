"""
GPU Optimizer
=============
Optimizes GPU usage for Kimera cognitive engines.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import torch, but provide fallback if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU optimization disabled")


class GPUOptimizer:
    """Optimizes GPU usage across cognitive engines"""
    
    def __init__(self):
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.optimizations_applied = []
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Enable tensor cores
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            self.optimizations_applied.extend([
                "tensor_cores_enabled",
                "cudnn_benchmark_enabled",
                "memory_fraction_set"
            ])
            
            logger.info(f"GPU optimizations applied: {self.optimizations_applied}")
    
    def optimize_model(self, model) -> Any:
        """Optimize a PyTorch model for GPU execution"""
        if not TORCH_AVAILABLE:
            return model
            
        if not torch.cuda.is_available():
            return model
        
        # Move to GPU
        model = model.to(self.device)
        
        # Enable mixed precision if supported
        if hasattr(torch.cuda, 'amp'):
            model = model.half()  # Convert to FP16
            self.optimizations_applied.append(f"mixed_precision_{model.__class__.__name__}")
        
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
            self.optimizations_applied.append(f"compiled_{model.__class__.__name__}")
        
        return model
    
    def optimize_batch_processing(self, batch_size: int) -> int:
        """Optimize batch size based on available GPU memory"""
        if not TORCH_AVAILABLE:
            return batch_size
            
        if not torch.cuda.is_available():
            return batch_size
        
        # Get available memory
        free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB
        
        # Adjust batch size based on memory
        if free_memory > 6:
            return min(batch_size * 4, 256)
        elif free_memory > 4:
            return min(batch_size * 2, 128)
        else:
            return batch_size
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get GPU optimization statistics"""
        stats = {
            "device": str(self.device),
            "optimizations_applied": self.optimizations_applied,
            "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "tensor_cores_enabled": torch.backends.cuda.matmul.allow_tf32
            })
        
        return stats


# Global optimizer instance
gpu_optimizer = GPUOptimizer()
