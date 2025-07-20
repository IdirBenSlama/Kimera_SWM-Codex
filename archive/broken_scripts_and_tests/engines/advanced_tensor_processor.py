"""
Advanced Tensor Processor for KIMERA Cognitive Architecture
==========================================================

Revolutionary tensor processing system providing comprehensive validation,
automatic correction, and enhanced safety for cognitive field operations.

Features:
- Multi-dimensional tensor validation
- Automatic shape correction with safety bounds
- Memory-efficient tensor operations
- GPU-optimized processing
- Cognitive field compatibility checking
- Quantum state tensor preparation
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class TensorType(Enum):
    """Supported tensor types for validation"""
    EMBEDDING = "embedding"
    COGNITIVE_FIELD = "cognitive_field"
    QUANTUM_STATE = "quantum_state"
    SEMANTIC_VECTOR = "semantic_vector"
    ATTENTION_WEIGHTS = "attention_weights"

@dataclass
class TensorValidationResult:
    """Result of tensor validation operation"""
    original_shape: Tuple[int, ...]
    corrected_shape: Tuple[int, ...]
    validation_passed: bool
    corrections_applied: List[str]
    safety_warnings: List[str]
    processing_time_ms: float
    memory_usage_mb: float

class AdvancedTensorProcessor:
    """
    Advanced tensor processing engine for KIMERA cognitive architecture.
    
    Provides comprehensive tensor validation, automatic correction, and
    enhanced safety mechanisms for cognitive field operations.
    """
    
    def __init__(self, 
                 device: Optional[torch.device] = None,
                 safety_bounds: Tuple[float, float] = (-100.0, 100.0),
                 memory_limit_gb: float = 16.0):
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.safety_bounds = safety_bounds
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        
        # Tensor validation rules
        self.shape_validators = {
            TensorType.EMBEDDING: self._validate_embedding_shape,
            TensorType.COGNITIVE_FIELD: self._validate_cognitive_field_shape,
            TensorType.QUANTUM_STATE: self._validate_quantum_state_shape,
            TensorType.SEMANTIC_VECTOR: self._validate_semantic_vector_shape,
            TensorType.ATTENTION_WEIGHTS: self._validate_attention_weights_shape
        }
        
        # Safety parameters
        self.max_tensor_size = 1024 * 1024 * 1024  # 1B elements
        self.min_tensor_size = 1
        
        logger.info(f"🔧 Advanced Tensor Processor initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Safety bounds: {self.safety_bounds}")
        logger.info(f"   Memory limit: {memory_limit_gb}GB")
    
    def validate_and_correct_tensor(self, 
                                  tensor: torch.Tensor, 
                                  tensor_type: TensorType,
                                  target_shape: Optional[Tuple[int, ...]] = None) -> Tuple[torch.Tensor, TensorValidationResult]:
        """
        Comprehensive tensor validation and automatic correction.
        
        Args:
            tensor: Input tensor to validate and correct
            tensor_type: Type of tensor for specific validation rules
            target_shape: Optional target shape for correction
            
        Returns:
            Tuple of (corrected_tensor, validation_result)
        """
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if start_time:
            start_time.record()
        
        import time
        cpu_start = time.time()
        
        original_shape = tuple(tensor.shape)
        corrections_applied = []
        safety_warnings = []
        
        try:
            # Step 1: Basic safety validation
            corrected_tensor = self._apply_basic_safety_checks(tensor, corrections_applied, safety_warnings)
            
            # Step 2: Type-specific validation
            validator = self.shape_validators.get(tensor_type)
            if validator:
                corrected_tensor = validator(corrected_tensor, target_shape, corrections_applied, safety_warnings)
            else:
                safety_warnings.append(f"No specific validator for tensor type: {tensor_type}")
            
            # Step 3: Memory optimization
            corrected_tensor = self._optimize_memory_layout(corrected_tensor, corrections_applied)
            
            # Step 4: Final safety bounds check
            corrected_tensor = self._apply_safety_bounds(corrected_tensor, corrections_applied, safety_warnings)
            
            # Calculate timing and memory usage
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time_ms = start_time.elapsed_time(end_time)
            else:
                processing_time_ms = (time.time() - cpu_start) * 1000
            
            memory_usage_mb = corrected_tensor.element_size() * corrected_tensor.numel() / (1024 * 1024)
            
            # Create validation result
            result = TensorValidationResult(
                original_shape=original_shape,
                corrected_shape=tuple(corrected_tensor.shape),
                validation_passed=len(safety_warnings) == 0,
                corrections_applied=corrections_applied,
                safety_warnings=safety_warnings,
                processing_time_ms=processing_time_ms,
                memory_usage_mb=memory_usage_mb
            )
            
            return corrected_tensor, result
            
        except Exception as e:
            logger.error(f"❌ Tensor validation failed: {e}")
            # Return original tensor with error result
            result = TensorValidationResult(
                original_shape=original_shape,
                corrected_shape=original_shape,
                validation_passed=False,
                corrections_applied=[],
                safety_warnings=[f"Validation failed: {str(e)}"],
                processing_time_ms=0.0,
                memory_usage_mb=0.0
            )
            return tensor, result
    
    def _apply_basic_safety_checks(self, 
                                 tensor: torch.Tensor, 
                                 corrections: List[str], 
                                 warnings: List[str]) -> torch.Tensor:
        """Apply basic safety checks to tensor"""
        
        # Check for NaN or Inf values
        if torch.isnan(tensor).any():
            warnings.append("NaN values detected in tensor")
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
            corrections.append("Replaced NaN values with zeros")
        
        if torch.isinf(tensor).any():
            warnings.append("Infinite values detected in tensor")
            tensor = torch.where(torch.isinf(tensor), torch.sign(tensor) * 100.0, tensor)
            corrections.append("Clipped infinite values to ±100")
        
        # Check tensor size
        total_elements = tensor.numel()
        if total_elements > self.max_tensor_size:
            warnings.append(f"Tensor too large: {total_elements} elements")
            # Flatten and truncate if too large
            tensor = tensor.flatten()[:self.max_tensor_size]
            corrections.append(f"Truncated tensor to {self.max_tensor_size} elements")
        
        if total_elements < self.min_tensor_size:
            warnings.append(f"Tensor too small: {total_elements} elements")
        
        # Ensure tensor is on correct device
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
            corrections.append(f"Moved tensor to {self.device}")
        
        return tensor
    
    def _validate_embedding_shape(self, 
                                tensor: torch.Tensor, 
                                target_shape: Optional[Tuple[int, ...]], 
                                corrections: List[str], 
                                warnings: List[str]) -> torch.Tensor:
        """Validate and correct embedding tensor shape"""
        
        if tensor.dim() == 0:
            warnings.append("Scalar tensor not suitable for embeddings")
            # Convert scalar to 1D tensor
            tensor = tensor.unsqueeze(0)
            corrections.append("Converted scalar to 1D tensor")
        
        elif tensor.dim() == 1:
            # Perfect for embeddings
            pass
            
        elif tensor.dim() == 2:
            if tensor.shape[0] == 1:
                # Batch dimension of 1 - can be squeezed
                tensor = tensor.squeeze(0)
                corrections.append("Removed singleton batch dimension")
            else:
                # Multiple batches - flatten to 1D
                original_shape = tensor.shape
                tensor = tensor.flatten()
                corrections.append(f"Flattened {original_shape} to 1D for embedding")
                warnings.append("Multi-batch tensor flattened - information may be lost")
        
        else:
            # Multi-dimensional - flatten to 1D
            original_shape = tensor.shape
            tensor = tensor.flatten()
            corrections.append(f"Flattened {original_shape} to 1D for embedding")
            warnings.append("High-dimensional tensor flattened - structure lost")
        
        # Apply target shape if specified
        if target_shape and len(target_shape) == 1:
            target_size = target_shape[0]
            current_size = tensor.shape[0]
            
            if current_size > target_size:
                # Truncate
                tensor = tensor[:target_size]
                corrections.append(f"Truncated from {current_size} to {target_size}")
            elif current_size < target_size:
                # Pad with zeros
                padding = target_size - current_size
                tensor = F.pad(tensor, (0, padding), mode='constant', value=0)
                corrections.append(f"Padded from {current_size} to {target_size}")
        
        return tensor
    
    def _validate_cognitive_field_shape(self, 
                                      tensor: torch.Tensor, 
                                      target_shape: Optional[Tuple[int, ...]], 
                                      corrections: List[str], 
                                      warnings: List[str]) -> torch.Tensor:
        """Validate and correct cognitive field tensor shape"""
        
        # Cognitive fields should be 1D vectors
        if tensor.dim() != 1:
            original_shape = tensor.shape
            tensor = tensor.flatten()
            corrections.append(f"Flattened cognitive field from {original_shape} to 1D")
        
        # Ensure reasonable size for cognitive processing
        if tensor.shape[0] < 64:
            warnings.append(f"Cognitive field dimension {tensor.shape[0]} may be too small")
        elif tensor.shape[0] > 8192:
            warnings.append(f"Cognitive field dimension {tensor.shape[0]} may be too large")
        
        return tensor
    
    def _validate_quantum_state_shape(self, 
                                    tensor: torch.Tensor, 
                                    target_shape: Optional[Tuple[int, ...]], 
                                    corrections: List[str], 
                                    warnings: List[str]) -> torch.Tensor:
        """Validate and correct quantum state tensor shape"""
        
        # Quantum states should be normalized
        if tensor.dim() == 1:
            # Check normalization
            norm = torch.norm(tensor)
            if abs(norm - 1.0) > 1e-6:
                tensor = tensor / (norm + 1e-8)
                corrections.append(f"Normalized quantum state (norm was {norm:.6f})")
        
        return tensor
    
    def _validate_semantic_vector_shape(self, 
                                      tensor: torch.Tensor, 
                                      target_shape: Optional[Tuple[int, ...]], 
                                      corrections: List[str], 
                                      warnings: List[str]) -> torch.Tensor:
        """Validate and correct semantic vector tensor shape"""
        
        # Similar to embedding validation but with semantic-specific checks
        tensor = self._validate_embedding_shape(tensor, target_shape, corrections, warnings)
        
        # Check for semantic meaningfulness
        if torch.std(tensor) < 1e-6:
            warnings.append("Semantic vector has very low variance - may lack semantic content")
        
        return tensor
    
    def _validate_attention_weights_shape(self, 
                                        tensor: torch.Tensor, 
                                        target_shape: Optional[Tuple[int, ...]], 
                                        corrections: List[str], 
                                        warnings: List[str]) -> torch.Tensor:
        """Validate and correct attention weights tensor shape"""
        
        # Attention weights should sum to 1 across relevant dimension
        if tensor.dim() >= 1:
            # Apply softmax normalization to last dimension
            tensor = F.softmax(tensor, dim=-1)
            corrections.append("Applied softmax normalization to attention weights")
        
        return tensor
    
    def _optimize_memory_layout(self, 
                              tensor: torch.Tensor, 
                              corrections: List[str]) -> torch.Tensor:
        """Optimize tensor memory layout for performance"""
        
        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
            corrections.append("Made tensor memory layout contiguous")
        
        # Use appropriate dtype for efficiency
        if tensor.dtype == torch.float64 and tensor.numel() > 1000:
            # Large tensors can use float32 for memory efficiency
            tensor = tensor.to(torch.float32)
            corrections.append("Converted to float32 for memory efficiency")
        
        return tensor
    
    def _apply_safety_bounds(self, 
                           tensor: torch.Tensor, 
                           corrections: List[str], 
                           warnings: List[str]) -> torch.Tensor:
        """Apply safety bounds to prevent numerical instability"""
        
        min_bound, max_bound = self.safety_bounds
        
        # Check for values outside safety bounds
        out_of_bounds = (tensor < min_bound) | (tensor > max_bound)
        if out_of_bounds.any():
            num_out_of_bounds = out_of_bounds.sum().item()
            warnings.append(f"{num_out_of_bounds} values outside safety bounds")
            
            # Clip to safety bounds
            tensor = torch.clamp(tensor, min_bound, max_bound)
            corrections.append(f"Clipped {num_out_of_bounds} values to safety bounds")
        
        return tensor
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            cached = torch.cuda.memory_reserved(self.device) / (1024**3)  # GB
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'utilization_percent': (allocated / 24.0) * 100  # Assuming 24GB GPU
            }
        else:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'allocated_gb': (memory.total - memory.available) / (1024**3),
                'cached_gb': 0.0,
                'utilization_percent': memory.percent
            }
    
    def validate_cognitive_compatibility(self, 
                                       tensor: torch.Tensor, 
                                       cognitive_field_dim: int) -> bool:
        """Check if tensor is compatible with cognitive field operations"""
        
        # Must be 1D for cognitive field compatibility
        if tensor.dim() != 1:
            return False
        
        # Must match or be adaptable to cognitive field dimension
        if tensor.shape[0] != cognitive_field_dim:
            # Check if it can be padded or truncated reasonably
            size_ratio = tensor.shape[0] / cognitive_field_dim
            if size_ratio < 0.1 or size_ratio > 10.0:
                return False  # Too different in size
        
        # Must not have extreme values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
        
        # Must have reasonable variance
        if torch.std(tensor) < 1e-8:
            return False  # Too uniform
        
        return True

# Factory function for easy instantiation
def create_advanced_tensor_processor(device: Optional[torch.device] = None,
                                   safety_bounds: Tuple[float, float] = (-100.0, 100.0),
                                   memory_limit_gb: float = 16.0) -> AdvancedTensorProcessor:
    """Create an AdvancedTensorProcessor with specified configuration"""
    return AdvancedTensorProcessor(device, safety_bounds, memory_limit_gb) 