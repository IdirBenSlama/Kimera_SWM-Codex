"""
Advanced Tensor Processor for KIMERA
====================================

A comprehensive tensor validation and processing module that ensures
tensor operations are safe, efficient, and scientifically rigorous.

Key Features:
- Automatic tensor shape validation and correction
- Memory-efficient operations with bounds checking
- GPU/CPU compatibility handling
- Comprehensive error recovery
- Performance monitoring and optimization
"""

import gc
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch

from ..config.settings import get_settings
from ..utils.config import get_api_settings

logger = logging.getLogger(__name__)


class TensorType(Enum):
    """Types of tensors for specialized processing."""

    EMBEDDING = "embedding"
    ATTENTION = "attention"
    HIDDEN_STATE = "hidden_state"
    LOGITS = "logits"
    NOISE = "noise"
    GENERAL = "general"


@dataclass
class TensorValidationResult:
    """Results from tensor validation and correction."""

    is_valid: bool
    original_shape: Tuple[int, ...]
    corrected_shape: Optional[Tuple[int, ...]]
    corrections_applied: List[str]
    safety_warnings: List[str]
    processing_time_ms: float
    memory_usage_mb: float


class AdvancedTensorProcessor:
    """
    Advanced tensor processing with comprehensive validation and safety.

    This processor ensures all tensor operations are:
    1. Memory-safe (prevents OOM errors)
    2. Numerically stable (prevents NaN/Inf)
    3. Shape-consistent (prevents dimension mismatches)
    4. Device-compatible (handles GPU/CPU transfers)
    """

    def __init__(
        self,
        device: torch.device,
        safety_bounds: Tuple[float, float] = (-100.0, 100.0),
        memory_limit_gb: float = 8.0,
        enable_mixed_precision: bool = True,
    ):
        """
        Initialize the advanced tensor processor.

        Args:
            device: Target device for tensor operations
            safety_bounds: Min/max values for tensor elements
            memory_limit_gb: Maximum memory usage in GB
            enable_mixed_precision: Whether to use mixed precision
        """
        self.device = device
        self.safety_bounds = safety_bounds
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)
        self.enable_mixed_precision = enable_mixed_precision and device.type == "cuda"

        # Expected dimensions for different tensor types
        self.expected_dims = {
            TensorType.EMBEDDING: (64, 8192),  # Min, max dimensions
            TensorType.ATTENTION: (1, 4096),
            TensorType.HIDDEN_STATE: (128, 4096),
            TensorType.LOGITS: (1, 100000),
            TensorType.NOISE: (1, 8192),
            TensorType.GENERAL: (1, 1000000),
        }

        logger.info(f"🔧 Advanced Tensor Processor initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Safety bounds: {safety_bounds}")
        logger.info(f"   Memory limit: {memory_limit_gb:.1f} GB")
        logger.info(f"   Mixed precision: {self.enable_mixed_precision}")

    def validate_and_correct_tensor(
        self,
        tensor: torch.Tensor,
        tensor_type: TensorType = TensorType.GENERAL,
        target_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[torch.Tensor, TensorValidationResult]:
        """
        Validate and correct a tensor with comprehensive safety checks.

        Args:
            tensor: Input tensor to validate
            tensor_type: Type of tensor for specialized handling
            target_shape: Optional target shape to conform to

        Returns:
            Tuple of (corrected_tensor, validation_result)
        """
        start_time = time.time()
        original_shape = tuple(tensor.shape)
        corrections = []
        warnings = []

        try:
            # Step 1: Device compatibility
            if tensor.device != self.device:
                tensor = tensor.to(self.device)
                corrections.append(
                    f"Moved tensor from {tensor.device} to {self.device}"
                )

            # Step 2: Check for NaN/Inf values
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                warnings.append(f"Found {nan_count} NaN and {inf_count} Inf values")

                # Replace NaN with 0 and Inf with bounds
                tensor = torch.nan_to_num(
                    tensor,
                    nan=0.0,
                    posinf=self.safety_bounds[1],
                    neginf=self.safety_bounds[0],
                )
                corrections.append("Replaced NaN/Inf values")

            # Step 3: Bounds checking
            if tensor.numel() > 0:
                min_val = tensor.min().item()
                max_val = tensor.max().item()

                if min_val < self.safety_bounds[0] or max_val > self.safety_bounds[1]:
                    warnings.append(
                        f"Values outside bounds: [{min_val:.2f}, {max_val:.2f}]"
                    )
                    tensor = torch.clamp(
                        tensor, min=self.safety_bounds[0], max=self.safety_bounds[1]
                    )
                    corrections.append(f"Clamped values to {self.safety_bounds}")

            # Step 4: Shape validation and correction
            tensor = self._validate_shape(
                tensor, tensor_type, target_shape, corrections, warnings
            )

            # Step 5: Memory usage check
            tensor_bytes = tensor.element_size() * tensor.numel()
            memory_mb = tensor_bytes / (1024 * 1024)

            if tensor_bytes > self.memory_limit_bytes:
                warnings.append(f"Tensor exceeds memory limit: {memory_mb:.1f} MB")
                # Optionally downsample or truncate
                tensor = self._reduce_tensor_memory(tensor, tensor_type, corrections)

            # Step 6: Numerical stability check
            if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                variance = tensor.var().item() if tensor.numel() > 1 else 0
                if variance > 1e6:
                    warnings.append(f"High variance detected: {variance:.2e}")
                    # Normalize if variance is too high
                    tensor = self._stabilize_tensor(tensor, corrections)

            # Step 7: Mixed precision handling
            if self.enable_mixed_precision and tensor.dtype == torch.float32:
                tensor = tensor.half()
                corrections.append("Converted to float16 for mixed precision")

            processing_time_ms = (time.time() - start_time) * 1000

            result = TensorValidationResult(
                is_valid=len(warnings) == 0,
                original_shape=original_shape,
                corrected_shape=tuple(tensor.shape),
                corrections_applied=corrections,
                safety_warnings=warnings,
                processing_time_ms=processing_time_ms,
                memory_usage_mb=memory_mb,
            )

            return tensor, result

        except Exception as e:
            logger.error(f"❌ Critical error in tensor validation: {e}")
            # Return original tensor with error result
            result = TensorValidationResult(
                is_valid=False,
                original_shape=original_shape,
                corrected_shape=None,
                corrections_applied=[],
                safety_warnings=[f"Critical error: {str(e)}"],
                processing_time_ms=(time.time() - start_time) * 1000,
                memory_usage_mb=0,
            )
            return tensor, result

    def _validate_shape(
        self,
        tensor: torch.Tensor,
        tensor_type: TensorType,
        target_shape: Optional[Tuple[int, ...]],
        corrections: List[str],
        warnings: List[str],
    ) -> torch.Tensor:
        """Validate and correct tensor shape."""
        min_dim, max_dim = self.expected_dims[tensor_type]

        # Handle empty tensors
        if tensor.numel() == 0:
            warnings.append("Empty tensor detected")
            # Create a minimal valid tensor
            if target_shape:
                tensor = torch.zeros(
                    target_shape, device=self.device, dtype=tensor.dtype
                )
            else:
                tensor = torch.zeros(min_dim, device=self.device, dtype=tensor.dtype)
            corrections.append("Replaced empty tensor with zeros")
            return tensor

        # Handle specific shape requirements
        if target_shape:
            if tensor.shape != target_shape:
                tensor = self._reshape_to_target(tensor, target_shape, corrections)
        else:
            # Validate dimensions for tensor type
            if tensor_type == TensorType.EMBEDDING:
                tensor = self._validate_embedding_shape(
                    tensor, min_dim, max_dim, corrections, warnings
                )
            elif tensor.dim() > 4:
                # Flatten very high dimensional tensors
                original_shape = tensor.shape
                tensor = tensor.flatten()
                corrections.append(f"Flattened {len(original_shape)}D tensor to 1D")

        return tensor

    def _validate_embedding_shape(
        self,
        tensor: torch.Tensor,
        min_dim: int,
        max_dim: int,
        corrections: List[str],
        warnings: List[str],
    ) -> torch.Tensor:
        """Special handling for embedding tensors."""
        # Ensure 1D or 2D
        if tensor.dim() > 2:
            tensor = tensor.flatten()
            corrections.append(f"Flattened embedding from {tensor.dim()}D to 1D")
        elif tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
            corrections.append("Added dimension to scalar tensor")

        # Check size constraints
        size = tensor.shape[-1] if tensor.dim() > 1 else tensor.shape[0]

        if size < min_dim:
            # Pad if too small
            padding_size = min_dim - size
            if tensor.dim() == 1:
                tensor = torch.cat(
                    [
                        tensor,
                        torch.zeros(
                            padding_size, device=tensor.device, dtype=tensor.dtype
                        ),
                    ]
                )
            else:
                tensor = torch.cat(
                    [
                        tensor,
                        torch.zeros(
                            tensor.shape[0],
                            padding_size,
                            device=tensor.device,
                            dtype=tensor.dtype,
                        ),
                    ],
                    dim=1,
                )
            corrections.append(f"Padded embedding from {size} to {min_dim} dimensions")
        elif size > max_dim:
            # Truncate if too large
            if tensor.dim() == 1:
                tensor = tensor[:max_dim]
            else:
                tensor = tensor[:, :max_dim]
            warnings.append(f"Truncated embedding from {size} to {max_dim} dimensions")
            corrections.append(f"Truncated embedding to {max_dim} dimensions")

        return tensor

    def _reshape_to_target(
        self,
        tensor: torch.Tensor,
        target_shape: Tuple[int, ...],
        corrections: List[str],
    ) -> torch.Tensor:
        """Reshape tensor to target shape with safety checks."""
        target_numel = np.prod(target_shape)
        current_numel = tensor.numel()

        if current_numel == target_numel:
            # Simple reshape
            tensor = tensor.reshape(target_shape)
            corrections.append(f"Reshaped from {tensor.shape} to {target_shape}")
        elif current_numel < target_numel:
            # Pad and reshape
            padding_size = target_numel - current_numel
            flat_tensor = tensor.flatten()
            padded = torch.cat(
                [
                    flat_tensor,
                    torch.zeros(padding_size, device=tensor.device, dtype=tensor.dtype),
                ]
            )
            tensor = padded.reshape(target_shape)
            corrections.append(f"Padded and reshaped to {target_shape}")
        else:
            # Truncate and reshape
            flat_tensor = tensor.flatten()[:target_numel]
            tensor = flat_tensor.reshape(target_shape)
            corrections.append(f"Truncated and reshaped to {target_shape}")

        return tensor

    def _reduce_tensor_memory(
        self, tensor: torch.Tensor, tensor_type: TensorType, corrections: List[str]
    ) -> torch.Tensor:
        """Reduce tensor memory usage."""
        if tensor_type == TensorType.EMBEDDING and tensor.dim() == 2:
            # Reduce batch size for embeddings
            max_batch = 32
            if tensor.shape[0] > max_batch:
                tensor = tensor[:max_batch]
                corrections.append(f"Reduced batch size to {max_batch}")
        else:
            # General reduction - keep first portion
            max_elements = self.memory_limit_bytes // (
                4 * tensor.element_size()
            )  # Safety factor of 4
            if tensor.numel() > max_elements:
                flat = tensor.flatten()
                tensor = flat[:max_elements].reshape(-1)
                corrections.append(f"Truncated to {max_elements} elements for memory")

        return tensor

    def _stabilize_tensor(
        self, tensor: torch.Tensor, corrections: List[str]
    ) -> torch.Tensor:
        """Stabilize tensor values for numerical stability."""
        # Standardize if variance is too high
        if tensor.numel() > 1:
            mean = tensor.mean()
            std = tensor.std()
            if std > 1e-6:  # Avoid division by zero
                tensor = (tensor - mean) / (std + 1e-8)
                corrections.append("Standardized tensor for stability")

        return tensor

    def merge_tensors(
        self, tensors: List[torch.Tensor], merge_strategy: str = "mean"
    ) -> torch.Tensor:
        """
        Safely merge multiple tensors.

        Args:
            tensors: List of tensors to merge
            merge_strategy: "mean", "sum", "concat", or "weighted"

        Returns:
            Merged tensor
        """
        if not tensors:
            raise ValueError("No tensors provided for merging")

        # Validate all tensors
        validated_tensors = []
        for i, t in enumerate(tensors):
            validated, _ = self.validate_and_correct_tensor(t, TensorType.GENERAL)
            validated_tensors.append(validated)

        # Ensure compatible shapes
        if merge_strategy in ["mean", "sum", "weighted"]:
            # Need same shape
            target_shape = validated_tensors[0].shape
            aligned_tensors = []
            for t in validated_tensors:
                if t.shape != target_shape:
                    t = self._reshape_to_target(t, target_shape, [])
                aligned_tensors.append(t)
            validated_tensors = aligned_tensors

        # Perform merge
        if merge_strategy == "mean":
            return torch.stack(validated_tensors).mean(dim=0)
        elif merge_strategy == "sum":
            return torch.stack(validated_tensors).sum(dim=0)
        elif merge_strategy == "concat":
            return torch.cat(validated_tensors, dim=0)
        elif merge_strategy == "weighted":
            weights = torch.softmax(torch.randn(len(validated_tensors)), dim=0).to(
                self.device
            )
            stacked = torch.stack(validated_tensors)
            return (stacked * weights.view(-1, 1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    def safe_matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Perform safe matrix multiplication with automatic shape correction.

        Args:
            a: First tensor
            b: Second tensor

        Returns:
            Result of matrix multiplication
        """
        # Validate inputs
        a, _ = self.validate_and_correct_tensor(a, TensorType.GENERAL)
        b, _ = self.validate_and_correct_tensor(b, TensorType.GENERAL)

        # Ensure 2D tensors
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(1)

        # Check compatibility
        if a.shape[-1] != b.shape[0]:
            # Try to make compatible
            common_dim = min(a.shape[-1], b.shape[0])
            a = a[..., :common_dim]
            b = b[:common_dim, ...]
            logger.warning(f"Adjusted matmul dimensions to {common_dim}")

        return torch.matmul(a, b)

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        }

        if self.device.type == "cuda":
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated(self.device) / (
                1024**3
            )
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved(self.device) / (
                1024**3
            )
            stats["gpu_free_gb"] = (
                torch.cuda.get_device_properties(self.device).total_memory
                - torch.cuda.memory_allocated(self.device)
            ) / (1024**3)

        return stats

    def cleanup_memory(self):
        """Force memory cleanup."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("🧹 Memory cleanup completed")
