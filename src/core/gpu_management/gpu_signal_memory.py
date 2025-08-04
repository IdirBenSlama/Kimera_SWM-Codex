"""
GPU Signal Memory Manager
=========================

This module implements an efficient memory management system for TCSE signal
properties on the GPU. It aims to minimize memory overhead while providing
fast access to signal data for GPU kernels.

Key Components:
- GPUSignalMemoryManager: Main class for managing signal-enhanced fields.
- GPUBufferPool: A simple pool for reusing GPU memory buffers.
- SignalEnhancedField: A dataclass linking a SemanticField to its GPU-based signal properties.
"""

import logging
from collections import deque
from typing import Any, Dict, Optional

import cupy as cp
import torch

from src.config.settings import get_settings
from src.core.cognitive_field_dynamics import SemanticField
from src.utils.config import get_api_settings
from src.utils.gpu_foundation import GPUFoundation

logger = logging.getLogger(__name__)


class GPUBufferPool:
    """A simple memory pool for reusing GPU buffers to reduce allocation overhead."""

    def __init__(
        self, initial_size=10, buffer_size_bytes=1024 * 1024
    ):  # Default 1MB buffers
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.pool = deque(maxlen=initial_size)
        self.buffer_size = buffer_size_bytes
        # In a real system, you'd integrate with CuPy's or PyTorch's memory pool.
        # This is a conceptual implementation.
        logger.info(
            f"GPUBufferPool initialized with {initial_size} buffers of {buffer_size_bytes} bytes."
        )

    def allocate(self, size_bytes: int) -> cp.ndarray:
        """Allocate a buffer from the pool or create a new one."""
        if self.pool and self.pool[0].nbytes >= size_bytes:
            return self.pool.popleft()
        return cp.zeros(size_bytes // 4, dtype=cp.float32)  # Assuming float32

    def deallocate(self, buffer: cp.ndarray):
        """Return a buffer to the pool."""
        self.pool.append(buffer)


class SignalEnhancedField:
    """
    Links a base SemanticField to its TCSE signal properties stored on the GPU.
    This structure avoids data duplication by referencing the base field's embedding
    while managing a separate, small buffer for thermodynamic properties.
    """

    def __init__(self, base_field: SemanticField, signal_properties_buffer: cp.ndarray):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.base_field = base_field
        self.signal_properties = signal_properties_buffer
        # 4 properties (e.g., temp, potential, coherence, flow) * 4 bytes/float
        self.memory_overhead_bytes = 16

    @property
    def geoid_id(self):
        return self.base_field.geoid_id

    @property
    def embedding(self):
        return self.base_field.embedding


class GPUSignalMemoryManager:
    """
    Manages the allocation and lifecycle of signal-enhanced fields on the GPU,
    focusing on memory efficiency.
    """

    def __init__(self, gpu_foundation: GPUFoundation):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.gpu_foundation = gpu_foundation
        self.signal_buffer_pool = GPUBufferPool()
        self.managed_fields: Dict[str, SignalEnhancedField] = {}
        self.total_memory_overhead = 0
        logger.info("GPUSignalMemoryManager initialized.")

    def allocate_signal_enhanced_field(
        self, base_field: SemanticField
    ) -> SignalEnhancedField:
        """
        Allocate GPU memory for a signal-enhanced field, reusing the base embedding.
        """
        if base_field.geoid_id in self.managed_fields:
            return self.managed_fields[base_field.geoid_id]

        # Allocate a minimal buffer for the 4 TCSE signal properties.
        # (signal_temperature, cognitive_potential, signal_coherence, entropic_flow_capacity)
        signal_buffer = self.signal_buffer_pool.allocate(16)  # 4 floats * 4 bytes

        enhanced_field = SignalEnhancedField(
            base_field=base_field, signal_properties_buffer=signal_buffer
        )

        self.managed_fields[base_field.geoid_id] = enhanced_field
        self.total_memory_overhead += enhanced_field.memory_overhead_bytes

        logger.debug(
            f"Allocated signal-enhanced field for {base_field.geoid_id}. Overhead: {self.total_memory_overhead} bytes."
        )
        return enhanced_field

    def deallocate_field(self, geoid_id: str):
        """Deallocate a field and return its buffer to the pool."""
        if geoid_id in self.managed_fields:
            enhanced_field = self.managed_fields.pop(geoid_id)
            self.signal_buffer_pool.deallocate(enhanced_field.signal_properties)
            self.total_memory_overhead -= enhanced_field.memory_overhead_bytes
            logger.debug(
                f"Deallocated signal-enhanced field for {geoid_id}. Overhead: {self.total_memory_overhead} bytes."
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Return memory management statistics."""
        return {
            "managed_fields_count": len(self.managed_fields),
            "total_memory_overhead_bytes": self.total_memory_overhead,
            "buffer_pool_size": len(self.signal_buffer_pool.pool),
        }
