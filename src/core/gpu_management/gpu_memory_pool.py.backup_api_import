"""
TCSE Optimized GPU Memory Pool
==============================

This module implements a specialized memory pool for TCSE GPU operations,
designed to minimize the overhead of frequent memory allocations and deallocations
by reusing pre-allocated memory blocks.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Dict, List, Optional

import cupy as cp

# Configuration Management
try:
    from src.utils.config import get_api_settings
except ImportError:
    try:
        from ...utils.config import get_api_settings
    except ImportError:
        try:
            from utils.config import get_api_settings
        except ImportError:
            # Emergency fallback
            def get_api_settings():
                return {"gpu": {"memory_limit": 8192}}


try:
    from src.config.settings import get_settings
except ImportError:
    try:
        from ...config.settings import get_settings
    except ImportError:
        try:
            from config.settings import get_settings
        except ImportError:
            # Emergency fallback
            def get_settings():
                return {"gpu": {"enabled": False}}


logger = logging.getLogger(__name__)


class TCSignalMemoryPool:
    """
    Manages a pool of CuPy memory allocations for efficient reuse in TCSE tasks.

    This class provides a simple but effective memory pooling mechanism. It helps
    avoid the performance penalties associated with `cudaMalloc` and `cudaFree`
    by holding onto allocated GPU memory and serving it to requesting components.
    When memory is no longer needed, it's returned to the pool instead of being
    freed, making it instantly available for the next request of the same size.
    """

    def __init__(
        self,
        initial_blocks: int = 10,
        block_size: int = 1024 * 1024 * 32,
        device_id: int = 0,
    ):
        """
        Initializes the memory pool.

        Args:
            initial_blocks (int): The number of memory blocks to pre-allocate.
            block_size (int): The size of each memory block in bytes (default: 32MB).
            device_id (int): The GPU device ID to allocate memory on.

        Raises:
            RuntimeError: If CUDA is not available or device is invalid
            ValueError: If parameters are invalid
        """
        # Input validation
        if initial_blocks <= 0:
            raise ValueError(f"Initial blocks must be positive, got {initial_blocks}")
        if block_size <= 0:
            raise ValueError(f"Block size must be positive, got {block_size}")
        if device_id < 0:
            raise ValueError(f"Device ID must be non-negative, got {device_id}")

        # Check CUDA availability
        if not cp.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available - GPU memory pool cannot be initialized"
            )

        # Validate device ID
        num_devices = cp.cuda.runtime.getDeviceCount()
        if device_id >= num_devices:
            raise ValueError(
                f"Invalid device ID {device_id}. Available devices: 0-{num_devices-1}"
            )

        self.device_id = device_id
        self.block_size = block_size
        self.lock = Lock()

        try:
            with cp.cuda.Device(device_id):
                # Log device information
                device_name = cp.cuda.runtime.getDeviceProperties(device_id)[
                    "name"
                ].decode()
                free_memory, total_memory = cp.cuda.runtime.memGetInfo()
                logger.info(
                    f"🖥️ GPU Memory Pool: Initializing on {device_name} (Device {device_id})"
                )
                logger.info(
                    f"   GPU Memory: {free_memory / 1024**3:.1f}GB free / {total_memory / 1024**3:.1f}GB total"
                )

                self.pool: Dict[int, List[cp.ndarray]] = (
                    {}
                )  # size -> list of available blocks
                self._preallocate_blocks(initial_blocks, block_size)

                logger.info(
                    f"🧠 TCSignalMemoryPool initialized on device {device_id} with {initial_blocks} blocks of {block_size // 1024**2}MB."
                )

        except cp.cuda.runtime.CudaError as e:
            logger.error(
                f"Failed to initialize GPU memory pool on device {device_id}: {e}"
            )
            raise RuntimeError(f"GPU memory pool initialization failed: {e}") from e

    def _preallocate_blocks(self, num_blocks: int, size: int):
        """Pre-allocates a number of memory blocks of a given size."""
        if size not in self.pool:
            self.pool[size] = []
        try:
            for _ in range(num_blocks):
                block = cp.empty(size, dtype=cp.uint8)
                self.pool[size].append(block)
        except cp.cuda.runtime.CudaError as e:
            logger.error(
                f"Failed to pre-allocate {num_blocks} blocks of size {size}: {e}"
            )
            raise

    def get_block(self, size: int) -> Optional[cp.ndarray]:
        """
        Retrieves a memory block of the specified size from the pool.

        If a block of the exact size is available, it's returned.
        If not, a new block is allocated.

        Args:
            size (int): The desired size of the memory block in bytes.

        Returns:
            Optional[cp.ndarray]: A CuPy array representing the memory block, or None if allocation fails.
        """
        with self.lock:
            if size in self.pool and self.pool[size]:
                logger.debug(f"Reusing block of size {size} from pool.")
                return self.pool[size].pop()
            else:
                logger.info(f"Allocating new block of size {size} as pool is empty.")
                try:
                    block = cp.empty(size, dtype=cp.uint8)
                    return block
                except cp.cuda.runtime.CudaError as e:
                    logger.critical(
                        f"Failed to allocate new GPU memory block of size {size}: {e}"
                    )
                    return None

    def release_block(self, block: cp.ndarray):
        """

        Returns a memory block to the pool for future reuse.

        Args:
            block (cp.ndarray): The CuPy array to return to the pool.
        """
        size = block.nbytes
        with self.lock:
            if size not in self.pool:
                self.pool[size] = []
            logger.debug(f"Returning block of size {size} to pool.")
            self.pool[size].append(block)

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the memory pool's current state."""
        with self.lock:
            stats = {
                "total_blocks": 0,
                "total_pooled_memory_mb": 0,
                "pool_breakdown": {},
            }
            for size, blocks in self.pool.items():
                num_blocks = len(blocks)
                stats["total_blocks"] += num_blocks
                stats["total_pooled_memory_mb"] += (num_blocks * size) / (1024**2)
                stats["pool_breakdown"][size] = num_blocks
            return stats
