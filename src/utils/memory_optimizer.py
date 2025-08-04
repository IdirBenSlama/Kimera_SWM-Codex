"""
Memory Optimizer
================
Optimizes memory usage and prevents leaks.
"""

import gc
import logging
import weakref
from collections import defaultdict
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Memory optimization and leak prevention"""

    def __init__(self):
        self._object_registry: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
        self._gc_threshold = 100000  # Trigger GC after this many objects
        self._last_gc_count = 0

    def register_object(self, category: str, obj: Any):
        """Register an object for tracking"""
        self._object_registry[category].add(obj)

    def get_object_counts(self) -> Dict[str, int]:
        """Get counts of registered objects"""
        return {
            category: len(objects)
            for category, objects in self._object_registry.items()
        }

    def optimize_memory(self):
        """Run memory optimization"""
        # Force garbage collection
        gc.collect()

        # Clear weakref sets
        for category, objects in list(self._object_registry.items()):
            if len(objects) == 0:
                del self._object_registry[category]

        # Log memory stats
        counts = self.get_object_counts()
        if counts:
            logger.info(f"Memory stats: {counts}")

    def check_memory_pressure(self):
        """Check if memory optimization is needed"""
        current_count = len(gc.get_objects())

        if current_count - self._last_gc_count > self._gc_threshold:
            logger.info(f"Memory pressure detected: {current_count} objects")
            self.optimize_memory()
            self._last_gc_count = current_count

    def clear_caches(self):
        """Clear all caches to free memory"""
        # Clear function caches
        import functools

        functools._lru_cache_clear_all()

        # Clear module caches
        import linecache

        linecache.clearcache()

        # Force GC
        gc.collect()

        logger.info("Caches cleared")


# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()
