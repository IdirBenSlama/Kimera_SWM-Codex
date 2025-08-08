"""
KIMERA SWM - Async Compatibility Helper for Python 3.13
=====================================================

Provides compatibility utilities for async operations across Python versions.
"""

import asyncio
import sys
from typing import Any, Awaitable, TypeVar, Union

T = TypeVar("T")


# Python 3.13 compatibility utilities
def create_task_group():
    """Create a TaskGroup with backward compatibility."""
    if hasattr(asyncio, "TaskGroup"):
        return asyncio.TaskGroup()
    else:
        # Fallback for older Python versions
        return _LegacyTaskGroup()
class _LegacyTaskGroup:
    """Auto-generated class."""
    pass
    """Legacy TaskGroup implementation for Python < 3.11."""

    def __init__(self):
        self._tasks = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        return False

    def create_task(self, coro, *, name=None):
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)
        return task


# Enhanced timeout with Python 3.13 features
def timeout_context(delay):
    """Create timeout context with backward compatibility."""
    if hasattr(asyncio, "timeout"):
        return asyncio.timeout(delay)
    else:
        return asyncio.wait_for(delay)


# Improved cancellation handling
async def safe_cancel_task(task: asyncio.Task, msg: str = None) -> bool:
    """Safely cancel a task with proper cleanup."""
    if task.done():
        return True

    # Use Python 3.13 enhanced cancellation if available
    if sys.version_info >= (3, 9) and msg:
        task.cancel(msg)
    else:
        task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        return True
    except Exception:
        return False

    return task.cancelled()


# Enhanced async context manager utilities
class AsyncContextManager:
    """Auto-generated class."""
    pass
    """Base class for async context managers with enhanced error handling."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is asyncio.CancelledError:
            # Enhanced cancellation handling for Python 3.13
            await self._handle_cancellation()
        return False

    async def _handle_cancellation(self):
        """Override in subclasses for custom cancellation handling."""
        pass


# Python 3.13 async iterator improvements
class CompatAsyncIterator:
    """Auto-generated class."""
    pass
    """Enhanced async iterator with Python 3.13 compatibility."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Enhanced async iteration for Python 3.13
        try:
            return await self._get_next()
        except StopAsyncIteration:
            raise
        except asyncio.CancelledError:
            # Proper cleanup for cancelled iterations
            await self._cleanup()
            raise

    async def _get_next(self):
        """Override in subclasses."""
        raise StopAsyncIteration

    async def _cleanup(self):
        """Override in subclasses for cleanup."""
        pass
