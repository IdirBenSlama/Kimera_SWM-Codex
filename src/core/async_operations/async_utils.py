"""
Async Utilities for KIMERA System
Utilities to prevent blocking calls in async contexts
Phase 2, Week 5: Async/Await Patterns Implementation
"""

import asyncio
import functools
import inspect
import json
import logging
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional, TypeVar

import aiofiles

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Global thread pool for blocking operations
_thread_pool: Optional[ThreadPoolExecutor] = None
_process_pool: Optional[ProcessPoolExecutor] = None


def get_thread_pool() -> ThreadPoolExecutor:
    """Get global thread pool for blocking operations"""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="kimera_blocking"
        )
    return _thread_pool


def get_process_pool() -> ProcessPoolExecutor:
    """Get global process pool for CPU-intensive operations"""
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=4)
    return _process_pool


async def run_in_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a blocking function in a thread pool

    Args:
        func: Blocking function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function
    """
    loop = asyncio.get_running_loop()

    # Use functools.partial to create a callable with arguments
    if args or kwargs:
        func_with_args = functools.partial(func, *args, **kwargs)
    else:
        func_with_args = func

    # Run in thread pool
    result = await loop.run_in_executor(get_thread_pool(), func_with_args)

    logger.debug(f"Executed {func.__name__} in thread pool")
    return result


async def run_in_process(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run a CPU-intensive function in a process pool

    Args:
        func: CPU-intensive function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function
    """
    loop = asyncio.get_running_loop()

    # Use functools.partial to create a callable with arguments
    if args or kwargs:
        func_with_args = functools.partial(func, *args, **kwargs)
    else:
        func_with_args = func

    # Run in process pool
    result = await loop.run_in_executor(get_process_pool(), func_with_args)

    logger.debug(f"Executed {func.__name__} in process pool")
    return result


def make_async(
    blocking_func: Callable[..., T],
) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Decorator to convert a blocking function to async by running it in a thread pool

    Args:
        blocking_func: Blocking function to convert

    Returns:
        Async version of the function
    """

    @functools.wraps(blocking_func)
    async def async_wrapper(*args, **kwargs):
        return await run_in_thread(blocking_func, *args, **kwargs)

    return async_wrapper


def cpu_bound_async(
    cpu_func: Callable[..., T],
) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Decorator to convert a CPU-intensive function to async by running it in a process pool

    Args:
        cpu_func: CPU-intensive function to convert

    Returns:
        Async version of the function
    """

    @functools.wraps(cpu_func)
    async def async_wrapper(*args, **kwargs):
        return await run_in_process(cpu_func, *args, **kwargs)

    return async_wrapper
class AsyncTimer:
    """Auto-generated class."""
    pass
    """Async context manager for timing operations"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} took {duration:.3f} seconds")

    @property
    def elapsed(self) -> float:
        """Get elapsed time"""
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time


async def gather_with_timeout(
    *coros: Coroutine, timeout: float = 30.0, return_exceptions: bool = True
) -> list:
    """
    Gather multiple coroutines with a timeout

    Args:
        *coros: Coroutines to gather
        timeout: Timeout in seconds
        return_exceptions: Whether to return exceptions or raise

    Returns:
        List of results
    """
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=return_exceptions), timeout=timeout
        )
        return results
    except asyncio.TimeoutError:
        logger.error(f"Gather operation timed out after {timeout}s")
        # Cancel remaining tasks
        for coro in coros:
            if hasattr(coro, "cancel"):
                coro.cancel()
        raise


async def retry_async(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args
    max_attempts: int = 3
    delay: float = 1.0
    backoff: float = 2.0
    exceptions: tuple = (Exception,),
    **kwargs
) -> T:
    """
    Retry an async function with exponential backoff

    Args:
        func: Async function to retry
        *args: Positional arguments for the function
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function

    Raises:
        Last exception if all attempts fail
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_attempts):
        try:
            result = await func(*args, **kwargs)
            if attempt > 0:
                logger.info(
                    f"Retry successful for {func.__name__} after {attempt} attempts"
                )
            return result

        except exceptions as e:
            last_exception = e
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                    f"Retrying in {current_delay}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_attempts} attempts failed for {func.__name__}")

    raise last_exception


def ensure_async(func: Callable) -> Callable[..., Coroutine]:
    """
    Ensure a function is async, converting if necessary

    Args:
        func: Function to ensure is async

    Returns:
        Async version of the function
    """
    if inspect.iscoroutinefunction(func):
        return func
    else:
        return make_async(func)
class AsyncLock:
    """Auto-generated class."""
    pass
    """Enhanced async lock with timeout and debugging"""

    def __init__(self, name: str = "lock", timeout: Optional[float] = None):
        self.name = name
        self.timeout = timeout
        self._lock = asyncio.Lock()
        self._holder: Optional[str] = None
        self._acquired_at: Optional[float] = None

    async def acquire(self):
        """Acquire the lock with optional timeout"""
        task_name = (
            asyncio.current_task().get_name() if asyncio.current_task() else "unknown"
        )

        if self.timeout:
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout=self.timeout)
            except asyncio.TimeoutError:
                logger.error(
                    f"Failed to acquire lock '{self.name}' within {self.timeout}s. "
                    f"Currently held by: {self._holder}"
                )
                raise
        else:
            await self._lock.acquire()

        self._holder = task_name
        self._acquired_at = time.time()
        logger.debug(f"Lock '{self.name}' acquired by {task_name}")

    def release(self):
        """Release the lock"""
        if self._acquired_at:
            hold_time = time.time() - self._acquired_at
            if hold_time > 1.0:  # Log if held for more than 1 second
                logger.warning(
                    f"Lock '{self.name}' held by {self._holder} for {hold_time:.2f}s"
                )

        self._lock.release()
        logger.debug(f"Lock '{self.name}' released by {self._holder}")
        self._holder = None
        self._acquired_at = None

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Async file operations
async def read_json_async(file_path: Path) -> dict:
    """Read JSON file asynchronously"""
    async with aiofiles.open(file_path, mode="r") as f:
        content = await f.read()
        return json.loads(content)


async def write_json_async(file_path: Path, data: dict, indent: int = 2) -> None:
    """Write JSON file asynchronously"""
    content = json.dumps(data, indent=indent)
    async with aiofiles.open(file_path, mode="w") as f:
        await f.write(content)


async def read_pickle_async(file_path: Path) -> Any:
    """Read pickle file asynchronously"""
    async with aiofiles.open(file_path, mode="rb") as f:
        content = await f.read()
        return pickle.loads(content)


async def write_pickle_async(file_path: Path, data: Any) -> None:
    """Write pickle file asynchronously"""
    content = pickle.dumps(data)
    async with aiofiles.open(file_path, mode="wb") as f:
        await f.write(content)


async def cleanup_resources():
    """Cleanup global resources"""
    global _thread_pool, _process_pool

    if _thread_pool:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None
        logger.info("Thread pool shut down")

    if _process_pool:
        _process_pool.shutdown(wait=True)
        _process_pool = None
        logger.info("Process pool shut down")


# Async queue with timeout
class AsyncQueue:
    """Auto-generated class."""
    pass
    """Enhanced async queue with timeout and monitoring"""

    def __init__(self, maxsize: int = 0, name: str = "queue"):
        self.name = name
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._total_put = 0
        self._total_get = 0

    async def put(self, item: Any, timeout: Optional[float] = None) -> None:
        """Put item in queue with optional timeout"""
        if timeout:
            await asyncio.wait_for(self._queue.put(item), timeout=timeout)
        else:
            await self._queue.put(item)
        self._total_put += 1

    async def get(self, timeout: Optional[float] = None) -> Any:
        """Get item from queue with optional timeout"""
        if timeout:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        else:
            item = await self._queue.get()
        self._total_get += 1
        return item

    def qsize(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()

    def full(self) -> bool:
        """Check if queue is full"""
        return self._queue.full()

    def stats(self) -> dict:
        """Get queue statistics"""
        return {
            "name": self.name
            "size": self.qsize(),
            "total_put": self._total_put
            "total_get": self._total_get
            "is_empty": self.empty(),
            "is_full": self.full(),
        }
