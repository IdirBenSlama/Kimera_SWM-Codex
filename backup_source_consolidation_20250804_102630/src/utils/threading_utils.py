"""
Threading utilities for Kimera SWM
Provides background task management and thread safety utilities
"""

import threading
import asyncio
import logging
import functools
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Global thread pool for background tasks
_thread_pool = None
_max_workers = 4

def get_thread_pool():
    """Get or create the global thread pool"""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=_max_workers)
    return _thread_pool

def start_background_task(func: Callable, *args, **kwargs) -> threading.Thread:
    """
    Start a function in a background thread
    
    Args:
        func: Function to run in background
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        threading.Thread: The thread object
    """
    def wrapper():
        try:
            logger.info(f"Starting background task: {func.__name__}")
            if asyncio.iscoroutinefunction(func):
                # Handle async functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(func(*args, **kwargs))
                finally:
                    loop.close()
            else:
                # Handle sync functions
                func(*args, **kwargs)
            logger.info(f"Background task completed: {func.__name__}")
        except Exception as e:
            logger.error(f"Background task failed: {func.__name__}: {e}", exc_info=True)
    
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    return thread

def run_in_thread_pool(func: Callable, *args, **kwargs):
    """
    Run a function in the thread pool
    
    Args:
        func: Function to run
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Future object
    """
    pool = get_thread_pool()
    return pool.submit(func, *args, **kwargs)

def thread_safe(func: Callable) -> Callable:
    """
    Decorator to make a function thread-safe using a lock
    
    Args:
        func: Function to make thread-safe
        
    Returns:
        Thread-safe version of the function
    """
    lock = threading.Lock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    
    return wrapper

class ThreadSafeCounter:
    """Thread-safe counter implementation"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment the counter and return new value"""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement the counter and return new value"""
        with self._lock:
            self._value -= amount
            return self._value
    
    @property
    def value(self) -> int:
        """Get current counter value"""
        with self._lock:
            return self._value
    
    def reset(self, new_value: int = 0) -> int:
        """Reset counter to new value"""
        with self._lock:
            old_value = self._value
            self._value = new_value
            return old_value

class ThreadSafeDict:
    """Thread-safe dictionary implementation"""
    
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get value by key"""
        with self._lock:
            return self._dict.get(key, default)
    
    def set(self, key: Any, value: Any) -> None:
        """Set value by key"""
        with self._lock:
            self._dict[key] = value
    
    def delete(self, key: Any) -> bool:
        """Delete key and return True if existed"""
        with self._lock:
            if key in self._dict:
                del self._dict[key]
                return True
            return False
    
    def keys(self):
        """Get all keys"""
        with self._lock:
            return list(self._dict.keys())
    
    def values(self):
        """Get all values"""
        with self._lock:
            return list(self._dict.values())
    
    def items(self):
        """Get all items"""
        with self._lock:
            return list(self._dict.items())
    
    def clear(self):
        """Clear all items"""
        with self._lock:
            self._dict.clear()
    
    def __len__(self):
        with self._lock:
            return len(self._dict)

def async_to_sync(coro):
    """
    Convert an async coroutine to a sync function
    
    Args:
        coro: Coroutine to convert
        
    Returns:
        Synchronous function result
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(coro)

def sync_to_async(func: Callable) -> Callable:
    """
    Convert a sync function to async using thread pool
    
    Args:
        func: Synchronous function to convert
        
    Returns:
        Async function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(get_thread_pool(), func, *args, **kwargs)
    
    return wrapper

def cleanup_threading():
    """Cleanup threading resources"""
    global _thread_pool
    if _thread_pool:
        logger.info("Shutting down thread pool...")
        _thread_pool.shutdown(wait=True)
        _thread_pool = None
        logger.info("Thread pool shutdown complete")

# Register cleanup on module exit
import atexit
atexit.register(cleanup_threading) 