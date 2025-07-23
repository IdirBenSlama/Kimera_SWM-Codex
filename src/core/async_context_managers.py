"""
Async Context Managers for KIMERA System
Implements proper async resource management patterns
Phase 2, Week 5: Async/Await Patterns Implementation
"""

import asyncio
import logging
from typing import Optional, Any, Dict, List, TypeVar, Generic, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import aiofiles
from pathlib import Path
import weakref
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncResourcePool(Generic[T]):
    """
    Generic async resource pool for managing limited resources
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        timeout: float = 30.0,
        cleanup: Optional[Callable[[T], None]] = None
    ):
        self.factory = factory
        self.max_size = max_size
        self.timeout = timeout
        self.cleanup = cleanup
        
        self._pool: List[T] = []
        self._in_use: weakref.WeakSet = weakref.WeakSet()
        self._semaphore = asyncio.Semaphore(max_size)
        self._lock = asyncio.Lock()
        self._closed = False
        
        logger.info(f"AsyncResourcePool initialized with max_size={max_size}")
    
    async def acquire(self) -> T:
        """Acquire a resource from the pool"""
        if self._closed:
            raise RuntimeError("Resource pool is closed")
        
        # Wait for available slot
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Failed to acquire resource within {self.timeout}s")
        
        async with self._lock:
            # Try to get from pool
            if self._pool:
                resource = self._pool.pop()
                logger.debug("Reusing resource from pool")
            else:
                # Create new resource
                resource = self.factory()
                logger.debug("Created new resource")
            
            self._in_use.add(resource)
            return resource
    
    async def release(self, resource: T) -> None:
        """Release a resource back to the pool"""
        async with self._lock:
            if resource in self._in_use:
                self._in_use.remove(resource)
                
                if not self._closed and len(self._pool) < self.max_size:
                    self._pool.append(resource)
                elif self.cleanup:
                    self.cleanup(resource)
            
            self._semaphore.release()
    
    async def close(self) -> None:
        """Close the pool and cleanup all resources"""
        async with self._lock:
            self._closed = True
            
            # Cleanup pooled resources
            if self.cleanup:
                for resource in self._pool:
                    self.cleanup(resource)
            
            self._pool.clear()
            logger.info("AsyncResourcePool closed")
    
    @asynccontextmanager
    async def acquire_context(self):
        """Context manager for acquiring resources"""
        resource = await self.acquire()
        try:
            yield resource
        finally:
            await self.release(resource)


@dataclass
class AsyncTimedOperation:
    """Track timing information for async operations"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None


class AsyncOperationTracker:
    """
    Track and monitor async operations with timing and error information
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.operations: Dict[str, AsyncTimedOperation] = {}
        self.completed_operations: List[AsyncTimedOperation] = []
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def track_operation(self, name: str):
        """Context manager to track an async operation"""
        operation = AsyncTimedOperation(
            name=name,
            start_time=time.time()
        )
        
        async with self._lock:
            self.operations[name] = operation
        
        try:
            yield operation
            operation.end_time = time.time()
            logger.debug(f"Operation '{name}' completed in {operation.duration:.3f}s")
            
        except Exception as e:
            operation.end_time = time.time()
            operation.error = str(e)
            logger.error(f"Operation '{name}' failed after {operation.duration:.3f}s: {e}")
            raise
            
        finally:
            async with self._lock:
                if name in self.operations:
                    del self.operations[name]
                
                self.completed_operations.append(operation)
                
                # Maintain history limit
                if len(self.completed_operations) > self.max_history:
                    self.completed_operations = self.completed_operations[-self.max_history:]
    
    def get_active_operations(self) -> List[AsyncTimedOperation]:
        """Get list of currently active operations"""
        return list(self.operations.values())
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get statistics about operations"""
        if not self.completed_operations:
            return {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "average_duration": 0,
                "active_operations": len(self.operations)
            }
        
        successful = [op for op in self.completed_operations if not op.error]
        failed = [op for op in self.completed_operations if op.error]
        
        durations = [op.duration for op in successful if op.duration]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_operations": len(self.completed_operations),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "average_duration": avg_duration,
            "active_operations": len(self.operations)
        }


class AsyncFileManager:
    """
    Async file operations with proper resource management
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self._open_files: weakref.WeakSet = weakref.WeakSet()
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def open_file(self, path: str, mode: str = 'r', encoding: str = 'utf-8'):
        """
        Async context manager for file operations
        
        Args:
            path: Relative path to file
            mode: File open mode
            encoding: File encoding
            
        Yields:
            Async file handle
        """
        full_path = self.base_path / path
        
        # Ensure directory exists for write modes
        if 'w' in mode or 'a' in mode:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, mode=mode, encoding=encoding) as f:
            async with self._lock:
                self._open_files.add(f)
            
            try:
                yield f
            finally:
                async with self._lock:
                    self._open_files.discard(f)
    
    async def read_file(self, path: str, encoding: str = 'utf-8') -> str:
        """Read entire file content"""
        async with self.open_file(path, 'r', encoding) as f:
            return await f.read()
    
    async def write_file(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """Write content to file"""
        async with self.open_file(path, 'w', encoding) as f:
            await f.write(content)
    
    async def append_file(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """Append content to file"""
        async with self.open_file(path, 'a', encoding) as f:
            await f.write(content)
    
    def get_open_file_count(self) -> int:
        """Get number of currently open files"""
        return len(self._open_files)


class AsyncBatchProcessor:
    """
    Process items in batches with async execution
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_concurrent_batches: int = 5,
        timeout_per_batch: float = 60.0
    ):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.timeout_per_batch = timeout_per_batch
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        logger.info(
            f"AsyncBatchProcessor initialized: batch_size={batch_size}, "
            f"max_concurrent_batches={max_concurrent_batches}"
        )
    
    async def process_items(
        self,
        items: List[T],
        processor: Callable[[List[T]], Any],
        error_handler: Optional[Callable[[Exception, List[T]], None]] = None
    ) -> List[Any]:
        """
        Process items in batches
        
        Args:
            items: List of items to process
            processor: Async function to process a batch
            error_handler: Optional error handler for failed batches
            
        Returns:
            List of results from all batches
        """
        results = []
        tasks = []
        
        # Create batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        logger.info(f"Processing {len(items)} items in {len(batches)} batches")
        
        # Process batches concurrently
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch, processor, error_handler))
            tasks.append(task)
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {result}")
                if error_handler:
                    error_handler(result, batches[i])
            else:
                results.extend(result if isinstance(result, list) else [result])
        
        return results
    
    async def _process_batch(
        self,
        batch: List[T],
        processor: Callable[[List[T]], Any],
        error_handler: Optional[Callable[[Exception, List[T]], None]]
    ) -> Any:
        """Process a single batch with semaphore control"""
        async with self._semaphore:
            try:
                # Apply timeout to batch processing
                result = await asyncio.wait_for(
                    processor(batch),
                    timeout=self.timeout_per_batch
                )
                return result
                
            except asyncio.TimeoutError as e:
                logger.error(f"Batch processing timeout after {self.timeout_per_batch}s")
                if error_handler:
                    error_handler(e, batch)
                raise
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                if error_handler:
                    error_handler(e, batch)
                raise


class AsyncRateLimiter:
    """
    Async rate limiter using token bucket algorithm
    """
    
    def __init__(self, rate: float, burst: int):
        """
        Initialize rate limiter
        
        Args:
            rate: Number of requests per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()
        
        logger.info(f"AsyncRateLimiter initialized: rate={rate}/s, burst={burst}")
    
    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens, waiting if necessary
        
        Args:
            tokens: Number of tokens to acquire
        """
        if tokens > self.burst:
            raise ValueError(f"Cannot acquire {tokens} tokens, burst size is {self.burst}")
        
        while True:
            async with self._lock:
                now = time.time()
                
                # Add tokens based on time elapsed
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                
                # Calculate wait time
                wait_time = (tokens - self.tokens) / self.rate
            
            # Wait before retrying
            await asyncio.sleep(wait_time)
    
    @asynccontextmanager
    async def limit(self, tokens: int = 1):
        """Context manager for rate limiting"""
        await self.acquire(tokens)
        yield


# Global instances
_operation_tracker: Optional[AsyncOperationTracker] = None


def get_operation_tracker() -> AsyncOperationTracker:
    """Get global operation tracker instance"""
    global _operation_tracker
    if _operation_tracker is None:
        _operation_tracker = AsyncOperationTracker()
    return _operation_tracker