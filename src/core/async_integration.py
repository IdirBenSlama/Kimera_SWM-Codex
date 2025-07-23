"""
Async Integration for KIMERA System
Integrates new async patterns into existing KIMERA components
Phase 2, Week 5: Async/Await Patterns Implementation
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import functools

from .task_manager import TaskManager, get_task_manager
from .async_context_managers import (
    AsyncResourcePool, AsyncOperationTracker, AsyncFileManager,
    AsyncBatchProcessor, AsyncRateLimiter, get_operation_tracker
)
from .async_performance_monitor import AsyncPerformanceMonitor, get_performance_monitor
from .async_utils import (
    run_in_thread, make_async, ensure_async, AsyncTimer,
    gather_with_timeout, retry_async
)

logger = logging.getLogger(__name__)


class AsyncKimeraIntegration:
    """
    Integration layer for async patterns in KIMERA system
    """
    
    def __init__(self):
        self.task_manager: Optional[TaskManager] = None
        self.performance_monitor: Optional[AsyncPerformanceMonitor] = None
        self.operation_tracker: Optional[AsyncOperationTracker] = None
        self.file_manager: Optional[AsyncFileManager] = None
        self.batch_processor: Optional[AsyncBatchProcessor] = None
        
        # Resource pools
        self.db_connection_pool: Optional[AsyncResourcePool] = None
        self.gpu_resource_pool: Optional[AsyncResourcePool] = None
        
        # Rate limiters for external APIs
        self.api_rate_limiters: Dict[str, AsyncRateLimiter] = {}
        
        self._initialized = False
        
        logger.info("AsyncKimeraIntegration created")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize all async components
        
        Args:
            config: Optional configuration dictionary
        """
        if self._initialized:
            logger.warning("AsyncKimeraIntegration already initialized")
            return
        
        config = config or {}
        
        # Initialize task manager
        self.task_manager = get_task_manager()
        logger.info("Task manager initialized")
        
        # Initialize performance monitor
        self.performance_monitor = get_performance_monitor()
        await self.performance_monitor.start()
        logger.info("Performance monitor started")
        
        # Initialize operation tracker
        self.operation_tracker = get_operation_tracker()
        logger.info("Operation tracker initialized")
        
        # Initialize file manager
        project_root = config.get("project_root", Path.cwd())
        self.file_manager = AsyncFileManager(project_root)
        logger.info(f"File manager initialized with root: {project_root}")
        
        # Initialize batch processor
        self.batch_processor = AsyncBatchProcessor(
            batch_size=config.get("batch_size", 100),
            max_concurrent_batches=config.get("max_concurrent_batches", 5)
        )
        logger.info("Batch processor initialized")
        
        # Initialize resource pools
        await self._initialize_resource_pools(config)
        
        # Initialize rate limiters
        self._initialize_rate_limiters(config)
        
        self._initialized = True
        logger.info("AsyncKimeraIntegration fully initialized")
    
    async def _initialize_resource_pools(self, config: Dict[str, Any]) -> None:
        """Initialize resource pools"""
        # Database connection pool
        db_pool_size = config.get("db_pool_size", 20)
        self.db_connection_pool = AsyncResourcePool(
            factory=lambda: None,  # Placeholder - should be actual DB connection factory
            max_size=db_pool_size,
            timeout=30.0
        )
        logger.info(f"Database connection pool initialized with size: {db_pool_size}")
        
        # GPU resource pool
        gpu_pool_size = config.get("gpu_pool_size", 4)
        self.gpu_resource_pool = AsyncResourcePool(
            factory=lambda: None,  # Placeholder - should be actual GPU resource factory
            max_size=gpu_pool_size,
            timeout=60.0
        )
        logger.info(f"GPU resource pool initialized with size: {gpu_pool_size}")
    
    def _initialize_rate_limiters(self, config: Dict[str, Any]) -> None:
        """Initialize API rate limiters"""
        rate_limits = config.get("rate_limits", {})
        
        # Default rate limiters
        default_limits = {
            "openai": (10, 10),  # 10 requests per second, burst of 10
            "cryptopanic": (5, 5),  # 5 requests per second, burst of 5
            "external_api": (20, 30)  # 20 requests per second, burst of 30
        }
        
        for api_name, (rate, burst) in {**default_limits, **rate_limits}.items():
            self.api_rate_limiters[api_name] = AsyncRateLimiter(rate=rate, burst=burst)
            logger.info(f"Rate limiter for '{api_name}' initialized: {rate}/s, burst={burst}")
    
    async def shutdown(self) -> None:
        """Shutdown all async components"""
        logger.info("AsyncKimeraIntegration shutdown initiated")
        
        # Shutdown task manager
        if self.task_manager:
            await self.task_manager.shutdown()
        
        # Stop performance monitor
        if self.performance_monitor:
            await self.performance_monitor.stop()
        
        # Close resource pools
        if self.db_connection_pool:
            await self.db_connection_pool.close()
        
        if self.gpu_resource_pool:
            await self.gpu_resource_pool.close()
        
        self._initialized = False
        logger.info("AsyncKimeraIntegration shutdown complete")
    
    # Convenience methods for common operations
    
    async def run_with_monitoring(self, operation_name: str, coro):
        """Run a coroutine with performance monitoring"""
        async with self.performance_monitor.track_operation(operation_name):
            async with self.operation_tracker.track_operation(operation_name):
                return await coro
    
    async def run_with_rate_limit(self, api_name: str, coro):
        """Run a coroutine with rate limiting"""
        if api_name not in self.api_rate_limiters:
            logger.warning(f"No rate limiter for '{api_name}', proceeding without limit")
            return await coro
        
        async with self.api_rate_limiters[api_name].limit():
            return await coro
    
    async def process_batch_with_monitoring(
        self,
        items: List[Any],
        processor: callable,
        operation_name: str
    ) -> List[Any]:
        """Process items in batches with monitoring"""
        async def monitored_processor(batch):
            async with self.performance_monitor.track_operation(f"{operation_name}_batch"):
                return await ensure_async(processor)(batch)
        
        return await self.batch_processor.process_items(
            items=items,
            processor=monitored_processor
        )


# Decorators for easy integration

def with_task_management(name: str, replace_existing: bool = True):
    """Decorator to manage async tasks"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            task_manager = get_task_manager()
            return await task_manager.create_managed_task(
                name=name,
                coro=func(*args, **kwargs),
                replace_existing=replace_existing
            )
        return wrapper
    return decorator


def with_performance_monitoring(operation_name: Optional[str] = None):
    """Decorator to add performance monitoring to async functions"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            monitor = get_performance_monitor()
            async with monitor.track_operation(name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def with_operation_tracking(operation_name: Optional[str] = None):
    """Decorator to add operation tracking to async functions"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            tracker = get_operation_tracker()
            async with tracker.track_operation(name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to add retry logic to async functions"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(
                func,
                *args,
                max_attempts=max_attempts,
                delay=delay,
                backoff=backoff,
                **kwargs
            )
        return wrapper
    return decorator


def with_timeout(timeout: float):
    """Decorator to add timeout to async functions"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout
            )
        return wrapper
    return decorator


# Example integration functions

async def initialize_kimera_with_async() -> AsyncKimeraIntegration:
    """
    Initialize KIMERA system with async patterns
    
    Returns:
        Configured AsyncKimeraIntegration instance
    """
    integration = AsyncKimeraIntegration()
    
    # Load configuration (this should come from actual config)
    config = {
        "project_root": Path.cwd(),
        "db_pool_size": 20,
        "gpu_pool_size": 4,
        "batch_size": 100,
        "max_concurrent_batches": 5,
        "rate_limits": {
            "openai": (10, 10),
            "cryptopanic": (5, 5)
        }
    }
    
    await integration.initialize(config)
    
    return integration


@with_performance_monitoring("kimera_parallel_init")
async def parallel_component_initialization(components: List[str]) -> Dict[str, Any]:
    """
    Initialize KIMERA components in parallel
    
    Args:
        components: List of component names to initialize
        
    Returns:
        Dictionary of initialized components
    """
    async def init_component(name: str) -> tuple:
        logger.info(f"Initializing component: {name}")
        
        # Simulate component initialization
        # In real implementation, this would call actual init functions
        await asyncio.sleep(0.1)  # Simulate work
        
        return name, f"{name}_initialized"
    
    # Initialize all components in parallel
    tasks = [init_component(comp) for comp in components]
    results = await gather_with_timeout(*tasks, timeout=30.0)
    
    # Convert to dictionary
    initialized_components = dict(results)
    
    logger.info(f"Initialized {len(initialized_components)} components in parallel")
    return initialized_components


# Global integration instance
_async_integration: Optional[AsyncKimeraIntegration] = None


async def get_async_integration() -> AsyncKimeraIntegration:
    """Get or create global async integration instance"""
    global _async_integration
    if _async_integration is None:
        _async_integration = await initialize_kimera_with_async()
    return _async_integration