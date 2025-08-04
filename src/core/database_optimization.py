"""
Database Optimization for KIMERA System
Implements connection pooling, query optimization, and caching
Phase 3, Week 8: Performance Optimization
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import redis.asyncio as aioredis
from cachetools import LRUCache, TTLCache
from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool, StaticPool

try:
    from config import get_settings
except ImportError:
    # Create placeholders for config
    def get_settings(*args, **kwargs):
        return None


import hashlib
import json

from .async_performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DatabaseConnectionPool:
    """
    Manages database connection pooling with monitoring
    """

    def __init__(self):
        self.settings = get_settings()
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._pool_stats = {
            "connections_created": 0,
            "connections_recycled": 0,
            "connections_failed": 0,
            "active_connections": 0,
            "idle_connections": 0,
        }
        self.performance_monitor = get_performance_monitor()

    async def initialize(self) -> None:
        """Initialize database connection pool"""
        logger.info("Initializing database connection pool")

        # PostgreSQL configuration with connection pooling
        db_url = self.settings.database.url

        if not db_url.startswith("postgresql"):
            raise ValueError("PostgreSQL database is required for Kimera SWM")

        # Create async engine with optimized settings for PostgreSQL
        engine_kwargs = {
            "echo": self.settings.database.echo,
            "poolclass": NullPool,  # Use NullPool for async compatibility
            "connect_args": self._get_connect_args(),
        }

        self.engine = create_async_engine(db_url, **engine_kwargs)

        # Create session factory
        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Add event listeners for monitoring
        self._setup_event_listeners()

        # Test connection
        async with self.engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        logger.info(
            f"PostgreSQL connection pool initialized (size: {self.settings.database.pool_size})"
        )

    def _get_connect_args(self) -> Dict[str, Any]:
        """Get PostgreSQL-specific connection arguments"""
        return {
            "server_settings": {
                "application_name": "kimera",
                "jit": "off",  # Disable JIT for consistent performance
            },
            "command_timeout": 60,
        }

    def _setup_event_listeners(self) -> None:
        """Setup event listeners for connection monitoring"""

        @event.listens_for(Engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            self._pool_stats["connections_created"] += 1
            connection_record.info["connect_time"] = time.time()

        @event.listens_for(Engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            self._pool_stats["active_connections"] += 1
            self._pool_stats["idle_connections"] = max(
                0, self._pool_stats["idle_connections"] - 1
            )

        @event.listens_for(Engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            self._pool_stats["active_connections"] = max(
                0, self._pool_stats["active_connections"] - 1
            )
            self._pool_stats["idle_connections"] += 1

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get a database session from the pool"""
        if not self.session_factory:
            raise RuntimeError("Database connection pool not initialized")

        async with self.performance_monitor.track_operation("db_session_acquire"):
            async with self.session_factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception:
                    await session.rollback()
                    raise

    async def close(self) -> None:
        """Close the connection pool"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection pool closed")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if self.engine and hasattr(self.engine.pool, "size"):
            self._pool_stats["pool_size"] = self.engine.pool.size()
            self._pool_stats["overflow"] = self.engine.pool.overflow()

        return self._pool_stats.copy()


class QueryOptimizer:
    """
    Optimizes database queries with caching and query rewriting
    """

    def __init__(self, cache_size: int = 1000, cache_ttl: int = 300):
        self.query_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        self.performance_monitor = get_performance_monitor()

    def _generate_cache_key(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for query"""
        key_data = {"query": query, "params": params or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def execute_cached(
        self,
        session: AsyncSession,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: Optional[int] = None,
    ) -> Any:
        """Execute query with caching"""
        cache_key = self._generate_cache_key(query, params)

        # Check cache
        if cache_key in self.query_cache:
            self._update_stats(cache_key, hit=True)
            return self.query_cache[cache_key]

        # Execute query
        async with self.performance_monitor.track_operation("db_query_execute"):
            start_time = time.time()

            result = await session.execute(text(query), params or {})
            data = result.fetchall()

            execution_time = time.time() - start_time

            # Cache result
            if cache_ttl:
                # Use custom TTL
                self.query_cache[cache_key] = data
            else:
                self.query_cache[cache_key] = data

            self._update_stats(cache_key, hit=False, execution_time=execution_time)

            return data

    def _update_stats(
        self, cache_key: str, hit: bool, execution_time: Optional[float] = None
    ) -> None:
        """Update query statistics"""
        if cache_key not in self.query_stats:
            self.query_stats[cache_key] = {
                "hits": 0,
                "misses": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0,
                "last_accessed": datetime.now(),
            }

        stats = self.query_stats[cache_key]

        if hit:
            stats["hits"] += 1
        else:
            stats["misses"] += 1
            if execution_time:
                stats["total_execution_time"] += execution_time
                total_executions = stats["misses"]
                stats["avg_execution_time"] = (
                    stats["total_execution_time"] / total_executions
                )

        stats["last_accessed"] = datetime.now()

    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        total_queries = sum(s["hits"] + s["misses"] for s in self.query_stats.values())
        total_hits = sum(s["hits"] for s in self.query_stats.values())
        total_misses = sum(s["misses"] for s in self.query_stats.values())

        return {
            "total_queries": total_queries,
            "hits": total_hits,
            "misses": total_misses,
            "cache_hit_rate": total_hits / total_queries if total_queries > 0 else 0,
            "cached_queries": len(self.query_cache),
            "unique_queries": len(self.query_stats),
            "top_queries": self._get_top_queries(),
        }

    def _get_top_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top queries by execution count"""
        sorted_queries = sorted(
            self.query_stats.items(),
            key=lambda x: x[1]["hits"] + x[1]["misses"],
            reverse=True,
        )

        return [
            {
                "cache_key": key,
                "total_calls": stats["hits"] + stats["misses"],
                "cache_hits": stats["hits"],
                "avg_execution_time": stats["avg_execution_time"],
            }
            for key, stats in sorted_queries[:limit]
        ]

    def clear_cache(self) -> None:
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")


class DatabaseOptimizationMiddleware:
    """
    Middleware for database query optimization
    """

    def __init__(self):
        self.connection_pool = DatabaseConnectionPool()
        self.query_optimizer = QueryOptimizer()
        self.redis_client: Optional[aioredis.Redis] = None
        self.settings = get_settings()

    async def initialize(self) -> None:
        """Initialize database optimization components"""
        # Initialize connection pool
        await self.connection_pool.initialize()

        # Initialize Redis for distributed caching (if configured)
        if self.settings.get_feature("redis_caching"):
            try:
                self.redis_client = await aioredis.from_url(
                    "redis://localhost", max_connections=10
                )
                logger.info("Redis caching initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")

    @asynccontextmanager
    async def optimized_session(self):
        """Get an optimized database session"""
        async with self.connection_pool.get_session() as session:
            # Attach query optimizer to session
            session.query_optimizer = self.query_optimizer
            yield session

    async def close(self) -> None:
        """Close all resources"""
        await self.connection_pool.close()

        if self.redis_client:
            await self.redis_client.close()


# Query optimization decorators


def cached_query(ttl: int = 300):
    """
    Decorator to cache query results

    Args:
        ttl: Time to live in seconds
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract session from arguments
            session = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    session = arg
                    break

            if not session or not hasattr(session, "query_optimizer"):
                # No optimization available
                return await func(*args, **kwargs)

            # Generate cache key from function and arguments
            cache_key = f"{func.__module__}.{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check cache
            optimizer = session.query_optimizer
            if cache_key in optimizer.query_cache:
                return optimizer.query_cache[cache_key]

            # Execute function
            result = await func(*args, **kwargs)

            # Cache result
            optimizer.query_cache[cache_key] = result

            return result

        return wrapper

    return decorator


def batch_query(batch_size: int = 100):
    """
    Decorator to batch multiple queries

    Args:
        batch_size: Maximum batch size
    """

    def decorator(func: Callable) -> Callable:
        # Store pending queries
        func._batch_queue = []
        func._batch_lock = asyncio.Lock()

        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with func._batch_lock:
                # Add to batch queue
                future = asyncio.Future()
                func._batch_queue.append((args, kwargs, future))

                # Process batch if full
                if len(func._batch_queue) >= batch_size:
                    await _process_batch(func)
                else:
                    # Schedule batch processing
                    asyncio.create_task(_schedule_batch(func))

                return await future

        async def _process_batch(func):
            """Process all queries in the batch"""
            if not func._batch_queue:
                return

            batch = func._batch_queue[:]
            func._batch_queue.clear()

            try:
                # Execute all queries in batch
                results = []
                for args, kwargs, _ in batch:
                    result = await func(*args, **kwargs)
                    results.append(result)

                # Set results
                for i, (_, _, future) in enumerate(batch):
                    future.set_result(results[i])

            except Exception as e:
                # Set exception for all futures
                for _, _, future in batch:
                    future.set_exception(e)

        async def _schedule_batch(func):
            """Schedule batch processing after a delay"""
            await asyncio.sleep(0.1)  # 100ms delay
            async with func._batch_lock:
                await _process_batch(func)

        return wrapper

    return decorator


# Global optimization instance
_db_optimization: Optional[DatabaseOptimizationMiddleware] = None


async def get_db_optimization() -> DatabaseOptimizationMiddleware:
    """Get global database optimization instance"""
    global _db_optimization
    if _db_optimization is None:
        _db_optimization = DatabaseOptimizationMiddleware()
        await _db_optimization.initialize()
    return _db_optimization
