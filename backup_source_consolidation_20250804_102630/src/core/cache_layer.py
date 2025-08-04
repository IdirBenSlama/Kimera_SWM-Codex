"""
Cache Layer for KIMERA System
Provides multi-tier caching (in-process LRU/TTL → optional Redis → optional disk)
Phase 3, Week 8: Performance Optimization
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Coroutine, TypeVar, Tuple
import asyncio
import logging
import time

import hashlib
import json
import pickle
try:
    import redis.asyncio as aioredis  # Use the new async redis client
except ImportError:  # Optional dependency
    aioredis = None

from cachetools import TTLCache, LRUCache

from src.config import get_settings, get_feature_flag

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ------------------------------------
# Dataclass helpers
# ------------------------------------


def _now() -> float:
    return time.time()


@dataclass
class CacheStats:
    """Statistics for a cache tier"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ------------------------------------
# Cache tiers
# ------------------------------------


class MemoryCache:
    """In-process LRU/TTL cache"""

    def __init__(self, maxsize: int = 2048, ttl: Optional[int] = None):
        self.ttl = ttl
        self.cache: Dict[str, Any]
        if ttl is not None:
            self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        else:
            self.cache = LRUCache(maxsize=maxsize)
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.stats.hits += 1
            return self.cache[key]
        self.stats.misses += 1
        return None

    def set(self, key: str, value: Any):
        prev = key in self.cache
        self.cache[key] = value
        self.stats.sets += 1
        if not prev and len(self.cache) == self.cache.maxsize:
            # rough evict count
            self.stats.evictions += 1

    def clear(self):
        self.cache.clear()


class DiskCache:
    """Very lightweight file-based cache (pickle) for large payloads."""

    def __init__(self, root: Path, ttl: int = 3600):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self.stats = CacheStats()

    def _path(self, key: str) -> Path:
        safe_key = key.replace("/", "_")
        return self.root / f"{safe_key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.exists():
            self.stats.misses += 1
            return None
        if _now() - path.stat().st_mtime > self.ttl:
            try:
                path.unlink()
            except OSError:
                pass
            self.stats.misses += 1
            return None
        try:
            with path.open("rb") as fh:
                self.stats.hits += 1
                return pickle.load(fh)
        except Exception as e:
            logger.warning(f"DiskCache read error: {e}")
            self.stats.misses += 1
            return None

    def set(self, key: str, value: Any):
        path = self._path(key)
        try:
            with path.open("wb") as fh:
                pickle.dump(value, fh)
            self.stats.sets += 1
        except Exception as e:
            logger.warning(f"DiskCache write error: {e}")

    def clear(self):
        for f in self.root.glob("*.pkl"):
            try:
                f.unlink()
            except OSError:
                pass


class RedisCache:
    """Async Redis cache wrapper (optional)."""

    def __init__(self, client: Any, ttl: int = 3600):  # Use Any to avoid type issues
        self.client = client
        self.ttl = ttl
        self.stats = CacheStats()

    async def get(self, key: str) -> Optional[Any]:
        try:
            data = await self.client.get(key)
            if data is None:
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return pickle.loads(data)
        except Exception as e:
            logger.warning(f"RedisCache get error: {e}")
            self.stats.misses += 1
            return None

    async def set(self, key: str, value: Any):
        try:
            await self.client.set(key, pickle.dumps(value), ex=self.ttl)  # Use 'ex' instead of 'expire'
            self.stats.sets += 1
        except Exception as e:
            logger.warning(f"RedisCache set error: {e}")


# ------------------------------------
# Multi-tier Cache Manager
# ------------------------------------


class CacheManager:
    """High-level cache manager orchestrating multiple cache tiers."""

    def __init__(self):
        settings = get_settings()
        perf = settings.performance

        self.memory_cache = MemoryCache(maxsize=perf.cache_size, ttl=3600)
        self.disk_cache: Optional[DiskCache] = None
        self.redis_cache: Optional[RedisCache] = None

        # Enable disk cache if feature flag
        if get_feature_flag("disk_cache"):
            self.disk_cache = DiskCache(root=settings.paths.temp_dir / "cache", ttl=86400)
        # Enable redis if feature flag and library available
        if get_feature_flag("redis_cache") and aioredis is not None:
            self.redis_cache = None  # async connect later

    async def initialize(self):
        if self.redis_cache is None and get_feature_flag("redis_cache") and aioredis is not None:
            try:
                client = await aioredis.from_url("redis://localhost", decode_responses=False)
                self.redis_cache = RedisCache(client)
                logger.info("RedisCache initialized")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")

    # ------------ basic get/set -------------

    async def get(self, key: str) -> Optional[Any]:
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        # Redis
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                self.memory_cache.set(key, value)
                return value
        # Disk
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                self.memory_cache.set(key, value)
                return value
        return None

    async def set(self, key: str, value: Any):
        self.memory_cache.set(key, value)
        if self.redis_cache:
            await self.redis_cache.set(key, value)
        if self.disk_cache:
            self.disk_cache.set(key, value)

    # ------------- decorator ----------------

    def cached(self, ttl: int = 3600):
        """Decorator to cache coroutine results based on arguments."""

        def decorator(func: Callable[..., Coroutine[Any, Any, T]]):
            async def wrapper(*args, **kwargs) -> T:
                key_data: Tuple = (func.__module__, func.__qualname__, args, kwargs)
                key = hashlib.sha256(pickle.dumps(key_data)).hexdigest()
                cached_val = await self.get(key)
                if cached_val is not None:
                    return cached_val  # type: ignore
                result: T = await func(*args, **kwargs)
                # short ttl respect? we still store but memory tier may evict later
                await self.set(key, result)
                return result

            return wrapper

        return decorator

    # ------------- stats/report -------------

    def stats(self) -> Dict[str, Any]:
        data = {
            "memory": self.memory_cache.stats.__dict__,
        }
        if self.disk_cache:
            data["disk"] = self.disk_cache.stats.__dict__
        if self.redis_cache:
            data["redis"] = self.redis_cache.stats.__dict__
        return data


# Global instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
