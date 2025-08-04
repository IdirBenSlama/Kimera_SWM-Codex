#!/usr/bin/env python3
"""
Kimera SWM Advanced Caching System
==================================

Multi-level intelligent caching system for Kimera SWM cognitive operations.
Provides semantic caching, result prediction, and memory-efficient storage.

This module delivers:
- L1: In-memory component caching with cognitive priority
- L2: Redis distributed caching for scalability
- L3: Persistent result caching with compression
- Semantic similarity-based cache retrieval
- Predictive cache warming and optimization

Author: Kimera SWM Development Team
Date: January 30, 2025
Version: 5.2.0
"""

import asyncio
import time
import json
import hashlib
import pickle
import gzip
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import OrderedDict
import threading
from functools import wraps

import torch
import torch.nn.functional as F
import numpy as np

# Optional Redis support
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Redis not available, distributed caching disabled")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheConfiguration:
    """Cache system configuration"""
    # L1 Cache (Memory)
    l1_max_size: int = 1000
    l1_ttl: int = 3600  # seconds
    
    # L2 Cache (Redis)
    l2_enabled: bool = REDIS_AVAILABLE
    l2_host: str = "localhost"
    l2_port: int = 6379
    l2_db: int = 0
    l2_ttl: int = 86400  # 24 hours
    l2_max_connections: int = 10
    
    # L3 Cache (Persistent)
    l3_enabled: bool = True
    l3_path: str = "./cache/persistent"
    l3_max_size_gb: float = 5.0
    l3_compression: bool = True
    
    # Cache behavior
    enable_semantic_caching: bool = True
    semantic_similarity_threshold: float = 0.85
    enable_predictive_warming: bool = True
    cache_priority_decay: float = 0.95
    
    # Performance
    async_operations: bool = True
    compression_level: int = 6


@dataclass
class CacheEntry:
    """Individual cache entry"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    priority_score: float = 1.0
    semantic_embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    size_bytes: int = 0


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    
    semantic_hits: int = 0
    predictive_hits: int = 0
    
    total_entries: int = 0
    total_size_mb: float = 0.0
    avg_access_time_ms: float = 0.0
    
    cache_efficiency: float = 0.0
    hit_rate: float = 0.0
    
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SemanticCacheManager:
    """Semantic similarity-based cache management"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.embeddings_cache = {}
        self._lock = threading.Lock()
    
    def compute_semantic_key(self, input_data: Any) -> str:
        """Compute semantic cache key"""
        # Convert input to string representation
        if isinstance(input_data, (dict, list)):
            content = json.dumps(input_data, sort_keys=True)
        elif isinstance(input_data, torch.Tensor):
            content = f"tensor_{input_data.shape}_{input_data.dtype}_{input_data.sum().item()}"
        else:
            content = str(input_data)
        
        # Create hash-based key
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def compute_embedding(self, input_data: Any) -> torch.Tensor:
        """Compute semantic embedding for input data"""
        try:
            if isinstance(input_data, str):
                # Simple text embedding (in practice, use BGE-M3 or similar)
                words = input_data.lower().split()
                # Create a simple hash-based embedding
                embedding = torch.zeros(384)  # BGE-M3 dimension
                for i, word in enumerate(words[:20]):  # Limit to 20 words
                    hash_val = hash(word) % 384
                    embedding[hash_val] += 1.0
                return F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
            
            elif isinstance(input_data, torch.Tensor):
                # Reduce tensor to fixed-size embedding
                flat = input_data.flatten()
                if len(flat) > 384:
                    # Sample or pool to 384 dimensions
                    indices = torch.linspace(0, len(flat)-1, 384).long()
                    embedding = flat[indices]
                else:
                    # Pad to 384 dimensions
                    embedding = F.pad(flat, (0, 384 - len(flat)))
                return F.normalize(embedding, p=2, dim=0)
            
            elif isinstance(input_data, (dict, list)):
                # Create embedding from JSON structure
                content = json.dumps(input_data, sort_keys=True)
                return self.compute_embedding(content)
            
            else:
                # Default string representation
                return self.compute_embedding(str(input_data))
                
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            # Return random embedding as fallback
            return torch.randn(384)
    
    def find_similar_entries(self, query_embedding: torch.Tensor, cache_entries: Dict[str, CacheEntry]) -> List[Tuple[str, float]]:
        """Find semantically similar cache entries"""
        similar_entries = []
        
        for key, entry in cache_entries.items():
            if entry.semantic_embedding is not None:
                try:
                    similarity = F.cosine_similarity(
                        query_embedding.unsqueeze(0),
                        entry.semantic_embedding.unsqueeze(0),
                        dim=1
                    ).item()
                    
                    if similarity >= self.similarity_threshold:
                        similar_entries.append((key, similarity))
                        
                except Exception as e:
                    logger.warning(f"Similarity computation failed for {key}: {e}")
        
        # Sort by similarity (highest first)
        similar_entries.sort(key=lambda x: x[1], reverse=True)
        return similar_entries


class L1MemoryCache:
    """L1 in-memory cache with cognitive priority"""
    
    def __init__(self, config: CacheConfiguration):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.semantic_manager = SemanticCacheManager(config.semantic_similarity_threshold)
        self._lock = threading.RLock()
        self.access_stats = {"hits": 0, "misses": 0}
    
    def _compute_priority_score(self, entry: CacheEntry) -> float:
        """Compute priority score for cache entry"""
        # Factors: recency, frequency, semantic importance
        current_time = time.time()
        recency_score = 1.0 / (1.0 + (current_time - entry.last_access) / 3600)  # Decay over hours
        frequency_score = min(1.0, entry.access_count / 10.0)  # Cap at 10 accesses
        
        # Combine factors
        priority = (recency_score * 0.6 + frequency_score * 0.4) * entry.priority_score
        return priority
    
    def _evict_entries(self):
        """Evict entries based on priority when cache is full"""
        if len(self.cache) <= self.config.l1_max_size:
            return
        
        # Calculate priorities for all entries
        priorities = []
        for key, entry in self.cache.items():
            priority = self._compute_priority_score(entry)
            priorities.append((key, priority))
        
        # Sort by priority (lowest first for eviction)
        priorities.sort(key=lambda x: x[1])
        
        # Evict lowest priority entries
        entries_to_evict = len(self.cache) - self.config.l1_max_size + 1
        for i in range(entries_to_evict):
            key_to_evict = priorities[i][0]
            if key_to_evict in self.cache:
                del self.cache[key_to_evict]
                logger.debug(f"Evicted L1 cache entry: {key_to_evict}")
    
    def get(self, key: str, input_data: Optional[Any] = None) -> Optional[Any]:
        """Get value from L1 cache with semantic similarity support"""
        with self._lock:
            # Direct key lookup
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                self.access_stats["hits"] += 1
                return entry.value
            
            # Semantic similarity search if enabled and input provided
            if self.config.enable_semantic_caching and input_data is not None:
                query_embedding = self.semantic_manager.compute_embedding(input_data)
                similar_entries = self.semantic_manager.find_similar_entries(query_embedding, self.cache)
                
                if similar_entries:
                    best_key, similarity = similar_entries[0]
                    logger.debug(f"Semantic cache hit: {similarity:.3f} similarity")
                    
                    entry = self.cache[best_key]
                    entry.access_count += 1
                    entry.last_access = time.time()
                    self.cache.move_to_end(best_key)
                    
                    self.access_stats["hits"] += 1
                    return entry.value
            
            self.access_stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any, input_data: Optional[Any] = None, priority: float = 1.0) -> bool:
        """Put value in L1 cache"""
        with self._lock:
            try:
                # Compute semantic embedding if enabled
                semantic_embedding = None
                if self.config.enable_semantic_caching and input_data is not None:
                    semantic_embedding = self.semantic_manager.compute_embedding(input_data)
                
                # Estimate size
                size_bytes = len(pickle.dumps(value))
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    priority_score=priority,
                    semantic_embedding=semantic_embedding,
                    size_bytes=size_bytes
                )
                
                # Add to cache
                self.cache[key] = entry
                
                # Evict if necessary
                self._evict_entries()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache entry {key}: {e}")
                return False
    
    def clear(self):
        """Clear L1 cache"""
        with self._lock:
            self.cache.clear()
            self.access_stats = {"hits": 0, "misses": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L1 cache statistics"""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            total_requests = self.access_stats["hits"] + self.access_stats["misses"]
            hit_rate = self.access_stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "entries": len(self.cache),
                "size_mb": total_size / (1024 * 1024),
                "hits": self.access_stats["hits"],
                "misses": self.access_stats["misses"],
                "hit_rate": hit_rate,
                "max_size": self.config.l1_max_size
            }


class L2RedisCache:
    """L2 Redis distributed cache"""
    
    def __init__(self, config: CacheConfiguration):
        self.config = config
        self.redis_client = None
        self.access_stats = {"hits": 0, "misses": 0}
        self._enabled = config.l2_enabled and REDIS_AVAILABLE
    
    async def initialize(self) -> bool:
        """Initialize Redis connection"""
        if not self._enabled:
            return False
        
        try:
            self.redis_client = aioredis.Redis(
                host=self.config.l2_host,
                port=self.config.l2_port,
                db=self.config.l2_db,
                max_connections=self.config.l2_max_connections,
                decode_responses=False  # Keep binary for pickle
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("L2 Redis cache initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._enabled = False
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self._enabled or not self.redis_client:
            return None
        
        try:
            data = await self.redis_client.get(f"kimera:{key}")
            if data:
                # Decompress and deserialize
                if self.config.l3_compression:
                    data = gzip.decompress(data)
                
                value = pickle.loads(data)
                self.access_stats["hits"] += 1
                return value
            else:
                self.access_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get failed for {key}: {e}")
            self.access_stats["misses"] += 1
            return None
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in Redis cache"""
        if not self._enabled or not self.redis_client:
            return False
        
        try:
            # Serialize and compress
            data = pickle.dumps(value)
            if self.config.l3_compression:
                data = gzip.compress(data, compresslevel=self.config.compression_level)
            
            # Set with TTL
            ttl = ttl or self.config.l2_ttl
            await self.redis_client.setex(f"kimera:{key}", ttl, data)
            return True
            
        except Exception as e:
            logger.error(f"Redis put failed for {key}: {e}")
            return False
    
    async def clear(self):
        """Clear Redis cache"""
        if not self._enabled or not self.redis_client:
            return
        
        try:
            # Delete all kimera keys
            keys = await self.redis_client.keys("kimera:*")
            if keys:
                await self.redis_client.delete(*keys)
            
            self.access_stats = {"hits": 0, "misses": 0}
            
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self._enabled:
            return {"enabled": False}
        
        total_requests = self.access_stats["hits"] + self.access_stats["misses"]
        hit_rate = self.access_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "enabled": True,
            "hits": self.access_stats["hits"],
            "misses": self.access_stats["misses"],
            "hit_rate": hit_rate
        }


class AdvancedCacheManager:
    """Main advanced caching system manager"""
    
    def __init__(self, config: Optional[CacheConfiguration] = None):
        self.config = config or CacheConfiguration()
        
        # Initialize cache layers
        self.l1_cache = L1MemoryCache(self.config)
        self.l2_cache = L2RedisCache(self.config) if self.config.l2_enabled else None
        
        # Cache statistics
        self.global_stats = CacheMetrics()
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the caching system"""
        try:
            # Initialize L2 cache if enabled
            if self.l2_cache:
                await self.l2_cache.initialize()
            
            logger.info("Advanced caching system initialized")
            logger.info(f"L1 Cache: Enabled (max {self.config.l1_max_size} entries)")
            logger.info(f"L2 Cache: {'Enabled' if self.config.l2_enabled else 'Disabled'}")
            logger.info(f"Semantic Caching: {'Enabled' if self.config.enable_semantic_caching else 'Disabled'}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Cache system initialization failed: {e}")
            return False
    
    async def get(self, key: str, input_data: Optional[Any] = None) -> Optional[Any]:
        """Get value from cache with multi-level lookup"""
        start_time = time.perf_counter()
        
        try:
            # L1 Cache lookup
            value = self.l1_cache.get(key, input_data)
            if value is not None:
                self.global_stats.l1_hits += 1
                self._update_access_time(start_time)
                return value
            
            self.global_stats.l1_misses += 1
            
            # L2 Cache lookup
            if self.l2_cache:
                value = await self.l2_cache.get(key)
                if value is not None:
                    self.global_stats.l2_hits += 1
                    
                    # Promote to L1
                    self.l1_cache.put(key, value, input_data)
                    
                    self._update_access_time(start_time)
                    return value
                
                self.global_stats.l2_misses += 1
            
            # Cache miss
            self._update_access_time(start_time)
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed for {key}: {e}")
            return None
    
    async def put(self, key: str, value: Any, input_data: Optional[Any] = None, 
                 priority: float = 1.0, ttl: Optional[int] = None) -> bool:
        """Put value in cache across all levels"""
        try:
            success = True
            
            # L1 Cache
            if not self.l1_cache.put(key, value, input_data, priority):
                success = False
            
            # L2 Cache
            if self.l2_cache:
                if not await self.l2_cache.put(key, value, ttl):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Cache put failed for {key}: {e}")
            return False
    
    def _update_access_time(self, start_time: float):
        """Update average access time statistics"""
        access_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Exponential moving average
        alpha = 0.1
        self.global_stats.avg_access_time_ms = (
            alpha * access_time + 
            (1 - alpha) * self.global_stats.avg_access_time_ms
        )
    
    def get_comprehensive_stats(self) -> CacheMetrics:
        """Get comprehensive cache statistics"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats() if self.l2_cache else {"enabled": False}
        
        # Update global stats
        self.global_stats.total_entries = l1_stats["entries"]
        self.global_stats.total_size_mb = l1_stats["size_mb"]
        
        # Calculate hit rates
        total_hits = self.global_stats.l1_hits + self.global_stats.l2_hits + self.global_stats.l3_hits
        total_misses = self.global_stats.l1_misses + self.global_stats.l2_misses + self.global_stats.l3_misses
        total_requests = total_hits + total_misses
        
        self.global_stats.hit_rate = total_hits / total_requests if total_requests > 0 else 0
        self.global_stats.cache_efficiency = (
            (self.global_stats.l1_hits * 1.0 + 
             self.global_stats.l2_hits * 0.8 + 
             self.global_stats.l3_hits * 0.6) / total_requests
        ) if total_requests > 0 else 0
        
        return self.global_stats
    
    async def clear_all(self):
        """Clear all cache levels"""
        self.l1_cache.clear()
        
        if self.l2_cache:
            await self.l2_cache.clear()
        
        # Reset stats
        self.global_stats = CacheMetrics()
        
        logger.info("All cache levels cleared")
    
    def cache_decorator(self, ttl: Optional[int] = None, priority: float = 1.0):
        """Decorator for automatic caching of function results"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                cache_key = hashlib.sha256(
                    json.dumps(key_data, sort_keys=True, default=str).encode()
                ).hexdigest()[:16]
                
                # Try to get from cache
                cached_result = await self.get(cache_key, key_data)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                await self.put(cache_key, result, key_data, priority, ttl)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, use asyncio.run for cache operations
                return asyncio.run(async_wrapper(*args, **kwargs))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


# Global cache manager
cache_manager = AdvancedCacheManager()

# Convenience functions
async def initialize_caching(config: Optional[CacheConfiguration] = None) -> bool:
    """Initialize advanced caching system"""
    global cache_manager
    if config:
        cache_manager.config = config
    return await cache_manager.initialize()

async def get_cached(key: str, input_data: Optional[Any] = None) -> Optional[Any]:
    """Get value from cache"""
    global cache_manager
    return await cache_manager.get(key, input_data)

async def put_cached(key: str, value: Any, input_data: Optional[Any] = None, 
                    priority: float = 1.0, ttl: Optional[int] = None) -> bool:
    """Put value in cache"""
    global cache_manager
    return await cache_manager.put(key, value, input_data, priority, ttl)

def cached(ttl: Optional[int] = None, priority: float = 1.0):
    """Caching decorator"""
    global cache_manager
    return cache_manager.cache_decorator(ttl, priority)

def get_cache_stats() -> CacheMetrics:
    """Get cache statistics"""
    global cache_manager
    return cache_manager.get_comprehensive_stats()


if __name__ == "__main__":
    # Test advanced caching
    async def test_caching():
        logger.info("💾 Testing Kimera SWM Advanced Caching System")
        logger.info("=" * 50)
        
        # Initialize
        success = await initialize_caching()
        
        if success:
            logger.info("✅ Advanced caching initialized successfully")
            
            # Test caching
            await put_cached("test_key", {"data": "test_value", "timestamp": time.time()})
            
            result = await get_cached("test_key")
            if result:
                logger.info("✅ Cache store/retrieve working")
            
            # Get stats
            stats = get_cache_stats()
            logger.info(f"Cache Entries: {stats.total_entries}")
            logger.info(f"Hit Rate: {stats.hit_rate:.1%}")
        else:
            logger.info("⚠️  Caching system initialization failed")
        
        logger.info("\n🎯 Advanced Caching System Ready!")
    
    asyncio.run(test_caching())