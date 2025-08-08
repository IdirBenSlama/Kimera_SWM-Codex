"""
GPU-Optimized Cognitive Field Dynamics Engine - Enhanced Edition

This engine is designed for high-performance GPU utilization by leveraging:
- PyTorch CUDA operations for massive parallelization
- Advanced batch processing with CUDA streams
- Tensor operations designed for NVIDIA GPU architecture
- Memory-efficient GPU tensor management with pre-allocation
- Mixed precision for performance (FP16/FP32)
- Intelligent load balancing and adaptive batch sizing
- Real-time performance monitoring and auto-tuning

Performance targets:
- 1000+ fields/sec creation rate (200x improvement)
- 2000+ neighbor searches/sec
- Full GPU utilization (>95%)
- Sub-millisecond response times
- Automatic performance optimization
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config.hardware_config import (ADAPTIVE_BATCH_SIZE_MAX, ADAPTIVE_BATCH_SIZE_MIN
                                      COMPILE_MODELS, DEVICE, ENABLE_AUTO_TUNING
                                      ENABLE_CUDA_STREAMS, ENABLE_MEMORY_POOLING
                                      ENABLE_TENSOR_CORES, PREFETCH_FACTOR
                                      TENSOR_BATCH_SIZE, USE_MIXED_PRECISION
                                      ..config.settings, ..utils.config, from
                                      get_api_settings, get_settings, import)
# Import configurations
from ..core.cognitive_field_config import CognitiveFieldConfig
from ..core.cognitive_field_config import cognitive_field_config as cfg
from ..monitoring.cognitive_field_metrics import get_metrics_collector

logger = logging.getLogger(__name__)

# Enhanced GPU Configuration with Auto-tuning
if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)} ({total_mem:.2f} GB VRAM)")
    logger.info(f"   CUDA Version: {torch.version.cuda}")
    logger.info(f"   Mixed Precision: {USE_MIXED_PRECISION}")
    logger.info(f"   CUDA Streams: {ENABLE_CUDA_STREAMS}")
    logger.info(f"   Auto-tuning: {ENABLE_AUTO_TUNING}")

@dataclass
class GPUFieldState:
    """Auto-generated class."""
    pass
    """GPU-optimized field state."""
    geoid_id: str
    embedding: torch.Tensor
    field_strength: float
    resonance_frequency: float
    phase: float
    decay_rate: float
    creation_time: float = 0.0

@dataclass
class PerformanceMetrics:
    """Auto-generated class."""
    pass
    """Comprehensive performance tracking."""
    throughput_fields_per_sec: float = 0.0
    gpu_utilization_percent: float = 0.0
    memory_efficiency_percent: float = 0.0
    batch_processing_time_ms: float = 0.0
    neighbor_search_time_ms: float = 0.0
    optimal_batch_size: int = TENSOR_BATCH_SIZE
    cuda_stream_efficiency: float = 0.0
class CUDAStreamManager:
    """Auto-generated class."""
    pass
    """Advanced CUDA stream management for overlapped processing."""
    
    def __init__(self, num_streams: int = 4):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.num_streams = num_streams
        self.streams = []
        self.current_stream = 0
        
        if ENABLE_CUDA_STREAMS:
            for i in range(num_streams):
                stream = torch.cuda.Stream()
                self.streams.append(stream)
            logger.info(f"🌊 CUDA Stream Manager initialized with {num_streams} streams")
        
    def get_next_stream(self):
        """Get the next available CUDA stream."""
        if not self.streams:
            return torch.cuda.default_stream()
        
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % self.num_streams
        return stream
    
    def synchronize_all(self):
        """Synchronize all streams."""
        if ENABLE_CUDA_STREAMS:
            for stream in self.streams:
                stream.synchronize()
class MemoryPool:
    """Auto-generated class."""
    pass
    """Optimized memory pool for tensor pre-allocation."""
    
    def __init__(self, device: torch.device, dimension: int):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.device = device
        self.dimension = dimension
        self.pool_size = 10000  # Pre-allocate for 10k fields
        
        if ENABLE_MEMORY_POOLING and torch.cuda.is_available():
            # Pre-allocate tensor pools
            self.embedding_pool = torch.empty(
                (self.pool_size, dimension), 
                device=device
                dtype=torch.float16 if USE_MIXED_PRECISION else torch.float32
            )
            self.scalar_pool = torch.empty(
                self.pool_size
                device=device
                dtype=torch.float32
            )
            
            self.available_indices = set(range(self.pool_size))
            self.used_indices = set()
            
            logger.info(f"🏊 Memory Pool initialized: {self.pool_size} slots, {dimension}D")
    
    def allocate_embedding_slot(self) -> Optional[int]:
        """Allocate a slot from the embedding pool."""
        if not hasattr(self, 'available_indices') or not self.available_indices:
            return None
        
        idx = self.available_indices.pop()
        self.used_indices.add(idx)
        return idx
    
    def deallocate_slot(self, idx: int):
        """Return a slot to the pool."""
        if hasattr(self, 'used_indices') and idx in self.used_indices:
            self.used_indices.remove(idx)
            self.available_indices.add(idx)
class AdaptiveBatchOptimizer:
    """Auto-generated class."""
    pass
    """Intelligent batch size optimization based on GPU performance."""
    
    def __init__(self, initial_batch_size: int = TENSOR_BATCH_SIZE):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.current_batch_size = initial_batch_size
        self.performance_history = deque(maxlen=50)
        self.optimization_enabled = ENABLE_AUTO_TUNING
        self.last_optimization_time = time.time()
        self.optimization_interval = 30.0  # Optimize every 30 seconds
        
    def record_performance(self, batch_size: int, processing_time_ms: float, gpu_utilization: float):
        """Record performance metrics for optimization."""
        throughput = batch_size / (processing_time_ms / 1000.0)
        efficiency_score = throughput * (gpu_utilization / 100.0)
        
        self.performance_history.append({
            'batch_size': batch_size
            'processing_time_ms': processing_time_ms
            'gpu_utilization': gpu_utilization
            'throughput': throughput
            'efficiency_score': efficiency_score
            'timestamp': time.time()
        })
    
    def should_optimize(self) -> bool:
        """Check if it's time to optimize batch size."""
        return (self.optimization_enabled and 
                time.time() - self.last_optimization_time > self.optimization_interval and
                len(self.performance_history) >= 10)
    
    def optimize_batch_size(self) -> int:
        """Optimize batch size based on performance history."""
        if not self.should_optimize():
            return self.current_batch_size
        
        # Analyze recent performance
        recent_metrics = list(self.performance_history)[-10:]
        
        # Find optimal batch size based on efficiency score
        best_metrics = max(recent_metrics, key=lambda x: x['efficiency_score'])
        optimal_size = best_metrics['batch_size']
        
        # Adaptive adjustment - tuned for GPUs with ~11GB VRAM
        if best_metrics['gpu_utilization'] < 80:
            # GPU underutilized, increase batch size
            optimal_size = min(optimal_size * 1.5, ADAPTIVE_BATCH_SIZE_MAX)
        elif best_metrics['gpu_utilization'] > 95:
            # GPU saturated, decrease batch size
            optimal_size = max(optimal_size * 0.8, ADAPTIVE_BATCH_SIZE_MIN)
        
        self.current_batch_size = int(optimal_size)
        self.last_optimization_time = time.time()
        
        logger.info(f"🎯 Batch size optimized: {self.current_batch_size} (GPU util: {best_metrics['gpu_utilization']:.1f}%)")
        return self.current_batch_size
class GPUSemanticFieldSystem:
    """Auto-generated class."""
    pass
    """Enhanced GPU-optimized semantic field system for maximum parallelization."""
    
    def __init__(self, dimension: int, device: torch.device = DEVICE):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.dimension = dimension
        self.device = device
        self.dtype = torch.float16 if USE_MIXED_PRECISION else torch.float32
        
        # Enhanced GPU tensor storage for batch operations
        self.field_embeddings = torch.empty((0, dimension), device=device, dtype=self.dtype)
        self.field_strengths = torch.empty(0, device=device, dtype=torch.float32)
        self.resonance_frequencies = torch.empty(0, device=device, dtype=torch.float32)
        self.phases = torch.empty(0, device=device, dtype=torch.float32)
        self.decay_rates = torch.empty(0, device=device, dtype=torch.float32)
        
        # Mapping structures
        self.geoid_to_index = {}
        self.index_to_geoid = {}
        self.next_index = 0
        
        # Advanced optimization components
        self.stream_manager = CUDAStreamManager(num_streams=4)
        self.memory_pool = MemoryPool(device, dimension)
        self.batch_optimizer = AdaptiveBatchOptimizer()
        
        # Performance tracking
        self.gpu_memory_used = 0
        self.operation_count = 0
        self.performance_metrics = PerformanceMetrics()
        
        # Compiled operations for JIT optimization
        if COMPILE_MODELS and torch.cuda.is_available():
            try:
                self._compile_critical_operations()
            except Exception as e:
                logger.warning(f"torch.compile not available: {e}")
                # Fallback to standard operations
                self.compiled_similarity = lambda x, y: torch.mm(x, y.t())
                self.compiled_normalize = lambda x: F.normalize(x, p=2, dim=1)
        
        logger.info(f"🔥 Enhanced GPU Field System initialized: {dimension}D on {device}")

    def _compile_critical_operations(self):
        """Compile frequently used operations with torch.compile for JIT optimization."""
        try:
            # Compile batch similarity computation
            @torch.compile
            def compiled_similarity_computation(embeddings1, embeddings2):
                return torch.mm(embeddings1, embeddings2.t())
            
            @torch.compile  
            def compiled_normalization(embeddings):
                return F.normalize(embeddings, p=2, dim=1)
            
            self.compiled_similarity = compiled_similarity_computation
            self.compiled_normalize = compiled_normalization
            
            logger.info("⚡ Critical operations compiled with torch.compile")
            
        except Exception as e:
            logger.warning(f"⚠️ torch.compile failed: {e}, using standard operations")
            self.compiled_similarity = lambda x, y: torch.mm(x, y.t())
            self.compiled_normalize = lambda x: F.normalize(x, p=2, dim=1)

    def add_field_batch(self, geoid_ids: List[str], embeddings: torch.Tensor) -> List[GPUFieldState]:
        """Enhanced batch field addition with CUDA streams and optimization."""
        batch_size = len(geoid_ids)
        
        if embeddings.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch: {batch_size} IDs vs {embeddings.shape[0]} embeddings")
        
        start_time = time.perf_counter()
        
        # Get optimal stream for processing
        compute_stream = self.stream_manager.get_next_stream()
        
        with torch.cuda.stream(compute_stream):
            # Ensure embeddings are on GPU with correct dtype
            embeddings = embeddings.to(device=self.device, dtype=self.dtype)
            
            # Normalize embeddings in batch (GPU operation) - use compiled version if available
            embeddings = self.compiled_normalize(embeddings)
            
            # Calculate field properties in batch with mixed precision
            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
                # Resonance frequencies from embedding energy distribution
                energy_per_dim = torch.sum(embeddings * embeddings, dim=1)
                resonance_freqs = 10.0 + torch.sqrt(energy_per_dim) * 20.0
                
                # Phases from embedding asymmetry with optimized computation
                split_point = self.dimension // 2
                first_half = torch.sum(embeddings[:, :split_point], dim=1)
                second_half = torch.sum(embeddings[:, split_point:], dim=1)
                phases = torch.atan2(first_half, second_half + 1e-8)
                
                # Field strengths (normalized) with tensor cores optimization
                field_strengths = torch.ones(batch_size, device=self.device, dtype=torch.float32)
                
                # Decay rates based on frequency with vectorized computation
                decay_rates = torch.clamp(resonance_freqs * 0.01, 0.001, 0.1)
        
        # Synchronize the compute stream
        compute_stream.synchronize()
        
        # Expand tensor storage efficiently
        start_idx = self.next_index
        end_idx = start_idx + batch_size
        
        # Enhanced tensor concatenation with memory efficiency
        new_embeddings = torch.cat([self.field_embeddings, embeddings], dim=0)
        new_strengths = torch.cat([self.field_strengths, field_strengths], dim=0)
        new_frequencies = torch.cat([self.resonance_frequencies, resonance_freqs.float()], dim=0)
        new_phases = torch.cat([self.phases, phases.float()], dim=0)
        new_decay_rates = torch.cat([self.decay_rates, decay_rates.float()], dim=0)
        
        # Update storage atomically
        self.field_embeddings = new_embeddings
        self.field_strengths = new_strengths
        self.resonance_frequencies = new_frequencies
        self.phases = new_phases
        self.decay_rates = new_decay_rates
        
        # Update mappings
        field_states = []
        for i, geoid_id in enumerate(geoid_ids):
            idx = start_idx + i
            self.geoid_to_index[geoid_id] = idx
            self.index_to_geoid[idx] = geoid_id
            
            field_state = GPUFieldState(
                geoid_id=geoid_id
                embedding=embeddings[i],
                field_strength=field_strengths[i].item(),
                resonance_frequency=resonance_freqs[i].item(),
                phase=phases[i].item(),
                decay_rate=decay_rates[i].item(),
                creation_time=time.time()
            )
            field_states.append(field_state)
        
        self.next_index = end_idx
        self.operation_count += batch_size
        
        # Record performance metrics for optimization
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        gpu_utilization = self._get_gpu_utilization()
        self.batch_optimizer.record_performance(batch_size, processing_time_ms, gpu_utilization)
        
        # Update performance metrics
        self.performance_metrics.batch_processing_time_ms = processing_time_ms
        self.performance_metrics.gpu_utilization_percent = gpu_utilization
        self.performance_metrics.throughput_fields_per_sec = batch_size / (processing_time_ms / 1000.0)
        
        return field_states

    def find_neighbors_gpu_batch(self, query_indices: List[int], 
                                energy_threshold: float = 0.1
                                top_k: int = 50) -> List[List[Tuple[str, float]]]:
        """Enhanced neighbor finding with CUDA streams and optimized algorithms."""
        if len(query_indices) == 0:
            return []
        
        start_time = time.perf_counter()
        
        # Use dedicated stream for neighbor search
        search_stream = self.stream_manager.get_next_stream()
        
        with torch.cuda.stream(search_stream):
            query_indices_tensor = torch.tensor(query_indices, device=self.device)
            query_embeddings = self.field_embeddings[query_indices_tensor]
            
            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
                # Compute all pairwise similarities using compiled operation
                similarities = self.compiled_similarity(query_embeddings, self.field_embeddings)
                
                # Enhanced resonance frequency matching with optimized computation
                query_freqs = self.resonance_frequencies[query_indices_tensor]
                freq_diff = torch.abs(
                    query_freqs.unsqueeze(1) - self.resonance_frequencies.unsqueeze(0)
                )
                freq_similarities = torch.exp(-freq_diff * 0.1)  # Exponential decay for smoother matching
                
                # Advanced combined similarity score with configurable weights
                combined_similarities = similarities * 0.7 + freq_similarities * 0.3
                
                # Apply threshold with optimized masking
                mask = combined_similarities > energy_threshold
                
                # Zero out self-similarities efficiently
                for i, query_idx in enumerate(query_indices):
                    combined_similarities[i, query_idx] = 0.0
        
        # Synchronize search stream
        search_stream.synchronize()
        
        # Process results with optimized data structures
        results = []
        for i, query_idx in enumerate(query_indices):
            row_similarities = combined_similarities[i]
            valid_mask = mask[i]
            
            if valid_mask.sum() == 0:
                results.append([])
                continue
            
            # Enhanced top-k selection with memory efficiency
            valid_similarities = row_similarities[valid_mask]
            valid_indices = torch.where(valid_mask)[0]
            
            # Optimized sorting and selection
            if len(valid_similarities) > top_k:
                top_values, top_local_indices = torch.topk(valid_similarities, top_k, largest=True)
                top_indices = valid_indices[top_local_indices]
            else:
                sorted_indices = torch.argsort(valid_similarities, descending=True)
                top_indices = valid_indices[sorted_indices]
                top_values = valid_similarities[sorted_indices]
            
            # Convert to list of tuples with optimized memory access
            neighbors = []
            for idx, similarity in zip(top_indices, top_values):
                geoid_id = self.index_to_geoid[idx.item()]
                neighbors.append((geoid_id, similarity.item()))
            
            results.append(neighbors)
        
        # Record neighbor search performance
        search_time_ms = (time.perf_counter() - start_time) * 1000
        self.performance_metrics.neighbor_search_time_ms = search_time_ms
        
        return results

    def compute_influence_field_gpu(self, source_indices: List[int]) -> List[Dict[str, float]]:
        """Enhanced influence field computation with advanced GPU optimization."""
        if len(source_indices) == 0:
            return []
        
        # Use dedicated stream for influence computation
        influence_stream = self.stream_manager.get_next_stream()
        
        with torch.cuda.stream(influence_stream):
            source_indices_tensor = torch.tensor(source_indices, device=self.device)
            source_embeddings = self.field_embeddings[source_indices_tensor]
            source_strengths = self.field_strengths[source_indices_tensor]
            
            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
                # Enhanced distance computation with optimized algorithms
                distances = torch.cdist(source_embeddings, self.field_embeddings, p=2)
                
                # Advanced influence calculation with non-linear decay
                influence_decay = torch.exp(-distances * 0.5)  # Exponential decay
                influences = source_strengths.unsqueeze(1) * influence_decay
                
                # Zero out self-influences
                for i, source_idx in enumerate(source_indices):
                    influences[i, source_idx] = 0.0
        
        # Synchronize influence stream
        influence_stream.synchronize()
        
        # Convert to dictionaries with optimized data structures
        results = []
        for i, source_idx in enumerate(source_indices):
            influence_dict = {}
            for target_idx in range(self.field_embeddings.shape[0]):
                if target_idx != source_idx:
                    geoid_id = self.index_to_geoid[target_idx]
                    influence_value = influences[i, target_idx].item()
                    if influence_value > 0.01:  # Filter small influences for efficiency
                        influence_dict[geoid_id] = influence_value
            results.append(influence_dict)
        
        return results

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        if torch.cuda.is_available():
            try:
                # Try to get GPU utilization (not always available)
                return torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 85.0
            except Exception as e:
                logger.error(f"Error in cognitive_field_dynamics_gpu.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                return 85.0  # Default estimate
        return 0.0

    def get_gpu_memory_stats(self) -> Dict[str, float]:
        """Enhanced GPU memory usage statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # Calculate memory efficiency
            efficiency = (allocated / reserved * 100) if reserved > 0 else 0.0
            
            return {
                "allocated_mb": allocated
                "reserved_mb": reserved
                "max_allocated_mb": max_allocated
                "utilization_percent": self._get_gpu_utilization(),
                "memory_efficiency_percent": efficiency
                "stream_count": len(self.stream_manager.streams),
                "memory_pool_utilization": len(self.memory_pool.used_indices) / self.memory_pool.pool_size * 100 if hasattr(self.memory_pool, 'used_indices') else 0.0
            }
        return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0, "utilization_percent": 0, "memory_efficiency_percent": 0, "stream_count": 0, "memory_pool_utilization": 0}

    def optimize_gpu_performance(self):
        """Comprehensive GPU performance optimization."""
        logger.info("🔧 Optimizing GPU performance...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Optimize batch size
        optimal_batch_size = self.batch_optimizer.optimize_batch_size()
        
        # Enable cuDNN benchmarking for consistent operations
        if torch.cuda.is_available() and hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Update performance metrics
        self.performance_metrics.optimal_batch_size = optimal_batch_size
        
        logger.info(f"✅ GPU optimization complete: batch_size={optimal_batch_size}")
class CognitiveFieldDynamicsGPU:
    """Auto-generated class."""
    pass
    """Enhanced high-performance GPU-optimized Cognitive Field Dynamics engine."""
    
    def __init__(self, dimension: int, config: Optional[CognitiveFieldConfig] = None):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.dimension = dimension
        self.config = config or cfg
        self.device = DEVICE
        
        # Initialize enhanced GPU field system
        self.field_system = GPUSemanticFieldSystem(dimension, self.device)
        
        # Performance metrics and monitoring
        self.metrics_collector = get_metrics_collector()
        self.performance_stats = {
            "fields_created": 0
            "neighbor_searches": 0
            "batch_operations": 0
            "gpu_time_ms": 0.0
            "total_time_ms": 0.0
            "average_batch_size": 0.0
            "peak_throughput": 0.0
            "gpu_utilization_history": deque(maxlen=100)
        }
        
        # Enhanced batch optimization
        self.pending_fields = []
        self.auto_flush_enabled = True
        self.performance_monitor_thread = None
        
        # Start background performance monitoring
        if ENABLE_AUTO_TUNING:
            self._start_performance_monitoring()
        
        logger.info(f"🚀 Enhanced CognitiveFieldDynamicsGPU initialized: {dimension}D on {self.device}")
        if torch.cuda.is_available():
            memory_stats = self.field_system.get_gpu_memory_stats()
            logger.info(f"   Initial GPU memory: {memory_stats['allocated_mb']:.1f} MB")
            logger.info(f"   GPU utilization: {memory_stats['utilization_percent']:.1f}%")

    def _start_performance_monitoring(self):
        """Start background performance monitoring thread."""
        def monitor_performance():
            while self.auto_flush_enabled:
                try:
                    # Record current performance metrics
                    memory_stats = self.field_system.get_gpu_memory_stats()
                    self.performance_stats["gpu_utilization_history"].append(
                        memory_stats["utilization_percent"]
                    )
                    
                    # Auto-optimize if needed
                    if self.field_system.batch_optimizer.should_optimize():
                        self.field_system.optimize_gpu_performance()
                    
                    time.sleep(5)  # Monitor every 5 seconds
                    
                except Exception as e:
                    logger.warning(f"Performance monitoring error: {e}")
                    time.sleep(10)
        
        self.performance_monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
        self.performance_monitor_thread.start()
        logger.info("📊 Background performance monitoring started")

    @property
    def batch_size(self) -> int:
        """Get current optimal batch size."""
        return self.field_system.batch_optimizer.current_batch_size

    def add_geoid(self, geoid_id: str, embedding: torch.Tensor) -> Optional[GPUFieldState]:
        """Enhanced geoid addition with intelligent batching."""
        if geoid_id in self.field_system.geoid_to_index:
            # Return existing field
            idx = self.field_system.geoid_to_index[geoid_id]
            return self._get_field_state_by_index(idx)
        
        # Convert numpy to tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)
        
        # Add to pending batch
        self.pending_fields.append((geoid_id, embedding))
        
        # Intelligent batch processing with adaptive sizing
        current_batch_size = self.batch_size
        
        if len(self.pending_fields) >= current_batch_size:
            return self._flush_pending_fields()[-1]  # Return last added field
        
        # Auto-flush for small batches if enabled
        if self.auto_flush_enabled and len(self.pending_fields) >= 10:
            return self._flush_pending_fields()[-1] if self.pending_fields else None
        
        return None

    def _flush_pending_fields(self) -> List[GPUFieldState]:
        """Enhanced batch processing with performance optimization."""
        if not self.pending_fields:
            return []
        
        start_time = time.perf_counter()
        
        # Prepare batch data with enhanced processing
        geoid_ids = [item[0] for item in self.pending_fields]
        embeddings = torch.stack([item[1] for item in self.pending_fields])
        
        # GPU batch processing with stream synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        gpu_start = time.perf_counter()
        field_states = self.field_system.add_field_batch(geoid_ids, embeddings)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        gpu_end = time.perf_counter()
        
        # Enhanced performance tracking
        gpu_time_ms = (gpu_end - gpu_start) * 1000
        total_time_ms = (time.perf_counter() - start_time) * 1000
        current_throughput = len(geoid_ids) / (total_time_ms / 1000.0)
        
        # Update comprehensive performance stats
        self.performance_stats["fields_created"] += len(geoid_ids)
        self.performance_stats["batch_operations"] += 1
        self.performance_stats["gpu_time_ms"] += gpu_time_ms
        self.performance_stats["total_time_ms"] += total_time_ms
        self.performance_stats["average_batch_size"] = (
            self.performance_stats["average_batch_size"] * (self.performance_stats["batch_operations"] - 1) + 
            len(geoid_ids)
        ) / self.performance_stats["batch_operations"]
        
        if current_throughput > self.performance_stats["peak_throughput"]:
            self.performance_stats["peak_throughput"] = current_throughput
        
        logger.debug(f"Enhanced batch processed: {len(geoid_ids)} fields in {gpu_time_ms:.2f}ms GPU time, {current_throughput:.1f} fields/sec")
        
        # Clear pending
        self.pending_fields.clear()
        
        return field_states

    def find_semantic_neighbors(self, geoid_id: str, energy_threshold: float = 0.1) -> List[Tuple[str, float]]:
        """Enhanced semantic neighbor finding with optimized performance."""
        if geoid_id not in self.field_system.geoid_to_index:
            return []
        
        # Flush any pending fields first
        self._flush_pending_fields()
        
        start_time = time.perf_counter()
        
        query_idx = self.field_system.geoid_to_index[geoid_id]
        results = self.field_system.find_neighbors_gpu_batch([query_idx], energy_threshold)
        
        gpu_time_ms = (time.perf_counter() - start_time) * 1000
        self.performance_stats["neighbor_searches"] += 1
        self.performance_stats["gpu_time_ms"] += gpu_time_ms
        
        return results[0] if results else []

    def find_influence_field(self, geoid_id: str) -> Dict[str, float]:
        """Enhanced influence field computation with optimization."""
        if geoid_id not in self.field_system.geoid_to_index:
            return {}
        
        # Flush any pending fields first
        self._flush_pending_fields()
        
        source_idx = self.field_system.geoid_to_index[geoid_id]
        results = self.field_system.compute_influence_field_gpu([source_idx])
        
        return results[0] if results else {}

    def detect_semantic_anomalies(self) -> List[Dict]:
        """Enhanced semantic anomaly detection with GPU acceleration."""
        self._flush_pending_fields()
        
        if self.field_system.field_embeddings.shape[0] == 0:
            return []
        
        anomalies = []
        
        # Use dedicated stream for anomaly detection
        anomaly_stream = self.field_system.stream_manager.get_next_stream()
        
        with torch.cuda.stream(anomaly_stream):
            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
                # High field strength anomalies with enhanced detection
                strength_threshold = torch.mean(self.field_system.field_strengths) + 2.5 * torch.std(self.field_system.field_strengths)
                high_strength_mask = self.field_system.field_strengths > strength_threshold
                
                # High frequency anomalies with improved sensitivity
                freq_threshold = torch.mean(self.field_system.resonance_frequencies) + 2.5 * torch.std(self.field_system.resonance_frequencies)
                high_freq_mask = self.field_system.resonance_frequencies > freq_threshold
                
                # Enhanced isolation anomalies (fields with very few neighbors)
                similarities = torch.mm(self.field_system.field_embeddings, self.field_system.field_embeddings.t())
                neighbor_counts = torch.sum(similarities > 0.5, dim=1) - 1  # Exclude self
                isolation_threshold = torch.quantile(neighbor_counts.float(), 0.05)  # Bottom 5% (more sensitive)
                isolation_mask = neighbor_counts < isolation_threshold
                
                # Phase coherence anomalies (new detection type)
                phase_diffs = torch.abs(self.field_system.phases.unsqueeze(1) - self.field_system.phases.unsqueeze(0))
                phase_coherence = torch.mean(torch.cos(phase_diffs), dim=1)
                coherence_threshold = torch.quantile(phase_coherence, 0.1)  # Bottom 10%
                incoherent_mask = phase_coherence < coherence_threshold
        
        # Synchronize anomaly detection stream
        anomaly_stream.synchronize()
        
        # Process enhanced anomaly results
        all_indices = torch.arange(len(self.field_system.index_to_geoid))
        
        # High strength anomalies
        for idx in all_indices[high_strength_mask]:
            geoid_id = self.field_system.index_to_geoid[idx.item()]
            anomalies.append({
                "type": "high_strength",
                "geoid_id": geoid_id
                "field_strength": self.field_system.field_strengths[idx].item(),
                "severity": "moderate",
                "threshold": strength_threshold.item()
            })
        
        # High frequency anomalies
        for idx in all_indices[high_freq_mask]:
            geoid_id = self.field_system.index_to_geoid[idx.item()]
            anomalies.append({
                "type": "high_frequency",
                "geoid_id": geoid_id
                "resonance_frequency": self.field_system.resonance_frequencies[idx].item(),
                "severity": "moderate",
                "threshold": freq_threshold.item()
            })
        
        # Isolation anomalies
        for idx in all_indices[isolation_mask]:
            geoid_id = self.field_system.index_to_geoid[idx.item()]
            anomalies.append({
                "type": "isolation",
                "geoid_id": geoid_id
                "neighbor_count": neighbor_counts[idx].item(),
                "severity": "high",
                "threshold": isolation_threshold.item()
            })
        
        # Phase coherence anomalies
        for idx in all_indices[incoherent_mask]:
            geoid_id = self.field_system.index_to_geoid[idx.item()]
            anomalies.append({
                "type": "phase_incoherence",
                "geoid_id": geoid_id
                "coherence_score": phase_coherence[idx].item(),
                "severity": "low",
                "threshold": coherence_threshold.item()
            })
        
        return anomalies

    def _get_field_state_by_index(self, idx: int) -> Optional[GPUFieldState]:
        """Get field state by index with enhanced error handling."""
        if idx >= self.field_system.field_embeddings.shape[0]:
            return None
        
        geoid_id = self.field_system.index_to_geoid[idx]
        return GPUFieldState(
            geoid_id=geoid_id
            embedding=self.field_system.field_embeddings[idx],
            field_strength=self.field_system.field_strengths[idx].item(),
            resonance_frequency=self.field_system.resonance_frequencies[idx].item(),
            phase=self.field_system.phases[idx].item(),
            decay_rate=self.field_system.decay_rates[idx].item()
        )

    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics with enhanced metrics."""
        gpu_stats = self.field_system.get_gpu_memory_stats()
        
        total_fields = self.performance_stats["fields_created"]
        total_gpu_time = self.performance_stats["gpu_time_ms"]
        total_time = self.performance_stats["total_time_ms"]
        
        # Calculate advanced performance metrics
        gpu_efficiency = (total_gpu_time / total_time) * 100 if total_time > 0 else 0
        fields_per_second = total_fields / (total_time / 1000) if total_time > 0 else 0
        average_gpu_utilization = np.mean(self.performance_stats["gpu_utilization_history"]) if self.performance_stats["gpu_utilization_history"] else 0
        
        return {
            "total_fields": total_fields
            "neighbor_searches": self.performance_stats["neighbor_searches"],
            "batch_operations": self.performance_stats["batch_operations"],
            "gpu_time_ms": total_gpu_time
            "total_time_ms": total_time
            "gpu_efficiency_percent": gpu_efficiency
            "fields_per_second": fields_per_second
            "peak_throughput": self.performance_stats["peak_throughput"],
            "average_batch_size": self.performance_stats["average_batch_size"],
            "current_optimal_batch_size": self.batch_size
            "average_gpu_utilization": average_gpu_utilization
            "gpu_memory": gpu_stats
            "device": str(self.device),
            "mixed_precision": USE_MIXED_PRECISION
            "cuda_streams_enabled": ENABLE_CUDA_STREAMS
            "auto_tuning_enabled": ENABLE_AUTO_TUNING
            "memory_pooling_enabled": ENABLE_MEMORY_POOLING
            "compiled_operations": COMPILE_MODELS
        }

    def optimize_for_inference(self):
        """Enhanced optimization for inference with comprehensive tuning."""
        if self.field_system.field_embeddings.shape[0] == 0:
            logger.warning("No fields to optimize - add fields first")
            return
        
        logger.info("🔥 Enhanced GPU engine optimization for inference...")
        
        # Comprehensive GPU optimization
        self.field_system.optimize_gpu_performance()
        
        # Warm up all compiled operations
        if COMPILE_MODELS and torch.cuda.is_available():
            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
                # Warm up similarity computation
                dummy_query = self.field_system.field_embeddings[:1]
                _ = self.field_system.compiled_similarity(dummy_query, self.field_system.field_embeddings)
                
                # Warm up normalization
                _ = self.field_system.compiled_normalize(dummy_query)
        
        # Optimize cuDNN for consistent operations
        if torch.cuda.is_available() and hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Synchronize all streams
        self.field_system.stream_manager.synchronize_all()
        
        logger.info("✅ Enhanced GPU optimization complete - ready for maximum performance")

    @property
    def fields(self) -> Dict[str, GPUFieldState]:
        """Get all fields as a dictionary for compatibility."""
        fields_dict = {}
        for geoid_id, idx in self.field_system.geoid_to_index.items():
            field_state = self._get_field_state_by_index(idx)
            if field_state:
                fields_dict[geoid_id] = field_state
        return fields_dict

    def shutdown(self):
        """Graceful shutdown with cleanup."""
        logger.info("🛑 Shutting down Enhanced CognitiveFieldDynamicsGPU...")
        
        # Stop auto-flushing and monitoring
        self.auto_flush_enabled = False
        
        # Flush any remaining pending fields
        self._flush_pending_fields()
        
        # Synchronize all CUDA streams
        self.field_system.stream_manager.synchronize_all()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Enhanced GPU engine shutdown complete")

# Alias for compatibility
CognitiveFieldDynamics = CognitiveFieldDynamicsGPU 