#!/usr/bin/env python3
"""
Comprehensive Speed and Latency Benchmark System

Measures:
- Processing speed (fields/second)
- Response latency (milliseconds)
- Throughput under load
- Memory access latency
- GPU compute latency
- System responsiveness
"""

import time
import json
import numpy as np
import torch
import psutil
import platform
from datetime import datetime
from pathlib import Path
import sys
import statistics
from typing import Dict, List, Any, Tuple

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Import Kimera components
sys.path.append(str(Path(__file__).parent / "backend"))

class SpeedLatencyBenchmark:
    """Comprehensive speed and latency testing system"""
    
    def __init__(self):
        self.results = []
        self.latency_measurements = []
        
        logger.info("KIMERA SPEED & LATENCY BENCHMARK SUITE")
        logger.info("=" * 60)
        logger.info(f"Platform: {platform.platform()
        logger.info(f"CUDA Available: {torch.cuda.is_available()
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)
        logger.info(f"Test Start: {datetime.now()
        logger.info()
    
    def measure_single_field_latency(self, iterations: int = 100) -> Dict[str, float]:
        """Measure latency for single field creation"""
        
        logger.info(f"SINGLE FIELD LATENCY TEST ({iterations} iterations)
        logger.info("-" * 40)
        
        # Initialize engine
        try:
            from backend.engines.cognitive_field_dynamics import CognitiveFieldDynamics
            engine = CognitiveFieldDynamics(dimension=128)
        except Exception as e:
            logger.error(f"ERROR: Failed to initialize engine: {e}")
            return {}
        
        latencies = []
        successful_operations = 0
        
        # Warm up
        for i in range(10):
            embedding = np.random.randn(128).astype(np.float32)
            engine.add_geoid(f"warmup_{i}", embedding)
        
        # Latency measurements
        for i in range(iterations):
            embedding = np.random.randn(128).astype(np.float32)
            
            start_time = time.perf_counter_ns()  # Nanosecond precision
            field = engine.add_geoid(f"latency_test_{i:04d}", embedding)
            end_time = time.perf_counter_ns()
            
            if field:
                latency_ns = end_time - start_time
                latency_ms = latency_ns / 1_000_000  # Convert to milliseconds
                latencies.append(latency_ms)
                successful_operations += 1
        
        # Statistical analysis
        if latencies:
            stats = {
                "iterations": iterations,
                "successful_operations": successful_operations,
                "success_rate": (successful_operations / iterations) * 100,
                "mean_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "latency_range_ms": max(latencies) - min(latencies)
            }
        else:
            stats = {"error": "No successful operations"}
        
        logger.info(f"  Successful Operations: {successful_operations}/{iterations}")
        if latencies:
            logger.info(f"  Mean Latency: {stats['mean_latency_ms']:.3f} ms")
            logger.info(f"  Median Latency: {stats['median_latency_ms']:.3f} ms")
            logger.info(f"  P95 Latency: {stats['p95_latency_ms']:.3f} ms")
            logger.info(f"  P99 Latency: {stats['p99_latency_ms']:.3f} ms")
            logger.info(f"  Min/Max: {stats['min_latency_ms']:.3f} / {stats['max_latency_ms']:.3f} ms")
        logger.info()
        
        return stats
    
    def measure_batch_throughput(self, batch_sizes: List[int]) -> Dict[str, Any]:
        """Measure throughput for different batch sizes"""
        
        logger.info("BATCH THROUGHPUT ANALYSIS")
        logger.info("-" * 40)
        
        batch_results = []
        
        for batch_size in batch_sizes:
            logger.info(f"  Testing batch size: {batch_size}")
            
            # Initialize engine
            try:
                from backend.engines.cognitive_field_dynamics import CognitiveFieldDynamics
                engine = CognitiveFieldDynamics(dimension=128)
            except Exception as e:
                logger.error(f"    ERROR: {e}")
                continue
            
            # Pre-generate embeddings for consistent testing
            embeddings = [np.random.randn(128).astype(np.float32) for _ in range(batch_size)]
            
            # Measure batch processing
            start_time = time.perf_counter()
            created_fields = []
            
            for i, embedding in enumerate(embeddings):
                field = engine.add_geoid(f"batch_{batch_size}_{i:06d}", embedding)
                if field:
                    created_fields.append(field)
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            duration = end_time - start_time
            throughput = len(created_fields) / duration
            avg_latency = (duration / len(created_fields)) * 1000 if created_fields else 0
            
            batch_result = {
                "batch_size": batch_size,
                "duration_seconds": duration,
                "fields_created": len(created_fields),
                "success_rate": (len(created_fields) / batch_size) * 100,
                "throughput_fields_per_sec": throughput,
                "average_latency_ms": avg_latency,
                "throughput_efficiency": throughput / batch_size  # Normalized throughput
            }
            
            batch_results.append(batch_result)
            
            logger.info(f"    Throughput: {throughput:.1f} fields/sec")
            logger.info(f"    Avg Latency: {avg_latency:.3f} ms")
            logger.info(f"    Success Rate: {batch_result['success_rate']:.1f}%")
        
        logger.info()
        return {"batch_results": batch_results}
    
    def measure_memory_latency(self) -> Dict[str, float]:
        """Measure memory access latency patterns"""
        
        logger.info("MEMORY ACCESS LATENCY TEST")
        logger.info("-" * 40)
        
        # GPU memory latency test
        if torch.cuda.is_available():
            # Small tensor operations
            small_tensor_times = []
            for _ in range(100):
                start = time.perf_counter_ns()
                tensor = torch.randn(128, device='cuda')
                result = tensor.sum()
                torch.cuda.synchronize()
                end = time.perf_counter_ns()
                small_tensor_times.append((end - start) / 1_000_000)  # ms
            
            # Large tensor operations
            large_tensor_times = []
            for _ in range(20):
                start = time.perf_counter_ns()
                tensor = torch.randn(10000, device='cuda')
                result = tensor.sum()
                torch.cuda.synchronize()
                end = time.perf_counter_ns()
                large_tensor_times.append((end - start) / 1_000_000)  # ms
            
            gpu_stats = {
                "small_tensor_mean_ms": statistics.mean(small_tensor_times),
                "small_tensor_min_ms": min(small_tensor_times),
                "small_tensor_max_ms": max(small_tensor_times),
                "large_tensor_mean_ms": statistics.mean(large_tensor_times),
                "large_tensor_min_ms": min(large_tensor_times),
                "large_tensor_max_ms": max(large_tensor_times),
                "memory_scaling_factor": statistics.mean(large_tensor_times) / statistics.mean(small_tensor_times)
            }
            
            logger.info(f"  Small Tensor (128)
            logger.info(f"  Large Tensor (10k)
            logger.info(f"  Scaling Factor: {gpu_stats['memory_scaling_factor']:.2f}x")
        
        else:
            gpu_stats = {"error": "CUDA not available"}
            logger.info("  CUDA not available for GPU memory testing")
        
        # CPU memory latency test
        cpu_times = []
        for _ in range(100):
            start = time.perf_counter_ns()
            array = np.random.randn(1000)
            result = np.sum(array)
            end = time.perf_counter_ns()
            cpu_times.append((end - start) / 1_000_000)  # ms
        
        cpu_stats = {
            "cpu_memory_mean_ms": statistics.mean(cpu_times),
            "cpu_memory_min_ms": min(cpu_times),
            "cpu_memory_max_ms": max(cpu_times)
        }
        
        logger.info(f"  CPU Memory Access: {cpu_stats['cpu_memory_mean_ms']:.3f} ms avg")
        logger.info()
        
        return {**gpu_stats, **cpu_stats}
    
    def measure_system_responsiveness(self) -> Dict[str, float]:
        """Measure system responsiveness under different loads"""
        
        logger.info("SYSTEM RESPONSIVENESS TEST")
        logger.info("-" * 40)
        
        # Baseline responsiveness (no load)
        baseline_times = []
        for _ in range(50):
            start = time.perf_counter_ns()
            # Simple operation
            result = sum(range(1000))
            end = time.perf_counter_ns()
            baseline_times.append((end - start) / 1_000)  # microseconds
        
        # Under computational load
        logger.info("  Testing under computational load...")
        
        # Initialize engine for load testing
        try:
            from backend.engines.cognitive_field_dynamics import CognitiveFieldDynamics
            engine = CognitiveFieldDynamics(dimension=128)
        except Exception as e:
            logger.error(f"    ERROR: {e}")
            return {}
        
        # Create background load
        loaded_times = []
        for i in range(50):
            # Background processing
            if i % 10 == 0:
                embeddings = [np.random.randn(128).astype(np.float32) for _ in range(10)]
                for j, embedding in enumerate(embeddings):
                    engine.add_geoid(f"load_test_{i}_{j}", embedding)
            
            # Measure responsiveness
            start = time.perf_counter_ns()
            result = sum(range(1000))
            end = time.perf_counter_ns()
            loaded_times.append((end - start) / 1_000)  # microseconds
        
        responsiveness_stats = {
            "baseline_mean_us": statistics.mean(baseline_times),
            "baseline_min_us": min(baseline_times),
            "baseline_max_us": max(baseline_times),
            "loaded_mean_us": statistics.mean(loaded_times),
            "loaded_min_us": min(loaded_times),
            "loaded_max_us": max(loaded_times),
            "responsiveness_degradation": statistics.mean(loaded_times) / statistics.mean(baseline_times),
            "load_impact_factor": (statistics.mean(loaded_times) - statistics.mean(baseline_times)) / statistics.mean(baseline_times)
        }
        
        logger.info(f"  Baseline: {responsiveness_stats['baseline_mean_us']:.1f} μs avg")
        logger.info(f"  Under Load: {responsiveness_stats['loaded_mean_us']:.1f} μs avg")
        logger.info(f"  Degradation: {responsiveness_stats['responsiveness_degradation']:.2f}x")
        logger.info(f"  Load Impact: {responsiveness_stats['load_impact_factor']*100:.1f}%")
        logger.info()
        
        return responsiveness_stats
    
    def measure_concurrent_performance(self) -> Dict[str, Any]:
        """Measure performance under concurrent operations"""
        
        logger.info("CONCURRENT OPERATIONS TEST")
        logger.info("-" * 40)
        
        try:
            from backend.engines.cognitive_field_dynamics import CognitiveFieldDynamics
            engine = CognitiveFieldDynamics(dimension=128)
        except Exception as e:
            logger.error(f"ERROR: {e}")
            return {}
        
        # Sequential processing baseline
        sequential_start = time.perf_counter()
        sequential_fields = []
        
        for i in range(100):
            embedding = np.random.randn(128).astype(np.float32)
            field = engine.add_geoid(f"sequential_{i:03d}", embedding)
            if field:
                sequential_fields.append(field)
        
        sequential_duration = time.perf_counter() - sequential_start
        sequential_throughput = len(sequential_fields) / sequential_duration
        
        # Simulated concurrent processing (interleaved operations)
        concurrent_start = time.perf_counter()
        concurrent_fields = []
        
        # Simulate concurrency by interleaving different types of operations
        for i in range(100):
            # Main operation
            embedding = np.random.randn(128).astype(np.float32)
            field = engine.add_geoid(f"concurrent_{i:03d}", embedding)
            if field:
                concurrent_fields.append(field)
            
            # Interleaved operations every 10 iterations
            if i % 10 == 0:
                # Additional memory operations
                temp_tensor = torch.randn(100) if not torch.cuda.is_available() else torch.randn(100, device='cuda')
                temp_result = temp_tensor.sum()
        
        concurrent_duration = time.perf_counter() - concurrent_start
        concurrent_throughput = len(concurrent_fields) / concurrent_duration
        
        concurrency_stats = {
            "sequential_fields": len(sequential_fields),
            "sequential_duration": sequential_duration,
            "sequential_throughput": sequential_throughput,
            "concurrent_fields": len(concurrent_fields),
            "concurrent_duration": concurrent_duration,
            "concurrent_throughput": concurrent_throughput,
            "throughput_ratio": concurrent_throughput / sequential_throughput,
            "overhead_percentage": ((concurrent_duration - sequential_duration) / sequential_duration) * 100
        }
        
        logger.info(f"  Sequential: {sequential_throughput:.1f} fields/sec")
        logger.info(f"  Concurrent: {concurrent_throughput:.1f} fields/sec")
        logger.info(f"  Ratio: {concurrency_stats['throughput_ratio']:.3f}")
        logger.info(f"  Overhead: {concurrency_stats['overhead_percentage']:.1f}%")
        logger.info()
        
        return concurrency_stats
    
    def run_comprehensive_speed_latency_suite(self) -> Dict[str, Any]:
        """Run complete speed and latency benchmark suite"""
        
        suite_start = time.perf_counter()
        
        logger.info("STARTING COMPREHENSIVE SPEED & LATENCY BENCHMARK")
        logger.info("=" * 60)
        logger.info()
        
        # 1. Single field latency
        single_field_stats = self.measure_single_field_latency(100)
        
        # 2. Batch throughput analysis
        batch_sizes = [10, 50, 100, 500, 1000]
        batch_stats = self.measure_batch_throughput(batch_sizes)
        
        # 3. Memory access latency
        memory_stats = self.measure_memory_latency()
        
        # 4. System responsiveness
        responsiveness_stats = self.measure_system_responsiveness()
        
        # 5. Concurrent performance
        concurrency_stats = self.measure_concurrent_performance()
        
        suite_duration = time.perf_counter() - suite_start
        
        # Comprehensive analysis
        comprehensive_results = {
            "benchmark_metadata": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": suite_duration,
                "platform": platform.platform(),
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            },
            "single_field_latency": single_field_stats,
            "batch_throughput": batch_stats,
            "memory_latency": memory_stats,
            "system_responsiveness": responsiveness_stats,
            "concurrent_performance": concurrency_stats
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"speed_latency_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Summary report
        logger.info("COMPREHENSIVE SPEED & LATENCY RESULTS")
        logger.info("=" * 50)
        logger.info(f"Total Benchmark Duration: {suite_duration:.1f} seconds")
        logger.info()
        
        if single_field_stats and 'mean_latency_ms' in single_field_stats:
            logger.info("LATENCY ANALYSIS:")
            logger.info(f"  Mean Single Field Latency: {single_field_stats['mean_latency_ms']:.3f} ms")
            logger.info(f"  P95 Latency: {single_field_stats['p95_latency_ms']:.3f} ms")
            logger.info(f"  P99 Latency: {single_field_stats['p99_latency_ms']:.3f} ms")
        
        if batch_stats and 'batch_results' in batch_stats:
            best_throughput = max(batch_stats['batch_results'], key=lambda x: x['throughput_fields_per_sec'])
            logger.info()
            logger.info("THROUGHPUT ANALYSIS:")
            logger.info(f"  Peak Throughput: {best_throughput['throughput_fields_per_sec']:.1f} fields/sec")
            logger.info(f"  Optimal Batch Size: {best_throughput['batch_size']}")
            logger.info(f"  Best Avg Latency: {best_throughput['average_latency_ms']:.3f} ms")
        
        if responsiveness_stats and 'responsiveness_degradation' in responsiveness_stats:
            logger.info()
            logger.info("RESPONSIVENESS ANALYSIS:")
            logger.info(f"  Baseline Response: {responsiveness_stats['baseline_mean_us']:.1f} μs")
            logger.info(f"  Under Load: {responsiveness_stats['loaded_mean_us']:.1f} μs")
            logger.info(f"  Load Impact: {responsiveness_stats['load_impact_factor']*100:.1f}%")
        
        if concurrency_stats and 'throughput_ratio' in concurrency_stats:
            logger.info()
            logger.info("CONCURRENCY ANALYSIS:")
            logger.info(f"  Sequential: {concurrency_stats['sequential_throughput']:.1f} fields/sec")
            logger.info(f"  Concurrent: {concurrency_stats['concurrent_throughput']:.1f} fields/sec")
            logger.info(f"  Efficiency Ratio: {concurrency_stats['throughput_ratio']:.3f}")
        
        logger.info()
        logger.info(f"Detailed results saved to: {filename}")
        logger.info("SPEED & LATENCY BENCHMARK COMPLETE")
        
        return comprehensive_results


def main():
    """Run comprehensive speed and latency benchmark"""
    
    benchmark = SpeedLatencyBenchmark()
    results = benchmark.run_comprehensive_speed_latency_suite()
    
    return results


if __name__ == "__main__":
    main() 