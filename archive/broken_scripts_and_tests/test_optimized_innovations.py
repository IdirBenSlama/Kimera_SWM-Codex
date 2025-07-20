#!/usr/bin/env python3
"""
Test the optimized innovation modules
"""
import asyncio
import time
import sys
import os

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)

sys.path.insert(0, os.path.abspath("."))

from innovations.quantum_batch_processor import QuantumBatchProcessor, get_quantum_metrics
from innovations.predictive_load_balancer import (
    PredictiveLoadBalancer, ResourceState, WorkloadRequest,
    register_resource, get_load_balancing_decision
)
from backend.core.geoid import GeoidState
import uuid

async def benchmark_quantum_processor():
    """Benchmark the optimized quantum processor"""
    logger.info("=== QUANTUM PROCESSOR BENCHMARK ===\n")
    
    # Test with optimized settings
    processor = QuantumBatchProcessor(
        max_batch_size=256,
        entanglement_threshold=0.0,  # Skip entanglement
        use_embeddings=False  # Skip embeddings
    )
    
    batch_sizes = [10, 50, 100, 200]
    
    for size in batch_sizes:
        # Create test geoids
        geoids = []
        for i in range(size):
            geoid = GeoidState(
                geoid_id=uuid.uuid4().hex,
                semantic_state={f"feature_{j}": float(j) for j in range(32)},
                symbolic_state={"data": f"test_{i}"},
                metadata={"index": i}
            )
            geoids.append(geoid)
        
        # Benchmark
        start = time.time()
        result = await processor.process_quantum_batch(geoids)
        duration = time.time() - start
        
        logger.info(f"Batch size {size}:")
        logger.info(f"  Processing time: {duration:.3f}s")
        logger.info(f"  Throughput: {size/duration:.1f} geoids/sec")
        logger.info(f"  Quantum efficiency: {result.quantum_efficiency:.1f}")
        logger.info()

def benchmark_load_balancer():
    """Benchmark the optimized load balancer"""
    logger.info("\n=== LOAD BALANCER BENCHMARK ===\n")
    
    balancer = PredictiveLoadBalancer()
    
    # Register minimal resources
    for i in range(4):  # Just 4 CPUs
        resource = ResourceState(
            resource_id=f"cpu_{i}",
            resource_type="cpu",
            utilization=0.3,
            capacity=100.0,
            temperature=50.0,
            power_consumption=50.0,
            latency=0.01,
            throughput=100.0,
            error_rate=0.0
        )
        register_resource(resource)
    
    # Benchmark decision making
    workload_sizes = [10, 50, 100, 200]
    
    for size in workload_sizes:
        workload = WorkloadRequest(
            request_id=f"workload_{size}",
            complexity_score=size / 200.0,
            resource_requirements={"cpu": 1.0},
            priority=5
        )
        
        start = time.time()
        decision = get_load_balancing_decision(workload)
        duration = time.time() - start
        
        logger.info(f"Workload size {size}:")
        logger.info(f"  Decision time: {duration*1000:.1f}ms")
        logger.info(f"  Confidence: {decision.confidence_score:.2f}")
        logger.info(f"  Resources allocated: {len(decision.assigned_resources)
        logger.info()

async def main():
    logger.info("OPTIMIZED INNOVATION MODULES TEST\n")
    
    # Test quantum processor
    await benchmark_quantum_processor()
    
    # Test load balancer
    benchmark_load_balancer()
    
    # Show final metrics
    logger.info("\n=== FINAL METRICS ===")
    quantum_metrics = get_quantum_metrics()
    logger.info(f"Quantum Processor:")
    logger.info(f"  Total superpositions: {quantum_metrics['total_superpositions']}")
    logger.info(f"  Speedup factor: {quantum_metrics['quantum_speedup_factor']:.2f}x")

if __name__ == "__main__":
    asyncio.run(main())