#!/usr/bin/env python3
"""
Simple test to verify innovation modules are working
"""
import asyncio
import sys
import os

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)

sys.path.insert(0, os.path.abspath("."))

from innovations.quantum_batch_processor import process_geoids_quantum, get_quantum_metrics
from innovations.predictive_load_balancer import (
    ResourceState, WorkloadRequest,
    register_resource, get_load_balancing_decision,
    start_optimization_loop, stop_optimization_loop
)
from backend.core.geoid import GeoidState
import uuid

async def test_quantum_processor():
    logger.info("Testing Quantum Batch Processor...")
    
    # Create a few test geoids
    geoids = []
    for i in range(5):
        geoid = GeoidState(
            geoid_id=uuid.uuid4().hex,
            semantic_state={f"feature_{j}": float(j) for j in range(10)},
            symbolic_state={"data": f"test_{i}"},
            metadata={"index": i}
        )
        geoids.append(geoid)
    
    logger.info(f"Created {len(geoids)
    
    try:
        # Process with quantum batch processor
        result = await asyncio.wait_for(
            process_geoids_quantum(geoids),
            timeout=10.0
        )
        
        logger.info(f"✅ Quantum processing successful!")
        logger.info(f"   - Processed: {len(result.processed_geoids)
        logger.info(f"   - Coherence score: {result.coherence_score:.2f}")
        logger.info(f"   - Processing time: {result.processing_time:.2f}s")
        logger.info(f"   - Quantum efficiency: {result.quantum_efficiency:.2f}")
        
        # Get metrics
        metrics = get_quantum_metrics()
        logger.info(f"\nQuantum Metrics:")
        for key, value in metrics.items():
            logger.info(f"   - {key}: {value}")
            
    except asyncio.TimeoutError:
        logger.error("❌ Quantum processing timed out!")
    except Exception as e:
        logger.error(f"❌ Quantum processing failed: {e}")
        import traceback
        traceback.print_exc()

def test_load_balancer():
    logger.info("\n\nTesting Predictive Load Balancer...")
    
    # Register a simple resource
    resource = ResourceState(
        resource_id="test_gpu",
        resource_type="gpu",
        utilization=0.3,
        capacity=100.0,
        temperature=50.0,
        power_consumption=100.0,
        latency=0.01,
        throughput=100.0,
        error_rate=0.0
    )
    register_resource(resource)
    logger.info("✅ Resource registered")
    
    # Start optimization loop
    start_optimization_loop()
    logger.info("✅ Optimization loop started")
    
    # Create a workload
    workload = WorkloadRequest(
        request_id="test_workload",
        complexity_score=0.5,
        resource_requirements={"gpu": 0.8},
        priority=5
    )
    
    # Get load balancing decision
    decision = get_load_balancing_decision(workload)
    logger.info(f"✅ Load balancing decision: {decision.reasoning}")
    logger.info(f"   - Confidence: {decision.confidence_score:.2f}")
    logger.info(f"   - Predicted completion: {decision.predicted_completion_time:.2f}")
    
    # Stop optimization loop
    stop_optimization_loop()
    logger.info("✅ Optimization loop stopped")

if __name__ == "__main__":
    logger.info("=== INNOVATION MODULES TEST ===\n")
    
    # Test quantum processor
    asyncio.run(test_quantum_processor())
    
    # Test load balancer
    test_load_balancer()
    
    logger.info("\n=== TEST COMPLETE ===")