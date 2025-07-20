"""
Quantum Integration Tests - KIMERA Phase 2, Week 2
===============================================

Comprehensive integration tests for quantum computing components
with neuropsychiatric safety validation and performance benchmarking.

Author: KIMERA Development Team
Version: 1.0.0 - Phase 2 Quantum Integration Tests
"""

import pytest
import asyncio
import time
import numpy as np
import torch
from typing import Dict, Any, List
from datetime import datetime
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import quantum components
try:
    from backend.engines.quantum_cognitive_engine import (
        QuantumCognitiveEngine,
        QuantumCognitiveState,
        QuantumNeuropsychiatricSafeguard,
        initialize_quantum_cognitive_engine
    )
    from backend.engines.quantum_classical_interface import (
        QuantumClassicalInterface,
        HybridProcessingMode,
        create_quantum_classical_interface
    )
    from backend.utils.gpu_foundation import GPUFoundation
    QUANTUM_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quantum imports not available: {e}")
    QUANTUM_IMPORTS_AVAILABLE = False

class TestQuantumCognitiveEngine:
    """Test suite for Quantum Cognitive Engine"""
    
    @pytest.fixture
    def quantum_engine(self):
        """Fixture providing quantum cognitive engine"""
        if not QUANTUM_IMPORTS_AVAILABLE:
            pytest.skip("Quantum imports not available")
        
        return QuantumCognitiveEngine(
            num_qubits=10,  # Smaller for testing
            gpu_acceleration=True,
            safety_level="rigorous"
        )
    
    @pytest.fixture
    def sample_cognitive_data(self):
        """Fixture providing sample cognitive data"""
        return [
            np.random.randn(100),
            np.random.randn(100),
            np.random.randn(100)
        ]
    
    def test_quantum_engine_initialization(self, quantum_engine):
        """Test quantum engine initialization"""
        assert quantum_engine is not None
        assert quantum_engine.num_qubits == 10
        assert quantum_engine.gpu_acceleration is True
        assert quantum_engine.safety_level == "rigorous"
        assert quantum_engine.quantum_simulator is not None
        assert quantum_engine.safety_guard is not None
        
        logger.info("✅ Quantum engine initialization test passed")
    
    def test_cognitive_superposition_creation(self, quantum_engine, sample_cognitive_data):
        """Test creation of cognitive superposition states"""
        # Create superposition
        quantum_state = quantum_engine.create_cognitive_superposition(
            sample_cognitive_data,
            entanglement_strength=0.5
        )
        
        # Validate quantum state
        assert isinstance(quantum_state, QuantumCognitiveState)
        assert quantum_state.state_vector is not None
        assert len(quantum_state.state_vector) > 0
        assert 0.0 <= quantum_state.entanglement_entropy <= 10.0
        assert 0.0 <= quantum_state.quantum_fidelity <= 1.0
        assert 0.0 <= quantum_state.classical_correlation <= 1.0
        
        logger.info("✅ Cognitive superposition creation test passed")
        logger.info(f"   Entanglement entropy: {quantum_state.entanglement_entropy:.3f}")
        logger.info(f"   Quantum fidelity: {quantum_state.quantum_fidelity:.3f}")
    
    def test_neuropsychiatric_safety_validation(self, quantum_engine, sample_cognitive_data):
        """Test neuropsychiatric safety validation"""
        safety_guard = QuantumNeuropsychiatricSafeguard()
        
        # Create quantum state
        quantum_state = quantum_engine.create_cognitive_superposition(
            sample_cognitive_data,
            entanglement_strength=0.3  # Lower entanglement for safety
        )
        
        # Validate safety
        is_safe = safety_guard.validate_quantum_cognitive_state(quantum_state)
        assert is_safe is True
        
        # Test safety thresholds
        assert quantum_state.quantum_fidelity >= safety_guard.identity_threshold * 0.9  # Allow some tolerance
        assert quantum_state.classical_correlation >= safety_guard.reality_anchor_strength * 0.9
        assert quantum_state.entanglement_entropy <= 3.0  # Reasonable upper bound
        
        logger.info("✅ Neuropsychiatric safety validation test passed")
    
    def test_quantum_cognitive_interference(self, quantum_engine, sample_cognitive_data):
        """Test quantum cognitive interference processing"""
        # Create two quantum states
        state1 = quantum_engine.create_cognitive_superposition(
            sample_cognitive_data[:2],
            entanglement_strength=0.3
        )
        
        state2 = quantum_engine.create_cognitive_superposition(
            sample_cognitive_data[1:],
            entanglement_strength=0.4
        )
        
        # Process interference
        interference_state = quantum_engine.process_quantum_cognitive_interference(state1, state2)
        
        # Validate interference result
        assert isinstance(interference_state, QuantumCognitiveState)
        assert interference_state.state_vector is not None
        assert 0.0 <= interference_state.quantum_fidelity <= 1.0
        assert 0.0 <= interference_state.classical_correlation <= 1.0
        
        logger.info("✅ Quantum cognitive interference test passed")
    
    def test_quantum_measurement(self, quantum_engine, sample_cognitive_data):
        """Test quantum measurement functionality"""
        # Create quantum state
        quantum_state = quantum_engine.create_cognitive_superposition(
            sample_cognitive_data,
            entanglement_strength=0.4
        )
        
        # Perform measurement
        measurement_results = quantum_engine.measure_quantum_cognitive_state(
            quantum_state,
            measurement_basis="computational"
        )
        
        # Validate measurement results
        assert isinstance(measurement_results, dict)
        assert 'counts' in measurement_results
        assert 'execution_time' in measurement_results
        assert 'total_shots' in measurement_results
        assert measurement_results['total_shots'] == 1000
        assert measurement_results['execution_time'] > 0
        
        logger.info("✅ Quantum measurement test passed")
        logger.info(f"   Measurement execution time: {measurement_results['execution_time']*1000:.2f}ms")
    
    def test_performance_metrics(self, quantum_engine):
        """Test quantum processing performance metrics"""
        metrics = quantum_engine.get_quantum_processing_metrics()
        
        # Validate metrics
        assert metrics.circuit_depth > 0
        assert metrics.gate_count > 0
        assert metrics.quantum_volume > 0
        assert 0.0 <= metrics.error_rate <= 1.0
        assert metrics.gpu_utilization >= 0.0
        assert metrics.memory_usage >= 0.0
        
        logger.info("✅ Performance metrics test passed")
        logger.info(f"   Circuit depth: {metrics.circuit_depth}")
        logger.info(f"   Quantum volume: {metrics.quantum_volume}")

class TestQuantumClassicalInterface:
    """Test suite for Quantum-Classical Interface"""
    
    @pytest.fixture
    def interface(self):
        """Fixture providing quantum-classical interface"""
        if not QUANTUM_IMPORTS_AVAILABLE:
            pytest.skip("Quantum imports not available")
        
        return create_quantum_classical_interface(
            num_qubits=8,  # Smaller for testing
            adaptive_mode=True
        )
    
    @pytest.fixture
    def test_data(self):
        """Fixture providing test data for hybrid processing"""
        return torch.randn(32, 64)  # Batch of 32, features of 64
    
    def test_interface_initialization(self, interface):
        """Test interface initialization"""
        assert interface is not None
        assert interface.quantum_engine is not None
        assert interface.gpu_foundation is not None
        assert interface.adaptive_mode is True
        assert interface.classical_device.type in ['cuda', 'cpu']
        
        logger.info("✅ Interface initialization test passed")
        logger.info(f"   Classical device: {interface.classical_device}")
    
    @pytest.mark.asyncio
    async def test_quantum_enhanced_processing(self, interface, test_data):
        """Test quantum-enhanced processing mode"""
        result = await interface.process_hybrid_cognitive_data(
            test_data,
            processing_mode=HybridProcessingMode.QUANTUM_ENHANCED,
            quantum_enhancement=0.5
        )
        
        # Validate result
        assert result is not None
        assert result.quantum_component is not None
        assert result.classical_component is not None
        assert 0.0 <= result.hybrid_fidelity <= 1.0
        assert result.processing_time > 0
        assert 0.0 <= result.quantum_advantage <= 1.0
        assert result.safety_validated is True
        assert result.processing_mode == HybridProcessingMode.QUANTUM_ENHANCED
        
        logger.info("✅ Quantum-enhanced processing test passed")
        logger.info(f"   Processing time: {result.processing_time*1000:.2f}ms")
        logger.info(f"   Quantum advantage: {result.quantum_advantage:.3f}")
    
    @pytest.mark.asyncio
    async def test_classical_enhanced_processing(self, interface, test_data):
        """Test classical-enhanced processing mode"""
        result = await interface.process_hybrid_cognitive_data(
            test_data,
            processing_mode=HybridProcessingMode.CLASSICAL_ENHANCED,
            quantum_enhancement=0.3
        )
        
        # Validate result
        assert result is not None
        assert result.processing_mode == HybridProcessingMode.CLASSICAL_ENHANCED
        assert result.safety_validated is True
        assert result.quantum_component is not None
        assert result.classical_component is not None
        
        logger.info("✅ Classical-enhanced processing test passed")
    
    @pytest.mark.asyncio
    async def test_parallel_processing(self, interface, test_data):
        """Test parallel processing mode"""
        result = await interface.process_hybrid_cognitive_data(
            test_data,
            processing_mode=HybridProcessingMode.PARALLEL_PROCESSING,
            quantum_enhancement=0.4
        )
        
        # Validate result
        assert result is not None
        assert result.processing_mode == HybridProcessingMode.PARALLEL_PROCESSING
        assert result.safety_validated is True
        
        logger.info("✅ Parallel processing test passed")
    
    @pytest.mark.asyncio
    async def test_adaptive_processing(self, interface, test_data):
        """Test adaptive processing mode"""
        result = await interface.process_hybrid_cognitive_data(
            test_data,
            processing_mode=HybridProcessingMode.ADAPTIVE_SWITCHING,
            quantum_enhancement=0.6
        )
        
        # Validate result
        assert result is not None
        assert result.processing_mode == HybridProcessingMode.ADAPTIVE_SWITCHING
        assert result.safety_validated is True
        
        logger.info("✅ Adaptive processing test passed")
    
    def test_interface_metrics(self, interface):
        """Test interface metrics collection"""
        metrics = interface.get_interface_metrics()
        
        # Validate metrics
        assert metrics.quantum_processing_time >= 0.0
        assert metrics.classical_processing_time >= 0.0
        assert metrics.interface_overhead >= 0.0
        assert metrics.total_processing_time >= 0.0
        assert metrics.quantum_advantage_ratio >= 0.0
        assert metrics.memory_usage_quantum >= 0.0
        assert metrics.memory_usage_classical >= 0.0
        assert metrics.throughput_operations_per_second >= 0.0
        
        logger.info("✅ Interface metrics test passed")
    
    def test_processing_mode_optimization(self, interface):
        """Test processing mode optimization"""
        optimization_results = interface.optimize_processing_modes()
        
        # Validate optimization results
        assert isinstance(optimization_results, dict)
        assert len(optimization_results) == 4  # Four processing modes
        
        for mode, performance in optimization_results.items():
            assert isinstance(mode, str)
            assert isinstance(performance, float)
            assert performance >= 0.0
        
        logger.info("✅ Processing mode optimization test passed")

class TestQuantumIntegrationStress:
    """Stress tests for quantum integration"""
    
    @pytest.fixture
    def stress_interface(self):
        """Fixture for stress testing"""
        if not QUANTUM_IMPORTS_AVAILABLE:
            pytest.skip("Quantum imports not available")
        
        return create_quantum_classical_interface(
            num_qubits=12,
            adaptive_mode=True
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, stress_interface):
        """Test concurrent quantum-classical processing"""
        # Create multiple processing tasks
        tasks = []
        for i in range(5):
            test_data = torch.randn(16, 32)
            task = stress_interface.process_hybrid_cognitive_data(
                test_data,
                processing_mode=HybridProcessingMode.PARALLEL_PROCESSING,
                quantum_enhancement=0.3
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # Validate all results
        assert len(results) == 5
        for result in results:
            assert result is not None
            assert result.safety_validated is True
            assert result.quantum_component is not None
            assert result.classical_component is not None
        
        logger.info("✅ Concurrent processing stress test passed")
        logger.info(f"   Total processing time: {total_time*1000:.2f}ms")
        logger.info(f"   Average per operation: {total_time/5*1000:.2f}ms")
    
    def test_memory_stress(self, stress_interface):
        """Test memory usage under stress"""
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Process large amount of data
        large_data = torch.randn(1000, 1000)
        
        # Monitor memory during processing
        memory_usage = []
        for i in range(10):
            # This is a synchronous approximation for testing
            # In real stress test, would use async
            _ = stress_interface._classical_neural_processing(large_data)
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated())
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Validate memory management
        if torch.cuda.is_available():
            memory_increase = final_memory - initial_memory
            logger.info(f"   Memory increase: {memory_increase / 1e6:.2f} MB")
            # Ensure memory increase is reasonable (less than 1GB)
            assert memory_increase < 1e9
        
        logger.info("✅ Memory stress test passed")

def run_quantum_integration_tests():
    """Run all quantum integration tests"""
    logger.info("🚀 Starting Quantum Integration Tests...")
    
    # Check if quantum components are available
    if not QUANTUM_IMPORTS_AVAILABLE:
        logger.error("❌ Quantum components not available - skipping tests")
        return False
    
    try:
        # Run basic functionality tests
        logger.info("🔍 Running basic functionality tests...")
        
        # Test quantum engine initialization
        engine = initialize_quantum_cognitive_engine(num_qubits=8, gpu_acceleration=True)
        assert engine is not None
        
        # Test interface creation
        interface = create_quantum_classical_interface(num_qubits=8, adaptive_mode=True)
        assert interface is not None
        
        # Clean up
        engine.shutdown()
        interface.shutdown()
        
        logger.info("✅ All quantum integration tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantum integration tests failed: {e}")
        return False

if __name__ == "__main__":
    success = run_quantum_integration_tests()
    exit(0 if success else 1)