"""
KIMERA Security Foundation Test Suite
=====================================
Phase 1, Week 4: Comprehensive Security Testing

This module tests all security components:
- GPU Cryptographic Engine
- Homomorphic Encryption
- Differential Privacy
- Quantum-Resistant Cryptography
- Security Orchestration

Author: KIMERA Team
Date: June 2025
Status: Production Testing
"""

import pytest
import numpy as np
import cupy as cp
import torch
import time
import json
import logging
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engines.gpu_cryptographic_engine import GPUCryptographicEngine
from backend.engines.homomorphic_cognitive_processor import HomomorphicCognitiveProcessor
from backend.engines.differential_privacy_engine import DifferentialPrivacyEngine
from backend.engines.quantum_resistant_crypto import QuantumResistantCrypto
from backend.engines.cognitive_security_orchestrator import (
    CognitiveSecurityOrchestrator, CognitiveSecurityPolicy, 
    SecurityLevel, SecureComputeRequest
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSecurityFoundation:
    """Comprehensive test suite for security foundation"""
    
    @pytest.fixture
    def crypto_engine(self):
        """Initialize GPU cryptographic engine"""
        return GPUCryptographicEngine()
    
    @pytest.fixture
    def he_processor(self):
        """Initialize homomorphic processor"""
        processor = HomomorphicCognitiveProcessor()
        processor.generate_keys()
        return processor
    
    @pytest.fixture
    def dp_engine(self):
        """Initialize differential privacy engine"""
        return DifferentialPrivacyEngine()
    
    @pytest.fixture
    def pqc_engine(self):
        """Initialize quantum-resistant crypto"""
        return QuantumResistantCrypto()
    
    @pytest.fixture
    def security_orchestrator(self):
        """Initialize security orchestrator"""
        policy = CognitiveSecurityPolicy(
            use_homomorphic=True,
            use_quantum_resistant=True
        )
        return CognitiveSecurityOrchestrator(policy)
    
    def test_gpu_encryption(self, crypto_engine):
        """Test GPU-accelerated encryption"""
        logger.info("Testing GPU encryption...")
        
        # Test different data sizes
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            # Generate test data
            data = cp.random.randn(size).astype(cp.float32)
            key = crypto_engine.generate_secure_key(32)
            
            # Encrypt
            start = time.time()
            ciphertext, nonce = crypto_engine.encrypt_cognitive_data(data, key)
            enc_time = time.time() - start
            
            # Verify encryption
            assert ciphertext.shape[0] >= data.size
            assert len(nonce) == 16
            
            # Performance check
            throughput = (data.nbytes / (1024 * 1024)) / enc_time
            logger.info(f"Encryption size {size}: {enc_time*1000:.2f}ms, {throughput:.1f} MB/s")
    
    def test_gpu_hashing(self, crypto_engine):
        """Test GPU-accelerated hashing"""
        logger.info("Testing GPU hashing...")
        
        # Test cognitive state hashing
        cognitive_state = cp.random.randn(100, 64).astype(cp.float32)
        
        # Hash state
        start = time.time()
        hash_value = crypto_engine.hash_cognitive_state(cognitive_state)
        hash_time = time.time() - start
        
        # Verify hash
        assert len(hash_value) == 32  # SHA3-256
        assert cp.all(hash_value >= 0) and cp.all(hash_value <= 255)
        
        # Test determinism
        hash_value2 = crypto_engine.hash_cognitive_state(cognitive_state)
        assert cp.array_equal(hash_value, hash_value2)
        
        logger.info(f"Hashing time: {hash_time*1000:.2f}ms")
    
    def test_stream_encryption(self, crypto_engine):
        """Test ChaCha20 stream encryption"""
        logger.info("Testing stream encryption...")
        
        # Test data
        data = cp.random.randint(0, 256, size=50000, dtype=cp.uint8)
        key = crypto_engine.generate_secure_key(32)
        
        # Encrypt
        ciphertext, nonce = crypto_engine.stream_encrypt(data, key)
        
        # Verify
        assert ciphertext.shape == data.shape
        assert len(nonce) == 12  # ChaCha20 nonce size
        assert not cp.array_equal(ciphertext, data)  # Should be different
        
        logger.info("Stream encryption successful")
    
    def test_homomorphic_encryption(self, he_processor):
        """Test homomorphic encryption operations"""
        logger.info("Testing homomorphic encryption...")
        
        # Test data
        data1 = cp.array([1.5, 2.3, -0.7, 4.2], dtype=cp.float32)
        data2 = cp.array([0.5, 1.7, 2.3, -1.2], dtype=cp.float32)
        
        # Encrypt
        enc1 = he_processor.encrypt_cognitive_tensor(data1)
        enc2 = he_processor.encrypt_cognitive_tensor(data2)
        
        # Test homomorphic addition
        enc_sum = he_processor.add_encrypted(enc1, enc2)
        dec_sum = he_processor.decrypt_cognitive_tensor(enc_sum)
        expected_sum = data1 + data2
        
        error = cp.max(cp.abs(dec_sum[:len(data1)] - expected_sum))
        assert error < 0.01  # Small error tolerance
        
        logger.info(f"Homomorphic addition error: {error:.6f}")
        
        # Test homomorphic multiplication
        enc_prod = he_processor.multiply_encrypted(enc1, enc2)
        dec_prod = he_processor.decrypt_cognitive_tensor(enc_prod)
        expected_prod = data1 * data2
        
        error = cp.max(cp.abs(dec_prod[:len(data1)] - expected_prod))
        assert error < 0.1  # Larger tolerance for multiplication
        
        logger.info(f"Homomorphic multiplication error: {error:.6f}")
    
    def test_privacy_metrics(self, he_processor):
        """Test homomorphic encryption privacy metrics"""
        logger.info("Testing privacy metrics...")
        
        # Encrypt data
        data = cp.random.randn(100).astype(cp.float32)
        encrypted = he_processor.encrypt_cognitive_tensor(data)
        
        # Get metrics
        metrics = he_processor.cognitive_privacy_metrics(encrypted)
        
        # Verify metrics
        assert metrics['noise_budget_bits'] > 0
        assert metrics['multiplication_depth'] >= 0
        assert metrics['ciphertext_size_kb'] > 0
        assert metrics['expansion_ratio'] > 1
        
        logger.info(f"Privacy metrics: {metrics}")
    
    def test_differential_privacy_laplace(self, dp_engine):
        """Test Laplace mechanism for differential privacy"""
        logger.info("Testing Laplace mechanism...")
        
        # Test data
        sensitive_data = cp.array([10, 20, 30, 40, 50], dtype=cp.float32)
        
        # Add Laplace noise
        epsilon = 1.0
        sensitivity = 10.0
        private_data = dp_engine.add_laplace_noise(sensitive_data, sensitivity, epsilon)
        
        # Verify noise was added
        assert not cp.array_equal(private_data, sensitive_data)
        
        # Check noise scale (should be roughly sensitivity/epsilon)
        noise = private_data - sensitive_data
        expected_scale = sensitivity / epsilon
        actual_scale = cp.mean(cp.abs(noise))
        
        # Laplace distribution: mean absolute deviation = scale
        assert 0.5 * expected_scale < actual_scale < 2 * expected_scale
        
        logger.info(f"Laplace noise scale: expected={expected_scale}, actual={actual_scale:.2f}")
    
    def test_differential_privacy_gaussian(self, dp_engine):
        """Test Gaussian mechanism for differential privacy"""
        logger.info("Testing Gaussian mechanism...")
        
        # Test on larger data
        data = cp.random.randn(1000).astype(cp.float32)
        
        # Add Gaussian noise
        epsilon = 2.0
        delta = 1e-5
        sensitivity = 1.0
        
        private_data = dp_engine.add_gaussian_noise(data, sensitivity, epsilon, delta)
        
        # Verify noise properties
        noise = private_data - data
        noise_std = cp.std(noise)
        
        # Expected noise scale
        c = np.sqrt(2 * np.log(1.25 / delta))
        expected_std = c * sensitivity / epsilon
        
        # Check within reasonable bounds
        assert 0.8 * expected_std < noise_std < 1.2 * expected_std
        
        logger.info(f"Gaussian noise std: expected={expected_std:.3f}, actual={noise_std:.3f}")
    
    def test_privacy_budget(self, dp_engine):
        """Test privacy budget tracking"""
        logger.info("Testing privacy budget...")
        
        # Initial budget
        initial_epsilon = dp_engine.budget.epsilon
        
        # Consume some budget
        data = cp.random.randn(100).astype(cp.float32)
        for i in range(5):
            _ = dp_engine.add_laplace_noise(data, sensitivity=1.0, epsilon=0.1)
        
        # Check budget consumption
        consumed = dp_engine.budget.consumed_epsilon
        remaining = dp_engine.budget.remaining_epsilon
        
        assert consumed == 0.5  # 5 * 0.1
        assert remaining == initial_epsilon - 0.5
        
        # Get privacy spent
        spent = dp_engine.get_privacy_spent()
        assert spent['epsilon_spent'] == consumed
        
        logger.info(f"Privacy budget: consumed={consumed}, remaining={remaining}")
    
    def test_quantum_resistant_kyber(self, pqc_engine):
        """Test Kyber encryption (quantum-resistant)"""
        logger.info("Testing Kyber encryption...")
        
        # Generate keys
        pk, sk = pqc_engine.generate_kyber_keypair()
        
        # Test message
        message = cp.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=cp.uint8)
        
        # Encrypt
        start = time.time()
        ciphertext = pqc_engine.kyber_encrypt(message, pk)
        enc_time = time.time() - start
        
        # Decrypt
        start = time.time()
        decrypted = pqc_engine.kyber_decrypt(ciphertext, sk)
        dec_time = time.time() - start
        
        # Verify correctness
        assert cp.array_equal(message, decrypted[:len(message)])
        
        logger.info(f"Kyber: encrypt={enc_time*1000:.2f}ms, decrypt={dec_time*1000:.2f}ms")
    
    def test_quantum_resistant_dilithium(self, pqc_engine):
        """Test Dilithium signatures (quantum-resistant)"""
        logger.info("Testing Dilithium signatures...")
        
        # Generate keys
        pk, sk = pqc_engine.generate_dilithium_keypair()
        
        # Test message
        message = b"Quantum-safe cognitive signature test"
        
        # Sign
        start = time.time()
        signature = pqc_engine.dilithium_sign(message, sk)
        sign_time = time.time() - start
        
        # Verify
        start = time.time()
        valid = pqc_engine.dilithium_verify(message, signature, pk)
        verify_time = time.time() - start
        
        assert valid == True
        
        # Test invalid signature
        tampered_message = b"Modified message"
        invalid = pqc_engine.dilithium_verify(tampered_message, signature, pk)
        assert invalid == False
        
        logger.info(f"Dilithium: sign={sign_time*1000:.2f}ms, verify={verify_time*1000:.2f}ms")
    
    def test_security_orchestrator_basic(self, security_orchestrator):
        """Test security orchestrator with basic security"""
        logger.info("Testing security orchestrator - basic level...")
        
        # Test data
        cognitive_data = cp.random.randn(50, 32).astype(cp.float32)
        
        # Create request
        request = SecureComputeRequest(
            operation='cognitive_update',
            data=cognitive_data,
            security_level=SecurityLevel.BASIC
        )
        
        # Process
        result = security_orchestrator.secure_compute(request)
        
        # Verify
        assert result.result.shape == cognitive_data.shape
        assert result.security_metadata['encryption_used'] == True
        assert result.security_metadata['privacy_preserved'] == False
        assert result.performance_metrics['latency_ms'] > 0
        
        logger.info(f"Basic security latency: {result.performance_metrics['latency_ms']:.2f}ms")
    
    def test_security_orchestrator_maximum(self, security_orchestrator):
        """Test security orchestrator with maximum security"""
        logger.info("Testing security orchestrator - maximum level...")
        
        # Test data
        cognitive_data = cp.random.randn(20, 20).astype(cp.float32)
        
        # Create request
        request = SecureComputeRequest(
            operation='identity_embedding',
            data=cognitive_data,
            security_level=SecurityLevel.MAXIMUM
        )
        
        # Process
        result = security_orchestrator.secure_compute(request)
        
        # Verify all security measures
        assert result.security_metadata['encryption_used'] == True
        assert result.security_metadata['privacy_preserved'] == True
        assert result.security_metadata['quantum_resistant'] == True
        
        logger.info(f"Maximum security latency: {result.performance_metrics['latency_ms']:.2f}ms")
    
    def test_secure_federated_aggregation(self, security_orchestrator):
        """Test secure federated aggregation"""
        logger.info("Testing secure federated aggregation...")
        
        # Simulate client updates
        num_clients = 5
        update_size = 100
        
        client_updates = [
            cp.random.randn(update_size).astype(cp.float32) * 0.1
            for _ in range(num_clients)
        ]
        
        # Aggregate securely
        aggregated = security_orchestrator.secure_federated_aggregation(
            client_updates, 'mean'
        )
        
        # Verify
        assert aggregated.shape == (update_size,)
        
        # Check that noise was added (result should differ from simple mean)
        simple_mean = cp.mean(cp.stack(client_updates), axis=0)
        assert not cp.allclose(aggregated, simple_mean)
        
        logger.info("Federated aggregation with privacy successful")
    
    def test_cognitive_integrity(self, security_orchestrator):
        """Test cognitive integrity verification"""
        logger.info("Testing cognitive integrity verification...")
        
        # Valid cognitive state
        valid_state = cp.random.randn(100, 64).astype(cp.float32)
        valid_state = valid_state / cp.linalg.norm(valid_state) * 10  # Normalize
        
        integrity_ok = security_orchestrator.verify_cognitive_integrity(valid_state)
        assert integrity_ok == True
        
        # Invalid state (with NaN)
        invalid_state = valid_state.copy()
        invalid_state[0, 0] = cp.nan
        
        integrity_bad = security_orchestrator.verify_cognitive_integrity(invalid_state)
        assert integrity_bad == False
        
        logger.info("Cognitive integrity verification working correctly")
    
    def test_performance_benchmarks(self, crypto_engine, he_processor, 
                                  dp_engine, pqc_engine, security_orchestrator):
        """Comprehensive performance benchmarking"""
        logger.info("\n=== SECURITY PERFORMANCE BENCHMARKS ===")
        
        results = {
            'crypto': {},
            'homomorphic': {},
            'differential_privacy': {},
            'quantum_resistant': {},
            'orchestrator': {}
        }
        
        # Benchmark GPU crypto
        logger.info("\nBenchmarking GPU cryptography...")
        crypto_bench = crypto_engine.benchmark_crypto_operations()
        results['crypto'] = crypto_bench
        
        # Benchmark homomorphic encryption
        logger.info("\nBenchmarking homomorphic encryption...")
        he_bench = he_processor.benchmark_homomorphic_ops()
        results['homomorphic'] = he_bench
        
        # Benchmark differential privacy
        logger.info("\nBenchmarking differential privacy...")
        dp_bench = dp_engine.benchmark_privacy_mechanisms()
        results['differential_privacy'] = dp_bench
        
        # Benchmark quantum-resistant crypto
        logger.info("\nBenchmarking quantum-resistant crypto...")
        pqc_bench = pqc_engine.benchmark_pqc_operations()
        results['quantum_resistant'] = pqc_bench
        
        # Benchmark orchestrator
        logger.info("\nBenchmarking security orchestrator...")
        orch_bench = security_orchestrator.benchmark_security_operations()
        results['orchestrator'] = orch_bench
        
        return results
    
    def test_end_to_end_security_pipeline(self, security_orchestrator):
        """Test complete security pipeline"""
        logger.info("\n=== END-TO-END SECURITY PIPELINE TEST ===")
        
        # Step 1: Create secure session
        logger.info("Step 1: Creating secure session...")
        session = security_orchestrator.create_secure_session(
            "test_e2e_session",
            SecurityLevel.MAXIMUM
        )
        assert 'session_id' in session
        
        # Step 2: Process cognitive data with maximum security
        logger.info("Step 2: Processing cognitive data...")
        cognitive_state = cp.random.randn(64, 128).astype(cp.float32)
        
        request = SecureComputeRequest(
            operation='cognitive_state_update',
            data=cognitive_state,
            security_level=SecurityLevel.MAXIMUM
        )
        
        result = security_orchestrator.secure_compute(request)
        assert result.result.shape == cognitive_state.shape
        
        # Step 3: Verify integrity
        logger.info("Step 3: Verifying integrity...")
        integrity_ok = security_orchestrator.verify_cognitive_integrity(result.result)
        assert integrity_ok == True
        
        # Step 4: Get security metrics
        logger.info("Step 4: Checking security metrics...")
        metrics = security_orchestrator.get_security_metrics()
        assert metrics['active_sessions'] == 1
        assert metrics['privacy_budget']['epsilon_remaining'] > 0
        
        logger.info("End-to-end security pipeline test completed successfully!")
        
        return {
            'session': session,
            'processing_latency_ms': result.performance_metrics['latency_ms'],
            'security_level': result.security_metadata,
            'final_metrics': metrics
        }


def run_all_tests():
    """Run all security foundation tests"""
    logger.info("Starting Security Foundation Test Suite...")
    
    # Initialize test class
    test_suite = TestSecurityFoundation()
    
    # Initialize components
    crypto_engine = GPUCryptographicEngine()
    
    he_processor = HomomorphicCognitiveProcessor()
    he_processor.generate_keys()
    
    dp_engine = DifferentialPrivacyEngine()
    pqc_engine = QuantumResistantCrypto()
    
    policy = CognitiveSecurityPolicy(
        use_homomorphic=True,
        use_quantum_resistant=True
    )
    security_orchestrator = CognitiveSecurityOrchestrator(policy)
    
    # Run tests
    try:
        # GPU Cryptography tests
        test_suite.test_gpu_encryption(crypto_engine)
        test_suite.test_gpu_hashing(crypto_engine)
        test_suite.test_stream_encryption(crypto_engine)
        
        # Homomorphic encryption tests
        test_suite.test_homomorphic_encryption(he_processor)
        test_suite.test_privacy_metrics(he_processor)
        
        # Differential privacy tests
        test_suite.test_differential_privacy_laplace(dp_engine)
        test_suite.test_differential_privacy_gaussian(dp_engine)
        test_suite.test_privacy_budget(dp_engine)
        
        # Quantum-resistant crypto tests
        test_suite.test_quantum_resistant_kyber(pqc_engine)
        test_suite.test_quantum_resistant_dilithium(pqc_engine)
        
        # Security orchestrator tests
        test_suite.test_security_orchestrator_basic(security_orchestrator)
        test_suite.test_security_orchestrator_maximum(security_orchestrator)
        test_suite.test_secure_federated_aggregation(security_orchestrator)
        test_suite.test_cognitive_integrity(security_orchestrator)
        
        # Performance benchmarks
        benchmark_results = test_suite.test_performance_benchmarks(
            crypto_engine, he_processor, dp_engine, pqc_engine, security_orchestrator
        )
        
        # End-to-end test
        e2e_results = test_suite.test_end_to_end_security_pipeline(security_orchestrator)
        
        # Save results
        results = {
            'status': 'SUCCESS',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'benchmarks': benchmark_results,
            'end_to_end': e2e_results,
            'security_metrics': security_orchestrator.get_security_metrics()
        }
        
        with open('security_foundation_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n=== ALL SECURITY TESTS PASSED ===")
        logger.info(f"Results saved to security_foundation_test_results.json")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()