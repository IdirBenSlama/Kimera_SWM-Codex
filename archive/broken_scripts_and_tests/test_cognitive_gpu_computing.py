"""
KIMERA Advanced GPU Computing Test Suite
========================================
Phase 1, Week 3: Comprehensive Testing

This module tests all advanced GPU computing components:
- Numba CUDA kernels
- Triton kernels
- CuGraph integration
- Custom GPU algorithms

Author: KIMERA Team
Date: June 2025
Status: Production Testing
"""

import unittest
import torch
import cupy as cp
import numpy as np
import time
import json
from typing import Dict, Any
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engines.cognitive_gpu_kernels import CognitiveGPUKernels
from backend.engines.triton_cognitive_kernels import TritonCognitiveKernels
from backend.engines.cognitive_graph_processor import CognitiveGraphProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCognitiveGPUComputing:
    """Comprehensive test suite for Cognitive GPU computing"""

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources"""
        try:
            cls.gpu_kernels = CognitiveGPUKernels()
            cls.device = cls.gpu_kernels.device
            logger.info(f"Test setup on GPU: {cls.device.name}")
        except Exception as e:
            logger.error(f"Failed to initialize GPU kernels: {e}")
            cls.gpu_kernels = None

    def test_init(self):
        """Test kernel initialization"""
        self.assertIsNotNone(self.gpu_kernels)
        self.assertIsNotNone(self.gpu_kernels.rng_states)

    def test_cognitive_field_transform(self):
        """Test Numba CUDA cognitive field transformation"""
        logger.info("Testing Numba cognitive field transformation...")
        
        # Test different sizes
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            # Create test data
            input_field = cp.random.randn(size).astype(cp.float32)
            
            # Apply transformation
            start_time = time.time()
            output = self.gpu_kernels.apply_cognitive_transform(
                input_field,
                entropy_threshold=0.5,
                coherence_factor=0.95
            )
            elapsed = time.time() - start_time
            
            # Verify output
            assert output.shape == input_field.shape
            assert cp.all(cp.isfinite(output))
            assert cp.all(cp.abs(output) <= 10.0)  # Safety bounds
            
            # Performance metrics
            throughput_gb = (size * 4 * 2) / (1e9 * elapsed)  # 2 arrays, 4 bytes each
            logger.info(f"Size {size}: {elapsed*1000:.2f}ms, {throughput_gb:.1f} GB/s")
    
    def test_numba_attention_mechanism(self):
        """Test Numba CUDA attention computation"""
        logger.info("Testing Numba attention mechanism...")
        
        # Test configurations
        configs = [
            (64, 32),   # (seq_len, d_model)
            (128, 64),
            (256, 128)
        ]
        
        for seq_len, d_model in configs:
            # Create test data
            query = cp.random.randn(seq_len, d_model).astype(cp.float32)
            key = cp.random.randn(seq_len, d_model).astype(cp.float32)
            value = cp.random.randn(seq_len, d_model).astype(cp.float32)
            
            # Compute attention
            start_time = time.time()
            output = self.gpu_kernels.compute_attention(query, key, value, temperature=1.0)
            elapsed = time.time() - start_time
            
            # Verify output
            assert output.shape == value.shape
            assert cp.all(cp.isfinite(output))
            
            logger.info(f"Attention ({seq_len}x{d_model}): {elapsed*1000:.2f}ms")
    
    def test_numba_stochastic_resonance(self):
        """Test Numba CUDA stochastic resonance"""
        logger.info("Testing Numba stochastic resonance...")
        
        # Create weak signal
        t = cp.linspace(0, 10 * cp.pi, 10000)
        weak_signal = cp.sin(t) * 0.1
        
        # Apply stochastic resonance
        enhanced = self.gpu_kernels.apply_stochastic_resonance(
            weak_signal,
            noise_level=0.1,
            threshold=0.5
        )
        
        # Verify enhancement
        enhancement_ratio = cp.std(enhanced) / cp.std(weak_signal)
        assert enhancement_ratio > 1.0
        assert cp.all(cp.isfinite(enhanced))
        
        logger.info(f"Signal enhancement ratio: {enhancement_ratio:.2f}x")
    
    def test_numba_wavelet_analysis(self):
        """Test Numba CUDA wavelet decomposition"""
        logger.info("Testing Numba wavelet analysis...")
        
        # Create test signal with multiple frequencies
        t = cp.linspace(0, 1, 1024)
        signal = cp.sin(2 * cp.pi * 10 * t) + cp.sin(2 * cp.pi * 50 * t)
        
        # Perform wavelet analysis
        coefficients = self.gpu_kernels.wavelet_analysis(signal)
        
        # Verify output
        assert coefficients.ndim == 2
        assert coefficients.shape[1] == signal.size
        assert cp.all(cp.isfinite(coefficients))
        assert cp.all(coefficients >= 0)  # Magnitude should be non-negative
        
        logger.info(f"Wavelet coefficients shape: {coefficients.shape}")
    
    def test_numba_neural_field_dynamics(self):
        """Test Numba CUDA neural field simulation"""
        logger.info("Testing Numba neural field dynamics...")
        
        # Create test network
        n_neurons = 100
        initial_field = cp.random.randn(n_neurons).astype(cp.float32) * 0.1
        
        # Create coupling matrix (Mexican hat connectivity)
        positions = cp.linspace(0, 1, n_neurons)
        distances = cp.abs(positions[:, None] - positions[None, :])
        coupling = cp.exp(-distances**2 / 0.1) - 0.5 * cp.exp(-distances**2 / 0.05)
        coupling = coupling.astype(cp.float32) * 0.1
        
        external_input = cp.random.randn(n_neurons).astype(cp.float32) * 0.5
        
        # Simulate dynamics
        final_field = self.gpu_kernels.simulate_neural_field(
            initial_field, coupling, external_input,
            dt=0.01, tau=10.0, n_steps=100
        )
        
        # Verify stability
        assert final_field.shape == initial_field.shape
        assert cp.all(cp.isfinite(final_field))
        assert cp.std(final_field) > 0  # Should have non-zero activity
        
        logger.info(f"Field evolution: initial std={cp.std(initial_field):.4f}, final std={cp.std(final_field):.4f}")
    
    def test_triton_cognitive_fusion(self):
        """Test Triton cognitive field fusion"""
        logger.info("Testing Triton cognitive field fusion...")
        
        # Test different sizes
        sizes = [1024, 16384, 65536]
        
        for size in sizes:
            # Create test fields
            field_a = torch.randn(size, device='cuda')
            field_b = torch.randn(size, device='cuda')
            
            # Fuse fields
            start_time = time.time()
            fused = TritonCognitiveKernels.cognitive_field_fusion(
                field_a, field_b,
                alpha=0.6, beta=0.4, gamma=1.5
            )
            elapsed = time.time() - start_time
            
            # Verify output
            assert fused.shape == field_a.shape
            assert torch.all(torch.isfinite(fused))
            assert torch.all(torch.abs(fused) <= 10.0)
            
            throughput_gb = (size * 4 * 3) / (1e9 * elapsed)
            logger.info(f"Triton fusion size {size}: {elapsed*1000:.2f}ms, {throughput_gb:.1f} GB/s")
    
    def test_triton_quantum_superposition(self):
        """Test Triton quantum-inspired superposition"""
        logger.info("Testing Triton quantum superposition...")
        
        # Create quantum states
        size = 10000
        state1 = torch.randn(size, device='cuda')
        state2 = torch.randn(size, device='cuda')
        phase = torch.rand(size, device='cuda') * 2 * np.pi
        amplitude = torch.rand(size, device='cuda')
        
        # Apply superposition
        real_part, imag_part = TritonCognitiveKernels.quantum_superposition(
            state1, state2, phase, amplitude
        )
        
        # Verify quantum properties
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        assert torch.all(torch.isfinite(magnitude))
        assert torch.abs(torch.mean(magnitude) - 1.0) < 0.1  # Normalized
        
        logger.info(f"Quantum superposition magnitude: mean={torch.mean(magnitude):.4f}, std={torch.std(magnitude):.4f}")
    
    def test_triton_entropy_attention(self):
        """Test Triton entropy-guided attention"""
        logger.info("Testing Triton entropy-guided attention...")
        
        # Create attention inputs
        seq_len, d_model = 128, 64
        query = torch.randn(seq_len, d_model, device='cuda')
        key = torch.randn(seq_len, d_model, device='cuda')
        value = torch.randn(seq_len, d_model, device='cuda')
        entropy = torch.rand(seq_len, device='cuda')
        
        # Compute attention
        output = TritonCognitiveKernels.entropy_guided_attention(
            query, key, value, entropy,
            temperature=1.0, entropy_weight=0.1
        )
        
        # Verify output
        assert output.shape == value.shape
        assert torch.all(torch.isfinite(output))
        
        logger.info(f"Entropy-guided attention output: mean={torch.mean(output):.4f}, std={torch.std(output):.4f}")
    
    def test_cugraph_network_creation(self):
        """Test CuGraph cognitive network creation"""
        logger.info("Testing CuGraph network creation...")
        
        network_types = ['small_world', 'scale_free', 'random']
        
        for net_type in network_types:
            # Create network
            graph = CognitiveGraphProcessor.create_cognitive_network(
                num_nodes=500,
                connectivity=0.1,
                network_type=net_type
            )
            
            # Verify properties
            assert graph.number_of_vertices() == 500
            assert graph.number_of_edges() > 0
            
            logger.info(f"{net_type} network: {graph.number_of_vertices()} nodes, {graph.number_of_edges()} edges")
    
    def test_cugraph_activation_propagation(self):
        """Test CuGraph activation propagation"""
        logger.info("Testing CuGraph activation propagation...")
        
        # Create network
        graph = CognitiveGraphProcessor.create_cognitive_network(
            num_nodes=200,
            connectivity=0.1,
            network_type='small_world'
        )
        
        # Initial activation
        initial = cp.zeros(200)
        initial[:10] = 1.0
        
        # Propagate
        final = CognitiveGraphProcessor.propagate_activation(initial, steps=20)
        
        # Verify spread
        active_nodes = cp.sum(cp.abs(final) > 0.1)
        assert active_nodes > 10  # Should spread beyond initial nodes
        assert cp.all(cp.isfinite(final))
        
        logger.info(f"Activation spread to {active_nodes} nodes")
    
    def test_cugraph_community_detection(self):
        """Test CuGraph community detection"""
        logger.info("Testing CuGraph community detection...")
        
        # Create network with community structure
        graph = CognitiveGraphProcessor.create_cognitive_network(
            num_nodes=300,
            connectivity=0.05,
            network_type='small_world'
        )
        
        # Detect communities
        communities = CognitiveGraphProcessor.detect_communities(resolution=1.0)
        
        # Verify results
        assert communities['num_communities'] > 1
        assert communities['modularity'] > 0
        assert len(communities['community_stats']) == communities['num_communities']
        
        logger.info(f"Found {communities['num_communities']} communities, modularity={communities['modularity']:.4f}")
    
    def test_cugraph_centrality_measures(self):
        """Test CuGraph centrality computations"""
        logger.info("Testing CuGraph centrality measures...")
        
        # Create network
        graph = CognitiveGraphProcessor.create_cognitive_network(
            num_nodes=100,
            connectivity=0.1,
            network_type='scale_free'
        )
        
        # Compute centrality
        centrality = CognitiveGraphProcessor.compute_centrality_measures()
        
        # Verify all measures computed
        assert 'degree' in centrality
        assert 'betweenness' in centrality
        assert 'pagerank' in centrality
        assert 'eigenvector' in centrality
        
        # Verify properties
        for measure, values in centrality.items():
            assert len(values) == 100
            assert cp.all(cp.isfinite(values))
            assert cp.all(values >= 0)
            
            logger.info(f"{measure} centrality: max={cp.max(values):.4f}, mean={cp.mean(values):.4f}")
    
    def test_cugraph_information_flow(self):
        """Test CuGraph information flow analysis"""
        logger.info("Testing CuGraph information flow...")
        
        # Create network
        graph = CognitiveGraphProcessor.create_cognitive_network(
            num_nodes=150,
            connectivity=0.08,
            network_type='small_world'
        )
        
        # Analyze flow
        flow_analysis = CognitiveGraphProcessor.analyze_information_flow(
            source_nodes=[0, 1, 2],
            time_steps=30
        )
        
        # Verify results
        assert 'activation_history' in flow_analysis
        assert 'entropy_history' in flow_analysis
        assert flow_analysis['activation_history'].shape == (30, 150)
        assert len(flow_analysis['entropy_history']) == 30
        assert flow_analysis['max_entropy'] > 0
        
        logger.info(f"Information flow: max entropy={flow_analysis['max_entropy']:.4f} at t={flow_analysis['convergence_time']}")
    
    def test_performance_benchmarks(self):
        """Comprehensive performance benchmarking"""
        logger.info("\n=== PERFORMANCE BENCHMARKS ===")
        
        results = {
            'numba_cuda': {},
            'triton': {},
            'cugraph': {}
        }
        
        # Benchmark Numba CUDA
        logger.info("\nBenchmarking Numba CUDA kernels...")
        
        # Large-scale cognitive transform
        size = 1_000_000
        input_field = cp.random.randn(size).astype(cp.float32)
        
        # Warmup
        for _ in range(5):
            _ = self.gpu_kernels.apply_cognitive_transform(input_field)
        
        # Benchmark
        cp.cuda.Stream.null.synchronize()
        start = time.time()
        for _ in range(100):
            _ = self.gpu_kernels.apply_cognitive_transform(input_field)
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.time() - start) / 100
        
        results['numba_cuda']['cognitive_transform_1M'] = {
            'time_ms': elapsed * 1000,
            'throughput_gb_s': (size * 4 * 2) / (1e9 * elapsed)
        }
        
        # Benchmark Triton
        logger.info("\nBenchmarking Triton kernels...")
        benchmarks = TritonCognitiveKernels.benchmark_performance()
        results['triton'] = benchmarks
        
        # Benchmark CuGraph
        logger.info("\nBenchmarking CuGraph operations...")
        
        # Large network
        graph = CognitiveGraphProcessor.create_cognitive_network(
            num_nodes=10000,
            connectivity=0.01,
            network_type='scale_free'
        )
        
        # PageRank benchmark
        start = time.time()
        _ = CognitiveGraphProcessor.compute_centrality_measures()
        elapsed = time.time() - start
        
        results['cugraph']['pagerank_10k_nodes'] = {
            'time_ms': elapsed * 1000,
            'nodes_per_second': 10000 / elapsed
        }
        
        # Print results
        logger.info("\n=== BENCHMARK RESULTS ===")
        logger.info(json.dumps(results, indent=2))
        
        return results
    
    def test_integration_pipeline(self):
        """Test integrated pipeline across all components"""
        logger.info("\n=== INTEGRATION PIPELINE TEST ===")
        
        # Step 1: Create cognitive network
        logger.info("Step 1: Creating cognitive network...")
        graph = CognitiveGraphProcessor.create_cognitive_network(
            num_nodes=500,
            connectivity=0.05,
            network_type='small_world'
        )
        
        # Step 2: Apply Numba cognitive transform to node features
        logger.info("Step 2: Applying Numba cognitive transforms...")
        node_features = CognitiveGraphProcessor.node_features['feature_vector']
        transformed_features = self.gpu_kernels.apply_cognitive_transform(
            node_features.ravel(),
            entropy_threshold=0.5,
            coherence_factor=0.95
        ).reshape(node_features.shape)
        
        # Step 3: Convert to PyTorch and apply Triton fusion
        logger.info("Step 3: Applying Triton cognitive fusion...")
        features_torch = torch.tensor(transformed_features.get(), device='cuda')
        random_field = torch.randn_like(features_torch)
        fused_features = TritonCognitiveKernels.cognitive_field_fusion(
            features_torch, random_field,
            alpha=0.7, beta=0.3, gamma=1.2
        )
        
        # Step 4: Propagate activation through network
        logger.info("Step 4: Propagating activation...")
        initial_activation = cp.zeros(500)
        initial_activation[:20] = 1.0
        final_activation = CognitiveGraphProcessor.propagate_activation(
            initial_activation, steps=15
        )
        
        # Step 5: Analyze results
        logger.info("Step 5: Analyzing results...")
        communities = CognitiveGraphProcessor.detect_communities()
        centrality = CognitiveGraphProcessor.compute_centrality_measures()
        
        # Verify integration
        assert cp.all(cp.isfinite(final_activation))
        assert communities['num_communities'] > 0
        assert all(len(v) == 500 for v in centrality.values())
        
        logger.info("Integration pipeline completed successfully!")
        
        return {
            'network_nodes': graph.number_of_vertices(),
            'network_edges': graph.number_of_edges(),
            'communities': communities['num_communities'],
            'modularity': communities['modularity'],
            'activation_spread': float(cp.sum(cp.abs(final_activation) > 0.1)),
            'max_pagerank': float(cp.max(centrality['pagerank']))
        }


def run_all_tests():
    """Run all Cognitive GPU computing tests"""
    logger.info("Starting Cognitive GPU Computing Test Suite...")
    
    # Create an instance of the test class
    test_suite = TestCognitiveGPUComputing()
    
    # Check for GPU availability
    if not test_suite.gpu_kernels:
        logger.error("GPU not available or failed to initialize. Skipping tests.")
        return
        
    # Setup class resources
    test_suite.setUpClass()
    
    # Discover and run tests
    results = {}
    for name in dir(test_suite):
        if name.startswith("test_"):
            test_method = getattr(test_suite, name)
            try:
                logger.info(f"--- Running test: {name} ---")
                test_method()
                results[name] = {"status": "PASSED"}
                logger.info(f"✅ {name} PASSED")
            except Exception as e:
                logger.error(f"❌ {name} FAILED: {e}", exc_info=True)
                results[name] = {"status": "FAILED", "error": str(e)}

    # Save results
    with open('cognitive_gpu_computing_test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info("Cognitive GPU Computing Test Suite finished.")
    logger.info(f"Results saved to cognitive_gpu_computing_test_results.json")


if __name__ == "__main__":
    run_all_tests()