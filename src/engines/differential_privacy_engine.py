"""
KIMERA Differential Privacy Engine
==================================
Phase 1, Week 4: Differential Privacy for Cognitive Data Protection

This module implements differential privacy mechanisms to protect
individual cognitive patterns while maintaining utility for analysis.

Author: KIMERA Team
Date: June 2025
Status: Production-Ready
"""

import numpy as np
import cupy as cp
import torch
from numba import cuda
from typing import Tuple, Optional, Dict, Any, Union, List, Callable
import logging
from dataclasses import dataclass
import math
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrivacyMechanism(Enum):
    """Available privacy mechanisms"""
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    RANDOMIZED_RESPONSE = "randomized_response"


@dataclass
class PrivacyBudget:
    """Privacy budget tracking for differential privacy"""
    epsilon: float  # Privacy parameter (smaller = more private)
    delta: float = 1e-5  # Probability of privacy breach
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    
    @property
    def remaining_epsilon(self) -> float:
        """Get remaining privacy budget"""
        return max(0, self.epsilon - self.consumed_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        """Get remaining delta budget"""
        return max(0, self.delta - self.consumed_delta)
    
    def consume(self, epsilon: float, delta: float = 0):
        """Consume privacy budget"""
        self.consumed_epsilon += epsilon
        self.consumed_delta += delta
        
        if self.consumed_epsilon > self.epsilon:
            logger.warning(f"Privacy budget exceeded: {self.consumed_epsilon} > {self.epsilon}")
    
    def reset(self):
        """Reset consumed budget"""
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0


@dataclass
class CognitivePrivacyConfig:
    """Configuration for cognitive data privacy"""
    # Global privacy parameters
    global_epsilon: float = 1.0
    global_delta: float = 1e-5
    
    # Mechanism-specific parameters
    clip_norm: float = 1.0  # L2 norm clipping for gradients
    noise_multiplier: float = 1.0  # Noise scale relative to sensitivity
    
    # Cognitive-specific parameters
    identity_protection_level: float = 0.9  # Protection for identity features
    memory_protection_level: float = 0.8  # Protection for memory patterns
    thought_protection_level: float = 0.95  # Protection for thought patterns
    
    # Advanced composition
    use_rdp: bool = True  # Use Rényi Differential Privacy
    rdp_orders: List[float] = None
    
    def __post_init__(self):
        if self.rdp_orders is None:
            self.rdp_orders = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 100.0]


class DifferentialPrivacyEngine:
    """GPU-accelerated differential privacy for cognitive data"""
    
    def __init__(self, config: Optional[CognitivePrivacyConfig] = None, device_id: int = 0):
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
"""Initialize differential privacy engine
        
        Args:
            config: Privacy configuration
            device_id: CUDA device ID
        """
        self.config = config or CognitivePrivacyConfig()
        self.device_id = device_id
        cuda.select_device(device_id)
        
        # Initialize privacy budget
        self.budget = PrivacyBudget(
            epsilon=self.config.global_epsilon,
            delta=self.config.global_delta
        )
        
        # RDP accountant for advanced composition
        self.rdp_accountant = self._initialize_rdp_accountant()
        
        logger.info(f"Differential Privacy Engine initialized")
        logger.info(f"Global privacy: ε={self.config.global_epsilon}, δ={self.config.global_delta}")
    
    def _initialize_rdp_accountant(self) -> Dict[float, float]:
        """Initialize Rényi Differential Privacy accountant"""
        return {order: 0.0 for order in self.config.rdp_orders}
    
    @staticmethod
    @cuda.jit
    def laplace_noise_kernel(data, output, scale, n_elements, seed):
        """CUDA kernel for Laplace noise addition"""
        idx = cuda.grid(1)
        
        if idx < n_elements:
            # Generate Laplace noise using inverse CDF method
            # U ~ Uniform(-0.5, 0.5)
            cuda.random.seed(seed)
            u = cuda.random.xoroshiro128p_uniform_float32(seed, idx) - 0.5
            
            # Laplace(0, scale) = -scale * sign(u) * log(1 - 2*|u|)
            if u > 0:
                noise = -scale * math.log(1 - 2 * u)
            else:
                noise = scale * math.log(1 + 2 * u)
            
            output[idx] = data[idx] + noise
    
    @staticmethod
    @cuda.jit
    def gaussian_noise_kernel(data, output, scale, n_elements, seed):
        """CUDA kernel for Gaussian noise addition"""
        idx = cuda.grid(1)
        
        if idx < n_elements:
            # Generate Gaussian noise
            cuda.random.seed(seed)
            noise = cuda.random.xoroshiro128p_normal_float32(seed, idx) * scale
            output[idx] = data[idx] + noise
    
    @staticmethod
    @cuda.jit
    def gradient_clipping_kernel(gradients, clip_norm, n_elements):
        """CUDA kernel for gradient clipping"""
        # Compute L2 norm
        norm_sq = 0.0
        for i in range(n_elements):
            norm_sq += gradients[i] * gradients[i]
        
        norm = math.sqrt(norm_sq)
        
        # Clip if necessary
        if norm > clip_norm:
            scale = clip_norm / norm
            for i in range(n_elements):
                gradients[i] *= scale
    
    @staticmethod
    @cuda.jit
    def exponential_mechanism_kernel(scores, output, epsilon, n_choices, seed):
        """CUDA kernel for exponential mechanism selection"""
        # Compute probabilities
        max_score = scores[0]
        for i in range(1, n_choices):
            if scores[i] > max_score:
                max_score = scores[i]
        
        # Compute exp(epsilon * score / 2) for numerical stability
        sum_exp = 0.0
        for i in range(n_choices):
            scores[i] = math.exp(epsilon * (scores[i] - max_score) / 2.0)
            sum_exp += scores[i]
        
        # Normalize to get probabilities
        for i in range(n_choices):
            scores[i] /= sum_exp
        
        # Sample from categorical distribution
        cuda.random.seed(seed)
        u = cuda.random.xoroshiro128p_uniform_float32(seed, 0)
        
        cumsum = 0.0
        selected = 0
        for i in range(n_choices):
            cumsum += scores[i]
            if u <= cumsum:
                selected = i
                break
        
        output[0] = selected
    
    def add_laplace_noise(self, data: cp.ndarray, sensitivity: float,
                         epsilon: Optional[float] = None) -> cp.ndarray:
        """Add Laplace noise for differential privacy
        
        Args:
            data: Data to protect
            sensitivity: L1 sensitivity of the query
            epsilon: Privacy parameter (uses config if None)
            
        Returns:
            Noisy data
        """
        epsilon = epsilon or self.config.global_epsilon
        scale = sensitivity / epsilon
        
        # Allocate output
        output = cp.zeros_like(data)
        n_elements = data.size
        
        # Configure kernel
        threads_per_block = 256
        blocks_per_grid = (n_elements + threads_per_block - 1) // threads_per_block
        
        # Generate random seed
        seed = int(cp.random.randint(0, 2**31))
        
        # Launch kernel
        self.laplace_noise_kernel[blocks_per_grid, threads_per_block](
            data.ravel(), output.ravel(), scale, n_elements, seed
        )
        
        # Update privacy budget
        self.budget.consume(epsilon)
        
        return output.reshape(data.shape)
    
    def add_gaussian_noise(self, data: cp.ndarray, sensitivity: float,
                          epsilon: Optional[float] = None,
                          delta: Optional[float] = None) -> cp.ndarray:
        """Add Gaussian noise for (ε,δ)-differential privacy
        
        Args:
            data: Data to protect
            sensitivity: L2 sensitivity of the query
            epsilon: Privacy parameter
            delta: Privacy parameter
            
        Returns:
            Noisy data
        """
        epsilon = epsilon or self.config.global_epsilon
        delta = delta or self.config.global_delta
        
        # Compute noise scale using analytic Gaussian mechanism
        c = np.sqrt(2 * np.log(1.25 / delta))
        scale = c * sensitivity / epsilon
        
        # Allocate output
        output = cp.zeros_like(data)
        n_elements = data.size
        
        # Configure kernel
        threads_per_block = 256
        blocks_per_grid = (n_elements + threads_per_block - 1) // threads_per_block
        
        # Generate random seed
        seed = int(cp.random.randint(0, 2**31))
        
        # Launch kernel
        self.gaussian_noise_kernel[blocks_per_grid, threads_per_block](
            data.ravel(), output.ravel(), scale, n_elements, seed
        )
        
        # Update privacy budget
        self.budget.consume(epsilon, delta)
        
        # Update RDP accountant
        if self.config.use_rdp:
            self._update_rdp_accountant(scale / sensitivity)
        
        return output.reshape(data.shape)
    
    def clip_gradients(self, gradients: cp.ndarray, clip_norm: float) -> cp.ndarray:
        """Clip gradients to bound sensitivity
        
        Args:
            gradients: Gradient tensor
            clip_norm: Maximum L2 norm
            
        Returns:
            Clipped gradients
        """
        # Compute current norm
        current_norm = cp.linalg.norm(gradients)
        
        # Clip if necessary
        if current_norm > clip_norm:
            gradients = gradients * (clip_norm / current_norm)
        
        return gradients
    
    def private_aggregation(self, data_list: List[cp.ndarray],
                          aggregation_fn: Callable,
                          sensitivity: float,
                          epsilon: Optional[float] = None) -> cp.ndarray:
        """Perform differentially private aggregation
        
        Args:
            data_list: List of data arrays from different sources
            aggregation_fn: Aggregation function (e.g., mean, sum)
            sensitivity: Sensitivity of aggregation
            epsilon: Privacy parameter
            
        Returns:
            Private aggregated result
        """
        # Aggregate data
        aggregated = aggregation_fn(data_list)
        
        # Add noise based on sensitivity
        private_result = self.add_laplace_noise(aggregated, sensitivity, epsilon)
        
        return private_result
    
    def exponential_mechanism(self, scores: cp.ndarray, sensitivity: float,
                            epsilon: Optional[float] = None) -> int:
        """Select option using exponential mechanism
        
        Args:
            scores: Utility scores for each option
            sensitivity: Sensitivity of scoring function
            epsilon: Privacy parameter
            
        Returns:
            Selected index
        """
        epsilon = epsilon or self.config.global_epsilon
        
        # Allocate output
        output = cp.zeros(1, dtype=cp.int32)
        n_choices = len(scores)
        
        # Generate random seed
        seed = int(cp.random.randint(0, 2**31))
        
        # Launch kernel
        self.exponential_mechanism_kernel[1, 1](
            scores, output, epsilon / sensitivity, n_choices, seed
        )
        
        # Update privacy budget
        self.budget.consume(epsilon)
        
        return int(output[0])
    
    def randomized_response(self, data: cp.ndarray, epsilon: float) -> cp.ndarray:
        """Apply randomized response for local differential privacy
        
        Args:
            data: Binary data (0 or 1)
            epsilon: Privacy parameter
            
        Returns:
            Randomized data
        """
        # Probability of keeping true value
        p = np.exp(epsilon) / (np.exp(epsilon) + 1)
        
        # Generate random mask
        mask = cp.random.random(data.shape) < p
        
        # Flip values based on mask
        randomized = cp.where(mask, data, 1 - data)
        
        return randomized
    
    def private_cognitive_embedding(self, embedding: cp.ndarray,
                                  feature_type: str = "general") -> cp.ndarray:
        """Apply privacy to cognitive embeddings based on feature type
        
        Args:
            embedding: Cognitive embedding vector
            feature_type: Type of cognitive feature
            
        Returns:
            Private embedding
        """
        # Determine protection level based on feature type
        protection_levels = {
            "identity": self.config.identity_protection_level,
            "memory": self.config.memory_protection_level,
            "thought": self.config.thought_protection_level,
            "general": 0.5
        }
        
        protection = protection_levels.get(feature_type, 0.5)
        
        # Scale epsilon based on protection level
        feature_epsilon = self.config.global_epsilon * (1 - protection)
        
        # Compute embedding sensitivity (assume normalized)
        sensitivity = 2.0  # Maximum L2 distance for normalized vectors
        
        # Add noise
        private_embedding = self.add_gaussian_noise(
            embedding, sensitivity, feature_epsilon
        )
        
        # Renormalize
        norm = cp.linalg.norm(private_embedding)
        if norm > 0:
            private_embedding = private_embedding / norm
        
        return private_embedding
    
    def private_histogram(self, data: cp.ndarray, bins: int,
                         range: Tuple[float, float],
                         epsilon: Optional[float] = None) -> cp.ndarray:
        """Compute differentially private histogram
        
        Args:
            data: Data values
            bins: Number of bins
            range: Range of values (min, max)
            epsilon: Privacy parameter
            
        Returns:
            Private histogram counts
        """
        # Compute histogram
        hist, _ = cp.histogram(data, bins=bins, range=range)
        
        # Add Laplace noise (sensitivity = 1 for counting queries)
        private_hist = self.add_laplace_noise(hist.astype(cp.float32), 1.0, epsilon)
        
        # Post-process: ensure non-negative counts
        private_hist = cp.maximum(private_hist, 0)
        
        return private_hist
    
    def _update_rdp_accountant(self, noise_multiplier: float):
        """Update RDP accountant for composition
        
        Args:
            noise_multiplier: Ratio of noise scale to sensitivity
        """
        for order in self.config.rdp_orders:
            # RDP guarantee for Gaussian mechanism
            if noise_multiplier > 0:
                rdp_epsilon = order / (2 * noise_multiplier**2)
                self.rdp_accountant[order] += rdp_epsilon
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """Get total privacy spent
        
        Returns:
            Dictionary with privacy metrics
        """
        result = {
            'epsilon_spent': self.budget.consumed_epsilon,
            'delta_spent': self.budget.consumed_delta,
            'epsilon_remaining': self.budget.remaining_epsilon,
            'delta_remaining': self.budget.remaining_delta
        }
        
        # Convert RDP to (ε,δ) if using RDP
        if self.config.use_rdp and any(self.rdp_accountant.values()):
            eps_rdp, delta_rdp = self._compute_rdp_privacy()
            result['rdp_epsilon'] = eps_rdp
            result['rdp_delta'] = delta_rdp
        
        return result
    
    def _compute_rdp_privacy(self, target_delta: Optional[float] = None) -> Tuple[float, float]:
        """Convert RDP guarantees to (ε,δ)-DP
        
        Args:
            target_delta: Target δ value
            
        Returns:
            Tuple of (epsilon, delta)
        """
        target_delta = target_delta or self.config.global_delta
        
        # Find minimum ε over all orders
        min_epsilon = float('inf')
        
        for order, rdp_eps in self.rdp_accountant.items():
            if rdp_eps > 0:
                # Convert RDP to (ε,δ)-DP
                epsilon = rdp_eps + np.log(1/target_delta) / (order - 1)
                min_epsilon = min(min_epsilon, epsilon)
        
        return min_epsilon, target_delta
    
    def reset_privacy_budget(self):
        """Reset privacy budget and accountants"""
        self.budget.reset()
        self.rdp_accountant = self._initialize_rdp_accountant()
        logger.info("Privacy budget reset")
    
    def benchmark_privacy_mechanisms(self) -> Dict[str, Any]:
        """Benchmark different privacy mechanisms
        
        Returns:
            Performance metrics
        """
        import time
from ..utils.config import get_api_settings
from ..config.settings import get_settings
        
        results = {}
        sizes = [1000, 10000, 100000, 1000000]
        
        for size in sizes:
            # Test data
            data = cp.random.randn(size).astype(cp.float32)
            
            # Benchmark Laplace mechanism
            start = time.time()
            _ = self.add_laplace_noise(data, sensitivity=1.0, epsilon=1.0)
            laplace_time = time.time() - start
            
            # Benchmark Gaussian mechanism
            start = time.time()
            _ = self.add_gaussian_noise(data, sensitivity=1.0, epsilon=1.0)
            gaussian_time = time.time() - start
            
            # Benchmark gradient clipping
            start = time.time()
            _ = self.clip_gradients(data, clip_norm=1.0)
            clip_time = time.time() - start
            
            results[f'size_{size}'] = {
                'laplace_ms': laplace_time * 1000,
                'gaussian_ms': gaussian_time * 1000,
                'clip_ms': clip_time * 1000,
                'throughput_gb_s': (size * 4) / (1e9 * laplace_time)
            }
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize engine
    dp_engine = DifferentialPrivacyEngine()
    
    # Test Laplace mechanism
    logger.info("Testing Laplace mechanism...")
    sensitive_data = cp.array([10.5, 20.3, 15.7, 18.2, 22.1], dtype=cp.float32)
    private_data = dp_engine.add_laplace_noise(sensitive_data, sensitivity=5.0, epsilon=1.0)
    logger.info(f"Original: {sensitive_data}")
    logger.info(f"Private: {private_data}")
    logger.info(f"Noise added: {private_data - sensitive_data}")
    
    # Test Gaussian mechanism
    logger.info("\nTesting Gaussian mechanism...")
    cognitive_features = cp.random.randn(10, 64).astype(cp.float32)
    private_features = dp_engine.add_gaussian_noise(
        cognitive_features, sensitivity=1.0, epsilon=2.0
    )
    logger.info(f"Feature shape: {private_features.shape}")
    logger.info(f"Average noise: {cp.mean(cp.abs(private_features - cognitive_features)
    
    # Test exponential mechanism
    logger.info("\nTesting exponential mechanism...")
    utility_scores = cp.array([5.0, 8.0, 6.0, 9.0, 7.0], dtype=cp.float32)
    selected = dp_engine.exponential_mechanism(utility_scores, sensitivity=1.0, epsilon=1.0)
    logger.info(f"Utility scores: {utility_scores}")
    logger.info(f"Selected option: {selected}")
    
    # Test private cognitive embedding
    logger.info("\nTesting private cognitive embedding...")
    identity_embedding = cp.random.randn(256).astype(cp.float32)
    identity_embedding = identity_embedding / cp.linalg.norm(identity_embedding)
    
    private_identity = dp_engine.private_cognitive_embedding(
        identity_embedding, feature_type="identity"
    )
    logger.info(f"Original norm: {cp.linalg.norm(identity_embedding)
    logger.info(f"Private norm: {cp.linalg.norm(private_identity)
    logger.info(f"Cosine similarity: {cp.dot(identity_embedding, private_identity)
    
    # Test private histogram
    logger.info("\nTesting private histogram...")
    data_points = cp.random.normal(50, 15, size=1000)
    private_hist = dp_engine.private_histogram(data_points, bins=10, range=(0, 100))
    logger.info(f"Private histogram counts: {private_hist}")
    
    # Check privacy budget
    logger.info("\nPrivacy budget status:")
    privacy_spent = dp_engine.get_privacy_spent()
    for key, value in privacy_spent.items():
        logger.info(f"  {key}: {value}")
    
    # Benchmark
    logger.info("\nBenchmarking privacy mechanisms...")
    benchmarks = dp_engine.benchmark_privacy_mechanisms()
    for size, metrics in benchmarks.items():
        logger.info(f"\n{size}:")
        logger.info(f"  Laplace: {metrics['laplace_ms']:.2f} ms")
        logger.info(f"  Gaussian: {metrics['gaussian_ms']:.2f} ms")
        logger.info(f"  Throughput: {metrics['throughput_gb_s']:.2f} GB/s")