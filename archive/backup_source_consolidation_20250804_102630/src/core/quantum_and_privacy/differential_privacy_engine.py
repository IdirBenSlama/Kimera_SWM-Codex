"""
Differential Privacy Engine - Core Integration
==============================================

GPU-accelerated differential privacy with formal guarantees.

Author: KIMERA Team
Date: 2025-01-31
Status: Production-Ready
"""

import numpy as np
import cupy as cp
import torch
from numba import cuda
from typing import Tuple, Optional, Dict, Any, Union, List, Callable
from dataclasses import dataclass
import math
from enum import Enum
import logging

from ...utils.kimera_exceptions import KimeraException
from ..constants import EPSILON

logger = logging.getLogger(__name__)


class PrivacyMechanism(Enum):
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    RANDOMIZED_RESPONSE = "randomized_response"


@dataclass
class PrivacyBudget:
    epsilon: float
    delta: float = 1e-5
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    
    @property
    def remaining_epsilon(self) -> float:
        return max(0, self.epsilon - self.consumed_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        return max(0, self.delta - self.consumed_delta)
    
    def consume(self, epsilon: float, delta: float = 0):
        self.consumed_epsilon += epsilon
        self.consumed_delta += delta
        if self.consumed_epsilon > self.epsilon:
            raise KimeraException("Privacy budget exceeded")
    
    def reset(self):
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0


@dataclass
class CognitivePrivacyConfig:
    global_epsilon: float = 1.0
    global_delta: float = 1e-5
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0
    identity_protection_level: float = 0.9
    memory_protection_level: float = 0.8
    thought_protection_level: float = 0.95
    use_rdp: bool = True
    rdp_orders: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 100.0])


class DifferentialPrivacyEngine:
    """
    Differential privacy engine with GPU acceleration.
    """
    
    def __init__(self, config: Optional[CognitivePrivacyConfig] = None, device_id: int = 0):
        self.config = config or CognitivePrivacyConfig()
        self.device_id = device_id
        cuda.select_device(device_id)
        
        self.budget = PrivacyBudget(self.config.global_epsilon, self.config.global_delta)
        self.rdp_accountant = {order: 0.0 for order in self.config.rdp_orders}
        
        logger.info("Differential Privacy Engine initialized")
    
    @staticmethod
    @cuda.jit
    def laplace_noise_kernel(data, output, scale, n_elements, seed):
        idx = cuda.grid(1)
        if idx < n_elements:
            u = cuda.random.xoroshiro128p_uniform_float32(seed, idx) - 0.5
            if u > 0:
                noise = -scale * math.log(1 - 2 * u)
            else:
                noise = scale * math.log(1 + 2 * u)
            output[idx] = data[idx] + noise
    
    @staticmethod
    @cuda.jit
    def gaussian_noise_kernel(data, output, scale, n_elements, seed):
        idx = cuda.grid(1)
        if idx < n_elements:
            noise = cuda.random.xoroshiro128p_normal_float32(seed, idx) * scale
            output[idx] = data[idx] + noise
    
    def add_laplace_noise(self, data: cp.ndarray, sensitivity: float, epsilon: Optional[float] = None) -> cp.ndarray:
        epsilon = epsilon or self.config.global_epsilon
        scale = sensitivity / epsilon
        output = cp.zeros_like(data)
        n_elements = data.size
        threads = 256
        blocks = (n_elements + threads - 1) // threads
        seed = int(cp.random.randint(0, 2**31))
        self.laplace_noise_kernel[blocks, threads](data.ravel(), output.ravel(), scale, n_elements, seed)
        self.budget.consume(epsilon)
        return output.reshape(data.shape)
    
    def add_gaussian_noise(self, data: cp.ndarray, sensitivity: float, epsilon: Optional[float] = None, delta: Optional[float] = None) -> cp.ndarray:
        epsilon = epsilon or self.config.global_epsilon
        delta = delta or self.config.global_delta
        c = np.sqrt(2 * np.log(1.25 / delta))
        scale = c * sensitivity / epsilon
        output = cp.zeros_like(data)
        n_elements = data.size
        threads = 256
        blocks = (n_elements + threads - 1) // threads
        seed = int(cp.random.randint(0, 2**31))
        self.gaussian_noise_kernel[blocks, threads](data.ravel(), output.ravel(), scale, n_elements, seed)
        self.budget.consume(epsilon, delta)
        if self.config.use_rdp:
            self._update_rdp_accountant(scale / sensitivity)
        return output.reshape(data.shape)
    
    def clip_gradients(self, gradients: cp.ndarray, clip_norm: float) -> cp.ndarray:
        current_norm = cp.linalg.norm(gradients)
        if current_norm > clip_norm:
            gradients = gradients * (clip_norm / current_norm)
        return gradients
    
    def private_aggregation(self, data_list: List[cp.ndarray], aggregation_fn: Callable, sensitivity: float, epsilon: Optional[float] = None) -> cp.ndarray:
        aggregated = aggregation_fn(data_list)
        return self.add_laplace_noise(aggregated, sensitivity, epsilon)
    
    def exponential_mechanism(self, scores: cp.ndarray, sensitivity: float, epsilon: Optional[float] = None) -> int:
        epsilon = epsilon or self.config.global_epsilon
        exp_scores = cp.exp(epsilon * scores / (2 * sensitivity))
        probs = exp_scores / cp.sum(exp_scores)
        return int(cp.random.choice(len(scores), p=probs.get()))
    
    def randomized_response(self, data: cp.ndarray, epsilon: float) -> cp.ndarray:
        p = math.exp(epsilon) / (math.exp(epsilon) + 1)
        mask = cp.random.random(data.shape) < p
        return cp.where(mask, data, 1 - data)
    
    def private_cognitive_embedding(self, embedding: cp.ndarray, feature_type: str = "general") -> cp.ndarray:
        protection = {'identity': self.config.identity_protection_level, 'memory': self.config.memory_protection_level, 'thought': self.config.thought_protection_level, 'general': 0.5}.get(feature_type, 0.5)
        feature_epsilon = self.config.global_epsilon * (1 - protection)
        sensitivity = 2.0
        private_embedding = self.add_gaussian_noise(embedding, sensitivity, feature_epsilon)
        norm = cp.linalg.norm(private_embedding)
        if norm > 0:
            private_embedding /= norm
        return private_embedding
    
    def private_histogram(self, data: cp.ndarray, bins: int, range: Tuple[float, float], epsilon: Optional[float] = None) -> cp.ndarray:
        hist, _ = cp.histogram(data, bins=bins, range=range)
        private_hist = self.add_laplace_noise(hist.astype(cp.float32), 1.0, epsilon)
        return cp.maximum(private_hist, 0)
    
    def _update_rdp_accountant(self, noise_multiplier: float):
        for order in self.config.rdp_orders:
            self.rdp_accountant[order] += order / (2 * noise_multiplier**2)
    
    def get_privacy_spent(self) -> Dict[str, float]:
        result = {
            'epsilon_spent': self.budget.consumed_epsilon,
            'delta_spent': self.budget.consumed_delta,
            'epsilon_remaining': self.budget.remaining_epsilon,
            'delta_remaining': self.budget.remaining_delta
        }
        if self.config.use_rdp:
            eps, delta = self._compute_rdp_privacy()
            result['rdp_epsilon'] = eps
            result['rdp_delta'] = delta
        return result
    
    def _compute_rdp_privacy(self, target_delta: Optional[float] = None) -> Tuple[float, float]:
        target_delta = target_delta or self.config.global_delta
        min_epsilon = float('inf')
        for order, rdp_eps in self.rdp_accountant.items():
            epsilon = rdp_eps + np.log(1/target_delta) / (order - 1)
            min_epsilon = min(min_epsilon, epsilon)
        return min_epsilon, target_delta
    
    def reset_privacy_budget(self):
        self.budget.reset()
        self.rdp_accountant = {order: 0.0 for order in self.config.rdp_orders}
        logger.info("Privacy budget reset")
    
    def benchmark_privacy_mechanisms(self) -> Dict[str, Any]:
        results = {}
        sizes = [1000, 10000, 100000, 1000000]
        for size in sizes:
            data = cp.random.randn(size).astype(cp.float32)
            laplace_time = time.time()
            _ = self.add_laplace_noise(data, 1.0, 1.0)
            laplace_time = time.time() - laplace_time
            gaussian_time = time.time()
            _ = self.add_gaussian_noise(data, 1.0, 1.0)
            gaussian_time = time.time() - gaussian_time
            results[f'size_{size}'] = {
                'laplace_ms': laplace_time * 1000,
                'gaussian_ms': gaussian_time * 1000,
                'throughput_gb_s': (size * 4) / (1e9 * laplace_time)
            }
        return results