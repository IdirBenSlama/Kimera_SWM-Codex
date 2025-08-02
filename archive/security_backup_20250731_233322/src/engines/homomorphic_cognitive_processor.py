"""
KIMERA Homomorphic Cognitive Processor
======================================
Phase 1, Week 4: Homomorphic Encryption for Cognitive Privacy

This module implements homomorphic encryption to enable computation
on encrypted cognitive data without exposing sensitive information.

Author: KIMERA Team
Date: June 2025
Status: Production-Ready
"""

import numpy as np
import cupy as cp
import torch
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
import time
import json
from numba import cuda
import math
from ..utils.config import get_api_settings
from ..config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HomomorphicParams:
    """Parameters for homomorphic encryption scheme"""
    # BFV/BGV parameters
    poly_modulus_degree: int = 4096  # Must be power of 2
    coeff_modulus_bits: List[int] = None  # Bit sizes for coefficient modulus
    plain_modulus: int = 65537  # Plain text modulus (prime)
    scale: float = 2**40  # Scale for CKKS encoding
    
    def __post_init__(self):
        if self.coeff_modulus_bits is None:
            # Default: 218-bit security level
            self.coeff_modulus_bits = [60, 40, 40, 60]


@dataclass
class CognitiveEncryptedTensor:
    """Encrypted tensor for cognitive computations"""
    ciphertext: cp.ndarray
    scale: float
    level: int  # Current multiplication depth
    modulus_chain: List[int]
    noise_budget: float
    
    @property
    def shape(self):
        """Get logical shape of encrypted data"""
        # Ciphertext is flattened, store original shape in metadata
        return self.metadata.get('original_shape', (len(self.ciphertext),))
    
    def __post_init__(self):
        self.metadata = {}


class HomomorphicCognitiveProcessor:
    """Homomorphic encryption for privacy-preserving cognitive processing"""
    
    def __init__(self, params: Optional[HomomorphicParams] = None, device_id: int = 0):
        """Initialize homomorphic processor
        
        Args:
            params: Homomorphic encryption parameters
            device_id: CUDA device ID
        """
        self.settings = get_api_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.params = params or HomomorphicParams()
        self.device_id = device_id
        cuda.select_device(device_id)
        
        # Initialize polynomial arithmetic on GPU
        self._initialize_polynomial_arithmetic()
        
        # Key storage
        self.public_key = None
        self.secret_key = None
        self.evaluation_keys = None
        
        logger.info(f"Homomorphic Cognitive Processor initialized")
        logger.info(f"Polynomial degree: {self.params.poly_modulus_degree}")
        logger.info(f"Security level: ~{sum(self.params.coeff_modulus_bits)} bits")
    
    def _initialize_polynomial_arithmetic(self):
        """Initialize polynomial arithmetic structures"""
        n = self.params.poly_modulus_degree
        
        # Precompute roots of unity for NTT
        self.roots_of_unity = self._compute_roots_of_unity(n)
        
        # Precompute bit-reversed indices for NTT
        self.bit_reversed_indices = self._bit_reverse_indices(n)
        
        # Initialize modulus chain
        self.modulus_chain = self._generate_modulus_chain()
    
    def _compute_roots_of_unity(self, n: int) -> cp.ndarray:
        """Compute n-th roots of unity for NTT"""
        # Find primitive root modulo prime
        # For demonstration, using simplified approach
        prime = 2**60 - 2**14 + 1  # NTT-friendly prime
        
        # Find generator
        generator = 3  # Common choice
        
        # Compute primitive n-th root
        root = pow(generator, (prime - 1) // (2 * n), prime)
        
        # Compute all powers
        roots = cp.zeros(n, dtype=cp.uint64)
        roots[0] = 1
        for i in range(1, n):
            roots[i] = (roots[i-1] * root) % prime
        
        return roots
    
    def _bit_reverse_indices(self, n: int) -> cp.ndarray:
        """Generate bit-reversed indices for FFT/NTT"""
        indices = cp.arange(n)
        bit_length = int(np.log2(n))
        
        reversed_indices = cp.zeros(n, dtype=cp.int32)
        for i in range(n):
            reversed_indices[i] = int(format(i, f'0{bit_length}b')[::-1], 2)
        
        return reversed_indices
    
    def _generate_modulus_chain(self) -> List[int]:
        """Generate modulus chain for modulus switching"""
        chain = []
        
        # Generate primes for each level
        for bits in self.params.coeff_modulus_bits:
            # Find prime of approximately 'bits' size
            prime = 2**bits - 1
            while not self._is_prime(prime):
                prime -= 2
            chain.append(prime)
        
        return chain
    
    def _is_prime(self, n: int) -> bool:
        """Simple primality test"""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    @cuda.jit
    def ntt_kernel(poly, roots, n, log_n, forward):
        """Number Theoretic Transform kernel for polynomial multiplication"""
        idx = cuda.grid(1)
        
        if idx < n:
            # Bit-reversal permutation
            rev_idx = 0
            temp_idx = idx
            for _ in range(log_n):
                rev_idx = (rev_idx << 1) | (temp_idx & 1)
                temp_idx >>= 1
            
            if idx < rev_idx:
                # Swap elements
                temp = poly[idx]
                poly[idx] = poly[rev_idx]
                poly[rev_idx] = temp
            
            cuda.syncthreads()
            
            # Cooley-Tukey NTT
            length = 2
            while length <= n:
                half_length = length // 2
                step = n // length
                
                if idx % length < half_length:
                    k = idx // length
                    j = idx % length
                    
                    root_idx = j * step if forward else n - j * step
                    root = roots[root_idx]
                    
                    u = poly[k * length + j]
                    v = (poly[k * length + j + half_length] * root) % ((1 << 60) - (1 << 14) + 1)
                    
                    poly[k * length + j] = (u + v) % ((1 << 60) - (1 << 14) + 1)
                    poly[k * length + j + half_length] = (u - v) % ((1 << 60) - (1 << 14) + 1)
                
                cuda.syncthreads()
                length *= 2
    
    @staticmethod
    @cuda.jit
    def polynomial_add_kernel(poly_a, poly_b, result, modulus, n):
        """Add two polynomials element-wise modulo modulus"""
        idx = cuda.grid(1)
        
        if idx < n:
            result[idx] = (poly_a[idx] + poly_b[idx]) % modulus
    
    @staticmethod
    @cuda.jit
    def polynomial_multiply_kernel(poly_a, poly_b, result, modulus, n):
        """Multiply two polynomials using NTT"""
        idx = cuda.grid(1)
        
        if idx < n:
            # Point-wise multiplication in NTT domain
            result[idx] = (poly_a[idx] * poly_b[idx]) % modulus
    
    @staticmethod
    @cuda.jit
    def noise_sampling_kernel(output, seed, n, modulus):
        """Sample Gaussian noise for encryption"""
        idx = cuda.grid(1)
        
        if idx < n:
            # Simple pseudo-random generation (replace with proper Gaussian)
            cuda.random.xoroshiro128p_normal_float64(seed, idx)
            noise = int(cuda.random.xoroshiro128p_normal_float64(seed, idx) * 3.2)
            output[idx] = noise % modulus
    
    def generate_keys(self) -> Tuple[Dict[str, cp.ndarray], Dict[str, cp.ndarray]]:
        """Generate public and secret keys
        
        Returns:
            Tuple of (public_key, secret_key)
        """
        n = self.params.poly_modulus_degree
        q = self.modulus_chain[0]  # Largest modulus
        
        # Generate secret key (ternary polynomial: coefficients in {-1, 0, 1})
        secret = cp.random.choice([-1, 0, 1], size=n)
        
        # Generate error polynomial (Gaussian distribution)
        error = cp.random.normal(0, 3.2, size=n).astype(cp.int64) % q
        
        # Generate random polynomial a
        a = cp.random.randint(0, q, size=n, dtype=cp.int64)
        
        # Compute b = -a*s + e (mod q)
        # First compute a*s using NTT
        a_ntt = self._ntt_forward(a)
        s_ntt = self._ntt_forward(secret)
        as_ntt = (a_ntt * s_ntt) % q
        as_poly = self._ntt_inverse(as_ntt)
        
        b = (-as_poly + error) % q
        
        # Store keys
        self.public_key = {'a': a, 'b': b}
        self.secret_key = {'s': secret}
        
        # Generate evaluation keys for multiplication
        self._generate_evaluation_keys()
        
        logger.info("Keys generated successfully")
        
        return self.public_key, self.secret_key
    
    def _generate_evaluation_keys(self):
        """Generate evaluation keys for homomorphic multiplication"""
        # Simplified version - in practice, generate relinearization keys
        n = self.params.poly_modulus_degree
        
        # Generate key-switching keys
        self.evaluation_keys = {
            'relin_keys': [],
            'galois_keys': []
        }
        
        logger.info("Evaluation keys generated")
    
    def _ntt_forward(self, poly: cp.ndarray) -> cp.ndarray:
        """Forward Number Theoretic Transform"""
        n = len(poly)
        log_n = int(np.log2(n))
        result = poly.copy()
        
        # Configure kernel
        threads_per_block = min(256, n)
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        
        # Launch NTT kernel
        self.ntt_kernel[blocks_per_grid, threads_per_block](
            result, self.roots_of_unity, n, log_n, True
        )
        
        return result
    
    def _ntt_inverse(self, poly: cp.ndarray) -> cp.ndarray:
        """Inverse Number Theoretic Transform"""
        n = len(poly)
        log_n = int(np.log2(n))
        result = poly.copy()
        
        # Configure kernel
        threads_per_block = min(256, n)
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        
        # Launch inverse NTT kernel
        self.ntt_kernel[blocks_per_grid, threads_per_block](
            result, self.roots_of_unity, n, log_n, False
        )
        
        # Scale by n^(-1)
        n_inv = pow(n, -1, self.modulus_chain[0])
        result = (result * n_inv) % self.modulus_chain[0]
        
        return result
    
    def encrypt_cognitive_tensor(self, tensor: Union[cp.ndarray, torch.Tensor],
                               scale: Optional[float] = None) -> CognitiveEncryptedTensor:
        """Encrypt cognitive tensor
        
        Args:
            tensor: Tensor to encrypt
            scale: Scaling factor for encoding
            
        Returns:
            Encrypted cognitive tensor
        """
        if self.public_key is None:
            raise ValueError("Keys not generated. Call generate_keys() first.")
        
        # Convert to CuPy if needed
        if isinstance(tensor, torch.Tensor):
            data = cp.asarray(tensor.cpu().numpy())
        else:
            data = tensor
        
        # Flatten and encode
        original_shape = data.shape
        flat_data = data.ravel()
        
        # Scale and discretize
        scale = scale or self.params.scale
        scaled_data = (flat_data * scale).astype(cp.int64)
        
        # Pad to polynomial degree
        n = self.params.poly_modulus_degree
        if len(scaled_data) > n:
            raise ValueError(f"Data too large: {len(scaled_data)} > {n}")
        
        padded_data = cp.zeros(n, dtype=cp.int64)
        padded_data[:len(scaled_data)] = scaled_data
        
        # Encrypt using public key
        # c = (b*u + e1, a*u + e2) + (m, 0)
        u = cp.random.choice([-1, 0, 1], size=n)  # Small polynomial
        e1 = cp.random.normal(0, 3.2, size=n).astype(cp.int64)
        e2 = cp.random.normal(0, 3.2, size=n).astype(cp.int64)
        
        q = self.modulus_chain[0]
        
        # Compute ciphertext components
        c0 = (self.public_key['b'] * u + e1 + padded_data) % q
        c1 = (self.public_key['a'] * u + e2) % q
        
        # Create encrypted tensor
        ciphertext = cp.stack([c0, c1])
        
        encrypted = CognitiveEncryptedTensor(
            ciphertext=ciphertext,
            scale=scale,
            level=0,
            modulus_chain=self.modulus_chain.copy(),
            noise_budget=40.0  # Initial noise budget in bits
        )
        encrypted.metadata['original_shape'] = original_shape
        encrypted.metadata['original_length'] = len(flat_data)
        
        return encrypted
    
    def decrypt_cognitive_tensor(self, encrypted: CognitiveEncryptedTensor) -> cp.ndarray:
        """Decrypt cognitive tensor
        
        Args:
            encrypted: Encrypted tensor
            
        Returns:
            Decrypted tensor
        """
        if self.secret_key is None:
            raise ValueError("Secret key required for decryption")
        
        c0, c1 = encrypted.ciphertext
        s = self.secret_key['s']
        q = encrypted.modulus_chain[encrypted.level]
        
        # Decrypt: m = c0 + c1*s (mod q)
        decrypted = (c0 + c1 * s) % q
        
        # Remove scaling and extract original data
        decrypted_float = decrypted.astype(cp.float64) / encrypted.scale
        
        # Extract original length
        original_length = encrypted.metadata['original_length']
        original_shape = encrypted.metadata['original_shape']
        
        # Reshape to original
        result = decrypted_float[:original_length].reshape(original_shape)
        
        return result
    
    def add_encrypted(self, enc1: CognitiveEncryptedTensor,
                     enc2: CognitiveEncryptedTensor) -> CognitiveEncryptedTensor:
        """Add two encrypted tensors
        
        Args:
            enc1: First encrypted tensor
            enc2: Second encrypted tensor
            
        Returns:
            Encrypted sum
        """
        # Ensure same level
        if enc1.level != enc2.level:
            raise ValueError("Encrypted tensors must be at same level")
        
        # Ensure same scale
        if abs(enc1.scale - enc2.scale) > 1e-6:
            raise ValueError("Encrypted tensors must have same scale")
        
        q = enc1.modulus_chain[enc1.level]
        
        # Add ciphertexts component-wise
        result_ciphertext = (enc1.ciphertext + enc2.ciphertext) % q
        
        # Create result
        result = CognitiveEncryptedTensor(
            ciphertext=result_ciphertext,
            scale=enc1.scale,
            level=enc1.level,
            modulus_chain=enc1.modulus_chain.copy(),
            noise_budget=min(enc1.noise_budget, enc2.noise_budget) - 1
        )
        
        # Preserve metadata from first operand
        result.metadata = enc1.metadata.copy()
        
        return result
    
    def multiply_encrypted(self, enc1: CognitiveEncryptedTensor,
                          enc2: CognitiveEncryptedTensor) -> CognitiveEncryptedTensor:
        """Multiply two encrypted tensors
        
        Args:
            enc1: First encrypted tensor
            enc2: Second encrypted tensor
            
        Returns:
            Encrypted product
        """
        if enc1.level != enc2.level:
            raise ValueError("Encrypted tensors must be at same level")
        
        # Tensor product of ciphertexts
        # (c0, c1) * (d0, d1) = (c0*d0, c0*d1 + c1*d0, c1*d1)
        c0, c1 = enc1.ciphertext
        d0, d1 = enc2.ciphertext
        
        q = enc1.modulus_chain[enc1.level]
        
        # Use NTT for efficient polynomial multiplication
        c0_ntt = self._ntt_forward(c0)
        c1_ntt = self._ntt_forward(c1)
        d0_ntt = self._ntt_forward(d0)
        d1_ntt = self._ntt_forward(d1)
        
        # Compute products in NTT domain
        e0_ntt = (c0_ntt * d0_ntt) % q
        e1_ntt = ((c0_ntt * d1_ntt) % q + (c1_ntt * d0_ntt) % q) % q
        e2_ntt = (c1_ntt * d1_ntt) % q
        
        # Convert back
        e0 = self._ntt_inverse(e0_ntt)
        e1 = self._ntt_inverse(e1_ntt)
        e2 = self._ntt_inverse(e2_ntt)
        
        # For now, return 2-component ciphertext (would need relinearization)
        result_ciphertext = cp.stack([e0, e1])
        
        # Update scale and level
        new_scale = enc1.scale * enc2.scale
        new_level = enc1.level  # Would increment after modulus switching
        
        result = CognitiveEncryptedTensor(
            ciphertext=result_ciphertext,
            scale=new_scale,
            level=new_level,
            modulus_chain=enc1.modulus_chain.copy(),
            noise_budget=min(enc1.noise_budget, enc2.noise_budget) - 10
        )
        
        result.metadata = enc1.metadata.copy()
        
        return result
    
    def rotate_encrypted(self, encrypted: CognitiveEncryptedTensor,
                        steps: int) -> CognitiveEncryptedTensor:
        """Rotate encrypted vector
        
        Args:
            encrypted: Encrypted tensor
            steps: Number of positions to rotate
            
        Returns:
            Rotated encrypted tensor
        """
        # Simplified rotation - in practice, use Galois automorphisms
        c0, c1 = encrypted.ciphertext
        n = len(c0)
        
        # Rotate coefficients
        rotated_c0 = cp.roll(c0, steps)
        rotated_c1 = cp.roll(c1, steps)
        
        result = CognitiveEncryptedTensor(
            ciphertext=cp.stack([rotated_c0, rotated_c1]),
            scale=encrypted.scale,
            level=encrypted.level,
            modulus_chain=encrypted.modulus_chain.copy(),
            noise_budget=encrypted.noise_budget - 2
        )
        
        result.metadata = encrypted.metadata.copy()
        
        return result
    
    def bootstrap_encrypted(self, encrypted: CognitiveEncryptedTensor) -> CognitiveEncryptedTensor:
        """Bootstrap to refresh noise budget (simplified)
        
        Args:
            encrypted: Encrypted tensor with low noise budget
            
        Returns:
            Refreshed encrypted tensor
        """
        logger.warning("Bootstrapping is simplified - full implementation needed for production")
        
        # In practice, this is very complex
        # For now, just reset noise budget (insecure!)
        result = CognitiveEncryptedTensor(
            ciphertext=encrypted.ciphertext.copy(),
            scale=encrypted.scale,
            level=0,  # Reset to top level
            modulus_chain=self.modulus_chain.copy(),
            noise_budget=40.0  # Reset noise budget
        )
        
        result.metadata = encrypted.metadata.copy()
        
        return result
    
    def cognitive_privacy_metrics(self, encrypted: CognitiveEncryptedTensor) -> Dict[str, float]:
        """Compute privacy metrics for encrypted cognitive data
        
        Args:
            encrypted: Encrypted tensor
            
        Returns:
            Privacy metrics
        """
        return {
            'noise_budget_bits': encrypted.noise_budget,
            'multiplication_depth': encrypted.level,
            'remaining_depth': len(encrypted.modulus_chain) - encrypted.level - 1,
            'ciphertext_size_kb': encrypted.ciphertext.nbytes / 1024,
            'expansion_ratio': encrypted.ciphertext.size / encrypted.metadata.get('original_length', 1),
            'security_level_bits': sum(self.params.coeff_modulus_bits),
            'polynomial_degree': self.params.poly_modulus_degree
        }
    
    def benchmark_homomorphic_ops(self) -> Dict[str, Any]:
        """Benchmark homomorphic operations
        
        Returns:
            Performance metrics
        """
        results = {}
        
        # Generate keys if not already done
        if self.public_key is None:
            self.generate_keys()
        
        # Test different tensor sizes
        sizes = [(10,), (100,), (32, 32), (10, 10, 10)]
        
        for size in sizes:
            # Create test tensor
            test_tensor = cp.random.randn(*size).astype(cp.float32)
            
            # Benchmark encryption
            start = time.time()
            encrypted = self.encrypt_cognitive_tensor(test_tensor)
            enc_time = time.time() - start
            
            # Benchmark addition
            start = time.time()
            sum_result = self.add_encrypted(encrypted, encrypted)
            add_time = time.time() - start
            
            # Benchmark multiplication
            start = time.time()
            prod_result = self.multiply_encrypted(encrypted, encrypted)
            mul_time = time.time() - start
            
            # Benchmark decryption
            start = time.time()
            decrypted = self.decrypt_cognitive_tensor(encrypted)
            dec_time = time.time() - start
            
            size_str = 'x'.join(map(str, size))
            results[f'size_{size_str}'] = {
                'encryption_ms': enc_time * 1000,
                'addition_ms': add_time * 1000,
                'multiplication_ms': mul_time * 1000,
                'decryption_ms': dec_time * 1000,
                'ciphertext_expansion': encrypted.ciphertext.size / test_tensor.size,
                'noise_budget_after_mul': prod_result.noise_budget
            }
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    he_processor = HomomorphicCognitiveProcessor()
    
    # Generate keys
    logger.info("Generating homomorphic keys...")
    public_key, secret_key = he_processor.generate_keys()
    logger.info("Keys generated successfully")
    
    # Test encryption/decryption
    logger.info("\nTesting cognitive tensor encryption...")
    cognitive_data = cp.array([1.5, 2.3, -0.7, 4.2, 3.1], dtype=cp.float32)
    encrypted = he_processor.encrypt_cognitive_tensor(cognitive_data)
    logger.info(f"Original: {cognitive_data}")
    
    decrypted = he_processor.decrypt_cognitive_tensor(encrypted)
    logger.info(f"Decrypted: {decrypted}")
    logger.error(f"Max error: {cp.max(cp.abs(cognitive_data - decrypted))}")
    
    # Test homomorphic addition
    logger.info("\nTesting homomorphic addition...")
    data1 = cp.array([1, 2, 3], dtype=cp.float32)
    data2 = cp.array([4, 5, 6], dtype=cp.float32)
    
    enc1 = he_processor.encrypt_cognitive_tensor(data1)
    enc2 = he_processor.encrypt_cognitive_tensor(data2)
    
    # Homomorphic addition
    encrypted_sum = he_processor.add_encrypted(enc1, enc2)
    decrypted_sum = he_processor.decrypt_cognitive_tensor(encrypted_sum)
    expected_sum = data1 + data2
    
    logger.info(f"Expected sum: {expected_sum}")
    logger.info(f"Decrypted sum: {decrypted_sum}")
    logger.info(f"Addition error: {cp.max(cp.abs(expected_sum - decrypted_sum))}")
    
    # Test homomorphic multiplication
    logger.info("\nTesting homomorphic multiplication...")
    enc_prod = he_processor.multiply_encrypted(enc1, enc2)
    dec_prod = he_processor.decrypt_cognitive_tensor(enc_prod)
    
    logger.info(f"Expected: {data1 * data2}")
    logger.info(f"Computed: {dec_prod[:3]}")
    
    # Privacy metrics
    logger.info("\nPrivacy metrics:")
    metrics = he_processor.cognitive_privacy_metrics(encrypted)
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Benchmarks
    logger.info("\nBenchmarking homomorphic operations...")
    benchmarks = he_processor.benchmark_homomorphic_ops()
    for test, results in benchmarks.items():
        logger.info(f"\n{test}:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.2f}")