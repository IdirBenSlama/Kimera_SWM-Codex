"""
Triton Kernels and Unsupervised Optimization Module
==================================================

This module provides high-performance cognitive kernels using Triton
and advanced unsupervised test optimization capabilities for the
Kimera SWM system.

Components:
- TritonCognitiveKernels: High-performance GPU kernels for cognitive processing
- UnsupervisedTestOptimization: Advanced self-optimizing test framework

Integration follows DO-178C Level A standards with:
- GPU-accelerated performance optimization
- Self-adapting test methodologies
- Nuclear-grade safety protocols
- Real-time performance monitoring
"""

from .triton_cognitive_kernels import TritonCognitiveKernels
from .unsupervised_test_optimization import UnsupervisedTestOptimization

__all__ = ["TritonCognitiveKernels", "UnsupervisedTestOptimization"]
