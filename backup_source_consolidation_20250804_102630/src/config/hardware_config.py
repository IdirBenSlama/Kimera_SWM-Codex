"""
Hardware Configuration for Kimera SWM

This file centralizes hardware-specific settings to allow for easier tuning
and adaptation to different GPU environments.
"""

import torch

# --------------------------------------------------------------------------
# Core GPU Settings
# --------------------------------------------------------------------------

# Automatically select CUDA if available, otherwise fallback to CPU.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use mixed-precision (FP16) for performance. Set to False for higher precision.
USE_MIXED_PRECISION = True

# Enable CUDA streams for overlapping compute and data transfer operations.
ENABLE_CUDA_STREAMS = torch.cuda.is_available()

# --------------------------------------------------------------------------
# Memory and Batch Size Configuration
# --------------------------------------------------------------------------

# Main tensor batch size for GPU operations.
# This is a critical parameter to adjust based on available VRAM.
# - 24GB VRAM (e.g., RTX 4090): Can handle 2048 or more.
# - 11GB VRAM (e.g., RTX 2080 Ti): 512 is a safe starting point.
# - Less VRAM: May require 128 or 256.
TENSOR_BATCH_SIZE = 512

# Enable pre-allocated memory pools to reduce memory allocation overhead.
ENABLE_MEMORY_POOLING = True

# Memory prefetch factor.
PREFETCH_FACTOR = 2

# --------------------------------------------------------------------------
# Performance and Optimization Flags
# --------------------------------------------------------------------------

# Enable auto-tuning for adaptive batch sizing and other performance tweaks.
ENABLE_AUTO_TUNING = True

# Use torch.compile for JIT optimization of critical code paths.
COMPILE_MODELS = True

# Leverage Tensor Cores for matrix operations if available.
ENABLE_TENSOR_CORES = True

# --------------------------------------------------------------------------
# Adaptive Optimizer Settings
# --------------------------------------------------------------------------
# These values are for the AdaptiveBatchOptimizer and should be tuned based on the GPU.

# The upper limit for the adaptive batch size.
ADAPTIVE_BATCH_SIZE_MAX = 1024

# The lower limit for the adaptive batch size.
ADAPTIVE_BATCH_SIZE_MIN = 128 