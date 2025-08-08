"""
KIMERA SWM - GPU MANAGEMENT SYSTEM
==================================

Comprehensive GPU detection, configuration, and optimization system for Kimera SWM.
Provides centralized GPU resource management with intelligent fallback to CPU.

Features:
- Automatic GPU detection and capability assessment
- CUDA environment configuration
- Memory management and optimization
- Performance monitoring
- Graceful CPU fallback
- Multi-GPU support
"""

import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class GPUCapabilityLevel(Enum):
    """GPU capability assessment levels"""

    NONE = "none"  # No GPU or CUDA available
    BASIC = "basic"  # Basic GPU with limited memory
    STANDARD = "standard"  # Standard GPU suitable for ML workloads
    HIGH_PERFORMANCE = "high"  # High-performance GPU for intensive tasks
    ENTERPRISE = "enterprise"  # Enterprise-grade GPU with extensive capabilities


class GPUStatus(Enum):
    """Current GPU system status"""

    INITIALIZING = "initializing"
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class GPUDevice:
    """Auto-generated class."""
    pass
    """Information about a GPU device"""

    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    available_memory_gb: float
    utilization_percent: float
    temperature_celsius: float
    power_limit_watts: float
    driver_version: str
    is_available: bool = True
    capability_level: GPUCapabilityLevel = GPUCapabilityLevel.BASIC

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "device_id": self.device_id
            "name": self.name
            "compute_capability": self.compute_capability
            "total_memory_gb": self.total_memory_gb
            "available_memory_gb": self.available_memory_gb
            "utilization_percent": self.utilization_percent
            "temperature_celsius": self.temperature_celsius
            "power_limit_watts": self.power_limit_watts
            "driver_version": self.driver_version
            "is_available": self.is_available
            "capability_level": self.capability_level.value
        }


@dataclass
class GPUConfiguration:
    """Auto-generated class."""
    pass
    """GPU system configuration"""

    enable_gpu: bool = True
    preferred_device_id: Optional[int] = None
    memory_fraction: float = 0.8  # Use 80% of GPU memory
    allow_memory_growth: bool = True
    enable_mixed_precision: bool = True
    optimization_level: str = "balanced"
    fallback_to_cpu: bool = True
    max_batch_size: int = 32
    enable_profiling: bool = False
    cuda_visible_devices: Optional[str] = None
class GPUManager:
    """Auto-generated class."""
    pass
    """Centralized GPU management system for Kimera SWM"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.config = GPUConfiguration()
        self.status = GPUStatus.INITIALIZING
        self.devices: List[GPUDevice] = []
        self.current_device: Optional[GPUDevice] = None
        self.capabilities: Dict[str, Any] = {}

        # Import flags
        self.torch_available = False
        self.cupy_available = False
        self.cuda_available = False

        # Initialize system
        self._setup_cuda_environment()
        self._import_gpu_libraries()
        self._detect_devices()
        self._assess_capabilities()
        self._configure_optimal_settings()

        self._initialized = True
        logger.info("🚀 GPU Manager initialized successfully")

    def _setup_cuda_environment(self) -> None:
        """Setup CUDA environment variables"""
        try:
            # Set CUDA_DEVICE_ORDER to PCI_BUS_ID for consistent ordering
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            # Configure memory management
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

            # Try to detect CUDA path automatically
            possible_cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                r"C:\tools\cuda",
                "/usr/local/cuda",
                "/opt/cuda",
            ]

            if "CUDA_PATH" not in os.environ:
                for cuda_path in possible_cuda_paths:
                    if os.path.exists(cuda_path):
                        # Find the latest version
                        versions = []
                        for item in os.listdir(cuda_path):
                            path = os.path.join(cuda_path, item)
                            if os.path.isdir(path) and item.startswith("v"):
                                versions.append((item, path))

                        if versions:
                            latest_version = sorted(versions)[-1]
                            os.environ["CUDA_PATH"] = latest_version[1]
                            logger.info(
                                f"🔧 Auto-detected CUDA path: {latest_version[1]}"
                            )
                            break

            logger.info("🔧 CUDA environment configured")

        except Exception as e:
            logger.warning(f"⚠️ CUDA environment setup failed: {e}")

    def _import_gpu_libraries(self) -> None:
        """Import GPU libraries with proper error handling"""
        # Try importing PyTorch
        try:
            import torch

            self.torch = torch
            self.torch_available = True
            self.cuda_available = torch.cuda.is_available()
            logger.info(f"✅ PyTorch {torch.__version__} imported successfully")
            logger.info(f"🔧 CUDA available: {self.cuda_available}")

        except ImportError as e:
            logger.warning(f"⚠️ PyTorch import failed: {e}")
            self.torch = None
            self.torch_available = False
            self.cuda_available = False

        # Try importing CuPy
        try:
            import cupy as cp

            self.cupy = cp
            self.cupy_available = True
            logger.info(f"✅ CuPy {cp.__version__} imported successfully")

        except ImportError as e:
            logger.warning(f"⚠️ CuPy import failed: {e}")
            self.cupy = None
            self.cupy_available = False

    def _detect_devices(self) -> None:
        """Detect available GPU devices"""
        self.devices = []

        if not self.cuda_available:
            logger.info("📱 No CUDA devices detected - CPU mode enabled")
            self.status = GPUStatus.OFFLINE
            return

        try:
            device_count = self.torch.cuda.device_count()
            logger.info(f"🔍 Detected {device_count} CUDA device(s)")

            for i in range(device_count):
                device_props = self.torch.cuda.get_device_properties(i)

                # Get memory info
                self.torch.cuda.set_device(i)
                memory_info = self.torch.cuda.mem_get_info()
                available_memory = memory_info[0] / (1024**3)  # GB
                total_memory = memory_info[1] / (1024**3)  # GB

                # Create device info
                device = GPUDevice(
                    device_id=i
                    name=device_props.name
                    compute_capability=(device_props.major, device_props.minor),
                    total_memory_gb=total_memory
                    available_memory_gb=available_memory
                    utilization_percent=0.0,  # Will be updated by monitoring
                    temperature_celsius=0.0,  # Will be updated by monitoring
                    power_limit_watts=0.0,  # Will be updated by monitoring
                    driver_version=self.torch.version.cuda or "Unknown",
                )

                # Assess capability level
                device.capability_level = self._assess_device_capability(device)

                self.devices.append(device)
                logger.info(
                    f"📱 Device {i}: {device.name} ({device.capability_level.value})"
                )
                logger.info(
                    f"   Memory: {device.available_memory_gb:.1f}GB / {device.total_memory_gb:.1f}GB"
                )
                logger.info(
                    f"   Compute: {device.compute_capability[0]}.{device.compute_capability[1]}"
                )

            # Select best device as current
            if self.devices:
                self.current_device = self._select_best_device()
                self.torch.cuda.set_device(self.current_device.device_id)
                self.status = GPUStatus.AVAILABLE
                logger.info(
                    f"🎯 Selected device {self.current_device.device_id}: {self.current_device.name}"
                )

        except Exception as e:
            logger.error(f"❌ GPU device detection failed: {e}")
            self.status = GPUStatus.ERROR

    def _assess_device_capability(self, device: GPUDevice) -> GPUCapabilityLevel:
        """Assess the capability level of a GPU device"""
        # Based on compute capability and memory
        major, minor = device.compute_capability
        memory_gb = device.total_memory_gb

        if major >= 8:  # RTX 30/40 series, A100, etc.
            if memory_gb >= 24:
                return GPUCapabilityLevel.ENTERPRISE
            elif memory_gb >= 10:
                return GPUCapabilityLevel.HIGH_PERFORMANCE
            else:
                return GPUCapabilityLevel.STANDARD
        elif major >= 7:  # RTX 20 series, V100, etc.
            if memory_gb >= 16:
                return GPUCapabilityLevel.HIGH_PERFORMANCE
            else:
                return GPUCapabilityLevel.STANDARD
        elif major >= 6:  # GTX 10 series, etc.
            return GPUCapabilityLevel.BASIC
        else:
            return GPUCapabilityLevel.NONE

    def _select_best_device(self) -> Optional[GPUDevice]:
        """Select the best available GPU device"""
        if not self.devices:
            return None

        # If preferred device specified, try to use it
        if self.config.preferred_device_id is not None:
            for device in self.devices:
                if device.device_id == self.config.preferred_device_id:
                    return device

        # Otherwise select based on capability and available memory
        scored_devices = []
        for device in self.devices:
            capability_score = {
                GPUCapabilityLevel.ENTERPRISE: 100
                GPUCapabilityLevel.HIGH_PERFORMANCE: 80
                GPUCapabilityLevel.STANDARD: 60
                GPUCapabilityLevel.BASIC: 40
                GPUCapabilityLevel.NONE: 0
            }[device.capability_level]

            memory_score = device.available_memory_gb * 10
            total_score = capability_score + memory_score

            scored_devices.append((total_score, device))

        # Return device with highest score
        scored_devices.sort(key=lambda x: x[0], reverse=True)
        return scored_devices[0][1]

    def _assess_capabilities(self) -> None:
        """Assess overall system GPU capabilities"""
        self.capabilities = {
            "cuda_available": self.cuda_available
            "torch_available": self.torch_available
            "cupy_available": self.cupy_available
            "device_count": len(self.devices),
            "mixed_precision_supported": False
            "tensor_cores_available": False
            "multi_gpu_supported": len(self.devices) > 1
        }

        if self.current_device:
            # Check for Tensor Cores (compute capability >= 7.0)
            major, minor = self.current_device.compute_capability
            self.capabilities["tensor_cores_available"] = major >= 7

            # Check mixed precision support
            self.capabilities["mixed_precision_supported"] = (
                self.torch_available and major >= 7
            )

        logger.info(f"🔬 GPU Capabilities: {self.capabilities}")

    def _configure_optimal_settings(self) -> None:
        """Configure optimal GPU settings based on detected hardware"""
        if not self.current_device:
            return

        try:
            # Configure memory management
            if self.torch_available:
                self.torch.cuda.empty_cache()

                # Set memory fraction if specified
                if self.config.memory_fraction < 1.0:
                    total_memory = self.current_device.total_memory_gb * (1024**3)
                    memory_to_use = int(total_memory * self.config.memory_fraction)
                    # Note: PyTorch doesn't have direct memory fraction setting
                    # This would be implemented with custom memory management

            # Configure optimization level
            if self.current_device.capability_level == GPUCapabilityLevel.ENTERPRISE:
                self.config.optimization_level = "maximum"
                self.config.max_batch_size = 64
            elif (
                self.current_device.capability_level
                == GPUCapabilityLevel.HIGH_PERFORMANCE
            ):
                self.config.optimization_level = "aggressive"
                self.config.max_batch_size = 32
            else:
                self.config.optimization_level = "balanced"
                self.config.max_batch_size = 16

            logger.info(
                f"⚙️ Optimized for {self.current_device.capability_level.value} performance"
            )

        except Exception as e:
            logger.warning(f"⚠️ GPU configuration optimization failed: {e}")

    def get_device_info(
        self, device_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get information about a specific device or current device"""
        if device_id is None:
            device = self.current_device
        else:
            device = next((d for d in self.devices if d.device_id == device_id), None)

        return device.to_dict() if device else None

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU system status"""
        return {
            "status": self.status.value
            "cuda_available": self.cuda_available
            "torch_available": self.torch_available
            "cupy_available": self.cupy_available
            "device_count": len(self.devices),
            "current_device": self.get_device_info() if self.current_device else None
            "all_devices": [device.to_dict() for device in self.devices],
            "capabilities": self.capabilities
            "configuration": {
                "optimization_level": self.config.optimization_level
                "max_batch_size": self.config.max_batch_size
                "memory_fraction": self.config.memory_fraction
                "mixed_precision": self.config.enable_mixed_precision
            },
        }

    def switch_device(self, device_id: int) -> bool:
        """Switch to a different GPU device"""
        try:
            device = next((d for d in self.devices if d.device_id == device_id), None)
            if not device:
                logger.error(f"❌ Device {device_id} not found")
                return False

            if self.torch_available:
                self.torch.cuda.set_device(device_id)

            self.current_device = device
            logger.info(f"🔄 Switched to device {device_id}: {device.name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to switch to device {device_id}: {e}")
            return False

    def optimize_for_task(self, task_type: str, **kwargs) -> Dict[str, Any]:
        """Optimize GPU settings for specific task type"""
        optimization_settings = {}

        if task_type == "geoid_processing":
            optimization_settings = {
                "batch_size": min(self.config.max_batch_size, 16),
                "precision": (
                    "mixed"
                    if self.capabilities.get("mixed_precision_supported")
                    else "float32"
                ),
                "memory_efficient": True
            }
        elif task_type == "cognitive_field":
            optimization_settings = {
                "batch_size": min(self.config.max_batch_size, 32),
                "precision": "float32",
                "parallel_streams": self.capabilities.get("device_count", 1),
            }
        elif task_type == "thermodynamic_evolution":
            optimization_settings = {
                "batch_size": min(self.config.max_batch_size, 8),
                "precision": "float64",  # Higher precision for thermodynamic calculations
                "memory_efficient": False
            }

        return optimization_settings

    def clear_cache(self) -> None:
        """Clear GPU memory cache"""
        try:
            if self.torch_available:
                self.torch.cuda.empty_cache()
                logger.info("🧹 GPU memory cache cleared")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clear GPU cache: {e}")

    def is_available(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.status == GPUStatus.AVAILABLE and self.current_device is not None


# Global GPU manager instance
gpu_manager = GPUManager()


# Convenience functions
def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance"""
    return gpu_manager


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available"""
    return gpu_manager.is_available()


def get_device_info() -> Optional[Dict[str, Any]]:
    """Get current GPU device information"""
    return gpu_manager.get_device_info()


def optimize_for_task(task_type: str, **kwargs) -> Dict[str, Any]:
    """Optimize GPU settings for specific task"""
    return gpu_manager.optimize_for_task(task_type, **kwargs)

    def optimize_batch_size(self, base_batch_size: int) -> int:
        """Optimize batch size based on GPU memory"""
        if not self.gpu_available:
            return base_batch_size

        try:
            free_memory = self._get_free_memory()
            total_memory = self._get_total_memory()
            memory_ratio = free_memory / total_memory

            # Scale batch size based on available memory
            if memory_ratio > 0.7:
                return int(base_batch_size * 2.0)
            elif memory_ratio > 0.5:
                return int(base_batch_size * 1.5)
            else:
                return base_batch_size
        except:
            return base_batch_size
