"""
CUDA Quantum Configuration Module
================================

Configuration management for CUDA Quantum integration with KIMERA,
providing scientific-grade parameter management and hardware optimization.

Configuration Categories:
- Hardware detection and optimization
- Quantum simulation parameters
- Variational algorithm settings
- Performance and memory management
- Cognitive-quantum correlation parameters
- Error handling and fallback strategies

Author: KIMERA Development Team
Version: 1.0.0 - CUDA Quantum Configuration
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
import json

# Configure logging
logger = logging.getLogger(__name__)

# Import Kimera infrastructure
try:
    from .settings import get_settings
    KIMERA_SETTINGS_AVAILABLE = True
except ImportError:
    KIMERA_SETTINGS_AVAILABLE = False
    logger.warning("Kimera settings not available - using standalone configuration")


class QuantumSimulationPrecision(Enum):
    """Quantum simulation precision levels"""
    SINGLE = "single"      # Single precision (faster, less memory)
    DOUBLE = "double"      # Double precision (slower, more memory)
    MIXED = "mixed"        # Mixed precision (balanced)

class QuantumOptimizationLevel(Enum):
    """Quantum circuit optimization levels"""
    NONE = "none"          # No optimization
    BASIC = "basic"        # Basic gate fusion
    STANDARD = "standard"  # Standard optimization
    AGGRESSIVE = "aggressive"  # Maximum optimization

class CognitiveMonitoringLevel(Enum):
    """Cognitive monitoring intensity levels"""
    DISABLED = "disabled"
    MINIMAL = "minimal"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

@dataclass
class HardwareConfiguration:
    """Hardware-specific configuration parameters"""
    # GPU Configuration
    preferred_gpu_backend: str = "nvidia"
    fallback_cpu_backend: str = "qpp-cpu"
    enable_multi_gpu: bool = False
    gpu_memory_fraction: float = 0.8  # Use 80% of available GPU memory
    
    # Memory Management
    max_qubits_single_gpu: int = 25   # Conservative default
    max_qubits_multi_gpu: int = 30    # With multi-GPU
    memory_pool_size_mb: int = 1024   # Memory pool size
    enable_memory_growth: bool = True
    
    # Performance Optimization
    enable_jit_compilation: bool = True
    optimization_level: QuantumOptimizationLevel = QuantumOptimizationLevel.STANDARD
    precision: QuantumSimulationPrecision = QuantumSimulationPrecision.DOUBLE
    
    # Hardware Detection
    auto_detect_capabilities: bool = True
    validation_on_startup: bool = True
    benchmark_on_startup: bool = False

@dataclass
class QuantumSimulationConfiguration:
    """Quantum simulation specific parameters"""
    # Default Simulation Parameters
    default_shots: int = 1024
    max_shots: int = 1000000
    default_seed: Optional[int] = None
    
    # Circuit Compilation
    enable_circuit_optimization: bool = True
    max_circuit_depth: int = 1000
    gate_fusion_threshold: int = 4
    
    # State Vector Simulation
    state_vector_threshold: int = 20  # Max qubits for state vector
    enable_state_caching: bool = True
    cache_size_mb: int = 512
    
    # Measurement and Sampling
    measurement_error_mitigation: bool = False
    readout_error_correction: bool = False
    statistical_error_bars: bool = True
    
    # Noise Modeling
    enable_noise_simulation: bool = False
    default_gate_error_rate: float = 0.001
    default_measurement_error_rate: float = 0.01
    default_decoherence_time: float = 100.0  # microseconds

@dataclass
class VariationalAlgorithmConfiguration:
    """Configuration for variational quantum algorithms"""
    # VQE Parameters
    default_optimizer: str = "BFGS"
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    gradient_method: str = "parameter_shift"
    
    # Parameter Initialization
    parameter_initialization: str = "random"  # "random", "zero", "heuristic"
    parameter_range: tuple = (0.0, 2.0)  # (min, max) for random initialization
    
    # Optimization Strategy
    adaptive_learning_rate: bool = True
    learning_rate_decay: float = 0.99
    momentum: float = 0.9
    
    # Circuit Ansatz
    default_ansatz: str = "hardware_efficient"
    max_layers: int = 10
    entangling_gate: str = "cx"  # "cx", "cz", "iswap"
    
    # Hamiltonian Settings
    pauli_grouping: bool = True
    hamiltonian_simplification: bool = True
    symmetry_reduction: bool = False

@dataclass
class CognitiveQuantumConfiguration:
    """Configuration for cognitive-quantum correlation monitoring"""
    # Monitoring Settings
    monitoring_level: CognitiveMonitoringLevel = CognitiveMonitoringLevel.STANDARD
    assessment_frequency: int = 10  # Every N operations
    
    # Cognitive Metrics
    identity_coherence_threshold: float = 0.95
    memory_continuity_threshold: float = 0.98
    cognitive_drift_threshold: float = 0.02
    reality_testing_threshold: float = 0.85
    
    # Quantum-Cognitive Correlation
    entanglement_coherence_correlation: bool = True
    fidelity_stability_correlation: bool = True
    measurement_bias_detection: bool = True
    
    # Safety Protocols
    emergency_shutdown_threshold: float = 0.80  # For critical cognitive metrics
    graceful_degradation: bool = True
    fallback_classical_mode: bool = True
    
    # Data Collection
    collect_correlation_data: bool = True
    correlation_data_retention_days: int = 30
    anonymize_cognitive_data: bool = True

@dataclass
class PerformanceConfiguration:
    """Performance monitoring and optimization configuration"""
    # Performance Monitoring
    enable_performance_tracking: bool = True
    detailed_profiling: bool = False
    memory_profiling: bool = True
    
    # Benchmarking
    auto_benchmark_circuits: bool = False
    benchmark_frequency_hours: int = 24
    benchmark_suite: List[str] = field(default_factory=lambda: ["ghz", "qft", "vqe_simple"])
    
    # Scaling Analysis
    enable_scaling_analysis: bool = True
    scaling_test_qubits: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])
    
    # Resource Management
    max_memory_usage_gb: float = 32.0
    max_execution_time_seconds: float = 3600.0  # 1 hour
    enable_resource_limits: bool = True
    
    # Reporting
    generate_performance_reports: bool = True
    report_frequency_hours: int = 24
    performance_data_retention_days: int = 90

@dataclass
class ErrorHandlingConfiguration:
    """Error handling and recovery configuration"""
    # Error Detection
    enable_comprehensive_error_checking: bool = True
    circuit_validation: bool = True
    parameter_validation: bool = True
    
    # Recovery Strategies
    auto_retry_on_failure: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    
    # Fallback Mechanisms
    enable_backend_fallback: bool = True
    fallback_chain: List[str] = field(default_factory=lambda: ["nvidia", "qpp-cpu"])
    
    # Error Reporting
    detailed_error_logging: bool = True
    error_context_collection: bool = True
    send_error_reports: bool = False  # Privacy consideration
    
    # Circuit Simplification on Error
    enable_circuit_simplification: bool = True
    simplification_max_attempts: int = 3
    preserve_essential_gates: bool = True

@dataclass
class CUDAQuantumConfiguration:
    """Comprehensive CUDA Quantum configuration"""
    hardware: HardwareConfiguration = field(default_factory=HardwareConfiguration)
    simulation: QuantumSimulationConfiguration = field(default_factory=QuantumSimulationConfiguration)
    variational: VariationalAlgorithmConfiguration = field(default_factory=VariationalAlgorithmConfiguration)
    cognitive: CognitiveQuantumConfiguration = field(default_factory=CognitiveQuantumConfiguration)
    performance: PerformanceConfiguration = field(default_factory=PerformanceConfiguration)
    error_handling: ErrorHandlingConfiguration = field(default_factory=ErrorHandlingConfiguration)
    
    # Global Settings
    debug_mode: bool = False
    verbose_logging: bool = True
    configuration_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'hardware': self.hardware.__dict__,
            'simulation': self.simulation.__dict__,
            'variational': self.variational.__dict__,
            'cognitive': self.cognitive.__dict__,
            'performance': self.performance.__dict__,
            'error_handling': self.error_handling.__dict__,
            'debug_mode': self.debug_mode,
            'verbose_logging': self.verbose_logging,
            'configuration_version': self.configuration_version
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CUDAQuantumConfiguration':
        """Create configuration from dictionary"""
        config = cls()
        
        if 'hardware' in config_dict:
            config.hardware = HardwareConfiguration(**config_dict['hardware'])
        if 'simulation' in config_dict:
            config.simulation = QuantumSimulationConfiguration(**config_dict['simulation'])
        if 'variational' in config_dict:
            config.variational = VariationalAlgorithmConfiguration(**config_dict['variational'])
        if 'cognitive' in config_dict:
            config.cognitive = CognitiveQuantumConfiguration(**config_dict['cognitive'])
        if 'performance' in config_dict:
            config.performance = PerformanceConfiguration(**config_dict['performance'])
        if 'error_handling' in config_dict:
            config.error_handling = ErrorHandlingConfiguration(**config_dict['error_handling'])
        
        config.debug_mode = config_dict.get('debug_mode', False)
        config.verbose_logging = config_dict.get('verbose_logging', True)
        config.configuration_version = config_dict.get('configuration_version', '1.0.0')
        
        return config


class QuantumConfigurationManager:
    """Configuration manager for CUDA Quantum integration"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/quantum_config.yaml")
        self._config: Optional[CUDAQuantumConfiguration] = None
        self._environment_overrides: Dict[str, Any] = {}
        
        # Load configuration
        self._load_configuration()
        self._apply_environment_overrides()
    
    def _load_configuration(self) -> None:
        """Load configuration from file or create default"""
        try:
            if self.config_path.exists():
                logger.info(f"Loading quantum configuration from {self.config_path}")
                
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() == '.yaml':
                        config_dict = yaml.safe_load(f)
                    else:
                        config_dict = json.load(f)
                
                self._config = CUDAQuantumConfiguration.from_dict(config_dict)
                logger.info("âœ… Quantum configuration loaded successfully")
                
            else:
                logger.info("No configuration file found, creating default configuration")
                self._config = CUDAQuantumConfiguration()
                self.save_configuration()
                
        except Exception as e:
            logger.error(f"Failed to load quantum configuration: {e}")
            logger.info("Using default configuration")
            self._config = CUDAQuantumConfiguration()
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides"""
        # Hardware overrides
        if 'KIMERA_QUANTUM_GPU_BACKEND' in os.environ:
            self._config.hardware.preferred_gpu_backend = os.environ['KIMERA_QUANTUM_GPU_BACKEND']
        
        if 'KIMERA_QUANTUM_GPU_MEMORY_FRACTION' in os.environ:
            try:
                self._config.hardware.gpu_memory_fraction = float(os.environ['KIMERA_QUANTUM_GPU_MEMORY_FRACTION'])
            except ValueError:
                logger.warning("Invalid GPU memory fraction in environment variable")
        
        # Simulation overrides
        if 'KIMERA_QUANTUM_DEFAULT_SHOTS' in os.environ:
            try:
                self._config.simulation.default_shots = int(os.environ['KIMERA_QUANTUM_DEFAULT_SHOTS'])
            except ValueError:
                logger.warning("Invalid default shots in environment variable")
        
        # Debug mode override
        if 'KIMERA_QUANTUM_DEBUG' in os.environ:
            self._config.debug_mode = os.environ['KIMERA_QUANTUM_DEBUG'].lower() in ['true', '1', 'yes']
        
        # Cognitive monitoring override
        if 'KIMERA_QUANTUM_COGNITIVE_MONITORING' in os.environ:
            monitoring_level = os.environ['KIMERA_QUANTUM_COGNITIVE_MONITORING'].lower()
            if monitoring_level in [level.value for level in CognitiveMonitoringLevel]:
                self._config.cognitive.monitoring_level = CognitiveMonitoringLevel(monitoring_level)
        
        logger.info("Environment variable overrides applied")
    
    @property
    def config(self) -> CUDAQuantumConfiguration:
        """Get current configuration"""
        return self._config
    
    def update_configuration(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        try:
            # Create new configuration with updates
            current_dict = self._config.to_dict()
            
            # Apply updates (deep merge)
            self._deep_merge(current_dict, updates)
            
            # Create new configuration
            self._config = CUDAQuantumConfiguration.from_dict(current_dict)
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise
    
    def save_configuration(self, path: Optional[Path] = None) -> None:
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() == '.yaml':
                    yaml.dump(self._config.to_dict(), f, default_flow_style=False, indent=2)
                else:
                    json.dump(self._config.to_dict(), f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_hardware_config(self) -> HardwareConfiguration:
        """Get hardware configuration"""
        return self._config.hardware
    
    def get_simulation_config(self) -> QuantumSimulationConfiguration:
        """Get simulation configuration"""
        return self._config.simulation
    
    def get_variational_config(self) -> VariationalAlgorithmConfiguration:
        """Get variational algorithm configuration"""
        return self._config.variational
    
    def get_cognitive_config(self) -> CognitiveQuantumConfiguration:
        """Get cognitive monitoring configuration"""
        return self._config.cognitive
    
    def get_performance_config(self) -> PerformanceConfiguration:
        """Get performance configuration"""
        return self._config.performance
    
    def get_error_handling_config(self) -> ErrorHandlingConfiguration:
        """Get error handling configuration"""
        return self._config.error_handling
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep merge dictionaries"""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Hardware validation
        hw = self._config.hardware
        if not 0.1 <= hw.gpu_memory_fraction <= 1.0:
            issues.append("GPU memory fraction must be between 0.1 and 1.0")
        
        if hw.max_qubits_single_gpu <= 0:
            issues.append("Maximum qubits for single GPU must be positive")
        
        # Simulation validation
        sim = self._config.simulation
        if sim.default_shots <= 0:
            issues.append("Default shots must be positive")
        
        if sim.max_shots < sim.default_shots:
            issues.append("Maximum shots must be >= default shots")
        
        # Variational algorithm validation
        var = self._config.variational
        if var.max_iterations <= 0:
            issues.append("Maximum iterations must be positive")
        
        if var.convergence_tolerance <= 0:
            issues.append("Convergence tolerance must be positive")
        
        # Cognitive monitoring validation
        cog = self._config.cognitive
        if not 0.0 <= cog.identity_coherence_threshold <= 1.0:
            issues.append("Identity coherence threshold must be between 0.0 and 1.0")
        
        return issues
    
    def get_optimized_configuration_for_hardware(self) -> CUDAQuantumConfiguration:
        """Get configuration optimized for detected hardware"""
        # This would be enhanced with actual hardware detection
        optimized = CUDAQuantumConfiguration.from_dict(self._config.to_dict())
        
        # Example optimizations based on detected hardware
        # (In real implementation, this would use actual hardware detection)
        
        # Placeholder optimizations
        logger.info("Applying hardware-optimized configuration")
        
        return optimized


# Global configuration manager instance
_config_manager: Optional[QuantumConfigurationManager] = None

def get_quantum_config() -> CUDAQuantumConfiguration:
    """Get global quantum configuration"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = QuantumConfigurationManager()
    
    return _config_manager.config

def get_quantum_config_manager() -> QuantumConfigurationManager:
    """Get global quantum configuration manager"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = QuantumConfigurationManager()
    
    return _config_manager

def initialize_quantum_configuration(config_path: Optional[Path] = None) -> QuantumConfigurationManager:
    """Initialize quantum configuration system"""
    global _config_manager
    
    logger.info("Initializing quantum configuration system...")
    
    _config_manager = QuantumConfigurationManager(config_path)
    
    # Validate configuration
    issues = _config_manager.validate_configuration()
    if issues:
        logger.warning("Configuration validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("âœ… Configuration validation passed")
    
    logger.info("âœ… Quantum configuration system initialized")
    return _config_manager


# Export main classes and functions
__all__ = [
    'CUDAQuantumConfiguration',
    'HardwareConfiguration',
    'QuantumSimulationConfiguration', 
    'VariationalAlgorithmConfiguration',
    'CognitiveQuantumConfiguration',
    'PerformanceConfiguration',
    'ErrorHandlingConfiguration',
    'QuantumConfigurationManager',
    'get_quantum_config',
    'get_quantum_config_manager',
    'initialize_quantum_configuration'
] 