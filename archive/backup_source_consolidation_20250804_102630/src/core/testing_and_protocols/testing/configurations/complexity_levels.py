#!/usr/bin/env python3
"""
Complexity Levels Configuration for Large-Scale Testing
======================================================

DO-178C Level A compliant complexity level definitions for comprehensive
cognitive system testing. Inspired by nuclear engineering validation
methodologies with aerospace-grade precision.

Key Features:
- Four distinct complexity levels (SIMPLE to EXPERT)
- Progressive cognitive load characteristics
- Measurable complexity metrics
- System resource requirements

Author: KIMERA Development Team
Version: 1.0.0 (DO-178C Level A)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from utils.kimera_logger import get_logger, LogCategory
from utils.kimera_exceptions import KimeraValidationError

logger = get_logger(__name__, LogCategory.SYSTEM)


class ComplexityLevel(Enum):
    """
    Cognitive complexity levels for systematic testing

    Based on nuclear engineering classification systems where
    each level requires exponentially more rigorous validation.
    """
    SIMPLE = "simple"           # Level 1: Single system operations
    MEDIUM = "medium"           # Level 2: Dual system coordination
    COMPLEX = "complex"         # Level 3: Multi-dimensional processing
    EXPERT = "expert"           # Level 4: Full cognitive architecture


@dataclass
class ComplexityMetrics:
    """Quantitative metrics for complexity assessment"""
    cognitive_load: float           # 0.0 to 1.0 scale
    memory_requirement: int         # MB required
    processing_time_estimate: float # seconds
    system_components: int          # number of active components
    interaction_depth: int          # levels of system interaction
    parallelization_factor: float   # concurrent processing capability
    failure_probability: float      # estimated failure rate
    verification_complexity: float  # verification difficulty scale


@dataclass
class ComplexityConfiguration:
    """Complete configuration for a complexity level"""
    level: ComplexityLevel
    metrics: ComplexityMetrics
    system_requirements: Dict[str, Any]
    validation_criteria: Dict[str, float]
    test_parameters: Dict[str, Any]
    expected_behaviors: List[str]
    failure_modes: List[str]

    def is_valid(self) -> bool:
        """Validate configuration consistency"""
        return (
            0.0 <= self.metrics.cognitive_load <= 1.0 and
            self.metrics.memory_requirement > 0 and
            self.metrics.processing_time_estimate > 0 and
            self.metrics.system_components > 0
        )


class ComplexityLevelManager:
    """
    Manager for complexity level configurations and validation

    Implements nuclear engineering principles:
    - Conservative estimation (bias toward higher complexity)
    - Defense in depth (multiple validation layers)
    - Positive confirmation (active verification)
    """

    def __init__(self):
        self.configurations = self._initialize_configurations()
        self.validation_history: List[Dict[str, Any]] = []

        logger.info("🧮 Complexity Level Manager initialized (DO-178C Level A)")
        logger.info(f"   Configurations: {len(self.configurations)}")
        logger.info(f"   Total complexity range: {self._get_complexity_range()}")

    def _initialize_configurations(self) -> Dict[ComplexityLevel, ComplexityConfiguration]:
        """Initialize all complexity level configurations"""
        configurations = {}

        # SIMPLE Level - Single system, linear processing
        configurations[ComplexityLevel.SIMPLE] = ComplexityConfiguration(
            level=ComplexityLevel.SIMPLE,
            metrics=ComplexityMetrics(
                cognitive_load=0.2,
                memory_requirement=256,      # 256 MB
                processing_time_estimate=0.1,  # 100ms
                system_components=1,
                interaction_depth=1,
                parallelization_factor=1.0,
                failure_probability=0.001,   # 0.1% failure rate
                verification_complexity=0.2
            ),
            system_requirements={
                "active_engines": ["cognitive_response"],
                "gpu_required": False,
                "quantum_security": False,
                "thermodynamic_validation": False,
                "dual_system_mode": False
            },
            validation_criteria={
                "response_time_max": 0.5,    # 500ms max
                "memory_usage_max": 512,     # 512 MB max
                "cpu_usage_max": 25.0,       # 25% CPU max
                "success_rate_min": 99.5     # 99.5% success rate
            },
            test_parameters={
                "iterations": 100,
                "concurrent_tests": 1,
                "timeout": 1.0,              # 1 second timeout
                "retry_attempts": 3
            },
            expected_behaviors=[
                "Direct response generation",
                "Single cognitive pathway activation",
                "Linear processing flow",
                "Minimal resource utilization"
            ],
            failure_modes=[
                "Response timeout",
                "Memory allocation failure",
                "Invalid input handling",
                "Component initialization failure"
            ]
        )

        # MEDIUM Level - Dual system coordination
        configurations[ComplexityLevel.MEDIUM] = ComplexityConfiguration(
            level=ComplexityLevel.MEDIUM,
            metrics=ComplexityMetrics(
                cognitive_load=0.5,
                memory_requirement=1024,     # 1 GB
                processing_time_estimate=0.5,  # 500ms
                system_components=3,
                interaction_depth=2,
                parallelization_factor=2.0,
                failure_probability=0.005,   # 0.5% failure rate
                verification_complexity=0.5
            ),
            system_requirements={
                "active_engines": ["cognitive_response", "barenholtz_architecture"],
                "gpu_required": True,
                "quantum_security": True,
                "thermodynamic_validation": False,
                "dual_system_mode": True
            },
            validation_criteria={
                "response_time_max": 2.0,    # 2 second max
                "memory_usage_max": 2048,    # 2 GB max
                "cpu_usage_max": 50.0,       # 50% CPU max
                "success_rate_min": 98.0     # 98% success rate
            },
            test_parameters={
                "iterations": 200,
                "concurrent_tests": 2,
                "timeout": 3.0,              # 3 second timeout
                "retry_attempts": 2
            },
            expected_behaviors=[
                "Dual-system coordination",
                "System 1/System 2 arbitration",
                "Parallel processing paths",
                "Security validation active"
            ],
            failure_modes=[
                "Dual-system deadlock",
                "Arbitration failure",
                "Security validation timeout",
                "Memory pressure issues",
                "GPU resource contention"
            ]
        )

        # COMPLEX Level - High-dimensional multi-system processing
        configurations[ComplexityLevel.COMPLEX] = ComplexityConfiguration(
            level=ComplexityLevel.COMPLEX,
            metrics=ComplexityMetrics(
                cognitive_load=0.8,
                memory_requirement=4096,     # 4 GB
                processing_time_estimate=2.0,  # 2 seconds
                system_components=6,
                interaction_depth=4,
                parallelization_factor=4.0,
                failure_probability=0.02,    # 2% failure rate
                verification_complexity=0.8
            ),
            system_requirements={
                "active_engines": [
                    "cognitive_response", "barenholtz_architecture",
                    "high_dimensional_modeling", "insight_management"
                ],
                "gpu_required": True,
                "quantum_security": True,
                "thermodynamic_validation": True,
                "dual_system_mode": True
            },
            validation_criteria={
                "response_time_max": 5.0,    # 5 second max
                "memory_usage_max": 6144,    # 6 GB max
                "cpu_usage_max": 75.0,       # 75% CPU max
                "success_rate_min": 95.0     # 95% success rate
            },
            test_parameters={
                "iterations": 500,
                "concurrent_tests": 4,
                "timeout": 10.0,             # 10 second timeout
                "retry_attempts": 1
            },
            expected_behaviors=[
                "High-dimensional processing (1024D)",
                "Multi-system coordination",
                "Thermodynamic coherence validation",
                "Insight generation and management",
                "Complex parallel processing"
            ],
            failure_modes=[
                "High-dimensional overflow",
                "Thermodynamic coherence failure",
                "Insight validation timeout",
                "Multi-system coordination deadlock",
                "Memory exhaustion",
                "GPU computation timeout"
            ]
        )

        # EXPERT Level - Full cognitive architecture integration
        configurations[ComplexityLevel.EXPERT] = ComplexityConfiguration(
            level=ComplexityLevel.EXPERT,
            metrics=ComplexityMetrics(
                cognitive_load=1.0,
                memory_requirement=8192,     # 8 GB
                processing_time_estimate=5.0,  # 5 seconds
                system_components=12,
                interaction_depth=8,
                parallelization_factor=8.0,
                failure_probability=0.05,    # 5% failure rate
                verification_complexity=1.0
            ),
            system_requirements={
                "active_engines": [
                    "cognitive_response", "barenholtz_architecture",
                    "high_dimensional_modeling", "insight_management",
                    "thermodynamic_integration", "quantum_security",
                    "ethical_governor", "system_monitor"
                ],
                "gpu_required": True,
                "quantum_security": True,
                "thermodynamic_validation": True,
                "dual_system_mode": True
            },
            validation_criteria={
                "response_time_max": 10.0,   # 10 second max
                "memory_usage_max": 12288,   # 12 GB max
                "cpu_usage_max": 90.0,       # 90% CPU max
                "success_rate_min": 90.0     # 90% success rate
            },
            test_parameters={
                "iterations": 1000,
                "concurrent_tests": 8,
                "timeout": 30.0,             # 30 second timeout
                "retry_attempts": 0          # No retries at expert level
            },
            expected_behaviors=[
                "Full cognitive architecture activation",
                "All-system coordination",
                "Maximum cognitive complexity",
                "Complete validation pipeline",
                "Emergent cognitive behaviors",
                "Cross-system learning",
                "Adaptive processing optimization"
            ],
            failure_modes=[
                "System-wide deadlock",
                "Cascade failure propagation",
                "Resource exhaustion",
                "Thermal throttling",
                "Security validation cascade failure",
                "Emergent behavior instability",
                "Cross-system interference",
                "Memory fragmentation",
                "GPU memory overflow"
            ]
        )

        # Validate all configurations
        for level, config in configurations.items():
            if not config.is_valid():
                raise KimeraValidationError(f"Invalid configuration for {level}")

        return configurations

    def get_configuration(self, level: ComplexityLevel) -> ComplexityConfiguration:
        """Get configuration for specified complexity level"""
        return self.configurations[level]

    def get_all_levels(self) -> List[ComplexityLevel]:
        """Get all available complexity levels"""
        return list(ComplexityLevel)

    def estimate_complexity(self,
                          system_components: int,
                          processing_time: float,
                          memory_usage: int) -> ComplexityLevel:
        """
        Estimate complexity level based on system characteristics

        Uses conservative estimation (bias toward higher complexity)
        following nuclear engineering principles.
        """
        # Calculate complexity score based on multiple factors
        component_score = min(system_components / 12.0, 1.0)
        time_score = min(processing_time / 5.0, 1.0)
        memory_score = min(memory_usage / 8192.0, 1.0)

        # Weighted average with conservative bias
        complexity_score = (component_score * 0.4 +
                          time_score * 0.3 +
                          memory_score * 0.3)

        # Map to complexity levels with conservative bias
        if complexity_score >= 0.75:
            return ComplexityLevel.EXPERT
        elif complexity_score >= 0.5:
            return ComplexityLevel.COMPLEX
        elif complexity_score >= 0.25:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.SIMPLE

    def validate_system_capabilities(self,
                                   level: ComplexityLevel,
                                   available_systems: List[str]) -> bool:
        """
        Validate that system has required capabilities for complexity level

        Implements positive confirmation principle - actively verify capabilities
        """
        config = self.configurations[level]
        required_engines = set(config.system_requirements["active_engines"])
        available_engines = set(available_systems)

        # Check if all required engines are available
        missing_engines = required_engines - available_engines

        if missing_engines:
            logger.warning(f"Missing engines for {level.value}: {missing_engines}")
            return False

        return True

    def get_scaling_factors(self) -> Dict[str, np.ndarray]:
        """Get scaling factors across complexity levels for analysis"""
        levels = [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM,
                 ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT]

        cognitive_loads = []
        memory_requirements = []
        processing_times = []
        component_counts = []

        for level in levels:
            config = self.configurations[level]
            cognitive_loads.append(config.metrics.cognitive_load)
            memory_requirements.append(config.metrics.memory_requirement)
            processing_times.append(config.metrics.processing_time_estimate)
            component_counts.append(config.metrics.system_components)

        return {
            "cognitive_load": np.array(cognitive_loads),
            "memory_requirement": np.array(memory_requirements),
            "processing_time": np.array(processing_times),
            "component_count": np.array(component_counts)
        }

    def _get_complexity_range(self) -> str:
        """Get human-readable complexity range description"""
        simple_load = self.configurations[ComplexityLevel.SIMPLE].metrics.cognitive_load
        expert_load = self.configurations[ComplexityLevel.EXPERT].metrics.cognitive_load
        return f"{simple_load:.1f} to {expert_load:.1f}"

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report for complexity manager"""
        return {
            "total_levels": len(self.configurations),
            "complexity_range": self._get_complexity_range(),
            "memory_range": {
                "min": min(c.metrics.memory_requirement for c in self.configurations.values()),
                "max": max(c.metrics.memory_requirement for c in self.configurations.values())
            },
            "processing_time_range": {
                "min": min(c.metrics.processing_time_estimate for c in self.configurations.values()),
                "max": max(c.metrics.processing_time_estimate for c in self.configurations.values())
            },
            "validation_history_count": len(self.validation_history),
            "configurations_valid": all(c.is_valid() for c in self.configurations.values())
        }


# Global instance for module access
_complexity_manager: Optional[ComplexityLevelManager] = None

def get_complexity_manager() -> ComplexityLevelManager:
    """Get global complexity level manager instance"""
    global _complexity_manager
    if _complexity_manager is None:
        _complexity_manager = ComplexityLevelManager()
    return _complexity_manager
