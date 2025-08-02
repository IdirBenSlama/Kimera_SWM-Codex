"""
Kimera SWM - Integration Layer
=============================

Integration and Orchestration Systems
This module provides the integration layer that enables seamless
communication, transparency, and performance optimization across
all cognitive systems:

- Interoperability Bus: High-performance component communication
- Transparency Monitor: Complete system observability and tracing
- Performance Optimizer: Efficiency and optimization management
- Architecture Orchestrator: Master system coordination

This layer ensures high interconnectedness, complete transparency,
and maximum efficiency across the entire cognitive architecture.
"""

from .interoperability_bus import (
    CognitiveInteroperabilityBus, 
    MessageRouter, 
    EventStream,
    ComponentRegistry
)
from .transparency_monitor import (
    CognitiveTransparencyMonitor,
    ProcessTracer,
    PerformanceMonitor,
    StateObserver,
    DecisionAuditor
)
from .performance_optimizer import (
    CognitivePerformanceOptimizer,
    GPUOptimizer,
    MemoryOptimizer,
    ParallelProcessor
)
from .architecture_orchestrator import (
    KimeraCoreArchitecture,
    InterconnectionMatrix,
    SystemCoordinator
)

__all__ = [
    'CognitiveInteroperabilityBus',
    'CognitiveTransparencyMonitor',
    'CognitivePerformanceOptimizer',
    'KimeraCoreArchitecture',
    'MessageRouter',
    'EventStream',
    'ComponentRegistry',
    'ProcessTracer',
    'PerformanceMonitor',
    'StateObserver',
    'DecisionAuditor',
    'GPUOptimizer',
    'MemoryOptimizer',
    'ParallelProcessor',
    'InterconnectionMatrix',
    'SystemCoordinator'
]

# Version information
__version__ = "1.0.0"
__status__ = "Production"
__architecture_tier__ = "INTEGRATION_LAYER"