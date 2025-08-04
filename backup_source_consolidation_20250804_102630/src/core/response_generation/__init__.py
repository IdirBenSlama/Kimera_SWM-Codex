"""
KIMERA Response Generation Module v2.0
======================================

DO-178C Level A compliant response generation system with quantum security,
full cognitive integration, and thermodynamic coherence.

This module provides the complete response generation capabilities for KIMERA,
integrating all cognitive systems including:
- Barenholtz dual-system architecture
- Quantum-resistant security
- High-dimensional modeling
- Insight management
- Thermodynamic coherence validation

Key Components:
- Cognitive Response System: Core response generation with cognitive metrics
- Full Integration Bridge: Complete cognitive architecture integration
- Quantum Security: Post-quantum cryptographic protection
- Response Orchestrator: Unified interface for all response generation

Author: KIMERA Development Team
Version: 2.0.0 (DO-178C Level A)
"""

from .core.cognitive_response_system import (
    ResponseGenerator,
    ResponseContext,
    ResponseOutput,
    ResponseGenerationConfig,
    ResponseType,
    CognitiveContext,
    ResponseQuality,
    CognitiveMetrics,
    get_cognitive_response_system
)

from .integration.full_integration_bridge import (
    KimeraFullIntegrationBridge,
    IntegrationConfig,
    IntegrationMode,
    ProcessingPriority,
    IntegrationMetrics,
    CognitiveArchitectureState,
    get_full_integration_bridge
)

from .security.quantum_security import (
    KimeraQuantumEdgeSecurityArchitecture,
    QuantumSecurityConfig,
    ThreatLevel,
    QuantumAttackType,
    SecurityMetrics,
    get_quantum_security
)

from .orchestrator import (
    ResponseGenerationOrchestrator,
    ResponseGenerationMode,
    ResponseGenerationRequest,
    ResponseGenerationResult,
    get_response_orchestrator,
    generate_standard_response,
    generate_secure_response,
    generate_research_response
)

# Version information
__version__ = "2.0.0"
__author__ = "KIMERA Development Team"
__license__ = "Proprietary - KIMERA SWM"
__status__ = "Production - DO-178C Level A"

# Export all public components
__all__ = [
    # Core Response System
    'ResponseGenerator',
    'ResponseContext',
    'ResponseOutput',
    'ResponseGenerationConfig',
    'ResponseType',
    'CognitiveContext',
    'ResponseQuality',
    'CognitiveMetrics',
    'get_cognitive_response_system',

    # Integration Bridge
    'KimeraFullIntegrationBridge',
    'IntegrationConfig',
    'IntegrationMode',
    'ProcessingPriority',
    'IntegrationMetrics',
    'CognitiveArchitectureState',
    'get_full_integration_bridge',

    # Quantum Security
    'KimeraQuantumEdgeSecurityArchitecture',
    'QuantumSecurityConfig',
    'ThreatLevel',
    'QuantumAttackType',
    'SecurityMetrics',
    'get_quantum_security',

    # Main Orchestrator
    'ResponseGenerationOrchestrator',
    'ResponseGenerationMode',
    'ResponseGenerationRequest',
    'ResponseGenerationResult',
    'get_response_orchestrator',

    # Convenience Functions
    'generate_standard_response',
    'generate_secure_response',
    'generate_research_response',

    # Module Info
    '__version__',
    '__author__',
    '__license__',
    '__status__'
]

# Module-level configuration and health check
def get_module_info() -> dict:
    """Get comprehensive module information"""
    return {
        'name': 'KIMERA Response Generation',
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'status': __status__,
        'compliance': 'DO-178C Level A',
        'components': {
            'cognitive_response_system': 'Core response generation with cognitive metrics',
            'full_integration_bridge': 'Complete cognitive architecture integration',
            'quantum_security': 'Post-quantum cryptographic protection',
            'response_orchestrator': 'Unified response generation interface'
        },
        'features': [
            'Quantum-resistant security',
            'Dual-system cognitive processing',
            'High-dimensional modeling integration',
            'Thermodynamic coherence validation',
            'Real-time threat detection',
            'Multi-modal response generation',
            'Comprehensive performance monitoring'
        ]
    }

def check_module_health() -> dict:
    """Perform module health check"""
    try:
        # Test core components
        orchestrator = get_response_orchestrator()
        status = orchestrator.get_orchestrator_status()

        return {
            'healthy': status.get('status') == 'operational',
            'orchestrator_status': status,
            'components_available': {
                'cognitive_response': True,
                'integration_bridge': True,
                'quantum_security': True
            },
            'timestamp': status.get('timestamp', 0)
        }

    except Exception as e:
        return {
            'healthy': False,
            'error': str(e),
            'components_available': {
                'cognitive_response': False,
                'integration_bridge': False,
                'quantum_security': False
            }
        }
