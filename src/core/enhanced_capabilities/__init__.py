"""
Enhanced Capabilities - Advanced Cognitive Processing Systems
===========================================================

This module contains the enhanced cognitive capabilities that build upon
the foundational systems to provide advanced cognitive processing.

Phase 3 Enhanced Capabilities:
- Understanding Core: Genuine understanding with self-model and causal reasoning
- Consciousness Core: Consciousness detection using thermodynamic signatures
- Meta Insight Core: Higher-order insight generation and meta-cognition
- Field Dynamics Core: Cognitive field processing with geoids
- Learning Core: Unsupervised cognitive learning engine
- Linguistic Intelligence Core: Advanced language processing

These capabilities integrate seamlessly with the foundational systems
(KCCL, SPDE, Barenholtz, Cognitive Cycle, Interoperability Bus) to
provide a complete cognitive processing platform.
"""

# Import all enhanced capabilities
try:
    from .understanding_core import (
        UnderstandingCore,
        GenuineUnderstanding,
        SelfModelSystem,
        CausalReasoningEngine,
        MultimodalGroundingSystem
    )
except ImportError as e:
    print(f"Warning: Could not import understanding_core: {e}")
    UnderstandingCore = None

try:
    from .consciousness_core import (
        ConsciousnessCore,
        ThermodynamicConsciousnessDetector,
        QuantumCoherenceAnalyzer,
        IntegratedInformationProcessor,
        ConsciousnessSignature
    )
except ImportError as e:
    print(f"Warning: Could not import consciousness_core: {e}")
    ConsciousnessCore = None

try:
    from .meta_insight_core import (
        MetaInsightCore,
        HigherOrderProcessor,
        MetaCognitionEngine,
        PatternRecognitionSystem,
        InsightGenerationEngine
    )
except ImportError as e:
    print(f"Warning: Could not import meta_insight_core: {e}")
    MetaInsightCore = None

try:
    from .field_dynamics_core import (
        FieldDynamicsCore,
        CognitiveFieldProcessor,
        GeoidFieldManager,
        SemanticFieldEvolution,
        EnergyFieldDynamics
    )
except ImportError as e:
    print(f"Warning: Could not import field_dynamics_core: {e}")
    FieldDynamicsCore = None

try:
    from .learning_core import (
        LearningCore,
        UnsupervisedCognitiveLearning,
        PhysicsBasedLearning,
        ResonanceClusteringEngine,
        ThermodynamicOrganization
    )
except ImportError as e:
    print(f"Warning: Could not import learning_core: {e}")
    LearningCore = None

try:
    from .linguistic_intelligence_core import (
        LinguisticIntelligenceCore,
        AdvancedLanguageProcessor,
        UniversalTranslationSystem,
        SemanticEntropyAnalyzer,
        GrammarSyntaxEngine
    )
except ImportError as e:
    print(f"Warning: Could not import linguistic_intelligence_core: {e}")
    LinguisticIntelligenceCore = None

__all__ = [
    # Understanding Core
    'UnderstandingCore',
    'GenuineUnderstanding', 
    'SelfModelSystem',
    'CausalReasoningEngine',
    'MultimodalGroundingSystem',
    
    # Consciousness Core
    'ConsciousnessCore',
    'ThermodynamicConsciousnessDetector',
    'QuantumCoherenceAnalyzer', 
    'IntegratedInformationProcessor',
    'ConsciousnessSignature',
    
    # Meta Insight Core
    'MetaInsightCore',
    'HigherOrderProcessor',
    'MetaCognitionEngine',
    'PatternRecognitionSystem', 
    'InsightGenerationEngine',
    
    # Field Dynamics Core
    'FieldDynamicsCore',
    'CognitiveFieldProcessor',
    'GeoidFieldManager',
    'SemanticFieldEvolution',
    'EnergyFieldDynamics',
    
    # Learning Core
    'LearningCore', 
    'UnsupervisedCognitiveLearning',
    'PhysicsBasedLearning',
    'ResonanceClusteringEngine',
    'ThermodynamicOrganization',
    
    # Linguistic Intelligence Core
    'LinguisticIntelligenceCore',
    'AdvancedLanguageProcessor',
    'UniversalTranslationSystem', 
    'SemanticEntropyAnalyzer',
    'GrammarSyntaxEngine'
]