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
    from .understanding_core import (CausalReasoningEngine, GenuineUnderstanding
                                     MultimodalGroundingSystem, SelfModelSystem
                                     UnderstandingCore)
except ImportError as e:
    logger.info(f"Warning: Could not import understanding_core: {e}")
    UnderstandingCore = None

try:
    from .consciousness_core import (ConsciousnessCore, ConsciousnessSignature
                                     IntegratedInformationProcessor
                                     QuantumCoherenceAnalyzer
                                     ThermodynamicConsciousnessDetector)
except ImportError as e:
    logger.info(f"Warning: Could not import consciousness_core: {e}")
    ConsciousnessCore = None

try:
    from .meta_insight_core import (HigherOrderProcessor, InsightGenerationEngine
                                    MetaCognitionEngine, MetaInsightCore
                                    PatternRecognitionSystem)
except ImportError as e:
    logger.info(f"Warning: Could not import meta_insight_core: {e}")
    MetaInsightCore = None

try:
    from .field_dynamics_core import (CognitiveFieldProcessor, EnergyFieldDynamics
                                      FieldDynamicsCore, GeoidFieldManager
                                      SemanticFieldEvolution)
except ImportError as e:
    logger.info(f"Warning: Could not import field_dynamics_core: {e}")
    FieldDynamicsCore = None

try:
    from .learning_core import (LearningCore, PhysicsBasedLearning
                                ResonanceClusteringEngine, ThermodynamicOrganization
                                UnsupervisedCognitiveLearning)
except ImportError as e:
    logger.info(f"Warning: Could not import learning_core: {e}")
    LearningCore = None

try:
    from .linguistic_intelligence_core import (=, __name__, er, logging
                                               logging.getLogger, rt)
        LinguisticIntelligenceCore
        AdvancedLanguageProcessor
        UniversalTranslationSystem
        SemanticEntropyAnalyzer
        GrammarSyntaxEngine
    )
except ImportError as e:
    logger.info(f"Warning: Could not import linguistic_intelligence_core: {e}")
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