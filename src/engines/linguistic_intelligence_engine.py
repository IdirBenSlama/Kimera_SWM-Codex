"""
Linguistic Intelligence Engine
==============================

Comprehensive linguistic processing engine that integrates all non-financial
language processing capabilities in Kimera, including:

- BGE-M3 embedding processing
- Universal translation capabilities  
- Grammar and syntax analysis
- Linguistic entropy and complexity analysis
- Meta-commentary elimination
- Context-aware semantic processing
- Multi-modal language understanding

This engine excludes all financial, trading, and market-related processing.
"""

import asyncio
import logging
import time
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

# Core linguistic components
from ..core.embedding_utils import (
    encode_text, 
    extract_semantic_features, 
    get_embedding_model,
    initialize_embedding_model
)
from ..core.context_field_selector import ContextFieldSelector, ContextFieldConfig, ProcessingLevel
from ..core.relevance_assessment import RelevanceAssessmentEngine
from ..linguistic.grammar import ECHOFORM_VOCABULARY, CORE_EVENT_TYPES, INSIGHT_SUB_TYPES
from ..linguistic.echoform import parse_echoform
from ..linguistic.entropy_formulas import calculate_linguistic_complexity, calculate_multiscale_entropy

# Translation and communication systems
from ..engines.gyroscopic_universal_translator import (
    GyroscopicUniversalTranslator,
    TranslationModality,
    TranslationRequest,
    TranslationResult
)
from ..engines.meta_commentary_eliminator import MetaCommentaryEliminator
from ..engines.human_interface import HumanInterface

# Configuration
from ..config.settings import get_settings
from ..utils.config import get_api_settings

logger = logging.getLogger(__name__)


class LinguisticCapability(Enum):
    """Core linguistic processing capabilities"""
    SEMANTIC_EMBEDDING = "semantic_embedding"
    UNIVERSAL_TRANSLATION = "universal_translation"
    GRAMMAR_ANALYSIS = "grammar_analysis"
    ENTROPY_ANALYSIS = "entropy_analysis"
    CONTEXT_PROCESSING = "context_processing"
    META_COMMENTARY_ELIMINATION = "meta_commentary_elimination"
    RELEVANCE_ASSESSMENT = "relevance_assessment"
    ECHOFORM_PARSING = "echoform_parsing"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    HUMAN_INTERFACE_OPTIMIZATION = "human_interface_optimization"


@dataclass
class LinguisticAnalysis:
    """Comprehensive linguistic analysis result"""
    
    # Input metadata
    input_text: str
    input_length: int
    language_detected: Optional[str] = None
    processing_time_ms: float = 0.0
    
    # Semantic analysis
    semantic_embedding: Optional[List[float]] = None
    semantic_features: Optional[Dict[str, float]] = None
    semantic_similarity_score: float = 0.0
    
    # Grammar and syntax
    grammar_analysis: Optional[Dict[str, Any]] = None
    echoform_parsed: Optional[List] = None
    vocabulary_matches: List[str] = field(default_factory=list)
    
    # Linguistic complexity
    complexity_metrics: Optional[Dict[str, float]] = None
    entropy_analysis: Optional[Dict[str, Any]] = None
    
    # Context and relevance
    context_assessment: Optional[Dict[str, Any]] = None
    relevance_score: float = 0.0
    context_type: Optional[str] = None
    
    # Communication optimization
    meta_commentary_detected: bool = False
    cleaned_response: Optional[str] = None
    human_optimized: bool = False
    
    # Translation capabilities
    translation_modalities: List[str] = field(default_factory=list)
    translation_confidence: float = 0.0
    
    # Processing metadata
    capabilities_used: List[str] = field(default_factory=list)
    processing_stages: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class LinguisticEngineConfig:
    """Configuration for the linguistic intelligence engine"""
    
    # Core processing settings
    enable_semantic_processing: bool = True
    enable_universal_translation: bool = True
    enable_grammar_analysis: bool = True
    enable_entropy_analysis: bool = True
    enable_context_processing: bool = True
    enable_meta_commentary_elimination: bool = True
    enable_relevance_assessment: bool = True
    
    # Performance settings
    max_input_length: int = 8192
    batch_processing_enabled: bool = True
    cache_enabled: bool = True
    lightweight_mode: bool = False
    
    # Context processing level
    processing_level: ProcessingLevel = ProcessingLevel.STANDARD
    
    # Translation settings
    default_translation_modality: TranslationModality = TranslationModality.NATURAL_LANGUAGE
    enable_polyglot_processing: bool = True
    
    # Quality settings
    min_confidence_threshold: float = 0.6
    enable_human_optimization: bool = True


class LinguisticIntelligenceEngine:
    """
    Comprehensive linguistic intelligence engine for Kimera
    
    Integrates all non-financial linguistic processing capabilities:
    - Advanced embedding processing with BGE-M3
    - Universal translation across multiple modalities
    - Grammar analysis and EchoForm parsing
    - Linguistic complexity and entropy analysis
    - Context-aware semantic processing
    - Meta-commentary elimination
    - Human-optimized communication interface
    """
    
    def __init__(self, config: Optional[LinguisticEngineConfig] = None):
        self.config = config or LinguisticEngineConfig()
        self.settings = get_settings()
        self.api_settings = get_api_settings()
        
        # Core components (initialized lazily)
        self._embedding_model = None
        self._universal_translator = None
        self._context_selector = None
        self._relevance_assessor = None
        self._meta_eliminator = None
        self._human_interface = None
        
        # Performance tracking
        self.performance_stats = {
            'total_analyses': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Component status
        self.component_status = {
            capability.value: False for capability in LinguisticCapability
        }
        
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        
        logger.info("🧠 Linguistic Intelligence Engine created")
        logger.info(f"   Configuration: {self.config.processing_level.value} processing level")
        logger.info(f"   Capabilities enabled: {sum(1 for k, v in self.config.__dict__.items() if k.startswith('enable_') and v)}")
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all linguistic components"""
        if self._initialized:
            return self.component_status
        
        async with self._initialization_lock:
            if self._initialized:
                return self.component_status
            
            logger.info("🔄 Initializing Linguistic Intelligence Engine...")
            start_time = time.time()
            
            # Initialize core components
            await self._initialize_embedding_system()
            await self._initialize_universal_translator()
            await self._initialize_context_processing()
            await self._initialize_relevance_assessment()
            await self._initialize_meta_commentary_elimination()
            await self._initialize_human_interface()
            
            self._initialized = True
            initialization_time = time.time() - start_time
            
            logger.info(f"✅ Linguistic Intelligence Engine initialized in {initialization_time:.2f}s")
            logger.info(f"   Active capabilities: {[k for k, v in self.component_status.items() if v]}")
            
            return self.component_status
    
    async def _initialize_embedding_system(self):
        """Initialize semantic embedding system"""
        try:
            if self.config.enable_semantic_processing:
                # Initialize BGE-M3 embedding model
                initialize_embedding_model()
                self._embedding_model = get_embedding_model()
                self.component_status[LinguisticCapability.SEMANTIC_EMBEDDING.value] = True
                logger.info("✅ Semantic embedding system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding system: {e}")
            self.component_status[LinguisticCapability.SEMANTIC_EMBEDDING.value] = False
    
    async def _initialize_universal_translator(self):
        """Initialize universal translation system"""
        try:
            if self.config.enable_universal_translation:
                self._universal_translator = GyroscopicUniversalTranslator()
                await self._universal_translator.initialize()
                self.component_status[LinguisticCapability.UNIVERSAL_TRANSLATION.value] = True
                logger.info("✅ Universal translator initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize universal translator: {e}")
            self.component_status[LinguisticCapability.UNIVERSAL_TRANSLATION.value] = False
    
    async def _initialize_context_processing(self):
        """Initialize context processing system"""
        try:
            if self.config.enable_context_processing:
                context_config = ContextFieldConfig(
                    processing_level=self.config.processing_level,
                    include_confidence_scores=True,
                    include_processing_metadata=True
                )
                self._context_selector = ContextFieldSelector(context_config)
                self.component_status[LinguisticCapability.CONTEXT_PROCESSING.value] = True
                logger.info("✅ Context processing system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize context processing: {e}")
            self.component_status[LinguisticCapability.CONTEXT_PROCESSING.value] = False
    
    async def _initialize_relevance_assessment(self):
        """Initialize relevance assessment system"""
        try:
            if self.config.enable_relevance_assessment:
                self._relevance_assessor = RelevanceAssessmentEngine()
                self.component_status[LinguisticCapability.RELEVANCE_ASSESSMENT.value] = True
                logger.info("✅ Relevance assessment system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize relevance assessment: {e}")
            self.component_status[LinguisticCapability.RELEVANCE_ASSESSMENT.value] = False
    
    async def _initialize_meta_commentary_elimination(self):
        """Initialize meta-commentary elimination system"""
        try:
            if self.config.enable_meta_commentary_elimination:
                self._meta_eliminator = MetaCommentaryEliminator()
                self.component_status[LinguisticCapability.META_COMMENTARY_ELIMINATION.value] = True
                logger.info("✅ Meta-commentary elimination initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize meta-commentary elimination: {e}")
            self.component_status[LinguisticCapability.META_COMMENTARY_ELIMINATION.value] = False
    
    async def _initialize_human_interface(self):
        """Initialize human interface optimization"""
        try:
            if self.config.enable_human_optimization:
                self._human_interface = HumanInterface()
                self.component_status[LinguisticCapability.HUMAN_INTERFACE_OPTIMIZATION.value] = True
                logger.info("✅ Human interface optimization initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize human interface: {e}")
            self.component_status[LinguisticCapability.HUMAN_INTERFACE_OPTIMIZATION.value] = False
    
    async def analyze_text(self, 
                          text: str, 
                          context: Optional[Dict[str, Any]] = None,
                          capabilities: Optional[List[LinguisticCapability]] = None) -> LinguisticAnalysis:
        """
        Perform comprehensive linguistic analysis on input text
        
        Args:
            text: Input text to analyze
            context: Optional context information
            capabilities: Specific capabilities to use (if None, uses all enabled)
            
        Returns:
            LinguisticAnalysis object with comprehensive results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        context = context or {}
        
        # Create analysis object
        analysis = LinguisticAnalysis(
            input_text=text,
            input_length=len(text),
            processing_stages=[]
        )
        
        try:
            # Validate input
            if len(text) > self.config.max_input_length:
                text = text[:self.config.max_input_length]
                logger.warning(f"Input truncated to {self.config.max_input_length} characters")
            
            analysis.processing_stages.append("input_validation")
            
            # Determine which capabilities to use
            if capabilities is None:
                capabilities = [cap for cap in LinguisticCapability 
                              if self.component_status.get(cap.value, False)]
            
            analysis.capabilities_used = [cap.value for cap in capabilities]
            
            # Semantic embedding analysis
            if LinguisticCapability.SEMANTIC_EMBEDDING in capabilities:
                await self._perform_semantic_analysis(analysis, text, context)
            
            # Grammar and syntax analysis
            if LinguisticCapability.GRAMMAR_ANALYSIS in capabilities:
                await self._perform_grammar_analysis(analysis, text)
            
            # EchoForm parsing
            if LinguisticCapability.ECHOFORM_PARSING in capabilities:
                await self._perform_echoform_analysis(analysis, text)
            
            # Entropy and complexity analysis
            if LinguisticCapability.ENTROPY_ANALYSIS in capabilities:
                await self._perform_entropy_analysis(analysis, text)
            
            # Context processing
            if LinguisticCapability.CONTEXT_PROCESSING in capabilities:
                await self._perform_context_analysis(analysis, text, context)
            
            # Relevance assessment
            if LinguisticCapability.RELEVANCE_ASSESSMENT in capabilities:
                await self._perform_relevance_analysis(analysis, text, context)
            
            # Meta-commentary elimination
            if LinguisticCapability.META_COMMENTARY_ELIMINATION in capabilities:
                await self._perform_meta_commentary_analysis(analysis, text)
            
            # Human interface optimization
            if LinguisticCapability.HUMAN_INTERFACE_OPTIMIZATION in capabilities:
                await self._perform_human_optimization(analysis, text)
            
            # Universal translation readiness
            if LinguisticCapability.UNIVERSAL_TRANSLATION in capabilities:
                await self._assess_translation_capabilities(analysis, text)
            
            # Calculate final metrics
            processing_time = time.time() - start_time
            analysis.processing_time_ms = processing_time * 1000
            analysis.performance_metrics = {
                'total_time_ms': analysis.processing_time_ms,
                'stages_completed': len(analysis.processing_stages),
                'capabilities_used': len(analysis.capabilities_used)
            }
            
            # Update performance stats
            self.performance_stats['total_analyses'] += 1
            self.performance_stats['total_processing_time'] += processing_time
            self.performance_stats['average_processing_time'] = (
                self.performance_stats['total_processing_time'] / 
                self.performance_stats['total_analyses']
            )
            self.performance_stats['successful_analyses'] += 1
            
            logger.debug(f"✅ Linguistic analysis completed in {processing_time:.3f}s")
            return analysis
            
        except Exception as e:
            self.performance_stats['failed_analyses'] += 1
            logger.error(f"❌ Linguistic analysis failed: {e}", exc_info=True)
            analysis.processing_time_ms = (time.time() - start_time) * 1000
            return analysis
    
    async def _perform_semantic_analysis(self, analysis: LinguisticAnalysis, text: str, context: Dict[str, Any]):
        """Perform semantic embedding analysis"""
        try:
            analysis.processing_stages.append("semantic_analysis")
            
            # Generate semantic embedding
            embedding = encode_text(text)
            if isinstance(embedding, torch.Tensor):
                analysis.semantic_embedding = embedding.cpu().numpy().tolist()
            elif isinstance(embedding, list):
                analysis.semantic_embedding = embedding
            else:
                analysis.semantic_embedding = embedding.tolist()
            
            # Extract semantic features
            analysis.semantic_features = extract_semantic_features(text)
            
            logger.debug("✅ Semantic analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Semantic analysis failed: {e}")
    
    async def _perform_grammar_analysis(self, analysis: LinguisticAnalysis, text: str):
        """Perform grammar and vocabulary analysis"""
        try:
            analysis.processing_stages.append("grammar_analysis")
            
            # Check for EchoForm vocabulary matches
            text_lower = text.lower()
            analysis.vocabulary_matches = [
                word for word in ECHOFORM_VOCABULARY 
                if word.lower() in text_lower
            ]
            
            # Basic grammar analysis
            analysis.grammar_analysis = {
                'vocabulary_matches': len(analysis.vocabulary_matches),
                'contains_event_types': any(event in text_lower for event in CORE_EVENT_TYPES),
                'contains_insights': any(insight in text_lower for insight in INSIGHT_SUB_TYPES),
                'sentence_count': text.count('.') + text.count('!') + text.count('?'),
                'word_count': len(text.split()),
                'character_count': len(text)
            }
            
            logger.debug("✅ Grammar analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Grammar analysis failed: {e}")
    
    async def _perform_echoform_analysis(self, analysis: LinguisticAnalysis, text: str):
        """Perform EchoForm parsing analysis"""
        try:
            analysis.processing_stages.append("echoform_analysis")
            
            # Try to parse as EchoForm
            try:
                analysis.echoform_parsed = parse_echoform(text)
                logger.debug("✅ EchoForm parsing successful")
            except (ValueError, SyntaxError):
                # Not valid EchoForm, which is normal for natural language
                analysis.echoform_parsed = None
                logger.debug("📝 Text is not valid EchoForm (normal for natural language)")
            
        except Exception as e:
            logger.error(f"❌ EchoForm analysis failed: {e}")
    
    async def _perform_entropy_analysis(self, analysis: LinguisticAnalysis, text: str):
        """Perform linguistic entropy and complexity analysis"""
        try:
            analysis.processing_stages.append("entropy_analysis")
            
            # Calculate linguistic complexity
            analysis.complexity_metrics = calculate_linguistic_complexity(text)
            
            # Calculate multiscale entropy if text is long enough
            if len(text) > 50:
                # Convert text to numeric data for entropy analysis
                char_values = [ord(c) for c in text if c.isprintable()]
                if len(char_values) > 10:
                    entropy_scales = calculate_multiscale_entropy(char_values, max_scale=5)
                    analysis.entropy_analysis = {
                        'multiscale_entropy': entropy_scales,
                        'average_entropy': sum(entropy_scales) / len(entropy_scales),
                        'entropy_complexity': max(entropy_scales) - min(entropy_scales)
                    }
            
            logger.debug("✅ Entropy analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Entropy analysis failed: {e}")
    
    async def _perform_context_analysis(self, analysis: LinguisticAnalysis, text: str, context: Dict[str, Any]):
        """Perform context processing analysis"""
        try:
            analysis.processing_stages.append("context_analysis")
            
            if self._context_selector:
                # Create semantic state for context processing
                semantic_state = {
                    'text': text,
                    'length': len(text),
                    'context': context,
                    'semantic_features': analysis.semantic_features or {}
                }
                
                # Filter semantic state through context selector
                filtered_state = self._context_selector.filter_semantic_state(semantic_state)
                analysis.context_assessment = filtered_state
            
            logger.debug("✅ Context analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Context analysis failed: {e}")
    
    async def _perform_relevance_analysis(self, analysis: LinguisticAnalysis, text: str, context: Dict[str, Any]):
        """Perform relevance assessment"""
        try:
            analysis.processing_stages.append("relevance_analysis")
            
            if self._relevance_assessor:
                # Assess context relevance
                assessment = self._relevance_assessor.assess_context_relevance(text, context)
                analysis.relevance_score = assessment.get('relevance_score', 0.0)
                analysis.context_type = assessment.get('context_type')
            
            logger.debug("✅ Relevance analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Relevance analysis failed: {e}")
    
    async def _perform_meta_commentary_analysis(self, analysis: LinguisticAnalysis, text: str):
        """Perform meta-commentary detection and elimination"""
        try:
            analysis.processing_stages.append("meta_commentary_analysis")
            
            if self._meta_eliminator:
                detection_result = await self._meta_eliminator.detect_meta_commentary(text)
                analysis.meta_commentary_detected = detection_result.has_dissociation
                
                if analysis.meta_commentary_detected:
                    analysis.cleaned_response = await self._meta_eliminator.eliminate_meta_commentary(text)
            
            logger.debug("✅ Meta-commentary analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Meta-commentary analysis failed: {e}")
    
    async def _perform_human_optimization(self, analysis: LinguisticAnalysis, text: str):
        """Perform human interface optimization"""
        try:
            analysis.processing_stages.append("human_optimization")
            
            if self._human_interface:
                # Optimize for human readability
                optimized = self._human_interface._clean_response(text)
                if optimized != text:
                    analysis.cleaned_response = optimized
                    analysis.human_optimized = True
            
            logger.debug("✅ Human optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Human optimization failed: {e}")
    
    async def _assess_translation_capabilities(self, analysis: LinguisticAnalysis, text: str):
        """Assess universal translation capabilities"""
        try:
            analysis.processing_stages.append("translation_assessment")
            
            if self._universal_translator:
                # Assess what translation modalities are available
                available_modalities = []
                
                # Check for natural language
                if any(c.isalpha() for c in text):
                    available_modalities.append(TranslationModality.NATURAL_LANGUAGE.value)
                
                # Check for mathematical content
                if any(c in text for c in '+-*/=()[]{}^∂∇∆∑∏∫'):
                    available_modalities.append(TranslationModality.MATHEMATICAL.value)
                
                # Check for EchoForm structure
                if '(' in text and ')' in text:
                    available_modalities.append(TranslationModality.ECHOFORM.value)
                
                analysis.translation_modalities = available_modalities
                analysis.translation_confidence = min(0.9, len(available_modalities) * 0.3)
            
            logger.debug("✅ Translation assessment completed")
            
        except Exception as e:
            logger.error(f"❌ Translation assessment failed: {e}")
    
    async def translate_text(self, 
                           text: str, 
                           source_modality: TranslationModality,
                           target_modality: TranslationModality,
                           context: Optional[Dict[str, Any]] = None) -> TranslationResult:
        """
        Translate text between different linguistic modalities
        
        Args:
            text: Input text to translate
            source_modality: Source modality
            target_modality: Target modality
            context: Optional context information
            
        Returns:
            TranslationResult object
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._universal_translator:
            raise RuntimeError("Universal translator not initialized")
        
        request = TranslationRequest(
            content=text,
            source_modality=source_modality,
            target_modality=target_modality,
            context=context or {}
        )
        
        return await self._universal_translator.translate(request)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_stats,
            'component_status': self.component_status,
            'configuration': {
                'processing_level': self.config.processing_level.value,
                'enabled_capabilities': [
                    k for k, v in self.config.__dict__.items() 
                    if k.startswith('enable_') and v
                ]
            }
        }
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all linguistic components"""
        return self.component_status.copy()
    
    async def shutdown(self):
        """Shutdown the linguistic intelligence engine"""
        logger.info("🔄 Shutting down Linguistic Intelligence Engine...")
        
        # Shutdown components
        if self._universal_translator:
            await self._universal_translator.shutdown()
        
        # Reset state
        self._initialized = False
        
        logger.info("✅ Linguistic Intelligence Engine shutdown complete")


# Factory functions for easy integration
def create_linguistic_engine(config: Optional[LinguisticEngineConfig] = None) -> LinguisticIntelligenceEngine:
    """Create a new linguistic intelligence engine"""
    return LinguisticIntelligenceEngine(config)

def create_lightweight_linguistic_engine() -> LinguisticIntelligenceEngine:
    """Create a lightweight linguistic engine for testing"""
    config = LinguisticEngineConfig(
        lightweight_mode=True,
        processing_level=ProcessingLevel.MINIMAL,
        enable_universal_translation=False,
        enable_entropy_analysis=False
    )
    return LinguisticIntelligenceEngine(config)

def create_comprehensive_linguistic_engine() -> LinguisticIntelligenceEngine:
    """Create a comprehensive linguistic engine with all capabilities"""
    config = LinguisticEngineConfig(
        processing_level=ProcessingLevel.ENHANCED,
        enable_semantic_processing=True,
        enable_universal_translation=True,
        enable_grammar_analysis=True,
        enable_entropy_analysis=True,
        enable_context_processing=True,
        enable_meta_commentary_elimination=True,
        enable_relevance_assessment=True
    )
    return LinguisticIntelligenceEngine(config)

# Global instance for easy access
_global_linguistic_engine: Optional[LinguisticIntelligenceEngine] = None
_linguistic_engine_lock = asyncio.Lock()

async def get_linguistic_engine() -> LinguisticIntelligenceEngine:
    """Get the global linguistic intelligence engine instance"""
    global _global_linguistic_engine
    
    if _global_linguistic_engine is None:
        async with _linguistic_engine_lock:
            if _global_linguistic_engine is None:
                _global_linguistic_engine = create_comprehensive_linguistic_engine()
                await _global_linguistic_engine.initialize()
    
    return _global_linguistic_engine 