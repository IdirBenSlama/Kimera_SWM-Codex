"""
Linguistic Intelligence Router
==============================

FastAPI router for accessing all linguistic processing capabilities in Kimera.
Provides endpoints for:

- Text analysis and semantic processing
- Universal translation between modalities
- Grammar and syntax analysis
- Linguistic complexity assessment
- Context-aware processing
- Meta-commentary elimination
- Human-optimized communication

Excludes all financial and trading-related functionality.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

# Import processing levels
from ...core.context_field_selector import ProcessingLevel
# Import core system
from ...engines.gyroscopic_universal_translator import (
    TranslationModality,
)
from ...engines.linguistic_intelligence_engine import (
    LinguisticCapability,
    LinguisticEngineConfig,
    LinguisticIntelligenceEngine,
    create_linguistic_engine,
    get_linguistic_engine,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/linguistic", tags=["Linguistic Intelligence"])


# --- Pydantic Models for API ---


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis"""

    text: str = Field(..., description="Text to analyze", max_length=8192)
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context information"
    )
    capabilities: Optional[List[str]] = Field(
        default=None,
        description="Specific capabilities to use (if None, uses all enabled)",
    )
    processing_level: Optional[str] = Field(
        default="standard", description="Processing level: minimal, standard, enhanced"
    )


class TextAnalysisResponse(BaseModel):
    """Response model for text analysis"""

    success: bool
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)


class TranslationRequestModel(BaseModel):
    """Request model for universal translation"""

    text: str = Field(..., description="Text to translate", max_length=8192)
    source_modality: str = Field(..., description="Source modality")
    target_modality: str = Field(..., description="Target modality")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context"
    )


class TranslationResponse(BaseModel):
    """Response model for translation"""

    success: bool
    translation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)


class LinguisticCapabilitiesResponse(BaseModel):
    """Response model for capabilities listing"""

    capabilities: List[str]
    component_status: Dict[str, bool]
    engine_status: str
    total_capabilities: int


class PerformanceStatsResponse(BaseModel):
    """Response model for performance statistics"""

    performance_stats: Dict[str, Any]
    component_status: Dict[str, bool]
    configuration: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class EngineConfigRequest(BaseModel):
    """Request model for engine configuration"""

    enable_semantic_processing: bool = True
    enable_universal_translation: bool = True
    enable_grammar_analysis: bool = True
    enable_entropy_analysis: bool = True
    enable_context_processing: bool = True
    enable_meta_commentary_elimination: bool = True
    enable_relevance_assessment: bool = True
    processing_level: str = "standard"
    lightweight_mode: bool = False
    max_input_length: int = 8192


# --- Dependency Functions ---


async def get_linguistic_engine_dependency() -> LinguisticIntelligenceEngine:
    """Dependency to get the linguistic intelligence engine"""
    try:
        return await get_linguistic_engine()
    except Exception as e:
        logger.error(f"Failed to get linguistic engine: {e}")
        raise HTTPException(
            status_code=500, detail=f"Linguistic engine unavailable: {str(e)}"
        )


def validate_processing_level(level: str) -> ProcessingLevel:
    """Validate and convert processing level string"""
    level_map = {
        "minimal": ProcessingLevel.MINIMAL,
        "standard": ProcessingLevel.STANDARD,
        "enhanced": ProcessingLevel.ENHANCED,
    }

    if level.lower() not in level_map:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid processing level. Must be one of: {list(level_map.keys())}"
            ),
        )

    return level_map[level.lower()]


def validate_modality(modality: str) -> TranslationModality:
    """Validate and convert modality string"""
    try:
        return TranslationModality(modality.lower())
    except ValueError:
        valid_modalities = [m.value for m in TranslationModality]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid modality '{modality}'. Must be one of: {valid_modalities}",
        )


def validate_capabilities(
    capabilities: Optional[List[str]],
) -> Optional[List[LinguisticCapability]]:
    """Validate and convert capability strings"""
    if capabilities is None:
        return None

    validated_caps = []
    valid_cap_values = [cap.value for cap in LinguisticCapability]

    for cap in capabilities:
        if cap not in valid_cap_values:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid capability '{cap}'. Must be one of: {valid_cap_values}"
                ),
            )
        validated_caps.append(LinguisticCapability(cap))

    return validated_caps


# --- API Endpoints ---


@router.get("/", summary="Linguistic Intelligence Overview")
async def linguistic_overview():
    """Get overview of linguistic intelligence capabilities"""
    return {
        "service": "Kimera Linguistic Intelligence Engine",
        "description": "Comprehensive linguistic processing and analysis",
        "capabilities": [cap.value for cap in LinguisticCapability],
        "endpoints": {
            "analyze": "POST /analyze - Comprehensive text analysis",
            "translate": "POST /translate - Universal translation",
            "capabilities": "GET /capabilities - List all capabilities",
            "performance": "GET /performance - Performance statistics",
            "health": "GET /health - Engine health status",
        },
        "version": "1.0.0",
        "excludes": "All financial and market-related processing",
    }


@router.post("/analyze", response_model=TextAnalysisResponse, summary="Analyze Text")
async def analyze_text(
    request: TextAnalysisRequest,
    engine: LinguisticIntelligenceEngine = Depends(get_linguistic_engine_dependency),
):
    """
    Perform comprehensive linguistic analysis on text

    Capabilities include:
    - Semantic embedding processing
    - Grammar and syntax analysis
    - EchoForm parsing
    - Linguistic entropy and complexity
    - Context processing
    - Relevance assessment
    - Meta-commentary elimination
    - Human interface optimization
    """
    start_time = datetime.now()

    try:
        # Validate capabilities
        capabilities = validate_capabilities(request.capabilities)

        # Perform analysis
        analysis = await engine.analyze_text(
            text=request.text, context=request.context, capabilities=capabilities
        )

        # Convert analysis to dict for JSON serialization
        analysis_dict = {
            "input_text": analysis.input_text,
            "input_length": analysis.input_length,
            "language_detected": analysis.language_detected,
            "processing_time_ms": analysis.processing_time_ms,
            "semantic_embedding": analysis.semantic_embedding,
            "semantic_features": analysis.semantic_features,
            "semantic_similarity_score": analysis.semantic_similarity_score,
            "grammar_analysis": analysis.grammar_analysis,
            "echoform_parsed": analysis.echoform_parsed,
            "vocabulary_matches": analysis.vocabulary_matches,
            "complexity_metrics": analysis.complexity_metrics,
            "entropy_analysis": analysis.entropy_analysis,
            "context_assessment": analysis.context_assessment,
            "relevance_score": analysis.relevance_score,
            "context_type": analysis.context_type,
            "meta_commentary_detected": analysis.meta_commentary_detected,
            "cleaned_response": analysis.cleaned_response,
            "human_optimized": analysis.human_optimized,
            "translation_modalities": analysis.translation_modalities,
            "translation_confidence": analysis.translation_confidence,
            "capabilities_used": analysis.capabilities_used,
            "processing_stages": analysis.processing_stages,
            "performance_metrics": analysis.performance_metrics,
        }

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return TextAnalysisResponse(
            success=True, analysis=analysis_dict, processing_time_ms=processing_time
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Text analysis failed: {e}", exc_info=True)

        return TextAnalysisResponse(
            success=False, error=str(e), processing_time_ms=processing_time
        )


@router.post(
    "/translate", response_model=TranslationResponse, summary="Universal Translation"
)
async def translate_text(
    request: TranslationRequestModel,
    engine: LinguisticIntelligenceEngine = Depends(get_linguistic_engine_dependency),
):
    """
    Translate text between different linguistic modalities

    Supported modalities:
    - natural_language: Human text/speech
    - echoform: KIMERA's native s-expressions
    - mathematical: Equations, formulas, logic
    - visual_patterns: Image descriptions
    - cognitive_states: Internal consciousness
    - semantic_fields: Cognitive field dynamics
    """
    start_time = datetime.now()

    try:
        # Validate modalities
        source_modality = validate_modality(request.source_modality)
        target_modality = validate_modality(request.target_modality)

        # Perform translation
        result = await engine.translate_text(
            text=request.text,
            source_modality=source_modality,
            target_modality=target_modality,
            context=request.context,
        )

        # Convert result to dict
        translation_dict = {
            "source_content": (
                result.source_content
                if hasattr(result, "source_content")
                else request.text
            ),
            "translated_content": (
                result.translated_content
                if hasattr(result, "translated_content")
                else None
            ),
            "source_modality": request.source_modality,
            "target_modality": request.target_modality,
            "confidence_score": (
                result.confidence_score if hasattr(result, "confidence_score") else 0.0
            ),
            "processing_metadata": (
                result.processing_metadata
                if hasattr(result, "processing_metadata")
                else {}
            ),
        }

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return TranslationResponse(
            success=True,
            translation=translation_dict,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(f"Translation failed: {e}", exc_info=True)

        return TranslationResponse(
            success=False, error=str(e), processing_time_ms=processing_time
        )


@router.get(
    "/capabilities",
    response_model=LinguisticCapabilitiesResponse,
    summary="List Capabilities",
)
async def get_capabilities(
    engine: LinguisticIntelligenceEngine = Depends(get_linguistic_engine_dependency),
):
    """Get list of all available linguistic processing capabilities"""
    try:
        component_status = engine.get_component_status()

        return LinguisticCapabilitiesResponse(
            capabilities=[cap.value for cap in LinguisticCapability],
            component_status=component_status,
            engine_status="initialized" if engine._initialized else "not_initialized",
            total_capabilities=len(LinguisticCapability),
        )

    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get capabilities: {str(e)}"
        )


@router.get(
    "/performance",
    response_model=PerformanceStatsResponse,
    summary="Performance Statistics",
)
async def get_performance_stats(
    engine: LinguisticIntelligenceEngine = Depends(get_linguistic_engine_dependency),
):
    """Get detailed performance statistics for the linguistic engine"""
    try:
        stats = engine.get_performance_stats()

        return PerformanceStatsResponse(
            performance_stats=stats.get("performance_stats", {}),
            component_status=stats.get("component_status", {}),
            configuration=stats.get("configuration", {}),
        )

    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance stats: {str(e)}"
        )


@router.get("/health", summary="Engine Health Check")
async def health_check():
    """Check health status of the linguistic intelligence engine"""
    try:
        # Try to get the engine
        engine = await get_linguistic_engine()

        # Check component status
        component_status = engine.get_component_status()
        active_components = sum(1 for status in component_status.values() if status)
        total_components = len(component_status)

        # Calculate health score
        health_score = (
            active_components / total_components if total_components > 0 else 0.0
        )

        status = (
            "healthy"
            if health_score >= 0.8
            else "degraded" if health_score >= 0.5 else "unhealthy"
        )

        return {
            "status": status,
            "health_score": health_score,
            "engine_initialized": engine._initialized,
            "active_components": active_components,
            "total_components": total_components,
            "component_status": component_status,
            "timestamp": datetime.now().isoformat(),
            "capabilities_available": [cap.value for cap in LinguisticCapability],
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "health_score": 0.0,
            "engine_initialized": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/modalities", summary="Available Translation Modalities")
async def get_translation_modalities():
    """Get list of all available translation modalities"""
    return {
        "modalities": [
            {
                "value": modality.value,
                "description": {
                    "quantum_actions": "Contradictions, entropy, geometry",
                    "natural_language": "Human text/speech",
                    "echoform": "KIMERA's native s-expressions",
                    "mathematical": "Equations, formulas, logic",
                    "visual_patterns": "Images, diagrams, visualizations",
                    "sensory_data": "Audio, tactile, multi-sensory",
                    "cognitive_states": "Internal consciousness states",
                    "semantic_fields": "Cognitive field dynamics",
                    "dolphin_communication": "Cetacean acoustic patterns",
                }.get(modality.value, "Advanced linguistic modality"),
            }
            for modality in TranslationModality
        ],
        "total_modalities": len(TranslationModality),
        "note": "All modalities exclude financial and market-related processing",
    }


@router.post("/configure", summary="Configure Engine")
async def configure_engine(
    config: EngineConfigRequest,
    engine: LinguisticIntelligenceEngine = Depends(get_linguistic_engine_dependency),
):
    """
    Configure the linguistic intelligence engine
    Note: This creates a new engine instance with the specified configuration
    """
    try:
        # Validate processing level
        processing_level = validate_processing_level(config.processing_level)

        # Create new configuration
        new_config = LinguisticEngineConfig(
            enable_semantic_processing=config.enable_semantic_processing,
            enable_universal_translation=config.enable_universal_translation,
            enable_grammar_analysis=config.enable_grammar_analysis,
            enable_entropy_analysis=config.enable_entropy_analysis,
            enable_context_processing=config.enable_context_processing,
            enable_meta_commentary_elimination=config.enable_meta_commentary_elimination,
            enable_relevance_assessment=config.enable_relevance_assessment,
            processing_level=processing_level,
            lightweight_mode=config.lightweight_mode,
            max_input_length=config.max_input_length,
        )

        # Create new engine with configuration
        new_engine = create_linguistic_engine(new_config)
        await new_engine.initialize()

        return {
            "success": True,
            "message": "Engine configured successfully",
            "configuration": {
                "processing_level": processing_level.value,
                "enabled_capabilities": [
                    k
                    for k, v in config.__dict__.items()
                    if k.startswith("enable_") and v
                ],
                "lightweight_mode": config.lightweight_mode,
                "max_input_length": config.max_input_length,
            },
            "component_status": new_engine.get_component_status(),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Engine configuration failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Configuration failed: {str(e)}"
        )


@router.get("/processing-levels", summary="Available Processing Levels")
async def get_processing_levels():
    """Get information about available processing levels"""
    return {
        "processing_levels": [
            {
                "value": "minimal",
                "description": "Basic text processing with minimal resource usage",
                "features": ["Basic tokenization", "Simple grammar analysis"],
            },
            {
                "value": "standard",
                "description": "Full linguistic analysis with balanced performance",
                "features": [
                    "Semantic embedding",
                    "Grammar analysis",
                    "Context processing",
                    "Meta-commentary elimination",
                ],
            },
            {
                "value": "enhanced",
                "description": "Comprehensive analysis with all advanced features",
                "features": [
                    "All standard features",
                    "Entropy analysis",
                    "Universal translation",
                    "Advanced context assessment",
                ],
            },
        ],
        "default": "standard",
        "note": (
            "Processing level affects both capability availability and resource usage"
        ),
    }


# Initialize router metadata
router.summary = "Linguistic Intelligence Engine API"
router.description = """
Comprehensive linguistic processing API for Kimera, providing:

- **Text Analysis**: Semantic, grammatical, and complexity analysis
- **Universal Translation**: Between different linguistic modalities  
- **Context Processing**: Context-aware semantic understanding
- **Communication Enhancement**: Meta-commentary elimination and human optimization
- **Performance Monitoring**: Real-time statistics and health checks

**Note**: This API excludes all financial, trading, and market-related functionality.
"""

logger.info("🧠 Linguistic Intelligence Router initialized")
