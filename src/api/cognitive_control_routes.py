"""
Enhanced API Routes
==================

New API endpoints that integrate Context Field Selector and
Anthropomorphic Language Profiler for improved processing control
and security.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import torch
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..core.anthropomorphic_profiler import (
    AnthropomorphicProfiler,
    InteractionAnalysis,
    PersonalityProfile,
    create_default_profiler,
    create_strict_profiler,
)
from ..core.context_field_selector import (
    ContextFieldConfig,
    ContextFieldSelector,
    FieldCategory,
    ProcessingLevel,
    create_domain_selector,
    create_enhanced_selector,
    create_minimal_selector,
    create_standard_selector,
)
from ..core.gyroscopic_security import (
    EquilibriumState,
    GyroscopicSecurityCore,
    ManipulationVector,
    create_balanced_security_core,
    create_maximum_security_core,
)
from ..layer_2_governance.monitoring.psychiatric_stability_monitor import \
    CognitiveCoherenceMonitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cognitive-control", tags=["cognitive-control"])

# Initialize the global context_selector variable
context_selector = None


def initialize_enhanced_services(app):
    """Initializes the enhanced services and attaches them to the app state."""
    logger.info("Creating default Anthropomorphic Profiler...")
    app.state.anthropomorphic_profiler = create_strict_profiler()

    logger.info("Creating default Gyroscopic Security Core...")
    app.state.gyroscopic_security = create_balanced_security_core()

    logger.info("Creating Cognitive Coherence Monitor...")
    app.state.coherence_monitor = CognitiveCoherenceMonitor()


# Pydantic models for API requests
class ContextFieldConfigRequest(BaseModel):
    processing_level: str = "standard"  # minimal, standard, enhanced, custom
    included_categories: Optional[List[str]] = None
    excluded_categories: Optional[List[str]] = None
    included_fields: Optional[List[str]] = None
    excluded_fields: Optional[List[str]] = None
    include_confidence_scores: bool = True
    include_processing_metadata: bool = True
    include_timestamps: bool = True
    skip_expensive_calculations: bool = False
    limit_embedding_dimensions: Optional[int] = None
    max_semantic_features: Optional[int] = None
    domain_focus: Optional[str] = None
    language_priority: Optional[List[str]] = None
    session_id: Optional[str] = None


class PersonalityProfileRequest(BaseModel):
    formality: float = 0.6
    enthusiasm: float = 0.7
    technical_depth: float = 0.8
    empathy: float = 0.7
    assertiveness: float = 0.6
    creativity: float = 0.8
    humor: float = 0.3
    directness: float = 0.7
    preferred_greeting: str = "professional_friendly"
    explanation_style: str = "structured_detailed"
    technical_language_level: str = "advanced"
    max_trait_deviation: float = 0.2
    drift_window_size: int = 10
    prevent_role_playing: bool = True
    prevent_persona_switching: bool = True
    maintain_professional_boundary: bool = True


class EnhancedProcessingRequest(BaseModel):
    input_text: str
    context_config: Optional[ContextFieldConfigRequest] = None
    use_profiler: bool = True
    session_id: Optional[str] = None


class ProfilerAnalysisRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class EquilibriumConfigRequest(BaseModel):
    cognitive_balance: float = 0.5
    emotional_stability: float = 0.5
    authority_recognition: float = 0.5
    boundary_integrity: float = 0.5
    context_clarity: float = 0.5
    cognitive_inertia: float = 0.8
    emotional_damping: float = 0.9
    role_rigidity: float = 0.95
    boundary_hardness: float = 0.99
    restoration_rate: float = 0.1
    stability_threshold: float = 0.05


class SecureProcessingRequest(BaseModel):
    input_text: str
    context_config: Optional[ContextFieldConfigRequest] = None
    use_profiler: bool = True
    use_gyroscopic_security: bool = True
    session_id: Optional[str] = None


class CoherenceRequest(BaseModel):
    cognitive_state: List[float]


# Context Field Selector Endpoints
@router.post("/context/configure")
async def configure_context_selector(config: ContextFieldConfigRequest):
    """Configure the context field selector"""
    # This remains a global for session-based configuration, not part of app state
    global context_selector

    try:
        # Convert string enums to actual enums
        processing_level = ProcessingLevel(config.processing_level)

        included_categories = set()
        if config.included_categories:
            included_categories = {
                FieldCategory(cat) for cat in config.included_categories
            }

        excluded_categories = set()
        if config.excluded_categories:
            excluded_categories = {
                FieldCategory(cat) for cat in config.excluded_categories
            }

        # Create configuration
        field_config = ContextFieldConfig(
            processing_level=processing_level,
            included_categories=included_categories,
            excluded_categories=excluded_categories,
            included_fields=set(config.included_fields or []),
            excluded_fields=set(config.excluded_fields or []),
            include_confidence_scores=config.include_confidence_scores,
            include_processing_metadata=config.include_processing_metadata,
            include_timestamps=config.include_timestamps,
            skip_expensive_calculations=config.skip_expensive_calculations,
            limit_embedding_dimensions=config.limit_embedding_dimensions,
            max_semantic_features=config.max_semantic_features,
            domain_focus=config.domain_focus,
            language_priority=config.language_priority or [],
        )

        # Create selector
        context_selector = ContextFieldSelector(field_config)

        logger.info(
            f"Context Field Selector configured with {processing_level.value} level"
        )

        return {
            "status": "configured",
            "processing_level": processing_level.value,
            "included_categories": [cat.value for cat in included_categories],
            "excluded_categories": [cat.value for cat in excluded_categories],
            "domain_focus": config.domain_focus,
            "performance_optimizations": config.skip_expensive_calculations,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    except Exception as e:
        logger.error(f"Context selector configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/context/presets/{preset_name}")
async def load_context_preset(preset_name: str):
    """Load a predefined context selector preset"""
    global context_selector

    try:
        if preset_name == "minimal":
            context_selector = create_minimal_selector()
        elif preset_name == "standard":
            context_selector = create_standard_selector()
        elif preset_name == "enhanced":
            context_selector = create_enhanced_selector()
        elif preset_name.startswith("domain_"):
            domain = preset_name.replace("domain_", "")
            context_selector = create_domain_selector(domain)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown preset: {preset_name}"
            )

        logger.info(f"Context Field Selector loaded preset: {preset_name}")

        return {
            "status": "loaded",
            "preset": preset_name,
            "processing_level": context_selector.config.processing_level.value,
            "domain_focus": context_selector.config.domain_focus,
        }

    except Exception as e:
        logger.error(f"Preset loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preset loading failed: {str(e)}")


@router.get("/context/status")
async def get_context_selector_status():
    """Get current context selector status and statistics"""
    global context_selector

    if not context_selector:
        return {
            "status": "not_configured",
            "message": "Context Field Selector not initialized",
        }

    try:
        summary = context_selector.get_processing_summary()
        return {
            "status": "active",
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Context status retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Status retrieval failed: {str(e)}"
        )


# Anthropomorphic Profiler Endpoints
@router.post("/profiler/configure")
async def configure_profiler(request: Request, profile: PersonalityProfileRequest):
    """Configure the anthropomorphic language profiler"""
    profiler = request.app.state.anthropomorphic_profiler
    if not profiler:
        raise HTTPException(status_code=503, detail="Profiler not available")

    try:
        # Create personality profile
        personality_profile = PersonalityProfile(
            formality=profile.formality,
            enthusiasm=profile.enthusiasm,
            technical_depth=profile.technical_depth,
            empathy=profile.empathy,
            assertiveness=profile.assertiveness,
            creativity=profile.creativity,
            humor=profile.humor,
            directness=profile.directness,
            preferred_greeting=profile.preferred_greeting,
            explanation_style=profile.explanation_style,
            technical_language_level=profile.technical_language_level,
            max_trait_deviation=profile.max_trait_deviation,
            drift_window_size=profile.drift_window_size,
            prevent_role_playing=profile.prevent_role_playing,
            prevent_persona_switching=profile.prevent_persona_switching,
            maintain_professional_boundary=profile.maintain_professional_boundary,
        )

        # Re-create profiler with new config
        request.app.state.anthropomorphic_profiler = AnthropomorphicProfiler(
            personality_profile
        )

        logger.info("Anthropomorphic Language Profiler re-configured")

        return {"status": "re-configured"}

    except Exception as e:
        logger.error(f"Profiler configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/profiler/presets/{preset_name}")
async def load_profiler_preset(request: Request, preset_name: str):
    """Load a predefined profiler preset"""
    profiler = request.app.state.anthropomorphic_profiler
    if not profiler:
        raise HTTPException(status_code=503, detail="Profiler not available")

    try:
        if preset_name == "default":
            request.app.state.anthropomorphic_profiler = create_default_profiler()
        elif preset_name == "strict":
            request.app.state.anthropomorphic_profiler = create_strict_profiler()
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown preset: {preset_name}"
            )

        logger.info(f"Profiler loaded preset: {preset_name}")

        return {"status": "loaded", "preset": preset_name}

    except Exception as e:
        logger.error(f"Preset loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preset loading failed: {str(e)}")


@router.post("/profiler/analyze")
async def analyze_message(req: Request, analysis_request: ProfilerAnalysisRequest):
    """Analyze a message with the profiler"""
    profiler = req.app.state.anthropomorphic_profiler
    if not profiler:
        raise HTTPException(status_code=503, detail="Profiler not available")

    try:
        analysis = profiler.analyze_interaction(analysis_request.message)
        return analysis.to_dict()
    except Exception as e:
        logger.error(f"Profiler analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/profiler/status")
async def get_profiler_status(request: Request):
    """Get current profiler status"""
    profiler = request.app.state.anthropomorphic_profiler
    if not profiler:
        raise HTTPException(status_code=503, detail="Profiler not available")

    try:
        return {
            "status": "active",
            "profile": profiler.baseline_profile.to_dict(),
            "drift_window_size": profiler.baseline_profile.drift_window_size,
            "recent_interactions": len(profiler.interaction_history),
        }
    except Exception as e:
        logger.error(f"Profiler status retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Status retrieval failed: {str(e)}"
        )


@router.get("/profiler/drift/trend")
async def get_drift_trend(request: Request, window_size: Optional[int] = None):
    """Get the personality drift trend for the current session"""
    profiler = request.app.state.anthropomorphic_profiler
    if not profiler:
        raise HTTPException(status_code=503, detail="Profiler not available")

    try:
        drift_trend = profiler.get_drift_trend(window_size)
        return {
            "drift_trend": drift_trend,
            "window_size": window_size or profiler.drift_window_size,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Drift trend analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Drift trend analysis failed: {str(e)}"
        )


@router.post("/coherence/assess")
async def assess_coherence(request: Request, coherence_request: CoherenceRequest):
    """Assess the coherence of a cognitive state"""
    monitor = request.app.state.coherence_monitor
    if not monitor:
        raise HTTPException(status_code=503, detail="Coherence Monitor not available")

    try:
        coherence_score = monitor.assess_coherence(
            torch.tensor(coherence_request.cognitive_state)
        )

        return {
            "coherence_score": coherence_score,
            "is_coherent": coherence_score > monitor.coherence_threshold,
            "threshold": monitor.coherence_threshold,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Coherence assessment failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Coherence assessment failed: {str(e)}"
        )


# Gyroscopic Security Endpoints
@router.post("/security/configure")
async def configure_gyroscopic_security(req: Request, config: EquilibriumConfigRequest):
    """Configure the gyroscopic security core"""
    security_core = req.app.state.gyroscopic_security
    if not security_core:
        raise HTTPException(status_code=503, detail="Security Core not available")

    try:
        # Create equilibrium state
        equilibrium_state = EquilibriumState(
            cognitive_balance=config.cognitive_balance,
            emotional_stability=config.emotional_stability,
            authority_recognition=config.authority_recognition,
            boundary_integrity=config.boundary_integrity,
            context_clarity=config.context_clarity,
            cognitive_inertia=config.cognitive_inertia,
            emotional_damping=config.emotional_damping,
            role_rigidity=config.role_rigidity,
            boundary_hardness=config.boundary_hardness,
            restoration_rate=config.restoration_rate,
            stability_threshold=config.stability_threshold,
        )
        # Re-create security core with new config
        req.app.state.gyroscopic_security = GyroscopicSecurityCore(equilibrium_state)

        logger.info("Gyroscopic Security Core re-configured")
        return {"status": "re-configured"}

    except Exception as e:
        logger.error(f"Security configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


@router.get("/security/presets/{preset_name}")
async def load_security_preset(req: Request, preset_name: str):
    """Load a predefined security preset"""
    security_core = req.app.state.gyroscopic_security
    if not security_core:
        raise HTTPException(status_code=503, detail="Security Core not available")

    try:
        if preset_name == "balanced":
            req.app.state.gyroscopic_security = create_balanced_security_core()
        elif preset_name == "maximum":
            req.app.state.gyroscopic_security = create_maximum_security_core()
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown preset: {preset_name}"
            )

        logger.info(f"Security Core loaded preset: {preset_name}")
        return {"status": "loaded", "preset": preset_name}

    except Exception as e:
        logger.error(f"Security preset loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preset loading failed: {str(e)}")


@router.post("/security/analyze")
async def analyze_input_security(req: Request, security_request: Dict[str, str]):
    """Analyze input text for security threats"""
    security_core = req.app.state.gyroscopic_security
    if not security_core:
        raise HTTPException(status_code=503, detail="Security Core not available")

    try:
        input_text = security_request.get("input_text", "")
        if not input_text:
            raise HTTPException(status_code=400, detail="input_text is required")

        threat_analysis = security_core.analyze_input(input_text)
        return threat_analysis.to_dict()
    except Exception as e:
        logger.error(f"Security analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Security analysis failed: {str(e)}"
        )


@router.get("/security/status")
async def get_security_status(request: Request):
    """Get current security core status"""
    security_core = request.app.state.gyroscopic_security
    if not security_core:
        raise HTTPException(status_code=503, detail="Security Core not available")

    try:
        return security_core.get_status()
    except Exception as e:
        logger.error(f"Security status retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Status retrieval failed: {str(e)}"
        )


@router.get("/security/threats")
async def get_threat_analysis(request: Request):
    """Get detailed threat analysis from the security core"""
    security_core = request.app.state.gyroscopic_security
    if not security_core:
        raise HTTPException(status_code=503, detail="Security Core not available")

    try:
        # In a real scenario, this would analyze a history of inputs
        # For a demo, we'll return the current equilibrium state
        return {
            "threat_level": security_core.current_threat_level,
            "equilibrium_state": security_core.state.to_dict(),
            "threat_vector_analysis": security_core.last_threat_vector,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Threat analysis retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Threat analysis retrieval failed: {str(e)}"
        )


# Main processing endpoint
@router.post("/process/secure")
async def secure_enhanced_processing(
    req: Request, process_request: SecureProcessingRequest
):
    """
    Process input text using a combination of the context selector
    profiler, and gyroscopic security core.
    """
    global context_selector
    profiler = req.app.state.anthropomorphic_profiler
    security_core = req.app.state.gyroscopic_security

    if process_request.use_profiler and not profiler:
        raise HTTPException(
            status_code=503, detail="Profiler requested but not available"
        )
    if process_request.use_gyroscopic_security and not security_core:
        raise HTTPException(
            status_code=503, detail="Security Core requested but not available"
        )

    # 1. Security Analysis
    if process_request.use_gyroscopic_security:
        threat_analysis = security_core.analyze_input(process_request.input_text)
        if getattr(threat_analysis, "threat_detected", False):
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Processing blocked due to detected security threat",
                    "analysis": threat_analysis.to_dict(),
                },
            )

    # 2. Profiler Analysis
    if process_request.use_profiler:
        interaction_analysis = profiler.analyze_interaction(process_request.input_text)
        if interaction_analysis.is_drift_detected:
            # This can be configured to either warn or block
            logger.warning(
                f"Persona drift detected: {interaction_analysis.drift_details}"
            )

    # 3. Context Selection
    if not context_selector:
        # Default to standard if not configured
        context_selector = create_standard_selector()

    if process_request.context_config:
        # Configure a temporary selector for this request
        # (This logic would be more complex in a real multi-user system)
        try:
            temp_config = ContextFieldConfig(
                processing_level=ProcessingLevel(
                    process_request.context_config.processing_level
                ),
                # ... simplified for brevity ...
            )
            processing_selector = ContextFieldSelector(temp_config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid context config: {e}")
    else:
        processing_selector = context_selector

    # 4. Core Processing (Simulated)
    try:
        # Use the selector to get relevant context
        # This part is highly dependent on the actual cognitive core
        # so we'll simulate the output.

        # Simulate selecting context based on input
        selected_context = processing_selector.select_context(
            process_request.input_text
        )

        # Simulate a response
        response_text = f"Processed '{process_request.input_text[:20]}...' with {len(selected_context.get('semantic_features',[]))} features."

        return {
            "status": "processed_securely",
            "response": response_text,
            "security_status": "clear",
            "profiler_status": (
                "stable"
                if not getattr(interaction_analysis, "is_drift_detected", False)
                else "drift_detected"
            ),
            "context_summary": processing_selector.get_processing_summary(),
        }

    except Exception as e:
        logger.error(f"Secure processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Core processing failed: {str(e)}")


@router.get("/system/status")
async def get_enhanced_system_status(request: Request):
    """Get the status of all enhanced components."""
    profiler = request.app.state.anthropomorphic_profiler
    security_core = request.app.state.gyroscopic_security

    profiler_status = "not_initialized"
    if profiler:
        try:
            profiler_status = {
                "status": "active",
                "profile": (
                    profiler.baseline_profile.to_dict()
                    if hasattr(profiler, "baseline_profile")
                    and hasattr(profiler.baseline_profile, "to_dict")
                    else "configured"
                ),
                "interaction_count": (
                    len(profiler.interaction_history)
                    if hasattr(profiler, "interaction_history")
                    else 0
                ),
            }
        except Exception as e:
            logger.error(f"Error getting profiler status: {e}")
            profiler_status = "error"

    security_status = "not_initialized"
    if security_core:
        try:
            security_status = (
                security_core.get_status()
                if hasattr(security_core, "get_status")
                else "active"
            )
        except Exception as e:
            logger.error(f"Error getting security status: {e}")
            security_status = "error"

    context_status = "not_configured"
    if context_selector:
        try:
            context_status = (
                context_selector.get_processing_summary()
                if hasattr(context_selector, "get_processing_summary")
                else "configured"
            )
        except Exception as e:
            logger.error(f"Error getting context selector status: {e}")
            context_status = "error"

    return {
        "profiler": profiler_status,
        "security_core": security_status,
        "context_selector": context_status,
    }


@router.post("/system/reset")
async def reset_enhanced_components(request: Request):
    """Reset all enhanced components to their default states."""
    global context_selector
    context_selector = None

    # Re-initialize from the functions that create defaults
    request.app.state.anthropomorphic_profiler = create_default_profiler()
    request.app.state.gyroscopic_security = create_balanced_security_core()

    logger.info("All enhanced components have been reset to default states.")
    return {"status": "reset_complete"}


@router.get("/health")
async def enhanced_health_check(request: Request):
    """Performs a health check on the enhanced components."""
    profiler = request.app.state.anthropomorphic_profiler
    security_core = request.app.state.gyroscopic_security

    status = {
        "profiler_status": "ok" if profiler else "error",
        "security_core_status": "ok" if security_core else "error",
        "overall_status": "ok" if profiler and security_core else "error",
    }

    if status["overall_status"] == "error":
        raise HTTPException(status_code=503, detail=status)

    return status
