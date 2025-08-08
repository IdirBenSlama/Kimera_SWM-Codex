"""
Cognitive Architecture Router
=============================

Comprehensive API router for accessing the complete cognitive architecture of Kimera.
Provides endpoints for:

- Cognitive request processing through the full architecture
- System status and health monitoring
- Component status and interconnection analysis
- Flow management and optimization
- Transparency and debugging capabilities
- Performance metrics and analytics
- Coherence monitoring and recommendations

This router exposes the unified cognitive nervous system of Kimera.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import cognitive architecture core
from ...core.cognitive_architecture_core import (
    CognitiveArchitectureCore,
    CognitiveComponent,
    CognitiveFlowResult,
    CognitiveFlowStage,
    CognitiveState,
    get_cognitive_architecture,
)
# Import core system
from ...core.kimera_system import get_kimera_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/cognitive", tags=["Cognitive Architecture"])


# --- Pydantic Models for API ---


class CognitiveRequest(BaseModel):
    """Request model for cognitive processing"""

    input_text: str = Field(
        ..., description="Text input for cognitive processing", max_length=16384
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context information"
    )
    processing_preferences: Optional[Dict[str, Any]] = Field(
        default=None, description="Processing preferences"
    )


class CognitiveResponse(BaseModel):
    """Response model for cognitive processing"""

    success: bool
    response: Optional[str] = None
    confidence: float
    processing_time: float

    # Cognitive Analysis
    understanding_analysis: Dict[str, Any]
    insight_events: List[Dict[str, Any]]
    consciousness_indicators: Dict[str, float]

    # Flow Information
    stages_completed: List[str]
    component_contributions: Dict[str, float]

    # Transparency
    processing_trace: List[Dict[str, Any]]
    confidence_breakdown: Dict[str, float]

    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ArchitectureStatusResponse(BaseModel):
    """Response model for architecture status"""

    initialized: bool
    coherence_score: float
    active_components: int
    total_components: int
    component_status: Dict[str, bool]
    system_health: str
    performance_metrics: Dict[str, Any]
    flow_stage: str
    transparency_report: Dict[str, Any]


class ComponentAnalysisResponse(BaseModel):
    """Response model for component analysis"""

    components: List[str]
    interconnections: Dict[str, List[str]]
    component_health: Dict[str, Dict[str, Any]]
    processing_flows: Dict[str, Any]
    recommendations: List[str]


class CoherenceReportResponse(BaseModel):
    """Response model for coherence analysis"""

    overall_coherence: float
    component_coherence: float
    flow_coherence: float
    interconnection_coherence: float
    health_indicators: Dict[str, float]
    recommendations: List[str]
    improvement_suggestions: List[str]


class FlowOptimizationRequest(BaseModel):
    """Request model for flow optimization"""

    target_components: Optional[List[str]] = Field(
        default=None, description="Target components to optimize"
    )
    optimization_goals: List[str] = Field(
        default=["speed", "accuracy"], description="Optimization goals"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default=None, description="Optimization constraints"
    )


class TransparencyRequest(BaseModel):
    """Request model for transparency analysis"""

    component_filter: Optional[List[str]] = Field(
        default=None, description="Components to analyze"
    )
    time_range: Optional[Dict[str, str]] = Field(
        default=None, description="Time range for analysis"
    )
    detail_level: str = Field(
        default="standard", description="Detail level: minimal, standard, comprehensive"
    )


# --- Dependency Functions ---


async def get_cognitive_architecture_dependency() -> CognitiveArchitectureCore:
    """Dependency to get the cognitive architecture core"""
    try:
        return await get_cognitive_architecture()
    except Exception as e:
        logger.error(f"Failed to get cognitive architecture: {e}")
        raise HTTPException(
            status_code=500, detail=f"Cognitive architecture unavailable: {str(e)}"
        )


def validate_component_names(components: Optional[List[str]]) -> Optional[List[str]]:
    """Validate component names"""
    if components is None:
        return None

    valid_components = [comp.value for comp in CognitiveComponent]
    invalid_components = [comp for comp in components if comp not in valid_components]

    if invalid_components:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid components: {invalid_components}. Valid options: {valid_components}",
        )

    return components


# --- API Endpoints ---


@router.get("/", summary="Cognitive Architecture Overview")
async def cognitive_overview():
    """Get overview of the cognitive architecture"""
    return {
        "service": "Kimera Cognitive Architecture Core",
        "description": "Unified cognitive nervous system with complete transparency and coherence",
        "components": [comp.value for comp in CognitiveComponent],
        "flow_stages": [stage.value for stage in CognitiveFlowStage],
        "endpoints": {
            "process": "POST /process - Process cognitive requests",
            "status": "GET /status - Architecture status",
            "components": "GET /components - Component analysis",
            "coherence": "GET /coherence - Coherence report",
            "transparency": "POST /transparency - Transparency analysis",
            "optimize": "POST /optimize - Flow optimization",
        },
        "capabilities": [
            "Zetetic cognitive processing",
            "Complete system transparency",
            "Interconnected component flow",
            "Coherence monitoring",
            "Adaptive optimization",
        ],
        "version": "2.0.0",
    }


@router.post(
    "/process", response_model=CognitiveResponse, summary="Process Cognitive Request"
)
async def process_cognitive_request(
    request: CognitiveRequest,
    architecture: CognitiveArchitectureCore = Depends(
        get_cognitive_architecture_dependency
    ),
):
    """
    Process a cognitive request through the complete cognitive architecture

    This endpoint provides access to the full cognitive processing pipeline:
    - Linguistic intelligence processing
    - Understanding and comprehension analysis
    - Meta-insight generation
    - Consciousness emergence detection
    - Revolutionary intelligence processing
    - Response optimization and human interface

    All processing is transparent and traceable.
    """
    start_time = datetime.now()

    try:
        # Process through cognitive architecture
        flow_result = await architecture.process_cognitive_request(
            request.input_text, request.context
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return CognitiveResponse(
            success=True,
            response=flow_result.response,
            confidence=flow_result.confidence,
            processing_time=processing_time,
            understanding_analysis=flow_result.understanding_analysis,
            insight_events=flow_result.insight_events,
            consciousness_indicators=flow_result.consciousness_indicators,
            stages_completed=flow_result.stages_completed,
            component_contributions=flow_result.component_contributions,
            processing_trace=flow_result.processing_trace,
            confidence_breakdown=flow_result.confidence_breakdown,
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Cognitive processing failed: {e}", exc_info=True)

        return CognitiveResponse(
            success=False,
            response=None,
            confidence=0.0,
            processing_time=processing_time,
            understanding_analysis={},
            insight_events=[],
            consciousness_indicators={},
            stages_completed=[],
            component_contributions={},
            processing_trace=[],
            confidence_breakdown={},
            error=str(e),
        )


@router.get(
    "/status", response_model=ArchitectureStatusResponse, summary="Architecture Status"
)
async def get_architecture_status(
    architecture: CognitiveArchitectureCore = Depends(
        get_cognitive_architecture_dependency
    ),
):
    """Get comprehensive status of the cognitive architecture"""
    try:
        status = architecture.get_architecture_status()

        return ArchitectureStatusResponse(
            initialized=status["initialized"],
            coherence_score=status["coherence_score"],
            active_components=status["active_components"],
            total_components=status["total_components"],
            component_status=status["component_status"],
            system_health=status["system_health"],
            performance_metrics=status["performance_metrics"],
            flow_stage=status["flow_stage"],
            transparency_report=status["transparency_report"],
        )

    except Exception as e:
        logger.error(f"Failed to get architecture status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Status retrieval failed: {str(e)}"
        )


@router.get(
    "/components",
    response_model=ComponentAnalysisResponse,
    summary="Component Analysis",
)
async def analyze_components(
    architecture: CognitiveArchitectureCore = Depends(
        get_cognitive_architecture_dependency
    ),
):
    """Get detailed analysis of all cognitive components"""
    try:
        # Get component information
        component_names = [comp.value for comp in CognitiveComponent]

        # Get interconnections
        interconnections = {}
        if hasattr(architecture.interconnection_matrix, "established_connections"):
            interconnections = (
                architecture.interconnection_matrix.established_connections
            )

        # Get component health
        component_health = {}
        for comp_name in component_names:
            health_data = {
                "active": architecture.component_status.get(comp_name, False),
                "initialized": comp_name in architecture.components,
                "has_instance": architecture.components.get(comp_name) is not None,
            }

            # Add performance data if available
            if comp_name in architecture.transparency_layer.component_metrics:
                metrics = architecture.transparency_layer.component_metrics[comp_name]
                health_data.update(
                    {
                        "total_processes": metrics.get("total_processes", 0),
                        "success_rate": metrics.get("success_rate", 0.0),
                        "last_activity": metrics.get(
                            "last_activity", datetime.now()
                        ).isoformat(),
                    }
                )

            component_health[comp_name] = health_data

        # Get processing flows
        processing_flows = {}
        if hasattr(architecture.flow_manager, "calibration_results"):
            processing_flows = architecture.flow_manager.calibration_results

        # Generate recommendations
        recommendations = []
        inactive_components = [
            name for name, active in architecture.component_status.items() if not active
        ]
        if inactive_components:
            recommendations.append(
                f"Initialize inactive components: {', '.join(inactive_components)}"
            )

        if architecture.state.coherence_score < 0.8:
            recommendations.append(
                "Improve system coherence through component optimization"
            )

        return ComponentAnalysisResponse(
            components=component_names,
            interconnections=interconnections,
            component_health=component_health,
            processing_flows=processing_flows,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Component analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Component analysis failed: {str(e)}"
        )


@router.get(
    "/coherence", response_model=CoherenceReportResponse, summary="Coherence Report"
)
async def get_coherence_report(
    architecture: CognitiveArchitectureCore = Depends(
        get_cognitive_architecture_dependency
    ),
):
    """Get detailed coherence analysis of the cognitive system"""
    try:
        # Perform coherence check
        coherence_result = await architecture._perform_coherence_check()

        # Calculate health indicators
        health_indicators = {
            "component_availability": len(
                [c for c in architecture.component_status.values() if c]
            )
            / len(architecture.component_status),
            "processing_efficiency": architecture.performance_metrics.get(
                "flow_efficiency", 0.0
            ),
            "system_stability": architecture.state.coherence_score,
            "transparency_coverage": (
                1.0 if architecture.transparency_layer.component_metrics else 0.0
            ),
        }

        # Generate improvement suggestions
        improvement_suggestions = []

        if coherence_result["metrics"]["component_coherence"] < 0.9:
            improvement_suggestions.append(
                "Consider initializing missing cognitive components"
            )

        if coherence_result["metrics"]["flow_coherence"] < 0.9:
            improvement_suggestions.append(
                "Optimize cognitive flow pathways for better efficiency"
            )

        if coherence_result["metrics"]["interconnection_coherence"] < 0.9:
            improvement_suggestions.append("Strengthen component interconnections")

        if (
            architecture.performance_metrics["successful_processes"]
            < architecture.performance_metrics["total_processes"] * 0.9
        ):
            improvement_suggestions.append(
                "Investigate and resolve processing failures"
            )

        return CoherenceReportResponse(
            overall_coherence=coherence_result["coherence_score"],
            component_coherence=coherence_result["metrics"]["component_coherence"],
            flow_coherence=coherence_result["metrics"]["flow_coherence"],
            interconnection_coherence=coherence_result["metrics"][
                "interconnection_coherence"
            ],
            health_indicators=health_indicators,
            recommendations=coherence_result["recommendations"],
            improvement_suggestions=improvement_suggestions,
        )

    except Exception as e:
        logger.error(f"Coherence analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Coherence analysis failed: {str(e)}"
        )


@router.post("/transparency", summary="Transparency Analysis")
async def analyze_transparency(
    request: TransparencyRequest,
    architecture: CognitiveArchitectureCore = Depends(
        get_cognitive_architecture_dependency
    ),
):
    """Get detailed transparency analysis of cognitive processing"""
    try:
        # Validate component filter
        if request.component_filter:
            validate_component_names(request.component_filter)

        # Get transparency report
        transparency_report = architecture.transparency_layer.get_transparency_report()

        # Filter by components if requested
        if request.component_filter:
            filtered_performance = {
                comp: data
                for comp, data in transparency_report["component_performance"].items()
                if comp in request.component_filter
            }
            transparency_report["component_performance"] = filtered_performance

        # Adjust detail level
        if request.detail_level == "minimal":
            # Remove detailed traces
            transparency_report.pop("recent_activity", None)
            transparency_report.pop("flow_patterns", None)
        elif request.detail_level == "comprehensive":
            # Add extra analysis
            transparency_report["processing_analysis"] = {
                "total_processing_time": sum(
                    metrics.get("total_time", 0)
                    for metrics in transparency_report["component_performance"].values()
                ),
                "component_efficiency": {
                    comp: metrics.get("success_rate", 0) * 100
                    for comp, metrics in transparency_report[
                        "component_performance"
                    ].items()
                },
                "bottleneck_analysis": architecture.transparency_layer._analyze_flow_patterns(),
            }

        return {
            "success": True,
            "transparency_report": transparency_report,
            "analysis_timestamp": datetime.now().isoformat(),
            "detail_level": request.detail_level,
            "components_analyzed": len(transparency_report["component_performance"]),
        }

    except Exception as e:
        logger.error(f"Transparency analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Transparency analysis failed: {str(e)}"
        )


@router.post("/optimize", summary="Flow Optimization")
async def optimize_cognitive_flow(
    request: FlowOptimizationRequest,
    architecture: CognitiveArchitectureCore = Depends(
        get_cognitive_architecture_dependency
    ),
):
    """Optimize cognitive flow based on specified goals and constraints"""
    try:
        # Validate target components
        if request.target_components:
            validate_component_names(request.target_components)

        # Analyze current performance
        current_metrics = architecture.performance_metrics.copy()

        # Generate optimization recommendations
        optimization_recommendations = []

        if "speed" in request.optimization_goals:
            # Recommend parallel processing optimizations
            if current_metrics["average_processing_time"] > 1.0:
                optimization_recommendations.append(
                    {
                        "goal": "speed",
                        "recommendation": "Enable parallel component processing",
                        "expected_improvement": "30-50% faster processing",
                        "implementation": "Use asyncio.gather for concurrent component execution",
                    }
                )

        if "accuracy" in request.optimization_goals:
            # Recommend accuracy improvements
            success_rate = current_metrics.get("successful_processes", 0) / max(
                current_metrics.get("total_processes", 1), 1
            )
            if success_rate < 0.9:
                optimization_recommendations.append(
                    {
                        "goal": "accuracy",
                        "recommendation": "Improve component error handling and fallback mechanisms",
                        "expected_improvement": f"Increase success rate from {success_rate:.1%} to >90%",
                        "implementation": "Add robust error recovery in critical components",
                    }
                )

        if "coherence" in request.optimization_goals:
            # Recommend coherence improvements
            if architecture.state.coherence_score < 0.9:
                optimization_recommendations.append(
                    {
                        "goal": "coherence",
                        "recommendation": "Strengthen component interconnections",
                        "expected_improvement": "Improved system coherence and reliability",
                        "implementation": "Initialize missing components and optimize flow pathways",
                    }
                )

        # Apply constraints
        if request.constraints:
            max_components = request.constraints.get("max_components")
            if max_components and len(optimization_recommendations) > max_components:
                optimization_recommendations = optimization_recommendations[
                    :max_components
                ]

        return {
            "success": True,
            "optimization_analysis": {
                "current_performance": current_metrics,
                "optimization_goals": request.optimization_goals,
                "target_components": request.target_components or "all",
                "recommendations": optimization_recommendations,
                "constraints_applied": request.constraints or {},
            },
            "implementation_priority": [
                rec["recommendation"]
                for rec in sorted(
                    optimization_recommendations
                    key=lambda x: len(x.get("expected_improvement", "")),
                    reverse=True
                )
            ],
            "analysis_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Flow optimization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Flow optimization failed: {str(e)}"
        )


@router.get("/health", summary="System Health Check")
async def health_check():
    """Check health status of the cognitive architecture"""
    try:
        # Try to get the architecture
        architecture = await get_cognitive_architecture()

        # Get detailed status
        status = architecture.get_architecture_status()

        # Calculate health score
        health_factors = [
            status["coherence_score"],
            status["active_components"] / max(status["total_components"], 1),
            1.0 if status["initialized"] else 0.0
        ]

        health_score = sum(health_factors) / len(health_factors)

        # Determine health status
        if health_score >= 0.9:
            health_status = "excellent"
        elif health_score >= 0.8:
            health_status = "good"
        elif health_score >= 0.6:
            health_status = "fair"
        else:
            health_status = "poor"

        return {
            "status": health_status,
            "health_score": health_score,
            "system_initialized": status["initialized"],
            "coherence_score": status["coherence_score"],
            "active_components": status["active_components"],
            "total_components": status["total_components"],
            "component_availability": status["active_components"]
            / max(status["total_components"], 1),
            "performance_metrics": status["performance_metrics"],
            "timestamp": datetime.now().isoformat(),
            "recommendations": [
                (
                    "System operating within normal parameters"
                    if health_score >= 0.8
                    else (
                        "Consider component optimization"
                        if health_score >= 0.6
                        else "System requires attention - multiple components may need initialization"
                    )
                )
            ],
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "health_score": 0.0,
            "system_initialized": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@router.get("/flows", summary="Available Cognitive Flows")
async def get_cognitive_flows():
    """Get information about available cognitive processing flows"""
    return {
        "cognitive_flows": [
            {
                "name": "linguistic_to_understanding",
                "description": "Process text through linguistic intelligence to understanding",
                "components": ["linguistic_intelligence", "understanding_engine"],
                "typical_time": "100-300ms",
                "use_cases": ["Text comprehension", "Semantic analysis"],
            },
            {
                "name": "understanding_to_insight",
                "description": "Generate insights from understanding",
                "components": ["understanding_engine", "meta_insight_engine"],
                "typical_time": "200-500ms",
                "use_cases": ["Pattern recognition", "Knowledge extraction"],
            },
            {
                "name": "insight_to_consciousness",
                "description": "Detect consciousness emergence from insights",
                "components": ["meta_insight_engine", "consciousness_detector"],
                "typical_time": "300-800ms",
                "use_cases": ["Consciousness detection", "Emergent behavior analysis"],
            },
            {
                "name": "full_cognitive_pipeline",
                "description": "Complete cognitive processing pipeline",
                "components": [
                    "linguistic_intelligence",
                    "understanding_engine",
                    "meta_insight_engine",
                    "consciousness_detector",
                    "revolutionary_intelligence",
                ],
                "typical_time": "500-1500ms",
                "use_cases": ["Comprehensive cognitive analysis", "Advanced reasoning"],
            },
        ],
        "flow_stages": [stage.value for stage in CognitiveFlowStage],
        "optimization_options": [
            "parallel_processing",
            "component_caching",
            "early_termination",
            "adaptive_routing",
        ],
    }


# Initialize router metadata
router.summary = "Cognitive Architecture Core API"
router.description = """
Comprehensive cognitive architecture API for Kimera, providing:

- **Cognitive Processing**: Complete cognitive request processing through integrated architecture
- **System Status**: Real-time architecture status and health monitoring
- **Component Analysis**: Detailed component status and interconnection analysis
- **Coherence Monitoring**: System coherence analysis and optimization recommendations
- **Transparency**: Complete transparency into cognitive processing flows
- **Flow Optimization**: Cognitive flow optimization and performance tuning

**Zetetic Principles**: This API embodies investigative creativity with complete transparency
ensuring functioning, interconnectedness, flow, interoperability, and coherence.
"""

logger.info("🧠 Cognitive Architecture Router initialized")
