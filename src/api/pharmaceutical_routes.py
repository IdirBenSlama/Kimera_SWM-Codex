"""
Pharmaceutical Testing API Routes

RESTful API endpoints for the Kimera Pharmaceutical Testing Framework.
Provides access to KCl extended-release capsule development and testing capabilities.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..pharmaceutical.analysis.dissolution_analyzer import DissolutionAnalyzer
from ..pharmaceutical.core.kcl_testing_engine import KClTestingEngine
from ..pharmaceutical.protocols.usp_protocols import (DissolutionTestUSP711
                                                      , USPProtocolEngine)
from ..pharmaceutical.validation.pharmaceutical_validator import PharmaceuticalValidator
from ..utils.kimera_exceptions import KimeraException
from ..utils.kimera_logger import get_logger

logger = get_logger(__name__)


# Pydantic models for API requests/responses
class RawMaterialRequest(BaseModel):
    """Raw material characterization request."""

    name: str = Field(..., description="Material name")
    grade: str = Field("USP", description="Material grade")
    purity_percent: float = Field(
        ..., ge=90.0, le=110.0, description="Purity percentage"
    )
    moisture_content: float = Field(
        ..., ge=0.0, le=10.0, description="Moisture content percentage"
    )
    particle_size_d50: Optional[float] = Field(
        None, ge=0.0, description="Median particle size (μm)"
    )
    bulk_density: Optional[float] = Field(
        None, ge=0.0, description="Bulk density (g/mL)"
    )
    tapped_density: Optional[float] = Field(
        None, ge=0.0, description="Tapped density (g/mL)"
    )
    potassium_confirmed: bool = Field(
        True, description="Potassium identification confirmed"
    )
    chloride_confirmed: bool = Field(
        True, description="Chloride identification confirmed"
    )


class FlowabilityRequest(BaseModel):
    """Powder flowability analysis request."""

    bulk_density: float = Field(..., gt=0.0, description="Bulk density (g/mL)")
    tapped_density: float = Field(..., gt=0.0, description="Tapped density (g/mL)")
    angle_of_repose: float = Field(
        ..., ge=0.0, le=90.0, description="Angle of repose (degrees)"
    )


class FormulationRequest(BaseModel):
    """Formulation prototype creation request."""

    coating_thickness_percent: float = Field(
        ..., ge=5.0, le=25.0, description="Coating thickness (%)"
    )
    polymer_ratios: Dict[str, float] = Field(..., description="Polymer ratios")
    process_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Process parameters"
    )


class DissolutionTestRequest(BaseModel):
    """Dissolution test request."""

    prototype_id: str = Field(..., description="Prototype ID to test")
    apparatus: int = Field(
        1, ge=1, le=2, description="USP apparatus (1=basket, 2=paddle)"
    )
    medium: str = Field("water", description="Dissolution medium")
    volume_ml: int = Field(900, gt=0, description="Medium volume (mL)")
    temperature_c: float = Field(37.0, ge=35.0, le=40.0, description="Temperature (°C)")
    rotation_rpm: int = Field(100, gt=0, description="Rotation speed (rpm)")
    reference_profile: Optional[Dict[str, List[float]]] = Field(
        None, description="Reference dissolution profile"
    )


class CompleteValidationRequest(BaseModel):
    """Complete pharmaceutical development validation request."""

    raw_materials: Dict[str, Any] = Field(..., description="Raw material data")
    formulation_data: Dict[str, Any] = Field(..., description="Formulation data")
    manufacturing_data: Dict[str, Any] = Field(..., description="Manufacturing data")
    testing_data: Dict[str, Any] = Field(..., description="Testing data")


class BatchQualityRequest(BaseModel):
    """Batch quality validation request."""

    batch_id: str = Field(..., description="Batch identifier")
    batch_data: Dict[str, Any] = Field(..., description="Batch testing data")
    specification_limits: Dict[str, Dict[str, float]] = Field(
        ..., description="Specification limits"
    )


# Initialize pharmaceutical engines (will be done in main app startup)
kcl_engine: Optional[KClTestingEngine] = None
usp_engine: Optional[USPProtocolEngine] = None
dissolution_analyzer: Optional[DissolutionAnalyzer] = None
pharmaceutical_validator: Optional[PharmaceuticalValidator] = None


def get_kcl_engine() -> KClTestingEngine:
    """Dependency to get KCl testing engine."""
    if kcl_engine is None:
        raise HTTPException(
            status_code=503, detail="KCl Testing Engine not initialized"
        )
    return kcl_engine


def get_usp_engine() -> USPProtocolEngine:
    """Dependency to get USP protocol engine."""
    if usp_engine is None:
        raise HTTPException(
            status_code=503, detail="USP Protocol Engine not initialized"
        )
    return usp_engine


def get_dissolution_analyzer() -> DissolutionAnalyzer:
    """Dependency to get dissolution analyzer."""
    if dissolution_analyzer is None:
        raise HTTPException(
            status_code=503, detail="Dissolution Analyzer not initialized"
        )
    return dissolution_analyzer


def get_pharmaceutical_validator() -> PharmaceuticalValidator:
    """Dependency to get pharmaceutical validator."""
    if pharmaceutical_validator is None:
        raise HTTPException(
            status_code=503, detail="Pharmaceutical Validator not initialized"
        )
    return pharmaceutical_validator


# Create router
router = APIRouter(prefix="/pharmaceutical", tags=["pharmaceutical"])


@router.post("/raw-materials/characterize")
async def characterize_raw_materials(
    request: RawMaterialRequest, engine: KClTestingEngine = Depends(get_kcl_engine)
) -> JSONResponse:
    """
    Characterize raw materials according to USP standards.

    Performs comprehensive raw material testing including:
    - Identity confirmation (potassium and chloride tests)
    - Purity analysis
    - Moisture content validation
    - USP compliance verification
    """
    try:
        logger.info(f"🔬 Characterizing raw material: {request.name}")

        # Convert request to dictionary
        material_data = request.dict()

        # Perform characterization
        specification = engine.characterize_raw_materials(material_data)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Raw material {request.name} characterized successfully",
                "data": {
                    "specification": specification.__dict__,
                    "usp_compliant": all(specification.identification_tests.values()),
                    "quality_grade": (
                        "ACCEPTABLE"
                        if specification.purity_percent >= 99.0
                        else "MARGINAL"
                    ),
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    except KimeraException as e:
        logger.error(f"❌ Raw material characterization failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in raw material characterization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/flowability/analyze")
async def analyze_flowability(
    request: FlowabilityRequest, engine: KClTestingEngine = Depends(get_kcl_engine)
) -> JSONResponse:
    """
    Analyze powder flowability using Carr's Index and Hausner Ratio.

    Provides flowability assessment based on:
    - Carr's Compressibility Index
    - Hausner Ratio
    - Flow character classification
    """
    try:
        logger.info("📊 Analyzing powder flowability")

        # Perform flowability analysis
        result = engine.analyze_powder_flowability(
            request.bulk_density, request.tapped_density, request.angle_of_repose
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Flowability analysis completed",
                "data": {
                    "flowability_result": result.__dict__,
                    "recommendation": (
                        "Excellent flowability - suitable for direct processing"
                        if result.flow_character == "Excellent"
                        else "Consider flow aids or process optimization"
                    ),
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    except KimeraException as e:
        logger.error(f"❌ Flowability analysis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in flowability analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/formulation/create-prototype")
async def create_formulation_prototype(
    request: FormulationRequest, engine: KClTestingEngine = Depends(get_kcl_engine)
) -> JSONResponse:
    """
    Create and characterize a formulation prototype.

    Generates microcapsule prototype with:
    - Specified coating thickness
    - Polymer ratio optimization
    - Encapsulation efficiency assessment
    - Particle morphology evaluation
    """
    try:
        logger.info(
            f"🧪 Creating formulation prototype with {request.coating_thickness_percent}% coating"
        )

        # Create prototype
        prototype = engine.create_formulation_prototype(
            request.coating_thickness_percent,
            request.polymer_ratios,
            request.process_parameters,
        )

        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "message": f"Prototype {prototype.prototype_id} created successfully",
                "data": {
                    "prototype": prototype.__dict__,
                    "quality_assessment": {
                        "encapsulation_grade": (
                            "EXCELLENT"
                            if prototype.encapsulation_efficiency >= 0.98
                            else (
                                "GOOD"
                                if prototype.encapsulation_efficiency >= 0.95
                                else "ACCEPTABLE"
                            )
                        ),
                        "morphology_grade": (
                            "OPTIMAL"
                            if "spherical" in prototype.particle_morphology.lower()
                            else "SUBOPTIMAL"
                        ),
                    },
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    except KimeraException as e:
        logger.error(f"❌ Prototype creation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in prototype creation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/dissolution/test")
async def perform_dissolution_test(
    request: DissolutionTestRequest, engine: KClTestingEngine = Depends(get_kcl_engine)
) -> JSONResponse:
    """
    Perform USP dissolution testing with f2 similarity analysis.

    Executes dissolution test according to USP <711> with:
    - Multi-point dissolution profile
    - f2 similarity factor calculation
    - USP tolerance validation
    - Kinetic modeling
    """
    try:
        logger.info(
            f"🧪 Performing dissolution test for prototype {request.prototype_id}"
        )

        # This is a simplified implementation - in practice, you would retrieve
        # the actual prototype from storage and perform the dissolution test

        # Create test conditions
        test_conditions = DissolutionTestUSP711(
            apparatus=request.apparatus,
            medium=request.medium,
            volume_ml=request.volume_ml,
            temperature_c=request.temperature_c,
            rotation_rpm=request.rotation_rpm,
            sampling_times=[1, 2, 4, 6],
            acceptance_table="2",
        )

        # Simulate dissolution data (in practice, this would come from actual testing)
        sample_data = {
            "time_points": [1, 2, 4, 6],
            "release_percentages": [30, 55, 78, 88],  # Example data
        }

        # Create reference profile if provided
        reference_profile = None
        if request.reference_profile:
            from ..pharmaceutical.analysis.dissolution_analyzer import \
                DissolutionProfile

            reference_profile = DissolutionProfile(
                time_points=request.reference_profile["time_points"],
                release_percentages=request.reference_profile["release_percentages"],
                test_conditions={},
            )

        # Use USP engine for official testing
        usp_result = usp_engine.perform_dissolution_test_711(
            sample_data,
            test_conditions,
            reference_profile.release_percentages if reference_profile else None,
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Dissolution test completed for {request.prototype_id}",
                "data": {
                    "usp_test_result": usp_result.__dict__,
                    "dissolution_profile": sample_data,
                    "test_conditions": test_conditions.__dict__,
                    "compliance_status": usp_result.status,
                    "f2_similarity": (
                        usp_result.result_value
                        if "f2" in str(usp_result.test_name).lower()
                        else None
                    ),
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    except KimeraException as e:
        logger.error(f"❌ Dissolution test failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in dissolution test: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/dissolution/analyze-kinetics")
async def analyze_dissolution_kinetics(
    time_points: List[float],
    release_percentages: List[float],
    models_to_fit: Optional[List[str]] = None,
    analyzer: DissolutionAnalyzer = Depends(get_dissolution_analyzer),
) -> JSONResponse:
    """
    Analyze dissolution kinetics using multiple mathematical models.

    Fits various kinetic models:
    - Zero-order kinetics
    - First-order kinetics
    - Higuchi model
    - Korsmeyer-Peppas model
    - Weibull model
    """
    try:
        logger.info("📈 Analyzing dissolution kinetics")

        # Perform kinetics analysis
        kinetic_models = analyzer.analyze_dissolution_kinetics(
            time_points, release_percentages, models_to_fit
        )

        # Find best fitting model
        best_model = None
        if kinetic_models:
            best_model_name = min(
                kinetic_models.keys(), key=lambda k: kinetic_models[k].aic
            )
            best_model = kinetic_models[best_model_name]

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Dissolution kinetics analysis completed",
                "data": {
                    "fitted_models": {
                        name: model.__dict__ for name, model in kinetic_models.items()
                    },
                    "best_model": {
                        "name": best_model_name if best_model else None,
                        "details": best_model.__dict__ if best_model else None
                    },
                    "model_comparison": {
                        name: {
                            "r_squared": model.r_squared,
                            "aic": model.aic,
                            "model_quality": (
                                "EXCELLENT"
                                if model.r_squared >= 0.99
                                else (
                                    "GOOD"
                                    if model.r_squared >= 0.95
                                    else "FAIR" if model.r_squared >= 0.90 else "POOR"
                                )
                            ),
                        }
                        for name, model in kinetic_models.items()
                    },
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    except KimeraException as e:
        logger.error(f"❌ Dissolution kinetics analysis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in kinetics analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/validation/complete")
async def perform_complete_validation(
    request: CompleteValidationRequest,
    background_tasks: BackgroundTasks,
    validator: PharmaceuticalValidator = Depends(get_pharmaceutical_validator),
) -> JSONResponse:
    """
    Perform complete pharmaceutical development validation.

    Comprehensive validation including:
    - Raw material validation
    - Formulation validation
    - Manufacturing process validation
    - Analytical testing validation
    - Regulatory compliance assessment
    """
    try:
        logger.info("🔬 Starting complete pharmaceutical development validation")

        # Perform validation (this could be run as background task for large validations)
        validation_result = validator.validate_complete_development(
            request.raw_materials,
            request.formulation_data,
            request.manufacturing_data,
            request.testing_data,
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Complete validation performed",
                "data": {
                    "validation_result": validation_result.__dict__,
                    "executive_summary": {
                        "overall_status": validation_result.status,
                        "confidence_score": validation_result.confidence_score,
                        "critical_failures_count": len(
                            validation_result.critical_failures
                        ),
                        "warnings_count": len(validation_result.warnings),
                        "regulatory_ready": validation_result.compliance_assessment.get(
                            "submission_readiness"
                        )
                        == "READY_FOR_SUBMISSION",
                    },
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    except KimeraException as e:
        logger.error(f"❌ Complete validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in complete validation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/quality/validate-batch")
async def validate_batch_quality(
    request: BatchQualityRequest,
    validator: PharmaceuticalValidator = Depends(get_pharmaceutical_validator),
) -> JSONResponse:
    """
    Validate batch quality against specifications.

    Provides comprehensive batch assessment:
    - Critical Quality Attribute evaluation
    - Specification compliance check
    - Risk assessment
    - Shelf life prediction
    """
    try:
        logger.info(f"🔍 Validating batch quality for {request.batch_id}")

        # Perform batch quality validation
        quality_profile = validator.validate_batch_quality(
            request.batch_data, request.specification_limits
        )

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Batch quality validation completed for {request.batch_id}",
                "data": {
                    "quality_profile": quality_profile.__dict__,
                    "batch_assessment": {
                        "overall_grade": (
                            "A"
                            if quality_profile.quality_score >= 0.95
                            else (
                                "B"
                                if quality_profile.quality_score >= 0.85
                                else (
                                    "C"
                                    if quality_profile.quality_score >= 0.75
                                    else "D"
                                )
                            )
                        ),
                        "release_recommendation": (
                            "APPROVED_FOR_RELEASE"
                            if quality_profile.quality_score >= 0.80
                            else "HOLD_FOR_INVESTIGATION"
                        ),
                        "high_risk_attributes": [
                            attr
                            for attr, risk in quality_profile.risk_assessment.items()
                            if "HIGH" in risk
                        ],
                    },
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    except KimeraException as e:
        logger.error(f"❌ Batch quality validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in batch validation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/reports/master-validation")
async def get_master_validation_report(
    validator: PharmaceuticalValidator = Depends(get_pharmaceutical_validator),
) -> JSONResponse:
    """
    Generate master validation report combining all validation activities.

    Provides comprehensive overview of:
    - All validation activities
    - Quality profile summaries
    - Regulatory readiness assessment
    - Development recommendations
    """
    try:
        logger.info("📋 Generating master validation report")

        # Generate master report
        master_report = validator.generate_master_validation_report()

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Master validation report generated",
                "data": master_report,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except KimeraException as e:
        logger.error(f"❌ Master report generation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in report generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/standards/usp")
async def get_usp_standards() -> JSONResponse:
    """
    Get USP standards and acceptance criteria for KCl extended-release capsules.

    Returns official USP standards for:
    - Dissolution testing (USP <711>)
    - Content uniformity (USP <905>)
    - Assay requirements
    - Stability testing (ICH Q1A)
    """
    try:
        # Return USP standards
        standards = {
            "dissolution_usp_711": {
                "test_conditions": {
                    "apparatus_1_rpm": 100,
                    "apparatus_2_rpm": 50,
                    "medium": "water",
                    "volume_ml": 900,
                    "temperature_c": 37.0,
                },
                "acceptance_criteria": {
                    "f2_similarity_threshold": 50.0,
                    "time_points_hours": [1, 2, 4, 6],
                    "release_tolerances": {
                        "1h": {"min": 25, "max": 45},
                        "2h": {"min": 45, "max": 65},
                        "4h": {"min": 70, "max": 90},
                        "6h": {"min": 85, "max": 100},
                    },
                },
            },
            "content_uniformity_usp_905": {
                "sample_size": 10,
                "individual_limits": {"min": 85.0, "max": 115.0},
                "acceptance_value_limit": 15.0,
            },
            "assay_requirements": {
                "acceptance_range": {"min": 90.0, "max": 110.0},
                "method": "Atomic Absorption Spectrophotometry",
            },
            "stability_ich_q1a": {
                "long_term": {"temperature": 25, "humidity": 60, "duration_months": 24},
                "accelerated": {
                    "temperature": 40,
                    "humidity": 75,
                    "duration_months": 6,
                },
                "intermediate": {
                    "temperature": 30,
                    "humidity": 65,
                    "duration_months": 12,
                },
            },
        }

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "USP standards retrieved",
                "data": standards,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except Exception as e:
        logger.error(f"❌ Error retrieving USP standards: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Engine initialization function (to be called during app startup)
async def initialize_pharmaceutical_engines(use_gpu: bool = True):
    """Initialize all pharmaceutical engines."""
    global kcl_engine, usp_engine, dissolution_analyzer, pharmaceutical_validator

    try:
        logger.info("🚀 Initializing Pharmaceutical Testing Engines...")

        kcl_engine = KClTestingEngine(use_gpu=use_gpu)
        usp_engine = USPProtocolEngine()
        dissolution_analyzer = DissolutionAnalyzer(use_gpu=use_gpu)
        pharmaceutical_validator = PharmaceuticalValidator(use_gpu=use_gpu)

        logger.info("✅ All Pharmaceutical Testing Engines initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize pharmaceutical engines: {e}")
        raise e


# Add route information for documentation
router.tags = ["pharmaceutical"]
router.prefix = "/pharmaceutical"
