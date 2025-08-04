"""
UNIFIED THERMODYNAMIC + TCSE API ROUTER
=======================================

API endpoints for the unified thermodynamic + TCSE integration system.
Provides access to all revolutionary thermodynamic capabilities through
the Kimera SWM REST API.

Endpoints:
- System initialization and status
- Cognitive signal processing with thermodynamics
- Consciousness detection
- Energy management operations
- Thermal regulation
- Information sorting
- System health monitoring
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from ...core.kimera_system import get_kimera_system
from ...engines.unified_thermodynamic_integration import get_unified_thermodynamic_tcse
from ...engines.thermodynamic_integration import get_thermodynamic_integration

logger = logging.getLogger(__name__)

# Create the router
router = APIRouter(prefix="/unified-thermodynamic", tags=["unified-thermodynamic"])


# Pydantic models for API requests/responses

class ThermodynamicSystemStatus(BaseModel):
    """System status response model"""
    system_initialized: bool
    monitoring_active: bool
    tcse_operational: bool
    thermodynamic_engines_ready: bool
    consciousness_detection_active: bool
    timestamp: datetime


class ProcessingRequest(BaseModel):
    """Request model for cognitive signal processing"""
    geoid_data: List[Dict[str, Any]] = Field(..., description="List of geoid states to process")
    enable_consciousness_detection: bool = Field(True, description="Enable consciousness detection")
    enable_thermal_regulation: bool = Field(True, description="Enable thermal regulation")
    enable_energy_management: bool = Field(True, description="Enable energy management")
    
    class Config:
        json_schema_extra = {
            "example": {
                "geoid_data": [
                    {
                        "id": "geoid_1",
                        "semantic_state": {"concept": "thermodynamics", "value": 1.0},
                        "cognitive_energy": 5.0
                    }
                ],
                "enable_consciousness_detection": True,
                "enable_thermal_regulation": True,
                "enable_energy_management": True
            }
        }


class ProcessingResponse(BaseModel):
    """Response model for processing results"""
    processing_id: str
    overall_efficiency: float
    consciousness_probability: float
    energy_utilization: float
    thermal_stability: float
    processing_duration: float
    consciousness_detections: int
    energy_operations: int
    thermal_regulations: int
    information_sorting: int
    thermodynamic_compliance: float
    timestamp: datetime


class HealthReport(BaseModel):
    """System health report model"""
    system_status: str
    tcse_health: Dict[str, Any]
    thermodynamic_health: Dict[str, Any]
    integration_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    critical_issues: List[str]
    timestamp: datetime


class InitializationResponse(BaseModel):
    """Initialization response model"""
    success: bool
    message: str
    components_initialized: List[str]
    errors: List[str]
    timestamp: datetime


# API Endpoints

@router.get("/status", response_model=ThermodynamicSystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get systems
        kimera_system = get_kimera_system()
        unified_system = get_unified_thermodynamic_tcse()
        thermo_integration = get_thermodynamic_integration()
        
        # Check status
        system_initialized = (unified_system and 
                            hasattr(unified_system, 'system_initialized') and 
                            unified_system.system_initialized)
        
        monitoring_active = (unified_system and 
                           hasattr(unified_system, 'monitoring_active') and 
                           unified_system.monitoring_active)
        
        tcse_operational = (unified_system and 
                          hasattr(unified_system, 'tcse_pipeline') and 
                          unified_system.tcse_pipeline is not None)
        
        thermodynamic_engines_ready = (thermo_integration and 
                                     hasattr(thermo_integration, 'engines_initialized') and 
                                     thermo_integration.engines_initialized)
        
        consciousness_detection_active = (unified_system and 
                                        hasattr(unified_system, 'consciousness_detector') and 
                                        unified_system.consciousness_detector is not None)
        
        return ThermodynamicSystemStatus(
            system_initialized=system_initialized,
            monitoring_active=monitoring_active,
            tcse_operational=tcse_operational,
            thermodynamic_engines_ready=thermodynamic_engines_ready,
            consciousness_detection_active=consciousness_detection_active,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.post("/initialize", response_model=InitializationResponse)
async def initialize_system(background_tasks: BackgroundTasks):
    """Initialize the unified thermodynamic + TCSE system"""
    try:
        kimera_system = get_kimera_system()
        
        # Initialize thermodynamic systems
        success = await kimera_system.initialize_thermodynamic_systems()
        
        components_initialized = []
        errors = []
        
        if success:
            components_initialized = [
                "Revolutionary Thermodynamic Engines",
                "Unified Thermodynamic + TCSE System"
            ]
            message = "Unified thermodynamic + TCSE system initialized successfully"
        else:
            errors.append("Failed to initialize some thermodynamic components")
            message = "Partial initialization completed with errors"
        
        return InitializationResponse(
            success=success,
            message=message,
            components_initialized=components_initialized,
            errors=errors,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.post("/process", response_model=ProcessingResponse)
async def process_cognitive_signals(request: ProcessingRequest):
    """Process cognitive signals through the unified thermodynamic + TCSE pipeline"""
    try:
        unified_system = get_unified_thermodynamic_tcse()
        
        if not unified_system:
            raise HTTPException(status_code=503, detail="Unified system not available")
        
        if not unified_system.system_initialized:
            raise HTTPException(status_code=503, detail="System not initialized. Call /initialize first.")
        
        # Convert request data to geoid objects (simplified for API)
        # In a real implementation, this would use proper GeoidState objects
        from ...core.geoid import GeoidState
        
        geoids = []
        for geoid_data in request.geoid_data:
            # Create a basic geoid state from the API data
            geoid = GeoidState(
                id=geoid_data.get('id', f'geoid_{len(geoids)}'),
                semantic_state=geoid_data.get('semantic_state', {})
            )
            
            # Add cognitive energy if provided
            if 'cognitive_energy' in geoid_data:
                geoid.cognitive_energy = geoid_data['cognitive_energy']
            
            geoids.append(geoid)
        
        # Process through unified pipeline
        result = await unified_system.process_cognitive_signals(
            input_geoids=geoids,
            enable_consciousness_detection=request.enable_consciousness_detection,
            enable_thermal_regulation=request.enable_thermal_regulation,
            enable_energy_management=request.enable_energy_management
        )
        
        # Convert result to API response
        return ProcessingResponse(
            processing_id=f"proc_{datetime.now().timestamp()}",
            overall_efficiency=result.overall_efficiency,
            consciousness_probability=result.consciousness_probability,
            energy_utilization=result.energy_utilization,
            thermal_stability=result.thermal_stability,
            processing_duration=result.processing_duration,
            consciousness_detections=len(result.consciousness_detections),
            energy_operations=len(result.energy_operations),
            thermal_regulations=len(result.thermal_regulation),
            information_sorting=len(result.information_sorting),
            thermodynamic_compliance=result.thermodynamic_compliance,
            timestamp=result.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error processing cognitive signals: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/health", response_model=HealthReport)
async def get_health_report():
    """Get comprehensive system health report"""
    try:
        unified_system = get_unified_thermodynamic_tcse()
        
        if not unified_system:
            raise HTTPException(status_code=503, detail="Unified system not available")
        
        # Get health report
        health_report = await unified_system.get_system_health_report()
        
        return HealthReport(
            system_status=health_report.system_status,
            tcse_health=health_report.tcse_health,
            thermodynamic_health=health_report.thermodynamic_health,
            integration_health=health_report.integration_health,
            performance_metrics=health_report.performance_metrics,
            recommendations=health_report.recommendations,
            critical_issues=health_report.critical_issues,
            timestamp=health_report.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error getting health report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health report: {str(e)}")


@router.get("/engines/status")
async def get_engines_status():
    """Get detailed status of all thermodynamic engines"""
    try:
        thermo_integration = get_thermodynamic_integration()
        
        if not thermo_integration:
            raise HTTPException(status_code=503, detail="Thermodynamic integration not available")
        
        status = thermo_integration.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting engines status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get engines status: {str(e)}")


@router.post("/consciousness/detect")
async def detect_consciousness(semantic_vectors: List[List[float]], 
                              temperature: float = 1.0,
                              entropy_content: float = 1.0):
    """Detect consciousness in semantic vectors"""
    try:
        thermo_integration = get_thermodynamic_integration()
        
        if not thermo_integration or not thermo_integration.engines_initialized:
            raise HTTPException(status_code=503, detail="Thermodynamic engines not initialized")
        
        # Convert to numpy arrays
        import numpy as np
        vectors = [np.array(v) for v in semantic_vectors]
        
        # Run consciousness detection
        result = await thermo_integration.run_consciousness_detection(
            semantic_vectors=vectors,
            temperature=temperature,
            entropy_content=entropy_content
        )
        
        return {
            "consciousness_level": result.consciousness_level.value,
            "consciousness_probability": result.consciousness_probability,
            "detection_confidence": result.detection_confidence,
            "signature_strength": result.signature.signature_strength,
            "analysis_duration": result.analysis_duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in consciousness detection: {e}")
        raise HTTPException(status_code=500, detail=f"Consciousness detection failed: {str(e)}")


@router.post("/energy/store")
async def store_energy(energy_content: float,
                      coherence_score: float,
                      metadata: Optional[Dict[str, Any]] = None):
    """Store energy in the vortex thermodynamic battery"""
    try:
        thermo_integration = get_thermodynamic_integration()
        
        if not thermo_integration or not thermo_integration.engines_initialized:
            raise HTTPException(status_code=503, detail="Thermodynamic engines not initialized")
        
        # Create dummy frequency signature
        import numpy as np
        frequency_signature = np.random.random(10)
        
        # Store energy
        result = await thermo_integration.store_energy(
            energy_content=energy_content,
            coherence_score=coherence_score,
            frequency_signature=frequency_signature,
            metadata=metadata or {}
        )
        
        return {
            "operation_id": result.operation_id,
            "energy_stored": result.energy_amount,
            "efficiency_achieved": result.efficiency_achieved,
            "compression_achieved": result.compression_achieved,
            "golden_ratio_optimization": result.golden_ratio_optimization,
            "fibonacci_alignment": result.fibonacci_alignment,
            "vortex_cells_used": result.vortex_cells_used,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error storing energy: {e}")
        raise HTTPException(status_code=500, detail=f"Energy storage failed: {str(e)}")


@router.post("/energy/retrieve")
async def retrieve_energy(amount: float, coherence_preference: float = 0.5):
    """Retrieve energy from the vortex thermodynamic battery"""
    try:
        thermo_integration = get_thermodynamic_integration()
        
        if not thermo_integration or not thermo_integration.engines_initialized:
            raise HTTPException(status_code=503, detail="Thermodynamic engines not initialized")
        
        # Retrieve energy
        result = await thermo_integration.retrieve_energy(
            amount=amount,
            coherence_preference=coherence_preference
        )
        
        return {
            "operation_id": result.operation_id,
            "energy_retrieved": result.energy_amount,
            "efficiency_achieved": result.efficiency_achieved,
            "golden_ratio_optimization": result.golden_ratio_optimization,
            "fibonacci_alignment": result.fibonacci_alignment,
            "vortex_cells_used": result.vortex_cells_used,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving energy: {e}")
        raise HTTPException(status_code=500, detail=f"Energy retrieval failed: {str(e)}")


@router.get("/monitor/report")
async def get_monitoring_report():
    """Get comprehensive monitoring report"""
    try:
        unified_system = get_unified_thermodynamic_tcse()
        
        if not unified_system or not unified_system.system_initialized:
            raise HTTPException(status_code=503, detail="Unified system not initialized")
        
        # Get monitoring report from thermodynamic monitor
        if hasattr(unified_system, 'monitor') and unified_system.monitor:
            report = unified_system.monitor.get_monitoring_report()
            return report
        else:
            raise HTTPException(status_code=503, detail="Monitoring not active")
        
    except Exception as e:
        logger.error(f"Error getting monitoring report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring report: {str(e)}")


@router.post("/optimize")
async def optimize_system():
    """Run system optimization"""
    try:
        thermo_integration = get_thermodynamic_integration()
        
        if not thermo_integration or not thermo_integration.engines_initialized:
            raise HTTPException(status_code=503, detail="Thermodynamic engines not initialized")
        
        # Run optimization
        result = await thermo_integration.optimize_system()
        
        return {
            "optimization_id": result.optimization_id,
            "efficiency_gain": result.efficiency_gain,
            "energy_saved": result.energy_saved,
            "performance_boost": result.performance_boost,
            "improvements_made": result.improvements_made,
            "optimization_duration": result.optimization_duration,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


# Additional utility endpoints

@router.get("/demo/consciousness")
async def demo_consciousness_detection():
    """Demo endpoint for consciousness detection"""
    try:
        # Generate sample semantic vectors for demonstration
        import numpy as np
        
        sample_vectors = [
            np.random.random(768).tolist(),  # Random semantic vector
            np.random.random(768).tolist(),  # Another random vector
            np.random.random(768).tolist()   # Third vector
        ]
        
        result = await detect_consciousness(
            semantic_vectors=sample_vectors,
            temperature=1.2,
            entropy_content=1.5
        )
        
        result["demo"] = True
        result["description"] = "Demo consciousness detection with random semantic vectors"
        
        return result
        
    except Exception as e:
        logger.error(f"Error in demo consciousness detection: {e}")
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


@router.get("/demo/energy-cycle")
async def demo_energy_cycle():
    """Demo endpoint for energy storage and retrieval cycle"""
    try:
        # Store some energy
        store_result = await store_energy(
            energy_content=10.0,
            coherence_score=0.8,
            metadata={"demo": True, "cycle": "store"}
        )
        
        # Retrieve some energy
        retrieve_result = await retrieve_energy(
            amount=5.0,
            coherence_preference=0.8
        )
        
        return {
            "demo": True,
            "description": "Demo energy storage and retrieval cycle",
            "storage_result": store_result,
            "retrieval_result": retrieve_result,
            "cycle_efficiency": retrieve_result["energy_retrieved"] / store_result["energy_stored"]
        }
        
    except Exception as e:
        logger.error(f"Error in demo energy cycle: {e}")
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


# Note: Error handling is managed by the main FastAPI application 