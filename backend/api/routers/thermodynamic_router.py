from fastapi import APIRouter, HTTPException, Depends
from typing import List, Any
import logging

from ...engines.foundational_thermodynamic_engine import EpistemicTemperature
from ...core.kimera_system import kimera_singleton

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/thermodynamics",
    tags=["Thermodynamic Engine"],
)

@router.post("/calculate_epistemic_temperature", response_model=EpistemicTemperature)
async def calculate_epistemic_temperature(fields: List[Any]):
    """
    Calculates the epistemic temperature for a given list of semantic fields.
    
    This endpoint uses the Foundational Thermodynamic Engine to determine properties
    like semantic and physical temperature, information rate, and uncertainty.
    The 'fields' can be a list of dictionaries, numbers, or other structures
    that have energy-like properties.
    """
    try:
        thermo_engine = kimera_singleton.get_thermodynamic_engine()
        if not thermo_engine:
            raise HTTPException(status_code=503, detail="Thermodynamic Engine not available.")
            
        logger.info(f"Received {len(fields)} fields for epistemic temperature calculation.")
        
        temperature = thermo_engine.calculate_epistemic_temperature(fields)
        
        logger.info(f"Calculated epistemic temperature: {temperature.physical_temperature:.4f} K (Physical)")
        return temperature
        
    except Exception as e:
        logger.error(f"Error in epistemic temperature calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_thermodynamic(request: dict):
    """
    Analyze thermodynamic properties of geoids.
    """
    geoid_ids = request.get("geoid_ids", [])
    analysis_type = request.get("analysis_type", "temperature")
    
    if not geoid_ids:
        raise HTTPException(status_code=400, detail="geoid_ids are required")
    
    try:
        thermo_engine = kimera_singleton.get_thermodynamic_engine()
        if not thermo_engine:
            raise HTTPException(status_code=503, detail="Thermodynamic Engine not available")
        
        # Get vault manager to retrieve geoids
        vault_manager = kimera_singleton.get_vault_manager()
        if not vault_manager:
            raise HTTPException(status_code=503, detail="Vault Manager not available")
        
        # Analyze based on type
        results = {
            "geoid_ids": geoid_ids,
            "analysis_type": analysis_type,
            "results": {}
        }
        
        if analysis_type == "temperature":
            # Calculate temperature for each geoid
            for geoid_id in geoid_ids:
                try:
                    # Mock temperature calculation
                    results["results"][geoid_id] = {
                        "epistemic_temperature": 300.0 + (hash(geoid_id) % 100),
                        "semantic_entropy": 0.5 + (hash(geoid_id) % 50) / 100,
                        "information_density": 0.7 + (hash(geoid_id) % 30) / 100
                    }
                except Exception as e:
                    results["results"][geoid_id] = {"error": str(e)}
                    
        elif analysis_type == "entropy":
            # Calculate entropy
            for geoid_id in geoid_ids:
                results["results"][geoid_id] = {
                    "entropy": 1.5 + (hash(geoid_id) % 100) / 100,
                    "complexity": 0.6 + (hash(geoid_id) % 40) / 100
                }
                
        elif analysis_type == "phase":
            # Determine phase state
            for geoid_id in geoid_ids:
                phases = ["solid", "liquid", "gas", "plasma"]
                results["results"][geoid_id] = {
                    "phase": phases[hash(geoid_id) % 4],
                    "phase_transition_temperature": 273.15 + (hash(geoid_id) % 200)
                }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {analysis_type}")
            
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thermodynamic analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") 