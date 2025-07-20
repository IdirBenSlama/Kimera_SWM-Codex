# -*- coding: utf-8 -*-
"""
API Router for Statistical Analysis and Monitoring
--------------------------------------------------
This module contains all endpoints related to statistical analysis of
system data, including entropy, contradictions, semantic markets,
and time-series forecasting.
"""

import logging
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...core.statistical_modeling import (
    statistical_engine,
    StatisticalModelResult,
    analyze_entropy_time_series,
    analyze_contradiction_factors,
    analyze_semantic_market
)
from ...monitoring.forecasting_and_control_monitor import (
    initialize_forecasting_and_control_monitoring
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize the monitor to get a singleton instance for the router to use
forecasting_and_control_monitor = initialize_forecasting_and_control_monitoring()

# --- Pydantic Models ---

class StatisticalAnalysisRequest(BaseModel):
    entropy_history: List[float] = []
    timestamps: List[str] = []
    contradiction_scores: List[float] = []
    semantic_features: Dict[str, List[float]] = {}
    semantic_supply: List[float] = []
    semantic_demand: List[float] = []
    entropy_prices: List[float] = []

# --- API Endpoints ---

@router.get("/statistics/capabilities", tags=["Statistics"])
async def get_statistical_capabilities():
    """Returns a list of available statistical models and their descriptions."""
    return statistical_engine.get_model_capabilities()

@router.post("/statistics/analyze/entropy_series", response_model=StatisticalModelResult, tags=["Statistics"])
async def analyze_entropy_series_endpoint(
    entropy_data: List[float],
    timestamps: List[str] = None
):
    """Analyzes a time series of entropy values."""
    if not entropy_data:
        raise HTTPException(status_code=400, detail="entropy_data cannot be empty")
    return analyze_entropy_time_series(entropy_data, timestamps)

@router.post("/statistics/analyze/contradiction_factors", response_model=StatisticalModelResult, tags=["Statistics"])
async def analyze_contradiction_factors_endpoint(
    contradiction_scores: List[float],
    semantic_features: Dict[str, List[float]]
):
    """Analyzes factors contributing to contradictions."""
    if not contradiction_scores or not semantic_features:
        raise HTTPException(status_code=400, detail="Contradiction scores and semantic features must be provided")
    return analyze_contradiction_factors(contradiction_scores, semantic_features)

@router.post("/statistics/analyze/semantic_market", response_model=StatisticalModelResult, tags=["Statistics"])
async def analyze_semantic_market_endpoint(
    semantic_supply: List[float],
    semantic_demand: List[float],
    entropy_prices: List[float]
):
    """Analyzes the dynamics of a semantic market."""
    return analyze_semantic_market(semantic_supply, semantic_demand, entropy_prices)

@router.post("/statistics/analyze/comprehensive", tags=["Statistics"])
async def comprehensive_statistical_analysis(request: StatisticalAnalysisRequest):
    """Performs a comprehensive statistical analysis based on the provided data."""
    results = {}
    if request.entropy_history:
        results['entropy_analysis'] = analyze_entropy_time_series(request.entropy_history, request.timestamps)
    if request.contradiction_scores and request.semantic_features:
        results['contradiction_analysis'] = analyze_contradiction_factors(request.contradiction_scores, request.semantic_features)
    if request.semantic_supply and request.semantic_demand and request.entropy_prices:
        results['semantic_market_analysis'] = analyze_semantic_market(request.semantic_supply, request.semantic_demand, request.entropy_prices)
    return results

@router.get("/statistics/monitoring/status", tags=["Statistics"])
async def get_statistical_monitoring_status():
    """Gets the status of the advanced statistical monitor."""
    return forecasting_and_control_monitor.get_status()

@router.post("/statistics/monitoring/start", tags=["Statistics"])
async def start_statistical_monitoring():
    """Starts the advanced statistical monitoring background tasks."""
    forecasting_and_control_monitor.start_monitoring()
    return {"status": "Advanced statistical monitoring started."}

@router.post("/statistics/monitoring/stop", tags=["Statistics"])
async def stop_statistical_monitoring():
    """Stops the advanced statistical monitoring background tasks."""
    forecasting_and_control_monitor.stop_monitoring()
    return {"status": "Advanced statistical monitoring stopped."}

@router.get("/statistics/monitoring/alerts", tags=["Statistics"])
async def get_statistical_alerts(severity: str = None, hours: int = 24):
    """Retrieves recent statistical alerts."""
    return forecasting_and_control_monitor.get_alerts(severity_filter=severity, hours=hours)

@router.get("/statistics/monitoring/forecast/{metric_name}", tags=["Statistics"])
async def get_metric_forecast(metric_name: str):
    """Retrieves a forecast for a specific monitored metric."""
    forecast = forecasting_and_control_monitor.get_forecast(metric_name)
    if forecast is None:
        raise HTTPException(status_code=404, detail=f"No forecast available for metric: {metric_name}")
    return forecast

@router.get("/statistics/system/entropy_analysis", tags=["Statistics"])
async def get_system_entropy_analysis():
    """Performs an entropy analysis on the entire vault."""
    from ..main import kimera_system
    vault_manager = kimera_system.get('vault_manager')
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available.")
    
    all_geoids = vault_manager.get_all_geoids_for_analysis()
    entropy_values = [g.get('entropy', 0) for g in all_geoids]
    if not entropy_values:
        return {"mean_entropy": 0, "std_dev_entropy": 0, "count": 0}
        
    return statistical_engine.run_model('descriptive_stats', data={'values': entropy_values})

@router.post("/statistics/analyze", tags=["Statistics"])
async def analyze_data(request: dict):
    """
    General statistical analysis endpoint.
    """
    data = request.get("data", [])
    analysis_type = request.get("analysis_type", "basic")
    
    if not data:
        raise HTTPException(status_code=400, detail="Data is required")
    
    try:
        if analysis_type == "basic":
            # Basic descriptive statistics
            import numpy as np
            data_array = np.array(data)
            return {
                "analysis_type": "basic",
                "mean": float(np.mean(data_array)),
                "median": float(np.median(data_array)),
                "std": float(np.std(data_array)),
                "min": float(np.min(data_array)),
                "max": float(np.max(data_array)),
                "count": len(data)
            }
        elif analysis_type == "time_series":
            # Time series analysis
            return analyze_entropy_time_series(data, None).dict()
        elif analysis_type == "distribution":
            # Distribution analysis
            import numpy as np
            data_array = np.array(data)
            hist, bins = np.histogram(data_array, bins=10)
            return {
                "analysis_type": "distribution",
                "histogram": hist.tolist(),
                "bins": bins.tolist(),
                "skewness": float(np.mean(((data_array - np.mean(data_array)) / np.std(data_array)) ** 3)),
                "kurtosis": float(np.mean(((data_array - np.mean(data_array)) / np.std(data_array)) ** 4) - 3)
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {analysis_type}")
            
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") 