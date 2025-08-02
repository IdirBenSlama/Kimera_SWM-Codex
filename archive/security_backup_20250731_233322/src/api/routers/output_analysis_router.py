# -*- coding: utf-8 -*-
"""
API Router for Output Comprehension and Self-Analysis
------------------------------------------------------
This module contains endpoints related to KIMERA's ability to analyze,
comprehend, and assess its own outputs, forming a meta-cognitive loop.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid

from ...core.kimera_output_intelligence import OutputDomain

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models ---

class OutputComprehensionRequest(BaseModel):
    """Request for output comprehension analysis"""
    output_content: str
    domain: str = "cognitive"
    context: Dict[str, Any] = {}
    require_zeteic_validation: bool = True
    gwf_protection_level: str = "standard"

class OutputIntelligenceRequest(BaseModel):
    """Request for output intelligence analysis"""
    output_content: str
    domain: str = "cognitive"
    context: Dict[str, Any] = {}

# --- API Endpoints ---

@router.post("/output/comprehend", tags=["Output Analysis"])
async def comprehend_output(request: OutputComprehensionRequest):
    """Analyzes a given output for comprehension, trust, and implications."""
    from ..main import kimera_system
    comprehension_engine = kimera_system.get('universal_comprehension')
    if not comprehension_engine:
        raise HTTPException(status_code=503, detail="Universal Output Comprehension Engine not available")
    
    try:
        result = await comprehension_engine.comprehend_output(
            output_content=request.output_content,
            context=request.context,
            require_zeteic_validation=request.require_zeteic_validation,
            gwf_protection_level=request.gwf_protection_level
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Failed to comprehend output: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to comprehend output: {str(e)}")

@router.post("/output/intelligence", tags=["Output Analysis"])
async def analyze_output_intelligence(request: OutputIntelligenceRequest):
    """Performs an intelligence analysis on a given output."""
    from ..main import kimera_system
    intelligence_system = kimera_system.get('output_intelligence')
    if not intelligence_system:
        raise HTTPException(status_code=503, detail="KIMERA Output Intelligence System not available")
        
    try:
        result = await intelligence_system.analyze_output(
            output_content=request.output_content,
            domain=OutputDomain(request.domain),
            context=request.context
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Failed to analyze output intelligence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze output intelligence: {str(e)}")

@router.get("/output/comprehension/history", tags=["Output Analysis"])
async def get_comprehension_history(limit: int = 10):
    """Gets the history of KIMERA's output comprehension reports."""
    from ..main import kimera_system
    comprehension_engine = kimera_system.get('universal_comprehension')
    if not comprehension_engine:
        raise HTTPException(status_code=503, detail="Universal Output Comprehension Engine not available")
    
    report = comprehension_engine.get_comprehension_report()
    return {
        "status": "success",
        "comprehension_history": report[:limit],
        "limit_applied": limit
    }

@router.post("/output/self_analysis", tags=["Output Analysis"])
async def kimera_self_analysis(content: str = "KIMERA system operational status"):
    """Enables KIMERA to perform a self-analysis of its own operational outputs."""
    from ..main import kimera_system
    comprehension_engine = kimera_system.get('universal_comprehension')
    intelligence_system = kimera_system.get('output_intelligence')
    
    if not comprehension_engine or not intelligence_system:
        raise HTTPException(status_code=503, detail="Self-analysis systems not available")
        
    logger.info("ðŸ¤– KIMERA performing self-analysis...")
    
    comprehension = await comprehension_engine.comprehend_output(
        output_content=content,
        context={"analysis_type": "self_reflection", "system": "KIMERA"}
    )
    
    intelligence = await intelligence_system.analyze_output(
        output_content=content,
        domain=OutputDomain.COGNITIVE,
        context={"analysis_type": "self_reflection", "system": "KIMERA"}
    )
    
    self_analysis = {
        "self_analysis_id": f"SELF_{uuid.uuid4().hex[:8]}",
        "analysis_content": content,
        "comprehension_analysis": comprehension.to_dict(),
        "intelligence_analysis": intelligence.to_dict(),
        "timestamp": datetime.now().isoformat()
    }
    
    return self_analysis

@router.post("/output/analyze", tags=["Output Analysis"])
async def analyze_output(request: dict):
    """
    General output analysis endpoint.
    """
    content = request.get("content", "")
    context = request.get("context", {})
    
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")
    
    try:
        # Perform basic output analysis
        analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
        
        # Analyze various aspects of the output
        word_count = len(content.split())
        char_count = len(content)
        
        # Simple sentiment analysis (mock)
        sentiment_score = 0.5 + (hash(content) % 100 - 50) / 100
        sentiment = "positive" if sentiment_score > 0.6 else "negative" if sentiment_score < 0.4 else "neutral"
        
        # Complexity analysis
        avg_word_length = sum(len(word) for word in content.split()) / max(word_count, 1)
        complexity = "high" if avg_word_length > 6 else "medium" if avg_word_length > 4 else "low"
        
        # Topic extraction (mock)
        topics = []
        if "AI" in content or "artificial" in content.lower():
            topics.append("artificial_intelligence")
        if "data" in content.lower():
            topics.append("data_analysis")
        if "system" in content.lower():
            topics.append("systems")
        
        analysis_result = {
            "analysis_id": analysis_id,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "metrics": {
                "word_count": word_count,
                "character_count": char_count,
                "average_word_length": round(avg_word_length, 2)
            },
            "sentiment": {
                "score": round(sentiment_score, 3),
                "label": sentiment
            },
            "complexity": complexity,
            "topics": topics,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Output analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") 