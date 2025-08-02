#!/usr/bin/env python3
"""
Kimera SWM Cognitive Services API
================================

Production-ready FastAPI service that exposes the Kimera SWM Cognitive Architecture
as enterprise-grade RESTful web services.

This API provides:
- Cognitive processing endpoints
- Real-time consciousness analysis
- Understanding and insight generation
- Learning and pattern recognition
- Multi-language processing
- System monitoring and health checks

Author: Kimera SWM Development Team
Date: January 30, 2025
Version: 5.0.0
"""

import asyncio
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Kimera SWM Core
from ..core.master_cognitive_architecture_v2 import (
    MasterCognitiveArchitecture,
    CognitiveRequest,
    CognitiveResponse,
    CognitiveWorkflow,
    ProcessingMode,
    ArchitectureState
)

# Configuration and logging
from ..config.config_loader import ConfigManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global architecture instance
cognitive_architecture: Optional[MasterCognitiveArchitecture] = None

# API Models
class CognitiveProcessingRequest(BaseModel):
    """Request model for cognitive processing"""
    input_data: Union[str, Dict[str, Any], List[Any]] = Field(
        ..., 
        description="Input data for cognitive processing",
        example="Analyze this complex philosophical statement about consciousness."
    )
    workflow_type: str = Field(
        default="basic_cognition",
        description="Type of cognitive workflow to execute",
        example="deep_understanding"
    )
    processing_mode: str = Field(
        default="adaptive",
        description="Processing mode (sequential, parallel, adaptive)",
        example="adaptive"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for processing",
        example={"priority": "high", "domain": "philosophy"}
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Processing priority (1-10 scale)",
        example=7
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        le=300,
        description="Processing timeout in seconds",
        example=60.0
    )

    @validator('workflow_type')
    def validate_workflow_type(cls, v):
        valid_workflows = [
            "basic_cognition", "deep_understanding", "creative_insight",
            "learning_integration", "consciousness_analysis", "linguistic_processing"
        ]
        if v not in valid_workflows:
            raise ValueError(f"Invalid workflow type. Must be one of: {valid_workflows}")
        return v

    @validator('processing_mode')
    def validate_processing_mode(cls, v):
        valid_modes = ["sequential", "parallel", "adaptive", "distributed"]
        if v not in valid_modes:
            raise ValueError(f"Invalid processing mode. Must be one of: {valid_modes}")
        return v


class CognitiveProcessingResponse(BaseModel):
    """Response model for cognitive processing"""
    request_id: str = Field(..., description="Unique request identifier")
    success: bool = Field(..., description="Whether processing was successful")
    workflow_type: str = Field(..., description="Executed workflow type")
    processing_time: float = Field(..., description="Processing time in seconds")
    quality_score: float = Field(..., description="Quality score (0-1)")
    confidence: float = Field(..., description="Confidence level (0-1)")
    
    # Results
    understanding: Optional[Dict[str, Any]] = Field(None, description="Understanding analysis results")
    consciousness: Optional[Dict[str, Any]] = Field(None, description="Consciousness analysis results")
    insights: List[Dict[str, Any]] = Field(default_factory=list, description="Generated insights")
    patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Discovered patterns")
    results: Dict[str, Any] = Field(default_factory=dict, description="Additional results")
    
    # Metadata
    components_used: List[str] = Field(default_factory=list, description="Components involved in processing")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    error_log: List[str] = Field(default_factory=list, description="Error messages if any")
    timestamp: str = Field(..., description="Response timestamp")


class SystemHealthResponse(BaseModel):
    """Response model for system health"""
    system_id: str
    state: str
    uptime: float
    device: str
    processing_mode: str
    components: Dict[str, Any]
    performance: Dict[str, Any]
    health: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class UnderstandingRequest(BaseModel):
    """Request model for understanding analysis"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze for understanding",
        example="What is the nature of consciousness in artificial intelligence?"
    )
    understanding_type: str = Field(
        default="semantic",
        description="Type of understanding analysis",
        example="conceptual"
    )
    depth: str = Field(
        default="standard",
        description="Analysis depth (basic, standard, deep)",
        example="deep"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for understanding",
        example={"domain": "philosophy", "complexity": "high"}
    )


class ConsciousnessRequest(BaseModel):
    """Request model for consciousness analysis"""
    cognitive_state: Optional[List[float]] = Field(
        None,
        description="Cognitive state vector for analysis",
        example=[0.1, 0.5, -0.2, 0.8]
    )
    text_input: Optional[str] = Field(
        None,
        description="Text input to analyze for consciousness markers",
        example="I am aware that I am thinking about my own thoughts."
    )
    analysis_mode: str = Field(
        default="unified",
        description="Analysis mode (unified, thermodynamic, quantum, iit)",
        example="unified"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for consciousness analysis"
    )


# Authentication (placeholder for production)
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authentication dependency (placeholder)"""
    if credentials and credentials.credentials:
        # In production, validate JWT token here
        return {"user_id": "authenticated_user", "roles": ["user"]}
    return {"user_id": "anonymous", "roles": ["guest"]}


# Startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global cognitive_architecture
    
    # Startup
    logger.info("🚀 Starting Kimera SWM Cognitive Services API...")
    
    try:
        # Initialize cognitive architecture
        cognitive_architecture = MasterCognitiveArchitecture(
            device="auto",
            enable_gpu=True,
            processing_mode=ProcessingMode.ADAPTIVE,
            max_concurrent_operations=50
        )
        
        # Initialize the architecture
        success = await cognitive_architecture.initialize_architecture()
        if not success:
            logger.error("❌ Failed to initialize cognitive architecture")
            raise RuntimeError("Architecture initialization failed")
        
        logger.info("✅ Kimera SWM Cognitive Services API ready")
        logger.info(f"   System ID: {cognitive_architecture.system_id}")
        logger.info(f"   Device: {cognitive_architecture.device}")
        logger.info(f"   Components: {len(cognitive_architecture.component_registry)}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Kimera SWM Cognitive Services API...")
    if cognitive_architecture:
        await cognitive_architecture.shutdown()
    logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Kimera SWM Cognitive Services API",
    description="Enterprise-grade cognitive processing services powered by Kimera SWM Architecture",
    version="5.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# Health and status endpoints
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "Kimera SWM Cognitive Services",
        "version": "5.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/status", response_model=SystemHealthResponse)
async def system_status(user: dict = Depends(get_current_user)):
    """Comprehensive system status"""
    if not cognitive_architecture:
        raise HTTPException(status_code=503, detail="Cognitive architecture not initialized")
    
    try:
        status = cognitive_architecture.get_system_status()
        return SystemHealthResponse(**status)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")


# Core cognitive processing endpoints
@app.post("/cognitive/process", response_model=CognitiveProcessingResponse)
async def process_cognitive_request(
    request: CognitiveProcessingRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Main cognitive processing endpoint"""
    if not cognitive_architecture:
        raise HTTPException(status_code=503, detail="Cognitive architecture not initialized")
    
    if cognitive_architecture.state != ArchitectureState.READY:
        raise HTTPException(status_code=503, detail=f"System not ready: {cognitive_architecture.state.value}")
    
    try:
        # Convert request to internal format
        workflow_map = {
            "basic_cognition": CognitiveWorkflow.BASIC_COGNITION,
            "deep_understanding": CognitiveWorkflow.DEEP_UNDERSTANDING,
            "creative_insight": CognitiveWorkflow.CREATIVE_INSIGHT,
            "learning_integration": CognitiveWorkflow.LEARNING_INTEGRATION,
            "consciousness_analysis": CognitiveWorkflow.CONSCIOUSNESS_ANALYSIS,
            "linguistic_processing": CognitiveWorkflow.LINGUISTIC_PROCESSING
        }
        
        mode_map = {
            "sequential": ProcessingMode.SEQUENTIAL,
            "parallel": ProcessingMode.PARALLEL,
            "adaptive": ProcessingMode.ADAPTIVE,
            "distributed": ProcessingMode.DISTRIBUTED
        }
        
        # Create internal request
        internal_request = CognitiveRequest(
            request_id=f"api_{uuid.uuid4().hex[:8]}",
            workflow_type=workflow_map[request.workflow_type],
            input_data=request.input_data,
            processing_mode=mode_map[request.processing_mode],
            context={**request.context, "user_id": user["user_id"], "api_request": True},
            priority=request.priority,
            timeout=request.timeout
        )
        
        # Process the request
        response = await cognitive_architecture.process_cognitive_request(internal_request)
        
        # Convert response to API format
        api_response = CognitiveProcessingResponse(
            request_id=response.request_id,
            success=response.success,
            workflow_type=request.workflow_type,
            processing_time=response.processing_time,
            quality_score=response.quality_score,
            confidence=response.confidence,
            understanding=response.understanding,
            consciousness=response.consciousness,
            insights=response.insights,
            patterns=response.patterns,
            results=response.results,
            components_used=response.components_used,
            performance_metrics=response.performance_metrics,
            error_log=response.error_log,
            timestamp=response.timestamp
        )
        
        # Log successful processing
        logger.info(f"Processed request {internal_request.request_id[:8]} in {response.processing_time:.3f}s")
        
        return api_response
        
    except Exception as e:
        logger.error(f"Cognitive processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/cognitive/understand")
async def analyze_understanding(
    request: UnderstandingRequest,
    user: dict = Depends(get_current_user)
):
    """Specialized understanding analysis endpoint"""
    if not cognitive_architecture:
        raise HTTPException(status_code=503, detail="Cognitive architecture not initialized")
    
    try:
        # Create understanding-focused request
        internal_request = CognitiveRequest(
            request_id=f"understand_{uuid.uuid4().hex[:8]}",
            workflow_type=CognitiveWorkflow.DEEP_UNDERSTANDING,
            input_data=request.text,
            context={
                **request.context,
                "understanding_type": request.understanding_type,
                "depth": request.depth,
                "user_id": user["user_id"]
            }
        )
        
        response = await cognitive_architecture.process_cognitive_request(internal_request)
        
        return {
            "request_id": response.request_id,
            "success": response.success,
            "understanding": response.understanding,
            "quality_score": response.quality_score,
            "processing_time": response.processing_time,
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Understanding analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Understanding analysis failed: {str(e)}")


@app.post("/cognitive/consciousness")
async def analyze_consciousness(
    request: ConsciousnessRequest,
    user: dict = Depends(get_current_user)
):
    """Specialized consciousness analysis endpoint"""
    if not cognitive_architecture:
        raise HTTPException(status_code=503, detail="Cognitive architecture not initialized")
    
    try:
        # Prepare input data
        input_data = request.text_input or request.cognitive_state
        if not input_data:
            raise HTTPException(status_code=400, detail="Either text_input or cognitive_state must be provided")
        
        # Create consciousness-focused request
        internal_request = CognitiveRequest(
            request_id=f"consciousness_{uuid.uuid4().hex[:8]}",
            workflow_type=CognitiveWorkflow.CONSCIOUSNESS_ANALYSIS,
            input_data=input_data,
            context={
                **request.context,
                "analysis_mode": request.analysis_mode,
                "user_id": user["user_id"]
            }
        )
        
        response = await cognitive_architecture.process_cognitive_request(internal_request)
        
        return {
            "request_id": response.request_id,
            "success": response.success,
            "consciousness": response.consciousness,
            "quality_score": response.quality_score,
            "processing_time": response.processing_time,
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        logger.error(f"Consciousness analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consciousness analysis failed: {str(e)}")


# Batch processing endpoint
@app.post("/cognitive/batch")
async def batch_process(
    requests: List[CognitiveProcessingRequest],
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Batch cognitive processing endpoint"""
    if not cognitive_architecture:
        raise HTTPException(status_code=503, detail="Cognitive architecture not initialized")
    
    if len(requests) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size limited to 10 requests")
    
    try:
        # Process all requests concurrently
        tasks = []
        for req in requests:
            workflow_map = {
                "basic_cognition": CognitiveWorkflow.BASIC_COGNITION,
                "deep_understanding": CognitiveWorkflow.DEEP_UNDERSTANDING,
                "creative_insight": CognitiveWorkflow.CREATIVE_INSIGHT,
                "learning_integration": CognitiveWorkflow.LEARNING_INTEGRATION,
                "consciousness_analysis": CognitiveWorkflow.CONSCIOUSNESS_ANALYSIS,
                "linguistic_processing": CognitiveWorkflow.LINGUISTIC_PROCESSING
            }
            
            internal_request = CognitiveRequest(
                request_id=f"batch_{uuid.uuid4().hex[:8]}",
                workflow_type=workflow_map[req.workflow_type],
                input_data=req.input_data,
                context={**req.context, "user_id": user["user_id"], "batch": True}
            )
            tasks.append(cognitive_architecture.process_cognitive_request(internal_request))
        
        responses = await asyncio.gather(*tasks)
        
        return {
            "batch_id": f"batch_{uuid.uuid4().hex[:8]}",
            "total_requests": len(requests),
            "successful_requests": sum(1 for r in responses if r.success),
            "responses": [
                {
                    "request_id": r.request_id,
                    "success": r.success,
                    "quality_score": r.quality_score,
                    "processing_time": r.processing_time,
                    "results": {
                        "understanding": r.understanding,
                        "consciousness": r.consciousness,
                        "insights": r.insights
                    }
                }
                for r in responses
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


# Metrics and monitoring
@app.get("/metrics")
async def get_metrics(user: dict = Depends(get_current_user)):
    """Get system metrics (Prometheus format)"""
    if not cognitive_architecture:
        raise HTTPException(status_code=503, detail="Cognitive architecture not initialized")
    
    try:
        status = cognitive_architecture.get_system_status()
        
        # Simple Prometheus-style metrics
        metrics = [
            f"kimera_total_operations {status['performance']['total_operations']}",
            f"kimera_success_rate {status['performance']['success_rate']}",
            f"kimera_avg_processing_time {status['performance']['average_processing_time']}",
            f"kimera_active_requests {status['performance']['active_requests']}",
            f"kimera_uptime {status['uptime']}",
            f"kimera_components_total {status['components']['total']}"
        ]
        
        return Response(
            content="\n".join(metrics),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect metrics")


# Development and testing endpoints
@app.get("/dev/info")
async def development_info():
    """Development information endpoint"""
    return {
        "service": "Kimera SWM Cognitive Services API",
        "version": "5.0.0",
        "environment": "development",
        "features": {
            "cognitive_processing": True,
            "understanding_analysis": True,
            "consciousness_analysis": True,
            "batch_processing": True,
            "real_time_monitoring": True,
            "gpu_acceleration": True
        },
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "process": "/cognitive/process",
            "understand": "/cognitive/understand", 
            "consciousness": "/cognitive/consciousness",
            "batch": "/cognitive/batch",
            "metrics": "/metrics"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "cognitive_services_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )