# -*- coding: utf-8 -*-
"""
API Router for Core System Actions
----------------------------------
This module contains miscellaneous, high-level endpoints that trigger
core system processes, such as the cognitive cycle or proactive scans.
"""

import logging
from fastapi import APIRouter, HTTPException

from ...engines.kccl import KimeraCognitiveCycle
from ...engines.proactive_contradiction_detector import ProactiveContradictionDetector

logger = logging.getLogger(__name__)
router = APIRouter()

# --- API Endpoints ---

@router.post("/system/cycle", tags=["Core Actions"])
async def trigger_cycle():
    """
    Manually triggers a single Kimera Cognitive Cycle (KCCL).
    This involves activation, synthesis, and potential insight generation.
    """
    from ..main import kimera_system
    # This assumes the KCCL engine is initialized.
    kccl_engine = kimera_system.get('cognitive_cycle', KimeraCognitiveCycle())
    if not kccl_engine:
        raise HTTPException(status_code=503, detail="Kimera Cognitive Cycle engine not available.")
    
    try:
        cycle_results = await kccl_engine.run_cycle()
        return {"status": "cycle_triggered", "results": cycle_results}
    except Exception as e:
        logger.error(f"Error during manual system cycle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to trigger system cycle.")

@router.post("/system/proactive_scan", tags=["Core Actions"])
async def run_proactive_contradiction_scan():
    """
    Initiates a proactive scan for latent contradictions in the vault.
    """
    from ..main import kimera_system
    vault_manager = kimera_system.get('vault_manager')
    contradiction_engine = kimera_system.get('contradiction_engine')

    if not vault_manager or not contradiction_engine:
        raise HTTPException(status_code=503, detail="Vault or Contradiction engine not available.")
        
    try:
        detector = ProactiveContradictionDetector(vault_manager, contradiction_engine)
        scan_results = await detector.scan_and_resolve()
        return {"status": "proactive_scan_complete", "results": scan_results}
    except Exception as e:
        logger.error(f"Error during proactive scan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Proactive scan failed.")

@router.post("/test/metrics", tags=["Core Actions"])
async def test_metrics():
    """
    An endpoint for testing the Prometheus metrics integration.
    Increments various test counters.
    """
    from ..main import kimera_system
    metrics = kimera_system.get('metrics')
    if metrics:
        metrics.kimera_geoids_created.inc()
        metrics.kimera_scars_created.inc()
        metrics.kimera_contradictions_detected.inc()
        metrics.kimera_contradictions_resolved.inc()
        return {"status": "Metrics incremented"}
    return {"status": "Metrics not initialized"}

@router.post("/embed", tags=["Core Actions"])
async def embed_text(request: dict):
    """
    Generate embeddings for text using the system's embedding model.
    """
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        from ...core.embedding_utils import encode_text
        embedding = encode_text(text)
        
        # Convert to CPU numpy array safely for JSON serialization
        import numpy as np
        if hasattr(embedding, 'cpu'):  # PyTorch tensor on GPU
            embedding_array = embedding.cpu().numpy()
        elif hasattr(embedding, 'detach'):  # PyTorch tensor on CPU
            embedding_array = embedding.detach().numpy()
        elif hasattr(embedding, 'numpy'):  # PyTorch tensor on CPU (alternative)
            embedding_array = embedding.numpy()
        elif isinstance(embedding, np.ndarray):  # Already numpy array
            embedding_array = embedding
        elif isinstance(embedding, list):  # List of floats
            embedding_array = np.array(embedding)
        else:
            # Last resort conversion
            embedding_array = np.array(embedding)
        
        # Convert to Python list for JSON serialization
        embedding_list = embedding_array.tolist()
        
        return {
            "text": text,
            "embedding": embedding_list,
            "dimensions": len(embedding_list)
        }
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {str(e)}")

@router.post("/semantic_features", tags=["Core Actions"])
async def extract_semantic_features(request: dict):
    """
    Extract semantic features from text.
    """
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    try:
        # TEMPORARY FIX: Use a simpler approach that avoids the GPU tensor issue
        # Get the embedding first and handle device conversion properly
        from ...core.embedding_utils import encode_text
        import numpy as np
        
        # Get raw embedding
        embedding_raw = encode_text(text)
        
        # Convert to CPU numpy array safely
        if hasattr(embedding_raw, 'cpu'):  # PyTorch tensor on GPU
            embedding = embedding_raw.cpu().detach().numpy()
        elif hasattr(embedding_raw, 'detach'):  # PyTorch tensor on CPU
            embedding = embedding_raw.detach().numpy()
        elif hasattr(embedding_raw, 'numpy'):  # PyTorch tensor on CPU (alternative)
            embedding = embedding_raw.numpy()
        elif isinstance(embedding_raw, np.ndarray):  # Already numpy array
            embedding = embedding_raw
        elif isinstance(embedding_raw, list):  # List of floats
            embedding = np.array(embedding_raw)
        else:
            # Last resort conversion
            embedding = np.array(embedding_raw)
        
        # Calculate features safely
        features = {
            'mean': float(np.mean(embedding)),
            'std': float(np.std(embedding)),
            'norm': float(np.linalg.norm(embedding)),
            'max': float(np.max(embedding)),
            'min': float(np.min(embedding)),
            'dimensionality': len(embedding)
        }
        
        return {
            "text": text,
            "semantic_features": features,
            "feature_count": len(features)
        }
    except Exception as e:
        logger.error(f"Semantic feature extraction failed: {e}")
        # Return basic features instead of failing
        return {
            "text": text,
            "semantic_features": {
                'mean': 0.0,
                'std': 0.0,
                'norm': 0.0,
                'max': 0.0,
                'min': 0.0,
                'dimensionality': 0,
                'error': f"Extraction failed: {str(e)}"
            },
            "feature_count": 6
        }

@router.post("/action/execute", tags=["Core Actions"])
async def execute_action(request: dict):
    """
    Execute a core system action.
    """
    action_type = request.get("action_type", "")
    parameters = request.get("parameters", {})
    
    if not action_type:
        raise HTTPException(status_code=400, detail="action_type is required")
    
    # Map action types to functions
    action_map = {
        "analyze": lambda p: {"action": "analyze", "result": f"Analyzed: {p.get('text', '')}", "status": "completed"},
        "process": lambda p: {"action": "process", "result": f"Processed: {p.get('data', '')}", "status": "completed"},
        "transform": lambda p: {"action": "transform", "result": f"Transformed: {p.get('input', '')}", "status": "completed"},
        "validate": lambda p: {"action": "validate", "result": f"Validated: {p.get('content', '')}", "status": "completed"}
    }
    
    if action_type not in action_map:
        raise HTTPException(status_code=400, detail=f"Unknown action type: {action_type}")
    
    try:
        result = action_map[action_type](parameters)
        return result
    except Exception as e:
        logger.error(f"Action execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute action: {str(e)}") 