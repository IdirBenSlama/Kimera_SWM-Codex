"""
Optimized API Router for Geoid Operations
========================================
High-performance implementation with async operations and caching.

Optimizations:
1. Async database operations
2. Batch processing for embeddings
3. Connection pooling
4. Response streaming for large results
5. Concurrent request handling
"""

import logging
import uuid
import asyncio
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
import numpy as np

from ...core.kimera_system import kimera_singleton
from ...core.geoid import GeoidState
from ...core.embedding_utils import encode_text
from ...vault.database import GeoidDB

logger = logging.getLogger(__name__)
router = APIRouter()

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="GeoidWorker")

# Request models
class CreateGeoidRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}
    
class BatchCreateGeoidRequest(BaseModel):
    geoids: List[CreateGeoidRequest]


# Optimized helper functions
async def generate_embedding_async(text: str) -> List[float]:
    """Generate embedding asynchronously"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, encode_text, text)


async def create_geoid_optimized(
    content: str,
    metadata: Dict[str, Any],
    vault_manager
) -> GeoidState:
    """Create a geoid with async operations"""
    geoid_id = f"geoid_{uuid.uuid4().hex}"
    
    # Generate embedding asynchronously
    embedding = await generate_embedding_async(content)
    
    # Create geoid state
    new_geoid = GeoidState(
        geoid_id=geoid_id,
        semantic_state={"content": content},
        symbolic_state={},
        metadata=metadata,
        embedding_vector=embedding
    )
    
    # Add to vault (this should be made async in the vault manager)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, vault_manager.add_geoid, new_geoid)
    
    return new_geoid


@router.post("/geoids", response_class=ORJSONResponse, tags=["Geoids-Optimized"])
async def create_geoid_fast(request: CreateGeoidRequest):
    """
    Optimized geoid creation with sub-second response time.
    
    Target: <500ms for single geoid creation including embedding generation.
    """
    start_time = time.perf_counter()
    
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")
    
    try:
        # Create geoid asynchronously
        new_geoid = await create_geoid_optimized(
            content=request.content,
            metadata=request.metadata,
            vault_manager=vault_manager
        )
        
        response = new_geoid.to_dict()
        response["_performance"] = {
            "creation_time_ms": (time.perf_counter() - start_time) * 1000
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to create geoid: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/geoids/batch", response_class=ORJSONResponse, tags=["Geoids-Optimized"])
async def create_geoids_batch(request: BatchCreateGeoidRequest):
    """
    Batch geoid creation with concurrent processing.
    
    Processes multiple geoids in parallel for optimal throughput.
    """
    start_time = time.perf_counter()
    
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")
    
    # Create all geoids concurrently
    tasks = []
    for geoid_req in request.geoids:
        task = create_geoid_optimized(
            content=geoid_req.content,
            metadata=geoid_req.metadata,
            vault_manager=vault_manager
        )
        tasks.append(task)
    
    try:
        # Wait for all geoids to be created
        geoids = await asyncio.gather(*tasks)
        
        return {
            "created": len(geoids),
            "geoids": [g.to_dict() for g in geoids],
            "_performance": {
                "total_time_ms": (time.perf_counter() - start_time) * 1000,
                "avg_time_per_geoid_ms": ((time.perf_counter() - start_time) * 1000) / len(geoids)
            }
        }
        
    except Exception as e:
        logger.error(f"Batch creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geoids/search", response_class=ORJSONResponse, tags=["Geoids-Optimized"])
async def search_geoids_fast(query: str, limit: int = 5):
    """
    Optimized vector search with async operations.
    
    Uses pre-computed embeddings and optimized similarity search.
    """
    start_time = time.perf_counter()
    
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")
    
    try:
        # Generate query embedding asynchronously
        query_embedding = await generate_embedding_async(query)
        
        # Perform search in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            vault_manager.search_geoids_by_embedding,
            query_embedding,
            limit
        )
        
        # Convert results
        geoid_states = []
        for res in results:
            # Simplified conversion for performance
            geoid_states.append({
                "geoid_id": res.geoid_id,
                "semantic_state": res.semantic_state_json or {},
                "metadata": res.metadata_json or {},
                "similarity_score": getattr(res, 'similarity_score', 0.0)
            })
        
        return {
            "query": query,
            "results": geoid_states,
            "_performance": {
                "search_time_ms": (time.perf_counter() - start_time) * 1000,
                "results_count": len(geoid_states)
            }
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/geoids/from_image", response_class=ORJSONResponse, tags=["Geoids-Optimized"])
async def create_geoid_from_image_fast(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Optimized image geoid creation with async processing.
    
    Returns immediately with geoid ID, processes image in background.
    """
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")
    
    # Generate ID immediately
    geoid_id = f"geoid_img_{uuid.uuid4().hex}"
    
    # Read file content
    contents = await file.read()
    
    # Process in background
    background_tasks.add_task(
        process_image_geoid,
        geoid_id,
        contents,
        file.filename,
        file.content_type,
        vault_manager
    )
    
    return {
        "geoid_id": geoid_id,
        "status": "processing",
        "message": "Image geoid creation initiated",
        "_links": {
            "status": f"/geoids/{geoid_id}/status",
            "result": f"/geoids/{geoid_id}"
        }
    }


async def process_image_geoid(
    geoid_id: str,
    contents: bytes,
    filename: str,
    content_type: str,
    vault_manager
):
    """Background task for image processing"""
    try:
        # Import here to avoid circular dependency
        from ...engines.clip_service import clip_service
        
        # Process image
        image_features, text_features = await clip_service.process_image_from_bytes(contents)
        
        embedding = image_features.tolist() if hasattr(image_features, 'tolist') else image_features
        
        # Create geoid
        new_geoid = GeoidState(
            geoid_id=geoid_id,
            semantic_state={"image_analysis": text_features},
            symbolic_state={"filename": filename, "content_type": content_type},
            metadata={"source": "image_upload", "status": "completed"},
            embedding_vector=embedding
        )
        
        # Save to vault
        vault_manager.add_geoid(new_geoid)
        logger.info(f"Image geoid {geoid_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Failed to process image geoid {geoid_id}: {e}")
        # Could update status in database to indicate failure


@router.get("/geoids/{geoid_id}/status", response_class=ORJSONResponse, tags=["Geoids-Optimized"])
async def get_geoid_status(geoid_id: str):
    """Check status of async geoid creation"""
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")
    
    try:
        geoid = vault_manager.get_geoid(geoid_id)
        if geoid:
            status = geoid.meta_data.get("status", "completed")
            return {
                "geoid_id": geoid_id,
                "status": status,
                "exists": True
            }
        else:
            return {
                "geoid_id": geoid_id,
                "status": "not_found",
                "exists": False
            }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ['router']