# -*- coding: utf-8 -*-
"""
API Router for Geoid and Scar Operations
-----------------------------------------
This module contains all API endpoints related to the creation, retrieval,
and management of Geoids (Geometrical-Semantic Objects) and Scars
(Semantic Contradiction and Resolution records).
"""

import logging
import uuid
from typing import Dict, Any

import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Request
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ...core.kimera_system import kimera_singleton
from ...core.geoid import GeoidState
from ...core.scar import ScarRecord
from ...core.embedding_utils import encode_text, extract_semantic_features
from ...vault.database import GeoidDB
from ...linguistic.echoform import parse_echoform
from ...engines.clip_service import clip_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Custom validation exception handler function (not used on router level)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with detailed error messages."""
    logger.error(f"Validation error in /geoids: {exc}")
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": errors,
            "detail": "Request validation failed. Check the 'errors' field for details."
        }
    )

# --- Pydantic Models for Request/Response ---

class CreateGeoidRequest(BaseModel):
    semantic_features: Dict[str, float] | None = None
    symbolic_content: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    echoform_text: str | None = None

    class Config:
        # Allow extra fields to be ignored
        extra = "ignore"


# --- Helper Functions ---

def to_state(row: GeoidDB) -> GeoidState:
    """Converts a GeoidDB database object to a GeoidState object."""
    # Convert numpy array to Python list for compatibility
    embedding_list = row.semantic_vector.tolist() if hasattr(row.semantic_vector, 'tolist') else (row.semantic_vector or [])
    return GeoidState(
        geoid_id=row.geoid_id,
        semantic_state=row.semantic_state_json or {},
        symbolic_state=row.symbolic_state or {},
        metadata=row.metadata_json or {},
        embedding_vector=embedding_list
    )

# --- API Endpoints ---

@router.post("/geoids", tags=["Geoids"])
async def create_geoid(request: CreateGeoidRequest):
    """
    Creates a new Geoid in the system.
    
    A Geoid can be created from semantic features, symbolic content, or
    by parsing an Echoform text. If Echoform text is provided, it will be
    parsed to extract semantic features.
    """
    logger.info(f"/geoids called with request: {request}")
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")

    geoid_id = f"geoid_{uuid.uuid4().hex}"
    
    semantic_features = request.semantic_features or {}
    
    if request.echoform_text:
        try:
            parsed_features = parse_echoform(request.echoform_text)
            semantic_features.update(parsed_features)
        except Exception as e:
            logger.warning(f"Failed to parse echoform text: {e}")

    if not semantic_features and not request.symbolic_content:
        raise HTTPException(status_code=400, detail="Either semantic_features or symbolic_content must be provided")

    # Generate embedding from semantic features if they exist
    embedding = None
    if semantic_features:
        try:
            # Create a text representation for embedding
            text_to_embed = " ".join([f"{k}:{v}" for k,v in semantic_features.items()])
            embedding_raw = encode_text(text_to_embed)

            # Convert to CPU numpy array safely first
            if hasattr(embedding_raw, 'cpu'):  # PyTorch tensor on GPU
                embedding_array = embedding_raw.cpu().numpy()
            elif hasattr(embedding_raw, 'detach'):  # PyTorch tensor on CPU
                embedding_array = embedding_raw.detach().numpy()
            elif hasattr(embedding_raw, 'numpy'):  # PyTorch tensor on CPU (alternative)
                embedding_array = embedding_raw.numpy()
            elif isinstance(embedding_raw, np.ndarray):  # Already numpy array
                embedding_array = embedding_raw
            elif isinstance(embedding_raw, list):  # List of floats
                embedding_array = np.array(embedding_raw)
            else:
                # Last resort conversion
                embedding_array = np.array(embedding_raw)
            
            # Convert to Python list for GeoidState
            embedding = embedding_array.tolist()
            
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
    
    # If no embedding was generated, create a default zero vector with correct dimensions
    if embedding is None:
        from ...core.constants import EMBEDDING_DIM
        embedding = [0.0] * EMBEDDING_DIM

    new_geoid = GeoidState(
        geoid_id=geoid_id,
        semantic_state=semantic_features,
        symbolic_state=request.symbolic_content,
        metadata=request.metadata,
        embedding_vector=embedding
    )

    try:
        vault_manager.add_geoid(new_geoid)
        logger.info(f"Geoid '{geoid_id}' created and added to vault.")
    except Exception as e:
        logger.error(f"Failed to add geoid to vault: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save geoid: {str(e)}")

    return new_geoid.to_dict()


@router.post("/geoids/from_image", response_model=Dict[str, Any], tags=["Geoids"])
async def create_geoid_from_image(file: UploadFile = File(...)):
    """
    Creates a new Geoid from an image file.
    
    The image is processed to extract semantic features using a vision model (CLIP),
    and a corresponding Geoid is created.
    """
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")

    geoid_id = f"geoid_img_{uuid.uuid4().hex}"
    
    try:
        contents = await file.read()
        image_features, text_features = await clip_service.process_image_from_bytes(contents)
        
        # We use image features for embedding and text features for semantic representation
        embedding = image_features.tolist() if hasattr(image_features, 'tolist') else image_features
        
        # Ensure embedding has correct dimensions
        if not embedding:
            from ...core.constants import EMBEDDING_DIM
            embedding = [0.0] * EMBEDDING_DIM

        # Create a GeoidState object
        new_geoid = GeoidState(
            geoid_id=geoid_id,
            semantic_state={"image_analysis": text_features},
            symbolic_state={"filename": file.filename, "content_type": file.content_type},
            metadata={"source": "image_upload"},
            embedding_vector=embedding
        )
        
        # Add geoid to the vault
        vault_manager.add_geoid(new_geoid)
        logger.info(f"Geoid from image '{geoid_id}' created successfully.")
        
        return new_geoid.to_dict()

    except Exception as e:
        logger.error(f"Error creating geoid from image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geoids/search", tags=["Geoids"])
async def search_geoids(query: str, limit: int = 5):
    """
    Searches for Geoids based on a text query.
    
    The query is converted to an embedding and a vector search is performed
    to find the most similar Geoids in the vault.
    """
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")
        
    try:
        query_embedding_raw = encode_text(query)
        
        # Convert to CPU numpy array safely first
        import numpy as np
        if hasattr(query_embedding_raw, 'cpu'):  # PyTorch tensor on GPU
            query_embedding = query_embedding_raw.cpu().numpy()
        elif hasattr(query_embedding_raw, 'detach'):  # PyTorch tensor on CPU
            query_embedding = query_embedding_raw.detach().numpy()
        elif hasattr(query_embedding_raw, 'numpy'):  # PyTorch tensor on CPU (alternative)
            query_embedding = query_embedding_raw.numpy()
        elif isinstance(query_embedding_raw, np.ndarray):  # Already numpy array
            query_embedding = query_embedding_raw
        elif isinstance(query_embedding_raw, list):  # List of floats
            query_embedding = np.array(query_embedding_raw)
        else:
            # Last resort conversion
            query_embedding = np.array(query_embedding_raw)
        
        results = vault_manager.search_geoids_by_embedding(query_embedding, limit=limit)
        
        # Convert DB objects to state objects
        geoid_states = [to_state(res) for res in results]
        
        return {"query": query, "results": geoid_states}
    except Exception as e:
        logger.error(f"Error searching geoids: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scars/search", tags=["Scars"])
async def search_scars(query: str, limit: int = 3):
    """
    Searches for Scars based on a text query.
    
    The query is converted to an embedding and a vector search is performed
    to find the most similar Scars in the vault.
    """
    vault_manager = kimera_singleton.get_vault_manager()
    if not vault_manager:
        raise HTTPException(status_code=503, detail="Vault Manager not available")

    try:
        query_embedding_raw = encode_text(query)
        
        # Convert to CPU numpy array safely first
        import numpy as np
        if hasattr(query_embedding_raw, 'cpu'):  # PyTorch tensor on GPU
            query_embedding = query_embedding_raw.cpu().numpy()
        elif hasattr(query_embedding_raw, 'detach'):  # PyTorch tensor on CPU
            query_embedding = query_embedding_raw.detach().numpy()
        elif hasattr(query_embedding_raw, 'numpy'):  # PyTorch tensor on CPU (alternative)
            query_embedding = query_embedding_raw.numpy()
        elif isinstance(query_embedding_raw, np.ndarray):  # Already numpy array
            query_embedding = query_embedding_raw
        elif isinstance(query_embedding_raw, list):  # List of floats
            query_embedding = np.array(query_embedding_raw)
        else:
            # Last resort conversion
            query_embedding = np.array(query_embedding_raw)
        
        search_result = vault_manager.search_scars_by_embedding(query_embedding, top_k=limit)
        
        # Extract results from the search response (already a dict)
        if isinstance(search_result, dict) and "results" in search_result:
            scars = search_result["results"]
            status = search_result.get("status", "unknown")
        else:
            # Fallback if format is unexpected
            scars = []
            status = "error"
        
        return {
            "query": query, 
            "results": scars,
            "status": status,
            "count": len(scars)
        }
    except Exception as e:
        logger.error(f"Error searching scars: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to search scars: {str(e)}") 