# -*- coding: utf-8 -*-
"""
API Router for Contradiction Processing
---------------------------------------
This module contains endpoints for detecting, processing, and resolving
cognitive contradictions within the KIMERA system.
"""

import logging
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from ...core.geoid import GeoidState
from ...core.scar import ScarRecord
from ...engines.contradiction_engine import ContradictionEngine, TensionGradient
from ...vault.database import GeoidDB
from ...vault.vault_manager import VaultManager
from .dependencies import get_contradiction_engine, get_vault_manager

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/contradiction",
    tags=["Contradiction Engine"],
)

# --- Pydantic Models ---


class ProcessContradictionRequest(BaseModel):
    trigger_geoid_id: str
    search_limit: int = 5
    force_collapse: bool = False


class DetectContradictionRequest(BaseModel):
    geoid_pairs: List[List[str]]


# --- Helper Functions ---


def to_state(row: GeoidDB) -> GeoidState:
    """Converts a GeoidDB database object to a GeoidState object."""
    embedding_list = (
        row.semantic_vector.tolist()
        if hasattr(row.semantic_vector, "tolist")
        else (row.semantic_vector or [])
    )
    return GeoidState(
        geoid_id=row.geoid_id,
        semantic_state=row.semantic_state_json or {},
        symbolic_state=row.symbolic_state or {},
        metadata=row.metadata_json or {},
        embedding_vector=embedding_list,
    )


def create_scar_from_tension(
    tension: TensionGradient,
    geoids_dict: Dict[str, GeoidState],
    decision: str = "collapse",
) -> Tuple[ScarRecord, List[float]]:
    """Helper function to create a ScarRecord from a tension gradient."""
    scar_id = f"scar_{uuid.uuid4().hex}"
    embeddings = [
        np.array(g.embedding) for g in geoids_dict.values() if g.embedding is not None
    ]
    if not embeddings:
        avg_embedding = np.zeros(768).tolist()  # Assuming embedding size
    else:
        avg_embedding = np.mean(embeddings, axis=0).tolist()

    geoid_ids = list(geoids_dict.keys())

    scar = ScarRecord(
        scar_id=scar_id,
        geoid_ids=geoid_ids,
        tension_gradient=tension.to_dict(),
        resolution_strategy=decision,
        metadata={"reason": "Contradiction detected"},
        embedding=avg_embedding,
    )
    return scar, avg_embedding


# --- Background Task ---


def run_contradiction_processing_bg(
    body: ProcessContradictionRequest,
    contradiction_engine: ContradictionEngine,
    vault_manager: VaultManager,
):
    """The actual logic for contradiction processing, run in the background."""
    if not contradiction_engine or not vault_manager:
        logger.error(
            "Contradiction Engine or Vault Manager not available for background task."
        )
        return

    logger.info(
        "Starting background contradiction processing for geoid: %s",
        body.trigger_geoid_id,
    )
    try:
        # Fetch the trigger geoid
        trigger_geoid_db = vault_manager.get_geoid(body.trigger_geoid_id)
        if not trigger_geoid_db:
            logger.error(f"Trigger geoid {body.trigger_geoid_id} not found.")
            return

        trigger_geoid_state = to_state(trigger_geoid_db)

        # Find potential contradictions
        potential_matches_df = vault_manager.search_geoids_by_embedding(
            trigger_geoid_state.embedding,
            limit=body.search_limit + 1,
            include_distances=True,
        )

        # Filter out the trigger geoid itself
        potential_matches_df = potential_matches_df[
            potential_matches_df["geoid_id"] != body.trigger_geoid_id
        ]

        if potential_matches_df.empty:
            logger.info(
                f"No potential contradictions found for geoid {body.trigger_geoid_id}"
            )
            return

        # Process each potential contradiction
        for _, row in potential_matches_df.iterrows():
            target_geoid_db = vault_manager.get_geoid(row["geoid_id"])
            if not target_geoid_db:
                continue
            target_geoid_state = to_state(target_geoid_db)

            tension = contradiction_engine.check_contradiction(
                trigger_geoid_state, target_geoid_state
            )

            if contradiction_engine.is_significant(tension):
                logger.info(
                    "Significant tension %.3f found between %s and %s",
                    tension.tension_score,
                    trigger_geoid_state.geoid_id,
                    target_geoid_state.geoid_id,
                )

                decision = "collapse"  # Or some other logic
                scar, _ = create_scar_from_tension(
                    tension,
                    {
                        trigger_geoid_state.geoid_id: trigger_geoid_state,
                        target_geoid_state.geoid_id: target_geoid_state,
                    },
                    decision=decision,
                )

                vault_manager.add_scar(scar)
                logger.info(f"Scar {scar.scar_id} created and logged.")

    except Exception as e:
        logger.error(
            "Error during background contradiction processing: %s",
            e,
            exc_info=True,
        )


# --- API Endpoints ---


@router.post("/process/contradictions", status_code=202, tags=["Contradiction"])
async def process_contradictions(
    body: ProcessContradictionRequest,
    background_tasks: BackgroundTasks,
    contradiction_engine: ContradictionEngine = Depends(get_contradiction_engine),
    vault_manager: VaultManager = Depends(get_vault_manager),
):
    """
    Triggers the asynchronous processing of contradictions for a given Geoid.
    """
    background_tasks.add_task(
        run_contradiction_processing_bg, body, contradiction_engine, vault_manager
    )
    return {"message": "Contradiction processing initiated in the background."}


@router.post("/process/contradictions/sync", tags=["Contradiction"])
async def process_contradictions_sync(
    body: ProcessContradictionRequest,
    contradiction_engine: ContradictionEngine = Depends(get_contradiction_engine),
    vault_manager: VaultManager = Depends(get_vault_manager),
):
    """
    Triggers the synchronous processing of contradictions for a given Geoid.
    Returns the result directly.
    """
    try:
        # Fetch the trigger geoid
        trigger_geoid_db = vault_manager.get_geoid(body.trigger_geoid_id)
        if not trigger_geoid_db:
            raise HTTPException(
            status_code=404,
            detail=f"Trigger geoid {body.trigger_geoid_id} not found.",
        )

        trigger_geoid_state = to_state(trigger_geoid_db)

        # Find potential contradictions
        potential_matches_df = vault_manager.search_geoids_by_embedding(
            trigger_geoid_state.embedding,
            limit=body.search_limit + 1,
            include_distances=True,
        )

        potential_matches_df = potential_matches_df[
            potential_matches_df["geoid_id"] != body.trigger_geoid_id
        ]

        if potential_matches_df.empty:
            return {"message": "No potential contradictions found.", "results": []}

        results = []
        for _, row in potential_matches_df.iterrows():
            target_geoid_db = vault_manager.get_geoid(row["geoid_id"])
            if not target_geoid_db:
                continue
            target_geoid_state = to_state(target_geoid_db)
            tension = contradiction_engine.check_contradiction(
                trigger_geoid_state, target_geoid_state
            )

            if contradiction_engine.is_significant(tension):
                decision = "collapse"
                scar, _ = create_scar_from_tension(
                    tension,
                    {
                        trigger_geoid_state.geoid_id: trigger_geoid_state,
                        target_geoid_state.geoid_id: target_geoid_state,
                    },
                    decision=decision,
                )
                vault_manager.add_scar(scar)
                results.append(
                    {"tension": tension.to_dict(), "scar_created": scar.to_dict()}
                )

        return {
            "message": (
                f"Processed {len(potential_matches_df)} potential contradictions, "
                f"found {len(results)} significant."
            ),
            "results": results,
        }

    except Exception as e:
        logger.error(
            f"Error during synchronous contradiction processing: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect")
async def detect_contradictions(
    request: DetectContradictionRequest,
    contradiction_engine: ContradictionEngine = Depends(get_contradiction_engine),
    vault_manager: VaultManager = Depends(get_vault_manager),
):
    """
    Detect contradictions between pairs of geoids.
    """
    geoid_pairs = request.geoid_pairs

    if not geoid_pairs:
        raise HTTPException(status_code=400, detail="geoid_pairs are required")

    try:
        results = []
        for pair in geoid_pairs:
            if len(pair) != 2:
                results.append(
                    {
                        "pair": pair,
                        "error": "Each pair must contain exactly 2 geoid IDs",
                    }
                )
                continue

            geoid1_id, geoid2_id = pair

            try:
                # Get geoids from vault
                geoid1_db = vault_manager.get_geoid(geoid1_id)
                geoid2_db = vault_manager.get_geoid(geoid2_id)

                if not geoid1_db or not geoid2_db:
                    results.append(
                        {"pair": pair, "error": "One or both geoids not found"}
                    )
                    continue

                # Convert to states
                geoid1_state = to_state(geoid1_db)
                geoid2_state = to_state(geoid2_db)

                # Check contradiction
                tension = contradiction_engine.check_contradiction(
                    geoid1_state, geoid2_state
                )

                results.append(
                    {
                        "pair": pair,
                        "is_contradiction": contradiction_engine.is_significant(
                            tension
                        ),
                        "tension_details": tension.to_dict(),
                    }
                )

            except Exception as e:
                logger.warning(f"Could not process pair {pair}: {e}")
                results.append({"pair": pair, "error": str(e)})

        return {"results": results}

    except Exception as e:
        logger.error(f"Error in detect_contradictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/detect_tension", response_model=Dict[str, Any])
async def detect_tension(
    geoid_pairs: List[List[str]],
    contradiction_engine: ContradictionEngine = Depends(get_contradiction_engine),
    vault_manager: VaultManager = Depends(get_vault_manager),
) -> Dict[str, Any]:
    """
    Detect tension between pairs of geoids.
    """
    try:
        if not contradiction_engine:
            return {
                "status": "error",
                "message": "Contradiction engine not available",
                "tensions": [],
            }

        tensions = []
        for pair in geoid_pairs:
            if len(pair) >= 2:
                geoid1_id, geoid2_id = pair
                geoid1_db = vault_manager.get_geoid(geoid1_id)
                geoid2_db = vault_manager.get_geoid(geoid2_id)
                if geoid1_db and geoid2_db:
                    geoid1_state = to_state(geoid1_db)
                    geoid2_state = to_state(geoid2_db)
                    tension = contradiction_engine.check_contradiction(
                        geoid1_state, geoid2_state
                    )
                    tensions.append(tension.to_dict())

        return {
            "status": "success",
            "tensions": tensions,
            "total_pairs_analyzed": len(geoid_pairs),
        }
    except Exception as e:
        logger.error(f"Error detecting tension: {e}")
        return {"status": "error", "message": str(e), "tensions": []}


@router.post("/contradictions/resolve", response_model=Dict[str, Any])
async def resolve_contradiction(
    contradiction_id: str,
    resolution_strategy: str = "entropy_minimization",
    contradiction_engine: ContradictionEngine = Depends(get_contradiction_engine),
) -> Dict[str, Any]:
    """Resolve a detected contradiction"""
    try:
        if not contradiction_engine:
            raise HTTPException(
                status_code=503, detail="Contradiction engine not available"
            )

        result = await contradiction_engine.resolve_contradiction(
            contradiction_id, strategy=resolution_strategy
        )
        return {
            "status": "success",
            "resolution": result,
            "contradiction_id": contradiction_id,
        }
    except Exception as e:
        logger.error(f"Error resolving contradiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
