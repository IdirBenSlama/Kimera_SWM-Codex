"""
CRUD operations for interacting with the database models.
"""

from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from ..core.insight import InsightScar
from . import database as models


def get_insight(db: Session, insight_id: str):
    return (
        db.query(models.InsightDB)
        .filter(models.InsightDB.insight_id == insight_id)
        .first()
    )


def get_insights_by_type(
    db: Session, insight_type: str, skip: int = 0, limit: int = 100
):
    return (
        db.query(models.InsightDB)
        .filter(models.InsightDB.insight_type == insight_type)
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_insight(db: Session, insight: InsightScar) -> models.InsightDB:
    """
    Creates a new InsightDB record from an InsightScar object.
    """
    db_insight = models.InsightDB(
        insight_id=insight.insight_id,
        insight_type=insight.insight_type,
        source_resonance_id=insight.source_resonance_id,
        echoform_repr=insight.echoform_repr,
        application_domains=insight.application_domains,
        confidence=insight.confidence,
        entropy_reduction=insight.entropy_reduction,
        utility_score=insight.utility_score,
        status=insight.status,
        created_at=insight.created_at,
        last_reinforced_cycle=str(insight.last_reinforced_cycle),  # Store as string
    )
    db.add(db_insight)
    db.commit()
    db.refresh(db_insight)
    return db_insight


def update_insight_status(
    db: Session, insight_id: str, status: str, utility_score: float
):
    """
    Updates the status and utility score of an existing insight.
    """
    db_insight = get_insight(db, insight_id=insight_id)
    if db_insight:
        db_insight.status = status
        db_insight.utility_score = utility_score
        db.commit()
        db.refresh(db_insight)
    return db_insight


def get_geoid_by_id(db: Session, geoid_id: str) -> Optional[models.GeoidDB]:
    """
    Retrieves a single Geoid by its ID.
    """
    if not hasattr(models, "GeoidDB"):
        return None
    return db.query(models.GeoidDB).filter(models.GeoidDB.geoid_id == geoid_id).first()


def get_geoid_stability(db: Session, geoid_id: str) -> float:
    """
    Retrieves or calculates the stability of a Geoid.
    NOTE: This is a placeholder for a more complex calculation.
    """
    geoid = get_geoid_by_id(db, geoid_id)
    if geoid and hasattr(geoid, "stability_score"):
        return geoid.stability_score
    return 0.5  # Return a default neutral stability if not found


def get_geoid_connectivity(db: Session, geoid_id: str) -> int:
    """
    Retrieves or calculates the connectivity (degree) of a Geoid in the cognitive graph.
    NOTE: This is a placeholder. A real implementation would query a graph database
    or analyze relationship tables.
    """
    # This is a conceptual placeholder. A true implementation would be complex.
    # For now, we simulate a basic return value.
    return hash(geoid_id) % 100  # Pseudo-random connectivity for demonstration


def get_scar_count_for_geoid(db: Session, geoid_id: str) -> int:
    """
    Counts the number of InsightScars a Geoid has been involved in.
    This serves as a measure of how much a concept has been tested and refined.
    """
    # This query assumes a relationship or stored data linking scars to geoids,
    # which would need to be architected. For now, we query based on a
    # hypothetical text search in the echoform representation.
    return (
        db.query(models.InsightDB)
        .filter(
            models.InsightDB.insight_type == "SCAR",
            models.InsightDB.echoform_repr.contains(geoid_id),
        )
        .count()
    )


def get_geoids_with_embeddings(
    limit: int = 100,
) -> List[Tuple[str, Optional[List[float]]]]:
    """
    Retrieve geoids with their embeddings from the database.

    Args:
        limit: Maximum number of geoids to retrieve

    Returns:
        List of tuples containing (geoid_id, embedding) where embedding is a list of floats or None
    """
    from .database import SessionLocal

    with SessionLocal() as db:
        # Check if GeoidDB exists in the models
        if hasattr(models, "GeoidDB"):
            geoids_db = db.query(models.GeoidDB).limit(limit).all()

            result = []
            for geoid_db in geoids_db:
                # Check for semantic_vector attribute
                if (
                    hasattr(geoid_db, "semantic_vector")
                    and geoid_db.semantic_vector is not None
                ):
                    result.append((geoid_db.geoid_id, geoid_db.semantic_vector))
                else:
                    result.append((geoid_db.geoid_id, None))

            return result
        else:
            # Return empty list if GeoidDB doesn't exist
            return []
