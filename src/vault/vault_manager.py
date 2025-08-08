from __future__ import annotations

import json
import logging
import math
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from ..core.geoid import GeoidState
from ..core.scar import ScarRecord
from ..utils.kimera_exceptions import (KimeraDatabaseError, KimeraValidationError,
                                       handle_exception)
from ..utils.kimera_logger import get_database_logger
from .database import (GeoidDB, ScarDB, SessionLocal, create_tables, engine,
                       get_db_status, initialize_database)
from .database_connection_manager import (get_connection_manager,
                                          initialize_database_connection)

# Initialize structured logger
logger = get_database_logger(__name__)

# Try to import Neo4j components, but don't fail if not available
try:
    from neo4j.exceptions import Neo4jError

    from ..graph.models import create_geoid, create_scar

    NEO4J_AVAILABLE = True
except ImportError:
    logger.warning(
        "Neo4j modules not available. Graph database functionality will be disabled."
    )
    NEO4J_AVAILABLE = False

    # Create dummy functions to avoid errors
    def create_scar(data):
        pass

    def create_geoid(data):
        pass

    class Neo4jError(Exception):
        pass
class VaultManager:
    """Auto-generated class."""
    pass
    """
    Manages the persistence of core system objects, including Geoids and Scars.

    This class abstracts all database interactions for reading and writing these
    critical data structures. It also contains the logic for maintaining
    balance between the dual scar vaults.
    """

    def __init__(self) -> None:
        """Initialize the vault manager."""
        logger.info("VaultManager initializing...")
        self.db_initialized = False
        self.neo4j_available = NEO4J_AVAILABLE
        self.retry_count = 0
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

        # Initialize database using connection manager
        try:
            connection_success = initialize_database_connection()

            if connection_success:
                # Initialize database extensions and tables
                db_init_success = initialize_database()

                if db_init_success:
                    self.db_initialized = True
                    logger.info("Database initialization completed successfully")

                    # Check database status
                    connection_manager = get_connection_manager()
                    db_status = connection_manager.get_status()
                    if db_status["status"] == "connected":
                        logger.info(
                            f"Connected to {db_status['type']} {db_status['version']}"
                        )
                    else:
                        logger.warning(
                            f"Database connection issues: {db_status['error']}"
                        )
                else:
                    logger.warning(
                        "Database initialization incomplete - operating with limited persistence"
                    )
                    self.db_initialized = False
            else:
                logger.warning(
                    "Database connection failed - operating without persistence"
                )
                self.db_initialized = False

        except Exception as e:
            logger.warning(
                f"Database initialization failed: {e} - operating without persistence"
            )
            self.db_initialized = False

        # Log Neo4j availability
        if self.neo4j_available:
            logger.info("Neo4j integration available")
        else:
            logger.warning(
                "Neo4j integration not available - graph database features disabled"
            )

        logger.info("VaultManager initialized")

    def _execute_with_retry(self, operation_func, *args, **kwargs):
        """Execute a database operation with retry logic"""
        retry_count = 0
        last_error = None

        while retry_count < self.max_retries:
            try:
                return operation_func(*args, **kwargs)
            except SQLAlchemyError as e:
                retry_count += 1
                last_error = e
                logger.warning(
                    f"Database operation failed (attempt {retry_count}/{self.max_retries}): {e}"
                )

                if retry_count < self.max_retries:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)

        # If we get here, all retries failed
        logger.error(
            f"Database operation failed after {self.max_retries} attempts: {last_error}"
        )
        raise KimeraDatabaseError(f"Database operation failed: {last_error}")

    def _select_vault(self) -> str:
        """Choose a vault based on current counts."""
        try:
            a_count = self.get_total_scar_count("vault_a")
            b_count = self.get_total_scar_count("vault_b")
            return "vault_a" if a_count <= b_count else "vault_b"
        except Exception as e:
            logger.warning(f"Error selecting vault: {e}. Defaulting to vault_a")
            return "vault_a"

    def get_all_geoids(self) -> List[GeoidState]:
        """
        Fetches all geoids from the database and reconstitutes them into GeoidState objects.

        This method queries the GeoidDB table and reconstructs the full, in-memory
        representation of each geoid, including its semantic and symbolic states.

        :raises KimeraDatabaseError: If there is an issue with the database query or connection.
        :return: A list of all GeoidState objects currently in the system.
        """
        if not self.db_initialized:
            logger.warning("Database not initialized. Returning empty geoid list.")
            return []

        operation_id = f"get_all_geoids_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting operation {operation_id}: get_all_geoids")

        try:
            with SessionLocal() as db:
                geoid_db_records = db.query(GeoidDB).all()

                geoids = []
                for g in geoid_db_records:
                    try:
                        geoid = GeoidState(
                            geoid_id=g.geoid_id,
                            semantic_state=g.semantic_state_json,
                            symbolic_state=g.symbolic_state,
                            embedding_vector=g.semantic_vector,
                            metadata=g.metadata_json,
                        )
                        geoids.append(geoid)
                    except (ValueError, TypeError) as e:
                        logger.error(
                            f"Failed to reconstruct geoid {g.geoid_id} due to data format error: {e}",
                            geoid_id=g.geoid_id,
                            error=e,
                        )
                        continue
                    except Exception as e:
                        logger.error(
                            f"Unexpected error reconstructing geoid {g.geoid_id}: {e}",
                            geoid_id=g.geoid_id,
                            error=e,
                        )
                        continue

                # --- dual-write to Neo4j (idempotent) ---
                if self.neo4j_available:
                    neo4j_errors = 0
                    for geo in geoids:
                        try:
                            create_geoid(geo.to_dict())
                        except Neo4jError as e:
                            neo4j_errors += 1
                            logger.debug(
                                f"Neo4j geoid creation failed for {geo.geoid_id}: {e}",
                                geoid_id=geo.geoid_id,
                                error=e,
                            )
                        except Exception as e:
                            neo4j_errors += 1
                            logger.warning(
                                f"Unexpected error in Neo4j geoid creation for {geo.geoid_id}: {e}",
                                geoid_id=geo.geoid_id,
                                error=e,
                            )

                    if neo4j_errors > 0:
                        logger.warning(
                            f"Some Neo4j operations failed: {neo4j_errors} out of {len(geoids)} geoids"
                        )

                logger.info(
                    f"Operation {operation_id} completed successfully: retrieved {len(geoids)} geoids"
                )
                return geoids

        except SQLAlchemyError as e:
            logger.error(f"Database error in operation {operation_id}: {e}", error=e)
            # Return empty list instead of raising an exception
            logger.warning("Returning empty geoid list due to database error")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in operation {operation_id}: {e}", error=e)
            # Return empty list instead of raising an exception
            logger.warning("Returning empty geoid list due to unexpected error")
            return []

    def insert_scar(
        self, scar: ScarRecord, vector: List[float], db: Session = None
    ) -> Optional[ScarDB]:
        """
        Inserts a new SCAR into the database and vector index,
        using an existing session if provided.

        Returns the created ScarDB object if successful, None otherwise.
        """
        if not self.db_initialized:
            logger.warning("Database not initialized. Skipping scar insertion.")
            return None

        try:
            if db:
                return self._insert_scar_data(db, scar, vector)
            else:
                with SessionLocal() as session:
                    return self._insert_scar_data(session, scar, vector)
        except Exception as e:
            logger.error(f"Failed to insert scar: {e}")
            return None

    def _insert_scar_data(
        self, db: Session, scar: ScarRecord, vector: List[float]
    ) -> Optional[ScarDB]:
        """Internal method to insert scar data with error handling"""
        try:
            if isinstance(scar, GeoidState):
                scar = ScarRecord(
                    scar_id=f"SCAR_{uuid.uuid4().hex[:8]}",
                    geoids=[scar.geoid_id],
                    reason="auto-generated",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    resolved_by="vault_manager",
                    pre_entropy=0.0,
                    post_entropy=0.0,
                    delta_entropy=0.0,
                    cls_angle=0.0,
                    semantic_polarity=0.0,
                    mutation_frequency=0.0,
                    weight=1.0,
                )

            vault_id = self._select_vault()
            scar_db = ScarDB(
                scar_id=scar.scar_id,
                geoids=scar.geoids,
                reason=scar.reason,
                timestamp=datetime.now(timezone.utc),
                resolved_by=scar.resolved_by,
                pre_entropy=scar.pre_entropy,
                post_entropy=scar.post_entropy,
                delta_entropy=scar.delta_entropy,
                cls_angle=scar.cls_angle,
                semantic_polarity=scar.semantic_polarity,
                mutation_frequency=scar.mutation_frequency,
                weight=getattr(scar, "weight", 1.0),
                scar_vector=vector,
                vault_id=vault_id,
            )

            db.add(scar_db)
            db.commit()
            db.refresh(scar_db)

            # --- async write to Neo4j (fire-and-forget) ---
            if self.neo4j_available:
                try:
                    create_scar(
                        {
                            **scar.__dict__,
                            "vault_id": vault_id,
                            "scar_vector": vector,
                        }
                    )
                except Neo4jError as e:
                    logger.warning(
                        f"Neo4j scar creation failed for {scar.scar_id}: {e}",
                        scar_id=scar.scar_id,
                        vault_id=vault_id,
                    )
                except Exception as e:
                    logger.warning(
                        f"Unexpected error in Neo4j scar creation for {scar.scar_id}: {e}",
                        scar_id=scar.scar_id,
                        vault_id=vault_id,
                        error=e,
                    )

            return scar_db

        except SQLAlchemyError as e:
            logger.error(f"Database error inserting scar: {e}")
            db.rollback()
            return None
        except Exception as e:
            logger.error(f"Unexpected error inserting scar: {e}")
            db.rollback()
            return None

    def get_scars_from_vault(self, vault_id: str, limit: int = 100) -> List[ScarDB]:
        """Return recent scars from the requested vault."""
        if not self.db_initialized:
            logger.warning("Database not initialized. Returning empty scar list.")
            return []

        try:
            with SessionLocal() as db:
                scars = (
                    db.query(ScarDB)
                    .filter(ScarDB.vault_id == vault_id)
                    .order_by(ScarDB.timestamp.desc())
                    .limit(limit)
                    .all()
                )
                now = datetime.now(timezone.utc)
                for s in scars:
                    s.last_accessed = now
                db.commit()
                return scars
        except SQLAlchemyError as e:
            logger.error(f"Database error getting scars from vault {vault_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting scars from vault {vault_id}: {e}")
            return []

    def get_total_scar_count(self, vault_id: str) -> int:
        """Return the total number of scars stored in the given vault."""
        if not self.db_initialized:
            logger.warning("Database not initialized. Returning zero scar count.")
            return 0

        try:
            with SessionLocal() as db:
                return db.query(ScarDB).filter(ScarDB.vault_id == vault_id).count()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting scar count for vault {vault_id}: {e}")
            return 0
        except Exception as e:
            logger.error(
                f"Unexpected error getting scar count for vault {vault_id}: {e}"
            )
            return 0

    def get_total_scar_weight(self, vault_id: str) -> float:
        """Return the sum of scar weights in the given vault."""
        if not self.db_initialized:
            logger.warning("Database not initialized. Returning zero scar weight.")
            return 0.0

        try:
            with SessionLocal() as db:
                total = (
                    db.query(func.sum(ScarDB.weight))
                    .filter(ScarDB.vault_id == vault_id)
                    .scalar()
                )
                return float(total or 0.0)
        except SQLAlchemyError as e:
            logger.error(
                f"Database error getting scar weight for vault {vault_id}: {e}"
            )
            return 0.0
        except Exception as e:
            logger.error(
                f"Unexpected error getting scar weight for vault {vault_id}: {e}"
            )
            return 0.0

    def detect_vault_imbalance(
        self,
        *,
        by_weight: bool = False,
        threshold: float = 1.5,
    ) -> Tuple[bool, str, str]:
        """Determine if one vault significantly outweighs the other."""

        Returns a tuple of (imbalanced?, overloaded_vault, underloaded_vault).
        """
        if not self.db_initialized:
            logger.warning("Database not initialized. Cannot detect vault imbalance.")
            return False, "", ""

        try:
            if by_weight:
                a_val = self.get_total_scar_weight("vault_a")
                b_val = self.get_total_scar_weight("vault_b")
            else:
                a_val = self.get_total_scar_count("vault_a")
                b_val = self.get_total_scar_count("vault_b")

            if b_val == 0 and a_val == 0:
                return False, "", ""

            if a_val > threshold * max(b_val, 1e-9):
                return True, "vault_a", "vault_b"
            if b_val > threshold * max(a_val, 1e-9):
                return True, "vault_b", "vault_a"
            return False, "", ""
        except Exception as e:
            logger.error(f"Error detecting vault imbalance: {e}")
            return False, "", ""

    def rebalance_vaults(
        self,
        *,
        by_weight: bool = False,
        threshold: float = 1.5,
    ) -> int:
        """Move low priority scars from the overloaded vault to the other."""
        if not self.db_initialized:
            logger.warning("Database not initialized. Cannot rebalance vaults.")
            return 0

        try:
            imbalanced, from_vault, to_vault = self.detect_vault_imbalance(
                by_weight=by_weight, threshold=threshold
            )
            if not imbalanced:
                return 0

            # Get the counts to determine how many to move
            from_count = self.get_total_scar_count(from_vault)
            to_count = self.get_total_scar_count(to_vault)
            target_count = int((from_count + to_count) / 2)
            to_move = from_count - target_count

            logger.info(
                f"Rebalancing vaults: moving {to_move} scars from {from_vault} to {to_vault}"
            )

            # Get the scars to move (oldest accessed first)
            with SessionLocal() as db:
                scars_to_move = (
                    db.query(ScarDB)
                    .filter(ScarDB.vault_id == from_vault)
                    .order_by(ScarDB.last_accessed)
                    .limit(to_move)
                    .all()
                )

                # Update their vault_id
                for scar in scars_to_move:
                    scar.vault_id = to_vault

                db.commit()

            return len(scars_to_move)
        except SQLAlchemyError as e:
            logger.error(f"Database error during vault rebalancing: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during vault rebalancing: {e}")
            return 0

    def add_geoid(self, geoid: GeoidState) -> bool:
        """
        Add a geoid to the database.

        Returns True if successful, False otherwise.
        """
        if not self.db_initialized:
            logger.warning("Database not initialized. Cannot add geoid.")
            return False

        if not isinstance(geoid, GeoidState):
            logger.error(f"Invalid geoid type: {type(geoid)}")
            return False

        try:
            with SessionLocal() as db:
                # Check if geoid already exists
                existing = (
                    db.query(GeoidDB).filter(GeoidDB.geoid_id == geoid.geoid_id).first()
                )

                if existing:
                    # Update existing geoid
                    existing.symbolic_state = geoid.symbolic_state
                    existing.metadata_json = geoid.meta_data
                    existing.semantic_state_json = geoid.semantic_state
                    existing.semantic_vector = geoid.embedding_vector
                else:
                    # Create new geoid
                    geoid_db = GeoidDB(
                        geoid_id=geoid.geoid_id,
                        symbolic_state=geoid.symbolic_state,
                        metadata_json=geoid.meta_data,
                        semantic_state_json=geoid.semantic_state,
                        semantic_vector=geoid.embedding_vector,
                    )
                    db.add(geoid_db)

                db.commit()

                # Dual-write to Neo4j
                if self.neo4j_available:
                    try:
                        create_geoid(geoid.to_dict())
                    except Exception as e:
                        logger.warning(
                            f"Neo4j geoid creation failed for {geoid.geoid_id}: {e}"
                        )

                return True

        except SQLAlchemyError as e:
            logger.error(f"Database error adding geoid {geoid.geoid_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error adding geoid {geoid.geoid_id}: {e}")
            return False

    def search_geoids_by_embedding(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[GeoidDB]:
        """
        Search for geoids by embedding vector similarity.

        For PostgreSQL with pgvector, uses vector similarity search.
        For other databases, falls back to a simplified approach.
        """
        if not self.db_initialized:
            logger.warning("Database not initialized. Cannot search geoids.")
            return []

        try:
            with SessionLocal() as db:
                from sqlalchemy import text

                # Check if we're using PostgreSQL with pgvector
                if (
                    "postgresql" in db.bind.dialect.name
                    and hasattr(GeoidDB, "semantic_vector")
                    and hasattr(GeoidDB.semantic_vector.type, "cosine_distance")
                ):
                    # Use pgvector for efficient similarity search
                    from pgvector.sqlalchemy import Vector

                    # Convert query embedding to vector
                    query_vector = Vector(query_embedding)

                    # Perform cosine distance search
                    results = (
                        db.query(GeoidDB)
                        .order_by(GeoidDB.semantic_vector.cosine_distance(query_vector))
                        .limit(limit)
                        .all()
                    )

                    return results
                else:
                    # Fallback for databases without vector support
                    logger.warning(
                        "Vector search not supported by database. Returning random geoids."
                    )
                    return db.query(GeoidDB).limit(limit).all()

        except SQLAlchemyError as e:
            logger.error(f"Database error searching geoids by embedding: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching geoids by embedding: {e}")
            return []

    def search_scars_by_embedding(
        self, query_embedding: List[float], top_k: int = 3
    ) -> Dict:
        """
        Search for scars by embedding vector similarity.

        For PostgreSQL with pgvector, uses vector similarity search.
        For other databases, falls back to a simplified approach.
        """
        if not self.db_initialized:
            logger.warning("Database not initialized. Cannot search scars.")
            return {
                "results": [],
                "status": "error",
                "message": "Database not initialized",
            }

        try:
            with SessionLocal() as db:
                from sqlalchemy import text

                # Check if we're using PostgreSQL with pgvector
                if (
                    "postgresql" in db.bind.dialect.name
                    and hasattr(ScarDB, "scar_vector")
                    and hasattr(ScarDB.scar_vector.type, "cosine_distance")
                ):
                    # Use pgvector for efficient similarity search
                    from pgvector.sqlalchemy import Vector

                    # Convert query embedding to vector
                    query_vector = Vector(query_embedding)

                    # Perform cosine distance search
                    results = (
                        db.query(ScarDB)
                        .order_by(ScarDB.scar_vector.cosine_distance(query_vector))
                        .limit(top_k)
                        .all()
                    )

                    # Convert results to dictionary with safe attribute access
                    scar_list = []
                    for item in results:
                        try:
                            # Safe timestamp handling
                            timestamp_str = (
                                item.timestamp.isoformat()
                                if hasattr(item.timestamp, "isoformat")
                                else str(item.timestamp)
                            )

                            scar_dict = {
                                "scar_id": getattr(item, "scar_id", None),
                                "reason": getattr(item, "reason", None),
                                "geoids": getattr(item, "geoids", []),
                                "timestamp": timestamp_str,
                                "vault_id": getattr(item, "vault_id", None),
                                "semantic_polarity": getattr(
                                    item, "semantic_polarity", 0.0
                                ),
                                "delta_entropy": getattr(item, "delta_entropy", 0.0),
                            }
                            scar_list.append(scar_dict)
                        except Exception as e:
                            logger.warning(f"Error converting scar item: {e}")
                            # Add a minimal scar entry
                            scar_list.append(
                                {
                                    "scar_id": str(getattr(item, "scar_id", "unknown")),
                                    "reason": "Error accessing scar data",
                                    "geoids": [],
                                    "timestamp": "unknown",
                                    "vault_id": "unknown",
                                    "semantic_polarity": 0.0,
                                    "delta_entropy": 0.0,
                                }
                            )

                    return {
                        "results": scar_list,
                        "status": "success",
                        "count": len(scar_list),
                    }
                else:
                    # Fallback for databases without vector support
                    logger.warning(
                        "Vector search not supported by database. Returning random scars."
                    )
                    results = db.query(ScarDB).limit(top_k).all()

                    # Convert results to dictionary with safe attribute access (fallback)
                    scar_list = []
                    for item in results:
                        try:
                            # Safe timestamp handling
                            timestamp_str = (
                                item.timestamp.isoformat()
                                if hasattr(item.timestamp, "isoformat")
                                else str(item.timestamp)
                            )

                            scar_dict = {
                                "scar_id": getattr(item, "scar_id", None),
                                "reason": getattr(item, "reason", None),
                                "geoids": getattr(item, "geoids", []),
                                "timestamp": timestamp_str,
                                "vault_id": getattr(item, "vault_id", None),
                                "semantic_polarity": getattr(
                                    item, "semantic_polarity", 0.0
                                ),
                                "delta_entropy": getattr(item, "delta_entropy", 0.0),
                            }
                            scar_list.append(scar_dict)
                        except Exception as e:
                            logger.warning(
                                f"Error converting scar item (fallback): {e}"
                            )
                            # Add a minimal scar entry
                            scar_list.append(
                                {
                                    "scar_id": str(getattr(item, "scar_id", "unknown")),
                                    "reason": "Error accessing scar data",
                                    "geoids": [],
                                    "timestamp": "unknown",
                                    "vault_id": "unknown",
                                    "semantic_polarity": 0.0,
                                    "delta_entropy": 0.0,
                                }
                            )

                    return {
                        "results": scar_list,
                        "status": "fallback",
                        "count": len(scar_list),
                        "message": "Vector search not supported, using fallback",
                    }

        except SQLAlchemyError as e:
            logger.error(f"Database error searching scars by embedding: {e}")
            return {"results": [], "status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error searching scars by embedding: {e}")
            return {"results": [], "status": "error", "message": str(e)}

    def get_status(self) -> Dict:
        """Get the status of the vault manager and database"""
        status = {
            "initialized": self.db_initialized,
            "neo4j_available": self.neo4j_available,
            "timestamp": datetime.now().isoformat(),
        }

        if self.db_initialized:
            try:
                # Add vault statistics
                vault_a_count = self.get_total_scar_count("vault_a")
                vault_b_count = self.get_total_scar_count("vault_b")
                vault_a_weight = self.get_total_scar_weight("vault_a")
                vault_b_weight = self.get_total_scar_weight("vault_b")

                status["vaults"] = {
                    "vault_a": {"count": vault_a_count, "weight": vault_a_weight},
                    "vault_b": {"count": vault_b_count, "weight": vault_b_weight},
                    "total_count": vault_a_count + vault_b_count,
                    "total_weight": vault_a_weight + vault_b_weight,
                    "balance_ratio": max(vault_a_count, vault_b_count)
                    / (min(vault_a_count, vault_b_count) or 1),
                }

                # Add database status
                status["database"] = get_db_status()

            except Exception as e:
                status["error"] = str(e)

        return status

    def get_geoid_count(self) -> int:
        """Get total number of geoids in the database."""
        if not self.db_initialized:
            logger.warning("Database not initialized. Returning zero geoid count.")
            return 0

        try:
            with SessionLocal() as db:
                return db.query(GeoidDB).count()
        except Exception as e:
            logger.error(f"Error getting geoid count: {e}")
            return 0

    def get_scar_count(self) -> int:
        """Get total number of scars in the database."""
        if not self.db_initialized:
            logger.warning("Database not initialized. Returning zero scar count.")
            return 0

        try:
            with SessionLocal() as db:
                return db.query(ScarDB).count()
        except Exception as e:
            logger.error(f"Error getting scar count: {e}")
            return 0

    def get_all_scars(self, limit: int = 1000) -> List[ScarDB]:
        """Get all scars from the database."""
        if not self.db_initialized:
            logger.warning("Database not initialized. Returning empty scar list.")
            return []

        try:
            with SessionLocal() as db:
                return (
                    db.query(ScarDB)
                    .order_by(ScarDB.timestamp.desc())
                    .limit(limit)
                    .all()
                )
        except Exception as e:
            logger.error(f"Error getting all scars: {e}")
            return []
