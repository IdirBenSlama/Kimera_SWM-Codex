"""
Dynamic Database Schema for Kimera SWM
======================================

This module creates database schema dynamically based on the actual database engine,
ensuring compatibility with both PostgreSQL and SQLite.
"""

import logging
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

logger = logging.getLogger(__name__)


def create_dynamic_schema(engine):
    """
    Create database schema dynamically based on the engine type.

    Args:
        engine: SQLAlchemy engine instance

    Returns:
        MetaData: SQLAlchemy metadata with all tables
    """
    metadata = MetaData()
    is_postgresql = engine.dialect.name == "postgresql"

    logger.info(f"Creating dynamic schema for {engine.dialect.name}")

    # Helper functions for column types
    def get_uuid_column():
        if is_postgresql:
            return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        else:
            return Column(
                String(36), primary_key=True, default=lambda: str(uuid.uuid4())
            )

    def get_uuid_fk_column(table_name):
        if is_postgresql:
            return Column(
                UUID(as_uuid=True), ForeignKey(f"{table_name}.id"), nullable=False
            )
        else:
            return Column(String(36), ForeignKey(f"{table_name}.id"), nullable=False)

    def get_array_column():
        if is_postgresql:
            return Column(ARRAY(Text))
        else:
            return Column(Text)  # Store as JSON string for SQLite

    def get_vector_column(dimensions=768):
        if is_postgresql:
            try:
                from pgvector.sqlalchemy import Vector

                return Column(Vector(dimensions), nullable=False)
            except ImportError:
                pass
        return Column(Text, nullable=False)  # Store as JSON string

    def get_json_column():
        if is_postgresql:
            return Column(JSONB, default={})
        else:
            return Column(JSON, default={})

    # Create GeoidState table
    state_vector_col = get_vector_column(768)
    state_vector_col.name = "state_vector"
    meta_data_col = get_json_column()
    meta_data_col.name = "meta_data"
    tags_col = get_array_column()
    tags_col.name = "tags"

    geoid_states = Table(
        "geoid_states",
        metadata,
        Column(
            "id",
            String(36) if not is_postgresql else UUID(as_uuid=True),
            primary_key=True,
            default=lambda: str(uuid.uuid4()) if not is_postgresql else uuid.uuid4,
        ),
        Column("timestamp", DateTime, default=datetime.utcnow),
        state_vector_col,
        meta_data_col,
        Column("entropy", Float, nullable=False),
        Column("coherence_factor", Float, nullable=False),
        Column("energy_level", Float, nullable=False, default=1.0),
        Column("creation_context", Text),
        tags_col,
    )

    # Create CognitiveTransition table
    source_id_col = get_uuid_fk_column("geoid_states")
    source_id_col.name = "source_id"
    target_id_col = get_uuid_fk_column("geoid_states")
    target_id_col.name = "target_id"
    ct_meta_data_col = get_json_column()
    ct_meta_data_col.name = "meta_data"

    cognitive_transitions = Table(
        "cognitive_transitions",
        metadata,
        Column(
            "id",
            String(36) if not is_postgresql else UUID(as_uuid=True),
            primary_key=True,
            default=lambda: str(uuid.uuid4()) if not is_postgresql else uuid.uuid4,
        ),
        source_id_col,
        target_id_col,
        Column("transition_energy", Float, nullable=False),
        Column("conservation_error", Float, nullable=False),
        Column("transition_type", String(50), nullable=False),
        Column("timestamp", DateTime, default=datetime.utcnow),
        ct_meta_data_col,
    )

    # Create SemanticEmbedding table
    se_meta_data_col = get_json_column()
    se_meta_data_col.name = "meta_data"

    semantic_embeddings = Table(
        "semantic_embeddings",
        metadata,
        Column(
            "id",
            String(36) if not is_postgresql else UUID(as_uuid=True),
            primary_key=True,
            default=lambda: str(uuid.uuid4()) if not is_postgresql else uuid.uuid4,
        ),
        Column("text_content", Text, nullable=False),
        Column("embedding", Text, nullable=False),  # JSON string for compatibility
        Column("timestamp", DateTime, default=datetime.utcnow),
        Column("source", String(100)),
        se_meta_data_col,
    )

    # Create simplified tables for core functionality
    sm_metric_data_col = get_json_column()
    sm_metric_data_col.name = "metric_data"

    system_metrics = Table(
        "system_metrics",
        metadata,
        Column(
            "id",
            String(36) if not is_postgresql else UUID(as_uuid=True),
            primary_key=True,
            default=lambda: str(uuid.uuid4()) if not is_postgresql else uuid.uuid4,
        ),
        Column("timestamp", DateTime, default=datetime.utcnow),
        Column("metric_name", String(100), nullable=False),
        Column("metric_value", Float),
        sm_metric_data_col,
    )

    vault_entries = Table(
        "vault_entries",
        metadata,
        Column(
            "id",
            String(36) if not is_postgresql else UUID(as_uuid=True),
            primary_key=True,
            default=lambda: str(uuid.uuid4()) if not is_postgresql else uuid.uuid4,
        ),
        Column("timestamp", DateTime, default=datetime.utcnow),
        Column("entry_type", String(50), nullable=False),
        Column("entry_data", Text, nullable=False),
        Column("encryption_status", String(20), default="none"),
    )

    logger.info(f"Created {len(metadata.tables)} tables for {engine.dialect.name}")
    return metadata


def create_tables_safely(engine):
    """
    Safely create all tables with proper error handling.

    Args:
        engine: SQLAlchemy engine instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        metadata = create_dynamic_schema(engine)

        # Create all tables
        metadata.create_all(bind=engine)

        # Create PostgreSQL specific optimizations
        if engine.dialect.name == "postgresql":
            try:
                with engine.connect() as conn:
                    # Create vector extension if available
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                    logger.info("PostgreSQL vector extension created")

                    # Create basic indexes
                    try:
                        conn.execute(
                            text(
                                """
                            CREATE INDEX IF NOT EXISTS idx_geoid_timestamp 
                            ON geoid_states (timestamp);
                        """
                            )
                        )
                        conn.execute(
                            text(
                                """
                            CREATE INDEX IF NOT EXISTS idx_transitions_source 
                            ON cognitive_transitions (source_id);
                        """
                            )
                        )
                        conn.commit()
                        logger.info("PostgreSQL indexes created")
                    except Exception as idx_error:
                        logger.warning(f"Could not create indexes: {idx_error}")

            except Exception as pg_error:
                logger.warning(f"PostgreSQL optimizations failed: {pg_error}")

        logger.info("✅ Database tables created successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        return False
