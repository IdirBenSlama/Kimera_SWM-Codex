"""
Enhanced Database Schema for Kimera SWM

This module defines the enhanced database schema for the Kimera SWM system.
It implements lazy table creation to prevent import-time database connection failures.
"""

import logging
import os
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
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Database-agnostic imports
try:
    from pgvector.sqlalchemy import Vector

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False


# Check if we're using PostgreSQL or SQLite
def is_postgresql():
    """Check if we're using PostgreSQL based on environment variables"""
    db_url = os.getenv("DATABASE_URL", os.getenv("KIMERA_DATABASE_URL", ""))
    return "postgresql" in db_url.lower()


# Database-agnostic column types
def get_array_column():
    """Get appropriate array column type based on database"""
    if is_postgresql():
        return ARRAY(Text)
    else:
        return Text  # Store as JSON string for SQLite


def get_vector_column(dimensions=768):
    """Get appropriate vector column type based on database"""
    if is_postgresql() and PGVECTOR_AVAILABLE:
        return Vector(dimensions)
    else:
        return Text  # Store as JSON string for SQLite


def get_uuid_column():
    """Get appropriate UUID column type based on database"""
    if is_postgresql():
        return UUID(as_uuid=True)
    else:
        return String(36)  # Store as string for SQLite


logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()


class GeoidState(Base):
    """
    Represents a cognitive state as a high-dimensional vector.
    """

    __tablename__ = "geoid_states"

    id = Column(get_uuid_column(), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow)
    state_vector = Column(get_vector_column(768), nullable=False)
    meta_data = Column(JSON, default={})
    entropy = Column(Float, nullable=False)
    coherence_factor = Column(Float, nullable=False)
    energy_level = Column(Float, nullable=False, default=1.0)
    creation_context = Column(Text)
    tags = Column(get_array_column())

    # Relationships
    transitions_as_source = relationship(
        "CognitiveTransition",
        foreign_keys="CognitiveTransition.source_id",
        back_populates="source",
    )
    transitions_as_target = relationship(
        "CognitiveTransition",
        foreign_keys="CognitiveTransition.target_id",
        back_populates="target",
    )


class CognitiveTransition(Base):
    """
    Represents a transition between cognitive states.
    """

    __tablename__ = "cognitive_transitions"

    id = Column(get_uuid_column(), primary_key=True, default=uuid.uuid4)
    source_id = Column(get_uuid_column(), ForeignKey("geoid_states.id"), nullable=False)
    target_id = Column(get_uuid_column(), ForeignKey("geoid_states.id"), nullable=False)
    transition_energy = Column(Float, nullable=False)
    conservation_error = Column(Float, nullable=False)
    transition_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(JSON, default={})

    # Relationships
    source = relationship(
        "GeoidState", foreign_keys=[source_id], back_populates="transitions_as_source"
    )
    target = relationship(
        "GeoidState", foreign_keys=[target_id], back_populates="transitions_as_target"
    )


class SemanticEmbedding(Base):
    """
    Stores text embeddings for semantic operations.
    """

    __tablename__ = "semantic_embeddings"

    id = Column(get_uuid_column(), primary_key=True, default=uuid.uuid4)
    text_content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # Stored as JSON string for compatibility
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String(100))
    meta_data = Column(JSON, default={})


class PortalConfiguration(Base):
    """
    Stores configurations for interdimensional portals.
    """

    __tablename__ = "portal_configurations"

    id = Column(get_uuid_column(), primary_key=True, default=uuid.uuid4)
    source_dimension = Column(Integer, nullable=False)
    target_dimension = Column(Integer, nullable=False)
    radius = Column(Float, nullable=False)
    energy_requirement = Column(Float, nullable=False)
    stability_factor = Column(Float, nullable=False)
    creation_timestamp = Column(DateTime, default=datetime.utcnow)
    last_used_timestamp = Column(DateTime)
    configuration_parameters = Column(JSONB, nullable=False)
    status = Column(String(20), nullable=False)


class SystemMetric(Base):
    """
    Records system performance metrics.
    """

    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    component = Column(String(100), nullable=False)
    context = Column(JSON, default={})


class InsightDB(Base):
    """
    Stores insights and scars from the understanding system.
    """

    __tablename__ = "insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    insight_id = Column(String(100), unique=True, nullable=False)
    insight_type = Column(String(50), nullable=False)
    source_resonance_id = Column(String(100))
    echoform_repr = Column(Text)
    application_domains = Column(get_array_column())
    confidence = Column(Float, nullable=False)
    entropy_reduction = Column(Float, nullable=False)
    utility_score = Column(Float, nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_reinforced_cycle = Column(String(100))


# Understanding-oriented database models for cognitive evolution
class MultimodalGroundingDB(Base):
    """Multimodal grounding for connecting abstract concepts to real-world experiences"""

    __tablename__ = "multimodal_grounding"

    id = Column(Integer, primary_key=True, autoincrement=True)
    grounding_id = Column(String(100), unique=True, nullable=False)
    concept_id = Column(String(100), nullable=False)
    visual_features = Column(JSON, default={})
    auditory_features = Column(JSON, default={})
    tactile_features = Column(JSON, default={})
    temporal_context = Column(JSON, default={})
    physical_properties = Column(JSON, default={})
    confidence_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class CausalRelationshipDB(Base):
    """Causal relationships between concepts"""

    __tablename__ = "causal_relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    relationship_id = Column(String(100), unique=True, nullable=False)
    cause_concept_id = Column(String(100), nullable=False)
    effect_concept_id = Column(String(100), nullable=False)
    causal_strength = Column(Float, nullable=False)
    evidence_quality = Column(Float, default=0.0)
    mechanism_description = Column(Text)
    counterfactual_scenarios = Column(JSONB, default=[])
    causal_delay = Column(Float, default=0.0)
    temporal_pattern = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class SelfModelDB(Base):
    """System's model of itself"""

    __tablename__ = "self_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(100), unique=True, nullable=False)
    model_version = Column(Integer, nullable=False)
    processing_capabilities = Column(JSON, default={})
    knowledge_domains = Column(JSON, default={})
    reasoning_patterns = Column(JSON, default={})
    limitation_awareness = Column(JSON, default={})
    self_assessment_accuracy = Column(Float, default=0.0)
    introspection_depth = Column(Integer, default=0)
    metacognitive_awareness = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


class IntrospectionLogDB(Base):
    """Introspection logs for self-awareness"""

    __tablename__ = "introspection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(String(100), unique=True, nullable=False)
    introspection_type = Column(String(50), nullable=False)
    current_state_analysis = Column(JSON, default={})
    predicted_state = Column(JSON, default={})
    actual_state = Column(JSON, default={})
    accuracy_score = Column(Float, default=0.0)
    processing_context = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


class CompositionSemanticDB(Base):
    """Compositional semantic understanding"""

    __tablename__ = "composition_semantics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    composition_id = Column(String(100), unique=True, nullable=False)
    component_concepts = Column(get_array_column())
    composition_rules = Column(JSON, default={})
    emergent_meaning = Column(JSON, default={})
    context_variations = Column(JSON, default={})
    understanding_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ConceptualAbstractionDB(Base):
    """Conceptual abstraction for understanding"""

    __tablename__ = "conceptual_abstractions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    concept_id = Column(String(100), unique=True, nullable=False)
    concept_name = Column(String(200), nullable=False)
    essential_properties = Column(JSON, default={})
    concrete_instances = Column(JSONB, default=[])
    abstraction_level = Column(Integer, default=0)
    concept_coherence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ValueSystemDB(Base):
    """Value system for ethical reasoning"""

    __tablename__ = "value_systems"

    id = Column(Integer, primary_key=True, autoincrement=True)
    value_id = Column(String(100), unique=True, nullable=False)
    value_name = Column(String(200), nullable=False)
    value_description = Column(Text)
    learning_source = Column(String(100))
    learning_evidence = Column(JSON, default={})
    value_strength = Column(Float, default=0.0)
    value_priority = Column(Integer, default=5)
    created_at = Column(DateTime, default=datetime.utcnow)


class GenuineOpinionDB(Base):
    """Genuine opinions formed by the system"""

    __tablename__ = "genuine_opinions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    opinion_id = Column(String(100), unique=True, nullable=False)
    topic = Column(String(200), nullable=False)
    stance = Column(String(100), nullable=False)
    reasoning = Column(Text)
    supporting_values = Column(get_array_column())
    supporting_evidence = Column(JSON, default={})
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class EthicalReasoningDB(Base):
    """Ethical reasoning processes"""

    __tablename__ = "ethical_reasoning"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reasoning_id = Column(String(100), unique=True, nullable=False)
    ethical_dilemma = Column(Text, nullable=False)
    stakeholders = Column(JSONB, default=[])
    potential_harms = Column(JSONB, default=[])
    potential_benefits = Column(JSONB, default=[])
    reasoning_approach = Column(String(100))
    decision_rationale = Column(Text)
    confidence_level = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class EnhancedScarDB(Base):
    """Enhanced SCAR with understanding depth"""

    __tablename__ = "enhanced_scars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scar_id = Column(String(100), unique=True, nullable=False)
    traditional_scar_id = Column(String(100))
    understanding_depth = Column(Float, default=0.0)
    causal_understanding = Column(JSON, default={})
    compositional_analysis = Column(JSON, default={})
    contextual_factors = Column(JSON, default={})
    introspective_accuracy = Column(Float, default=0.0)
    value_implications = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


class EnhancedGeoidDB(Base):
    """Enhanced Geoid with understanding components"""

    __tablename__ = "enhanced_geoids"

    id = Column(Integer, primary_key=True, autoincrement=True)
    geoid_id = Column(String(100), unique=True, nullable=False)
    traditional_geoid_id = Column(String(100))
    compositional_structure = Column(JSON, default={})
    abstraction_level = Column(Integer, default=0)
    causal_relationships = Column(JSON, default={})
    understanding_confidence = Column(Float, default=0.0)
    multimodal_groundings = Column(get_array_column())
    created_at = Column(DateTime, default=datetime.utcnow)


class UnderstandingTestDB(Base):
    """Understanding tests and results"""

    __tablename__ = "understanding_tests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    test_id = Column(String(100), unique=True, nullable=False)
    test_type = Column(String(100), nullable=False)
    test_description = Column(Text)
    test_input = Column(JSON, default={})
    expected_output = Column(JSON, default={})
    actual_output = Column(JSON, default={})
    understanding_accuracy = Column(Float, default=0.0)
    system_state = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


class ComplexityIndicatorDB(Base):
    """Complexity indicators for consciousness measurement"""

    __tablename__ = "complexity_indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    indicator_id = Column(String(100), unique=True, nullable=False)
    phi_value = Column(Float)
    global_accessibility = Column(Float)
    reportability_score = Column(Float)
    experience_report = Column(JSON, default={})
    processing_context = Column(JSON, default={})
    awareness_level = Column(Float, default=0.0)
    processing_integration = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


# Define function to create all tables
def create_tables(engine):
    """
    Create all tables in the database.

    Args:
        engine: SQLAlchemy engine instance
    """
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    # Create vector extension and indexes if PostgreSQL
    if engine.dialect.name == "postgresql":
        try:
            with engine.connect() as conn:
                # Create vector extension if not exists
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                # Create vector indexes
                conn.execute(
                    text(
                        """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE c.relname = 'idx_geoid_vector' AND n.nspname = 'public'
                        ) THEN
                            CREATE INDEX idx_geoid_vector ON geoid_states USING ivfflat (state_vector vector_cosine_ops);
                        END IF;
                        
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                            WHERE c.relname = 'idx_embedding_vector' AND n.nspname = 'public'
                        ) THEN
                            CREATE INDEX idx_embedding_vector ON semantic_embeddings USING ivfflat (embedding vector_cosine_ops);
                        END IF;
                    END
                    $$;
                """
                    )
                )

                logger.info("PostgreSQL vector extension created successfully")
        except Exception as e:
            logger.warning(f"Could not create vector extension or indexes: {e}")

    logger.info("Database tables created successfully")


# Do not create tables at import time - this will be done explicitly when needed
# This prevents database connection failures during import
