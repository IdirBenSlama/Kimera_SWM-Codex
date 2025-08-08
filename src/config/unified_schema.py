"""
Unified Database Schema for KIMERA SWM
======================================

Consolidated schema definitions with proper relationships and indexing.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Index
                        Integer, String, Text, UniqueConstraint)
from sqlalchemy.dialects.postgresql import UUID, VECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship
from sqlalchemy.sql import func

Base = declarative_base()
class TimestampMixin:
    """Auto-generated class."""
    pass
    """Mixin for created/updated timestamps"""

    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )


class GeoidState(Base, TimestampMixin):
    """Core geoid state representation"""

    __tablename__ = "geoid_states"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Geoid coordinates and properties
    theta = Column(Float, nullable=False)
    phi = Column(Float, nullable=False)
    radius = Column(Float, nullable=False, default=1.0)

    # Thermodynamic properties
    temperature = Column(Float, nullable=False, default=1.0)
    energy = Column(Float, nullable=False, default=0.0)
    entropy = Column(Float, nullable=False, default=0.0)

    # Cognitive properties
    cognitive_potential = Column(Float, nullable=False, default=0.0)
    consciousness_level = Column(Float, nullable=False, default=0.0)

    # Metadata
    system_state = Column(String, nullable=False, default="stable")
    metadata = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_geoid_coordinates", "theta", "phi"),
        Index("idx_geoid_energy", "energy"),
        Index("idx_geoid_cognitive", "cognitive_potential"),
    )


class TradingSignal(Base, TimestampMixin):
    """Trading signals and decisions"""

    __tablename__ = "trading_signals"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Signal properties
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(20), nullable=False)  # buy, sell, hold
    confidence = Column(Float, nullable=False)
    strength = Column(Float, nullable=False)

    # Market data
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)

    # Cognitive analysis
    geoid_state_id = Column(String, ForeignKey("geoid_states.id"), nullable=True)
    cognitive_analysis = Column(JSON, nullable=True)

    # Execution
    executed = Column(Boolean, default=False)
    execution_price = Column(Float, nullable=True)
    execution_time = Column(DateTime, nullable=True)

    # Relationships
    geoid_state = relationship("GeoidState", backref="trading_signals")

    __table_args__ = (
        Index("idx_trading_symbol", "symbol"),
        Index("idx_trading_signal_type", "signal_type"),
        Index("idx_trading_confidence", "confidence"),
        Index("idx_trading_executed", "executed"),
    )


class PerformanceMetric(Base, TimestampMixin):
    """Performance and monitoring metrics"""

    __tablename__ = "performance_metrics"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Metric identification
    component = Column(String(100), nullable=False)  # gpu, thermodynamic, cognitive
    metric_type = Column(
        String(50), nullable=False
    )  # utilization, temperature, efficiency

    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    threshold_warning = Column(Float, nullable=True)
    threshold_critical = Column(Float, nullable=True)

    # Context
    system_state = Column(String(50), nullable=True)
    additional_data = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_perf_component", "component"),
        Index("idx_perf_type", "metric_type"),
        Index("idx_perf_created", "created_at"),
        # Composite index for time-series queries
        Index("idx_perf_component_time", "component", "created_at"),
    )


class SystemEvent(Base, TimestampMixin):
    """System events and logging"""

    __tablename__ = "system_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Event classification
    event_type = Column(String(50), nullable=False)  # info, warning, error, critical
    component = Column(String(100), nullable=False)

    # Event details
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)

    # Context
    session_id = Column(String, nullable=True)
    user_id = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_event_type", "event_type"),
        Index("idx_event_component", "component"),
        Index("idx_event_created", "created_at"),
    )


class ConfigurationSetting(Base, TimestampMixin):
    """Dynamic configuration settings"""

    __tablename__ = "configuration_settings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Setting identification
    category = Column(String(50), nullable=False)  # database, gpu, thermodynamic
    key = Column(String(100), nullable=False)

    # Setting value
    value = Column(Text, nullable=False)  # JSON encoded
    value_type = Column(String(20), nullable=False)  # string, int, float, bool, json

    # Metadata
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint("category", "key", name="uq_config_category_key"),
        Index("idx_config_category", "category"),
        Index("idx_config_active", "is_active"),
    )


# Vector storage for PostgreSQL (optional, falls back gracefully)
try:
    from sqlalchemy.dialects.postgresql import VECTOR

    class VectorEmbedding(Base, TimestampMixin):
        """Vector embeddings for semantic analysis"""

        __tablename__ = "vector_embeddings"

        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

        # Embedding identification
        content_id = Column(String, nullable=False)
        content_type = Column(String(50), nullable=False)  # geoid, signal, text

        # Vector data
        embedding = Column(VECTOR(1536), nullable=False)  # OpenAI embedding size

        # Metadata
        model_version = Column(String(50), nullable=False)
        content_hash = Column(String(64), nullable=True)

        __table_args__ = (
            Index("idx_vector_content", "content_id"),
            Index("idx_vector_type", "content_type"),
        )

except ImportError:
    # Fallback for non-PostgreSQL databases
    class VectorEmbedding(Base, TimestampMixin):
        """Vector embeddings fallback (without VECTOR type)"""

        __tablename__ = "vector_embeddings"

        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        content_id = Column(String, nullable=False)
        content_type = Column(String(50), nullable=False)

        # Store as JSON for non-PostgreSQL databases
        embedding = Column(JSON, nullable=False)

        model_version = Column(String(50), nullable=False)
        content_hash = Column(String(64), nullable=True)


def create_tables(engine):
    """Create all tables with proper error handling"""
    try:
        Base.metadata.create_all(bind=engine)
        return True
    except Exception as e:
        print(f"Error creating tables: {e}")
        return False


def drop_tables(engine):
    """Drop all tables (use with caution)"""
    try:
        Base.metadata.drop_all(bind=engine)
        return True
    except Exception as e:
        print(f"Error dropping tables: {e}")
        return False
