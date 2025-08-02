"""
SQLite-Compatible Database Schema for Kimera SWM
"""

from sqlalchemy import Column, String, Float, DateTime, Text, Integer, ForeignKey, JSON, Boolean, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class GeoidState(Base):
    """SQLite-compatible geoid state table"""
    __tablename__ = "geoid_states"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow)
    state_vector = Column(Text, nullable=False)  # JSON string for vector data
    meta_data = Column(JSON, default={})  # Use JSON instead of JSONB
    entropy = Column(Float, nullable=False)
    coherence_factor = Column(Float, nullable=False) 
    energy_level = Column(Float, nullable=False, default=1.0)
    creation_context = Column(Text)
    tags = Column(Text)  # JSON string for tags array
    
    # Relationships
    transitions_as_source = relationship("CognitiveTransition", 
                                        foreign_keys="CognitiveTransition.source_id",
                                        back_populates="source")
    transitions_as_target = relationship("CognitiveTransition", 
                                        foreign_keys="CognitiveTransition.target_id",
                                        back_populates="target")

class CognitiveTransition(Base):
    """SQLite-compatible cognitive transition table"""
    __tablename__ = "cognitive_transitions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String, ForeignKey("geoid_states.id"), nullable=False)
    target_id = Column(String, ForeignKey("geoid_states.id"), nullable=False)
    transition_energy = Column(Float, nullable=False)
    conservation_error = Column(Float, nullable=False)
    transition_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_data = Column(JSON, default={})
    
    # Relationships
    source = relationship("GeoidState", foreign_keys=[source_id], back_populates="transitions_as_source")
    target = relationship("GeoidState", foreign_keys=[target_id], back_populates="transitions_as_target")

class SemanticEmbedding(Base):
    """SQLite-compatible semantic embedding table"""
    __tablename__ = "semantic_embeddings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    text_content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON string for embedding vector
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String(100))
    meta_data = Column(JSON, default={})

def create_sqlite_tables(engine):
    """Create all tables in SQLite database"""
    try:
        Base.metadata.create_all(engine)
        return True
    except Exception as e:
        print(f"Failed to create tables: {e}")
        return False
