"""
Database module for Kimera SWM

This module provides database connectivity and session management.
It implements multiple authentication strategies and lazy table creation.
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import OperationalError
from typing import Optional, Dict, Any

from .database_connection_manager import DatabaseConnectionManager
from .enhanced_database_schema import Base, create_tables

logger = logging.getLogger(__name__)

# Initialize the database connection manager
connection_manager = DatabaseConnectionManager(
    pool_size=5,
    max_overflow=10,
    pool_timeout=30
)

# Initialize engine and session
engine = None
SessionLocal = None

def initialize_database():
    """
    Initialize the database connection and create tables if needed.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global engine, SessionLocal
    
    try:
        # Initialize connection using the connection manager
        engine = connection_manager.initialize_connection()
        
        # Check if pgvector extension is available (PostgreSQL only)
        if engine.dialect.name == 'postgresql':
            with engine.connect() as conn:
                try:
                    conn.execute(text("SELECT 'dummy'::vector"))
                    logger.info("pgvector extension available for PostgreSQL vector operations")
                except Exception as e:
                    logger.warning(f"pgvector extension not available: {e}")
        
        # Log database information
        if engine.dialect.name == 'postgresql':
            logger.info("Using PostgreSQL with optimized connection settings")
        elif engine.dialect.name == 'sqlite':
            logger.info("Using SQLite database (development/fallback mode)")
        
        # Create session factory
        SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
        
        # Create tables using the dynamic schema
        from .dynamic_schema import create_tables_safely
        if not create_tables_safely(engine):
            logger.warning("Failed to create dynamic schema, attempting basic schema")
            # Fallback to simple table creation
            try:
                from .sqlite_schema import create_sqlite_tables
                create_sqlite_tables(engine)
            except Exception as fallback_error:
                logger.error(f"Fallback schema creation failed: {fallback_error}")
                raise
        
        logger.info("Database engine created successfully")
        logger.info("Database session factory created successfully")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def get_db():
    """
    Get a database session.
    
    Yields:
        Session: SQLAlchemy session
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_engine():
    """
    Get the SQLAlchemy engine instance.
    
    Returns:
        Engine: SQLAlchemy engine instance
    """
    if engine is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    
    return engine

def get_database_info() -> Dict[str, Any]:
    """
    Get information about the connected database.
    
    Returns:
        Dict[str, Any]: Database information
    """
    if engine is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    
    return connection_manager.get_database_info()

def get_db_status() -> Dict[str, Any]:
    """
    Get the current database connection status.
    
    Returns:
        Dict[str, Any]: Database status information
    """
    if engine is None:
        return {
            "status": "disconnected",
            "message": "Database not initialized"
        }
    
    try:
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return {
            "status": "connected",
            "dialect": engine.dialect.name,
            "message": "Database connection is healthy"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Database connection failed: {e}"
        }

class GeoidDB:
    """
    Database access class for Geoid operations.
    """
    def __init__(self, session=None):
        """
        Initialize GeoidDB with an optional session.
        
        Args:
            session: SQLAlchemy session (optional)
        """
        self.session = session
    
    def store_geoid(self, geoid_data):
        """
        Store a geoid in the database.
        
        Args:
            geoid_data: Geoid data to store
            
        Returns:
            str: ID of the stored geoid
        """
        # Implementation details...
        pass
    
    def retrieve_geoid(self, geoid_id):
        """
        Retrieve a geoid from the database.
        
        Args:
            geoid_id: ID of the geoid to retrieve
            
        Returns:
            dict: Retrieved geoid data
        """
        # Implementation details...
        pass

class ScarDB:
    """
    Database access class for Scar operations.
    """
    def __init__(self, session=None):
        """
        Initialize ScarDB with an optional session.
        
        Args:
            session: SQLAlchemy session (optional)
        """
        self.session = session
    
    def store_transition(self, source_id, target_id, transition_data):
        """
        Store a cognitive transition in the database.
        
        Args:
            source_id: ID of the source geoid
            target_id: ID of the target geoid
            transition_data: Transition data to store
            
        Returns:
            str: ID of the stored transition
        """
        # Implementation details...
        pass
    
    def retrieve_transitions(self, geoid_id):
        """
        Retrieve transitions for a geoid from the database.
        
        Args:
            geoid_id: ID of the geoid
            
        Returns:
            list: Retrieved transition data
        """
        # Implementation details...
        pass

# Import models from enhanced_database_schema for backward compatibility
from .enhanced_database_schema import InsightDB

def create_tables(engine):
    """
    Create all database tables.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        bool: True if tables were created successfully, False otherwise
    """
    try:
        from .dynamic_schema import create_tables_safely
        return create_tables_safely(engine)
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        # Try fallback to simple SQLite schema
        try:
            from .sqlite_schema import create_sqlite_tables  
            return create_sqlite_tables(engine)
        except Exception as fallback_error:
            logger.error(f"Fallback schema creation failed: {fallback_error}")
            return False

