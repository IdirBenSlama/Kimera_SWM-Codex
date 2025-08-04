"""
Database Connection Manager for Kimera SWM

This module provides a robust connection management system with multiple
authentication strategies and graceful fallback mechanisms.
"""

import os
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import OperationalError, SQLAlchemyError

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """
    Database connection manager with multiple authentication strategies.
    
    This class implements a two-stage connection strategy pattern:
    1. Primary Strategy: Attempts connection using Kimera-specific credentials
    2. Secondary Strategy: Falls back to environment variable configuration
    
    Attributes:
        pool_size (int): Connection pool size
        max_overflow (int): Maximum number of connections to overflow
        pool_timeout (int): Timeout for acquiring a connection from the pool
        engine (Engine): SQLAlchemy engine instance
        session_factory (sessionmaker): SQLAlchemy session factory
    """
    
    def __init__(
        self,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30
    ):
        """
        Initialize the database connection manager.
        
        Args:
            pool_size (int): Connection pool size
            max_overflow (int): Maximum number of connections to overflow
            pool_timeout (int): Timeout for acquiring a connection from the pool
        """
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.engine = None
        self.session_factory = None
        
    def initialize_connection(self) -> Engine:
        """
        Initialize database connection using multiple strategies.
        
        Returns:
            Engine: SQLAlchemy engine instance
            
        Raises:
            RuntimeError: If all connection strategies fail
        """
        # Try primary strategy (Kimera-specific credentials)
        try:
            logger.info("Attempting connection with Kimera credentials...")
            self.engine = self._connect_with_kimera_credentials()
            self._verify_connection()
            logger.info("Database connection successful using Kimera credentials")
            self._initialize_session_factory()
            return self.engine
        except Exception as e:
            logger.warning(f"Primary connection strategy failed: {e}")
        
        # Try secondary strategy (Environment variables)
        try:
            logger.info("Attempting connection with environment variables...")
            self.engine = self._connect_with_env_variables()
            self._verify_connection()
            logger.info("Database connection successful using environment variables")
            self._initialize_session_factory()
            return self.engine
        except Exception as e:
            logger.warning(f"Secondary connection strategy failed: {e}")
        
        logger.error("All connection strategies failed")
        raise RuntimeError("Failed to establish database connection using any strategy")
    
    def _connect_with_kimera_credentials(self) -> Engine:
        """
        Connect using Kimera-specific credentials.
        
        Returns:
            Engine: SQLAlchemy engine instance
        """
        url = "postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm"
        return self._create_engine(url)
    
    def _connect_with_env_variables(self) -> Engine:
        """
        Connect using environment variables.
        
        Returns:
            Engine: SQLAlchemy engine instance
            
        Raises:
            ValueError: If DATABASE_URL environment variable is not set
        """
        url = os.environ.get("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL environment variable not set")
        return self._create_engine(url)
    
    def _create_engine(self, url: str) -> Engine:
        """Create SQLAlchemy engine with optimized parameters."""
        connect_args = {
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            "application_name": "Kimera SWM",
        }
        engine_args = {
            "pool_pre_ping": True,
            "pool_recycle": 3600,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
        }
        return create_engine(url, connect_args=connect_args, **engine_args)
    
    def _verify_connection(self) -> None:
        """
        Verify that the database connection is working.
        
        Raises:
            OperationalError: If the connection fails
        """
        if not self.engine:
            raise OperationalError("Engine not initialized")
        
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result != 1:
                raise OperationalError("Connection verification failed")
    
    def _initialize_session_factory(self) -> None:
        """Initialize the session factory."""
        if not self.engine:
            raise RuntimeError("Cannot initialize session factory: engine not initialized")
        
        Session = sessionmaker(bind=self.engine)
        self.session_factory = scoped_session(Session)
        logger.info("Database connection established successfully")
    
    def get_session(self):
        """
        Get a new database session.
        
        Returns:
            Session: SQLAlchemy session
            
        Raises:
            RuntimeError: If session factory is not initialized
        """
        if not self.session_factory:
            raise RuntimeError("Session factory not initialized")
        return self.session_factory()
    
    def close_all_sessions(self) -> None:
        """Close all sessions."""
        if self.session_factory:
            self.session_factory.remove()
    
    def get_engine(self) -> Optional[Engine]:
        """
        Get the SQLAlchemy engine instance.
        
        Returns:
            Optional[Engine]: SQLAlchemy engine instance or None if not initialized
        """
        return self.engine
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the connected database.
        
        Returns:
            Dict[str, Any]: Database information
            
        Raises:
            RuntimeError: If engine is not initialized
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized")
        
        try:
            with self.engine.connect() as conn:
                version = conn.execute(text("SELECT version()")).scalar()
                
                # Check if pgvector is available (PostgreSQL only)
                pgvector_available = False
                if "postgresql" in self.engine.url.drivername:
                    try:
                        conn.execute(text("SELECT 'dummy'::vector"))
                        pgvector_available = True
                    except SQLAlchemyError:
                        pgvector_available = False
                
                return {
                    "version": version,
                    "driver": self.engine.url.drivername,
                    "pgvector_available": pgvector_available,
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow
                }
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {
                "error": str(e),
                "driver": self.engine.url.drivername if self.engine else "unknown"
            } 

# Global connection manager instance
_connection_manager = None

def get_connection_manager() -> DatabaseConnectionManager:
    """
    Get the global connection manager instance.
    
    Returns:
        DatabaseConnectionManager: The global connection manager instance
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = DatabaseConnectionManager()
    return _connection_manager

def initialize_database_connection() -> Engine:
    """
    Initialize the database connection using the global connection manager.
    
    Returns:
        Engine: SQLAlchemy engine instance
    """
    manager = get_connection_manager()
    return manager.initialize_connection() 