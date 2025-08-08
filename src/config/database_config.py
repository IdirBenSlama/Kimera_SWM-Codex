"""
Unified Database Configuration for KIMERA SWM
=============================================

Single source of truth for all database connectivity and configuration.
Supports PostgreSQL (production) and SQLite (development/testing).
"""

import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)
class DatabaseConfig:
    """Unified database configuration"""

    def __init__(self):
        self.database_url = self._get_database_url()
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self.scoped_session: Optional[scoped_session] = None
        self._initialized = False

    def force_reload_config(self) -> None:
        """Force reload configuration from environment"""
        self._initialized = False
        self.database_url = self._get_database_url()
        self.engine = None
        self.session_factory = None
        self.scoped_session = None
        logger.info(f"Database config reloaded: {self.database_url}")

    def _get_database_url(self) -> str:
        """Get database URL from environment with fallbacks"""
        import importlib
        import os

        # Reload environment variables
        if hasattr(os, "environ"):
            os.environ.reload = True  # Force reload

        # Priority order for database URL
        url_candidates = [
            os.getenv("KIMERA_DATABASE_URL"),
            os.getenv("DATABASE_URL"),
            os.getenv("POSTGRESQL_URL"),
            "sqlite:///kimera_swm.db",  # Default fallback
        ]

        for url in url_candidates:
            if url:
                parsed = urlparse(url)
                if parsed.scheme in ["postgresql", "sqlite"]:
                    logger.info(
                        f"Using database: {parsed.scheme}://{parsed.netloc}/{parsed.path}"
                    )
                    return url

        # Should never reach here, but safety fallback
        return "sqlite:///kimera_swm.db"

    def _configure_engine_for_postgresql(self, url: str) -> Engine:
        """Configure engine optimally for PostgreSQL"""
        return create_engine(
            url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False,
            future=True,
        )

    def _configure_engine_for_sqlite(self, url: str) -> Engine:
        """Configure engine optimally for SQLite"""
        return create_engine(
            url,
            pool_pre_ping=True,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )

    def initialize(self) -> bool:
        """Initialize database connection and session factory"""
        if self._initialized:
            return True

        try:
            # Create engine based on database type
            parsed_url = urlparse(self.database_url)

            if parsed_url.scheme == "postgresql":
                self.engine = self._configure_engine_for_postgresql(self.database_url)

                # Test PostgreSQL specific features
                with self.engine.connect() as conn:
                    try:
                        # Check for pgvector extension
                        conn.execute(text("SELECT 'test'::vector"))
                        logger.info("✅ pgvector extension available")
                    except Exception:
                        logger.warning("⚠️ pgvector extension not available")

            elif parsed_url.scheme == "sqlite":
                self.engine = self._configure_engine_for_sqlite(self.database_url)
                logger.info("✅ SQLite database configured")

            else:
                raise ValueError(f"Unsupported database scheme: {parsed_url.scheme}")

            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.fetchone()[0] == 1

            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )

            # Create scoped session
            self.scoped_session = scoped_session(self.session_factory)

            self._initialized = True
            logger.info("✅ Database initialization successful")
            return True

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False

    def get_engine(self) -> Engine:
        """Get the database engine"""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Database not initialized")
        return self.engine

    def get_session(self):
        """Get a database session (use as context manager)"""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Database not initialized")

        session = self.scoped_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session_factory(self) -> sessionmaker:
        """Get the session factory"""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Database not initialized")
        return self.session_factory

    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        if not self._initialized:
            return {"status": "not_initialized"}

        try:
            with self.engine.connect() as conn:
                # Basic connectivity test
                result = conn.execute(text("SELECT 1"))
                assert result.fetchone()[0] == 1

                # Database-specific checks
                parsed_url = urlparse(self.database_url)

                if parsed_url.scheme == "postgresql":
                    # PostgreSQL specific checks
                    version_result = conn.execute(text("SELECT version()"))
                    version = version_result.fetchone()[0]

                    return {
                        "status": "healthy",
                        "database_type": "postgresql",
                        "version": version,
                        "url": f"postgresql://{parsed_url.netloc}/{parsed_url.path}",
                    }

                elif parsed_url.scheme == "sqlite":
                    # SQLite specific checks
                    version_result = conn.execute(text("SELECT sqlite_version()"))
                    version = version_result.fetchone()[0]

                    return {
                        "status": "healthy",
                        "database_type": "sqlite",
                        "version": version,
                        "file_path": parsed_url.path,
                    }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def close(self) -> None:
        """Close database connections"""
        if self.scoped_session:
            self.scoped_session.remove()

        if self.engine:
            self.engine.dispose()

        self._initialized = False
        logger.info("✅ Database connections closed")


# Placeholder classes for backward compatibility
class GeoidDB:
    """Auto-generated class."""
    pass
    """GeoidDB placeholder for backward compatibility"""

    def __init__(self):
        pass
class ScarDB:
    """Auto-generated class."""
    pass
    """ScarDB placeholder for backward compatibility"""

    def __init__(self):
        pass
class InsightDB:
    """Auto-generated class."""
    pass
    """InsightDB placeholder for backward compatibility"""

    def __init__(self):
        pass


# Global database configuration instance
db_config = DatabaseConfig()


# Compatibility functions for existing code
def initialize_database() -> bool:
    """Initialize database (compatibility function)"""
    return db_config.initialize()


def get_engine() -> Engine:
    """Get database engine (compatibility function)"""
    return db_config.get_engine()


def get_db():
    """Get database session (compatibility function)"""
    return db_config.get_session()


def get_session_factory() -> sessionmaker:
    """Get session factory (compatibility function)"""
    return db_config.get_session_factory()


def database_health_check() -> Dict[str, Any]:
    """Perform database health check (compatibility function)"""
    return db_config.health_check()


# Legacy compatibility
SessionLocal = None


def get_db_status():
    """Legacy function for compatibility"""
    return database_health_check()


def create_tables(engine=None):
    """Create database tables (compatibility function)"""
    try:
        if engine is None:
            engine = db_config.get_engine()

        # Import unified schema
        from .unified_schema import create_tables as schema_create_tables

        return schema_create_tables(engine)

    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False


def get_db_status():
    """Get database status (compatibility function)"""
    return db_config.health_check()


# Event listeners for connection optimization
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for optimal performance"""
    if "sqlite" in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        # Performance optimizations
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()


@event.listens_for(Engine, "connect")
def set_postgresql_settings(dbapi_connection, connection_record):
    """Set PostgreSQL connection settings"""
    if hasattr(dbapi_connection, "server_version"):  # PostgreSQL connection
        cursor = dbapi_connection.cursor()
        # Optimize for KIMERA workloads
        cursor.execute("SET statement_timeout = '30s'")
        cursor.execute("SET lock_timeout = '10s'")
        cursor.close()
