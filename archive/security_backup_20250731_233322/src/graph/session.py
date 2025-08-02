"""Neo4j session/driver factory.

This module centralises Neo4j connectivity so that **all** graph queries go
through the same connection pool.  Other packages should *only* import
`get_driver()` / `get_session()` rather than instantiating their own `neo4j.Driver`.

Environment variables
---------------------
NEO4J_URI   bolt URI (e.g. ``bolt://localhost:7687``)
NEO4J_USER  database username (defaults to ``neo4j``)
NEO4J_PASS  password
NEO4J_ENCRYPTED  "0" to disable encryption (default ``1``)

The driver is lazily initialised on first access to avoid slowing unit tests
when the graph is not required.
"""
from __future__ import annotations

import os
import threading
import logging
from contextlib import contextmanager
from typing import Generator, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Neo4j driver, but don't fail if it's not available
try:
    from neo4j import GraphDatabase, Driver, Session
    NEO4J_AVAILABLE = True
    logger.info("Neo4j driver available")
except ImportError:
    logger.warning("Neo4j driver not available. Graph database functionality will be disabled.")
    NEO4J_AVAILABLE = False
    # Create dummy classes to avoid errors
    class Driver: pass
    class Session: pass
    class GraphDatabase:
        @staticmethod
        def driver(*args, **kwargs): 
            return None

__all__ = ["get_driver", "get_session", "driver_liveness_check", "is_neo4j_available"]

# ---------------------------------------------------------------------------
# Internal state â€“ lazily created singleton
# ---------------------------------------------------------------------------

_driver: Optional[Driver] = None
_lock = threading.Lock()
_connection_error = None
_connection_attempted = False


def _create_driver() -> Optional[Driver]:
    """Instantiate a Neo4j :class:`neo4j.Driver` from env vars."""
    global _connection_error, _connection_attempted
    
    if not NEO4J_AVAILABLE:
        _connection_error = "Neo4j driver not available"
        _connection_attempted = True
        return None
        
    # Check if required environment variables are set
    uri = os.getenv("NEO4J_URI")
    if not uri:
        logger.warning("NEO4J_URI environment variable not set. Neo4j integration disabled.")
        _connection_error = "NEO4J_URI environment variable not set"
        _connection_attempted = True
        return None
        
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASS")
    if not pwd:
        logger.warning("NEO4J_PASS environment variable not set. Neo4j integration disabled.")
        _connection_error = "NEO4J_PASS environment variable not set"
        _connection_attempted = True
        return None
    
    encrypted = os.getenv("NEO4J_ENCRYPTED", "1") != "0"
    
    try:
        # Try to create the driver with connection timeout
        driver = GraphDatabase.driver(
            uri,
            auth=(user, pwd),
            encrypted=encrypted,
            # tuned defaults â€“ tweak as needed
            max_connection_lifetime=1800,  # seconds
            max_connection_pool_size=int(os.getenv("NEO4J_POOL_SIZE", "20")),
            connection_timeout=5.0,  # 5 second timeout for connection attempts
        )
        
        # Test the connection
        with driver.session() as session:
            session.run("RETURN 1").single()
            
        logger.info(f"Successfully connected to Neo4j at {uri}")
        _connection_attempted = True
        return driver
    except Exception as e:
        logger.warning(f"Failed to connect to Neo4j: {e}")
        _connection_error = str(e)
        _connection_attempted = True
        return None


def get_driver() -> Optional[Driver]:
    """Return the lazily-initialised Neo4j driver (singleton).
    
    Returns None if Neo4j is not available or connection fails.
    """
    global _driver
    if _driver is None and not _connection_attempted:
        with _lock:
            if _driver is None and not _connection_attempted:  # double-checked locking
                _driver = _create_driver()
    return _driver


@contextmanager
def get_session(**kwargs) -> Generator[Optional[Session], None, None]:
    """Context manager yielding a Neo4j session.

    Example
    -------
    >>> from src.graph.session import get_session
    >>> with get_session() as s:
    ...     if s:
    ...         s.run("RETURN 1").single()
    ...     else:
    ...         print("Neo4j not available")
    """
    driver = get_driver()
    if driver is None:
        # Yield None if driver is not available
        yield None
        return
        
    try:
        session: Session = driver.session(**kwargs)
        try:
            yield session
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Failed to create Neo4j session: {e}")
        yield None


# ---------------------------------------------------------------------------
# Health / readiness check helpers
# ---------------------------------------------------------------------------

def driver_liveness_check(timeout: float = 3.0) -> bool:
    """Quick *RETURN 1* to confirm the database is reachable."""
    if not NEO4J_AVAILABLE:
        return False
        
    try:
        with get_session() as s:
            if s is None:
                return False
                
            record = s.run("RETURN 1 AS ok").single()
            return record["ok"] == 1
    except Exception:
        return False


def is_neo4j_available() -> bool:
    """Check if Neo4j is available.
    
    Returns True if the Neo4j driver is available and connected,
    False otherwise.
    """
    return NEO4J_AVAILABLE and get_driver() is not None


def get_connection_status() -> dict:
    """Get Neo4j connection status information."""
    return {
        "driver_available": NEO4J_AVAILABLE,
        "connection_attempted": _connection_attempted,
        "connection_error": _connection_error,
        "connected": is_neo4j_available(),
        "uri": os.getenv("NEO4J_URI", "not set")
    }
