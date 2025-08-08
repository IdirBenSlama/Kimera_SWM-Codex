"""
KIMERA SWM - DATABASE MANAGER
=============================

The Database Manager provides unified access to multiple database backends
for complex queries, analytics, and advanced data management beyond the
basic vault system. It handles structured queries, aggregations, and
real-time analytics across the knowledge base.

This is the system's advanced query and analytics layer.
"""

import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ..data_structures.geoid_state import GeoidProcessingState, GeoidState, GeoidType
from ..data_structures.scar_state import ScarSeverity, ScarState, ScarStatus, ScarType

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types"""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    ELASTICSEARCH = "elasticsearch"
    REDIS = "redis"


class QueryType(Enum):
    """Types of database queries"""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    SEARCH = "search"
    ANALYTICS = "analytics"


@dataclass
class DatabaseConfiguration:
    """Auto-generated class."""
    pass
    """Configuration for database connections"""

    db_type: DatabaseType
    connection_string: str
    pool_size: int = 10
    timeout: int = 30
    auto_commit: bool = True
    enable_ssl: bool = False
    schema_name: Optional[str] = None
    index_settings: Dict[str, Any] = None


@dataclass
class QueryResult:
    """Auto-generated class."""
    pass
    """Result of a database query"""

    success: bool
    data: List[Dict[str, Any]]
    row_count: int
    execution_time: float
    query_type: QueryType
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class AnalyticsResult:
    """Auto-generated class."""
    pass
    """Result of analytics query"""

    metric_name: str
    value: Union[int, float, Dict[str, Any]]
    aggregation_type: str
    time_range: Optional[Tuple[datetime, datetime]]
    filters: Dict[str, Any]
    metadata: Dict[str, Any]


class DatabaseConnection(ABC):
    """Abstract base class for database connections"""

    @abstractmethod
    def connect(self) -> bool:
        """Establish database connection"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection"""
        pass

    @abstractmethod
    def execute_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> QueryResult:
        """Execute a database query"""
        pass

    @abstractmethod
    def execute_analytics(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> AnalyticsResult:
        """Execute analytics query"""
        pass

    @abstractmethod
    def create_indexes(self) -> bool:
        """Create necessary indexes"""
        pass

    @abstractmethod
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        pass


class SQLiteConnection(DatabaseConnection):
    """SQLite database connection"""

    def __init__(self, config: DatabaseConfiguration):
        self.config = config
        self.connection = None
        self.lock = threading.RLock()
        self.is_connected = False

        # Extract database path from connection string
        if config.connection_string.startswith("sqlite://"):
            self.db_path = config.connection_string[9:]
        else:
            self.db_path = config.connection_string

    def connect(self) -> bool:
        """Establish SQLite connection"""
        try:
            with self.lock:
                self.connection = sqlite3.connect(
                    self.db_path, timeout=self.config.timeout, check_same_thread=False
                )
                self.connection.row_factory = (
                    sqlite3.Row
                )  # Enable column access by name
                self.is_connected = True

                # Initialize schema
                self._initialize_schema()

                logger.info(f"Connected to SQLite database: {self.db_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {str(e)}")
            return False

    def disconnect(self) -> None:
        """Close SQLite connection"""
        try:
            with self.lock:
                if self.connection:
                    self.connection.close()
                    self.connection = None
                    self.is_connected = False
                    logger.info("Disconnected from SQLite database")

        except Exception as e:
            logger.error(f"Error disconnecting from SQLite: {str(e)}")

    def execute_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> QueryResult:
        """Execute SQLite query"""
        start_time = time.time()

        try:
            with self.lock:
                if not self.is_connected:
                    self.connect()

                cursor = self.connection.cursor()

                if parameters:
                    cursor.execute(query, parameters)
                else:
                    cursor.execute(query)

                # Determine query type
                query_type = self._determine_query_type(query)

                # Fetch results for SELECT queries
                if query_type == QueryType.SELECT:
                    rows = cursor.fetchall()
                    data = [dict(row) for row in rows]
                    row_count = len(data)
                else:
                    data = []
                    row_count = cursor.rowcount
                    if self.config.auto_commit:
                        self.connection.commit()

                execution_time = time.time() - start_time

                return QueryResult(
                    success=True,
                    data=data,
                    row_count=row_count,
                    execution_time=execution_time,
                    query_type=query_type,
                    metadata={"database_type": "sqlite"},
                )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"SQLite query error: {error_msg}")

            return QueryResult(
                success=False
                data=[],
                row_count=0
                execution_time=execution_time
                query_type=QueryType.SELECT
                metadata={"database_type": "sqlite"},
                error_message=error_msg
            )

    def execute_analytics(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> AnalyticsResult:
        """Execute analytics query"""
        result = self.execute_query(query, parameters)

        # Extract analytics information from result
        if result.success and result.data:
            # Assume first row contains the analytics result
            first_row = result.data[0]

            # Extract metric name and value from query result
            if "count" in first_row:
                metric_name = "count"
                value = first_row["count"]
                aggregation_type = "count"
            elif "avg" in first_row:
                metric_name = "average"
                value = first_row["avg"]
                aggregation_type = "average"
            elif "sum" in first_row:
                metric_name = "sum"
                value = first_row["sum"]
                aggregation_type = "sum"
            else:
                # Use all data as complex result
                metric_name = "complex_analytics"
                value = result.data
                aggregation_type = "complex"

            return AnalyticsResult(
                metric_name=metric_name
                value=value
                aggregation_type=aggregation_type
                time_range=None
                filters=parameters or {},
                metadata={"execution_time": result.execution_time},
            )
        else:
            return AnalyticsResult(
                metric_name="error",
                value=0
                aggregation_type="error",
                time_range=None
                filters=parameters or {},
                metadata={"error": result.error_message},
            )

    def create_indexes(self) -> bool:
        """Create necessary indexes for performance"""
        indexes = [
            # Geoid indexes
            "CREATE INDEX IF NOT EXISTS idx_geoids_type ON geoids(geoid_type)",
            "CREATE INDEX IF NOT EXISTS idx_geoids_state ON geoids(processing_state)",
            "CREATE INDEX IF NOT EXISTS idx_geoids_created ON geoids(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_geoids_coherence ON geoids(coherence_score)",
            "CREATE INDEX IF NOT EXISTS idx_geoids_energy ON geoids(cognitive_energy)",
            # SCAR indexes
            "CREATE INDEX IF NOT EXISTS idx_scars_type ON scars(scar_type)",
            "CREATE INDEX IF NOT EXISTS idx_scars_severity ON scars(severity)",
            "CREATE INDEX IF NOT EXISTS idx_scars_status ON scars(status)",
            "CREATE INDEX IF NOT EXISTS idx_scars_created ON scars(created_at)",
            # Relationship indexes
            "CREATE INDEX IF NOT EXISTS idx_geoid_relations_source ON geoid_relations(source_geoid_id)",
            "CREATE INDEX IF NOT EXISTS idx_geoid_relations_target ON geoid_relations(target_geoid_id)",
            "CREATE INDEX IF NOT EXISTS idx_scar_geoids_scar ON scar_geoids(scar_id)",
            "CREATE INDEX IF NOT EXISTS idx_scar_geoids_geoid ON scar_geoids(geoid_id)",
        ]

        try:
            for index_sql in indexes:
                result = self.execute_query(index_sql)
                if not result.success:
                    logger.warning(f"Failed to create index: {index_sql}")
                    return False

            logger.info("Successfully created all database indexes")
            return True

        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            return False

    def get_schema_info(self) -> Dict[str, Any]:
        """Get SQLite schema information"""
        try:
            # Get table information
            tables_result = self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )

            schema_info = {
                "database_type": "sqlite",
                "tables": [],
                "indexes": [],
                "total_size": 0
            }

            if tables_result.success:
                for row in tables_result.data:
                    table_name = row["name"]
                    schema_info["tables"].append(table_name)

            # Get index information
            indexes_result = self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )

            if indexes_result.success:
                schema_info["indexes"] = [row["name"] for row in indexes_result.data]

            return schema_info

        except Exception as e:
            logger.error(f"Error getting schema info: {str(e)}")
            return {"error": str(e)}

    def _initialize_schema(self) -> None:
        """Initialize database schema"""
        schema_queries = [
            # Geoids table
            """
            CREATE TABLE IF NOT EXISTS geoids (
                geoid_id TEXT PRIMARY KEY
                geoid_type TEXT NOT NULL
                processing_state TEXT NOT NULL
                coherence_score REAL
                cognitive_energy REAL
                has_semantic_state BOOLEAN
                has_symbolic_state BOOLEAN
                has_thermodynamic_state BOOLEAN
                processing_depth INTEGER
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                metadata_json TEXT
            )
            ""","""
            # SCARs table
            """
            CREATE TABLE IF NOT EXISTS scars (
                scar_id TEXT PRIMARY KEY
                scar_type TEXT NOT NULL
                severity TEXT NOT NULL
                status TEXT NOT NULL
                title TEXT
                description TEXT
                root_cause TEXT
                resolution_summary TEXT
                impact_score REAL
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                resolved_at TIMESTAMP
                metadata_json TEXT
            )
            ""","""
            # Geoid relationships table
            """
            CREATE TABLE IF NOT EXISTS geoid_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT
                source_geoid_id TEXT NOT NULL
                target_geoid_id TEXT NOT NULL
                relation_type TEXT NOT NULL
                strength REAL DEFAULT 1.0
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                FOREIGN KEY (source_geoid_id) REFERENCES geoids(geoid_id),
                FOREIGN KEY (target_geoid_id) REFERENCES geoids(geoid_id)
            )
            ""","""
            # SCAR-Geoid associations
            """
            CREATE TABLE IF NOT EXISTS scar_geoids (
                id INTEGER PRIMARY KEY AUTOINCREMENT
                scar_id TEXT NOT NULL
                geoid_id TEXT NOT NULL
                relationship_type TEXT DEFAULT 'affected',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                FOREIGN KEY (scar_id) REFERENCES scars(scar_id),
                FOREIGN KEY (geoid_id) REFERENCES geoids(geoid_id)
            )
            ""","""
            # System metrics table
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT
                metric_name TEXT NOT NULL
                metric_value REAL NOT NULL
                metric_type TEXT NOT NULL
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                metadata_json TEXT
            )
            ""","""
        ]

        for query in schema_queries:
            result = self.execute_query(query)
            if not result.success:
                logger.error(f"Failed to create schema: {result.error_message}")

    def _determine_query_type(self, query: str) -> QueryType:
        """Determine the type of SQL query"""
        query_lower = query.strip().lower()

        if query_lower.startswith("select"):
            return QueryType.SELECT
        elif query_lower.startswith("insert"):
            return QueryType.INSERT
        elif query_lower.startswith("update"):
            return QueryType.UPDATE
        elif query_lower.startswith("delete"):
            return QueryType.DELETE
        else:
            return QueryType.SELECT
class DatabaseManager:
    """Auto-generated class."""
    pass
    """
    Database Manager - Advanced Query and Analytics Layer
    =====================================================

    The DatabaseManager provides unified access to multiple database backends
    for complex queries, analytics, and advanced data management. It handles
    structured queries, aggregations, and real-time analytics across the
    knowledge base.

    Key Features:
    - Multi-database backend support
    - Complex query execution
    - Real-time analytics and aggregations
    - Performance monitoring and optimization
    - Schema management and migrations
    """

    def __init__(self, config: DatabaseConfiguration):
        self.config = config
        self.connection = self._create_connection(config)
        self.query_cache = {}
        self.performance_metrics = {
            "total_queries": 0
            "average_query_time": 0.0
            "cache_hit_rate": 0.0
            "error_rate": 0.0
        }

        # Connect to database
        if not self.connection.connect():
            raise ConnectionError(
                f"Failed to connect to {config.db_type.value} database"
            )

        # Create indexes for performance
        self.connection.create_indexes()

        logger.info(f"DatabaseManager initialized with {config.db_type.value}")

    def _create_connection(self, config: DatabaseConfiguration) -> DatabaseConnection:
        """Create database connection based on configuration"""
        if config.db_type == DatabaseType.SQLITE:
            return SQLiteConnection(config)
        else:
            raise ValueError(f"Unsupported database type: {config.db_type}")

    def store_geoid_metadata(self, geoid: GeoidState) -> bool:
        """Store geoid metadata in database for querying"""
        try:
            query = """
                INSERT OR REPLACE INTO geoids 
                (geoid_id, geoid_type, processing_state, coherence_score
                 cognitive_energy, has_semantic_state, has_symbolic_state
                 has_thermodynamic_state, processing_depth, metadata_json)
                VALUES (:geoid_id, :geoid_type, :processing_state, :coherence_score
                        :cognitive_energy, :has_semantic_state, :has_symbolic_state
                        :has_thermodynamic_state, :processing_depth, :metadata_json)
            """

            parameters = {
                "geoid_id": geoid.geoid_id
                "geoid_type": geoid.geoid_type.value
                "processing_state": geoid.processing_state.value
                "coherence_score": geoid.coherence_score
                "cognitive_energy": geoid.cognitive_energy
                "has_semantic_state": geoid.semantic_state is not None
                "has_symbolic_state": geoid.symbolic_state is not None
                "has_thermodynamic_state": geoid.thermodynamic is not None
                "processing_depth": geoid.metadata.processing_depth
                "metadata_json": json.dumps(
                    {
                        "source_engine": geoid.metadata.source_engine
                        "parent_geoids": geoid.metadata.parent_geoids
                        "child_geoids": geoid.metadata.child_geoids
                    }
                ),
            }

            result = self.connection.execute_query(query, parameters)
            return result.success

        except Exception as e:
            logger.error(f"Error storing geoid metadata: {str(e)}")
            return False

    def store_scar_metadata(self, scar: ScarState) -> bool:
        """Store SCAR metadata in database for querying"""
        try:
            query = """
                INSERT OR REPLACE INTO scars 
                (scar_id, scar_type, severity, status, title, description
                 root_cause, resolution_summary, impact_score, metadata_json)
                VALUES (:scar_id, :scar_type, :severity, :status, :title, :description
                        :root_cause, :resolution_summary, :impact_score, :metadata_json)
            """

            parameters = {
                "scar_id": scar.scar_id
                "scar_type": scar.scar_type.value
                "severity": scar.severity.value
                "status": scar.status.value
                "title": scar.title
                "description": scar.description
                "root_cause": scar.root_cause
                "resolution_summary": scar.resolution_summary
                "impact_score": scar.calculate_impact_score(),
                "metadata_json": json.dumps(
                    {
                        "affected_geoids": scar.affected_geoids
                        "affected_engines": scar.affected_engines
                        "evidence_count": len(scar.evidence),
                        "resolution_actions_count": len(scar.resolution_actions),
                    }
                ),
            }

            result = self.connection.execute_query(query, parameters)

            # Store SCAR-geoid relationships
            if result.success:
                for geoid_id in scar.affected_geoids:
                    self._store_scar_geoid_relation(scar.scar_id, geoid_id, "affected")

            return result.success

        except Exception as e:
            logger.error(f"Error storing SCAR metadata: {str(e)}")
            return False

    def query_geoids(
        self, filters: Dict[str, Any] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query geoids with filters"""
        try:
            where_clauses = []
            parameters = {}

            if filters:
                if "geoid_type" in filters:
                    where_clauses.append("geoid_type = :geoid_type")
                    parameters["geoid_type"] = filters["geoid_type"]

                if "processing_state" in filters:
                    where_clauses.append("processing_state = :processing_state")
                    parameters["processing_state"] = filters["processing_state"]

                if "min_coherence" in filters:
                    where_clauses.append("coherence_score >= :min_coherence")
                    parameters["min_coherence"] = filters["min_coherence"]

                if "max_coherence" in filters:
                    where_clauses.append("coherence_score <= :max_coherence")
                    parameters["max_coherence"] = filters["max_coherence"]

                if "min_energy" in filters:
                    where_clauses.append("cognitive_energy >= :min_energy")
                    parameters["min_energy"] = filters["min_energy"]

                if "has_semantic_state" in filters:
                    where_clauses.append("has_semantic_state = :has_semantic")
                    parameters["has_semantic"] = filters["has_semantic_state"]

            where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            query = f"""
                SELECT * FROM geoids 
                {where_sql}
                ORDER BY created_at DESC 
                LIMIT {limit}
            """

            result = self.connection.execute_query(query, parameters)
            return result.data if result.success else []

        except Exception as e:
            logger.error(f"Error querying geoids: {str(e)}")
            return []

    def query_scars(
        self, filters: Dict[str, Any] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query SCARs with filters"""
        try:
            where_clauses = []
            parameters = {}

            if filters:
                if "scar_type" in filters:
                    where_clauses.append("scar_type = :scar_type")
                    parameters["scar_type"] = filters["scar_type"]

                if "severity" in filters:
                    where_clauses.append("severity = :severity")
                    parameters["severity"] = filters["severity"]

                if "status" in filters:
                    where_clauses.append("status = :status")
                    parameters["status"] = filters["status"]

                if "min_impact" in filters:
                    where_clauses.append("impact_score >= :min_impact")
                    parameters["min_impact"] = filters["min_impact"]

            where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

            query = f"""
                SELECT * FROM scars 
                {where_sql}
                ORDER BY created_at DESC 
                LIMIT {limit}
            """

            result = self.connection.execute_query(query, parameters)
            return result.data if result.success else []

        except Exception as e:
            logger.error(f"Error querying SCARs: {str(e)}")
            return []

    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        analytics = {}

        try:
            # Geoid analytics
            geoid_stats = self.connection.execute_analytics(
                """
                SELECT 
                    COUNT(*) as total_geoids
                    AVG(coherence_score) as avg_coherence
                    AVG(cognitive_energy) as avg_energy
                    MAX(processing_depth) as max_depth
                FROM geoids
            """
            )

            if geoid_stats.aggregation_type != "error":
                analytics["geoids"] = geoid_stats.value

            # SCAR analytics
            scar_stats = self.connection.execute_analytics(
                """
                SELECT 
                    COUNT(*) as total_scars
                    AVG(impact_score) as avg_impact
                    COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_count
                FROM scars
            """
            )

            if scar_stats.aggregation_type != "error":
                analytics["scars"] = scar_stats.value

            # Type distributions
            geoid_types = self.connection.execute_query(
                """
                SELECT geoid_type, COUNT(*) as count 
                FROM geoids 
                GROUP BY geoid_type
            """
            )

            if geoid_types.success:
                analytics["geoid_type_distribution"] = {
                    row["geoid_type"]: row["count"] for row in geoid_types.data
                }

            scar_types = self.connection.execute_query(
                """
                SELECT scar_type, COUNT(*) as count 
                FROM scars 
                GROUP BY scar_type
            """
            )

            if scar_types.success:
                analytics["scar_type_distribution"] = {
                    row["scar_type"]: row["count"] for row in scar_types.data
                }

            return analytics

        except Exception as e:
            logger.error(f"Error getting system analytics: {str(e)}")
            return {"error": str(e)}

    def _store_scar_geoid_relation(
        self, scar_id: str, geoid_id: str, relationship_type: str
    ) -> bool:
        """Store SCAR-geoid relationship"""
        try:
            query = """
                INSERT OR REPLACE INTO scar_geoids 
                (scar_id, geoid_id, relationship_type)
                VALUES (:scar_id, :geoid_id, :relationship_type)
            """

            parameters = {
                "scar_id": scar_id
                "geoid_id": geoid_id
                "relationship_type": relationship_type
            }

            result = self.connection.execute_query(query, parameters)
            return result.success

        except Exception as e:
            logger.error(f"Error storing SCAR-geoid relation: {str(e)}")
            return False

    def close(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.disconnect()


# Global database manager instance
_global_db_manager: Optional[DatabaseManager] = None


def get_global_database_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _global_db_manager
    if _global_db_manager is None:
        # Default SQLite configuration
        config = DatabaseConfiguration(
            db_type=DatabaseType.SQLITE, connection_string="sqlite://kimera_system.db"
        )
        _global_db_manager = DatabaseManager(config)
    return _global_db_manager


def initialize_database_manager(config: DatabaseConfiguration) -> DatabaseManager:
    """Initialize the global database manager with custom configuration"""
    global _global_db_manager
    _global_db_manager = DatabaseManager(config)
    return _global_db_manager


# Convenience functions
def query_geoids_by_type(geoid_type: GeoidType) -> List[Dict[str, Any]]:
    """Convenience function to query geoids by type"""
    db_manager = get_global_database_manager()
    return db_manager.query_geoids({"geoid_type": geoid_type.value})


def query_active_scars() -> List[Dict[str, Any]]:
    """Convenience function to query active SCARs"""
    db_manager = get_global_database_manager()
    return db_manager.query_scars({"status": "active"})


def get_system_health_metrics() -> Dict[str, Any]:
    """Convenience function to get system health metrics"""
    db_manager = get_global_database_manager()
    return db_manager.get_system_analytics()
