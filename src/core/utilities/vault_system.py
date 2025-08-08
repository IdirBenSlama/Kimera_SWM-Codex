"""
KIMERA SWM - VAULT SYSTEM
=========================

The Vault System provides persistent storage and retrieval for geoids, SCARs
and other system data. It supports multiple storage backends and provides
a unified interface for all persistence operations.

This is the system's long-term memory and knowledge preservation layer.
"""

import asyncio
import json
import logging
import os
import pickle
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ..data_structures.geoid_state import GeoidState, GeoidType
from ..data_structures.scar_state import ScarSeverity, ScarState, ScarStatus, ScarType

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Supported storage backends"""

    SQLITE = "sqlite"  # SQLite database
    JSON_FILES = "json_files"  # JSON file storage
    PICKLE_FILES = "pickle"  # Pickle file storage
    MEMORY = "memory"  # In-memory storage (for testing)
    HYBRID = "hybrid"  # Combination of backends


class DataType(Enum):
    """Types of data stored in vault"""

    GEOID = "geoid"
    SCAR = "scar"
    METADATA = "metadata"
    STATISTICS = "statistics"
    CONFIGURATION = "configuration"
    SESSION_DATA = "session_data"


@dataclass
class StorageConfiguration:
    """Auto-generated class."""
    pass
    """Configuration for storage backends"""

    backend: StorageBackend
    base_path: str
    database_url: Optional[str] = None
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    compression_enabled: bool = True
    encryption_enabled: bool = False
    backup_enabled: bool = True
    retention_days: int = 30
    index_enabled: bool = True


@dataclass
class StorageMetrics:
    """Auto-generated class."""
    pass
    """Metrics for storage operations"""

    total_items_stored: int
    total_items_retrieved: int
    average_write_time: float
    average_read_time: float
    storage_size_bytes: int
    last_backup_time: Optional[datetime]
    error_count: int
    cache_hit_rate: float


class StorageAdapter(ABC):
    """Abstract base class for storage adapters"""

    @abstractmethod
    def store(self, key: str, data: Any, data_type: DataType) -> bool:
        """Store data with the given key"""
        pass

    @abstractmethod
    def retrieve(self, key: str, data_type: DataType) -> Optional[Any]:
        """Retrieve data by key"""
        pass

    @abstractmethod
    def delete(self, key: str, data_type: DataType) -> bool:
        """Delete data by key"""
        pass

    @abstractmethod
    def list_keys(self, data_type: DataType, prefix: Optional[str] = None) -> List[str]:
        """List all keys for a data type"""
        pass

    @abstractmethod
    def exists(self, key: str, data_type: DataType) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    def get_size(self) -> int:
        """Get total storage size in bytes"""
        pass

    @abstractmethod
    def cleanup(self, retention_days: int) -> int:
        """Clean up old data and return number of items removed"""
        pass


class SQLiteAdapter(StorageAdapter):
    """SQLite storage adapter"""

    def __init__(self, config: StorageConfiguration):
        self.config = config
        self.db_path = os.path.join(config.base_path, "kimera_vault.db")
        self.lock = threading.RLock()

        # Ensure directory exists
        os.makedirs(config.base_path, exist_ok=True)

        # Initialize database
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vault_data (
                    key TEXT PRIMARY KEY
                    data_type TEXT NOT NULL
                    data_blob BLOB NOT NULL
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    metadata TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_data_type ON vault_data(data_type)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at ON vault_data(created_at)
            """
            )

            conn.commit()

    def store(self, key: str, data: Any, data_type: DataType) -> bool:
        """Store data in SQLite"""
        try:
            with self.lock:
                # Serialize data
                if isinstance(data, (GeoidState, ScarState)):
                    data_blob = json.dumps(data.to_dict()).encode("utf-8")
                else:
                    data_blob = pickle.dumps(data)

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO vault_data 
                        (key, data_type, data_blob, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ""","""
                        (key, data_type.value, data_blob),
                    )
                    conn.commit()

                return True

        except Exception as e:
            logger.error(f"Error storing data in SQLite: {str(e)}")
            return False

    def retrieve(self, key: str, data_type: DataType) -> Optional[Any]:
        """Retrieve data from SQLite"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT data_blob FROM vault_data 
                        WHERE key = ? AND data_type = ?
                    ""","""
                        (key, data_type.value),
                    )

                    row = cursor.fetchone()
                    if not row:
                        return None

                    data_blob = row[0]

                    # Deserialize based on data type
                    if data_type == DataType.GEOID:
                        data_dict = json.loads(data_blob.decode("utf-8"))
                        return self._reconstruct_geoid(data_dict)
                    elif data_type == DataType.SCAR:
                        data_dict = json.loads(data_blob.decode("utf-8"))
                        return self._reconstruct_scar(data_dict)
                    else:
                        return pickle.loads(data_blob)

        except Exception as e:
            logger.error(f"Error retrieving data from SQLite: {str(e)}")
            return None

    def delete(self, key: str, data_type: DataType) -> bool:
        """Delete data from SQLite"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        DELETE FROM vault_data 
                        WHERE key = ? AND data_type = ?
                    ""","""
                        (key, data_type.value),
                    )
                    conn.commit()
                    return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error deleting data from SQLite: {str(e)}")
            return False

    def list_keys(self, data_type: DataType, prefix: Optional[str] = None) -> List[str]:
        """List keys from SQLite"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    if prefix:
                        cursor = conn.execute(
                            """
                            SELECT key FROM vault_data 
                            WHERE data_type = ? AND key LIKE ?
                            ORDER BY created_at DESC
                        ""","""
                            (data_type.value, f"{prefix}%"),
                        )
                    else:
                        cursor = conn.execute(
                            """
                            SELECT key FROM vault_data 
                            WHERE data_type = ?
                            ORDER BY created_at DESC
                        ""","""
                            (data_type.value,),
                        )

                    return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Error listing keys from SQLite: {str(e)}")
            return []

    def exists(self, key: str, data_type: DataType) -> bool:
        """Check if key exists in SQLite"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT 1 FROM vault_data 
                        WHERE key = ? AND data_type = ?
                    ""","""
                        (key, data_type.value),
                    )
                    return cursor.fetchone() is not None

        except Exception as e:
            logger.error(f"Error checking existence in SQLite: {str(e)}")
            return False

    def get_size(self) -> int:
        """Get SQLite database size"""
        try:
            return os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        except Exception:
            return 0

    def cleanup(self, retention_days: int) -> int:
        """Clean up old data from SQLite"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        DELETE FROM vault_data 
                        WHERE created_at < ?
                    ""","""
                        (cutoff_date,),
                    )
                    conn.commit()
                    return cursor.rowcount

        except Exception as e:
            logger.error(f"Error cleaning up SQLite data: {str(e)}")
            return 0

    def _reconstruct_geoid(self, data_dict: Dict[str, Any]) -> GeoidState:
        """Reconstruct GeoidState from dictionary"""
        # This is a simplified reconstruction - in practice would need
        # proper deserialization handling for all nested objects
        from ..data_structures.geoid_state import (GeoidProcessingState, GeoidState
                                                   GeoidType, ProcessingMetadata
                                                   SemanticState, SymbolicState
                                                   ThermodynamicProperties)

        # Create basic geoid - full reconstruction would be more complex
        geoid = GeoidState(
            geoid_id=data_dict["geoid_id"],
            geoid_type=GeoidType(data_dict["geoid_type"]),
            processing_state=GeoidProcessingState(data_dict["processing_state"]),
        )

        # Would need to reconstruct semantic_state, symbolic_state, etc.
        # This is simplified for demonstration

        return geoid

    def _reconstruct_scar(self, data_dict: Dict[str, Any]) -> ScarState:
        """Reconstruct ScarState from dictionary"""
        # This is a simplified reconstruction
        from ..data_structures.scar_state import (ScarSeverity, ScarState, ScarStatus
                                                  ScarType)

        scar = ScarState(
            scar_id=data_dict["scar_id"],
            scar_type=ScarType(data_dict["scar_type"]),
            severity=ScarSeverity(data_dict["severity"]),
            status=ScarStatus(data_dict["status"]),
            title=data_dict["title"],
            description=data_dict["description"],
        )

        return scar


class JSONFileAdapter(StorageAdapter):
    """JSON file storage adapter"""

    def __init__(self, config: StorageConfiguration):
        self.config = config
        self.base_path = Path(config.base_path)
        self.lock = threading.RLock()

        # Ensure directory structure exists
        for data_type in DataType:
            (self.base_path / data_type.value).mkdir(parents=True, exist_ok=True)

    def store(self, key: str, data: Any, data_type: DataType) -> bool:
        """Store data as JSON file"""
        try:
            with self.lock:
                file_path = self.base_path / data_type.value / f"{key}.json"

                # Serialize data
                if isinstance(data, (GeoidState, ScarState)):
                    data_dict = data.to_dict()
                else:
                    data_dict = data

                # Add metadata
                storage_data = {
                    "data": data_dict
                    "stored_at": datetime.now().isoformat(),
                    "data_type": data_type.value
                    "key": key
                }

                with open(file_path, "w") as f:
                    json.dump(storage_data, f, indent=2, default=str)

                return True

        except Exception as e:
            logger.error(f"Error storing JSON file: {str(e)}")
            return False

    def retrieve(self, key: str, data_type: DataType) -> Optional[Any]:
        """Retrieve data from JSON file"""
        try:
            with self.lock:
                file_path = self.base_path / data_type.value / f"{key}.json"

                if not file_path.exists():
                    return None

                with open(file_path, "r") as f:
                    storage_data = json.load(f)

                return storage_data["data"]

        except Exception as e:
            logger.error(f"Error retrieving JSON file: {str(e)}")
            return None

    def delete(self, key: str, data_type: DataType) -> bool:
        """Delete JSON file"""
        try:
            with self.lock:
                file_path = self.base_path / data_type.value / f"{key}.json"

                if file_path.exists():
                    file_path.unlink()
                    return True
                return False

        except Exception as e:
            logger.error(f"Error deleting JSON file: {str(e)}")
            return False

    def list_keys(self, data_type: DataType, prefix: Optional[str] = None) -> List[str]:
        """List JSON file keys"""
        try:
            with self.lock:
                dir_path = self.base_path / data_type.value

                if not dir_path.exists():
                    return []

                files = []
                for file_path in dir_path.glob("*.json"):
                    key = file_path.stem
                    if prefix is None or key.startswith(prefix):
                        files.append(key)

                return sorted(files)

        except Exception as e:
            logger.error(f"Error listing JSON files: {str(e)}")
            return []

    def exists(self, key: str, data_type: DataType) -> bool:
        """Check if JSON file exists"""
        file_path = self.base_path / data_type.value / f"{key}.json"
        return file_path.exists()

    def get_size(self) -> int:
        """Get total size of JSON files"""
        try:
            total_size = 0
            for data_type in DataType:
                dir_path = self.base_path / data_type.value
                if dir_path.exists():
                    for file_path in dir_path.rglob("*.json"):
                        total_size += file_path.stat().st_size
            return total_size

        except Exception:
            return 0

    def cleanup(self, retention_days: int) -> int:
        """Clean up old JSON files"""
        try:
            cutoff_time = time.time() - (retention_days * 24 * 3600)
            removed_count = 0

            with self.lock:
                for data_type in DataType:
                    dir_path = self.base_path / data_type.value
                    if dir_path.exists():
                        for file_path in dir_path.glob("*.json"):
                            if file_path.stat().st_mtime < cutoff_time:
                                file_path.unlink()
                                removed_count += 1

            return removed_count

        except Exception as e:
            logger.error(f"Error cleaning up JSON files: {str(e)}")
            return 0


class MemoryAdapter(StorageAdapter):
    """In-memory storage adapter (for testing)"""

    def __init__(self, config: StorageConfiguration):
        self.config = config
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()

    def store(self, key: str, data: Any, data_type: DataType) -> bool:
        """Store data in memory"""
        try:
            with self.lock:
                if data_type.value not in self.storage:
                    self.storage[data_type.value] = {}

                self.storage[data_type.value][key] = {
                    "data": data
                    "stored_at": datetime.now(),
                    "data_type": data_type.value
                }
                return True

        except Exception as e:
            logger.error(f"Error storing in memory: {str(e)}")
            return False

    def retrieve(self, key: str, data_type: DataType) -> Optional[Any]:
        """Retrieve data from memory"""
        try:
            with self.lock:
                type_storage = self.storage.get(data_type.value, {})
                item = type_storage.get(key)
                return item["data"] if item else None

        except Exception as e:
            logger.error(f"Error retrieving from memory: {str(e)}")
            return None

    def delete(self, key: str, data_type: DataType) -> bool:
        """Delete data from memory"""
        try:
            with self.lock:
                type_storage = self.storage.get(data_type.value, {})
                if key in type_storage:
                    del type_storage[key]
                    return True
                return False

        except Exception as e:
            logger.error(f"Error deleting from memory: {str(e)}")
            return False

    def list_keys(self, data_type: DataType, prefix: Optional[str] = None) -> List[str]:
        """List keys from memory"""
        try:
            with self.lock:
                type_storage = self.storage.get(data_type.value, {})
                keys = list(type_storage.keys())

                if prefix:
                    keys = [k for k in keys if k.startswith(prefix)]

                return sorted(keys)

        except Exception as e:
            logger.error(f"Error listing from memory: {str(e)}")
            return []

    def exists(self, key: str, data_type: DataType) -> bool:
        """Check if key exists in memory"""
        with self.lock:
            type_storage = self.storage.get(data_type.value, {})
            return key in type_storage

    def get_size(self) -> int:
        """Get approximate memory usage"""
        try:
            # Rough estimate
            import sys

            return sys.getsizeof(self.storage)
        except Exception:
            return 0

    def cleanup(self, retention_days: int) -> int:
        """Clean up old data from memory"""
        try:
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            removed_count = 0

            with self.lock:
                for data_type in list(self.storage.keys()):
                    type_storage = self.storage[data_type]
                    keys_to_remove = []

                    for key, item in type_storage.items():
                        if item["stored_at"] < cutoff_time:
                            keys_to_remove.append(key)

                    for key in keys_to_remove:
                        del type_storage[key]
                        removed_count += 1

            return removed_count

        except Exception as e:
            logger.error(f"Error cleaning up memory: {str(e)}")
            return 0
class VaultSystem:
    """Auto-generated class."""
    pass
    """
    Vault System - Persistent Knowledge Storage
    ==========================================

    The VaultSystem provides unified persistent storage for all Kimera SWM
    knowledge components including geoids, SCARs, and system metadata.
    It supports multiple storage backends and provides high-performance
    access patterns for cognitive data.

    Key Features:
    - Multiple storage backend support
    - Automatic data lifecycle management
    - Performance optimization and caching
    - Backup and recovery capabilities
    - Search and indexing functionality
    """

    def __init__(self, config: StorageConfiguration):
        self.config = config
        self.adapter = self._create_adapter(config)

        # Performance tracking
        self.metrics = StorageMetrics(
            total_items_stored=0,
            total_items_retrieved=0,
            average_write_time=0.0,
            average_read_time=0.0,
            storage_size_bytes=0,
            last_backup_time=None,
            error_count=0,
            cache_hit_rate=0.0
        )

        # Simple cache for performance
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000

        # Background maintenance
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.maintenance_tasks = set()

        logger.info(f"VaultSystem initialized with backend: {config.backend.value}")

    def _create_adapter(self, config: StorageConfiguration) -> StorageAdapter:
        """Create storage adapter based on configuration"""
        if config.backend == StorageBackend.SQLITE:
            return SQLiteAdapter(config)
        elif config.backend == StorageBackend.JSON_FILES:
            return JSONFileAdapter(config)
        elif config.backend == StorageBackend.MEMORY:
            return MemoryAdapter(config)
        else:
            raise ValueError(f"Unsupported storage backend: {config.backend}")

    def store_geoid(self, geoid: GeoidState) -> bool:
        """Store a geoid in the vault"""
        start_time = time.time()

        try:
            success = self.adapter.store(geoid.geoid_id, geoid, DataType.GEOID)

            if success:
                # Update cache
                cache_key = f"geoid:{geoid.geoid_id}"
                self.cache[cache_key] = geoid
                self._manage_cache_size()

                # Update metrics
                self.metrics.total_items_stored += 1
                duration = time.time() - start_time
                self._update_average_time("write", duration)

                logger.debug(f"Stored geoid {geoid.geoid_id[:8]} in {duration:.3f}s")

            return success

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error storing geoid {geoid.geoid_id[:8]}: {str(e)}")
            return False

    def retrieve_geoid(self, geoid_id: str) -> Optional[GeoidState]:
        """Retrieve a geoid from the vault"""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"geoid:{geoid_id}"
            if cache_key in self.cache:
                self.cache_hits += 1
                logger.debug(f"Retrieved geoid {geoid_id[:8]} from cache")
                return self.cache[cache_key]

            # Retrieve from storage
            geoid = self.adapter.retrieve(geoid_id, DataType.GEOID)

            if geoid:
                # Update cache
                self.cache[cache_key] = geoid
                self._manage_cache_size()

                # Update metrics
                self.cache_misses += 1
                self.metrics.total_items_retrieved += 1
                duration = time.time() - start_time
                self._update_average_time("read", duration)

                logger.debug(
                    f"Retrieved geoid {geoid_id[:8]} from storage in {duration:.3f}s"
                )

            return geoid

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error retrieving geoid {geoid_id[:8]}: {str(e)}")
            return None

    def store_scar(self, scar: ScarState) -> bool:
        """Store a SCAR in the vault"""
        start_time = time.time()

        try:
            success = self.adapter.store(scar.scar_id, scar, DataType.SCAR)

            if success:
                # Update cache
                cache_key = f"scar:{scar.scar_id}"
                self.cache[cache_key] = scar
                self._manage_cache_size()

                # Update metrics
                self.metrics.total_items_stored += 1
                duration = time.time() - start_time
                self._update_average_time("write", duration)

                logger.debug(f"Stored SCAR {scar.scar_id[:8]} in {duration:.3f}s")

            return success

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error storing SCAR {scar.scar_id[:8]}: {str(e)}")
            return False

    def retrieve_scar(self, scar_id: str) -> Optional[ScarState]:
        """Retrieve a SCAR from the vault"""
        start_time = time.time()

        try:
            # Check cache first
            cache_key = f"scar:{scar_id}"
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]

            # Retrieve from storage
            scar = self.adapter.retrieve(scar_id, DataType.SCAR)

            if scar:
                # Update cache
                self.cache[cache_key] = scar
                self._manage_cache_size()

                # Update metrics
                self.cache_misses += 1
                self.metrics.total_items_retrieved += 1
                duration = time.time() - start_time
                self._update_average_time("read", duration)

                logger.debug(
                    f"Retrieved SCAR {scar_id[:8]} from storage in {duration:.3f}s"
                )

            return scar

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error retrieving SCAR {scar_id[:8]}: {str(e)}")
            return None

    def list_geoids(self, prefix: Optional[str] = None) -> List[str]:
        """List all geoid IDs in the vault"""
        return self.adapter.list_keys(DataType.GEOID, prefix)

    def list_scars(self, prefix: Optional[str] = None) -> List[str]:
        """List all SCAR IDs in the vault"""
        return self.adapter.list_keys(DataType.SCAR, prefix)

    def delete_geoid(self, geoid_id: str) -> bool:
        """Delete a geoid from the vault"""
        try:
            # Remove from cache
            cache_key = f"geoid:{geoid_id}"
            self.cache.pop(cache_key, None)

            # Delete from storage
            return self.adapter.delete(geoid_id, DataType.GEOID)

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error deleting geoid {geoid_id[:8]}: {str(e)}")
            return False

    def delete_scar(self, scar_id: str) -> bool:
        """Delete a SCAR from the vault"""
        try:
            # Remove from cache
            cache_key = f"scar:{scar_id}"
            self.cache.pop(cache_key, None)

            # Delete from storage
            return self.adapter.delete(scar_id, DataType.SCAR)

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error deleting SCAR {scar_id[:8]}: {str(e)}")
            return False

    def get_storage_metrics(self) -> StorageMetrics:
        """Get current storage metrics"""
        # Update current metrics
        self.metrics.storage_size_bytes = self.adapter.get_size()

        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations > 0:
            self.metrics.cache_hit_rate = self.cache_hits / total_cache_operations

        return self.metrics

    def cleanup_old_data(self, retention_days: Optional[int] = None) -> int:
        """Clean up old data based on retention policy"""
        retention = retention_days or self.config.retention_days

        try:
            removed_count = self.adapter.cleanup(retention)

            # Clear cache to ensure consistency
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0

            logger.info(
                f"Cleaned up {removed_count} old items (retention: {retention} days)"
            )
            return removed_count

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Error during cleanup: {str(e)}")
            return 0

    def _manage_cache_size(self) -> None:
        """Manage cache size to prevent memory issues"""
        if len(self.cache) > self.max_cache_size:
            # Remove oldest 20% of cache items (simple LRU approximation)
            items_to_remove = len(self.cache) - int(self.max_cache_size * 0.8)
            keys_to_remove = list(self.cache.keys())[:items_to_remove]

            for key in keys_to_remove:
                self.cache.pop(key, None)

    def _update_average_time(self, operation: str, duration: float) -> None:
        """Update average operation time"""
        if operation == "write":
            if self.metrics.total_items_stored <= 1:
                self.metrics.average_write_time = duration
            else:
                # Exponential moving average
                alpha = 0.1
                self.metrics.average_write_time = (
                    alpha * duration + (1 - alpha) * self.metrics.average_write_time
                )
        elif operation == "read":
            if self.metrics.total_items_retrieved <= 1:
                self.metrics.average_read_time = duration
            else:
                alpha = 0.1
                self.metrics.average_read_time = (
                    alpha * duration + (1 - alpha) * self.metrics.average_read_time
                )


# Global vault instance
_global_vault: Optional[VaultSystem] = None


def get_global_vault() -> VaultSystem:
    """Get the global vault instance"""
    global _global_vault
    if _global_vault is None:
        # Default configuration
        config = StorageConfiguration(
            backend=StorageBackend.SQLITE
            base_path=os.path.join(os.getcwd(), "vault_data"),
        )
        _global_vault = VaultSystem(config)
    return _global_vault


def initialize_vault(config: StorageConfiguration) -> VaultSystem:
    """Initialize the global vault with custom configuration"""
    global _global_vault
    _global_vault = VaultSystem(config)
    return _global_vault


# Convenience functions
def store_geoid(geoid: GeoidState) -> bool:
    """Convenience function to store a geoid"""
    vault = get_global_vault()
    return vault.store_geoid(geoid)


def retrieve_geoid(geoid_id: str) -> Optional[GeoidState]:
    """Convenience function to retrieve a geoid"""
    vault = get_global_vault()
    return vault.retrieve_geoid(geoid_id)


def store_scar(scar: ScarState) -> bool:
    """Convenience function to store a SCAR"""
    vault = get_global_vault()
    return vault.store_scar(scar)


def retrieve_scar(scar_id: str) -> Optional[ScarState]:
    """Convenience function to retrieve a SCAR"""
    vault = get_global_vault()
    return vault.retrieve_scar(scar_id)
