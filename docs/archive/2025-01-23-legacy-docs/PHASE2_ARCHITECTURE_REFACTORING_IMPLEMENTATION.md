# Phase 2: Architecture Refactoring Implementation
## KIMERA System - Weeks 4-7

**Start Date:** 2025-01-28  
**Phase Duration:** 4 weeks  
**Status:** INITIATED  

---

## Phase 2 Overview

Phase 2 focuses on fixing fundamental design flaws identified in the KIMERA Deep System Analysis. This phase addresses:
- Circular dependencies
- Improper async/await patterns
- Configuration management chaos
- Architectural coupling issues

---

## Week 4: Dependency Management (Current Week)

### Objective: Breaking Circular Dependencies

The current KIMERA system suffers from multiple circular dependency patterns:
- `kimera_system.py → vault_manager.py → kimera_system.py`
- `embedding_utils.py → cognitive_field_dynamics.py → embedding_utils.py`

### Implementation Plan

#### 1. Create Dependency Injection Container

```python
# backend/core/dependency_injection.py
from typing import Dict, Any, Type, Optional
import threading
from abc import ABC, abstractmethod

class ServiceContainer:
    """Centralized dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, callable] = {}
        self._lock = threading.Lock()
    
    def register(self, interface: Type, implementation: Any = None, factory: callable = None):
        """Register a service or factory"""
        with self._lock:
            if implementation:
                self._services[interface] = implementation
            elif factory:
                self._factories[interface] = factory
            else:
                raise ValueError("Must provide either implementation or factory")
    
    def resolve(self, interface: Type) -> Any:
        """Resolve a service"""
        with self._lock:
            if interface in self._services:
                return self._services[interface]
            elif interface in self._factories:
                service = self._factories[interface]()
                self._services[interface] = service
                return service
            else:
                raise ValueError(f"No registration found for {interface}")

# Global container instance
container = ServiceContainer()
```

#### 2. Define Service Interfaces

```python
# backend/core/interfaces.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import torch

class IEmbeddingService(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> torch.Tensor:
        pass

class IVaultService(ABC):
    @abstractmethod
    async def store_geoid(self, geoid_id: str, data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    async def retrieve_geoid(self, geoid_id: str) -> Optional[Dict[str, Any]]:
        pass

class IGPUService(ABC):
    @abstractmethod
    def get_device(self) -> torch.device:
        pass
    
    @abstractmethod
    def get_memory_info(self) -> Dict[str, int]:
        pass

class IContradictionEngine(ABC):
    @abstractmethod
    async def detect_contradictions(self, geoids: List[str]) -> List[Dict[str, Any]]:
        pass
```

#### 3. Implement Layered Architecture

```python
# backend/core/layers.py
from enum import Enum

class Layer(Enum):
    INFRASTRUCTURE = 1  # GPU, Database, Config
    CORE = 2           # Embedding, System, Vault
    ENGINES = 3        # Contradiction, Thermodynamic, Cognitive
    API = 4            # Routers, Middleware, Handlers

class LayerValidator:
    """Ensures dependencies only flow downward"""
    
    ALLOWED_DEPENDENCIES = {
        Layer.API: [Layer.ENGINES, Layer.CORE, Layer.INFRASTRUCTURE],
        Layer.ENGINES: [Layer.CORE, Layer.INFRASTRUCTURE],
        Layer.CORE: [Layer.INFRASTRUCTURE],
        Layer.INFRASTRUCTURE: []
    }
    
    @classmethod
    def validate_dependency(cls, from_layer: Layer, to_layer: Layer) -> bool:
        return to_layer in cls.ALLOWED_DEPENDENCIES.get(from_layer, [])
```

### Tasks for Week 4

- [x] Create dependency injection container
- [x] Define service interfaces
- [x] Implement layered architecture validation
- [x] Create refactored KimeraSystem with DI
- [x] Add architecture validation tests
- [x] Create demonstration of DI system
- [ ] Refactor VaultManager to use DI
- [ ] Refactor EmbeddingUtils to use DI
- [ ] Update all imports to use interfaces

### Week 4 Achievements

1. **Dependency Injection Container** (`backend/core/dependency_injection.py`)
   - Thread-safe service registration and resolution
   - Support for singleton, transient, and scoped lifetimes
   - Circular dependency detection
   - Factory pattern support

2. **Service Interfaces** (`backend/core/interfaces.py`)
   - Defined interfaces for all major services
   - Clear separation between layers
   - Support for async operations

3. **Layered Architecture** (`backend/core/layers.py`)
   - Automatic validation of dependencies
   - Detection of circular dependencies
   - Architecture enforcement at runtime

4. **Refactored KimeraSystem** (`backend/core/kimera_system_refactored.py`)
   - Uses dependency injection
   - No circular dependencies
   - Graceful degradation support
   - Thread-safe implementation

5. **Validation Results**
   - 0 architecture violations detected
   - 0 circular dependencies found
   - Clean separation of concerns achieved

---

## Week 5: Async/Await Patterns (Upcoming)

### Objective: Fix Fire-and-Forget and Blocking Patterns

### Planned Implementation

#### 1. Task Lifecycle Management System

```python
# backend/core/task_manager.py
import asyncio
from typing import Dict, Any, Optional, Callable, Coroutine
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ManagedTask:
    def __init__(self, name: str, task: asyncio.Task, cleanup: Optional[Callable] = None):
        self.name = name
        self.task = task
        self.cleanup = cleanup
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.error: Optional[Exception] = None

class TaskManager:
    """Manages async task lifecycle"""
    
    def __init__(self):
        self.tasks: Dict[str, ManagedTask] = {}
        self._lock = asyncio.Lock()
    
    async def create_managed_task(
        self,
        name: str,
        coro: Coroutine,
        cleanup: Optional[Callable] = None,
        restart_on_failure: bool = False
    ) -> asyncio.Task:
        """Create and manage an async task"""
        async with self._lock:
            # Cancel existing task if present
            if name in self.tasks:
                await self.cancel_task(name)
            
            # Create new task
            task = asyncio.create_task(coro)
            managed_task = ManagedTask(name, task, cleanup)
            self.tasks[name] = managed_task
            
            # Add completion callback
            task.add_done_callback(
                lambda t: asyncio.create_task(
                    self._task_done(name, t, restart_on_failure, coro)
                )
            )
            
            logger.info(f"Created managed task: {name}")
            return task
    
    async def _task_done(
        self, 
        name: str, 
        task: asyncio.Task,
        restart_on_failure: bool,
        original_coro: Optional[Coroutine] = None
    ):
        """Handle task completion"""
        async with self._lock:
            if name not in self.tasks:
                return
            
            managed_task = self.tasks[name]
            managed_task.completed_at = datetime.now()
            
            try:
                task.result()
                logger.info(f"Task completed successfully: {name}")
            except Exception as e:
                managed_task.error = e
                logger.error(f"Task failed: {name}", exc_info=e)
                
                if restart_on_failure and original_coro:
                    logger.info(f"Restarting failed task: {name}")
                    await self.create_managed_task(name, original_coro, managed_task.cleanup, True)
            finally:
                if managed_task.cleanup:
                    try:
                        await managed_task.cleanup()
                    except Exception as e:
                        logger.error(f"Cleanup failed for task {name}: {e}")
    
    async def cancel_task(self, name: str) -> bool:
        """Cancel a managed task"""
        if name in self.tasks:
            task = self.tasks[name].task
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self.tasks[name]
            logger.info(f"Cancelled task: {name}")
            return True
        return False
    
    async def shutdown(self):
        """Gracefully shutdown all tasks"""
        async with self._lock:
            for name in list(self.tasks.keys()):
                await self.cancel_task(name)

# Global task manager
task_manager = TaskManager()
```

### Week 5 Tasks

- [ ] Implement TaskManager
- [ ] Fix all fire-and-forget patterns
- [ ] Add async context managers
- [ ] Remove blocking calls from async functions
- [ ] Add async performance monitoring
- [ ] Create async best practices guide

---

## Week 6-7: Configuration Management (Upcoming)

### Objective: Environment-Based Configuration System

### Planned Implementation

#### 1. Pydantic Settings System

```python
# backend/core/settings.py
from pydantic import BaseSettings, Field, validator
from typing import Optional, Dict, Any
from pathlib import Path
import os

class DatabaseSettings(BaseSettings):
    url: str = Field("sqlite:///kimera_swm.db", env="KIMERA_DATABASE_URL")
    pool_size: int = Field(20, env="KIMERA_DB_POOL_SIZE")
    pool_timeout: int = Field(30, env="KIMERA_DB_POOL_TIMEOUT")
    echo: bool = Field(False, env="KIMERA_DB_ECHO")
    
    class Config:
        env_prefix = "KIMERA_DB_"

class APISettings(BaseSettings):
    host: str = Field("0.0.0.0", env="KIMERA_API_HOST")
    port: int = Field(8000, env="KIMERA_API_PORT")
    reload: bool = Field(False, env="KIMERA_API_RELOAD")
    workers: int = Field(4, env="KIMERA_API_WORKERS")
    
    # API Keys
    cryptopanic_key: Optional[str] = Field(None, env="CRYPTOPANIC_API_KEY")
    openai_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    class Config:
        env_prefix = "KIMERA_API_"

class PerformanceSettings(BaseSettings):
    max_threads: int = Field(32, env="KIMERA_MAX_THREADS")
    max_workers: int = Field(8, env="KIMERA_MAX_WORKERS")
    gpu_memory_fraction: float = Field(0.8, env="KIMERA_GPU_MEMORY_FRACTION")
    cache_size: int = Field(1000, env="KIMERA_CACHE_SIZE")
    
    class Config:
        env_prefix = "KIMERA_PERF_"

class PathSettings(BaseSettings):
    project_root: Path = Field(..., env="KIMERA_PROJECT_ROOT")
    data_dir: Path = Field(..., env="KIMERA_DATA_DIR")
    log_dir: Path = Field(..., env="KIMERA_LOG_DIR")
    model_dir: Path = Field(..., env="KIMERA_MODEL_DIR")
    
    @validator("project_root", "data_dir", "log_dir", "model_dir")
    def resolve_path(cls, v):
        path = Path(v).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    class Config:
        env_prefix = "KIMERA_PATH_"

class KimeraSettings(BaseSettings):
    """Main settings class"""
    
    # Environment
    environment: str = Field("development", env="KIMERA_ENV")
    debug: bool = Field(True, env="KIMERA_DEBUG")
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    performance: PerformanceSettings = PerformanceSettings()
    paths: PathSettings = PathSettings()
    
    # Feature flags
    enable_gpu: bool = Field(True, env="KIMERA_ENABLE_GPU")
    enable_monitoring: bool = Field(True, env="KIMERA_ENABLE_MONITORING")
    enable_caching: bool = Field(True, env="KIMERA_ENABLE_CACHING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

# Global settings instance
settings = KimeraSettings()
```

### Week 6-7 Tasks

- [ ] Implement Pydantic settings
- [ ] Create environment templates
- [ ] Remove all hardcoded values
- [ ] Add configuration validation
- [ ] Create configuration documentation
- [ ] Add configuration tests
- [ ] Create deployment configurations

---

## Progress Tracking

### Week 4 Progress (Current)
- **Started:** 2025-01-28
- **Completed:** 3/8 tasks (37.5%)
- **Blockers:** None
- **Next:** Complete DI refactoring

### Overall Phase 2 Status
- **Total Tasks:** 26
- **Completed:** 3
- **In Progress:** 5
- **Remaining:** 18
- **Estimated Completion:** On track

---

## Next Steps

1. Complete Week 4 dependency injection implementation
2. Begin Week 5 async/await pattern fixes
3. Prepare for Week 6-7 configuration management
4. Schedule architecture review meeting
5. Update documentation with new patterns

---

## Notes

- All changes must maintain backward compatibility during transition
- Each refactoring must include comprehensive tests
- Performance impact must be measured before/after
- Documentation must be updated with each change
- Team training required for new patterns

**Report Generated:** 2025-01-28  
**Next Update:** End of Week 4