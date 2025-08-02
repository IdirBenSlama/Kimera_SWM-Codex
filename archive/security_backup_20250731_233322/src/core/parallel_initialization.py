"""
Parallel Initialization System for KIMERA
Optimizes startup time through parallel component initialization
Phase 3, Week 8: Performance Optimization
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import traceback
from collections import defaultdict
import networkx as nx

from src.config import get_settings
from src.core.task_manager import get_task_manager, initialize_task_manager
from src.core.async_performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Component initialization status"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ComponentInfo:
    """Information about a component to initialize"""
    name: str
    init_func: Callable
    dependencies: Set[str] = field(default_factory=set)
    optional: bool = False
    timeout: float = 30.0
    retry_count: int = 3
    status: ComponentStatus = ComponentStatus.PENDING
    error: Optional[str] = None
    init_time: Optional[float] = None
    result: Any = None


@dataclass
class InitializationProgress:
    """Track initialization progress"""
    total_components: int
    initialized: int = 0
    failed: int = 0
    skipped: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_components == 0:
            return 100.0
        completed = self.initialized + self.failed + self.skipped
        return (completed / self.total_components) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            "total_components": self.total_components,
            "initialized": self.initialized,
            "failed": self.failed,
            "skipped": self.skipped,
            "progress_percentage": self.progress_percentage,
            "elapsed_time": self.elapsed_time
        }


class ParallelInitializer:
    """
    Manages parallel initialization of KIMERA components
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.initialization_graph = nx.DiGraph()
        self.progress = InitializationProgress(total_components=0)
        self._progress_callbacks: List[Callable[[InitializationProgress], None]] = []
        self._initialized_components: Set[str] = set()
        self._lock = asyncio.Lock()
        
        # Performance monitoring
        self.performance_monitor = get_performance_monitor()
        self.task_manager = None  # Will be initialized when needed
        
        logger.info("ParallelInitializer created")
    
    async def _get_task_manager(self):
        """Get task manager, initializing if necessary"""
        if self.task_manager is None:
            self.task_manager = await initialize_task_manager()
        return self.task_manager
    
    def register_component(
        self,
        name: str,
        init_func: Callable,
        dependencies: Optional[List[str]] = None,
        optional: bool = False,
        timeout: float = 30.0,
        retry_count: int = 3
    ) -> None:
        """
        Register a component for initialization
        
        Args:
            name: Unique component name
            init_func: Async initialization function
            dependencies: List of component names this depends on
            optional: Whether this component is optional
            timeout: Initialization timeout in seconds
            retry_count: Number of retry attempts
        """
        if name in self.components:
            raise ValueError(f"Component '{name}' already registered")
        
        component = ComponentInfo(
            name=name,
            init_func=init_func,
            dependencies=set(dependencies or []),
            optional=optional,
            timeout=timeout,
            retry_count=retry_count
        )
        
        self.components[name] = component
        self.initialization_graph.add_node(name)
        
        # Add edges for dependencies
        for dep in component.dependencies:
            self.initialization_graph.add_edge(dep, name)
        
        logger.debug(f"Registered component: {name} (dependencies: {component.dependencies})")
    
    def add_progress_callback(self, callback: Callable[[InitializationProgress], None]) -> None:
        """Add a callback to be called on progress updates"""
        self._progress_callbacks.append(callback)
    
    async def initialize_all(self) -> Dict[str, Any]:
        """
        Initialize all registered components in parallel
        
        Returns:
            Dictionary of initialized components and their results
            
        Raises:
            RuntimeError: If critical components fail to initialize
        """
        # Validate dependency graph
        if not nx.is_directed_acyclic_graph(self.initialization_graph):
            cycles = list(nx.simple_cycles(self.initialization_graph))
            raise RuntimeError(f"Circular dependencies detected: {cycles}")
        
        # Update progress
        self.progress.total_components = len(self.components)
        self.progress.start_time = time.time()
        
        # Get initialization order
        init_levels = self._get_initialization_levels()
        
        logger.info(f"Starting parallel initialization of {len(self.components)} components")
        logger.info(f"Initialization levels: {len(init_levels)}")
        
        # Initialize components level by level
        results = {}
        
        for level_num, level_components in enumerate(init_levels):
            logger.info(f"Initializing level {level_num + 1}/{len(init_levels)}: {level_components}")
            
            # Initialize all components in this level in parallel
            level_tasks = []
            for component_name in level_components:
                component = self.components[component_name]
                
                # Check if dependencies are satisfied
                if self._are_dependencies_satisfied(component):
                    task = asyncio.create_task(
                        self._initialize_component(component)
                    )
                    level_tasks.append((component_name, task))
                else:
                    # Skip if dependencies failed
                    component.status = ComponentStatus.SKIPPED
                    component.error = "Dependencies not satisfied"
                    self.progress.skipped += 1
                    self._update_progress()
            
            # Wait for all components in this level
            if level_tasks:
                await asyncio.gather(
                    *[task for _, task in level_tasks],
                    return_exceptions=True
                )
                
                # Collect results
                for component_name, task in level_tasks:
                    component = self.components[component_name]
                    if component.status == ComponentStatus.INITIALIZED:
                        results[component_name] = component.result
        
        # Check for critical failures
        critical_failures = [
            comp for comp in self.components.values()
            if not comp.optional and comp.status == ComponentStatus.FAILED
        ]
        
        if critical_failures:
            error_msg = "Critical components failed to initialize:\n"
            for comp in critical_failures:
                error_msg += f"  - {comp.name}: {comp.error}\n"
            raise RuntimeError(error_msg)
        
        # Log summary
        self._log_initialization_summary()
        
        return results
    
    def _get_initialization_levels(self) -> List[List[str]]:
        """
        Get components grouped by initialization level
        Components in the same level can be initialized in parallel
        """
        # Use topological generations to get levels
        try:
            levels = list(nx.topological_generations(self.initialization_graph))
            return levels
        except nx.NetworkXError:
            # Fallback to simple topological sort
            sorted_components = list(nx.topological_sort(self.initialization_graph))
            # Group into levels based on dependencies
            levels = []
            remaining = set(sorted_components)
            initialized = set()
            
            while remaining:
                current_level = []
                for comp in remaining:
                    deps = self.components[comp].dependencies
                    if deps.issubset(initialized):
                        current_level.append(comp)
                
                if not current_level:
                    # No progress possible
                    raise RuntimeError(f"Cannot resolve dependencies for: {remaining}")
                
                levels.append(current_level)
                initialized.update(current_level)
                remaining.difference_update(current_level)
            
            return levels
    
    def _are_dependencies_satisfied(self, component: ComponentInfo) -> bool:
        """Check if all dependencies are satisfied"""
        for dep in component.dependencies:
            if dep not in self.components:
                logger.warning(f"Unknown dependency '{dep}' for component '{component.name}'")
                return False
            
            dep_component = self.components[dep]
            if dep_component.status != ComponentStatus.INITIALIZED:
                if not dep_component.optional:
                    return False
        
        return True
    
    async def _initialize_component(self, component: ComponentInfo) -> None:
        """Initialize a single component with retry logic"""
        component.status = ComponentStatus.INITIALIZING
        start_time = time.time()
        
        # Track with performance monitor
        async with self.performance_monitor.track_operation(f"init_{component.name}"):
            for attempt in range(component.retry_count):
                try:
                    logger.info(f"Initializing component: {component.name} (attempt {attempt + 1}/{component.retry_count})")
                    
                    # Run initialization with timeout
                    component.result = await asyncio.wait_for(
                        component.init_func(),
                        timeout=component.timeout
                    )
                    
                    # Success
                    component.status = ComponentStatus.INITIALIZED
                    component.init_time = time.time() - start_time
                    
                    async with self._lock:
                        self._initialized_components.add(component.name)
                        self.progress.initialized += 1
                    
                    logger.info(f"Component '{component.name}' initialized in {component.init_time:.2f}s")
                    self._update_progress()
                    return
                    
                except asyncio.TimeoutError:
                    error_msg = f"Timeout after {component.timeout}s"
                    logger.error(f"Component '{component.name}' initialization timeout (attempt {attempt + 1})")
                    component.error = error_msg
                    
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"Component '{component.name}' initialization failed: {error_msg}")
                    logger.debug(traceback.format_exc())
                    component.error = error_msg
                
                # Wait before retry
                if attempt < component.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # All attempts failed
            component.status = ComponentStatus.FAILED
            component.init_time = time.time() - start_time
            
            async with self._lock:
                self.progress.failed += 1
            
            self._update_progress()
    
    def _update_progress(self) -> None:
        """Update progress and notify callbacks"""
        for callback in self._progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def _log_initialization_summary(self) -> None:
        """Log initialization summary"""
        total_time = self.progress.elapsed_time
        
        logger.info("=" * 60)
        logger.info("Initialization Summary")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Components initialized: {self.progress.initialized}/{self.progress.total_components}")
        logger.info(f"Components failed: {self.progress.failed}")
        logger.info(f"Components skipped: {self.progress.skipped}")
        
        # Log individual component times
        logger.info("\nComponent initialization times:")
        sorted_components = sorted(
            [c for c in self.components.values() if c.init_time is not None],
            key=lambda x: x.init_time,
            reverse=True
        )
        
        for comp in sorted_components[:10]:  # Top 10 slowest
            status_icon = "âœ“" if comp.status == ComponentStatus.INITIALIZED else "âœ—"
            logger.info(f"  {status_icon} {comp.name}: {comp.init_time:.2f}s")
        
        if len(sorted_components) > 10:
            logger.info(f"  ... and {len(sorted_components) - 10} more")
        
        logger.info("=" * 60)
    
    def get_initialization_report(self) -> Dict[str, Any]:
        """Get detailed initialization report"""
        return {
            "summary": self.progress.to_dict(),
            "components": {
                name: {
                    "status": comp.status.value,
                    "init_time": comp.init_time,
                    "error": comp.error,
                    "dependencies": list(comp.dependencies),
                    "optional": comp.optional
                }
                for name, comp in self.components.items()
            },
            "initialization_order": [
                list(level) for level in self._get_initialization_levels()
            ]
        }


# Global initializer instance
_parallel_initializer: Optional[ParallelInitializer] = None


def get_parallel_initializer() -> ParallelInitializer:
    """Get global parallel initializer instance"""
    global _parallel_initializer
    if _parallel_initializer is None:
        _parallel_initializer = ParallelInitializer()
    return _parallel_initializer


# Convenience decorator for component registration
def initialization_component(
    name: str,
    dependencies: Optional[List[str]] = None,
    optional: bool = False,
    timeout: float = 30.0,
    retry_count: int = 3
):
    """
    Decorator to register a component for parallel initialization
    
    Usage:
        @initialization_component("database", dependencies=["config"])
        async def initialize_database():
            # Initialize database
            return db_connection
    """
    def decorator(func: Callable) -> Callable:
        # Register with global initializer
        initializer = get_parallel_initializer()
        initializer.register_component(
            name=name,
            init_func=func,
            dependencies=dependencies,
            optional=optional,
            timeout=timeout,
            retry_count=retry_count
        )
        return func
    
    return decorator