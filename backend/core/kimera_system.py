"""kimera_system.py
Kimera System Core Module
=========================
A minimal but scientifically rigorous implementation of the Kimera System
singleton required by the API layer.  This implementation fulfils the
Zero-Debugging and Cognitive Fidelity constraints by providing explicit
logging, hardware awareness (GPU vs CPU), and clear, observable system
state transitions.

If richer functionality is required, extend this module rather than
introducing ad-hoc globals elsewhere in the codebase.
"""

from __future__ import annotations

import logging
import platform
import threading
import inspect  # Add import for inspect
from enum import Enum, auto
from typing import Optional, Dict, Any

# Fix critical import error - import GPUFoundation class directly
from backend.utils.gpu_foundation import GPUFoundation

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """Enumeration of Kimera System runtime states."""

    STOPPED = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    SHUTTING_DOWN = auto()

    def __str__(self) -> str:  # pragma: no cover – cosmetic
        return self.name.lower()


class KimeraSystem:  # pylint: disable=too-few-public-methods
    """Singleton orchestration class for Kimera runtime subsystems.

    The current implementation is intentionally lightweight.  It logs
    device information (GPU/CPU) and exposes lifecycle hooks consumed
    by the API layer.  All heavy initialisation (models, databases, etc.)
    **must** be delegated to dedicated modules to honour the Separation
    of Concerns principle.

    This class *must* remain import-safe: importing it should not trigger
    expensive side-effects.  Heavy work happens inside :py:meth:`initialize`.
    
    Thread Safety: This implementation uses double-checked locking pattern
    to ensure thread-safe singleton instantiation without performance penalty
    on subsequent accesses.
    """

    _instance: Optional["KimeraSystem"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    _initialization_complete: bool = False
    _initialization_event: threading.Event = threading.Event()

    def __new__(cls) -> "KimeraSystem":  # noqa: D401, N804 – singleton pattern
        # First check without lock for performance
        if cls._instance is None:
            # Acquire lock for thread safety
            with cls._lock:
                # Double-check pattern: verify again inside lock
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Initialize instance attributes only once
                    if not cls._initialized:
                        cls._instance._state = SystemState.STOPPED
                        cls._instance._device = "cpu"
                        cls._instance._components: Dict[str, Any] = {}
                        cls._instance._gpu_foundation = None
                        cls._instance._component_locks: Dict[str, threading.Lock] = {}
                        cls._instance._state_lock = threading.Lock()
                        # This flag should be set on the instance, not the class,
                        # to allow for re-initialization in tests.
                        cls._instance._initialized_once = True 
                    
                    # Reset flags for new instance
                    cls._initialized = False
                    cls._initialization_complete = False
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """
        Initialises all critical subsystems. This method acts as a thread-safe gatekeeper.
        The actual initialization logic is in _do_initialize_once().
        This function is idempotent and safe to call from multiple threads.
        """
        cls = self.__class__
        
        # Fast path: if initialization is already complete, do nothing.
        if cls._initialization_complete:
            return

        # Acquire lock to ensure only one thread proceeds with initialization checks.
        with cls._lock:
            if cls._initialization_complete:
                return  # Double-check inside lock
            
            # If another thread is already initializing, the current thread waits.
            if cls._initialized:
                logger.debug("Initialization in progress, waiting for completion...")
                # Release the lock and wait for the event
                cls._lock.release()
                try:
                    cls._initialization_event.wait(timeout=10) # 10 second timeout
                finally:
                    # Re-acquire the lock before returning
                    cls._lock.acquire()
                return
            
            # This is the first thread to pass the checks; mark as in-progress.
            cls._initialized = True

        # The first thread proceeds. Use try/finally to ensure cleanup.
        try:
            self._do_initialize_once()
        finally:
            # The initializing thread resets the in-progress flag and signals completion.
            with cls._lock:
                cls._initialized = False
                if cls._initialization_complete:
                    cls._initialization_event.set()

    def _do_initialize_once(self) -> None:
        """
        Performs the actual one-time initialization of the system.
        This method should only ever be called by the gatekeeper initialize() method.
        """
        logger.info("KimeraSystem initialising ...")
        self._state = SystemState.INITIALIZING

        # Hardware detection
        try:
            gpu_found = GPUFoundation()
            self._gpu_foundation = gpu_found
            self._device = str(gpu_found.get_device())
            self._set_component("gpu_foundation", gpu_found)
            logger.info("GPU detected – operations will use %s", self._device)
        except (RuntimeError, ImportError, AttributeError) as exc:
            self._device = "cpu"
            self._gpu_foundation = None
            self._set_component("gpu_foundation", None)
            logger.warning(
                "GPU unavailable or initialisation failed (%s). Falling back to CPU.",
                exc,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self._device = "cpu"
            self._gpu_foundation = None
            self._set_component("gpu_foundation", None)
            logger.error(
                "Unexpected error during GPU initialization (%s). Falling back to CPU.",
                exc,
                exc_info=True,
            )

        # Initialize core subsystems
        self._initialize_exception_handling()
        self._initialize_error_recovery()
        self._initialize_performance_manager()
        self._initialize_database_optimization()
        self._initialize_context_supremacy()
        self._initialize_statistical_modeling()
        self._initialize_universal_compassion()
        self._initialize_cache_layer()
        self._initialize_dependency_injection()
        self._initialize_task_manager()
        self._initialize_vault_manager()
        self._initialize_embedding_model()
        self._initialize_contradiction_engine()
        self._initialize_thermodynamics_engine()
        self._initialize_spde_engine()
        self._initialize_cognitive_cycle_engine()
        self._initialize_meta_insight_engine()
        self._initialize_proactive_detector()
        self._initialize_revolutionary_intelligence_engine()
        self._initialize_geoid_scar_manager()
        self._initialize_system_monitor()
        self._initialize_ethical_governor()

        # Mark initialization as complete
        with self.__class__._lock:
            self._state = SystemState.RUNNING
            self.__class__._initialization_complete = True
        
        logger.info("KimeraSystem initialised successfully - state: %s", self._state)

    def _initialize_vault_manager(self) -> None:
        """Initialize the VaultManager subsystem."""
        try:
            from backend.vault.vault_manager import VaultManager
            vault_manager = VaultManager()
            self._set_component("vault_manager", vault_manager)
            logger.info("VaultManager initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import VaultManager: %s", exc)
            self._set_component("vault_manager", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize VaultManager: %s", exc, exc_info=True)
            self._set_component("vault_manager", None)

    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model subsystem."""
        try:
            from backend.core import embedding_utils
            self._set_component("embedding_model", True)  # Placeholder - actual model loaded in embedding_utils
            logger.info("Embedding model initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import embedding utilities: %s", exc)
            self._set_component("embedding_model", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize embedding model: %s", exc, exc_info=True)
            self._set_component("embedding_model", None)

    def _initialize_contradiction_engine(self) -> None:
        """Initialize the contradiction engine subsystem."""
        try:
            from backend.engines.contradiction_engine import ContradictionEngine
            engine = ContradictionEngine(tension_threshold=0.4)
            self._set_component("contradiction_engine", engine)
            logger.info("Contradiction engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import ContradictionEngine: %s", exc)
            self._set_component("contradiction_engine", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize contradiction engine: %s", exc, exc_info=True)
            self._set_component("contradiction_engine", None)

    def _initialize_thermodynamics_engine(self) -> None:
        """Initialize the thermodynamics engine subsystem."""
        try:
            from backend.engines.foundational_thermodynamic_engine import FoundationalThermodynamicEngine
            engine = FoundationalThermodynamicEngine()
            self._set_component("thermodynamics_engine", engine)
            logger.info("Thermodynamic engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import FoundationalThermodynamicEngine: %s", exc)
            self._set_component("thermodynamics_engine", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize thermodynamics engine: %s", exc, exc_info=True)
            self._set_component("thermodynamics_engine", None)

    def _initialize_spde_engine(self) -> None:
        """Initialize the SPDE engine subsystem."""
        try:
            from backend.engines.spde_engine import create_spde_engine
            engine = create_spde_engine(device=self._device)
            self._set_component("spde_engine", engine)
            logger.info("SPDE engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import SPDE engine: %s", exc)
            self._set_component("spde_engine", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize SPDE engine: %s", exc, exc_info=True)
            self._set_component("spde_engine", None)

    def _initialize_cognitive_cycle_engine(self) -> None:
        """Initialize the Cognitive Cycle engine subsystem."""
        try:
            from backend.engines.cognitive_cycle_engine import create_cognitive_cycle_engine
            engine = create_cognitive_cycle_engine(device=self._device)
            self._set_component("cognitive_cycle_engine", engine)
            logger.info("Cognitive Cycle engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Cognitive Cycle engine: %s", exc)
            self._set_component("cognitive_cycle_engine", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Cognitive Cycle engine: %s", exc, exc_info=True)
            self._set_component("cognitive_cycle_engine", None)

    def _initialize_meta_insight_engine(self) -> None:
        """Initialize the Meta Insight engine subsystem."""
        try:
            from backend.engines.meta_insight_engine import create_meta_insight_engine
            engine = create_meta_insight_engine(device=self._device)
            self._set_component("meta_insight_engine", engine)
            logger.info("Meta Insight engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Meta Insight engine: %s", exc)
            self._set_component("meta_insight_engine", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Meta Insight engine: %s", exc, exc_info=True)
            self._set_component("meta_insight_engine", None)

    def _initialize_proactive_detector(self) -> None:
        """Initialize the Proactive Detector subsystem."""
        try:
            from backend.engines.proactive_detector import create_proactive_detector
            engine = create_proactive_detector(device=self._device)
            self._set_component("proactive_detector", engine)
            logger.info("Proactive Detector initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Proactive Detector: %s", exc)
            self._set_component("proactive_detector", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Proactive Detector: %s", exc, exc_info=True)
            self._set_component("proactive_detector", None)

    def _initialize_revolutionary_intelligence_engine(self) -> None:
        """Initialize the Revolutionary Intelligence Engine subsystem."""
        try:
            from backend.engines.revolutionary_intelligence_engine import create_revolutionary_intelligence_engine
            engine = create_revolutionary_intelligence_engine(device=self._device)
            self._set_component("revolutionary_intelligence_engine", engine)
            logger.info("Revolutionary Intelligence Engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Revolutionary Intelligence Engine: %s", exc)
            self._set_component("revolutionary_intelligence_engine", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Revolutionary Intelligence Engine: %s", exc, exc_info=True)
            self._set_component("revolutionary_intelligence_engine", None)

    def _initialize_geoid_scar_manager(self) -> None:
        """Initialize the Geoid SCAR Manager subsystem."""
        try:
            from backend.engines.geoid_scar_manager import GeoidScarManager
            manager = GeoidScarManager()
            self._set_component("geoid_scar_manager", manager)
            logger.info("Geoid SCAR Manager initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import GeoidScarManager: %s", exc)
            self._set_component("geoid_scar_manager", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Geoid SCAR Manager: %s", exc, exc_info=True)
            self._set_component("geoid_scar_manager", None)

    def _initialize_system_monitor(self) -> None:
        """Initialize the System Monitor subsystem."""
        try:
            from backend.monitoring.system_monitor import SystemMonitor
            monitor = SystemMonitor()
            self._set_component("system_monitor", monitor)
            logger.info("System Monitor initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import SystemMonitor: %s", exc)
            self._set_component("system_monitor", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize System Monitor: %s", exc, exc_info=True)
            self._set_component("system_monitor", None)

    def _initialize_ethical_governor(self) -> None:
        """Initialize the Ethical Governor subsystem."""
        try:
            from backend.governance.ethical_governor import EthicalGovernor
            governor = EthicalGovernor()
            self._set_component("ethical_governor", governor)
            logger.info("Ethical Governor initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import EthicalGovernor: %s", exc)
            self._set_component("ethical_governor", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Ethical Governor: %s", exc, exc_info=True)
            self._set_component("ethical_governor", None)

    def _initialize_exception_handling(self) -> None:
        """Initialize the Exception Handling subsystem."""
        try:
            from backend.core import exception_handling
            self._set_component("exception_handling", exception_handling.error_registry)
            logger.info("Exception Handling initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Exception Handling: %s", exc)
            self._set_component("exception_handling", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Exception Handling: %s", exc, exc_info=True)
            self._set_component("exception_handling", None)

    def _initialize_error_recovery(self) -> None:
        """Initialize the Error Recovery subsystem."""
        try:
            from backend.core.error_recovery import get_error_recovery_manager
            manager = get_error_recovery_manager()
            self._set_component("error_recovery", manager)
            logger.info("Error Recovery initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Error Recovery: %s", exc)
            self._set_component("error_recovery", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Error Recovery: %s", exc, exc_info=True)
            self._set_component("error_recovery", None)

    def _initialize_performance_manager(self) -> None:
        """Initialize the Performance Manager subsystem."""
        try:
            from backend.core.performance_integration import PerformanceManager
            manager = PerformanceManager()
            self._set_component("performance_manager", manager)
            logger.info("Performance Manager initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Performance Manager: %s", exc)
            self._set_component("performance_manager", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Performance Manager: %s", exc, exc_info=True)
            self._set_component("performance_manager", None)

    def _initialize_database_optimization(self) -> None:
        """Initialize the Database Optimization subsystem."""
        try:
            from backend.core.database_optimization import DatabaseConnectionPool
            db_pool = DatabaseConnectionPool()
            self._set_component("database_optimization", db_pool)
            logger.info("Database Optimization initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Database Optimization: %s", exc)
            self._set_component("database_optimization", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Database Optimization: %s", exc, exc_info=True)
            self._set_component("database_optimization", None)

    def _initialize_context_supremacy(self) -> None:
        """Initialize the Context Supremacy Engine subsystem."""
        try:
            from backend.core.context_supremacy import ContextSupremacyEngine
            engine = ContextSupremacyEngine()
            self._set_component("context_supremacy", engine)
            logger.info("Context Supremacy Engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Context Supremacy Engine: %s", exc)
            self._set_component("context_supremacy", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Context Supremacy Engine: %s", exc, exc_info=True)
            self._set_component("context_supremacy", None)

    def _initialize_statistical_modeling(self) -> None:
        """Initialize the Statistical Modeling subsystem."""
        try:
            from backend.core.statistical_modeling import StatisticalModelingEngine
            engine = StatisticalModelingEngine()
            self._set_component("statistical_modeling", engine)
            logger.info("Statistical Modeling Engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Statistical Modeling Engine: %s", exc)
            self._set_component("statistical_modeling", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Statistical Modeling Engine: %s", exc, exc_info=True)
            self._set_component("statistical_modeling", None)

    def _initialize_universal_compassion(self) -> None:
        """Initialize the Universal Compassion Engine subsystem."""
        try:
            from backend.core.universal_compassion import UniversalCompassionEngine
            engine = UniversalCompassionEngine()
            self._set_component("universal_compassion", engine)
            logger.info("Universal Compassion Engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Universal Compassion Engine: %s", exc)
            self._set_component("universal_compassion", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Universal Compassion Engine: %s", exc, exc_info=True)
            self._set_component("universal_compassion", None)

    def _initialize_cache_layer(self) -> None:
        """Initialize the Cache Layer subsystem."""
        try:
            from backend.core.cache_layer import CacheManager
            manager = CacheManager()
            self._set_component("cache_layer", manager)
            logger.info("Cache Layer initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Cache Layer: %s", exc)
            self._set_component("cache_layer", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Cache Layer: %s", exc, exc_info=True)
            self._set_component("cache_layer", None)

    def _initialize_dependency_injection(self) -> None:
        """Initialize the Dependency Injection subsystem."""
        try:
            from backend.core.dependency_injection import ServiceContainer
            container = ServiceContainer()
            self._set_component("dependency_injection", container)
            logger.info("Dependency Injection initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Dependency Injection: %s", exc)
            self._set_component("dependency_injection", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Dependency Injection: %s", exc, exc_info=True)
            self._set_component("dependency_injection", None)

    def _initialize_task_manager(self) -> None:
        """Initialize the Task Manager subsystem."""
        try:
            from backend.core.task_manager import TaskManager
            manager = TaskManager()
            self._set_component("task_manager", manager)
            logger.info("Task Manager initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Task Manager: %s", exc)
            self._set_component("task_manager", None)
        except Exception as exc:
            logger.error("❌ Failed to initialize Task Manager: %s", exc, exc_info=True)
            self._set_component("task_manager", None)

    async def shutdown(self) -> None:
        """Graceful shutdown of all subsystems.
        
        Thread Safety: State transitions and component access are protected.
        """
        with self._state_lock:
            if self._state is SystemState.SHUTTING_DOWN:
                return
            logger.info("🔻 KimeraSystem shutdown initiated …")
            self._state = SystemState.SHUTTING_DOWN

        # Create a snapshot of components to avoid modification during iteration
        components_snapshot = {}
        for name in list(self._components.keys()):
            component = self.get_component(name)
            if component:
                components_snapshot[name] = component

        # Shutdown subsystems, handling both sync and async shutdown methods
        for component_name, component in components_snapshot.items():
            if component and hasattr(component, 'shutdown'):
                try:
                    shutdown_method = getattr(component, 'shutdown')
                    if inspect.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
                    logger.info(f"✅ {component_name} shutdown complete")
                except Exception as exc:
                    logger.error(f"❌ Error shutting down {component_name}: {exc}", exc_info=True)

        # Clear all components
        for name in list(self._components.keys()):
            self._set_component(name, None)
            
        with self._state_lock:
            self._state = SystemState.STOPPED
        logger.info("🛑 KimeraSystem shutdown complete – state: %s", self._state)

    # ------------------------------------------------------------------
    # Component access methods (replaces circular import pattern)
    # ------------------------------------------------------------------
    def _get_component_lock(self, component_name: str) -> threading.Lock:
        """Get or create a lock for a specific component."""
        if component_name not in self._component_locks:
            with self._lock:  # Use class lock to protect lock creation
                if component_name not in self._component_locks:
                    self._component_locks[component_name] = threading.Lock()
        return self._component_locks[component_name]
    
    def _set_component(self, component_name: str, component: Any) -> None:
        """Thread-safe component setter."""
        lock = self._get_component_lock(component_name)
        with lock:
            self._components[component_name] = component
    
    def get_component(self, component_name: str) -> Any:
        """Get a specific component by name (thread-safe)."""
        lock = self._get_component_lock(component_name)
        with lock:
            return self._components.get(component_name)

    def get_vault_manager(self):
        """Get the VaultManager instance (thread-safe)."""
        return self.get_component("vault_manager")

    def get_embedding_model(self):
        """Get the embedding model (thread-safe)."""
        return self.get_component("embedding_model")

    def get_contradiction_engine(self):
        """Get the contradiction engine instance (thread-safe)."""
        return self.get_component("contradiction_engine")

    def get_thermodynamic_engine(self):
        """Get the thermodynamic engine instance (thread-safe)."""
        return self.get_component("thermodynamics_engine")

    def get_spde_engine(self):
        """Get the SPDE engine instance (thread-safe)."""
        return self.get_component("spde_engine")

    def get_cognitive_cycle_engine(self):
        """Get the Cognitive Cycle engine instance (thread-safe)."""
        return self.get_component("cognitive_cycle_engine")

    def get_meta_insight_engine(self):
        """Get the Meta Insight engine instance (thread-safe)."""
        return self.get_component("meta_insight_engine")

    def get_proactive_detector(self):
        """Get the Proactive Detector instance (thread-safe)."""
        return self.get_component("proactive_detector")

    def get_revolutionary_intelligence_engine(self):
        """Get the Revolutionary Intelligence Engine instance (thread-safe)."""
        return self.get_component("revolutionary_intelligence_engine")

    def get_gpu_foundation(self):
        """Get the GPU foundation instance (thread-safe)."""
        return self.get_component("gpu_foundation")

    def get_geoid_scar_manager(self):
        """Get the Geoid SCAR Manager instance (thread-safe)."""
        return self.get_component("geoid_scar_manager")

    def get_system_monitor(self):
        """Get the System Monitor instance (thread-safe)."""
        return self.get_component("system_monitor")

    def get_ethical_governor(self):
        """Get the Ethical Governor instance (thread-safe)."""
        return self.get_component("ethical_governor")

    def get_exception_handling(self):
        """Get the Exception Handling registry (thread-safe)."""
        return self.get_component("exception_handling")

    def get_error_recovery(self):
        """Get the Error Recovery manager (thread-safe)."""
        return self.get_component("error_recovery")

    def get_performance_manager(self):
        """Get the Performance Manager instance (thread-safe)."""
        return self.get_component("performance_manager")

    def get_database_optimization(self):
        """Get the Database Optimization instance (thread-safe)."""
        return self.get_component("database_optimization")

    def get_context_supremacy(self):
        """Get the Context Supremacy Engine instance (thread-safe)."""
        return self.get_component("context_supremacy")

    def get_statistical_modeling(self):
        """Get the Statistical Modeling Engine instance (thread-safe)."""
        return self.get_component("statistical_modeling")

    def get_universal_compassion(self):
        """Get the Universal Compassion Engine instance (thread-safe)."""
        return self.get_component("universal_compassion")

    def get_cache_layer(self):
        """Get the Cache Layer instance (thread-safe)."""
        return self.get_component("cache_layer")

    def get_dependency_injection(self):
        """Get the Dependency Injection container (thread-safe)."""
        return self.get_component("dependency_injection")

    def get_task_manager(self):
        """Get the Task Manager instance (thread-safe)."""
        return self.get_component("task_manager")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status (thread-safe)."""
        with self._state_lock:
            current_state = self._state
            
        # Get component status safely
        component_names = ["vault_manager", "gpu_foundation", "contradiction_engine", 
                          "thermodynamics_engine", "embedding_model", "geoid_scar_manager",
                          "system_monitor", "ethical_governor"]
        component_status = {}
        for name in component_names:
            component_status[name] = self.get_component(name) is not None
            
        return {
            "state": current_state,
            "components": component_status,
            "vault_manager_ready": component_status.get("vault_manager", False),
            "gpu_foundation_ready": component_status.get("gpu_foundation", False),
            "contradiction_engine_ready": component_status.get("contradiction_engine", False),
            "thermodynamic_engine_ready": component_status.get("thermodynamics_engine", False),
            "embedding_model_ready": component_status.get("embedding_model", False),
            "geoid_scar_manager_ready": component_status.get("geoid_scar_manager", False),
            "system_monitor_ready": component_status.get("system_monitor", False),
            "ethical_governor_ready": component_status.get("ethical_governor", False),
        }

    # ------------------------------------------------------------------
    # Legacy compatibility methods (for existing router patterns)
    # ------------------------------------------------------------------
    def get(self, key: str, default=None):
        """Legacy dict-style access for backward compatibility."""
        if key == "status":
            return str(self._state)
        elif key == "system_state":
            return {
                "state": str(self._state),
                "device": self._device,
                "components": {name: comp is not None for name, comp in self._components.items()}
            }
        return self._components.get(key, default)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_status(self) -> str:
        """Return a human-readable system status string (thread-safe)."""
        with self._state_lock:
            return str(self._state)

    def get_device(self) -> str:
        """Return the compute device in use (e.g. ``cuda:0`` or ``cpu``)."""
        return self._device

    def get_system_state(self) -> Dict[str, Any]:
        """Returns a dictionary representing the current system state."""
        return {
            "state": self._state.name,
            "device": self._device,
            "components": list(self._components.keys())
        }

    @property
    def state(self) -> SystemState:
        """Current system state."""
        return self._state


# ----------------------------------------------------------------------
# Singleton Accessor
# ----------------------------------------------------------------------
def get_kimera_system() -> "KimeraSystem":
    """Returns the singleton instance of the KimeraSystem."""
    return KimeraSystem()

# Convenience instance for direct import if needed, but get_kimera_system is preferred
kimera_singleton = get_kimera_system()

__all__ = [
    "KimeraSystem",
    "kimera_singleton",
    "SystemState",
] 