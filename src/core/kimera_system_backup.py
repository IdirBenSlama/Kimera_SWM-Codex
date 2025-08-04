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

import inspect  # Add import for inspect
import logging
import platform
import threading
from enum import Enum, auto
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# GPU System Integration
try:
    from src.core.gpu.gpu_integration import get_gpu_integration_system
    from src.core.gpu.gpu_manager import get_gpu_manager, is_gpu_available
    from src.engines.gpu.gpu_geoid_processor import get_gpu_geoid_processor
    from src.engines.gpu.gpu_thermodynamic_engine import get_gpu_thermodynamic_engine

    GPU_SYSTEM_AVAILABLE = True
    logger.info("GPU system imports successful")
except ImportError as e:
    logger.warning(f"GPU system not available: {e}")
    GPU_SYSTEM_AVAILABLE = False

# Legacy GPU Foundation fallback
try:
    from src.utils.gpu_foundation import GPUFoundation

    GPU_FOUNDATION_AVAILABLE = True
except ImportError:
    GPUFoundation = None
    GPU_FOUNDATION_AVAILABLE = False


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
                        # Legacy GPU Foundation
                        cls._instance._gpu_foundation = None
                        # New GPU System Components
                        cls._instance._gpu_manager = None
                        cls._instance._gpu_integration_system = None
                        cls._instance._gpu_geoid_processor = None
                        cls._instance._gpu_thermodynamic_engine = None
                        cls._instance._gpu_acceleration_enabled = False
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
                    cls._initialization_event.wait(timeout=10)  # 10 second timeout
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

        # GPU System Detection and Initialization
        self._initialize_gpu_system()

        # Legacy GPU Foundation (fallback)
        self._initialize_legacy_gpu_foundation()

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
        self._initialize_cognitive_architecture_core()
        self._initialize_understanding_engine()
        self._initialize_human_interface()
        # self._initialize_cognitive_security_orchestrator()  # TODO: Fix syntax errors in dependencies
        self._initialize_linguistic_intelligence_engine()
        self._initialize_contradiction_engine()
        self._initialize_thermodynamics_engine()
        self._initialize_enhanced_thermodynamic_scheduler()
        self._initialize_quantum_cognitive_engine()
        self._initialize_ethical_reasoning_engine()
        self._initialize_unsupervised_cognitive_learning_engine()
        self._initialize_complexity_analysis_engine()
        self._initialize_quantum_field_engine()
        self._initialize_gpu_cryptographic_engine()
        self._initialize_thermodynamic_integration()
        self._initialize_unified_thermodynamic_integration()
        self._initialize_revolutionary_thermodynamic_engines()
        self._initialize_unified_thermodynamic_tcse()
        self._initialize_spde_engine()
        self._initialize_cognitive_cycle_engine()
        self._initialize_meta_insight_engine()
        self._initialize_proactive_detector()
        self._initialize_revolutionary_intelligence_engine()
        self._initialize_geoid_scar_manager()
        self._initialize_system_monitor()
        self._initialize_ethical_governor()

        # GPU System Final Integration
        self._finalize_gpu_integration()

        # Mark initialization as complete
        with self.__class__._lock:
            self._state = SystemState.RUNNING
            self.__class__._initialization_complete = True

        logger.info("KimeraSystem initialised successfully - state: %s", self._state)

        # Complete async initializations if needed
        self._complete_async_initializations()

    def _complete_async_initializations(self) -> None:
        """Complete any pending async initializations"""
        try:
            import asyncio

            # Check if there are components that need async initialization
            thermo_integration = self.get_component(
                "revolutionary_thermodynamic_engines"
            )
            unified_system = self.get_component("unified_thermodynamic_tcse")

            if (
                thermo_integration
                and thermo_integration != "initializing"
                and hasattr(thermo_integration, "initialize_all_engines")
            ):

                # Schedule async initialization for later
                logger.info(
                    "🔥 Revolutionary Thermodynamic Engines ready for async initialization"
                )

            if (
                unified_system
                and unified_system != "initializing"
                and hasattr(unified_system, "initialize_complete_system")
            ):

                # Schedule async initialization for later
                logger.info(
                    "🌡️ Unified Thermodynamic + TCSE System ready for async initialization"
                )

        except Exception as exc:
            logger.warning(f"⚠️ Async initialization setup failed: {exc}")

    def _initialize_vault_manager(self) -> None:
        """Initialize the VaultManager subsystem."""
        try:
            from src.vault.vault_manager import VaultManager

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
            from src.core import embedding_utils

            self._set_component(
                "embedding_model", True
            )  # Placeholder - actual model loaded in embedding_utils
            logger.info("Embedding model initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import embedding utilities: %s", exc)
            self._set_component("embedding_model", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize embedding model: %s", exc, exc_info=True
            )
            self._set_component("embedding_model", None)

    def _initialize_human_interface(self) -> None:
        """Initialize the Human Interface subsystem for human-readable system outputs."""
        try:
            from src.engines.human_interface import ResponseMode, create_human_interface

            # Human Interface is synchronous, no need for asyncio handling
            interface = create_human_interface(mode=ResponseMode.HYBRID)
            self._set_component("human_interface", interface)
            logger.info(
                "👤 Human Interface initialized successfully - Human-readable outputs enabled"
            )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Human Interface: %s", exc)
            self._set_component("human_interface", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Human Interface: %s", exc, exc_info=True
            )
            self._set_component("human_interface", None)

    def _initialize_cognitive_security_orchestrator(self) -> None:
        """Initialize the Cognitive Security Orchestrator for comprehensive data protection."""
        try:
            from src.engines.cognitive_security_orchestrator import (
                CognitiveSecurityOrchestrator,
                CognitiveSecurityPolicy,
            )

            # Create security policy with enhanced settings for production
            policy = CognitiveSecurityPolicy(
                default_level="enhanced",
                use_homomorphic=True,
                use_quantum_resistant=True,
                require_gdpr=True,
                audit_logging=True,
            )

            # Get GPU device ID from existing GPU system if available
            device_id = 0
            if self._gpu_acceleration_enabled and self._gpu_manager:
                try:
                    device_info = self._gpu_manager.get_device_info()
                    device_id = device_info.get("device_id", 0)
                except Exception as e:
                    logger.error(f"Error in kimera_system.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling

            orchestrator = CognitiveSecurityOrchestrator(
                policy=policy, device_id=device_id
            )
            self._set_component("cognitive_security_orchestrator", orchestrator)
            logger.info(
                "🔒 Cognitive Security Orchestrator initialized successfully - Comprehensive cognitive data protection enabled"
            )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Cognitive Security Orchestrator: %s", exc)
            self._set_component("cognitive_security_orchestrator", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Cognitive Security Orchestrator: %s",
                exc,
                exc_info=True,
            )
            self._set_component("cognitive_security_orchestrator", None)

    def _initialize_linguistic_intelligence_engine(self) -> None:
        """Initialize the Linguistic Intelligence Engine subsystem."""
        try:
            # Use asyncio to initialize the engine properly
            import asyncio

            from src.engines.linguistic_intelligence_engine import get_linguistic_engine

            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in a running loop, schedule the initialization
                future = asyncio.ensure_future(get_linguistic_engine())
                # For now, we'll mark as initialized and let it initialize lazily
                self._set_component("linguistic_intelligence_engine", "initializing")
                logger.info("Linguistic Intelligence Engine initialization scheduled")
            except RuntimeError:
                # No event loop running, create one for initialization
                engine = asyncio.run(get_linguistic_engine())
                self._set_component("linguistic_intelligence_engine", engine)
                logger.info("Linguistic Intelligence Engine initialized successfully")

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Linguistic Intelligence Engine: %s", exc)
            self._set_component("linguistic_intelligence_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Linguistic Intelligence Engine: %s",
                exc,
                exc_info=True,
            )
            self._set_component("linguistic_intelligence_engine", None)

    def _initialize_cognitive_architecture_core(self) -> None:
        """Initialize the Cognitive Architecture Core subsystem."""
        try:
            # Use asyncio to initialize the cognitive architecture properly
            import asyncio

            from src.core.cognitive_architecture_core import get_cognitive_architecture

            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in a running loop, schedule the initialization
                future = asyncio.ensure_future(get_cognitive_architecture())
                # For now, we'll mark as initialized and let it initialize lazily
                self._set_component("cognitive_architecture_core", "initializing")
                logger.info("Cognitive Architecture Core initialization scheduled")
            except RuntimeError:
                # No event loop running, create one for initialization
                architecture = asyncio.run(get_cognitive_architecture())
                self._set_component("cognitive_architecture_core", architecture)
                logger.info("Cognitive Architecture Core initialized successfully")

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Cognitive Architecture Core: %s", exc)
            self._set_component("cognitive_architecture_core", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Cognitive Architecture Core: %s",
                exc,
                exc_info=True,
            )
            self._set_component("cognitive_architecture_core", None)

    def _initialize_understanding_engine(self) -> None:
        """Initialize the Understanding Engine subsystem for genuine understanding capabilities."""
        try:
            # Use asyncio to initialize the engine properly
            import asyncio

            from src.engines.understanding_engine import create_understanding_engine

            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in a running loop, schedule the initialization
                future = asyncio.ensure_future(create_understanding_engine())
                # For now, we'll mark as initialized and let it initialize lazily
                self._set_component("understanding_engine", "initializing")
                logger.info("🧠 Understanding Engine initialization scheduled")
            except RuntimeError:
                # No event loop running, create one for initialization
                engine = asyncio.run(create_understanding_engine())
                self._set_component("understanding_engine", engine)
                logger.info(
                    "🧠 Understanding Engine initialized successfully - Genuine understanding capabilities enabled"
                )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Understanding Engine: %s", exc)
            self._set_component("understanding_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Understanding Engine: %s", exc, exc_info=True
            )
            self._set_component("understanding_engine", None)

    def _initialize_contradiction_engine(self) -> None:
        """Initialize the contradiction engine subsystem."""
        try:
            from src.engines.contradiction_engine import ContradictionEngine

            engine = ContradictionEngine(tension_threshold=0.4)
            self._set_component("contradiction_engine", engine)
            logger.info("Contradiction engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import ContradictionEngine: %s", exc)
            self._set_component("contradiction_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize contradiction engine: %s", exc, exc_info=True
            )
            self._set_component("contradiction_engine", None)

    def _initialize_thermodynamics_engine(self) -> None:
        """Initialize the thermodynamics engine subsystem."""
        try:
            from src.engines.foundational_thermodynamic_engine import (
                FoundationalThermodynamicEngine,
            )

            engine = FoundationalThermodynamicEngine()
            self._set_component("thermodynamics_engine", engine)
            logger.info("Thermodynamic engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import FoundationalThermodynamicEngine: %s", exc)
            self._set_component("thermodynamics_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize thermodynamics engine: %s", exc, exc_info=True
            )
            self._set_component("thermodynamics_engine", None)

    def _initialize_enhanced_thermodynamic_scheduler(self) -> None:
        """Initialize the Enhanced Thermodynamic Scheduler subsystem."""
        try:
            # Use asyncio to initialize the scheduler properly
            import asyncio

            from src.engines.thermodynamic_scheduler import (
                get_enhanced_thermodynamic_scheduler,
            )

            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in a running loop, schedule the initialization
                future = asyncio.ensure_future(get_enhanced_thermodynamic_scheduler())
                # For now, we'll mark as initialized and let it initialize lazily
                self._set_component("enhanced_thermodynamic_scheduler", "initializing")
                logger.info(
                    "🌡️ Enhanced Thermodynamic Scheduler initialization scheduled"
                )
            except RuntimeError:
                # No event loop running, create one for initialization
                scheduler = asyncio.run(get_enhanced_thermodynamic_scheduler())
                self._set_component("enhanced_thermodynamic_scheduler", scheduler)
                logger.info(
                    "🌡️ Enhanced Thermodynamic Scheduler initialized successfully - Physics-based optimization enabled"
                )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error(
                "❌ Failed to import Enhanced Thermodynamic Scheduler: %s", exc
            )
            self._set_component("enhanced_thermodynamic_scheduler", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Enhanced Thermodynamic Scheduler: %s",
                exc,
                exc_info=True,
            )
            self._set_component("enhanced_thermodynamic_scheduler", None)

    def _initialize_quantum_cognitive_engine(self) -> None:
        """Initialize the Quantum Cognitive Engine subsystem."""
        try:
            from src.engines.quantum_cognitive_engine import (
                initialize_quantum_cognitive_engine,
            )

            # Configure based on GPU availability
            gpu_acceleration = self._gpu_acceleration_enabled
            num_qubits = (
                20 if gpu_acceleration else 10
            )  # Scale based on available resources

            engine = initialize_quantum_cognitive_engine(
                num_qubits=num_qubits, gpu_acceleration=gpu_acceleration
            )
            self._set_component("quantum_cognitive_engine", engine)
            logger.info(
                "⚛️ Quantum Cognitive Engine initialized successfully - Quantum-enhanced cognition enabled"
            )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Quantum Cognitive Engine: %s", exc)
            self._set_component("quantum_cognitive_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Quantum Cognitive Engine: %s",
                exc,
                exc_info=True,
            )
            self._set_component("quantum_cognitive_engine", None)

    def _initialize_ethical_reasoning_engine(self) -> None:
        """Initialize the Ethical Reasoning Engine subsystem."""
        try:
            import asyncio

            from src.engines.ethical_reasoning_engine import (
                create_ethical_reasoning_engine,
            )

            # Configure based on requirements
            try:
                loop = asyncio.get_running_loop()
                # If we're in a running loop, schedule the initialization
                future = asyncio.ensure_future(create_ethical_reasoning_engine())
                self._set_component("ethical_reasoning_engine", "initializing")
                logger.info("⚖️ Ethical Reasoning Engine initialization scheduled")
            except RuntimeError:
                # No event loop running, create one for initialization
                engine = asyncio.run(create_ethical_reasoning_engine())
                self._set_component("ethical_reasoning_engine", engine)
                logger.info(
                    "⚖️ Ethical Reasoning Engine initialized successfully - Advanced ethical decision-making enabled"
                )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Ethical Reasoning Engine: %s", exc)
            self._set_component("ethical_reasoning_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Ethical Reasoning Engine: %s",
                exc,
                exc_info=True,
            )
            self._set_component("ethical_reasoning_engine", None)

    def _initialize_unsupervised_cognitive_learning_engine(self) -> None:
        """Initialize the Unsupervised Cognitive Learning Engine subsystem."""
        try:
            from src.engines.cognitive_field_dynamics import CognitiveFieldDynamics
            from src.engines.unsupervised_cognitive_learning_engine import (
                UnsupervisedCognitiveLearningEngine,
            )

            # Create cognitive field engine for the learning engine
            cognitive_field = CognitiveFieldDynamics(dimension=1024)

            # Create the unsupervised learning engine
            learning_engine = UnsupervisedCognitiveLearningEngine(
                cognitive_field_engine=cognitive_field,
                learning_sensitivity=0.15,
                emergence_threshold=0.7,
                insight_threshold=0.85,
            )

            self._set_component(
                "unsupervised_cognitive_learning_engine", learning_engine
            )
            logger.info(
                "🧠 Unsupervised Cognitive Learning Engine initialized successfully - Revolutionary physics-based learning enabled"
            )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error(
                "❌ Failed to import Unsupervised Cognitive Learning Engine: %s", exc
            )
            self._set_component("unsupervised_cognitive_learning_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Unsupervised Cognitive Learning Engine: %s",
                exc,
                exc_info=True,
            )
            self._set_component("unsupervised_cognitive_learning_engine", None)

    def _initialize_complexity_analysis_engine(self) -> None:
        """Initialize the Complexity Analysis Engine subsystem."""
        try:
            import asyncio

            from src.engines.complexity_analysis_engine import (
                create_complexity_analysis_engine,
            )

            # Configure based on requirements
            try:
                loop = asyncio.get_running_loop()
                # If we're in a running loop, schedule the initialization
                future = asyncio.ensure_future(create_complexity_analysis_engine())
                self._set_component("complexity_analysis_engine", "initializing")
                logger.info("🔬 Complexity Analysis Engine initialization scheduled")
            except RuntimeError:
                # No event loop running, create one for initialization
                engine = asyncio.run(create_complexity_analysis_engine())
                self._set_component("complexity_analysis_engine", engine)
                logger.info(
                    "🔬 Complexity Analysis Engine initialized successfully - Advanced information integration analysis enabled"
                )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Complexity Analysis Engine: %s", exc)
            self._set_component("complexity_analysis_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Complexity Analysis Engine: %s",
                exc,
                exc_info=True,
            )
            self._set_component("complexity_analysis_engine", None)

    def _initialize_quantum_field_engine(self) -> None:
        """Initialize the Quantum Field Engine subsystem."""
        try:
            from src.engines.quantum_field_engine import create_quantum_field_engine

            # Configure based on GPU availability
            device = "cuda" if self._gpu_acceleration_enabled else "cpu"
            dimension = 20  # Good balance for quantum field complexity

            engine = create_quantum_field_engine(dimension=dimension, device=device)
            self._set_component("quantum_field_engine", engine)
            logger.info(
                "⚛️ Quantum Field Engine initialized successfully - Quantum field modeling of cognitive states enabled"
            )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Quantum Field Engine: %s", exc)
            self._set_component("quantum_field_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Quantum Field Engine: %s", exc, exc_info=True
            )
            self._set_component("quantum_field_engine", None)

    def _initialize_gpu_cryptographic_engine(self) -> None:
        """Initialize the GPU Cryptographic Engine subsystem."""
        try:
            # Check for GPU availability
            import cupy as cp

            from src.engines.gpu_cryptographic_engine import GPUCryptographicEngine

            if cp.cuda.is_available():
                device_count = cp.cuda.runtime.getDeviceCount()
                device_id = 0 if device_count > 0 else None

                if device_id is not None:
                    crypto_engine = GPUCryptographicEngine(device_id=device_id)
                    self._set_component("gpu_cryptographic_engine", crypto_engine)
                    logger.info("🔐 GPU Cryptographic Engine successfully initialized")
                else:
                    logger.warning(
                        "🔐 No CUDA devices available, GPU Cryptographic Engine disabled"
                    )
                    self._set_component("gpu_cryptographic_engine", None)
            else:
                logger.warning(
                    "🔐 CUDA not available, GPU Cryptographic Engine disabled"
                )
                self._set_component("gpu_cryptographic_engine", None)

        except Exception as e:
            logger.error(f"❌ Failed to initialize GPU Cryptographic Engine: {e}")
            self._set_component("gpu_cryptographic_engine", None)

    def _initialize_thermodynamic_integration(self) -> None:
        """Initialize the core Thermodynamic Integration subsystem."""
        try:
            import asyncio

            from src.engines.thermodynamic_integration import (
                get_thermodynamic_integration,
                initialize_thermodynamics,
            )

            # Get the thermodynamic integration instance
            integration = get_thermodynamic_integration()

            # Schedule async initialization if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # If we're in a running loop, schedule the initialization
                future = asyncio.ensure_future(initialize_thermodynamics())
                self._set_component("thermodynamic_integration", "initializing")
                logger.info(
                    "🔥 Thermodynamic Integration system initializing asynchronously..."
                )
            except RuntimeError:
                # No event loop running, set the integration directly
                self._set_component("thermodynamic_integration", integration)
                logger.info(
                    "🔥 Thermodynamic Integration system initialized (sync mode)"
                )

        except Exception as e:
            logger.error(f"❌ Failed to initialize Thermodynamic Integration: {e}")
            self._set_component("thermodynamic_integration", None)

    def _initialize_unified_thermodynamic_integration(self) -> None:
        """Initialize the Unified Thermodynamic + TCSE Integration subsystem."""
        try:
            from src.engines.unified_thermodynamic_integration import (
                UnifiedThermodynamicTCSE,
            )

            # Create the unified thermodynamic TCSE system
            unified_system = UnifiedThermodynamicTCSE(
                auto_start_monitoring=False,  # We'll control monitoring from the core system
                consciousness_threshold=0.75,
                thermal_regulation_enabled=True,
                energy_management_enabled=True,
            )

            self._set_component("unified_thermodynamic_integration", unified_system)
            logger.info("🌡️ Unified Thermodynamic + TCSE Integration system initialized")

        except Exception as e:
            logger.error(
                f"❌ Failed to initialize Unified Thermodynamic Integration: {e}"
            )
            self._set_component("unified_thermodynamic_integration", None)

    def _initialize_revolutionary_thermodynamic_engines(self) -> None:
        """Initialize all revolutionary thermodynamic engines."""
        try:
            import asyncio

            from src.engines.thermodynamic_integration import (
                get_thermodynamic_integration,
            )

            # Get the thermodynamic integration system
            thermo_integration = get_thermodynamic_integration()

            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # Schedule async initialization
                future = asyncio.ensure_future(
                    thermo_integration.initialize_all_engines()
                )
                self._set_component(
                    "revolutionary_thermodynamic_engines", "initializing"
                )
                logger.info(
                    "Revolutionary Thermodynamic Engines initialization scheduled"
                )
            except RuntimeError:
                # No event loop running, mark for later async initialization
                self._set_component(
                    "revolutionary_thermodynamic_engines", thermo_integration
                )
                logger.info(
                    "Revolutionary Thermodynamic Engines ready for async initialization"
                )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error(
                "❌ Failed to import revolutionary thermodynamic engines: %s", exc
            )
            self._set_component("revolutionary_thermodynamic_engines", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize revolutionary thermodynamic engines: %s",
                exc,
                exc_info=True,
            )
            self._set_component("revolutionary_thermodynamic_engines", None)

    def _initialize_unified_thermodynamic_tcse(self) -> None:
        """Initialize the unified thermodynamic + TCSE integration system."""
        try:
            import asyncio

            from src.engines.unified_thermodynamic_integration import (
                get_unified_thermodynamic_tcse,
            )

            # Get the unified system
            unified_system = get_unified_thermodynamic_tcse()

            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # Schedule async initialization
                future = asyncio.ensure_future(
                    unified_system.initialize_complete_system()
                )
                self._set_component("unified_thermodynamic_tcse", "initializing")
                logger.info(
                    "Unified Thermodynamic + TCSE System initialization scheduled"
                )
            except RuntimeError:
                # No event loop running, mark for later async initialization
                self._set_component("unified_thermodynamic_tcse", unified_system)
                logger.info(
                    "Unified Thermodynamic + TCSE System ready for async initialization"
                )

        except (ImportError, ModuleNotFoundError) as exc:
            logger.error(
                "❌ Failed to import unified thermodynamic TCSE system: %s", exc
            )
            self._set_component("unified_thermodynamic_tcse", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize unified thermodynamic TCSE system: %s",
                exc,
                exc_info=True,
            )
            self._set_component("unified_thermodynamic_tcse", None)

    def _initialize_spde_engine(self) -> None:
        """Initialize the SPDE engine subsystem."""
        try:
            from src.engines.spde_engine import create_spde_engine

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
            from src.engines.cognitive_cycle_engine import create_cognitive_cycle_engine

            engine = create_cognitive_cycle_engine(device=self._device)
            self._set_component("cognitive_cycle_engine", engine)
            logger.info("Cognitive Cycle engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Cognitive Cycle engine: %s", exc)
            self._set_component("cognitive_cycle_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Cognitive Cycle engine: %s", exc, exc_info=True
            )
            self._set_component("cognitive_cycle_engine", None)

    def _initialize_meta_insight_engine(self) -> None:
        """Initialize the Meta Insight engine subsystem."""
        try:
            from src.engines.meta_insight_engine import create_meta_insight_engine

            # Configure device based on GPU availability
            device = self._device if self._gpu_acceleration_enabled else "cpu"

            engine = create_meta_insight_engine(device=device)
            self._set_component("meta_insight_engine", engine)
            logger.info(
                "🧠 Meta Insight Engine initialized successfully - Higher-order cognitive processing enabled"
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Meta Insight engine: %s", exc)
            self._set_component("meta_insight_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Meta Insight engine: %s", exc, exc_info=True
            )
            self._set_component("meta_insight_engine", None)

    def _initialize_proactive_detector(self) -> None:
        """Initialize the Proactive Detector subsystem."""
        try:
            from src.engines.proactive_detector import create_proactive_detector

            engine = create_proactive_detector(device=self._device)
            self._set_component("proactive_detector", engine)
            logger.info("Proactive Detector initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Proactive Detector: %s", exc)
            self._set_component("proactive_detector", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Proactive Detector: %s", exc, exc_info=True
            )
            self._set_component("proactive_detector", None)

    def _initialize_revolutionary_intelligence_engine(self) -> None:
        """Initialize the Revolutionary Intelligence Engine subsystem."""
        try:
            from src.engines.revolutionary_intelligence_engine import (
                create_revolutionary_intelligence_engine,
            )

            # Configure device based on GPU availability
            device = self._device if self._gpu_acceleration_enabled else "cpu"

            engine = create_revolutionary_intelligence_engine(device=device)
            self._set_component("revolutionary_intelligence_engine", engine)
            logger.info(
                "🚀 Revolutionary Intelligence Engine initialized successfully - Advanced AI capabilities enabled"
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error(
                "❌ Failed to import Revolutionary Intelligence Engine: %s", exc
            )
            self._set_component("revolutionary_intelligence_engine", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Revolutionary Intelligence Engine: %s",
                exc,
                exc_info=True,
            )
            self._set_component("revolutionary_intelligence_engine", None)

    def _initialize_geoid_scar_manager(self) -> None:
        """Initialize the Geoid SCAR Manager subsystem."""
        try:
            from src.engines.geoid_scar_manager import GeoidScarManager

            manager = GeoidScarManager()
            self._set_component("geoid_scar_manager", manager)
            logger.info("Geoid SCAR Manager initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import GeoidScarManager: %s", exc)
            self._set_component("geoid_scar_manager", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Geoid SCAR Manager: %s", exc, exc_info=True
            )
            self._set_component("geoid_scar_manager", None)

    def _initialize_system_monitor(self) -> None:
        """Initialize the System Monitor subsystem."""
        try:
            from src.monitoring.system_monitor import SystemMonitor

            monitor = SystemMonitor()
            self._set_component("system_monitor", monitor)
            logger.info("System Monitor initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import SystemMonitor: %s", exc)
            self._set_component("system_monitor", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize System Monitor: %s", exc, exc_info=True
            )
            self._set_component("system_monitor", None)

    def _initialize_ethical_governor(self) -> None:
        """Initialize the Ethical Governor subsystem."""
        try:
            from src.governance.ethical_governor import EthicalGovernor

            governor = EthicalGovernor()
            self._set_component("ethical_governor", governor)
            logger.info("Ethical Governor initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import EthicalGovernor: %s", exc)
            self._set_component("ethical_governor", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Ethical Governor: %s", exc, exc_info=True
            )
            self._set_component("ethical_governor", None)

    def _initialize_exception_handling(self) -> None:
        """Initialize the Exception Handling subsystem."""
        try:
            from src.core import exception_handling

            self._set_component("exception_handling", exception_handling.error_registry)
            logger.info("Exception Handling initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Exception Handling: %s", exc)
            self._set_component("exception_handling", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Exception Handling: %s", exc, exc_info=True
            )
            self._set_component("exception_handling", None)

    def _initialize_error_recovery(self) -> None:
        """Initialize the Error Recovery subsystem."""
        try:
            from src.core.error_recovery import get_error_recovery_manager

            manager = get_error_recovery_manager()
            self._set_component("error_recovery", manager)
            logger.info("Error Recovery initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Error Recovery: %s", exc)
            self._set_component("error_recovery", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Error Recovery: %s", exc, exc_info=True
            )
            self._set_component("error_recovery", None)

    def _initialize_performance_manager(self) -> None:
        """Initialize the Performance Manager subsystem."""
        try:
            from src.core.performance_integration import PerformanceManager

            manager = PerformanceManager()
            self._set_component("performance_manager", manager)
            logger.info("Performance Manager initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Performance Manager: %s", exc)
            self._set_component("performance_manager", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Performance Manager: %s", exc, exc_info=True
            )
            self._set_component("performance_manager", None)

    def _initialize_database_optimization(self) -> None:
        """Initialize the Database Optimization subsystem."""
        try:
            from src.core.database_optimization import DatabaseConnectionPool

            db_pool = DatabaseConnectionPool()
            self._set_component("database_optimization", db_pool)
            logger.info("Database Optimization initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Database Optimization: %s", exc)
            self._set_component("database_optimization", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Database Optimization: %s", exc, exc_info=True
            )
            self._set_component("database_optimization", None)

    def _initialize_context_supremacy(self) -> None:
        """Initialize the Context Supremacy Engine subsystem."""
        try:
            from src.core.context_supremacy import ContextSupremacyEngine

            engine = ContextSupremacyEngine()
            self._set_component("context_supremacy", engine)
            logger.info("Context Supremacy Engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Context Supremacy Engine: %s", exc)
            self._set_component("context_supremacy", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Context Supremacy Engine: %s",
                exc,
                exc_info=True,
            )
            self._set_component("context_supremacy", None)

    def _initialize_statistical_modeling(self) -> None:
        """Initialize the Statistical Modeling subsystem."""
        try:
            from src.core.statistical_modeling import StatisticalModelingEngine

            engine = StatisticalModelingEngine()
            self._set_component("statistical_modeling", engine)
            logger.info("Statistical Modeling Engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Statistical Modeling Engine: %s", exc)
            self._set_component("statistical_modeling", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Statistical Modeling Engine: %s",
                exc,
                exc_info=True,
            )
            self._set_component("statistical_modeling", None)

    def _initialize_universal_compassion(self) -> None:
        """Initialize the Universal Compassion Engine subsystem."""
        try:
            from src.core.universal_compassion import UniversalCompassionEngine

            engine = UniversalCompassionEngine()
            self._set_component("universal_compassion", engine)
            logger.info("Universal Compassion Engine initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Universal Compassion Engine: %s", exc)
            self._set_component("universal_compassion", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Universal Compassion Engine: %s",
                exc,
                exc_info=True,
            )
            self._set_component("universal_compassion", None)

    def _initialize_cache_layer(self) -> None:
        """Initialize the Cache Layer subsystem."""
        try:
            from src.core.cache_layer import CacheManager

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
            from src.core.dependency_injection import ServiceContainer

            container = ServiceContainer()
            self._set_component("dependency_injection", container)
            logger.info("Dependency Injection initialized successfully")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.error("❌ Failed to import Dependency Injection: %s", exc)
            self._set_component("dependency_injection", None)
        except Exception as exc:
            logger.error(
                "❌ Failed to initialize Dependency Injection: %s", exc, exc_info=True
            )
            self._set_component("dependency_injection", None)

    def _initialize_task_manager(self) -> None:
        """Initialize the Task Manager subsystem."""
        try:
            from src.core.task_manager import TaskManager

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

        # Shutdown unified thermodynamic systems first (order matters)
        try:
            unified_system = self.get_component("unified_thermodynamic_tcse")
            if unified_system and hasattr(unified_system, "shutdown_unified_system"):
                await unified_system.shutdown_unified_system()
                logger.info("✅ Unified Thermodynamic + TCSE System shutdown complete")
        except Exception as exc:
            logger.error(
                f"❌ Error shutting down unified thermodynamic system: {exc}",
                exc_info=True,
            )

        try:
            thermo_integration = self.get_component(
                "revolutionary_thermodynamic_engines"
            )
            if thermo_integration and hasattr(thermo_integration, "shutdown_all"):
                await thermo_integration.shutdown_all()
                logger.info("✅ Revolutionary Thermodynamic Engines shutdown complete")
        except Exception as exc:
            logger.error(
                f"❌ Error shutting down revolutionary thermodynamic engines: {exc}",
                exc_info=True,
            )

        # Create a snapshot of components to avoid modification during iteration
        components_snapshot = {}
        for name in list(self._components.keys()):
            component = self.get_component(name)
            if component:
                components_snapshot[name] = component

        # Shutdown remaining subsystems, handling both sync and async shutdown methods
        for component_name, component in components_snapshot.items():
            if component and hasattr(component, "shutdown"):
                try:
                    shutdown_method = getattr(component, "shutdown")
                    if inspect.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
                    logger.info(f"✅ {component_name} shutdown complete")
                except Exception as exc:
                    logger.error(
                        f"❌ Error shutting down {component_name}: {exc}", exc_info=True
                    )

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

    def get_understanding_engine(self):
        """Get the Understanding Engine instance (thread-safe)."""
        return self.get_component("understanding_engine")

    def get_human_interface(self):
        """Get the Human Interface instance (thread-safe)."""
        return self.get_component("human_interface")

    def get_cognitive_security_orchestrator(self):
        """Get the Cognitive Security Orchestrator instance (thread-safe)."""
        return self.get_component("cognitive_security_orchestrator")

    def get_contradiction_engine(self):
        """Get the contradiction engine instance (thread-safe)."""
        return self.get_component("contradiction_engine")

    def get_thermodynamic_engine(self):
        """Get the thermodynamic engine instance (thread-safe)."""
        return self.get_component("thermodynamics_engine")

    def get_enhanced_thermodynamic_scheduler(self):
        """Get the Enhanced Thermodynamic Scheduler instance (thread-safe)."""
        return self.get_component("enhanced_thermodynamic_scheduler")

    def get_quantum_cognitive_engine(self):
        """Get the Quantum Cognitive Engine instance (thread-safe)."""
        return self.get_component("quantum_cognitive_engine")

    def get_ethical_reasoning_engine(self):
        """Get the Ethical Reasoning Engine instance (thread-safe)."""
        return self.get_component("ethical_reasoning_engine")

    def get_unsupervised_cognitive_learning_engine(self):
        """Get the Unsupervised Cognitive Learning Engine instance (thread-safe)."""
        return self.get_component("unsupervised_cognitive_learning_engine")

    def get_complexity_analysis_engine(self):
        """Get the Complexity Analysis Engine instance (thread-safe)."""
        return self.get_component("complexity_analysis_engine")

    def get_quantum_field_engine(self):
        """Get the Quantum Field Engine instance (thread-safe)."""
        return self.get_component("quantum_field_engine")

    def get_gpu_cryptographic_engine(self):
        """Get the GPU Cryptographic Engine instance (thread-safe)."""
        return self.get_component("gpu_cryptographic_engine")

    def get_thermodynamic_integration(self):
        """Get the Thermodynamic Integration system instance (thread-safe)."""
        return self.get_component("thermodynamic_integration")

    def get_unified_thermodynamic_integration(self):
        """Get the Unified Thermodynamic + TCSE Integration system instance (thread-safe)."""
        return self.get_component("unified_thermodynamic_integration")

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
        component_names = [
            "vault_manager",
            "gpu_foundation",
            "understanding_engine",
            "human_interface",
            "enhanced_thermodynamic_scheduler",
            "quantum_cognitive_engine",
            "revolutionary_intelligence_engine",
            "meta_insight_engine",
            "ethical_reasoning_engine",
            "unsupervised_cognitive_learning_engine",
            "complexity_analysis_engine",
            "quantum_field_engine",
            "gpu_cryptographic_engine",
            "thermodynamic_integration",
            "unified_thermodynamic_integration",
            "contradiction_engine",
            "thermodynamics_engine",
            "embedding_model",
            "geoid_scar_manager",
            "system_monitor",
            "ethical_governor",
        ]
        component_status = {}
        for name in component_names:
            component_status[name] = self.get_component(name) is not None

        return {
            "state": current_state,
            "components": component_status,
            "vault_manager_ready": component_status.get("vault_manager", False),
            "gpu_foundation_ready": component_status.get("gpu_foundation", False),
            "understanding_engine_ready": component_status.get(
                "understanding_engine", False
            ),
            "human_interface_ready": component_status.get("human_interface", False),
            "enhanced_thermodynamic_scheduler_ready": component_status.get(
                "enhanced_thermodynamic_scheduler", False
            ),
            "quantum_cognitive_engine_ready": component_status.get(
                "quantum_cognitive_engine", False
            ),
            "revolutionary_intelligence_engine_ready": component_status.get(
                "revolutionary_intelligence_engine", False
            ),
            "meta_insight_engine_ready": component_status.get(
                "meta_insight_engine", False
            ),
            "ethical_reasoning_engine_ready": component_status.get(
                "ethical_reasoning_engine", False
            ),
            "unsupervised_cognitive_learning_engine_ready": component_status.get(
                "unsupervised_cognitive_learning_engine", False
            ),
            "complexity_analysis_engine_ready": component_status.get(
                "complexity_analysis_engine", False
            ),
            "quantum_field_engine_ready": component_status.get(
                "quantum_field_engine", False
            ),
            "gpu_cryptographic_engine_ready": component_status.get(
                "gpu_cryptographic_engine", False
            ),
            "thermodynamic_integration_ready": component_status.get(
                "thermodynamic_integration", False
            ),
            "unified_thermodynamic_integration_ready": component_status.get(
                "unified_thermodynamic_integration", False
            ),
            "contradiction_engine_ready": component_status.get(
                "contradiction_engine", False
            ),
            "thermodynamic_engine_ready": component_status.get(
                "thermodynamics_engine", False
            ),
            "embedding_model_ready": component_status.get("embedding_model", False),
            "geoid_scar_manager_ready": component_status.get(
                "geoid_scar_manager", False
            ),
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
                "components": {
                    name: comp is not None for name, comp in self._components.items()
                },
            }
        return self._components.get(key, default)

    # ------------------------------------------------------------------
    # GPU System Initialization Methods
    # ------------------------------------------------------------------

    def _initialize_gpu_system(self) -> None:
        """Initialize the comprehensive GPU acceleration system."""
        logger.info("🚀 Initializing GPU acceleration system...")

        if not GPU_SYSTEM_AVAILABLE:
            logger.warning(
                "⚠️ GPU system components not available - GPU acceleration disabled"
            )
            self._device = "cpu"
            self._gpu_acceleration_enabled = False
            return

        try:
            # Initialize GPU Manager
            if is_gpu_available():
                gpu_manager = get_gpu_manager()
                self._gpu_manager = gpu_manager
                self._set_component("gpu_manager", gpu_manager)

                # Set device based on GPU availability
                device_info = gpu_manager.get_device_info()
                self._device = f"cuda:{device_info.get('device_id', 0)}"
                self._gpu_acceleration_enabled = True

                logger.info(
                    f"✅ GPU Manager initialized - Device: {device_info.get('name', 'Unknown')}"
                )
                logger.info(
                    f"🔥 GPU Memory: {device_info.get('total_memory_gb', 0):.1f}GB"
                )
                logger.info(
                    f"⚡ Compute Capability: {device_info.get('compute_capability', (0, 0))}"
                )

                # Initialize GPU Integration System
                try:
                    integration_system = get_gpu_integration_system()
                    self._gpu_integration_system = integration_system
                    self._set_component("gpu_integration_system", integration_system)
                    logger.info("✅ GPU Integration System initialized")
                except Exception as exc:
                    logger.error(
                        f"❌ Failed to initialize GPU Integration System: {exc}"
                    )
                    self._set_component("gpu_integration_system", None)

                # Initialize GPU Geoid Processor
                try:
                    gpu_geoid_processor = get_gpu_geoid_processor()
                    self._gpu_geoid_processor = gpu_geoid_processor
                    self._set_component("gpu_geoid_processor", gpu_geoid_processor)
                    logger.info("✅ GPU Geoid Processor initialized")
                except Exception as exc:
                    logger.error(f"❌ Failed to initialize GPU Geoid Processor: {exc}")
                    self._set_component("gpu_geoid_processor", None)

                # Initialize GPU Thermodynamic Engine
                try:
                    gpu_thermo_engine = get_gpu_thermodynamic_engine()
                    self._gpu_thermodynamic_engine = gpu_thermo_engine
                    self._set_component("gpu_thermodynamic_engine", gpu_thermo_engine)
                    logger.info("✅ GPU Thermodynamic Engine initialized")
                except Exception as exc:
                    logger.error(
                        f"❌ Failed to initialize GPU Thermodynamic Engine: {exc}"
                    )
                    self._set_component("gpu_thermodynamic_engine", None)

                logger.info("🎉 GPU acceleration system fully operational!")

            else:
                logger.warning("⚠️ GPU hardware not available - falling back to CPU")
                self._device = "cpu"
                self._gpu_acceleration_enabled = False
                self._set_component("gpu_manager", None)
                self._set_component("gpu_integration_system", None)
                self._set_component("gpu_geoid_processor", None)
                self._set_component("gpu_thermodynamic_engine", None)

        except Exception as exc:
            logger.error(f"❌ GPU system initialization failed: {exc}", exc_info=True)
            self._device = "cpu"
            self._gpu_acceleration_enabled = False
            # Set all GPU components to None
            for component in [
                "gpu_manager",
                "gpu_integration_system",
                "gpu_geoid_processor",
                "gpu_thermodynamic_engine",
            ]:
                self._set_component(component, None)

    def _initialize_legacy_gpu_foundation(self) -> None:
        """Initialize legacy GPU Foundation for backward compatibility."""
        if GPU_FOUNDATION_AVAILABLE and not self._gpu_acceleration_enabled:
            try:
                gpu_found = GPUFoundation()
                self._gpu_foundation = gpu_found
                if (
                    self._device == "cpu"
                ):  # Only override if GPU system didn't set device
                    self._device = str(gpu_found.get_device())
                self._set_component("gpu_foundation", gpu_found)
                logger.info(
                    f"✅ Legacy GPU Foundation initialized - Device: {self._device}"
                )
            except (RuntimeError, ImportError, AttributeError) as exc:
                self._gpu_foundation = None
                self._set_component("gpu_foundation", None)
                logger.warning(f"⚠️ Legacy GPU Foundation failed ({exc})")
            except Exception as exc:
                self._gpu_foundation = None
                self._set_component("gpu_foundation", None)
                logger.error(f"❌ Legacy GPU Foundation error: {exc}", exc_info=True)
        else:
            self._gpu_foundation = None
            self._set_component("gpu_foundation", None)
            if not GPU_FOUNDATION_AVAILABLE:
                logger.debug("Legacy GPU Foundation not available")

    def _finalize_gpu_integration(self) -> None:
        """Finalize GPU system integration and start monitoring."""
        if self._gpu_acceleration_enabled and self._gpu_integration_system:
            try:
                # Start GPU monitoring if available
                import asyncio

                loop = None
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule monitoring start
                        asyncio.create_task(self._start_gpu_monitoring())
                        logger.info("🔄 GPU monitoring scheduled")
                except RuntimeError:
                    logger.debug(
                        "No event loop available - GPU monitoring will start with first async operation"
                    )

                logger.info("🏁 GPU system integration completed successfully")

            except Exception as exc:
                logger.error(f"❌ GPU integration finalization failed: {exc}")

    async def _start_gpu_monitoring(self) -> None:
        """Start GPU monitoring (async)."""
        try:
            if self._gpu_integration_system:
                await self._gpu_integration_system._start_monitoring()
                logger.info("📊 GPU monitoring started")
        except Exception as exc:
            logger.error(f"❌ Failed to start GPU monitoring: {exc}")

    # ------------------------------------------------------------------
    # GPU System Access Methods
    # ------------------------------------------------------------------

    def get_gpu_manager(self) -> Optional[Any]:
        """Get the GPU manager instance."""
        return self._gpu_manager

    def get_gpu_integration_system(self) -> Optional[Any]:
        """Get the GPU integration system instance."""
        return self._gpu_integration_system

    def get_gpu_geoid_processor(self) -> Optional[Any]:
        """Get the GPU geoid processor instance."""
        return self._gpu_geoid_processor

    def get_gpu_thermodynamic_engine(self) -> Optional[Any]:
        """Get the GPU thermodynamic engine instance."""
        return self._gpu_thermodynamic_engine

    def is_gpu_acceleration_enabled(self) -> bool:
        """Check if GPU acceleration is enabled and operational."""
        return self._gpu_acceleration_enabled

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
            "gpu_acceleration_enabled": self._gpu_acceleration_enabled,
            "components": list(self._components.keys()),
            "gpu_components": {
                "gpu_manager": self._gpu_manager is not None,
                "gpu_integration_system": self._gpu_integration_system is not None,
                "gpu_geoid_processor": self._gpu_geoid_processor is not None,
                "gpu_thermodynamic_engine": self._gpu_thermodynamic_engine is not None,
            },
        }

    # ------------------------------------------------------------------
    # Thermodynamic System Access Methods
    # ------------------------------------------------------------------

    def get_thermodynamic_integration(self):
        """Get the revolutionary thermodynamic integration system"""
        return self.get_component("revolutionary_thermodynamic_engines")

    def get_unified_thermodynamic_tcse(self):
        """Get the unified thermodynamic + TCSE system"""
        return self.get_component("unified_thermodynamic_tcse")

    async def initialize_thermodynamic_systems(self) -> bool:
        """Initialize thermodynamic systems asynchronously"""
        try:
            success = True

            # Initialize revolutionary thermodynamic engines
            thermo_integration = self.get_thermodynamic_integration()
            if thermo_integration and hasattr(
                thermo_integration, "initialize_all_engines"
            ):
                result = await thermo_integration.initialize_all_engines()
                if result:
                    logger.info(
                        "✅ Revolutionary Thermodynamic Engines async initialization complete"
                    )
                else:
                    logger.error(
                        "❌ Revolutionary Thermodynamic Engines async initialization failed"
                    )
                    success = False

            # Initialize unified system
            unified_system = self.get_unified_thermodynamic_tcse()
            if unified_system and hasattr(unified_system, "initialize_complete_system"):
                result = await unified_system.initialize_complete_system()
                if result:
                    logger.info(
                        "✅ Unified Thermodynamic + TCSE System async initialization complete"
                    )
                else:
                    logger.error(
                        "❌ Unified Thermodynamic + TCSE System async initialization failed"
                    )
                    success = False

            return success

        except Exception as e:
            logger.error(f"❌ Thermodynamic systems async initialization failed: {e}")
            return False

    def is_thermodynamic_systems_ready(self) -> bool:
        """Check if thermodynamic systems are ready"""
        thermo_integration = self.get_thermodynamic_integration()
        unified_system = self.get_unified_thermodynamic_tcse()

        thermo_ready = (
            thermo_integration
            and hasattr(thermo_integration, "engines_initialized")
            and thermo_integration.engines_initialized
        )

        unified_ready = (
            unified_system
            and hasattr(unified_system, "system_initialized")
            and unified_system.system_initialized
        )

        return thermo_ready and unified_ready

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
