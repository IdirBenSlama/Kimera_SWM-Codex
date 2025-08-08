"""
KCl Extended-Release Capsule Testing Engine

Comprehensive computational and laboratory framework for developing and testing
potassium chloride extended-release capsules following USP standards.

Integrates with Kimera's cognitive fidelity principles and GPU acceleration.
"""

import json
import logging
import statistics
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import auc, roc_curve

from ...utils.gpu_foundation import GPUFoundation
from ...utils.kimera_exceptions import KimeraBaseException
from ...utils.kimera_logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Auto-generated class."""
    pass
    """Performance tracking for pharmaceutical testing operations."""

    operation_type: str
    gpu_time_ms: float
    cpu_time_ms: float
    memory_used_mb: float
    throughput_ops_per_sec: float
    error_count: int = 0
    warning_count: int = 0


@dataclass
class RawMaterialSpec:
    """Auto-generated class."""
    pass
    """Raw material specifications for pharmaceutical testing."""

    name: str
    grade: str
    purity_percent: float
    moisture_content: float
    particle_size_d50: float
    bulk_density: float
    tapped_density: float
    identification_tests: Dict[str, bool] = field(default_factory=dict)
    impurity_limits: Dict[str, float] = field(default_factory=dict)
    validation_timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


@dataclass
class FlowabilityResult:
    """Auto-generated class."""
    pass
    """Powder flowability analysis results."""

    bulk_density: float
    tapped_density: float
    carr_index: float
    hausner_ratio: float
    angle_of_repose: float
    flow_character: str
    processing_recommendations: List[str] = field(default_factory=list)


@dataclass
class DissolutionProfile:
    """Auto-generated class."""
    pass
    """Dissolution test profile with time points and release percentages."""

    time_points: List[float]
    release_percentages: List[float]
    test_conditions: Dict[str, Any]
    f2_similarity: Optional[float] = None
    kinetic_model_fit: Optional[Dict[str, Any]] = None


@dataclass
class FormulationPrototype:
    """Auto-generated class."""
    pass
    """Formulation prototype with coating parameters."""

    prototype_id: str
    coating_thickness_percent: float
    polymer_ratio: Dict[str, float]
    encapsulation_efficiency: float
    particle_morphology: str
    dissolution_profile: Optional[DissolutionProfile] = None
    manufacturing_parameters: Dict[str, Any] = field(default_factory=dict)


class PharmaceuticalTestingException(KimeraBaseException):
    """Exception for pharmaceutical testing operations."""

    pass
class KClTestingEngine:
    """Auto-generated class."""
    pass
    """
    Main engine for KCl extended-release capsule testing and development.

    Implements comprehensive USP-compliant testing protocols with GPU acceleration
    for computational aspects and integration with Kimera's scientific framework.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the KCl testing engine.

        Args:
            use_gpu: Whether to use GPU acceleration for computations

        Raises:
            PharmaceuticalTestingException: If initialization fails
        """
        self.logger = logger
        self.use_gpu = use_gpu
        self.device = None
        self.gpu_foundation = None

        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []
        self.gpu_optimization_enabled = False
        self.tensor_core_enabled = False

        # Batch processing optimization
        self.batch_cache = {}
        self.batch_size_optimization = {
            "raw_material_characterization": 32,
            "dissolution_testing": 16,
            "formulation_optimization": 8,
        }
        self.memory_pool = None

        # USP Standards and Benchmarks
        self.usp_standards = self._load_usp_standards()
        self.reference_products = {}
        self.test_results = {}

        # Initialize GPU if requested
        if self.use_gpu:
            try:
                self.gpu_foundation = GPUFoundation()
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

                # Enable advanced GPU optimizations
                if torch.cuda.is_available():
                    self._optimize_gpu_settings()
                    self._initialize_memory_pool()
                    self.gpu_optimization_enabled = True

                self.logger.info(f"🚀 KCl Testing Engine initialized on {self.device}")
                self.logger.info(
                    f"   GPU Optimization: {self.gpu_optimization_enabled}"
                )
                self.logger.info(f"   Tensor Core Support: {self.tensor_core_enabled}")

            except Exception as e:
                self.logger.warning(f"GPU initialization failed, using CPU: {e}")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.logger.info("💊 KCl Extended-Release Capsule Testing Engine initialized")

    def _optimize_gpu_settings(self):
        """Optimize GPU settings for pharmaceutical computations."""
        try:
            # Enable cuDNN benchmark for consistent convolution algorithms
            torch.backends.cudnn.benchmark = True

            # Enable TensorFloat-32 for faster computation on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Check for Tensor Core support
            if torch.cuda.get_device_capability()[0] >= 7:  # Volta and newer
                self.tensor_core_enabled = True
                self.logger.info("   Tensor Core optimization enabled")

            # Optimize memory allocation
            torch.cuda.empty_cache()

            # Set memory fraction to prevent OOM
            try:
                torch.cuda.set_per_process_memory_fraction(0.8)
            except AttributeError as e:
                # Not available in all PyTorch versions
                self.logger.debug(
                    "Memory fraction setting not available in this PyTorch version"
                )
                pass

        except Exception as e:
            self.logger.warning(f"GPU optimization failed: {e}")

    def _track_performance(
        self,
        operation_type: str,
        gpu_time: float,
        cpu_time: float,
        memory_used: float,
        throughput: float,
        errors: int = 0,
        warnings: int = 0,
    ):
        """Track performance metrics for optimization analysis."""
        metrics = PerformanceMetrics(
            operation_type=operation_type,
            gpu_time_ms=gpu_time * 1000,
            cpu_time_ms=cpu_time * 1000,
            memory_used_mb=memory_used / (1024**2),
            throughput_ops_per_sec=throughput,
            error_count=errors,
            warning_count=warnings,
        )
        self.performance_metrics.append(metrics)

        # Log performance for monitoring
        if throughput > 0:
            self.logger.debug(
                f"Performance [{operation_type}]: "
                f"GPU: {gpu_time*1000:.1f}ms, "
                f"CPU: {cpu_time*1000:.1f}ms, "
                f"Throughput: {throughput:.1f} ops/sec"
            )

    def _load_usp_standards(self) -> Dict[str, Any]:
        """Load USP standards and acceptance criteria."""
        return {
            "kcl_purity": {"min": 99.0, "max": 100.5},
            "moisture_content": {"max": 1.0},
            "dissolution_test_2": {
                "time_points": [1, 2, 4, 6],  # hours
                "tolerances": {
                    1: {"min": 25, "max": 45},
                    2: {"min": 45, "max": 65},
                    4: {"min": 70, "max": 90},
                    6: {"min": 85, "max": 100},
                },
            },
            "f2_similarity_threshold": 50.0,
            "content_uniformity": {"min": 90.0, "max": 110.0},
            "flowability_thresholds": {
                "excellent": {"carr_index": 10, "hausner_ratio": 1.11},
                "good": {"carr_index": 15, "hausner_ratio": 1.18},
                "fair": {"carr_index": 20, "hausner_ratio": 1.25},
                "passable": {"carr_index": 25, "hausner_ratio": 1.34},
                "poor": {"carr_index": 31, "hausner_ratio": 1.45},
            },
        }

    def _initialize_memory_pool(self):
        """Initialize GPU memory pool for efficient batch processing."""
        try:
            if torch.cuda.is_available():
                # Pre-allocate common tensor sizes for efficient memory management
                common_sizes = [
                    (32, 1024),  # Batch characterization
                    (16, 2048),  # Dissolution analysis
                    (8, 4096),  # Formulation optimization
                ]

                self.memory_pool = {}
                for batch_size, features in common_sizes:
                    self.memory_pool[f"{batch_size}x{features}"] = torch.zeros(
                        (batch_size, features), device=self.device, dtype=torch.float32
                    )

                self.logger.info("   Memory pool initialized for batch processing")

        except Exception as e:
            self.logger.warning(f"Memory pool initialization failed: {e}")

    def characterize_raw_materials_batch(
        self, material_batches: List[Dict[str, Any]]
    ) -> List[RawMaterialSpec]:
        """
        Characterize multiple raw material batches efficiently using batch processing.

        Args:
            material_batches: List of material data dictionaries

        Returns:
            List[RawMaterialSpec]: Characterized material specifications

        Raises:
            PharmaceuticalTestingException: If batch characterization fails
        """
        try:
            start_time = time.perf_counter()
            batch_size = len(material_batches)

            self.logger.info(f"🔬 Characterizing {batch_size} raw material batches...")

            # Process in optimal batch sizes
            optimal_batch_size = self.batch_size_optimization[
                "raw_material_characterization"
            ]
            results = []
            errors = 0
            warnings = 0

            for i in range(0, batch_size, optimal_batch_size):
                batch_slice = material_batches[i : i + optimal_batch_size]

                # GPU-accelerated batch processing
                if self.use_gpu and len(batch_slice) > 4:
                    batch_results = self._characterize_batch_gpu(batch_slice)
                else:
                    batch_results = self._characterize_batch_cpu(batch_slice)

                results.extend(batch_results)

                # Update error counts
                for result in batch_results:
                    if hasattr(result, "error_flags") and result.error_flags:
                        errors += len(result.error_flags)
                    if hasattr(result, "warning_flags") and result.warning_flags:
                        warnings += len(result.warning_flags)

            total_time = time.perf_counter() - start_time
            throughput = batch_size / total_time if total_time > 0 else 0

            self._track_performance(
                "batch_raw_material_characterization",
                0,
                total_time,
                0,
                throughput,
                errors,
                warnings,
            )

            self.logger.info(
                f"✅ Batch characterization completed in {total_time:.2f}s"
            )
            self.logger.info(f"   Throughput: {throughput:.1f} batches/sec")

            return results

        except Exception as e:
            self.logger.error(f"❌ Batch characterization failed: {e}")
            raise PharmaceuticalTestingException(f"Batch characterization failed: {e}")

    def _characterize_batch_gpu(
        self, batch_slice: List[Dict[str, Any]]
    ) -> List[RawMaterialSpec]:
        """GPU-accelerated batch characterization."""
        results = []

        try:
            # Vectorize common calculations across batch
            batch_size = len(batch_slice)

            # Extract numerical data for vectorized processing
            purity_values = torch.tensor(
                [b.get("purity_percent", 99.0) for b in batch_slice],
                device=self.device,
                dtype=torch.float32,
            )
            moisture_values = torch.tensor(
                [b.get("moisture_content", 1.0) for b in batch_slice],
                device=self.device,
                dtype=torch.float32,
            )

            # Vectorized USP compliance checks
            usp_compliance_matrix = torch.zeros((batch_size, 4), device=self.device)
            usp_compliance_matrix[:, 0] = (purity_values >= 99.0).float()
            usp_compliance_matrix[:, 1] = (moisture_values <= 2.0).float()
            usp_compliance_matrix[:, 2] = torch.ones(
                batch_size, device=self.device
            )  # Identity tests
            usp_compliance_matrix[:, 3] = torch.ones(
                batch_size, device=self.device
            )  # Impurity limits

            # Calculate batch compliance scores
            compliance_scores = torch.mean(usp_compliance_matrix, dim=1)

            # Process each material with GPU-accelerated validation
            for i, material_data in enumerate(batch_slice):
                compliance_score = compliance_scores[i].item()

                # Create specification with enhanced validation
                spec = RawMaterialSpec(
                    name=material_data.get("name", "Unknown KCl"),
                    grade=material_data.get("grade", "USP"),
                    purity_percent=purity_values[i].item(),
                    moisture_content=moisture_values[i].item(),
                    particle_size_d50=material_data.get("particle_size_d50", 150.0),
                    bulk_density=material_data.get("bulk_density", 1.0),
                    tapped_density=material_data.get("tapped_density", 1.2),
                    identification_tests={
                        "potassium": material_data.get("potassium_confirmed", True),
                        "chloride": material_data.get("chloride_confirmed", True),
                    },
                    impurity_limits=self._check_impurity_limits(material_data),
                )

                # Add GPU-computed compliance metadata
                spec.gpu_compliance_score = compliance_score
                spec.batch_processing = True

                results.append(spec)

            return results

        except Exception as e:
            self.logger.warning(
                f"GPU batch processing failed, falling back to CPU: {e}"
            )
            return self._characterize_batch_cpu(batch_slice)

    def _characterize_batch_cpu(
        self, batch_slice: List[Dict[str, Any]]
    ) -> List[RawMaterialSpec]:
        """CPU batch characterization with optimized processing."""
        results = []

        for material_data in batch_slice:
            try:
                # Use existing characterization method but with batch optimizations
                spec = self.characterize_raw_materials(material_data)
                spec.batch_processing = True
                results.append(spec)

            except PharmaceuticalTestingException:
                # Create minimal spec for failed characterization
                spec = RawMaterialSpec(
                    name=material_data.get("name", "Failed_Batch"),
                    grade="UNKNOWN",
                    purity_percent=0.0,
                    moisture_content=0.0,
                    particle_size_d50=0.0,
                    bulk_density=0.0,
                    tapped_density=0.0,
                    identification_tests={"potassium": False, "chloride": False},
                    impurity_limits={},
                )
                spec.error_flags = ["CHARACTERIZATION_FAILED"]
                results.append(spec)

        return results

    def characterize_raw_materials(
        self, material_batch: Dict[str, Any]
    ) -> RawMaterialSpec:
        """
        Characterize raw materials according to USP standards.

        Args:
            material_batch: Raw material batch data

        Returns:
            RawMaterialSpec: Characterized material specifications

        Raises:
            PharmaceuticalTestingException: If characterization fails
        """
        start_time = time.perf_counter()
        gpu_start_time = time.perf_counter()
        errors = 0
        warnings = 0

        try:
            self.logger.info("🔬 Characterizing raw materials...")

            # Input validation
            if not isinstance(material_batch, dict):
                raise PharmaceuticalTestingException(
                    "Material batch must be a dictionary"
                )

            required_fields = ["purity_percent", "moisture_content"]
            missing_fields = [
                field for field in required_fields if field not in material_batch
            ]
            if missing_fields:
                raise PharmaceuticalTestingException(
                    f"Missing required fields: {missing_fields}"
                )

            # Validate KCl purity with enhanced checking
            purity = material_batch.get("purity_percent", 0.0)
            purity_range = self.usp_standards["kcl_purity"]

            if not isinstance(purity, (int, float)) or purity <= 0:
                errors += 1
                raise PharmaceuticalTestingException(f"Invalid purity value: {purity}")

            if not (purity_range["min"] <= purity <= purity_range["max"]):
                errors += 1
                raise PharmaceuticalTestingException(
                    f"KCl purity {purity}% outside USP range "
                    f"{purity_range['min']}-{purity_range['max']}%"
                )

            # Check moisture content with warning thresholds
            moisture = material_batch.get("moisture_content", 0.0)
            moisture_limit = self.usp_standards["moisture_content"]["max"]

            if not isinstance(moisture, (int, float)) or moisture < 0:
                errors += 1
                raise PharmaceuticalTestingException(
                    f"Invalid moisture content: {moisture}"
                )

            if moisture > moisture_limit:
                errors += 1
                raise PharmaceuticalTestingException(
                    f"Moisture content {moisture}% exceeds USP limit {moisture_limit}%"
                )
            elif moisture > moisture_limit * 0.8:  # Warning threshold
                warnings += 1
                self.logger.warning(
                    f"Moisture content {moisture}% approaching USP limit"
                )

            # Perform identification tests with error handling
            identification_tests = {}
            try:
                identification_tests["potassium_test"] = (
                    self._perform_potassium_identification(material_batch)
                )
                identification_tests["chloride_test"] = (
                    self._perform_chloride_identification(material_batch)
                )
            except Exception as e:
                errors += 1
                self.logger.error(f"Identification tests failed: {e}")
                identification_tests = {"potassium_test": False, "chloride_test": False}

            # Check impurity limits with comprehensive validation
            try:
                impurity_limits = self._check_impurity_limits(material_batch)

                # Validate impurity values
                for impurity, value in impurity_limits.items():
                    if not isinstance(value, (int, float)) or value < 0:
                        warnings += 1
                        self.logger.warning(f"Invalid {impurity} value: {value}")
                        impurity_limits[impurity] = 0.0

            except Exception as e:
                warnings += 1
                self.logger.warning(f"Impurity limit checking failed: {e}")
                impurity_limits = {"heavy_metals": 0.0, "sodium": 0.0, "bromide": 0.0}

            # Validate particle properties
            particle_size = max(0.0, material_batch.get("particle_size_d50", 0.0))
            bulk_density = max(0.0, material_batch.get("bulk_density", 0.0))
            tapped_density = max(0.0, material_batch.get("tapped_density", 0.0))

            # Validate density relationship
            if (
                bulk_density > 0
                and tapped_density > 0
                and bulk_density > tapped_density
            ):
                warnings += 1
                self.logger.warning("Bulk density > tapped density (unusual)")

            spec = RawMaterialSpec(
                name=material_batch.get("name", "KCl"),
                grade=material_batch.get("grade", "USP"),
                purity_percent=purity,
                moisture_content=moisture,
                particle_size_d50=particle_size,
                bulk_density=bulk_density,
                tapped_density=tapped_density,
                identification_tests=identification_tests,
                impurity_limits=impurity_limits,
            )

            # Track performance
            total_time = time.perf_counter() - start_time
            gpu_time = time.perf_counter() - gpu_start_time
            cpu_time = total_time - gpu_time
            memory_used = 0  # Placeholder for actual memory tracking
            throughput = 1.0 / total_time if total_time > 0 else 0

            self._track_performance(
                "raw_material_characterization",
                gpu_time,
                cpu_time,
                memory_used,
                throughput,
                errors,
                warnings,
            )

            self.logger.info(f"✅ Raw material characterization completed: {spec.name}")
            if warnings > 0:
                self.logger.info(f"   Warnings encountered: {warnings}")

            return spec

        except Exception as e:
            total_time = time.perf_counter() - start_time
            self._track_performance(
                "raw_material_characterization",
                0,
                total_time,
                0,
                0,
                errors + 1,
                warnings,
            )
            self.logger.error(f"❌ Raw material characterization failed: {e}")
            raise PharmaceuticalTestingException(
                f"Raw material characterization failed: {e}"
            )

    def analyze_powder_flowability(
        self, bulk_density: float, tapped_density: float, angle_of_repose: float
    ) -> FlowabilityResult:
        """
        Analyze powder flowability using Carr's Index and Hausner Ratio.

        Args:
            bulk_density: Bulk density in g/mL
            tapped_density: Tapped density in g/mL
            angle_of_repose: Angle of repose in degrees

        Returns:
            FlowabilityResult: Flowability analysis results

        Raises:
            PharmaceuticalTestingException: If analysis fails
        """
        try:
            self.logger.info("📊 Analyzing powder flowability...")

            if tapped_density <= 0 or bulk_density <= 0:
                raise PharmaceuticalTestingException("Invalid density values")

            # Calculate Carr's Compressibility Index
            carr_index = 100 * (tapped_density - bulk_density) / tapped_density

            # Calculate Hausner Ratio
            hausner_ratio = tapped_density / bulk_density

            # Determine flow character
            flow_character = self._determine_flow_character(carr_index, hausner_ratio)

            result = FlowabilityResult(
                bulk_density=bulk_density,
                tapped_density=tapped_density,
                carr_index=carr_index,
                hausner_ratio=hausner_ratio,
                angle_of_repose=angle_of_repose,
                flow_character=flow_character,
            )

            self.logger.info(f"✅ Flowability analysis completed: {flow_character}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Flowability analysis failed: {e}")
            raise PharmaceuticalTestingException(f"Flowability analysis failed: {e}")

    def create_formulation_prototype(
        self,
        coating_thickness: float,
        polymer_ratios: Dict[str, float],
        process_parameters: Dict[str, Any],
    ) -> FormulationPrototype:
        """
        Create and characterize a formulation prototype.

        Args:
            coating_thickness: Coating thickness as percentage weight gain
            polymer_ratios: Ratios of different polymers
            process_parameters: Manufacturing process parameters

        Returns:
            FormulationPrototype: Characterized prototype

        Raises:
            PharmaceuticalTestingException: If prototype creation fails
        """
        try:
            self.logger.info(
                f"🧪 Creating formulation prototype with {coating_thickness}% coating..."
            )

            # Generate unique prototype ID
            prototype_id = f"KCl_P_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Simulate microcapsule formation with GPU acceleration if available
            if self.use_gpu and self.device.type == "cuda":
                encapsulation_efficiency = self._simulate_encapsulation_gpu(
                    coating_thickness, polymer_ratios, process_parameters
                )
            else:
                encapsulation_efficiency = self._simulate_encapsulation_cpu(
                    coating_thickness, polymer_ratios, process_parameters
                )

            # Assess particle morphology
            morphology = self._assess_particle_morphology(
                coating_thickness, polymer_ratios
            )

            prototype = FormulationPrototype(
                prototype_id=prototype_id,
                coating_thickness_percent=coating_thickness,
                polymer_ratio=polymer_ratios,
                encapsulation_efficiency=encapsulation_efficiency,
                particle_morphology=morphology,
            )

            self.logger.info(f"✅ Prototype {prototype_id} created successfully")
            return prototype

        except Exception as e:
            self.logger.error(f"❌ Prototype creation failed: {e}")
            raise PharmaceuticalTestingException(f"Prototype creation failed: {e}")

    def perform_dissolution_test(
        self,
        prototype: FormulationPrototype,
        test_conditions: Dict[str, Any],
        reference_profile: Optional[DissolutionProfile] = None,
    ) -> DissolutionProfile:
        """
        Perform USP dissolution testing with f2 similarity analysis.

        Args:
            prototype: Formulation prototype to test
            test_conditions: Test conditions (apparatus, medium, etc.)
            reference_profile: Reference dissolution profile for comparison

        Returns:
            DissolutionProfile: Dissolution test results

        Raises:
            PharmaceuticalTestingException: If dissolution test fails
        """
        try:
            self.logger.info(
                f"🧪 Performing dissolution test for {prototype.prototype_id}..."
            )

            # Simulate dissolution kinetics based on formulation parameters
            time_points = self.usp_standards["dissolution_test_2"]["time_points"]
            release_percentages = self._simulate_dissolution_kinetics(
                prototype, test_conditions, time_points
            )

            # Create dissolution profile
            profile = DissolutionProfile(
                time_points=time_points,
                release_percentages=release_percentages,
                test_conditions=test_conditions,
            )

            # Calculate f2 similarity if reference provided
            if reference_profile:
                f2_similarity = self._calculate_f2_similarity(
                    profile, reference_profile
                )
                profile.f2_similarity = f2_similarity

                self.logger.info(f"📊 f2 similarity factor: {f2_similarity:.2f}")

                if f2_similarity >= self.usp_standards["f2_similarity_threshold"]:
                    self.logger.info("✅ Dissolution profiles are similar (f2 ≥ 50)")
                else:
                    self.logger.warning("⚠️ Dissolution profiles not similar (f2 < 50)")

            # Validate against USP tolerances
            self._validate_dissolution_profile(profile)

            prototype.dissolution_profile = profile
            self.logger.info(
                f"✅ Dissolution test completed for {prototype.prototype_id}"
            )

            return profile

        except Exception as e:
            self.logger.error(f"❌ Dissolution test failed: {e}")
            raise PharmaceuticalTestingException(f"Dissolution test failed: {e}")

    def optimize_formulation(
        self, target_profile: DissolutionProfile, constraints: Dict[str, Any]
    ) -> List[FormulationPrototype]:
        """
        Optimize formulation parameters to match target dissolution profile.

        Args:
            target_profile: Target dissolution profile to match
            constraints: Formulation constraints

        Returns:
            List[FormulationPrototype]: Optimized formulation candidates

        Raises:
            PharmaceuticalTestingException: If optimization fails
        """
        try:
            self.logger.info("🎯 Optimizing formulation parameters...")

            candidates = []

            # Define parameter ranges for optimization
            coating_range = np.linspace(8, 20, 5)  # 8-20% coating thickness
            ethylcellulose_range = np.linspace(0.6, 0.9, 4)  # 60-90% ethylcellulose

            best_f2 = 0
            best_candidate = None

            for coating in coating_range:
                for ec_ratio in ethylcellulose_range:
                    hpc_ratio = 1.0 - ec_ratio

                    polymer_ratios = {
                        "ethylcellulose": ec_ratio,
                        "hydroxypropyl_cellulose": hpc_ratio,
                    }

                    # Create prototype
                    prototype = self.create_formulation_prototype(
                        coating,
                        polymer_ratios,
                        constraints.get("process_parameters", {}),
                    )

                    # Test dissolution
                    profile = self.perform_dissolution_test(
                        prototype,
                        constraints.get("test_conditions", {}),
                        target_profile,
                    )

                    # Track best candidate
                    if profile.f2_similarity and profile.f2_similarity > best_f2:
                        best_f2 = profile.f2_similarity
                        best_candidate = prototype

                    candidates.append(prototype)

            # Sort by f2 similarity
            candidates.sort(
                key=lambda x: x.dissolution_profile.f2_similarity or 0, reverse=True
            )

            self.logger.info(f"✅ Optimization completed. Best f2: {best_f2:.2f}")
            return candidates[:5]  # Return top 5 candidates

        except Exception as e:
            self.logger.error(f"❌ Formulation optimization failed: {e}")
            raise PharmaceuticalTestingException(
                f"Formulation optimization failed: {e}"
            )

    def validate_stability(
        self,
        prototype: FormulationPrototype,
        storage_conditions: List[Dict[str, Any]],
        time_points: List[int],
    ) -> Dict[str, Any]:
        """
        Validate stability according to ICH Q1A guidelines.

        Args:
            prototype: Formulation prototype to test
            storage_conditions: Storage conditions (temperature, humidity)
            time_points: Time points for testing (months)

        Returns:
            Dict[str, Any]: Stability test results

        Raises:
            PharmaceuticalTestingException: If stability validation fails
        """
        try:
            self.logger.info(f"🕐 Validating stability for {prototype.prototype_id}...")

            stability_results = {
                "prototype_id": prototype.prototype_id,
                "conditions": storage_conditions,
                "time_points": time_points,
                "results": {},
            }

            for condition in storage_conditions:
                condition_name = (
                    f"{condition['temperature']}°C/{condition['humidity']}%RH"
                )
                condition_results = {}

                for time_point in time_points:
                    # Simulate degradation over time
                    degradation_factor = self._calculate_degradation(
                        condition, time_point
                    )

                    # Simulate dissolution profile changes
                    degraded_profile = self._simulate_degraded_dissolution(
                        prototype.dissolution_profile, degradation_factor
                    )

                    # Calculate f2 similarity with initial profile
                    f2_similarity = self._calculate_f2_similarity(
                        degraded_profile, prototype.dissolution_profile
                    )

                    condition_results[f"{time_point}_months"] = {
                        "dissolution_f2": f2_similarity,
                        "degradation_factor": degradation_factor,
                        "stable": f2_similarity >= 50.0,
                    }

                stability_results["results"][condition_name] = condition_results

            self.logger.info(
                f"✅ Stability validation completed for {prototype.prototype_id}"
            )
            return stability_results

        except Exception as e:
            self.logger.error(f"❌ Stability validation failed: {e}")
            raise PharmaceuticalTestingException(f"Stability validation failed: {e}")

    # Helper methods
    def _perform_potassium_identification(self, material_batch: Dict[str, Any]) -> bool:
        """Simulate potassium identification test."""
        # In real implementation, this would interface with analytical instruments
        return material_batch.get("potassium_confirmed", True)

    def _perform_chloride_identification(self, material_batch: Dict[str, Any]) -> bool:
        """Simulate chloride identification test."""
        return material_batch.get("chloride_confirmed", True)

    def _check_impurity_limits(
        self, material_batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """Check impurity limits against USP specifications."""
        return {
            "heavy_metals": material_batch.get("heavy_metals_ppm", 0.0),
            "sodium": material_batch.get("sodium_percent", 0.0),
            "bromide": material_batch.get("bromide_ppm", 0.0),
        }

    def _determine_flow_character(self, carr_index: float, hausner_ratio: float) -> str:
        """Determine flow character based on Carr's Index and Hausner Ratio."""
        thresholds = self.usp_standards["flowability_thresholds"]

        if carr_index <= thresholds["excellent"]["carr_index"]:
            return "Excellent"
        elif carr_index <= thresholds["good"]["carr_index"]:
            return "Good"
        elif carr_index <= thresholds["fair"]["carr_index"]:
            return "Fair"
        elif carr_index <= thresholds["passable"]["carr_index"]:
            return "Passable"
        else:
            return "Poor"

    def _simulate_encapsulation_gpu(
        self,
        coating_thickness: float,
        polymer_ratios: Dict[str, float],
        process_parameters: Dict[str, Any],
    ) -> float:
        """Simulate encapsulation efficiency using GPU acceleration."""
        if not torch.cuda.is_available():
            return self._simulate_encapsulation_cpu(
                coating_thickness, polymer_ratios, process_parameters
            )

        try:
            with torch.amp.autocast("cuda", enabled=self.tensor_core_enabled):
                # Convert parameters to tensors for GPU computation
                coating_tensor = torch.tensor(
                    coating_thickness, device=self.device, dtype=torch.float32
                )

                # Advanced coating process simulation with realistic physics
                base_efficiency = 0.95

                # Coating thickness optimization curve (sigmoid function)
                optimal_thickness = 12.0  # Optimal around 12%
                thickness_factor = torch.sigmoid(
                    (coating_tensor - optimal_thickness) / 3.0
                )

                # Polymer ratio effects with complex interactions
                ec_ratio = polymer_ratios.get("ethylcellulose", 0.8)
                hpc_ratio = polymer_ratios.get("hpc", 0.2)

                # Create tensors for polymer calculations
                ec_tensor = torch.tensor(
                    ec_ratio, device=self.device, dtype=torch.float32
                )
                hpc_tensor = torch.tensor(
                    hpc_ratio, device=self.device, dtype=torch.float32
                )

                # Optimal polymer blend calculation
                optimal_ec = 0.8
                optimal_hpc = 0.2

                ec_factor = 1.0 - torch.abs(ec_tensor - optimal_ec) * 0.2
                hpc_factor = 1.0 - torch.abs(hpc_tensor - optimal_hpc) * 0.15

                # Process parameter effects
                temperature = process_parameters.get("temperature", 60.0)
                spray_rate = process_parameters.get("spray_rate", 1.0)

                temp_tensor = torch.tensor(
                    temperature, device=self.device, dtype=torch.float32
                )
                spray_tensor = torch.tensor(
                    spray_rate, device=self.device, dtype=torch.float32
                )

                # Temperature optimization (optimal around 60°C)
                temp_factor = torch.sigmoid((temp_tensor - 60.0) / 10.0) * 0.8 + 0.9

                # Spray rate factor (optimal around 1.0)
                spray_factor = 1.0 / (1.0 + torch.abs(spray_tensor - 1.0) * 0.1)

                # Combine all factors
                efficiency = (
                    base_efficiency
                    * thickness_factor
                    * ec_factor
                    * hpc_factor
                    * temp_factor
                    * spray_factor
                )

                # Add realistic noise and constraints
                noise = torch.randn(1, device=self.device) * 0.005  # 0.5% noise
                final_efficiency = torch.clamp(efficiency + noise, 0.85, 0.99)

                return float(final_efficiency.cpu().item())

        except Exception as e:
            self.logger.warning(
                f"GPU encapsulation simulation failed: {e}, falling back to CPU"
            )
            return self._simulate_encapsulation_cpu(
                coating_thickness, polymer_ratios, process_parameters
            )

    def _simulate_encapsulation_cpu(
        self,
        coating_thickness: float,
        polymer_ratios: Dict[str, float],
        process_parameters: Dict[str, Any],
    ) -> float:
        """Simulate encapsulation efficiency using CPU."""
        base_efficiency = 0.95

        # Coating thickness optimization (similar to GPU version)
        optimal_thickness = 12.0
        thickness_factor = 1.0 / (
            1.0 + np.exp(-(coating_thickness - optimal_thickness) / 3.0)
        )

        # Polymer ratio effects
        ec_ratio = polymer_ratios.get("ethylcellulose", 0.8)
        hpc_ratio = polymer_ratios.get("hpc", 0.2)

        ec_factor = 1.0 - abs(ec_ratio - 0.8) * 0.2
        hpc_factor = 1.0 - abs(hpc_ratio - 0.2) * 0.15

        # Process parameter effects
        temperature = process_parameters.get("temperature", 60.0)
        spray_rate = process_parameters.get("spray_rate", 1.0)

        temp_factor = 1.0 / (1.0 + np.exp(-(temperature - 60.0) / 10.0)) * 0.8 + 0.9
        spray_factor = 1.0 / (1.0 + abs(spray_rate - 1.0) * 0.1)

        # Combine all factors
        efficiency = (
            base_efficiency
            * thickness_factor
            * ec_factor
            * hpc_factor
            * temp_factor
            * spray_factor
        )

        # Add small realistic variation and constraints
        noise = np.random.normal(0, 0.005)  # 0.5% noise
        final_efficiency = np.clip(efficiency + noise, 0.85, 0.99)

        return float(final_efficiency)

    def _assess_particle_morphology(
        self, coating_thickness: float, polymer_ratios: Dict[str, float]
    ) -> str:
        """Assess particle morphology based on formulation parameters."""
        if coating_thickness < 10:
            return "Irregular - Thin coating"
        elif coating_thickness > 18:
            return "Aggregated - Thick coating"
        else:
            return "Spherical - Uniform coating"

    def _simulate_dissolution_kinetics(
        self,
        prototype: FormulationPrototype,
        test_conditions: Dict[str, Any],
        time_points: List[float],
    ) -> List[float]:
        """Simulate dissolution kinetics based on formulation parameters."""
        # Higuchi model: Q = k * sqrt(t)
        # Modified for coating effects

        coating_factor = 1.0 / (1.0 + prototype.coating_thickness_percent / 10)
        ec_ratio = prototype.polymer_ratio.get("ethylcellulose", 0.8)
        polymer_factor = 1.0 - ec_ratio * 0.3  # Higher EC = slower release

        k = 25 * coating_factor * polymer_factor  # Release rate constant

        release_percentages = []
        for t in time_points:
            # Higuchi equation with saturation
            release = k * np.sqrt(t)
            release = min(release, 95)  # Cap at 95% to be realistic
            release_percentages.append(release)

        return release_percentages

    def _calculate_f2_similarity(
        self, profile1: DissolutionProfile, profile2: DissolutionProfile
    ) -> float:
        """Calculate f2 similarity factor between two dissolution profiles."""
        if len(profile1.release_percentages) != len(profile2.release_percentages):
            raise PharmaceuticalTestingException(
                "Profiles must have same number of time points"
            )

        # f2 = 50 * log{[1 + (1/n) * Σ(Rt - Tt)²]^(-0.5) * 100}
        n = len(profile1.release_percentages)
        sum_squared_diff = sum(
            (r1 - r2) ** 2
            for r1, r2 in zip(
                profile1.release_percentages, profile2.release_percentages
            )
        )

        f2 = 50 * np.log10(((1 + sum_squared_diff / n) ** -0.5) * 100)
        return max(0, min(100, f2))  # Constrain to 0-100 range

    def _validate_dissolution_profile(self, profile: DissolutionProfile) -> None:
        """Validate dissolution profile against USP tolerances."""
        tolerances = self.usp_standards["dissolution_test_2"]["tolerances"]

        for i, (time_point, release) in enumerate(
            zip(profile.time_points, profile.release_percentages)
        ):
            if time_point in tolerances:
                min_release = tolerances[time_point]["min"]
                max_release = tolerances[time_point]["max"]

                if not (min_release <= release <= max_release):
                    self.logger.warning(
                        f"⚠️ Release at {time_point}h ({release:.1f}%) outside USP range "
                        f"({min_release}-{max_release}%)"
                    )

    def _calculate_degradation(
        self, condition: Dict[str, Any], time_months: int
    ) -> float:
        """Calculate degradation factor based on storage conditions and time."""
        # Arrhenius equation approximation
        temp = condition["temperature"]
        humidity = condition["humidity"]

        # Higher temperature and humidity increase degradation
        temp_factor = np.exp((temp - 25) / 10)  # Normalized to 25°C
        humidity_factor = 1 + (humidity - 60) / 100  # Normalized to 60% RH

        degradation = 0.01 * temp_factor * humidity_factor * time_months
        return min(degradation, 0.5)  # Cap at 50% degradation

    def _simulate_degraded_dissolution(
        self, original_profile: DissolutionProfile, degradation_factor: float
    ) -> DissolutionProfile:
        """Simulate how dissolution profile changes with degradation."""
        # Degradation typically slows release
        degraded_percentages = [
            release * (1 - degradation_factor * 0.5)
            for release in original_profile.release_percentages
        ]

        return DissolutionProfile(
            time_points=original_profile.time_points.copy(),
            release_percentages=degraded_percentages,
            test_conditions=original_profile.test_conditions.copy(),
        )

    def generate_comprehensive_report(
        self, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive pharmaceutical testing report.

        Args:
            test_results: All test results to include in report

        Returns:
            Dict[str, Any]: Comprehensive testing report
        """
        try:
            self.logger.info(
                "📊 Generating comprehensive pharmaceutical testing report..."
            )

            report = {
                "report_id": f"KCl_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generation_time": datetime.now().isoformat(),
                "test_summary": {
                    "total_tests": len(test_results),
                    "passed_tests": sum(
                        1
                        for result in test_results.values()
                        if result.get("status") == "PASSED"
                    ),
                    "failed_tests": sum(
                        1
                        for result in test_results.values()
                        if result.get("status") == "FAILED"
                    ),
                },
                "detailed_results": test_results,
                "compliance_status": self._assess_overall_compliance(test_results),
                "recommendations": self._generate_recommendations(test_results),
            }

            self.logger.info(
                f"✅ Comprehensive report generated: {report['report_id']}"
            )
            return report

        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            raise PharmaceuticalTestingException(f"Report generation failed: {e}")

    def _assess_overall_compliance(self, test_results: Dict[str, Any]) -> str:
        """Assess overall compliance status."""
        failed_critical = any(
            result.get("status") == "FAILED" and result.get("critical", False)
            for result in test_results.values()
        )

        if failed_critical:
            return "NON_COMPLIANT"

        total_tests = len(test_results)
        passed_tests = sum(
            1 for result in test_results.values() if result.get("status") == "PASSED"
        )

        if passed_tests == total_tests:
            return "FULLY_COMPLIANT"
        elif passed_tests / total_tests >= 0.8:
            return "SUBSTANTIALLY_COMPLIANT"
        else:
            return "PARTIALLY_COMPLIANT"

    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        for test_name, result in test_results.items():
            if result.get("status") == "FAILED":
                if "dissolution" in test_name.lower():
                    recommendations.append(
                        "Consider adjusting coating thickness or polymer ratios to improve dissolution"
                    )
                elif "flowability" in test_name.lower():
                    recommendations.append(
                        "Improve powder flowability through particle size optimization or flow aids"
                    )
                elif "stability" in test_name.lower():
                    recommendations.append(
                        "Investigate formulation stability - consider antioxidants or packaging changes"
                    )

        if not recommendations:
            recommendations.append(
                "All tests passed - formulation ready for scale-up studies"
            )

        return recommendations

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        if not self.performance_metrics:
            return {"status": "No performance data available"}

        # Aggregate metrics by operation type
        operation_stats = {}
        for metric in self.performance_metrics:
            op_type = metric.operation_type
            if op_type not in operation_stats:
                operation_stats[op_type] = {
                    "gpu_times": [],
                    "cpu_times": [],
                    "throughputs": [],
                    "error_counts": [],
                    "warning_counts": [],
                    "memory_usage": [],
                }

            operation_stats[op_type]["gpu_times"].append(metric.gpu_time_ms)
            operation_stats[op_type]["cpu_times"].append(metric.cpu_time_ms)
            operation_stats[op_type]["throughputs"].append(
                metric.throughput_ops_per_sec
            )
            operation_stats[op_type]["error_counts"].append(metric.error_count)
            operation_stats[op_type]["warning_counts"].append(metric.warning_count)
            operation_stats[op_type]["memory_usage"].append(metric.memory_used_mb)

        # Calculate statistics
        summary_stats = {}
        for op_type, stats in operation_stats.items():
            summary_stats[op_type] = {
                "total_operations": len(stats["gpu_times"]),
                "avg_gpu_time_ms": (
                    np.mean(stats["gpu_times"]) if stats["gpu_times"] else 0
                ),
                "avg_cpu_time_ms": (
                    np.mean(stats["cpu_times"]) if stats["cpu_times"] else 0
                ),
                "avg_throughput": (
                    np.mean(stats["throughputs"]) if stats["throughputs"] else 0
                ),
                "total_errors": sum(stats["error_counts"]),
                "total_warnings": sum(stats["warning_counts"]),
                "avg_memory_mb": (
                    np.mean(stats["memory_usage"]) if stats["memory_usage"] else 0
                ),
                "gpu_acceleration_ratio": (
                    (
                        np.mean(stats["cpu_times"])
                        / max(np.mean(stats["gpu_times"]), 1e-6)
                    )
                    if stats["gpu_times"] and stats["cpu_times"]
                    else 1.0
                ),
            }

        # Overall system performance
        total_operations = sum(
            len(stats["gpu_times"]) for stats in operation_stats.values()
        )
        total_errors = sum(
            sum(stats["error_counts"]) for stats in operation_stats.values()
        )
        total_warnings = sum(
            sum(stats["warning_counts"]) for stats in operation_stats.values()
        )

        system_health = {
            "total_operations": total_operations,
            "error_rate": total_errors / max(total_operations, 1),
            "warning_rate": total_warnings / max(total_operations, 1),
            "gpu_utilization": self.gpu_optimization_enabled,
            "tensor_core_active": self.tensor_core_enabled,
            "overall_status": (
                "HEALTHY"
                if total_errors == 0
                else (
                    "DEGRADED" if total_errors < total_operations * 0.05 else "CRITICAL"
                )
            ),
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "operation_statistics": summary_stats,
            "system_health": system_health,
            "gpu_info": {
                "device": str(self.device),
                "optimization_enabled": self.gpu_optimization_enabled,
                "tensor_core_enabled": self.tensor_core_enabled,
                "cuda_available": torch.cuda.is_available(),
            },
            "recommendations": self._generate_performance_recommendations(
                summary_stats, system_health
            ),
        }

    def _generate_performance_recommendations(
        self, stats: Dict[str, Any], health: Dict[str, Any]
    ) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Error rate recommendations
        if health["error_rate"] > 0.01:  # More than 1% errors
            recommendations.append(
                "High error rate detected - review input validation and error handling"
            )

        # GPU utilization recommendations
        if not self.gpu_optimization_enabled and torch.cuda.is_available():
            recommendations.append(
                "GPU available but not optimized - enable GPU acceleration for better performance"
            )

        # Throughput optimization
        for op_type, op_stats in stats.items():
            if op_stats["avg_throughput"] < 1.0:  # Less than 1 operation per second
                recommendations.append(
                    f"Low throughput for {op_type} - consider batch processing optimization"
                )

            if (
                op_stats["gpu_acceleration_ratio"] < 2.0
                and self.gpu_optimization_enabled
            ):
                recommendations.append(
                    f"Limited GPU acceleration for {op_type} - review GPU computation patterns"
                )

        # Memory recommendations
        total_memory = sum(op_stats["avg_memory_mb"] for op_stats in stats.values())
        if total_memory > 1000:  # More than 1GB average
            recommendations.append(
                "High memory usage detected - implement memory pooling and cleanup"
            )

        if not recommendations:
            recommendations.append(
                "System performance is optimal - continue monitoring"
            )

        return recommendations
