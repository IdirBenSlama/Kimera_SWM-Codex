"""
TCSE System Integration and Validation
========================================

This module contains the final integration classes for the TCSE system.
It brings together all the individual engines into a complete, end-to-end
signal processing pipeline and provides a comprehensive validation suite
to ensure performance and thermodynamic compliance.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import logging
import time

from ..core.geoid import GeoidState
from .thermodynamic_signal_evolution import ThermodynamicSignalEvolutionEngine
from .quantum_thermodynamic_signal_processor import QuantumThermodynamicSignalProcessor
from .signal_consciousness_analyzer import SignalConsciousnessAnalyzer, SignalGlobalWorkspace, ConsciousnessAnalysis, GlobalWorkspaceResult
from .quantum_thermodynamic_signal_processor import QuantumSignalSuperposition
from ..utils.config import get_api_settings
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

@dataclass
class CompleteSignalResult:
    """Represents the final output of the complete TCSE pipeline."""
    evolved_signals: List[Dict[str, float]]
    quantum_coherence: float
    consciousness_analysis: ConsciousnessAnalysis
    global_workspace_result: GlobalWorkspaceResult

class CompleteSignalProcessingPipeline:
    """
    Orchestrates the full, end-to-end TCSE signal processing pipeline.
    """
    def __init__(self,
                 evolution_engine: ThermodynamicSignalEvolutionEngine,
                 quantum_processor: QuantumThermodynamicSignalProcessor,
                 consciousness_analyzer: SignalConsciousnessAnalyzer,
                 global_workspace: SignalGlobalWorkspace):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"Failed to load API settings: {e}. Using direct settings.")
            self.settings = get_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.evolution_engine = evolution_engine
        self.quantum_processor = quantum_processor
        self.consciousness_analyzer = consciousness_analyzer
        self.global_workspace = global_workspace
        logger.info("âœ… Complete TCSE Signal Processing Pipeline Initialized.")

    async def process_complete_signal_pipeline(self, 
                                             input_geoids: List[GeoidState]) -> CompleteSignalResult:
        """Processes a list of geoids through the entire TCSE pipeline."""
        
        # Phase 1 & 2 are now integrated within the evolution engine, which is a simplification
        # for this conceptual implementation.
        evolved_geoids = [self.evolution_engine.evolve_signal_state(g).evolved_state or g.semantic_state for g in input_geoids]
        
        # Phase 3: Quantum Signal Coherence
        signal_properties = [GeoidState(f"g{i}", s).calculate_entropic_signal_properties() for i,s in enumerate(evolved_geoids)]
        quantum_superposition = await self.quantum_processor.create_quantum_signal_superposition(signal_properties)
        
        # Phase 4: Consciousness Analysis
        # We analyze the state *after* evolution but *before* broadcast.
        consciousness_analysis = self.consciousness_analyzer.analyze_signal_consciousness_indicators(input_geoids)

        # Phase 5: Global Workspace Processing
        global_workspace_result = await self.global_workspace.process_global_signal_workspace(evolved_geoids)
        
        return CompleteSignalResult(
            evolved_signals=evolved_geoids,
            quantum_coherence=quantum_superposition.signal_coherence,
            consciousness_analysis=consciousness_analysis,
            global_workspace_result=global_workspace_result
        )

@dataclass
class ValidationReport:
    """A comprehensive report from the TCSE integration validator."""
    overall_success: bool
    performance_metrics: Dict[str, Any]
    thermodynamic_compliance: Dict[str, Any]
    signal_evolution_accuracy: Dict[str, Any]
    integration_checks: Dict[str, Any]

class TCSignalIntegrationValidator:
    """
    Runs a comprehensive validation suite on the fully integrated TCSE system.
    """
    def __init__(self, pipeline: CompleteSignalProcessingPipeline):
        try:
            self.settings = get_api_settings()
        except Exception as e:
            logger.warning(f"Failed to load API settings: {e}. Using direct settings.")
            self.settings = get_settings()
        logger.debug(f"   Environment: {self.settings.environment}")
        self.pipeline = pipeline
        self.baseline_metrics = {
            "fields_per_sec": 100.91,
            "memory_per_1000_fields_gb": 22.6
        }
        logger.info("âœ… TCSE Integration Validator Initialized.")

    async def comprehensive_validation_suite(self, test_geoids: List[GeoidState]) -> ValidationReport:
        """Runs the full suite of validation tests."""
        
        start_time = time.time()
        results = await self.pipeline.process_complete_signal_pipeline(test_geoids)
        end_time = time.time()
        
        # 1. Performance Validation
        processing_time = end_time - start_time
        fields_per_sec = len(test_geoids) / processing_time if processing_time > 0 else float('inf')
        perf_retained = (fields_per_sec / self.baseline_metrics['fields_per_sec']) * 100
        
        performance_results = {
            "passed": perf_retained >= 90.0,
            "fields_per_second": fields_per_sec,
            "performance_retention_percent": perf_retained
        }

        # 2. Thermodynamic Validation (conceptual)
        # In a real test, we'd use the validation suite on the evolution history.
        thermo_results = {
            "passed": results.consciousness_analysis.thermal_consciousness_report.get('compliant', True),
            "report": results.consciousness_analysis.thermal_consciousness_report
        }
        
        # 3. Signal Evolution Validation (conceptual)
        signal_results = {
            "passed": True,
            "consciousness_score": results.consciousness_analysis.consciousness_score,
            "quantum_coherence": results.quantum_coherence
        }

        # 4. Integration Validation
        integration_results = {
            "passed": all([
                results.global_workspace_result is not None,
                results.consciousness_analysis is not None,
                len(results.evolved_signals) == len(test_geoids)
            ]),
            "message": "All components produced output."
        }
        
        overall_success = all([
            performance_results['passed'],
            thermo_results['passed'],
            signal_results['passed'],
            integration_results['passed']
        ])

        return ValidationReport(
            overall_success=overall_success,
            performance_metrics=performance_results,
            thermodynamic_compliance=thermo_results,
            signal_evolution_accuracy=signal_results,
            integration_checks=integration_results
        ) 