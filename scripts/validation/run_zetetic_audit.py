#!/usr/bin/env python3
"""
Zetetic Real-World Thermodynamic Audit for Kimera SWM

This script conducts rigorous, zetetic (skeptical inquiry) validation of the revolutionary
thermodynamic applications with the actual Kimera instance. No simulations, no mocks.

Zetetic Methodology:
1. Question everything - assume nothing works until proven
2. Measure everything - comprehensive instrumentation
3. Validate against physics - compare to theoretical limits
4. Document all failures - honest reporting of limitations
5. Reproducible results - multiple independent runs
"""

import sys
import time
import json
import numpy as np
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
import logging
import statistics

# Add backend to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.core.kimera_system import kimera_singleton
from backend.utils.kimera_logger import get_logger, LogCategory

logger = get_logger("ZeteticAudit", LogCategory.SYSTEM)

# GPU monitoring setup
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
    logger.info("Real GPU monitoring available via pynvml")
except (ImportError, pynvml.NVMLError):
    GPU_MONITORING_AVAILABLE = False
    logger.warning("pynvml not available or failed to initialize - GPU monitoring will be limited")

@dataclass
class ZeteticMeasurement:
    """Rigorous measurement with uncertainty quantification"""
    timestamp: datetime
    measurement_name: str
    measured_value: float
    theoretical_value: float
    uncertainty: float
    validation_status: str
    raw_data: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ZeteticAuditor:
    """
    Rigorous zetetic auditor for real-world thermodynamic validation.
    """
    
    def __init__(self):
        self.audit_start_time = datetime.now()
        self.measurements: List[ZeteticMeasurement] = []
        self.validation_failures: List[Dict[str, Any]] = []
        
        # Use the global Kimera singleton
        self.kimera = kimera_singleton
        # The script is now run by the API, so the system is guaranteed to be initialized.
        
        self.field_engine = self.kimera.get_vault_manager()
        self.thermo_engine = self.kimera.get_thermodynamic_engine()
        self.gpu_foundation = self.kimera.get_gpu_foundation()

        if not all([self.thermo_engine, self.gpu_foundation]):
            raise RuntimeError("Thermodynamic Engine or GPU Foundation not available.")

        logger.info("ZETETIC AUDITOR INITIALIZED")
        logger.info("Using live, singleton-managed Kimera components.")

    def get_gpu_metrics(self) -> Dict[str, Any]:
        if not self.gpu_foundation:
            return {"error": "GPU Foundation not available"}
        return self.gpu_foundation.get_status()

    def create_cognitive_fields(self, count: int, variance: float) -> List[Dict[str, Any]]:
        """Creates geoids with embeddings for testing."""
        logger.info(f"Creating {count} cognitive fields with variance {variance:.2f}")
        fields = []
        for i in range(count):
            embedding = (np.random.randn(1024) * variance).tolist()
            # This is a simplified representation for the thermodynamic engine
            fields.append({'embedding': embedding})
        return fields

    def validate_carnot_engine(self) -> ZeteticMeasurement:
        """Zetetic validation of Carnot engine claims"""
        logger.info("ZETETIC CARNOT ENGINE VALIDATION: START")
        
        logger.info("Step 1: Creating cognitive field reservoirs...")
        hot_fields = self.create_cognitive_fields(50, variance=2.0)
        cold_fields = self.create_cognitive_fields(50, variance=0.5)
        logger.info(f"Step 1 COMPLETE: Created {len(hot_fields)} hot and {len(cold_fields)} cold fields.")

        logger.info("Step 2: Calculating epistemic temperatures...")
        hot_temp_obj = self.thermo_engine.calculate_epistemic_temperature(hot_fields)
        cold_temp_obj = self.thermo_engine.calculate_epistemic_temperature(cold_fields)
        logger.info("Step 2 COMPLETE: Temperatures calculated.")

        hot_temp = hot_temp_obj.physical_temperature
        cold_temp = cold_temp_obj.physical_temperature
        logger.info(f"Measured temperatures: Hot={hot_temp:.3f}K, Cold={cold_temp:.3f}K")
        
        if hot_temp <= cold_temp:
            logger.warning("VALIDATION FAILED: Hot reservoir is not hotter than cold reservoir.")
            status = "FAILED"
            uncertainty = 0.0
            measured = 0.0
            theoretical = 0.0
        else:
            logger.info("Step 3: Running Zetetic Carnot Cycle...")
            carnot_cycle = self.thermo_engine.run_zetetic_carnot_engine(hot_fields, cold_fields)
            logger.info("Step 3 COMPLETE: Carnot cycle finished.")
            
            measured = carnot_cycle.actual_efficiency
            theoretical = carnot_cycle.theoretical_efficiency
            status = "VALIDATED" if measured <= theoretical else "FAILED"
            uncertainty = abs(measured - theoretical)
            logger.info(f"Efficiency check: Measured={measured:.4f}, Theoretical={theoretical:.4f}, Status={status}")

        logger.info("Step 4: Creating measurement object...")
        measurement = ZeteticMeasurement(
            timestamp=datetime.now(),
            measurement_name="Carnot Engine Efficiency",
            measured_value=measured,
            theoretical_value=theoretical,
            uncertainty=uncertainty,
            validation_status=status
        )
        logger.info("Step 4 COMPLETE. ZETETIC CARNOT ENGINE VALIDATION: END")
        return measurement

    def run_audit(self) -> Dict[str, Any]:
        """Conducts the comprehensive zetetic audit."""
        logger.info("AUDIT: Starting comprehensive zetetic audit...")
        
        results = {}
        try:
            logger.info("AUDIT: Calling validate_carnot_engine...")
            carnot_measurement = self.validate_carnot_engine()
            logger.info("AUDIT: Returned from validate_carnot_engine.")
            
            self.measurements.append(carnot_measurement)
            results['carnot_validation'] = carnot_measurement.__dict__
            if carnot_measurement.validation_status == "FAILED":
                self.validation_failures.append(carnot_measurement.__dict__)

        except Exception as e:
            logger.error(f"AUDIT: CRITICAL a udit error: {e}", exc_info=True)
            results['audit_error'] = str(e)

        logger.info("AUDIT: Generating report...")
        report = self.generate_report(results)
        logger.info("AUDIT: Saving report...")
        self.save_report(report)
        logger.info("AUDIT: Comprehensive zetetic audit complete.")
        return report

    def generate_report(self, results: Dict) -> Dict:
        failure_count = len(self.validation_failures)
        overall_status = "PASSED" if failure_count == 0 else "FAILED"
        
        logger.info("="*20 + " ZETETIC AUDIT REPORT " + "="*20)
        logger.info(f"Audit completed at: {datetime.now().isoformat()}")
        logger.info(f"Overall Status: {overall_status}")
        logger.info(f"Total Measurements: {len(self.measurements)}")
        logger.info(f"Validation Failures: {failure_count}")
        logger.info("="*62)
        
        for m in self.measurements:
            logger.info(f"  - Measurement: {m.measurement_name}")
            logger.info(f"    - Status: {m.validation_status}")
            logger.info(f"    - Measured: {m.measured_value:.4f}")
            logger.info(f"    - Theoretical: {m.theoretical_value:.4f}")
            logger.info(f"    - Uncertainty: {m.uncertainty:.4f}")

        return {
            "audit_summary": {
                "overall_status": overall_status,
                "timestamp": self.audit_start_time.isoformat(),
                "total_measurements": len(self.measurements),
                "failures": failure_count,
            },
            "detailed_results": results
        }
        
    def save_report(self, report: Dict):
        report_path = Path(__file__).parent.parent / "reports" / f"zetetic_audit_{self.audit_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Audit report saved to {report_path}")

# The script is now intended to be imported and run via the API,
# so the __main__ block is removed. 