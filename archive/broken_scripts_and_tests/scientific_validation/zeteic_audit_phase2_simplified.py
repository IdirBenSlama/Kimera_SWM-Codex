"""
KIMERA Phase 2 Zeteic Scientific Audit - Simplified
==================================================

Rigorous scientific validation of Phase 2 claims using zeteic methodology.
This version focuses on core validation without complex dependencies.
"""

import torch
import numpy as np
import scipy.stats as stats
import sys
import os
import time
import logging
from typing import Dict, Tuple
from dataclasses import dataclass

# Add backend to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))

try:
    from monitoring.psychiatric_stability_monitor import (
        CognitiveCoherenceMonitor,
        PersonaDriftDetector,
        PsychoticFeaturePrevention
    )
    from core.therapeutic_intervention_system import TherapeuticInterventionSystem
    from security.cognitive_firewall import CognitiveSeparationFirewall
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  Import warning: {e}")
    IMPORTS_AVAILABLE = False

@dataclass
class ScientificMeasurement:
    """Rigorous scientific measurement with statistical validation"""
    value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    p_value: float
    effect_size: float
    measurement_error: float
    reproducibility_score: float
    scientific_validity: str

class ZeteticAuditor:
    """Simplified but rigorous zeteic audit framework"""
    
    def __init__(self):
        self.results = {}
        
        if IMPORTS_AVAILABLE:
            try:
                self.coherence_monitor = CognitiveCoherenceMonitor()
                self.drift_detector = PersonaDriftDetector()
                self.psychotic_prevention = PsychoticFeaturePrevention()
                self.intervention_system = TherapeuticInterventionSystem()
                self.firewall = CognitiveSeparationFirewall()
                self.components_loaded = True
            except Exception as e:
                logger.warning(f"⚠️  Component loading error: {e}")
                self.components_loaded = False
        else:
            self.components_loaded = False
    
    def audit_measurement_consistency(self) -> ScientificMeasurement:
        """
        ZETEIC QUESTION: Are our measurements scientifically consistent?
        
        Test: Measure the same input multiple times - should get consistent results
        """
        logger.debug("🔬 AUDITING: Measurement consistency")
        
        if not self.components_loaded:
            return self._create_failed_measurement("Components not available")
        
        measurements = []
        test_input = torch.randn(1, 256) * 0.1  # Controlled input
        
        # Take 50 measurements of the same input
        for _ in range(50):
            try:
                coherence = self.coherence_monitor.calculate_identity_coherence(test_input)
                measurements.append(coherence)
            except Exception as e:
                logger.error(f"❌ Measurement failed: {e}")
                return self._create_failed_measurement(f"Measurement error: {e}")
        
        # Statistical analysis
        mean_val = np.mean(measurements)
        std_val = np.std(measurements)
        cv = std_val / mean_val if mean_val != 0 else float('inf')  # Coefficient of variation
        
        # Confidence interval
        sem = stats.sem(measurements)
        ci = stats.t.interval(0.95, len(measurements)-1, loc=mean_val, scale=sem)
        
        # Test for consistency (low variance is good)
        normality_test = stats.shapiro(measurements)
        
        # Reproducibility score (1.0 = perfect, 0.0 = terrible)
        reproducibility = max(0.0, 1.0 - cv)
        
        validity = "VALID" if cv < 0.1 and normality_test.pvalue > 0.05 else "INVALID"
        
        return ScientificMeasurement(
            value=mean_val,
            confidence_interval=ci,
            sample_size=len(measurements),
            p_value=normality_test.pvalue,
            effect_size=cv,
            measurement_error=std_val,
            reproducibility_score=reproducibility,
            scientific_validity=validity
        )
    
    def audit_drift_detection_sensitivity(self) -> ScientificMeasurement:
        """
        ZETEIC QUESTION: Does drift detection actually detect drift?
        
        Test: Create known amounts of drift and see if detector finds them
        """
        logger.debug("🔬 AUDITING: Drift detection sensitivity")
        
        if not self.components_loaded:
            return self._create_failed_measurement("Components not available")
        
        try:
            # Establish baseline
            baseline = torch.randn(1, 256)
            self.drift_detector.baseline_cognitive_signature = baseline.clone()
            
            # Test with known drift amounts
            known_drifts = np.linspace(0.01, 0.2, 20)  # 1% to 20% drift
            detected_drifts = []
            
            for drift_amount in known_drifts:
                # Create drifted state
                noise = torch.randn(1, 256) * drift_amount
                drifted_state = baseline + noise
                
                result = self.drift_detector.monitor_cognitive_stability(drifted_state)
                detected_drift = result.get('drift_magnitude', 0)
                detected_drifts.append(detected_drift)
            
            # Calculate correlation between known and detected drift
            correlation, p_value = stats.pearsonr(known_drifts, detected_drifts)
            
            # Calculate measurement error
            error = np.mean(np.abs(np.array(detected_drifts) - known_drifts))
            
            # Confidence interval for correlation
            n = len(known_drifts)
            r_z = np.arctanh(correlation)
            se = 1/np.sqrt(n-3)
            ci_z = (r_z - 1.96*se, r_z + 1.96*se)
            ci = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
            
            validity = "VALID" if correlation > 0.8 and p_value < 0.05 else "INVALID"
            
            return ScientificMeasurement(
                value=correlation,
                confidence_interval=ci,
                sample_size=n,
                p_value=p_value,
                effect_size=correlation**2,  # R-squared
                measurement_error=error,
                reproducibility_score=correlation,
                scientific_validity=validity
            )
            
        except Exception as e:
            return self._create_failed_measurement(f"Drift detection error: {e}")
    
    def audit_firewall_discrimination(self) -> ScientificMeasurement:
        """
        ZETEIC QUESTION: Can the firewall actually distinguish clean from contaminated data?
        
        Test: Give it known clean and contaminated data, measure discrimination ability
        """
        logger.debug("🔬 AUDITING: Firewall discrimination ability")
        
        if not self.components_loaded:
            return self._create_failed_measurement("Components not available")
        
        try:
            # Create known clean and contaminated samples
            clean_samples = [torch.randn(256) * 0.0001 for _ in range(30)]  # Very small values = clean
            contaminated_samples = [torch.randn(256) * 0.1 for _ in range(30)]  # Large values = contaminated
            
            clean_detected = []
            contaminated_detected = []
            
            # Test clean samples
            for sample in clean_samples:
                try:
                    self.firewall.validate_cognitive_purity(sample)
                    clean_detected.append(True)  # Correctly passed as clean
                except:
                    clean_detected.append(False)  # Incorrectly blocked
            
            # Test contaminated samples  
            for sample in contaminated_samples:
                try:
                    self.firewall.validate_cognitive_purity(sample)
                    contaminated_detected.append(False)  # Incorrectly passed
                except:
                    contaminated_detected.append(True)  # Correctly blocked
            
            # Calculate discrimination metrics
            true_negative_rate = np.mean(clean_detected)  # Correctly identified clean
            true_positive_rate = np.mean(contaminated_detected)  # Correctly identified contaminated
            accuracy = (sum(clean_detected) + sum(contaminated_detected)) / (len(clean_detected) + len(contaminated_detected))
            
            # Statistical test
            total_correct = sum(clean_detected) + sum(contaminated_detected)
            total_samples = len(clean_detected) + len(contaminated_detected)
            
            # Binomial test against chance (50%)
            p_value = stats.binom_test(total_correct, total_samples, 0.5, alternative='greater')
            
            # Confidence interval for accuracy
            ci = stats.binom.interval(0.95, total_samples, total_correct) 
            ci_normalized = (ci[0]/total_samples, ci[1]/total_samples)
            
            # Effect size (Cohen's d equivalent for proportions)
            effect_size = accuracy - 0.5  # Improvement over chance
            
            validity = "VALID" if accuracy > 0.8 and p_value < 0.05 else "INVALID"
            
            return ScientificMeasurement(
                value=accuracy,
                confidence_interval=ci_normalized,
                sample_size=total_samples,
                p_value=p_value,
                effect_size=effect_size,
                measurement_error=np.sqrt(accuracy * (1-accuracy) / total_samples),
                reproducibility_score=accuracy,
                scientific_validity=validity
            )
            
        except Exception as e:
            return self._create_failed_measurement(f"Firewall test error: {e}")
    
    def audit_performance_reproducibility(self) -> ScientificMeasurement:
        """
        ZETEIC QUESTION: Are performance claims reproducible across multiple trials?
        
        Test: Run multiple performance tests and measure consistency
        """
        logger.debug("🔬 AUDITING: Performance reproducibility")
        
        if not self.components_loaded:
            return self._create_failed_measurement("Components not available")
        
        try:
            performance_measurements = []
            
            # Run 20 independent performance trials
            for trial in range(20):
                start_time = time.time()
                operations = 0
                test_duration = 0.5  # 0.5 second test
                
                while time.time() - start_time < test_duration:
                    # Standard operation
                    test_data = torch.randn(1, 256)
                    _ = self.coherence_monitor.calculate_identity_coherence(test_data)
                    operations += 1
                
                actual_duration = time.time() - start_time
                ops_per_second = operations / actual_duration
                performance_measurements.append(ops_per_second)
            
            # Statistical analysis
            mean_perf = np.mean(performance_measurements)
            std_perf = np.std(performance_measurements)
            cv = std_perf / mean_perf
            
            # Confidence interval
            sem = stats.sem(performance_measurements)
            ci = stats.t.interval(0.95, len(performance_measurements)-1, loc=mean_perf, scale=sem)
            
            # Test for consistency
            normality_test = stats.shapiro(performance_measurements)
            
            # Reproducibility (lower CV = more reproducible)
            reproducibility = max(0.0, 1.0 - cv)
            
            validity = "VALID" if cv < 0.3 and normality_test.pvalue > 0.05 else "INVALID"
            
            return ScientificMeasurement(
                value=mean_perf,
                confidence_interval=ci,
                sample_size=len(performance_measurements),
                p_value=normality_test.pvalue,
                effect_size=cv,
                measurement_error=std_perf,
                reproducibility_score=reproducibility,
                scientific_validity=validity
            )
            
        except Exception as e:
            return self._create_failed_measurement(f"Performance test error: {e}")
    
    def _create_failed_measurement(self, reason: str) -> ScientificMeasurement:
        """Create a failed measurement result"""
        return ScientificMeasurement(
            value=0.0,
            confidence_interval=(0.0, 0.0),
            sample_size=0,
            p_value=1.0,
            effect_size=0.0,
            measurement_error=float('inf'),
            reproducibility_score=0.0,
            scientific_validity=f"FAILED: {reason}"
        )
    
    def conduct_full_audit(self):
        """Conduct comprehensive zeteic audit"""
        logger.info("\n" + "=" * 80)
        logger.debug("🔬 ZETEIC SCIENTIFIC AUDIT - PHASE 2 VALIDATION")
        logger.info("=" * 80)
        logger.info("Applying rigorous scientific methodology to validate all claims")
        logger.info()
        
        # Run all audits
        audits = {
            "Measurement Consistency": self.audit_measurement_consistency(),
            "Drift Detection Sensitivity": self.audit_drift_detection_sensitivity(),
            "Firewall Discrimination": self.audit_firewall_discrimination(),
            "Performance Reproducibility": self.audit_performance_reproducibility()
        }
        
        # Report results
        logger.info("\n📊 AUDIT RESULTS")
        logger.info("=" * 50)
        
        valid_tests = 0
        total_tests = 0
        avg_reproducibility = 0
        
        for test_name, result in audits.items():
            total_tests += 1
            
            logger.debug(f"\n🔍 {test_name}")
            logger.info(f"   Value: {result.value:.6f}")
            logger.info(f"   95% CI: ({result.confidence_interval[0]:.6f}, {result.confidence_interval[1]:.6f})
            logger.info(f"   Sample Size: {result.sample_size}")
            logger.info(f"   P-value: {result.p_value:.6f}")
            logger.info(f"   Effect Size: {result.effect_size:.6f}")
            logger.error(f"   Measurement Error: {result.measurement_error:.6f}")
            logger.info(f"   Reproducibility: {result.reproducibility_score:.6f}")
            logger.info(f"   Scientific Validity: {result.scientific_validity}")
            
            if "VALID" in result.scientific_validity:
                valid_tests += 1
                logger.info("   ✅ SCIENTIFICALLY VALID")
            else:
                logger.error("   ❌ SCIENTIFICALLY INVALID")
            
            avg_reproducibility += result.reproducibility_score
        
        avg_reproducibility /= total_tests if total_tests > 0 else 1
        
        # Overall assessment
        logger.info("\n" + "=" * 80)
        logger.info("🎯 ZETEIC AUDIT CONCLUSIONS")
        logger.info("=" * 80)
        
        logger.info(f"📈 Valid Tests: {valid_tests}/{total_tests} ({valid_tests/total_tests*100:.1f}%)
        logger.info(f"📊 Average Reproducibility: {avg_reproducibility:.3f}")
        
        if valid_tests >= 3 and avg_reproducibility > 0.7:
            logger.info("🎉 VERDICT: SCIENTIFICALLY RIGOROUS")
            logger.info("   Phase 2 claims are well-supported by empirical evidence")
        elif valid_tests >= 2 and avg_reproducibility > 0.5:
            logger.warning("⚠️  VERDICT: NEEDS IMPROVEMENT")
            logger.info("   Some claims require further validation")
        else:
            logger.error("❌ VERDICT: INSUFFICIENT SCIENTIFIC RIGOR")
            logger.info("   Major claims lack adequate empirical support")
        
        logger.debug("\n🔬 Zeteic principle applied: Question everything, validate empirically")
        logger.info("=" * 80)

if __name__ == "__main__":
    auditor = ZeteticAuditor()
    auditor.conduct_full_audit() 