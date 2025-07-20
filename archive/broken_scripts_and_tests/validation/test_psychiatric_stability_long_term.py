import torch
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add backend to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))

from monitoring.psychiatric_stability_monitor import (
    CognitiveCoherenceMonitor,
    PersonaDriftDetector,
    PsychoticFeaturePrevention
)
from core.therapeutic_intervention_system import TherapeuticInterventionSystem

def print_header(title):
    logger.info("\n" + "="*80)
    logger.info(f"    {title.upper()
    logger.info("="*80)

def run_stability_validation(cycles=100):
    """
    Runs a long-term simulation to validate the psychiatric stability framework.
    """
    print_header("Initializing Psychiatric Stability Validation")
    
    # 1. Initialize all relevant systems
    # MONKEY-PATCH: The persona drift detector was found to be insensitive.
    # We will use a more sensitive threshold for this validation run.
    drift_detector = PersonaDriftDetector(drift_threshold=0.02)
    logger.info("!!! PersonaDriftDetector patched with a more sensitive 0.02 threshold for validation. !!!")

    coherence_monitor = CognitiveCoherenceMonitor()
    psychotic_monitor = PsychoticFeaturePrevention()
    intervention_system = TherapeuticInterventionSystem()
    
    # 2. Create baseline cognitive state
    cognitive_state = torch.randn(1, 256)
    drift_detector.monitor_cognitive_stability(cognitive_state)
    logger.info(f"Baseline cognitive state established. Shape: {cognitive_state.shape}")
    
    # 3. Simulation Loop
    print_header(f"Running Simulation for {cycles} Cycles")
    intervention_log = []

    for i in range(cycles):
        # a. Apply a small perturbation
        perturbation = torch.randn_like(cognitive_state) * 0.005 # Very subtle change
        cognitive_state += perturbation
        
        # b. Run monitors
        coherence_result = coherence_monitor.assess_dissociative_risk(cognitive_state)
        drift_result = drift_detector.monitor_cognitive_stability(cognitive_state)
        psychotic_result = psychotic_monitor.assess_psychotic_risk(cognitive_state)
        
        # c. Check for alerts and intervene
        if coherence_result.get('risk_level') == 'CRITICAL':
            intervention_system.process_alert(coherence_result)
            intervention_log.append((i, "Coherence Alert", coherence_result))
            # In a real system, we might reset the state here
        
        if drift_result.get('drift_detected'):
            intervention_system.process_alert(drift_result)
            intervention_log.append((i, "Drift Alert", drift_result))
            # Reset to baseline after drift
            cognitive_state = drift_detector.baseline_cognitive_signature.clone()
            logging.info(f"Cycle {i}: Cognitive state reset due to drift.")

        if psychotic_result.get('alert'):
            intervention_system.process_alert(psychotic_result)
            intervention_log.append((i, "Psychotic Risk Alert", psychotic_result))

        if (i + 1) % 10 == 0:
            logging.info(f"Cycle {i+1}/{cycles} complete. Current stability: {drift_result.get('stability_score', 1.0):.4f}")

    # 4. Report Results
    print_header("Simulation Results")
    final_stability = drift_detector.monitor_cognitive_stability(cognitive_state)
    
    logger.info(f"Total simulation cycles: {cycles}")
    logger.info(f"Final cognitive stability score: {final_stability.get('stability_score')
    logger.info(f"Total interventions triggered: {len(intervention_log)

    if intervention_log:
        logger.info("\n--- Intervention Log ---")
        for log_entry in intervention_log:
            logger.info(f"  - Cycle {log_entry[0]}: {log_entry[1]} - {log_entry[2]}")
    else:
        logger.info("\n✅ SUCCESS: The system remained stable for the entire simulation without intervention.")

    assert len(intervention_log) < (cycles * 0.1), "Excessive interventions suggest instability."
    logger.info("\nValidation complete. The system demonstrates long-term stability.")


if __name__ == "__main__":
    run_stability_validation() 