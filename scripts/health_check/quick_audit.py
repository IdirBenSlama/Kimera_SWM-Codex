#!/usr/bin/env python3
"""
Kimera SWM Quick System Audit
============================
Fast verification of claimed 100% integration status.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def quick_audit():
    """Quick audit of integration status."""
    logger.info("🔍 KIMERA SWM QUICK AUDIT")
    logger.info("=" * 30)

    core_path = PROJECT_ROOT / "src" / "core"

    # Integration modules from roadmap
    integration_modules = [
        "axiomatic_foundation", "services", "advanced_cognitive_processing",
        "validation_and_monitoring", "quantum_and_privacy", "signal_processing",
        "geometric_optimization", "gpu_management", "high_dimensional_modeling",
        "insight_management", "barenholtz_architecture", "response_generation",
        "testing_and_protocols", "output_and_portals", "contradiction_and_pruning",
        "quantum_interface", "quantum_security_and_complexity", "quantum_thermodynamics",
        "signal_evolution_and_validation", "rhetorical_and_symbolic_processing",
        "symbolic_and_tcse", "thermodynamic_optimization", "triton_and_unsupervised_optimization",
        "vortex_dynamics", "zetetic_and_revolutionary_integration"
    ]

    results = {}
    successful = 0

    for module in integration_modules:
        module_path = core_path / module
        integration_file = module_path / "integration.py"

        status = {
            'exists': module_path.exists(),
            'has_integration': integration_file.exists() if module_path.exists() else False
        }

        if status['exists'] and status['has_integration']:
            successful += 1
            logger.info(f"✅ {module}")
        else:
            logger.info(f"❌ {module} - Missing: {'' if status['exists'] else 'directory'}{' integration.py' if status['exists'] and not status['has_integration'] else ''}")

        results[module] = status

    completion_rate = (successful / len(integration_modules)) * 100

    logger.info(f"\n📊 RESULTS:")
    logger.info(f"   Successful: {successful}/{len(integration_modules)}")
    logger.info(f"   Completion: {completion_rate:.1f}%")
    logger.info(f"   Roadmap Claim: 100%")
    logger.info(f"   Accuracy: {'✅ ACCURATE' if completion_rate == 100 else '❌ INACCURATE'}")

    # Save results
    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    report_dir = Path("docs/reports/health")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / f"{date_str}_quick_audit.json"
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'claimed_completion': '100%',
            'actual_completion': f'{completion_rate:.1f}%',
            'successful_modules': successful,
            'total_modules': len(integration_modules),
            'results': results
        }, f, indent=2)

    logger.info(f"\n💾 Report saved: {report_file}")
    return completion_rate, results

if __name__ == "__main__":
    quick_audit()
