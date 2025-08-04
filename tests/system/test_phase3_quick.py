#!/usr/bin/env python3
"""
QUICK PHASE 3 STATUS CHECK
==========================
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def quick_phase3_check():
    """Quick check of Phase 3 engine status"""

    print("QUICK PHASE 3 STATUS CHECK")
    print("=" * 30)

    try:
        from src.core.kimera_system import get_kimera_system

        kimera = get_kimera_system()
        kimera.initialize()

        status = kimera.get_system_status()

        phase3_engines = [
            "ethical_reasoning_engine_ready",
            "unsupervised_cognitive_learning_engine_ready",
            "complexity_analysis_engine_ready",
            "quantum_field_engine_ready",
        ]

        ready = sum(1 for engine in phase3_engines if status.get(engine, False))

        print(f"Phase 3 Engines: {ready}/4 ready")

        for engine in phase3_engines:
            name = engine.replace("_ready", "").replace("_", " ").title()
            state = "READY" if status.get(engine, False) else "NOT READY"
            print(f"  {name}: {state}")

        return ready >= 3

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = quick_phase3_check()
    print(f"\nResult: {'SUCCESS' if success else 'PARTIAL'}")
    sys.exit(0 if success else 1)
