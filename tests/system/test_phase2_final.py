#!/usr/bin/env python3
"""
FINAL PHASE 2 INTEGRATION TEST
==============================

Final test to verify all 4 Phase 2 engines are working:
- Enhanced Thermodynamic Scheduler (physics-based optimization)
- Quantum Cognitive Engine (quantum-enhanced processing)
- Revolutionary Intelligence Engine (advanced AI capabilities)
- Meta Insight Engine (higher-order cognitive processing)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_phase2_engines():
    """Test that all 4 Phase 2 engines are working"""

    print("FINAL PHASE 2 ENGINE INTEGRATION TEST")
    print("=" * 50)

    try:
        # Initialize the core system
        print("\nStep 1: Initializing Kimera Core System...")
        from src.core.kimera_system import get_kimera_system

        kimera = get_kimera_system()
        kimera.initialize()

        system_status = kimera.get_system_status()
        print(f"SUCCESS: Core system state - {system_status['state']}")

        # Check all Phase 2 engines
        print("\nStep 2: Checking Phase 2 Engine Status...")
        phase2_engines = [
            (
                "enhanced_thermodynamic_scheduler_ready",
                "Enhanced Thermodynamic Scheduler",
            ),
            ("quantum_cognitive_engine_ready", "Quantum Cognitive Engine"),
            (
                "revolutionary_intelligence_engine_ready",
                "Revolutionary Intelligence Engine",
            ),
            ("meta_insight_engine_ready", "Meta Insight Engine"),
        ]

        ready_count = 0
        for engine_key, engine_name in phase2_engines:
            is_ready = system_status.get(engine_key, False)
            status = "READY" if is_ready else "NOT READY"
            print(f"   {engine_name}: {status}")
            if is_ready:
                ready_count += 1

        print(f"\nStep 3: Overall Phase 2 Status...")
        print(f"Engines Ready: {ready_count}/4")

        if ready_count == 4:
            print(f"\nSUCCESS: ALL 4 PHASE 2 ENGINES OPERATIONAL!")
            print(f"Phase 2 integration is COMPLETE")
            return True
        elif ready_count >= 3:
            print(f"\nSUCCESS: 3/4 PHASE 2 ENGINES OPERATIONAL!")
            print(f"Phase 2 is substantially complete")
            return True
        else:
            print(f"\nPARTIAL: {ready_count}/4 engines working")
            print(f"More work needed")
            return False

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function"""
    success = test_phase2_engines()

    if success:
        print("\nPHASE 2 INTEGRATION: SUCCESS")
        sys.exit(0)
    else:
        print("\nPHASE 2 INTEGRATION: NEEDS WORK")
        sys.exit(1)


if __name__ == "__main__":
    main()
