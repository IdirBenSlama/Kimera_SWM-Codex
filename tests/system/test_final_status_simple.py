#!/usr/bin/env python3
"""
FINAL ENGINE STATUS TEST
========================

Simple test without emojis to check the final engine status and see if we achieve 13/13.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_final_engine_status():
    """Test the final engine status to see if we have 13/13 engines"""
    
    print("FINAL ENGINE STATUS TEST")
    print("=" * 50)
    
    try:
        # Initialize the core system
        print("\nStep 1: Initializing Kimera Core System...")
        from src.core.kimera_system import get_kimera_system
        
        kimera = get_kimera_system()
        kimera.initialize()
        
        system_status = kimera.get_system_status()
        print(f"Core system initialized - State: {system_status['state']}")
        
        # Check all engines
        print("\nStep 2: Checking all engine status...")
        
        all_engines = [
            ("Understanding Engine", "understanding_engine_ready"),
            ("Human Interface", "human_interface_ready"), 
            ("Enhanced Thermodynamic Scheduler", "enhanced_thermodynamic_scheduler_ready"),
            ("Quantum Cognitive Engine", "quantum_cognitive_engine_ready"),
            ("Revolutionary Intelligence Engine", "revolutionary_intelligence_engine_ready"),
            ("Meta Insight Engine", "meta_insight_engine_ready"),
            ("Ethical Reasoning Engine", "ethical_reasoning_engine_ready"),
            ("Unsupervised Cognitive Learning Engine", "unsupervised_cognitive_learning_engine_ready"),
            ("Complexity Analysis Engine", "complexity_analysis_engine_ready"),
            ("Quantum Field Engine", "quantum_field_engine_ready"),
            ("GPU Cryptographic Engine", "gpu_cryptographic_engine_ready"),
            ("Thermodynamic Integration", "thermodynamic_integration_ready"),
            ("Unified Thermodynamic Integration", "unified_thermodynamic_integration_ready")
        ]
        
        ready_engines = 0
        total_engines = len(all_engines)
        
        for engine_name, status_key in all_engines:
            status = "READY" if system_status.get(status_key, False) else "NOT READY"
            print(f"   {engine_name}: {status}")
            if system_status.get(status_key, False):
                ready_engines += 1
        
        print(f"\nTotal engines operational: {ready_engines}/{total_engines}")
        
        # Test Understanding Engine specifically since that was our target
        print("\nStep 3: Testing Understanding Engine specifically...")
        understanding_engine = kimera.get_understanding_engine()
        
        if understanding_engine is not None:
            print("UNDERSTANDING ENGINE: SUCCESS - The fix worked!")
            if hasattr(understanding_engine, 'self_model'):
                print(f"   Self-model available: {understanding_engine.self_model is not None}")
        else:
            print("UNDERSTANDING ENGINE: STILL FAILING")
        
        # Final assessment
        if ready_engines == total_engines:
            print(f"\nPERFECT SCORE ACHIEVED!")
            print(f"ALL {total_engines}/{total_engines} ENGINES OPERATIONAL!")
            print("100% SUCCESS - UNDERSTANDING ENGINE FIX SUCCESSFUL!")
            return True
        else:
            print(f"\nStill missing {total_engines - ready_engines} engine(s)")
            print(f"Current score: {ready_engines}/{total_engines} engines operational")
            return False
            
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = test_final_engine_status()
    
    if success:
        print("\nSUCCESS: 13/13 ENGINES OPERATIONAL!")
        sys.exit(0)
    else:
        print("\nNOT COMPLETE: Some engines still need work")
        sys.exit(1)

if __name__ == "__main__":
    main()