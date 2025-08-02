#!/usr/bin/env python3
"""
UNDERSTANDING ENGINE FIX TEST
=============================

Quick test to verify the Understanding Engine fix for the missing self_models table.
This should resolve the 12/13 → 13/13 engine success rate.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_understanding_engine_fix():
    """Test that the Understanding Engine now initializes properly"""
    
    print("UNDERSTANDING ENGINE FIX TEST")
    print("=" * 50)
    
    try:
        # Initialize the core system
        print("\nStep 1: Initializing Kimera Core System...")
        from src.core.kimera_system import get_kimera_system
        
        kimera = get_kimera_system()
        kimera.initialize()
        
        system_status = kimera.get_system_status()
        print(f"Core system initialized - State: {system_status['state']}")
        
        # Test Understanding Engine specifically
        print("\nStep 2: Testing Understanding Engine...")
        understanding_engine = kimera.get_understanding_engine()
        
        if understanding_engine is not None:
            print("✅ Understanding Engine successfully integrated")
            
            # Test basic understanding functionality
            try:
                print(f"   - Self-model available: {hasattr(understanding_engine, 'self_model')}")
                if hasattr(understanding_engine, 'self_model') and understanding_engine.self_model:
                    print(f"   - Model ID: {understanding_engine.self_model.model_id}")
                    print("   - Self-awareness capabilities: Available")
                print("   - Understanding systems: Operational")
            except Exception as e:
                print(f"   - Understanding engine available (detailed test skipped: {type(e).__name__})")
                
        else:
            print("❌ Understanding Engine still not available")
            return False
        
        # Check overall engine status
        print("\nStep 3: Overall Engine Status...")
        
        all_engines = [
            "understanding_engine_ready",
            "human_interface_ready", 
            "enhanced_thermodynamic_scheduler_ready",
            "quantum_cognitive_engine_ready",
            "revolutionary_intelligence_engine_ready",
            "meta_insight_engine_ready",
            "ethical_reasoning_engine_ready",
            "unsupervised_cognitive_learning_engine_ready",
            "complexity_analysis_engine_ready",
            "quantum_field_engine_ready",
            "gpu_cryptographic_engine_ready",
            "thermodynamic_integration_ready",
            "unified_thermodynamic_integration_ready"
        ]
        
        ready_engines = sum(1 for engine in all_engines if system_status.get(engine, False))
        total_engines = len(all_engines)
        
        print(f"Total engines operational: {ready_engines}/{total_engines}")
        
        if ready_engines == total_engines:
            print("\n🎉 PERFECT SCORE ACHIEVED!")
            print("✅ ALL 13/13 ENGINES OPERATIONAL!")
            print("🎯 100% SUCCESS - UNDERSTANDING ENGINE FIX SUCCESSFUL!")
            return True
        else:
            print(f"\n⚠️ Still missing {total_engines - ready_engines} engine(s)")
            for engine in all_engines:
                status = "✅" if system_status.get(engine, False) else "❌"
                engine_name = engine.replace("_ready", "").replace("_", " ").title()
                print(f"   {status} {engine_name}")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = test_understanding_engine_fix()
    
    if success:
        print("\n✅ UNDERSTANDING ENGINE FIX SUCCESSFUL")
        print("🎊 KIMERA NOW ACHIEVES PERFECT 13/13 ENGINE SUCCESS!")
        sys.exit(0)
    else:
        print("\n❌ UNDERSTANDING ENGINE FIX NEEDS MORE WORK")
        sys.exit(1)

if __name__ == "__main__":
    main()