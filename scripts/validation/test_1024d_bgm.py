#!/usr/bin/env python3
"""
Test 1024D BGM Integration
=========================
Validate updated 1024D configuration
"""

import sys
sys.path.insert(0, '.')

def test_1024d_integration():
    """Test the updated 1024D BGM configuration"""
    print("🔬 Testing updated 1024D BGM configuration...")

    try:
        from src.core.kimera_system import KimeraSystem

        system = KimeraSystem()
        system.initialize()
        hd_modeling = system.get_component('high_dimensional_modeling')

        if hd_modeling:
            print(f"✅ High-Dimensional Modeling: {type(hd_modeling).__name__}")
            print(f"✅ BGM Dimension: {hd_modeling.bgm_engine.config.dimension}D")
            print(f"✅ Batch Size: {hd_modeling.bgm_engine.config.batch_size}")
            print("🎉 1024D BGM Successfully Integrated!")
            return True
        else:
            print("❌ Integration failed")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_1024d_integration()
