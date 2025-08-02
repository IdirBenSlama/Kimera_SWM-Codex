#!/usr/bin/env python3
"""
UNDERSTANDING ENGINE DEBUG TEST
==============================

Debug test to see exactly why the Understanding Engine is failing to initialize.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_understanding_engine_debug():
    """Debug the Understanding Engine initialization"""
    
    print("UNDERSTANDING ENGINE DEBUG TEST")
    print("=" * 50)
    
    try:
        # Test direct import first
        print("\nStep 1: Testing direct Understanding Engine import...")
        try:
            from src.engines.understanding_engine import create_understanding_engine
            print("✅ Understanding Engine import successful")
        except Exception as e:
            print(f"❌ Understanding Engine import failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test direct initialization
        print("\nStep 2: Testing direct Understanding Engine initialization...")
        try:
            import asyncio
            engine = asyncio.run(create_understanding_engine())
            print("✅ Understanding Engine direct initialization successful")
            print(f"   Engine type: {type(engine)}")
            if hasattr(engine, 'self_model'):
                print(f"   Self-model: {engine.self_model}")
        except Exception as e:
            print(f"❌ Understanding Engine direct initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test through Kimera system
        print("\nStep 3: Testing Understanding Engine through Kimera system...")
        try:
            from src.core.kimera_system import get_kimera_system
            
            kimera = get_kimera_system()
            
            # Test just the Understanding Engine initialization method
            print("   Testing _initialize_understanding_engine method...")
            kimera._initialize_understanding_engine()
            
            understanding_engine = kimera.get_understanding_engine()
            if understanding_engine is not None:
                print("✅ Understanding Engine initialized through Kimera system")
                print(f"   Engine: {understanding_engine}")
                return True
            else:
                print("❌ Understanding Engine is None after Kimera initialization")
                return False
                
        except Exception as e:
            print(f"❌ Understanding Engine through Kimera system failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"\n❌ DEBUG TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    success = test_understanding_engine_debug()
    
    if success:
        print("\n✅ UNDERSTANDING ENGINE DEBUG SUCCESSFUL")
        sys.exit(0)
    else:
        print("\n❌ UNDERSTANDING ENGINE DEBUG FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()