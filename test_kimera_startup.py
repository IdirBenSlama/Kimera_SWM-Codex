#!/usr/bin/env python3
"""
KIMERA System Startup Test
=========================
This script tests the core KIMERA components to verify they can be imported and initialized.
"""

import sys
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_kimera_core():
    """Test core KIMERA system components"""
    results = {
        "config": False,
        "core_system": False,
        "engines": False,
        "gpu": False,
        "total_tests": 4,
        "passed_tests": 0,
        "timestamp": datetime.now().isoformat()
    }
    
    print("🚀 KIMERA CORE SYSTEM TEST")
    print("=" * 50)
    
    # Test 1: Configuration System
    try:
        print("1. Testing Configuration System...")
        from backend.config.config_integration import ConfigManager
        from backend.utils.config import get_api_settings
        settings = get_api_settings()
        print(f"   ✅ Configuration loaded - Environment: {settings.environment}")
        results["config"] = True
        results["passed_tests"] += 1
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
    
    # Test 2: Core System
    try:
        print("2. Testing Core System...")
        from backend.core.kimera_system import get_kimera_system
        kimera = get_kimera_system()
        print(f"   ✅ Core system initialized - Status: {kimera.get_status()}")
        results["core_system"] = True
        results["passed_tests"] += 1
    except Exception as e:
        print(f"   ❌ Core system failed: {e}")
    
    # Test 3: Engine Components
    try:
        print("3. Testing Engine Components...")
        from backend.engines.contradiction_engine import ContradictionEngine
        from backend.engines.foundational_thermodynamic_engine import FoundationalThermodynamicEngine
        
        contradiction_engine = ContradictionEngine()
        thermo_engine = FoundationalThermodynamicEngine()
        print(f"   ✅ Engines initialized successfully")
        results["engines"] = True
        results["passed_tests"] += 1
    except Exception as e:
        print(f"   ❌ Engine initialization failed: {e}")
        traceback.print_exc()
    
    # Test 4: GPU Detection
    try:
        print("4. Testing GPU Detection...")
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            print(f"   ✅ GPU Available: {gpu_name}")
            print(f"   📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"   ⚠️  GPU not available, using CPU")
        results["gpu"] = torch.cuda.is_available()
        results["passed_tests"] += 1
    except Exception as e:
        print(f"   ❌ GPU detection failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 KIMERA TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
    print(f"Success Rate: {(results['passed_tests']/results['total_tests']*100):.1f}%")
    
    if results["passed_tests"] >= 3:
        print("🟢 KIMERA CORE SYSTEM: OPERATIONAL")
        print("✅ Ready for full startup!")
    else:
        print("🟡 KIMERA CORE SYSTEM: PARTIAL")
        print("⚠️  Some components need attention")
    
    return results

if __name__ == "__main__":
    try:
        results = test_kimera_core()
        
        # Additional system info
        print(f"\n📋 System Information:")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   Platform: {sys.platform}")
        print(f"   Timestamp: {results['timestamp']}")
        
        # Exit with appropriate code
        if results["passed_tests"] >= 3:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Partial failure
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(2)  # Critical failure 