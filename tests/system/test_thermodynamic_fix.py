#!/usr/bin/env python3
"""
Quick test to verify Enhanced Thermodynamic Scheduler is fixed
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_thermodynamic_scheduler():
    """Test the Enhanced Thermodynamic Scheduler"""
    print("🌡️ Testing Enhanced Thermodynamic Scheduler...")

    try:
        # Test import
        from src.engines.thermodynamic_scheduler import (
            get_enhanced_thermodynamic_scheduler,
        )

        print("✅ Import successful")

        # Test initialization
        async def test_init():
            scheduler = await get_enhanced_thermodynamic_scheduler()
            return scheduler

        result = asyncio.run(test_init())
        print(f"✅ Initialization successful: {type(result).__name__}")
        print(f"   - Monitoring interval: {result.monitoring_interval}s")
        print(f"   - Target reversibility: {result.target_reversibility}")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_kimera_integration():
    """Test integration with Kimera core system"""
    print("\n🔧 Testing Kimera Core Integration...")

    try:
        from src.core.kimera_system import get_kimera_system

        kimera = get_kimera_system()
        kimera.initialize()

        status = kimera.get_system_status()
        print(f"✅ Core system state: {status['state']}")

        # Check Enhanced Thermodynamic Scheduler specifically
        scheduler_ready = status.get("enhanced_thermodynamic_scheduler_ready", False)
        print(
            f"🌡️ Enhanced Thermodynamic Scheduler: {'✅ Ready' if scheduler_ready else '❌ Not Ready'}"
        )

        return scheduler_ready

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 ENHANCED THERMODYNAMIC SCHEDULER FIX VERIFICATION")
    print("=" * 60)

    # Test 1: Direct scheduler test
    scheduler_works = test_thermodynamic_scheduler()

    # Test 2: Integration test
    integration_works = test_kimera_integration()

    # Results
    print(f"\n📊 RESULTS:")
    print(f"   Scheduler Direct Test: {'✅ PASS' if scheduler_works else '❌ FAIL'}")
    print(f"   Kimera Integration: {'✅ PASS' if integration_works else '❌ FAIL'}")

    if scheduler_works and integration_works:
        print(f"\n🎉 ENHANCED THERMODYNAMIC SCHEDULER FIXED!")
        print(f"   Phase 2 should now be complete")
        return True
    else:
        print(f"\n⚠️ Issues remain - needs further investigation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
