#!/usr/bin/env python3
"""
Verify Thermodynamic Systems Readiness
======================================
Check the actual state of thermodynamic components
"""

import asyncio
import sys
sys.path.insert(0, '.')

async def verify_thermodynamic_systems():
    """Verify thermodynamic systems readiness in detail"""
    logger.info("🔍 Verifying Thermodynamic Systems Status...")

    try:
        from src.core.kimera_system import KimeraSystem

        # Get instance
        system = KimeraSystem()
        system.initialize()

        # Get thermodynamic components
        thermo_integration = system.get_thermodynamic_integration()

        logger.info("\n📊 Thermodynamic Integration Status:")
        if thermo_integration:
            logger.info(f"   Component exists: ✅")
            logger.info(f"   Type: {type(thermo_integration).__name__}")

            # Check engines_initialized before async init
            if hasattr(thermo_integration, 'engines_initialized'):
                logger.info(f"   Engines initialized (before async): {thermo_integration.engines_initialized}")

            # Run async initialization
            logger.info("\n🔄 Running async initialization...")
            success = await system.initialize_thermodynamic_systems()
            logger.info(f"   Async init result: {'✅ Success' if success else '❌ Failed'}")

            # Check engines_initialized after async init
            if hasattr(thermo_integration, 'engines_initialized'):
                logger.info(f"   Engines initialized (after async): {thermo_integration.engines_initialized}")

            # Check individual engines
            logger.info("\n📦 Individual Engines:")
            for engine_name in ['heat_pump', 'maxwell_demon', 'vortex_battery', 'consciousness_detector', 'monitor']:
                if hasattr(thermo_integration, engine_name):
                    engine = getattr(thermo_integration, engine_name)
                    logger.info(f"   {engine_name}: {'✅ Present' if engine else '❌ Missing'}")
        else:
            logger.info(f"   Component exists: ❌")

        # Final readiness check
        logger.info("\n🎯 Final Readiness Check:")
        ready = system.is_thermodynamic_systems_ready()
        logger.info(f"   Thermodynamic Systems Ready: {'✅ YES' if ready else '❌ NO'}")

        # Show why not ready if false
        if not ready:
            logger.info("\n⚠️ Readiness Requirements:")
            logger.info("   - thermodynamic_integration must exist ✅")
            logger.info(f"   - engines_initialized must be True: {'✅' if thermo_integration and hasattr(thermo_integration, 'engines_initialized') and thermo_integration.engines_initialized else '❌'}")

        return ready

    except Exception as e:
        logger.info(f"❌ Error: {e}")
        import traceback
import logging
logger = logging.getLogger(__name__)
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    ready = asyncio.run(verify_thermodynamic_systems())
    if ready:
        logger.info("\n🎉 Thermodynamic Systems FULLY READY!")
        return 0
    else:
        logger.info("\n⚠️ Thermodynamic Systems Not Fully Ready")
        return 1

if __name__ == "__main__":
    exit(main())
