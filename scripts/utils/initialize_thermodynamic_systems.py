#!/usr/bin/env python3
"""
Initialize Thermodynamic Systems
================================
Ensure thermodynamic systems are properly initialized using async runtime
"""

import asyncio
import sys
sys.path.insert(0, '.')

async def initialize_thermodynamic_systems():
    """Initialize thermodynamic systems with proper async handling"""
    logger.info("🔥 Initializing Thermodynamic Systems...")

    try:
        from src.core.kimera_system import KimeraSystem

        # Get instance
        system = KimeraSystem()
        system.initialize()

        logger.info("📊 Current system state:")
        logger.info(f"   Components loaded: {len(system.get_system_state()['components'])}")
        logger.info(f"   Thermodynamic ready: {system.is_thermodynamic_systems_ready()}")

        if not system.is_thermodynamic_systems_ready():
            logger.info("\n🔄 Running async thermodynamic initialization...")
            success = await system.initialize_thermodynamic_systems()

            if success:
                logger.info("✅ Thermodynamic systems initialized successfully!")
            else:
                logger.info("❌ Failed to initialize thermodynamic systems")
                return False
        else:
            logger.info("✅ Thermodynamic systems already ready!")

        # Verify final state
        logger.info("\n📊 Final system state:")
        logger.info(f"   Thermodynamic ready: {system.is_thermodynamic_systems_ready()}")

        # Get components
        thermo_integration = system.get_thermodynamic_integration()
        unified_system = system.get_unified_thermodynamic_tcse()

        if thermo_integration:
            logger.info(f"   Thermodynamic Integration: {type(thermo_integration).__name__}")
            if hasattr(thermo_integration, 'engines_initialized'):
                logger.info(f"   Engines initialized: {thermo_integration.engines_initialized}")

        if unified_system:
            logger.info(f"   Unified TCSE System: {type(unified_system).__name__}")
            if hasattr(unified_system, 'system_initialized'):
                logger.info(f"   System initialized: {unified_system.system_initialized}")

        return True

    except Exception as e:
        logger.info(f"❌ Error: {e}")
        import traceback
import logging
logger = logging.getLogger(__name__)
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    success = asyncio.run(initialize_thermodynamic_systems())
    if success:
        logger.info("\n🎉 Thermodynamic Systems Ready!")
        return 0
    else:
        logger.info("\n❌ Thermodynamic Systems Initialization Failed")
        return 1

if __name__ == "__main__":
    exit(main())
