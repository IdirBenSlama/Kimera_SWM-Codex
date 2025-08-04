#!/usr/bin/env python3
"""
Simple System Health Check
=========================
Quick validation of Kimera system health
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.kimera_system import KimeraSystem
import logging
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("🔍 KIMERA SYSTEM QUICK HEALTH CHECK")
    logger.info("=" * 60)
    
    try:
        # Initialize system
        system = KimeraSystem()
        system.initialize()
        
        # Get system state
        state = system.get_system_state()
        
        logger.info(f"✅ System State: {state['state']}")
        logger.info(f"✅ Device: {state['device']}")
        logger.info(f"✅ GPU Acceleration: {state['gpu_acceleration_enabled']}")
        logger.info(f"✅ Components Loaded: {len(state['components'])}")
        
        # Check specific components
        logger.info("\n📦 Component Status:")
        
        # High-Dimensional Modeling
        hd_modeling = system.get_component('high_dimensional_modeling')
        if hd_modeling:
            logger.info(f"✅ High-Dimensional Modeling: Active")
            logger.info(f"   - BGM Dimension: {hd_modeling.bgm_engine.config.dimension}D")
            logger.info(f"   - Batch Size: {hd_modeling.bgm_engine.config.batch_size}")
        else:
            logger.info("❌ High-Dimensional Modeling: Not loaded")
        
        # Thermodynamic Systems
        if system.is_thermodynamic_systems_ready():
            logger.info("✅ Thermodynamic Systems: Ready")
        else:
            logger.info("⚠️ Thermodynamic Systems: Not ready")
        
        # GPU Status
        gpu_manager = system.get_gpu_manager()
        if gpu_manager:
            logger.info("✅ GPU Manager: Active")
        else:
            logger.info("⚠️ GPU Manager: Not available")
        
        logger.info("\n✅ HEALTH CHECK COMPLETE - System Operational")
        
    except Exception as e:
        logger.info(f"\n❌ HEALTH CHECK FAILED: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
