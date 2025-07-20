#!/usr/bin/env python3
"""
Quick test of Mirror Portal functionality
"""

import asyncio
import sys
import os

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def quick_test():
    logger.info("\n🌀 QUICK MIRROR PORTAL TEST\n")
    
    try:
        # Import and create engine
        from backend.engines.geoid_mirror_portal_engine import GeoidMirrorPortalEngine, QuantumSemanticState
        
        engine = GeoidMirrorPortalEngine()
        logger.info("✅ Engine created")
        
        # Create portal
        semantic_content = {"meaning": 0.8, "understanding": 0.7}
        symbolic_content = {"type": "test", "formula": "ψ"}
        
        sem_geoid, sym_geoid, portal = await engine.create_dual_state_geoid(
            semantic_content, symbolic_content, 0.9
        )
        
        logger.info(f"✅ Portal created: {portal.portal_id}")
        logger.info(f"   Coherence: {portal.coherence_strength:.2%}")
        logger.info(f"   Contact: {portal.contact_point}")
        
        # Test transition
        transition = await engine.transition_through_portal(
            portal.portal_id,
            QuantumSemanticState.WAVE_SUPERPOSITION,
            1.0
        )
        
        logger.info(f"✅ Transition: {transition.source_state.value} → {transition.target_state.value}")
        logger.info(f"   Success: {transition.source_state != transition.target_state}")
        
        logger.info("\n🎉 Mirror Portal is working!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())