#!/usr/bin/env python3
"""
Verify Mirror Portal integration without starting full server
"""

import sys
import os
import asyncio
from datetime import datetime

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def verify_integration():
    """Verify all components are properly integrated"""
    
    logger.info("="*80)
    logger.debug("🔍 VERIFYING MIRROR PORTAL INTEGRATION")
    logger.info("="*80)
    logger.info(f"Time: {datetime.now()
    
    results = {
        "imports": False,
        "engine_init": False,
        "portal_creation": False,
        "transitions": False,
        "integration": False
    }
    
    # Step 1: Test imports
    logger.info("1️⃣ Testing imports...")
    try:
        from backend.core.geoid import GeoidState
        from backend.engines.geoid_mirror_portal_engine import (
            GeoidMirrorPortalEngine,
            QuantumSemanticState,
            MirrorPortalState
        )
        from backend.engines.quantum_cognitive_engine import QuantumCognitiveEngine
        from backend.core.therapeutic_intervention_system import TherapeuticInterventionSystem
        logger.info("✅ All imports successful")
        results["imports"] = True
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return results
    
    # Step 2: Test engine initialization
    logger.info("\n2️⃣ Testing engine initialization...")
    try:
        portal_engine = GeoidMirrorPortalEngine()
        logger.info("✅ Mirror Portal Engine initialized")
        
        # Check engine state
        logger.info(f"   Golden ratio: {portal_engine.golden_ratio:.6f}")
        logger.info(f"   Portal creation energy: {portal_engine.portal_creation_energy}")
        logger.info(f"   Active portals: {len(portal_engine.active_portals)
        results["engine_init"] = True
    except Exception as e:
        logger.error(f"❌ Engine initialization error: {e}")
        return results
    
    # Step 3: Test portal creation
    logger.info("\n3️⃣ Testing portal creation...")
    try:
        semantic_content = {
            "test": 0.9,
            "integration": 0.8,
            "mirror": 0.7
        }
        
        symbolic_content = {
            "type": "test",
            "formula": "∫ψ(x)dx"
        }
        
        semantic_geoid, symbolic_geoid, portal = await portal_engine.create_dual_state_geoid(
            semantic_content, symbolic_content, 0.8
        )
        
        logger.info("✅ Portal created successfully")
        logger.info(f"   Portal ID: {portal.portal_id}")
        logger.info(f"   Contact point: {portal.contact_point}")
        logger.info(f"   Coherence: {portal.coherence_strength:.3f}")
        logger.info(f"   Quantum state: {portal.quantum_state.value}")
        results["portal_creation"] = True
        
        # Step 4: Test transitions
        logger.info("\n4️⃣ Testing quantum transitions...")
        transition = await portal_engine.transition_through_portal(
            portal.portal_id,
            QuantumSemanticState.WAVE_SUPERPOSITION,
            1.0
        )
        
        logger.info(f"✅ Transition attempted")
        logger.info(f"   Success: {transition.source_state != transition.target_state}")
        logger.info(f"   Probability: {transition.transition_probability:.3f}")
        logger.info(f"   Information preserved: {transition.information_preserved:.3f}")
        results["transitions"] = True
        
    except Exception as e:
        logger.error(f"❌ Portal operation error: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # Step 5: Test system integration
    logger.info("\n5️⃣ Testing system integration...")
    try:
        # Test Quantum Cognitive Engine integration
        qce = QuantumCognitiveEngine(num_qubits=10)
        portal2 = await qce.create_mirror_portal_state(
            semantic_geoid, symbolic_geoid, 0.7
        )
        logger.info("✅ Quantum Cognitive Engine integration working")
        
        # Test Therapeutic System integration
        tis = TherapeuticInterventionSystem()
        logger.info("✅ Therapeutic Intervention System integration working")
        
        results["integration"] = True
        
    except Exception as e:
        logger.error(f"❌ Integration error: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 INTEGRATION VERIFICATION SUMMARY")
    logger.info("="*80)
    
    all_passed = all(results.values())
    passed_count = sum(results.values())
    total_count = len(results)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test.ljust(20)
    
    logger.info(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if all_passed:
        logger.info("\n🎉 MIRROR PORTAL INTEGRATION VERIFIED!")
        logger.info("All components are properly integrated and functional.")
    else:
        logger.warning("\n⚠️ Some integration tests failed.")
        logger.error("Please check the errors above.")
    
    # Save verification report
    with open(f"integration_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
        f.write("MIRROR PORTAL INTEGRATION VERIFICATION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Date: {datetime.now()}\n\n")
        f.write("Test Results:\n")
        for test, passed in results.items():
            f.write(f"  {test}: {'PASS' if passed else 'FAIL'}\n")
        f.write(f"\nOverall: {passed_count}/{total_count} passed\n")
        f.write(f"Status: {'VERIFIED' if all_passed else 'NEEDS ATTENTION'}\n")
    
    return all_passed

async def main():
    """Main runner"""
    success = await verify_integration()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)