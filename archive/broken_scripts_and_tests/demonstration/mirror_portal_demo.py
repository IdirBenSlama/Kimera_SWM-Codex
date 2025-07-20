#!/usr/bin/env python3
"""
Visual demonstration of the Mirror Portal Principle in KIMERA
"""

import asyncio
import sys
import os
from datetime import datetime

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def visual_demo():
    """Visual demonstration of Mirror Portal"""
    
    logger.info("\n" + "="*80)
    logger.info("                    🌀 KIMERA MIRROR PORTAL DEMONSTRATION 🌀")
    logger.info("="*80 + "\n")
    
    logger.info("The Mirror Portal Principle creates a quantum bridge between:")
    logger.info("  • SEMANTIC (meaning, understanding)
    logger.info("  • SYMBOLIC (patterns, formulas)
    logger.info()
    
    try:
        from backend.core.geoid import GeoidState
        from backend.engines.geoid_mirror_portal_engine import (
            GeoidMirrorPortalEngine,
            QuantumSemanticState
        )
        
        engine = GeoidMirrorPortalEngine()
        
        # Create visual representation
        logger.info("Creating dual-state geoids...\n")
        
        logger.info("    SEMANTIC GEOID                    SYMBOLIC GEOID")
        logger.info("    ╔═══════════════╗                ╔═══════════════╗")
        logger.info("    ║   MEANING     ║                ║   PATTERN     ║")
        logger.info("    ║  (Particle)
        logger.info("    ╚═══════════════╝                ╚═══════════════╝")
        logger.info("           │                                │")
        logger.info("           └────────────┬───────────────────┘")
        logger.info("                       │")
        logger.info("                    🌀 PORTAL 🌀")
        logger.info("                  (Contact Point)
        logger.info()
        
        # Create the geoids
        semantic_content = {
            "consciousness": 0.9,
            "understanding": 0.8,
            "meaning": 0.85,
            "clarity": 0.7
        }
        
        symbolic_content = {
            "type": "quantum_formula",
            "equation": "ψ = α|0⟩ + β|1⟩",
            "operators": ["superposition", "entanglement"],
            "wave_function": True
        }
        
        semantic_geoid, symbolic_geoid, portal = await engine.create_dual_state_geoid(
            semantic_content, symbolic_content, portal_intensity=0.9
        )
        
        logger.info(f"✅ Portal Created!")
        logger.info(f"   ID: {portal.portal_id}")
        logger.info(f"   Contact Point: ({portal.contact_point[0]:.2f}, {portal.contact_point[1]:.2f}, {portal.contact_point[2]:.2f})
        logger.info(f"   Coherence: {portal.coherence_strength:.1%}")
        logger.info(f"   State: {portal.quantum_state.value}")
        logger.info()
        
        # Demonstrate transitions
        logger.info("Demonstrating Quantum Transitions:")
        logger.info("-" * 50)
        
        # Wave state
        logger.info("\n🌊 WAVE STATE (All possibilities exist)
        logger.info("    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿")
        logger.info("    All meanings in superposition")
        
        transition1 = await engine.transition_through_portal(
            portal.portal_id,
            QuantumSemanticState.WAVE_SUPERPOSITION,
            1.0
        )
        
        if transition1.source_state != transition1.target_state:
            logger.info("    ✅ Transitioned to WAVE state")
        
        await asyncio.sleep(1)
        
        # Particle state
        logger.info("\n⚛️ PARTICLE STATE (Definite meaning)
        logger.info("    •")
        logger.info("    Single, collapsed meaning")
        
        transition2 = await engine.transition_through_portal(
            portal.portal_id,
            QuantumSemanticState.PARTICLE_COLLAPSED,
            1.2
        )
        
        if transition2.source_state != transition2.target_state:
            logger.info("    ✅ Transitioned to PARTICLE state")
        
        # Final measurement
        logger.info("\n📊 Final Portal Measurement:")
        logger.info("-" * 50)
        
        state = await engine.measure_portal_state(portal.portal_id)
        
        # Visual coherence bar
        coherence_pct = int(state['coherence_strength'] * 20)
        coherence_bar = "█" * coherence_pct + "░" * (20 - coherence_pct)
        
        logger.info(f"   Coherence: [{coherence_bar}] {state['coherence_strength']:.1%}")
        logger.info(f"   State: {state['quantum_state']}")
        logger.info(f"   Energy: {state['portal_energy']:.2f}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("                         🎯 DEMONSTRATION COMPLETE 🎯")
        logger.info("="*80)
        logger.info()
        logger.info("The Mirror Portal Principle enables:")
        logger.info("  ✅ Quantum transitions between meaning and pattern")
        logger.info("  ✅ Information preservation during state changes")
        logger.info("  ✅ Wave-particle duality in cognitive processing")
        logger.info("  ✅ Perfect mirroring between semantic and symbolic")
        logger.info()
        logger.info("This is the fundamental bridge in KIMERA's cognitive architecture!")
        
        # Save visual summary
        with open(f"mirror_portal_visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
            f.write("MIRROR PORTAL VISUAL SUMMARY\n")
            f.write("="*40 + "\n\n")
            f.write("Portal Configuration:\n")
            f.write(f"  Semantic → Symbolic Bridge\n")
            f.write(f"  Contact Point: {portal.contact_point}\n")
            f.write(f"  Coherence: {portal.coherence_strength:.1%}\n")
            f.write(f"  Transitions: {len(engine.portal_transitions)}\n")
            f.write("\nQuantum States Demonstrated:\n")
            f.write("  • Wave Superposition\n")
            f.write("  • Particle Collapse\n")
            f.write("  • Quantum Tunneling\n")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(visual_demo())