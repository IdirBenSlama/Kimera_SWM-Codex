#!/usr/bin/env python3
"""
Debug consciousness core boolean tensor issue
"""

import asyncio
import torch
import traceback

async def debug_consciousness_issue():
    """Debug the specific consciousness core issue"""
    print("🔍 DEBUGGING CONSCIOUSNESS CORE BOOLEAN TENSOR ISSUE")
    print("=" * 60)
    
    try:
        from src.core.enhanced_capabilities.consciousness_core import (
            ConsciousnessCore, ConsciousnessMode
        )
        
        consciousness_core = ConsciousnessCore(
            default_mode=ConsciousnessMode.UNIFIED,
            consciousness_threshold=0.6,
            device="cpu"
        )
        print("✅ Consciousness Core created")
        
        # Complex cognitive state for consciousness detection
        cognitive_state = torch.randn(256) * 0.5 + 0.3  # Structured noise
        energy_field = torch.sin(torch.linspace(0, 4*3.14159, 256)) * 0.3 + 0.1
        
        print(f"   Cognitive state shape: {cognitive_state.shape}")
        print(f"   Energy field shape: {energy_field.shape}")
        
        signature = await consciousness_core.detect_consciousness(
            cognitive_state, energy_field, 
            context={"consciousness_test": True, "high_complexity": True}
        )
        
        print("❌ This should have failed with Boolean tensor error")
        print(f"   Result: {signature.consciousness_probability}")
        
    except Exception as e:
        print(f"✅ Found the error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Check if it's the boolean tensor issue
        if "Boolean value of Tensor" in str(e):
            print("\n🎯 CONFIRMED: This is the Boolean tensor ambiguity issue")
            print("💡 Need to fix tensor comparisons in consciousness detection")
        else:
            print(f"\n⚠️  Different error than expected: {type(e).__name__}")

if __name__ == "__main__":
    asyncio.run(debug_consciousness_issue())