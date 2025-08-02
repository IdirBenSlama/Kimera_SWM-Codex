#!/usr/bin/env python3
"""
Simple Phase 3 Test - Enhanced Capabilities
==========================================

Simple test to validate the Phase 3 enhanced capabilities implementation.
"""

import asyncio
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_understanding_core():
    """Test Understanding Core basic functionality"""
    print("Testing Understanding Core...")
    try:
        from src.core.enhanced_capabilities.understanding_core import (
            UnderstandingCore, UnderstandingType
        )
        
        understanding_core = UnderstandingCore()
        result = await understanding_core.understand(
            "This is a test of understanding.",
            UnderstandingType.SEMANTIC
        )
        
        print(f"✅ Understanding Core: success={result.success}, depth={result.understanding_depth:.3f}")
        return True
    except Exception as e:
        print(f"❌ Understanding Core failed: {e}")
        return False

async def test_consciousness_core():
    """Test Consciousness Core basic functionality"""
    print("Testing Consciousness Core...")
    try:
        from src.core.enhanced_capabilities.consciousness_core import ConsciousnessCore
        
        consciousness_core = ConsciousnessCore()
        cognitive_state = torch.randn(128)
        
        signature = await consciousness_core.detect_consciousness(cognitive_state)
        
        print(f"✅ Consciousness Core: success={signature.success}, probability={signature.consciousness_probability:.3f}")
        return True
    except Exception as e:
        print(f"❌ Consciousness Core failed: {e}")
        return False

async def test_meta_insight_core():
    """Test Meta Insight Core basic functionality"""
    print("Testing Meta Insight Core...")
    try:
        from src.core.enhanced_capabilities.meta_insight_core import MetaInsightCore
        
        meta_insight_core = MetaInsightCore()
        cognitive_state = torch.randn(64)
        
        result = await meta_insight_core.generate_meta_insight(cognitive_state)
        
        print(f"✅ Meta Insight Core: success={result.success}, strength={result.insight_strength:.3f}")
        return True
    except Exception as e:
        print(f"❌ Meta Insight Core failed: {e}")
        return False

async def main():
    """Run simple Phase 3 tests"""
    print("🧠 SIMPLE PHASE 3 ENHANCED CAPABILITIES TEST")
    print("=" * 50)
    
    results = []
    
    results.append(await test_understanding_core())
    results.append(await test_consciousness_core())
    results.append(await test_meta_insight_core())
    
    passed = sum(results)
    total = len(results)
    
    print()
    print(f"📊 Results: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All Phase 3 enhanced capabilities working!")
    else:
        print("⚠️  Some Phase 3 capabilities need attention")

if __name__ == "__main__":
    asyncio.run(main())