#!/usr/bin/env python3
"""
Simple test to check if imports work after fixes
"""

import sys
import traceback


def test_imports():
    """Test basic imports"""
    print("Testing imports...")

    try:
        # Test torch operations
        import torch

        print("✅ PyTorch imported")

        # Test cosine similarity
        a = torch.randn(10)
        b = torch.randn(10)
        sim = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        print(f"✅ torch.cosine_similarity works: {sim.item():.3f}")

        # Test FFT
        try:
            fft_result = torch.fft.fft(a)
            print("✅ torch.fft.fft works")
        except Exception as e:
            print(f"⚠️  torch.fft.fft failed: {e}")

        # Test enhanced capabilities imports
        from src.core.enhanced_capabilities.understanding_core import UnderstandingCore

        print("✅ UnderstandingCore imported")

        from src.core.enhanced_capabilities.consciousness_core import ConsciousnessCore

        print("✅ ConsciousnessCore imported")

        from src.core.enhanced_capabilities.meta_insight_core import MetaInsightCore

        print("✅ MetaInsightCore imported")

        from src.core.enhanced_capabilities.field_dynamics_core import FieldDynamicsCore

        print("✅ FieldDynamicsCore imported")

        from src.core.enhanced_capabilities.learning_core import LearningCore

        print("✅ LearningCore imported")

        from src.core.enhanced_capabilities.linguistic_intelligence_core import (
            LinguisticIntelligenceCore,
        )

        print("✅ LinguisticIntelligenceCore imported")

        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ FIXES APPEAR TO BE WORKING!")
    else:
        print("\n❌ STILL HAVE ISSUES TO FIX")
