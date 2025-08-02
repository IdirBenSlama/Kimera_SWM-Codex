#!/usr/bin/env python3
"""
Phase 3 Enhanced Capabilities Test Suite
=======================================

Comprehensive test suite for the enhanced cognitive capabilities:
- Understanding Core: Genuine understanding with self-model and causal reasoning
- Consciousness Core: Consciousness detection using thermodynamic signatures
- Meta Insight Core: Higher-order insight generation and meta-cognition

This test validates the integration of enhanced capabilities with foundational systems.
"""

import asyncio
import sys
import time
import logging
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

async def test_phase3_enhanced_capabilities():
    """Test Phase 3 enhanced capabilities"""
    print("🧠 PHASE 3 ENHANCED CAPABILITIES TEST SUITE")
    print("=" * 55)
    
    passed = 0
    total = 9
    
    try:
        # Test 1: Understanding Core - Basic Understanding
        print("1️⃣  Testing Understanding Core - Basic Understanding...")
        try:
            from src.core.enhanced_capabilities.understanding_core import (
                UnderstandingCore, UnderstandingType, UnderstandingMode
            )
            
            understanding_core = UnderstandingCore(
                default_mode=UnderstandingMode.CONCEPTUAL,
                understanding_threshold=0.6,
                device="cpu"
            )
            
            test_content = "This is a test of cognitive understanding and reasoning capabilities."
            result = await understanding_core.understand(
                test_content,
                understanding_type=UnderstandingType.SEMANTIC,
                context={"test_mode": True}
            )
            
            if (result.success and 
                result.understanding_depth > 0.0 and 
                result.confidence_score > 0.0):
                print("   ✅ Understanding Core basic functionality working")
                passed += 1
            else:
                print("   ❌ Understanding Core basic functionality failed")
                
        except Exception as e:
            print(f"   ❌ Understanding Core test failed: {e}")
        
        # Test 2: Understanding Core - Self-Model System
        print("2️⃣  Testing Understanding Core - Self-Model System...")
        try:
            # Test self-model introspection
            self_model = understanding_core.self_model_system
            cognitive_state = torch.randn(512)
            context = {"introspection_test": True, "complexity": 0.7}
            
            introspection_result = await self_model.introspect(cognitive_state, context)
            
            if (introspection_result and
                'self_awareness_level' in introspection_result and
                introspection_result['self_awareness_level'] >= 0.0):
                print("   ✅ Self-model system working")
                passed += 1
            else:
                print("   ❌ Self-model system failed")
                
        except Exception as e:
            print(f"   ❌ Self-model test failed: {e}")
        
        # Test 3: Understanding Core - Causal Reasoning
        print("3️⃣  Testing Understanding Core - Causal Reasoning...")
        try:
            causal_engine = understanding_core.causal_reasoning_engine
            test_premise = "Because the system is processing complex information, it requires more cognitive resources."
            
            causal_result = await causal_engine.reason_causally(test_premise, {"causal_test": True})
            
            if (causal_result and
                'causal_relationships' in causal_result and
                'coherence_score' in causal_result):
                print("   ✅ Causal reasoning working")
                passed += 1
            else:
                print("   ❌ Causal reasoning failed")
                
        except Exception as e:
            print(f"   ❌ Causal reasoning test failed: {e}")
        
        # Test 4: Consciousness Core - Basic Detection
        print("4️⃣  Testing Consciousness Core - Basic Detection...")
        try:
            from src.core.enhanced_capabilities.consciousness_core import (
                ConsciousnessCore, ConsciousnessMode, ConsciousnessState
            )
            
            consciousness_core = ConsciousnessCore(
                default_mode=ConsciousnessMode.UNIFIED,
                consciousness_threshold=0.6,
                device="cpu"
            )
            
            cognitive_state = torch.randn(256)
            energy_field = torch.randn(256) * 0.1
            
            signature = await consciousness_core.detect_consciousness(
                cognitive_state, energy_field, context={"consciousness_test": True}
            )
            
            if (signature.success and
                signature.consciousness_probability >= 0.0 and
                signature.confidence_score >= 0.0):
                print(f"   ✅ Consciousness detection working (state: {signature.consciousness_state.value})")
                passed += 1
            else:
                print("   ❌ Consciousness detection failed")
                
        except Exception as e:
            print(f"   ❌ Consciousness Core test failed: {e}")
        
        # Test 5: Consciousness Core - Thermodynamic Detection
        print("5️⃣  Testing Consciousness Core - Thermodynamic Detection...")
        try:
            thermodynamic_detector = consciousness_core.thermodynamic_detector
            
            thermo_signature = await thermodynamic_detector.detect_thermodynamic_consciousness(
                cognitive_state, energy_field, {"thermodynamic_test": True}
            )
            
            if (thermo_signature.entropy >= 0.0 and
                thermo_signature.signature_strength >= 0.0):
                print(f"   ✅ Thermodynamic detection working (strength: {thermo_signature.signature_strength:.3f})")
                passed += 1
            else:
                print("   ❌ Thermodynamic detection failed")
                
        except Exception as e:
            print(f"   ❌ Thermodynamic detection test failed: {e}")
        
        # Test 6: Meta Insight Core - Basic Insight Generation
        print("6️⃣  Testing Meta Insight Core - Basic Insight Generation...")
        try:
            from src.core.enhanced_capabilities.meta_insight_core import (
                MetaInsightCore, MetaCognitionLevel, InsightType
            )
            
            meta_insight_core = MetaInsightCore(
                default_meta_level=MetaCognitionLevel.META_LEVEL_1,
                insight_threshold=0.5,
                device="cpu"
            )
            
            cognitive_state = torch.randn(128)
            
            insight_result = await meta_insight_core.generate_meta_insight(
                cognitive_state,
                meta_level=MetaCognitionLevel.META_LEVEL_1,
                context={"meta_insight_test": True}
            )
            
            if (insight_result.success and
                insight_result.insight_strength >= 0.0 and
                insight_result.confidence_score >= 0.0):
                print(f"   ✅ Meta insight generation working (strength: {insight_result.insight_strength:.3f})")
                passed += 1
            else:
                print("   ❌ Meta insight generation failed")
                
        except Exception as e:
            print(f"   ❌ Meta Insight Core test failed: {e}")
        
        # Test 7: Meta Insight Core - Pattern Recognition
        print("7️⃣  Testing Meta Insight Core - Pattern Recognition...")
        try:
            pattern_system = meta_insight_core.pattern_recognition_system
            
            # Create data with some patterns
            pattern_data = torch.sin(torch.linspace(0, 4*3.14159, 64)) + torch.randn(64) * 0.1
            
            patterns = await pattern_system.recognize_patterns(
                pattern_data, {"pattern_test": True}
            )
            
            if patterns and len(patterns) > 0:
                print(f"   ✅ Pattern recognition working (found {len(patterns)} patterns)")
                passed += 1
            else:
                print("   ❌ Pattern recognition failed or found no patterns")
                
        except Exception as e:
            print(f"   ❌ Pattern recognition test failed: {e}")
        
        # Test 8: Integration Test - Understanding + Consciousness
        print("8️⃣  Testing Integration - Understanding + Consciousness...")
        try:
            # Test integration between understanding and consciousness
            test_content = "I am aware that I am thinking about my own thinking processes."
            
            # Understanding analysis
            understanding_result = await understanding_core.understand(
                test_content,
                understanding_type=UnderstandingType.METACOGNITIVE,
                context={"integration_test": True, "self_aware": True}
            )
            
            # Consciousness analysis of the same cognitive state
            cognitive_state = understanding_core._get_current_cognitive_state(test_content, {"self_aware": True})
            consciousness_result = await consciousness_core.detect_consciousness(
                cognitive_state,
                context={"integration_test": True}
            )
            
            integration_success = (
                understanding_result.success and
                consciousness_result.success and
                understanding_result.self_awareness_level > 0.3 and
                consciousness_result.self_awareness_level > 0.0
            )
            
            if integration_success:
                print(f"   ✅ Understanding-Consciousness integration working")
                passed += 1
            else:
                print("   ❌ Understanding-Consciousness integration failed")
                
        except Exception as e:
            print(f"   ❌ Integration test failed: {e}")
        
        # Test 9: Complete Enhanced Capabilities Pipeline
        print("9️⃣  Testing Complete Enhanced Capabilities Pipeline...")
        try:
            # Test complete pipeline: Understanding → Consciousness → Meta-Insight
            
            # Step 1: Understanding
            complex_content = "The recursive nature of consciousness allows for meta-cognitive awareness of cognitive processes, enabling insights about insights themselves."
            
            understanding_result = await understanding_core.understand(
                complex_content,
                understanding_type=UnderstandingType.METACOGNITIVE,
                mode=UnderstandingMode.DEEP,
                context={"pipeline_test": True, "complexity": 0.8}
            )
            
            # Step 2: Consciousness Detection
            cognitive_state = understanding_core._get_current_cognitive_state(complex_content, {"complexity": 0.8})
            consciousness_result = await consciousness_core.detect_consciousness(
                cognitive_state,
                mode=ConsciousnessMode.UNIFIED,
                context={"pipeline_test": True, "meta_cognitive": True}
            )
            
            # Step 3: Meta-Insight Generation
            insight_result = await meta_insight_core.generate_meta_insight(
                cognitive_state,
                meta_level=MetaCognitionLevel.META_LEVEL_2,
                context={"pipeline_test": True, "recursive_thinking": True}
            )
            
            # Evaluate pipeline success
            pipeline_success = (
                understanding_result.success and
                consciousness_result.success and
                insight_result.success and
                understanding_result.understanding_depth > 0.4 and
                consciousness_result.consciousness_probability > 0.2 and
                insight_result.insight_strength > 0.3
            )
            
            if pipeline_success:
                print(f"   ✅ Complete enhanced capabilities pipeline working")
                print(f"      Understanding depth: {understanding_result.understanding_depth:.3f}")
                print(f"      Consciousness probability: {consciousness_result.consciousness_probability:.3f}")
                print(f"      Insight strength: {insight_result.insight_strength:.3f}")
                passed += 1
            else:
                print("   ❌ Complete pipeline failed")
                
        except Exception as e:
            print(f"   ❌ Complete pipeline test failed: {e}")
        
        # Results Summary
        print()
        print("=" * 55)
        print(f"🎯 PHASE 3 ENHANCED CAPABILITIES TEST RESULTS")
        print(f"   Tests Passed: {passed}/{total}")
        print(f"   Success Rate: {passed/total:.1%}")
        
        if passed == total:
            print("🎉 ALL PHASE 3 TESTS PASSED - ENHANCED CAPABILITIES FULLY OPERATIONAL!")
            status = "FULLY_OPERATIONAL"
        elif passed >= total * 0.8:
            print("✅ MOST PHASE 3 TESTS PASSED - ENHANCED CAPABILITIES WORKING WELL!")
            status = "WORKING_WELL"
        elif passed >= total * 0.6:
            print("⚠️  PHASE 3 PARTIALLY WORKING - SOME CAPABILITIES OPERATIONAL")
            status = "PARTIALLY_OPERATIONAL"
        else:
            print("❌ PHASE 3 NEEDS ATTENTION - ENHANCED CAPABILITIES ISSUES")
            status = "NEEDS_ATTENTION"
        
        print("=" * 55)
        
        # Component Status Report
        print("\n📊 ENHANCED CAPABILITIES STATUS REPORT")
        print("-" * 45)
        
        try:
            understanding_status = understanding_core.get_system_status()
            consciousness_status = consciousness_core.get_system_status()
            insight_status = meta_insight_core.get_system_status()
            
            print(f"🧠 Understanding Core:")
            print(f"   Success Rate: {understanding_status['success_rate']:.1%}")
            print(f"   Requests: {understanding_status['total_understanding_requests']}")
            print(f"   Threshold: {understanding_status['understanding_threshold']}")
            
            print(f"🧠 Consciousness Core:")
            print(f"   Detection Rate: {consciousness_status['detection_rate']:.1%}")
            print(f"   Detections: {consciousness_status['total_detections']}")
            print(f"   Threshold: {consciousness_status['consciousness_threshold']}")
            
            print(f"🧠 Meta Insight Core:")
            print(f"   Success Rate: {insight_status['success_rate']:.1%}")
            print(f"   Requests: {insight_status['total_insight_requests']}")
            print(f"   Threshold: {insight_status['insight_threshold']}")
            
        except Exception as e:
            print(f"Status report error: {e}")
        
        return status, passed, total
        
    except Exception as e:
        print(f"❌ Phase 3 test suite failed: {e}")
        return "FAILED", 0, total


async def main():
    """Run Phase 3 enhanced capabilities test"""
    status, passed, total = await test_phase3_enhanced_capabilities()
    
    print(f"\n🎯 FINAL PHASE 3 STATUS: {status}")
    print(f"   Enhanced capabilities implementation: {passed}/{total} components working")
    
    if status in ["FULLY_OPERATIONAL", "WORKING_WELL"]:
        print("🚀 Ready for Phase 4: Complete Integration!")
    else:
        print("🔧 Review enhanced capabilities before proceeding to Phase 4")


if __name__ == "__main__":
    asyncio.run(main())