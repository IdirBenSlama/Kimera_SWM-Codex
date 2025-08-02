#!/usr/bin/env python3
"""
KIMERA CRITICAL ENGINE INTEGRATION TEST
======================================

Tests the integration of critical Phase 1 engines into the Kimera core system:
- Understanding Engine (genuine intelligence)
- Human Interface (human-readable outputs)  
- Cognitive Security Orchestrator (data protection)
- Linguistic Intelligence Engine (language processing)

This test verifies that the engines are properly initialized and accessible
through the core system.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_critical_engine_integration():
    """Test that critical engines are properly integrated into the core system"""
    
    print("\n🧪 KIMERA CRITICAL ENGINE INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Initialize the core system
        print("\n🔧 Step 1: Initializing Kimera Core System...")
        from src.core.kimera_system import get_kimera_system
        
        kimera = get_kimera_system()
        kimera.initialize()  # This is synchronous, not async
        
        system_status = kimera.get_system_status()
        print(f"✅ Core system initialized - State: {system_status['state']}")
        
        # Test Understanding Engine
        print("\n🧠 Step 2: Testing Understanding Engine...")
        understanding_engine = kimera.get_understanding_engine()
        
        if understanding_engine is not None and understanding_engine != "initializing":
            print("✅ Understanding Engine successfully integrated")
            
            # Test understanding capability
            from src.engines.understanding_engine import UnderstandingContext
            context = UnderstandingContext(
                input_content="Test cognitive processing",
                modalities={"text": True},
                goals=["analyze", "understand"],
                current_state={"test": True}
            )
            
            understanding_result = await understanding_engine.understand_content(context)
            print(f"   - Understanding confidence: {understanding_result.confidence_score:.2f}")
            print(f"   - Insights generated: {len(understanding_result.insights_generated)}")
            
        elif understanding_engine == "initializing":
            print("⏳ Understanding Engine is initializing asynchronously")
        else:
            print("❌ Understanding Engine not available")
        
        # Test Human Interface
        print("\n👤 Step 3: Testing Human Interface...")
        human_interface = kimera.get_human_interface()
        
        if human_interface is not None:
            print("✅ Human Interface successfully integrated")
            
            # Test human-readable output
            from src.engines.human_interface import ResponseMode
            test_response = human_interface.format_response(
                "System operational",
                thinking_summary="Analyzing system status",
                confidence=0.95
            )
            print(f"   - Response mode: {test_response.mode}")
            print(f"   - Response confidence: {test_response.confidence}")
            print(f"   - Human-readable output: {test_response.content[:50]}...")
            
        else:
            print("❌ Human Interface not available")
        
        # Test Cognitive Security Orchestrator
        print("\n🔒 Step 4: Testing Cognitive Security Orchestrator...")
        security_orchestrator = kimera.get_cognitive_security_orchestrator()
        
        if security_orchestrator is not None:
            print("✅ Cognitive Security Orchestrator successfully integrated")
            
            # Test security metrics
            security_metrics = security_orchestrator.get_security_metrics()
            print(f"   - Security policy active: {security_metrics.get('policy_active', False)}")
            print(f"   - GPU cryptography available: {security_metrics.get('gpu_crypto_available', False)}")
            print(f"   - Homomorphic encryption: {security_metrics.get('homomorphic_ready', False)}")
            
        else:
            print("❌ Cognitive Security Orchestrator not available")
        
        # Test Linguistic Intelligence Engine  
        print("\n🗣️ Step 5: Testing Linguistic Intelligence Engine...")
        linguistic_engine = kimera.get_component("linguistic_intelligence_engine")
        
        if linguistic_engine is not None and linguistic_engine != "initializing":
            print("✅ Linguistic Intelligence Engine successfully integrated")
            
            # Test basic linguistic processing
            test_text = "Hello world, this is a test of linguistic processing."
            analysis = await linguistic_engine.analyze_text(test_text)
            print(f"   - Input length: {analysis.input_length}")
            print(f"   - Processing time: {analysis.processing_time_ms:.2f}ms")
            print(f"   - Language detected: {analysis.language_detected}")
            
        elif linguistic_engine == "initializing":
            print("⏳ Linguistic Intelligence Engine is initializing asynchronously")
        else:
            print("❌ Linguistic Intelligence Engine not available")
        
        # Overall system status
        print("\n📊 Step 6: Overall System Status...")
        final_status = kimera.get_system_status()
        
        critical_engines = [
            "understanding_engine_ready",
            "human_interface_ready", 
            "cognitive_security_orchestrator_ready"
        ]
        
        ready_engines = sum(1 for engine in critical_engines if final_status.get(engine, False))
        total_engines = len(critical_engines)
        
        print(f"Critical engines ready: {ready_engines}/{total_engines}")
        
        for engine in critical_engines:
            status = "✅" if final_status.get(engine, False) else "❌"
            engine_name = engine.replace("_ready", "").replace("_", " ").title()
            print(f"   {status} {engine_name}")
        
        # Success assessment
        if ready_engines >= 2:  # At least 2 critical engines working
            print(f"\n🎉 INTEGRATION TEST PASSED!")
            print(f"   {ready_engines}/{total_engines} critical engines successfully integrated")
            print(f"   Kimera now has enhanced cognitive capabilities!")
            return True
        else:
            print(f"\n⚠️ INTEGRATION TEST PARTIAL SUCCESS")
            print(f"   {ready_engines}/{total_engines} critical engines integrated")
            print(f"   Some engines may still be initializing")
            return False
            
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_critical_engine_integration()
    
    if success:
        print("\n✅ KIMERA CRITICAL ENGINE INTEGRATION SUCCESSFUL")
        sys.exit(0)
    else:
        print("\n❌ KIMERA CRITICAL ENGINE INTEGRATION NEEDS ATTENTION")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 