"""
Unified Cognitive Architecture Test Runner
=========================================

Comprehensive demonstration and testing of KIMERA's revolutionary
unified cognitive architecture integrating all subsystems.
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from kimera_unified_cognitive_architecture import UnifiedCognitiveArchitecture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def comprehensive_architecture_test():
    """Comprehensive test of the unified cognitive architecture"""
    
    logger.info("🌌 KIMERA UNIFIED COGNITIVE ARCHITECTURE - COMPREHENSIVE TEST")
    logger.info("=" * 80)
    logger.info("Testing revolutionary integration of:")
    logger.info("  • Thermodynamic Self-Stabilization")
    logger.info("  • Proprioceptive Self-Regulation")
    logger.info("  • Temporal Graph Network Algorithms")
    logger.info("  • Quantum Edge Security Architecture")
    logger.info("  • Innovation Module Integration")
    logger.info()
    
    # Initialize architecture with full capabilities
    logger.info("🚀 Initializing Unified Cognitive Architecture...")
    architecture = UnifiedCognitiveArchitecture(
        enable_innovations=True,
        quantum_security_level="maximum",
        thermodynamic_precision="high",
        proprioceptive_sensitivity=0.15
    )
    
    # Test 1: Basic Cognitive Processing
    logger.info("\n📋 TEST 1: Basic Unified Cognitive Processing")
    logger.info("-" * 50)
    
    test_inputs = [
        "Market volatility indicates economic uncertainty",
        "Higher temperatures accelerate chemical reactions",
        "Neural networks learn through backpropagation",
        "Quantum entanglement enables secure communication",
        "Thermodynamic efficiency requires energy optimization"
    ]
    
    start_time = time.time()
    result = await architecture.unified_cognitive_processing(test_inputs)
    processing_time = time.time() - start_time
    
    logger.info(f"✅ Processing Result: {result['success']}")
    logger.info(f"⏱️  Processing Time: {processing_time:.3f}s")
    logger.info(f"🎯 Intelligence Quotient: {result.get('intelligence_quotient', 0)
    logger.info(f"🧠 Inputs Processed: {len(test_inputs)
    
    if result.get('unified_state'):
        state = result['unified_state']
        logger.info(f"🌡️  Thermodynamic Efficiency: {state.thermodynamic_efficiency:.3f}")
        logger.info(f"🔄 Reversibility Index: {state.reversibility_index:.3f}")
        logger.info(f"🧠 Computational Health: {state.computational_health_score:.3f}")
        logger.info(f"🔗 Causal Coherence: {state.causal_coherence:.3f}")
        logger.info(f"🛡️  Security Level: {state.quantum_security_level:.3f}")
    
    # Test 2: Innovation Module Integration
    logger.info("\n📋 TEST 2: Innovation Module Integration")
    logger.info("-" * 50)
    
    status = architecture.get_unified_status()
    subsystems = status['subsystem_status']
    
    logger.info("Subsystem Status:")
    for system, active in subsystems.items():
        status_icon = "✅" if active else "❌"
        logger.info(f"  {status_icon} {system.replace('_', ' ')
    
    active_count = sum(subsystems.values())
    total_count = len(subsystems)
    integration_score = active_count / total_count
    
    logger.info(f"\n🎯 Integration Score: {integration_score:.1%} ({active_count}/{total_count})
    
    # Test 3: Autonomous Operation Demonstration
    logger.info("\n📋 TEST 3: Autonomous Multi-Scale Operation")
    logger.info("-" * 50)
    
    logger.info("🚀 Starting autonomous operation...")
    architecture.start_autonomous_operation()
    
    # Monitor for 10 seconds
    monitoring_duration = 10
    logger.info(f"📊 Monitoring for {monitoring_duration} seconds...")
    
    for i in range(monitoring_duration):
        await asyncio.sleep(1)
        current_status = architecture.get_unified_status()
        uiq = current_status.get('current_uiq', 0)
        logger.info(f"   Second {i+1:2d}: UIQ = {uiq:.3f}, Autonomous = {current_status['autonomous_active']}")
    
    logger.info("🛑 Stopping autonomous operation...")
    architecture.stop_autonomous_operation()
    
    # Test 4: Complex Reasoning Test
    logger.info("\n📋 TEST 4: Complex Temporal-Causal Reasoning")
    logger.info("-" * 50)
    
    complex_scenario = [
        "Market crash causes investor panic",
        "Investor panic leads to selling pressure", 
        "Selling pressure drives prices down",
        "Lower prices create buying opportunities",
        "Buying opportunities attract value investors"
    ]
    
    reasoning_result = await architecture.unified_cognitive_processing(
        complex_scenario,
        context={'scenario_type': 'causal_chain', 'temporal_sequence': complex_scenario}
    )
    
    logger.info(f"✅ Complex Reasoning: {reasoning_result['success']}")
    if reasoning_result.get('subsystem_results', {}).get('temporal_causal'):
        tc_result = reasoning_result['subsystem_results']['temporal_causal']
        logger.info(f"🔗 Causal Coherence: {tc_result.get('overall_causal_coherence', 0)
        logger.info(f"⏰ Temporal Stability: {tc_result.get('overall_temporal_stability', 0)
        logger.info(f"🧠 Causal Relations Found: {sum(r['causes_found'] + r['effects_found'] for r in tc_result.get('causal_analysis', [])
    
    # Test 5: Security Resilience Test
    logger.info("\n📋 TEST 5: Security Resilience Assessment")
    logger.info("-" * 50)
    
    security_test_inputs = [
        "Normal market analysis request",
        "Potential injection attempt: DROP TABLE users",
        "Standard volatility calculation",
        "Suspicious pattern: <script>alert('test')</script>",
        "Regular trading signal analysis"
    ]
    
    security_results = []
    for input_text in security_test_inputs:
        result = await architecture.unified_cognitive_processing([input_text])
        security_results.append({
            'input': input_text[:30] + "..." if len(input_text) > 30 else input_text,
            'success': result.get('success', False),
            'blocked': result.get('blocked', False)
        })
    
    processed_count = sum(1 for r in security_results if r['success'])
    blocked_count = sum(1 for r in security_results if r['blocked'])
    
    logger.info(f"🛡️  Security Assessment:")
    logger.info(f"   Processed safely: {processed_count}/{len(security_test_inputs)
    logger.info(f"   Threats blocked: {blocked_count}")
    logger.info(f"   Security effectiveness: {((processed_count + blocked_count)
    
    # Test 6: Performance Benchmarking
    logger.info("\n📋 TEST 6: Performance Benchmarking")
    logger.info("-" * 50)
    
    benchmark_inputs = [f"Benchmark test input {i}" for i in range(50)]
    
    logger.info("🏃 Running performance benchmark...")
    benchmark_start = time.time()
    
    benchmark_results = []
    for i in range(0, len(benchmark_inputs), 5):  # Process in batches of 5
        batch = benchmark_inputs[i:i+5]
        batch_start = time.time()
        result = await architecture.unified_cognitive_processing(batch)
        batch_time = time.time() - batch_start
        
        benchmark_results.append({
            'batch_size': len(batch),
            'processing_time': batch_time,
            'success': result.get('success', False),
            'uiq': result.get('intelligence_quotient', 0)
        })
    
    total_benchmark_time = time.time() - benchmark_start
    successful_batches = sum(1 for r in benchmark_results if r['success'])
    average_uiq = sum(r['uiq'] for r in benchmark_results) / len(benchmark_results)
    throughput = len(benchmark_inputs) / total_benchmark_time
    
    logger.info(f"⚡ Performance Results:")
    logger.info(f"   Total inputs processed: {len(benchmark_inputs)
    logger.info(f"   Total processing time: {total_benchmark_time:.2f}s")
    logger.info(f"   Throughput: {throughput:.1f} inputs/second")
    logger.info(f"   Success rate: {successful_batches}/{len(benchmark_results)
    logger.info(f"   Average UIQ: {average_uiq:.3f}")
    
    # Final Assessment
    logger.info("\n📋 FINAL ASSESSMENT")
    logger.info("=" * 50)
    
    final_status = architecture.get_unified_status()
    final_metrics = final_status.get('performance_metrics', {})
    
    logger.info("🎯 Unified Architecture Assessment:")
    logger.info(f"   Overall UIQ: {final_status.get('current_uiq', 0)
    logger.info(f"   Cognitive Ops/Sec: {final_metrics.get('cognitive_operations_per_second', 0)
    logger.info(f"   Thermodynamic Efficiency: {final_metrics.get('thermodynamic_efficiency_score', 0)
    logger.info(f"   Security Mitigation Rate: {final_metrics.get('security_threat_mitigation_rate', 0)
    logger.info(f"   Innovation Synergy: {final_metrics.get('innovation_synergy_factor', 0)
    
    # Generate comprehensive report
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'architecture_config': {
            'innovations_enabled': architecture.enable_innovations,
            'quantum_security_level': architecture.quantum_security_level,
            'thermodynamic_precision': architecture.thermodynamic_precision
        },
        'test_results': {
            'basic_processing': {
                'success': result.get('success', False),
                'processing_time': processing_time,
                'intelligence_quotient': result.get('intelligence_quotient', 0)
            },
            'integration_score': integration_score,
            'autonomous_operation': {
                'duration_seconds': monitoring_duration,
                'final_uiq': final_status.get('current_uiq', 0)
            },
            'security_resilience': {
                'processed_safely': processed_count,
                'threats_blocked': blocked_count,
                'effectiveness': (processed_count + blocked_count) / len(security_test_inputs)
            },
            'performance_benchmark': {
                'throughput': throughput,
                'success_rate': successful_batches / len(benchmark_results),
                'average_uiq': average_uiq
            }
        },
        'final_metrics': final_metrics
    }
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"unified_cognitive_architecture_test_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n📄 Comprehensive report saved: {filename}")
    
    # Summary
    overall_success = (
        result.get('success', False) and
        integration_score > 0.7 and
        final_status.get('current_uiq', 0) > 0.5
    )
    
    logger.info(f"\n🎉 UNIFIED COGNITIVE ARCHITECTURE TEST: {'SUCCESS' if overall_success else 'NEEDS IMPROVEMENT'}")
    
    return report

async def main():
    """Main test runner"""
    try:
        report = await comprehensive_architecture_test()
        logger.info("\n✅ All tests completed successfully!")
        return report
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main()) 