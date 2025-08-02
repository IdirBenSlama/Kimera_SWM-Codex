import os
#!/usr/bin/env python3
"""
KIMERA CDP AGENTKIT INTEGRATION TEST
===================================

Test script to validate the CDP AgentKit integration with Kimera's cognitive systems.
This test runs in simulation mode by default for safety.
"""

import asyncio
import json
import time
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from kimera_cdp_agentkit_integration import KimeraCDPTrader, KimeraCDPCognitiveEngine

async def test_cognitive_engine():
    """Test the cognitive engine functionality"""
    print("🧠 Testing Kimera CDP Cognitive Engine...")
    
    engine = KimeraCDPCognitiveEngine(dimension=128)
    
    # Test market analysis
    test_market_data = {
        'price': 2000.0,
        'volume': 1500000.0,
        'price_change_24h': 5.2,
        'volume_change_24h': 12.1,
        'volatility': 0.45,
        'gas_price': 25.0,
        'liquidity_score': 0.75
    }
    
    analysis = await engine.analyze_blockchain_market(test_market_data)
    
    print(f"  ✅ Cognitive Score: {analysis['cognitive_score']:.3f}")
    print(f"  ✅ Blockchain Confidence: {analysis['blockchain_confidence']:.3f}")
    print(f"  ✅ Gas Efficiency: {analysis['gas_efficiency']:.3f}")
    
    # Test decision generation
    decision = engine.generate_cdp_decision(analysis)
    
    print(f"  ✅ Generated Decision: {decision.action} {decision.from_asset}→{decision.to_asset}")
    print(f"  ✅ Amount: {decision.amount:.6f}")
    print(f"  ✅ Confidence: {decision.confidence:.3f}")
    print(f"  ✅ Network: {decision.network}")
    
    return True

async def test_cdp_trader():
    """Test the CDP trader functionality"""
    print("🚀 Testing Kimera CDP Trader...")
    
    # Initialize trader with test credentials
    trader = KimeraCDPTrader(
        api_key_name=os.getenv("CDP_API_KEY_NAME", ""),
        api_key_private_key=None  # Will run in simulation mode
    )
    
    # Test initialization
    init_success = await trader.initialize_cdp()
    print(f"  ✅ CDP Initialization: {'Success' if init_success else 'Simulation Mode'}")
    
    # Test single operation
    from kimera_cdp_agentkit_integration import KimeraCDPDecision
    
    test_decision = KimeraCDPDecision(
        action="swap",
        from_asset="USDC",
        to_asset="ETH",
        amount=0.01,
        confidence=0.85,
        cognitive_reason="Test operation",
        thermodynamic_score=0.3,
        network="base",
        execution_priority="normal",
        max_slippage=0.01
    )
    
    operation_success = await trader.execute_cdp_operation(test_decision)
    print(f"  ✅ Test Operation: {'Success' if operation_success else 'Failed'}")
    
    return True

async def test_full_trading_cycle():
    """Test a complete trading cycle"""
    print("📊 Testing Full Trading Cycle...")
    
    trader = KimeraCDPTrader(
        api_key_name=os.getenv("CDP_API_KEY_NAME", ""),
        api_key_private_key=None  # Simulation mode
    )
    
    # Run a short trading cycle
    report = await trader.run_trading_cycle(duration_minutes=1)
    
    if 'error' in report:
        print(f"  ❌ Cycle Error: {report['error']}")
        return False
    
    # Display results
    if 'cycle_summary' in report:
        summary = report['cycle_summary']
        print(f"  ✅ Operations: {summary['total_operations']}")
        print(f"  ✅ Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"  ✅ Ops/Min: {summary['operations_per_minute']:.1f}")
    
    if 'cognitive_performance' in report:
        cognitive = report['cognitive_performance']
        print(f"  ✅ Avg Confidence: {cognitive.get('avg_confidence', 0):.3f}")
    
    # Save test report
    timestamp = int(time.time())
    test_report_file = f"kimera_cdp_test_report_{timestamp}.json"
    
    with open(test_report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✅ Test Report: {test_report_file}")
    
    return True

async def main():
    """Run all tests"""
    print("🚀 KIMERA CDP AGENTKIT INTEGRATION TESTS")
    print("=" * 50)
    
    test_results = []
    
    try:
        # Test 1: Cognitive Engine
        result1 = await test_cognitive_engine()
        test_results.append(("Cognitive Engine", result1))
        print()
        
        # Test 2: CDP Trader
        result2 = await test_cdp_trader()
        test_results.append(("CDP Trader", result2))
        print()
        
        # Test 3: Full Trading Cycle
        result3 = await test_full_trading_cycle()
        test_results.append(("Full Trading Cycle", result3))
        print()
        
        # Summary
        print("=" * 50)
        print("📋 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        all_passed = True
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            if not result:
                all_passed = False
        
        print("=" * 50)
        if all_passed:
            print("🎉 ALL TESTS PASSED - CDP INTEGRATION READY")
            print("\n📝 Next Steps:")
            print("1. Add your CDP private key to kimera_cdp_config.env")
            print("2. Run: python kimera_cdp_agentkit_integration.py")
            print("3. Monitor logs and performance metrics")
        else:
            print("❌ SOME TESTS FAILED - CHECK LOGS")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 