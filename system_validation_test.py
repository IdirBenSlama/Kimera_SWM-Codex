#!/usr/bin/env python3
"""
Kimera SWM System Validation Test
=================================

Comprehensive validation script to verify all implemented phases are working correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.append('backend')

def test_phase_1_thermodynamics():
    """Test Phase 1: Thermodynamic Engine"""
    print("Testing Phase 1: Thermodynamic Engine...")
    
    try:
        from engines.thermodynamic_engine import ThermodynamicEngine
        
        engine = ThermodynamicEngine()
        test_vectors = [np.random.randn(3) for _ in range(5)]
        
        # Test semantic temperature
        temperature = engine.calculate_semantic_temperature(test_vectors)
        assert temperature >= 0, "Temperature must be non-negative"
        
        # Test Carnot engine
        carnot_result = engine.run_semantic_carnot_engine(test_vectors[:3], test_vectors[3:])
        assert 'efficiency' in carnot_result, "Carnot result must include efficiency"
        assert 0 <= carnot_result['efficiency'] <= 1, "Efficiency must be in [0,1]"
        
        print(f"  ✓ Semantic Temperature: {temperature:.3f}")
        print(f"  ✓ Carnot Efficiency: {carnot_result['efficiency']:.3f}")
        return True
        
    except Exception as e:
        print(f"  ✗ Phase 1 FAILED: {e}")
        return False

def test_phase_2_trading():
    """Test Phase 2: Trading Engine"""
    print("Testing Phase 2: Trading Engine...")
    
    try:
        from trading.engine import TradingEngine
        from trading.connectors.coinbase import CoinbaseConnector
        from trading.models import Order
        
        # Initialize components
        connector = CoinbaseConnector()
        engine = TradingEngine(connector)
        
        # Test order creation
        test_order = Order(
            symbol="BTC-USD",
            side="BUY",
            quantity=0.1,
            price=50000.0
        )
        
        print(f"  ✓ Trading Engine initialized")
        print(f"  ✓ Order model: {test_order.symbol} {test_order.side}")
        print(f"  ✓ Portfolio: ${engine.portfolio.cash:,.2f} initial cash")
        return True
        
    except Exception as e:
        print(f"  ✗ Phase 2 FAILED: {e}")
        return False

def test_phase_3a_pharmaceutical():
    """Test Phase 3A: Pharmaceutical Quality Control"""
    print("Testing Phase 3A: Pharmaceutical Quality Control...")
    
    try:
        from backend.core.quality_control import QualityControlSystem
        
        qc = QualityControlSystem()
        
        # Test control limits establishment
        test_data = [100, 102, 98, 101, 99, 103, 97, 100, 104, 96]
        limits = qc.establish_control_limits("test_attribute", test_data)
        
        assert 'ucl' in limits, "Control limits must include UCL"
        assert 'lcl' in limits, "Control limits must include LCL"
        assert limits['ucl'] > limits['center_line'] > limits['lcl'], "UCL > Center > LCL"
        
        # Test quality monitoring
        result = qc.monitor_quality_point("test_attribute", 100.5)
        assert 'status' in result, "Monitoring result must include status"
        
        # Test process capability
        spec_limits = {"usl": 115, "lsl": 85}
        capability = qc.analyze_process_capability("test_attribute", spec_limits)
        assert 'cp' in capability, "Capability analysis must include Cp"
        assert capability['cp'] > 0, "Cp must be positive"
        
        print(f"  ✓ Control Limits: UCL={limits['ucl']:.2f}, LCL={limits['lcl']:.2f}")
        print(f"  ✓ Quality Monitoring: {result['status']}")
        print(f"  ✓ Process Capability: Cp={capability['cp']:.3f}")
        return True
        
    except Exception as e:
        print(f"  ✗ Phase 3A FAILED: {e}")
        return False

def test_phase_3b_quantum():
    """Test Phase 3B: Quantum Thermodynamics"""
    print("Testing Phase 3B: Quantum Thermodynamics...")
    
    try:
        from specialized.quantum_thermodynamics import QuantumThermodynamicEngine
        
        quantum = QuantumThermodynamicEngine()
        
        # Test quantum coherence
        test_states = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        coherence = quantum.calculate_quantum_coherence(test_states)
        assert 0 <= coherence <= 1, "Coherence must be in [0,1]"
        
        # Test quantum annealing
        def simple_cost(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2
        
        annealing_result = quantum.simulate_quantum_annealing(
            simple_cost, np.array([10.0, 10.0]), num_steps=20
        )
        assert 'best_cost' in annealing_result, "Annealing must return best_cost"
        assert annealing_result['improvement'] >= 0, "Annealing should improve or maintain cost"
        
        # Test entanglement entropy
        bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        entropy = quantum.calculate_entanglement_entropy(bell_state)
        assert entropy >= 0, "Entropy must be non-negative"
        
        print(f"  ✓ Quantum Coherence: {coherence:.3f}")
        print(f"  ✓ Annealing Improvement: {annealing_result['improvement']:.3f}")
        print(f"  ✓ Entanglement Entropy: {entropy:.3f}")
        return True
        
    except Exception as e:
        print(f"  ✗ Phase 3B FAILED: {e}")
        return False

def test_integration():
    """Test integration between phases"""
    print("Testing Phase Integration...")
    
    try:
        from engines.thermodynamic_engine import ThermodynamicEngine
        from specialized.quantum_thermodynamics import QuantumThermodynamicEngine
        from backend.core.quality_control import QualityControlSystem
        
        # Test thermodynamic integration in pharmaceutical QC
        qc = QualityControlSystem()
        test_data = [100, 101, 99, 102, 98]
        qc.establish_control_limits("integration_test", test_data)
        
        # Add enough points to trigger thermodynamic calculation
        for i in range(10):
            result = qc.monitor_quality_point("integration_test", 100 + np.random.normal(0, 1))
        
        assert 'process_entropy' in result, "Should include thermodynamic entropy"
        assert result['process_entropy'] >= 0, "Process entropy should be non-negative"
        
        # Test quantum-classical bridge
        quantum = QuantumThermodynamicEngine()
        test_field = [np.random.randn(3) for _ in range(5)]
        enhanced_result = quantum.quantum_enhanced_temperature(test_field)
        
        assert 'classical_temperature' in enhanced_result, "Should include classical temperature"
        assert 'quantum_coherence' in enhanced_result, "Should include quantum coherence"
        
        print(f"  ✓ Pharmaceutical-Thermodynamic integration working")
        print(f"  ✓ Quantum-Classical bridge working")
        print(f"  ✓ Process entropy: {result['process_entropy']:.3f}")
        return True
        
    except Exception as e:
        print(f"  ✗ Integration FAILED: {e}")
        return False

def main():
    """Run comprehensive system validation"""
    print("=" * 60)
    print("KIMERA SWM SYSTEM VALIDATION")
    print("=" * 60)
    
    results = []
    
    # Test all phases
    results.append(("Phase 1 (Thermodynamics)", test_phase_1_thermodynamics()))
    results.append(("Phase 2 (Trading)", test_phase_2_trading()))
    results.append(("Phase 3A (Pharmaceutical)", test_phase_3a_pharmaceutical()))
    results.append(("Phase 3B (Quantum)", test_phase_3b_quantum()))
    results.append(("Integration", test_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for phase_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{phase_name:25} : {status}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} phases passed")
    
    if passed == total:
        print("\nSYSTEM STATUS: FULLY OPERATIONAL")
        print("All phases restored and validated successfully!")
        return True
    else:
        print(f"\nSYSTEM STATUS: PARTIAL FAILURE")
        print(f"{total - passed} phase(s) require attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 