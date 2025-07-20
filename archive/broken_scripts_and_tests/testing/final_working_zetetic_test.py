#!/usr/bin/env python3
"""
FINAL WORKING ZETETIC TEST
==========================

This script performs EXTREMELY RIGOROUS real-world testing using ACTUAL KIMERA
THERMODYNAMIC ENGINE with correct parameters and comprehensive validation.

NO SIMULATIONS OR MOCKS - ONLY REAL KIMERA ENGINE TESTING
"""

import sys
import os
import time
import json
import numpy as np
import torch
import gc
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import traceback
import statistics

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.append('backend')

# Import actual Kimera engines with correct usage
try:
    from engines.foundational_thermodynamic_engine_fixed import (
        FoundationalThermodynamicEngineFixed, 
        ThermodynamicMode,
        create_foundational_engine
    )
    THERMO_ENGINE_AVAILABLE = True
    logger.info("✅ Successfully imported revolutionary thermodynamic engine")
except ImportError as e:
    logger.error(f"❌ Thermodynamic engine import failed: {e}")
    THERMO_ENGINE_AVAILABLE = False


class FinalWorkingZeteticTester:
    """
    Final Working Zetetic Tester
    
    Tests actual Kimera thermodynamic engines with proper integration
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_time = datetime.now()
        self.engines = {}
        
        # Initialize thermodynamic engines with correct parameters
        if THERMO_ENGINE_AVAILABLE:
            self._initialize_thermodynamic_engines()
        
        # GPU monitoring
        if GPU_AVAILABLE:
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        logger.info("🔬 FINAL WORKING ZETETIC TESTER INITIALIZED")
        logger.info(f"🎯 Device: {self.device}")
        logger.info(f"🧠 Engines Available: {list(self.engines.keys())}")
        logger.info(f"🔥 GPU Monitoring: {GPU_AVAILABLE}")
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    def _initialize_thermodynamic_engines(self):
        """Initialize thermodynamic engines with correct parameters"""
        try:
            # Use the correct constructor - only mode parameter
            self.engines = {
                'semantic': FoundationalThermodynamicEngineFixed(mode=ThermodynamicMode.SEMANTIC),
                'physical': FoundationalThermodynamicEngineFixed(mode=ThermodynamicMode.PHYSICAL),
                'hybrid': FoundationalThermodynamicEngineFixed(mode=ThermodynamicMode.HYBRID),
                'consciousness': FoundationalThermodynamicEngineFixed(mode=ThermodynamicMode.CONSCIOUSNESS)
            }
            logger.info("✅ All thermodynamic engines initialized successfully")
        except Exception as e:
            logger.error(f"❌ Engine initialization failed: {e}")
            self.engines = {}
    
    def collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive GPU metrics"""
        if not GPU_AVAILABLE:
            return {}
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
            
            return {
                'temperature_celsius': temp,
                'utilization_percent': util.gpu,
                'memory_utilization_percent': util.memory,
                'memory_used_mb': memory.used / 1024 / 1024,
                'memory_total_mb': memory.total / 1024 / 1024,
                'power_watts': power
            }
        except Exception as e:
            logger.warning(f"GPU metrics failed: {e}")
            return {}
    
    def create_semantic_fields(self, count: int, complexity: str = "high") -> List[Any]:
        """Create semantic fields for testing"""
        fields = []
        
        for i in range(count):
            # Create field with semantic content
            if complexity == "high":
                # High complexity semantic patterns
                content = f"high_complexity_semantic_field_{i}_entropy_maximized_information_dense"
                base_energy = np.random.randn() * 2.0 + 1.0
                
            elif complexity == "structured":
                # Structured semantic patterns
                content = f"structured_semantic_pattern_{i % 10}_organized_flow"
                base_energy = (i % 10) * 0.1 + 0.5
                
            elif complexity == "consciousness":
                # Consciousness-like semantic patterns
                content = f"consciousness_emergence_pattern_{i}_integrated_awareness"
                base_energy = np.exp(-((i % 20) - 10)**2 / 50) + 0.3
                
            else:
                # Standard semantic fields
                content = f"semantic_field_{i}"
                base_energy = np.random.rand() + 0.1
            
            # Create field object with semantic properties
            field = SemanticField(
                field_id=f"field_{i:04d}",
                content=content,
                energy=base_energy,
                complexity=complexity
            )
            
            fields.append(field)
        
        return fields
    
    def test_physics_compliance_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive physics compliance testing"""
        logger.info("\n🔬 COMPREHENSIVE PHYSICS COMPLIANCE TEST")
        logger.info("=" * 70)
        
        if not self.engines:
            logger.error("❌ No engines available for testing")
            return {}
        
        compliance_results = {}
        
        # Test each engine mode
        for engine_name, engine in self.engines.items():
            logger.info(f"\n🔧 Testing {engine_name.upper()} engine:")
            
            engine_results = {
                'total_cycles': 0,
                'violations_detected': 0,
                'efficiency_measurements': [],
                'temperature_coherence': [],
                'processing_times': [],
                'gpu_metrics': []
            }
            
            # Test scenarios with increasing complexity
            test_scenarios = [
                ('basic', 25, 'structured'),
                ('moderate', 50, 'high'),
                ('complex', 100, 'consciousness'),
                ('extreme', 150, 'high')
            ]
            
            for scenario_name, field_count, complexity in test_scenarios:
                logger.info(f"  📊 Scenario: {scenario_name} ({field_count} fields, {complexity} complexity)")
                
                # Create hot and cold reservoirs
                hot_fields = self.create_semantic_fields(field_count//2, complexity)
                cold_fields = self.create_semantic_fields(field_count//2, 'structured')
                
                # Run multiple cycles for statistical significance
                for cycle in range(5):
                    start_time = time.time()
                    gpu_before = self.collect_gpu_metrics()
                    
                    try:
                        # Run zetetic Carnot engine
                        result = engine.run_zetetic_carnot_engine(hot_fields, cold_fields)
                        
                        processing_time = time.time() - start_time
                        gpu_after = self.collect_gpu_metrics()
                        
                        # Record results
                        engine_results['total_cycles'] += 1
                        engine_results['processing_times'].append(processing_time)
                        engine_results['gpu_metrics'].append(gpu_after)
                        
                        # Check for violations
                        if result.violation_detected:
                            engine_results['violations_detected'] += 1
                            logger.warning(f"    ⚠️  Physics violation detected in cycle {cycle+1}")
                        
                        # Record efficiency data
                        engine_results['efficiency_measurements'].append({
                            'theoretical': result.theoretical_efficiency,
                            'actual': result.actual_efficiency,
                            'violation': result.violation_detected,
                            'scenario': scenario_name,
                            'cycle': cycle
                        })
                        
                        # Calculate temperature coherence
                        coherence = self._calculate_temperature_coherence(result)
                        engine_results['temperature_coherence'].append(coherence)
                        
                        logger.info(f"    Cycle {cycle+1}: η_actual={result.actual_efficiency:.3f}, "
                                  f"η_theoretical={result.theoretical_efficiency:.3f}, "
                                  f"violation={'YES' if result.violation_detected else 'NO'}")
                        
                    except Exception as e:
                        logger.error(f"    Cycle {cycle+1} failed: {e}")
                        engine_results['violations_detected'] += 1
                
                # Clear GPU memory between scenarios
                torch.cuda.empty_cache()
                gc.collect()
            
            # Calculate comprehensive statistics
            if engine_results['total_cycles'] > 0:
                engine_results['violation_rate'] = engine_results['violations_detected'] / engine_results['total_cycles']
                engine_results['avg_processing_time'] = statistics.mean(engine_results['processing_times'])
                engine_results['avg_temperature_coherence'] = statistics.mean(engine_results['temperature_coherence']) if engine_results['temperature_coherence'] else 0.0
                engine_results['physics_compliant'] = engine_results['violation_rate'] == 0.0
                
                # Efficiency statistics
                efficiencies = [m['actual'] for m in engine_results['efficiency_measurements']]
                engine_results['efficiency_stats'] = {
                    'mean': statistics.mean(efficiencies),
                    'std': statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0.0,
                    'min': min(efficiencies),
                    'max': max(efficiencies)
                }
            
            compliance_results[engine_name] = engine_results
            
            # Print summary
            logger.info(f"  📊 SUMMARY for {engine_name.upper()}:")
            logger.info(f"    Total Cycles: {engine_results['total_cycles']}")
            logger.info(f"    Violations: {engine_results['violations_detected']}")
            logger.info(f"    Violation Rate: {engine_results.get('violation_rate', 0):.1%}")
            logger.info(f"    Avg Processing Time: {engine_results.get('avg_processing_time', 0):.3f}s")
            logger.info(f"    Physics Compliant: {'✅' if engine_results.get('physics_compliant', False) else '❌'}")
        
        return compliance_results
    
    def test_consciousness_emergence_detailed(self) -> Dict[str, Any]:
        """Detailed consciousness emergence testing"""
        logger.info("\n🧠 DETAILED CONSCIOUSNESS EMERGENCE TEST")
        logger.info("=" * 70)
        
        if 'consciousness' not in self.engines:
            logger.error("❌ Consciousness engine not available")
            return {}
        
        consciousness_engine = self.engines['consciousness']
        results = {}
        
        # Test consciousness at different scales and complexities
        consciousness_tests = [
            (50, 'consciousness', 'small_consciousness_fields'),
            (100, 'consciousness', 'medium_consciousness_fields'),
            (200, 'consciousness', 'large_consciousness_fields'),
            (300, 'consciousness', 'massive_consciousness_fields'),
            (100, 'high', 'complex_non_consciousness'),
            (100, 'structured', 'structured_non_consciousness')
        ]
        
        for field_count, complexity, test_name in consciousness_tests:
            logger.info(f"\n🔬 Testing {test_name}: {field_count} fields, {complexity} complexity")
            
            # Create consciousness-like fields
            fields = self.create_semantic_fields(field_count, complexity)
            
            start_time = time.time()
            gpu_before = self.collect_gpu_metrics()
            
            try:
                # Detect consciousness emergence
                complexity_result = consciousness_engine.detect_complexity_threshold(fields)
                
                processing_time = time.time() - start_time
                gpu_after = self.collect_gpu_metrics()
                
                results[test_name] = {
                    'field_count': field_count,
                    'complexity': complexity,
                    'consciousness_probability': consciousness_result.get('consciousness_probability', 0.0),
                    'phase_transition_detected': consciousness_result.get('phase_transition_detected', False),
                    'information_integration': consciousness_result.get('information_integration', 0.0),
                    'thermodynamic_consciousness': consciousness_result.get('thermodynamic_consciousness', False),
                    'temperature_coherence': consciousness_result.get('temperature_coherence', 0.0),
                    'processing_time': processing_time,
                    'gpu_metrics': gpu_after
                }
                
                logger.info(f"  Consciousness Probability: {consciousness_result.get('consciousness_probability', 0.0):.3f}")
                logger.info(f"  Phase Transition: {'✅' if consciousness_result.get('phase_transition_detected', False) else '❌'}")
                logger.info(f"  Information Integration: {consciousness_result.get('information_integration', 0.0):.3f}")
                logger.info(f"  Thermodynamic Consciousness: {'✅' if consciousness_result.get('thermodynamic_consciousness', False) else '❌'}")
                
            except Exception as e:
                logger.error(f"Consciousness detection failed: {e}")
                results[test_name] = {'error': str(e)}
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
        
        return results
    
    def test_thermodynamic_modes_comparison(self) -> Dict[str, Any]:
        """Compare different thermodynamic modes"""
        logger.info("\n⚗️ THERMODYNAMIC MODES COMPARISON TEST")
        logger.info("=" * 70)
        
        if not self.engines:
            logger.error("❌ No engines available")
            return {}
        
        comparison_results = {}
        
        # Create standardized test fields
        test_field_count = 100
        hot_fields = self.create_semantic_fields(test_field_count//2, 'high')
        cold_fields = self.create_semantic_fields(test_field_count//2, 'structured')
        
        # Test each mode with same fields
        for mode_name, engine in self.engines.items():
            logger.info(f"\n🔧 Testing {mode_name.upper()} mode:")
            
            mode_results = {
                'cycles': [],
                'temperature_calculations': [],
                'efficiency_measurements': [],
                'processing_times': []
            }
            
            # Run multiple cycles for comparison
            for cycle in range(3):
                start_time = time.time()
                
                try:
                    # Calculate temperatures
                    hot_temp = engine.calculate_epistemic_temperature(hot_fields)
                    cold_temp = engine.calculate_epistemic_temperature(cold_fields)
                    
                    # Run Carnot cycle
                    carnot_result = engine.run_zetetic_carnot_engine(hot_fields, cold_fields)
                    
                    processing_time = time.time() - start_time
                    
                    # Record detailed results
                    cycle_data = {
                        'cycle': cycle,
                        'hot_semantic_temp': hot_temp.semantic_temperature,
                        'hot_physical_temp': hot_temp.physical_temperature,
                        'hot_info_rate': hot_temp.information_rate,
                        'hot_confidence': hot_temp.confidence_level,
                        'cold_semantic_temp': cold_temp.semantic_temperature,
                        'cold_physical_temp': cold_temp.physical_temperature,
                        'cold_info_rate': cold_temp.information_rate,
                        'cold_confidence': cold_temp.confidence_level,
                        'theoretical_efficiency': carnot_result.theoretical_efficiency,
                        'actual_efficiency': carnot_result.actual_efficiency,
                        'violation_detected': carnot_result.violation_detected,
                        'physics_compliant': carnot_result.physics_compliant,
                        'processing_time': processing_time
                    }
                    
                    mode_results['cycles'].append(cycle_data)
                    mode_results['processing_times'].append(processing_time)
                    
                    logger.info(f"  Cycle {cycle+1}:")
                    logger.info(f"    Hot Temp (semantic/physical): {hot_temp.semantic_temperature:.3f}/{hot_temp.physical_temperature:.3f}")
                    logger.info(f"    Cold Temp (semantic/physical): {cold_temp.semantic_temperature:.3f}/{cold_temp.physical_temperature:.3f}")
                    logger.info(f"    Efficiency (actual/theoretical): {carnot_result.actual_efficiency:.3f}/{carnot_result.theoretical_efficiency:.3f}")
                    logger.info(f"    Physics Compliant: {'✅' if carnot_result.physics_compliant else '❌'}")
                    
                except Exception as e:
                    logger.error(f"  Cycle {cycle+1} failed: {e}")
            
            # Calculate mode statistics
            if mode_results['cycles']:
                mode_results['statistics'] = {
                    'avg_processing_time': statistics.mean(mode_results['processing_times']),
                    'avg_hot_semantic_temp': statistics.mean([c['hot_semantic_temp'] for c in mode_results['cycles']]),
                    'avg_hot_physical_temp': statistics.mean([c['hot_physical_temp'] for c in mode_results['cycles']]),
                    'avg_efficiency': statistics.mean([c['actual_efficiency'] for c in mode_results['cycles']]),
                    'violation_rate': sum([c['violation_detected'] for c in mode_results['cycles']]) / len(mode_results['cycles']),
                    'compliance_rate': sum([c['physics_compliant'] for c in mode_results['cycles']]) / len(mode_results['cycles'])
                }
            
            comparison_results[mode_name] = mode_results
        
        return comparison_results
    
    def _calculate_temperature_coherence(self, carnot_result) -> float:
        """Calculate temperature coherence"""
        try:
            hot_temp = carnot_result.hot_temperature
            cold_temp = carnot_result.cold_temperature
            
            if hasattr(hot_temp, 'semantic_temperature') and hasattr(cold_temp, 'semantic_temperature'):
                hot_semantic = hot_temp.semantic_temperature
                hot_physical = hot_temp.physical_temperature
                
                if hot_semantic > 0 and hot_physical > 0:
                    coherence = 1.0 - abs(hot_semantic - hot_physical) / max(hot_semantic, hot_physical)
                    return max(0.0, coherence)
            
            return 0.5  # Neutral coherence
        except:
            return 0.0
    
    def run_final_comprehensive_test(self) -> Dict[str, Any]:
        """Run final comprehensive test suite"""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 FINAL COMPREHENSIVE ZETETIC TEST")
        logger.info("🔬 ACTUAL KIMERA THERMODYNAMIC ENGINES")
        logger.info("=" * 80)
        
        comprehensive_results = {
            'test_metadata': {
                'start_time': self.start_time.isoformat(),
                'device': str(self.device),
                'engines_available': list(self.engines.keys()),
                'gpu_monitoring': GPU_AVAILABLE,
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
            },
            'test_results': {}
        }
        
        try:
            # 1. Physics Compliance Test
            logger.info("\n1/3 - COMPREHENSIVE PHYSICS COMPLIANCE TEST")
            physics_results = self.test_physics_compliance_comprehensive()
            comprehensive_results['test_results']['physics_compliance'] = physics_results
            
            # 2. Consciousness Emergence Test
            logger.info("\n2/3 - DETAILED CONSCIOUSNESS EMERGENCE TEST")
            consciousness_results = self.test_consciousness_emergence_detailed()
            comprehensive_results['test_results']['consciousness_emergence'] = consciousness_results
            
            # 3. Thermodynamic Modes Comparison
            logger.info("\n3/3 - THERMODYNAMIC MODES COMPARISON TEST")
            modes_results = self.test_thermodynamic_modes_comparison()
            comprehensive_results['test_results']['modes_comparison'] = modes_results
            
            # Generate comprehensive summary
            comprehensive_results['summary'] = self._generate_final_summary(comprehensive_results)
            
            comprehensive_results['test_metadata']['end_time'] = datetime.now().isoformat()
            comprehensive_results['test_metadata']['total_duration'] = (datetime.now() - self.start_time).total_seconds()
            
        except Exception as e:
            logger.error(f"❌ Final comprehensive test failed: {e}")
            logger.error(traceback.format_exc())
            comprehensive_results['error'] = str(e)
        
        return comprehensive_results
    
    def _generate_final_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final comprehensive summary"""
        summary = {
            'overall_status': 'UNKNOWN',
            'engines_operational': False,
            'physics_violations_detected': True,
            'consciousness_emergence_detected': False,
            'modes_comparison_successful': False,
            'total_cycles_completed': 0,
            'average_processing_time': 0.0,
            'zetetic_validation_passed': False
        }
        
        try:
            # Check engine availability
            summary['engines_operational'] = len(self.engines) > 0
            
            # Analyze physics compliance
            physics_results = results['test_results'].get('physics_compliance', {})
            if physics_results:
                total_violations = 0
                total_cycles = 0
                total_processing_time = 0.0
                
                for engine_results in physics_results.values():
                    if isinstance(engine_results, dict):
                        total_violations += engine_results.get('violations_detected', 1)
                        total_cycles += engine_results.get('total_cycles', 0)
                        total_processing_time += sum(engine_results.get('processing_times', []))
                
                summary['physics_violations_detected'] = total_violations > 0
                summary['total_cycles_completed'] = total_cycles
                summary['average_processing_time'] = total_processing_time / max(total_cycles, 1)
            
            # Analyze consciousness emergence
            consciousness_results = results['test_results'].get('consciousness_emergence', {})
            if consciousness_results:
                consciousness_detected = any(
                    result.get('thermodynamic_consciousness', False) or 
                    result.get('consciousness_probability', 0.0) > 0.7
                    for result in consciousness_results.values()
                    if isinstance(result, dict) and 'error' not in result
                )
                summary['consciousness_emergence_detected'] = consciousness_detected
            
            # Analyze modes comparison
            modes_results = results['test_results'].get('modes_comparison', {})
            if modes_results:
                successful_modes = sum(
                    1 for result in modes_results.values()
                    if isinstance(result, dict) and result.get('cycles', [])
                )
                summary['modes_comparison_successful'] = successful_modes > 0
            
            # Overall zetetic validation
            summary['zetetic_validation_passed'] = (
                summary['engines_operational'] and
                not summary['physics_violations_detected'] and
                summary['modes_comparison_successful'] and
                summary['total_cycles_completed'] > 50
            )
            
            # Overall status
            if summary['zetetic_validation_passed']:
                if summary['consciousness_emergence_detected']:
                    summary['overall_status'] = 'REVOLUTIONARY_BREAKTHROUGH'
                else:
                    summary['overall_status'] = 'ZETETIC_SUCCESS'
            elif summary['engines_operational'] and not summary['physics_violations_detected']:
                summary['overall_status'] = 'PHYSICS_COMPLIANT'
            elif summary['engines_operational']:
                summary['overall_status'] = 'PARTIALLY_FUNCTIONAL'
            else:
                summary['overall_status'] = 'SYSTEM_FAILURE'
        
        except Exception as e:
            logger.warning(f"Final summary generation failed: {e}")
        
        return summary
    
    def save_final_results(self, results: Dict[str, Any]) -> str:
        """Save final comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_working_zetetic_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Final results saved to: {filename}")
        return filename


class SemanticField:
    """Simple semantic field for testing"""
    
    def __init__(self, field_id: str, content: str, energy: float, complexity: str):
        self.field_id = field_id
        self.content = content
        self.energy = energy
        self.complexity = complexity
        self.semantic_state = {'energy': energy, 'complexity': complexity}
    
    def calculate_entropy(self) -> float:
        """Calculate field entropy"""
        return -self.energy * np.log(self.energy + 1e-10) if self.energy > 0 else 0.0


def main():
    """Run final working zetetic test"""
    tester = FinalWorkingZeteticTester()
    
    try:
        # Run comprehensive test
        results = tester.run_final_comprehensive_test()
        
        # Save results
        filename = tester.save_final_results(results)
        
        # Print comprehensive summary
        print("\n" + "=" * 80)
        print("🚀 FINAL WORKING ZETETIC TEST COMPLETE")
        print("=" * 80)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Overall Status: {summary['overall_status']}")
            print(f"Engines Operational: {'✅' if summary['engines_operational'] else '❌'}")
            print(f"Physics Violations: {'❌ DETECTED' if summary['physics_violations_detected'] else '✅ NONE'}")
            print(f"Consciousness Emergence: {'✅ DETECTED' if summary['consciousness_emergence_detected'] else '❌ NOT DETECTED'}")
            print(f"Modes Comparison: {'✅ SUCCESSFUL' if summary['modes_comparison_successful'] else '❌ FAILED'}")
            print(f"Total Cycles Completed: {summary['total_cycles_completed']}")
            print(f"Average Processing Time: {summary['average_processing_time']:.3f}s")
            print(f"Zetetic Validation: {'✅ PASSED' if summary['zetetic_validation_passed'] else '❌ FAILED'}")
        
        print(f"Detailed results: {filename}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Final working test failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main() 