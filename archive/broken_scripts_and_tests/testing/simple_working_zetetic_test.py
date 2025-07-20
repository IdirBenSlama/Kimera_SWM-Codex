#!/usr/bin/env python3
"""
SIMPLE WORKING ZETETIC TEST
===========================

This script performs rigorous real-world testing using ACTUAL KIMERA THERMODYNAMIC ENGINE
with the correct data format and comprehensive validation.

NO SIMULATIONS OR MOCKS - ONLY REAL KIMERA ENGINE TESTING
"""

import sys
import os
import time
import json
import numpy as np
import torch
import gc
from datetime import datetime
from typing import Dict, List, Any, Optional
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

# Import actual Kimera engines
try:
    from engines.foundational_thermodynamic_engine_fixed import (
        FoundationalThermodynamicEngineFixed, 
        ThermodynamicMode
    )
    THERMO_ENGINE_AVAILABLE = True
    logger.info("✅ Successfully imported revolutionary thermodynamic engine")
except ImportError as e:
    logger.error(f"❌ Thermodynamic engine import failed: {e}")
    THERMO_ENGINE_AVAILABLE = False


class SimpleField:
    """Simple field that matches engine expectations"""
    
    def __init__(self, field_id: str, energy: float, semantic_data: Dict[str, float]):
        self.field_id = field_id
        self.energy = energy
        self.semantic_state = semantic_data
        self.embedding = torch.tensor([energy], dtype=torch.float32)
    
    def calculate_entropy(self) -> float:
        """Calculate field entropy"""
        return -self.energy * np.log(self.energy + 1e-10) if self.energy > 0 else 0.0


class SimpleWorkingZeteticTester:
    """
    Simple Working Zetetic Tester
    
    Tests actual Kimera thermodynamic engines with correct data format
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_time = datetime.now()
        self.engines = {}
        
        # Initialize thermodynamic engines
        if THERMO_ENGINE_AVAILABLE:
            self._initialize_engines()
        
        # GPU monitoring
        if GPU_AVAILABLE:
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        logger.info("🔬 SIMPLE WORKING ZETETIC TESTER INITIALIZED")
        logger.info(f"🎯 Device: {self.device}")
        logger.info(f"🧠 Engines Available: {list(self.engines.keys())}")
        logger.info(f"🔥 GPU Monitoring: {GPU_AVAILABLE}")
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    def _initialize_engines(self):
        """Initialize thermodynamic engines"""
        try:
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
        """Collect GPU metrics"""
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
                'memory_used_mb': memory.used / 1024 / 1024,
                'power_watts': power
            }
        except Exception as e:
            logger.warning(f"GPU metrics failed: {e}")
            return {}
    
    def create_simple_fields(self, count: int, energy_pattern: str = "random") -> List[SimpleField]:
        """Create simple fields with correct format"""
        fields = []
        
        for i in range(count):
            if energy_pattern == "random":
                energy = np.random.rand() * 2.0 + 0.1  # 0.1 to 2.1
            elif energy_pattern == "high":
                energy = np.random.rand() * 2.0 + 1.5  # 1.5 to 3.5
            elif energy_pattern == "low":
                energy = np.random.rand() * 0.5 + 0.1  # 0.1 to 0.6
            elif energy_pattern == "structured":
                energy = (i % 10) * 0.2 + 0.5  # 0.5 to 2.3
            else:
                energy = 1.0
            
            semantic_data = {
                'energy': energy,
                'field_index': float(i),
                'pattern_type': hash(energy_pattern) % 1000 / 1000.0
            }
            
            field = SimpleField(
                field_id=f"field_{i:04d}",
                energy=energy,
                semantic_data=semantic_data
            )
            
            fields.append(field)
        
        return fields
    
    def test_basic_thermodynamic_operations(self) -> Dict[str, Any]:
        """Test basic thermodynamic operations"""
        logger.info("\n🔬 BASIC THERMODYNAMIC OPERATIONS TEST")
        logger.info("=" * 70)
        
        if not self.engines:
            logger.error("❌ No engines available")
            return {}
        
        results = {}
        
        # Test each engine with simple operations
        for engine_name, engine in self.engines.items():
            logger.info(f"\n🔧 Testing {engine_name.upper()} engine:")
            
            engine_results = {
                'temperature_calculations': [],
                'carnot_cycles': [],
                'processing_times': [],
                'physics_compliance': []
            }
            
            # Create test fields
            hot_fields = self.create_simple_fields(20, "high")
            cold_fields = self.create_simple_fields(20, "low")
            
            try:
                # Test temperature calculation
                start_time = time.time()
                hot_temp = engine.calculate_epistemic_temperature(hot_fields)
                cold_temp = engine.calculate_epistemic_temperature(cold_fields)
                temp_time = time.time() - start_time
                
                logger.info(f"  Temperature Calculation:")
                logger.info(f"    Hot Semantic/Physical: {hot_temp.semantic_temperature:.3f}/{hot_temp.physical_temperature:.3f}")
                logger.info(f"    Cold Semantic/Physical: {cold_temp.semantic_temperature:.3f}/{cold_temp.physical_temperature:.3f}")
                logger.info(f"    Confidence: {hot_temp.confidence_level:.3f}")
                
                engine_results['temperature_calculations'].append({
                    'hot_semantic': hot_temp.semantic_temperature,
                    'hot_physical': hot_temp.physical_temperature,
                    'cold_semantic': cold_temp.semantic_temperature,
                    'cold_physical': cold_temp.physical_temperature,
                    'confidence': hot_temp.confidence_level,
                    'processing_time': temp_time
                })
                
                # Test Carnot cycle
                start_time = time.time()
                carnot_result = engine.run_zetetic_carnot_engine(hot_fields, cold_fields)
                carnot_time = time.time() - start_time
                
                logger.info(f"  Carnot Cycle:")
                logger.info(f"    Theoretical Efficiency: {carnot_result.theoretical_efficiency:.3f}")
                logger.info(f"    Actual Efficiency: {carnot_result.actual_efficiency:.3f}")
                logger.info(f"    Physics Compliant: {'✅' if carnot_result.physics_compliant else '❌'}")
                logger.info(f"    Violation Detected: {'YES' if carnot_result.violation_detected else 'NO'}")
                
                engine_results['carnot_cycles'].append({
                    'theoretical_efficiency': carnot_result.theoretical_efficiency,
                    'actual_efficiency': carnot_result.actual_efficiency,
                    'physics_compliant': carnot_result.physics_compliant,
                    'violation_detected': carnot_result.violation_detected,
                    'work_extracted': carnot_result.work_extracted,
                    'processing_time': carnot_time
                })
                
                engine_results['processing_times'].extend([temp_time, carnot_time])
                engine_results['physics_compliance'].append(carnot_result.physics_compliant)
                
            except Exception as e:
                logger.error(f"  Engine {engine_name} failed: {e}")
                engine_results['error'] = str(e)
            
            results[engine_name] = engine_results
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
        
        return results
    
    def test_consciousness_detection_simple(self) -> Dict[str, Any]:
        """Test consciousness detection with simple fields"""
        logger.info("\n🧠 SIMPLE CONSCIOUSNESS DETECTION TEST")
        logger.info("=" * 70)
        
        if 'consciousness' not in self.engines:
            logger.error("❌ Consciousness engine not available")
            return {}
        
        consciousness_engine = self.engines['consciousness']
        results = {}
        
        # Test different field configurations
        test_configs = [
            (30, 'structured', 'small_structured'),
            (50, 'high', 'medium_high_energy'),
            (100, 'random', 'large_random')
        ]
        
        for field_count, pattern, test_name in test_configs:
            logger.info(f"\n🔬 Testing {test_name}: {field_count} fields, {pattern} pattern")
            
            try:
                # Create fields
                fields = self.create_simple_fields(field_count, pattern)
                
                start_time = time.time()
                gpu_before = self.collect_gpu_metrics()
                
                # Detect consciousness
                complexity_result = consciousness_engine.detect_complexity_threshold(fields)
                
                processing_time = time.time() - start_time
                gpu_after = self.collect_gpu_metrics()
                
                results[test_name] = {
                    'field_count': field_count,
                    'pattern': pattern,
                    'consciousness_probability': consciousness_result.get('consciousness_probability', 0.0),
                    'phase_transition_detected': consciousness_result.get('phase_transition_detected', False),
                    'information_integration': consciousness_result.get('information_integration', 0.0),
                    'thermodynamic_consciousness': consciousness_result.get('thermodynamic_consciousness', False),
                    'processing_time': processing_time,
                    'gpu_metrics': gpu_after
                }
                
                logger.info(f"  Consciousness Probability: {consciousness_result.get('consciousness_probability', 0.0):.3f}")
                logger.info(f"  Phase Transition: {'✅' if consciousness_result.get('phase_transition_detected', False) else '❌'}")
                logger.info(f"  Thermodynamic Consciousness: {'✅' if consciousness_result.get('thermodynamic_consciousness', False) else '❌'}")
                
            except Exception as e:
                logger.error(f"Consciousness detection failed: {e}")
                results[test_name] = {'error': str(e)}
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
        
        return results
    
    def test_physics_compliance_validation(self) -> Dict[str, Any]:
        """Test physics compliance across multiple cycles"""
        logger.info("\n⚖️ PHYSICS COMPLIANCE VALIDATION TEST")
        logger.info("=" * 70)
        
        if not self.engines:
            logger.error("❌ No engines available")
            return {}
        
        validation_results = {}
        
        # Test each engine with multiple cycles
        for engine_name, engine in self.engines.items():
            logger.info(f"\n🔧 Validating {engine_name.upper()} physics compliance:")
            
            validation_data = {
                'total_cycles': 0,
                'violations_detected': 0,
                'efficiency_data': [],
                'temperature_data': [],
                'compliance_rate': 0.0
            }
            
            # Run multiple validation cycles
            for cycle in range(10):
                try:
                    # Create fresh fields for each cycle
                    hot_fields = self.create_simple_fields(15, "high")
                    cold_fields = self.create_simple_fields(15, "low")
                    
                    # Run Carnot cycle
                    result = engine.run_zetetic_carnot_engine(hot_fields, cold_fields)
                    
                    validation_data['total_cycles'] += 1
                    
                    if result.violation_detected:
                        validation_data['violations_detected'] += 1
                    
                    validation_data['efficiency_data'].append({
                        'cycle': cycle,
                        'theoretical': result.theoretical_efficiency,
                        'actual': result.actual_efficiency,
                        'violation': result.violation_detected
                    })
                    
                    validation_data['temperature_data'].append({
                        'cycle': cycle,
                        'hot_semantic': result.hot_temperature.semantic_temperature,
                        'hot_physical': result.hot_temperature.physical_temperature,
                        'cold_semantic': result.cold_temperature.semantic_temperature,
                        'cold_physical': result.cold_temperature.physical_temperature
                    })
                    
                except Exception as e:
                    logger.error(f"  Cycle {cycle} failed: {e}")
                    validation_data['violations_detected'] += 1
            
            # Calculate compliance rate
            if validation_data['total_cycles'] > 0:
                validation_data['compliance_rate'] = 1.0 - (validation_data['violations_detected'] / validation_data['total_cycles'])
            
            validation_results[engine_name] = validation_data
            
            logger.info(f"  Total Cycles: {validation_data['total_cycles']}")
            logger.info(f"  Violations: {validation_data['violations_detected']}")
            logger.info(f"  Compliance Rate: {validation_data['compliance_rate']:.1%}")
            logger.info(f"  Physics Compliant: {'✅' if validation_data['compliance_rate'] > 0.9 else '❌'}")
        
        return validation_results
    
    def run_simple_comprehensive_test(self) -> Dict[str, Any]:
        """Run simple comprehensive test suite"""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 SIMPLE COMPREHENSIVE ZETETIC TEST")
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
            # 1. Basic Thermodynamic Operations
            logger.info("\n1/3 - BASIC THERMODYNAMIC OPERATIONS TEST")
            basic_results = self.test_basic_thermodynamic_operations()
            comprehensive_results['test_results']['basic_operations'] = basic_results
            
            # 2. Consciousness Detection
            logger.info("\n2/3 - SIMPLE CONSCIOUSNESS DETECTION TEST")
            consciousness_results = self.test_consciousness_detection_simple()
            comprehensive_results['test_results']['consciousness_detection'] = consciousness_results
            
            # 3. Physics Compliance Validation
            logger.info("\n3/3 - PHYSICS COMPLIANCE VALIDATION TEST")
            validation_results = self.test_physics_compliance_validation()
            comprehensive_results['test_results']['physics_validation'] = validation_results
            
            # Generate summary
            comprehensive_results['summary'] = self._generate_simple_summary(comprehensive_results)
            
            comprehensive_results['test_metadata']['end_time'] = datetime.now().isoformat()
            comprehensive_results['test_metadata']['total_duration'] = (datetime.now() - self.start_time).total_seconds()
            
        except Exception as e:
            logger.error(f"❌ Simple comprehensive test failed: {e}")
            logger.error(traceback.format_exc())
            comprehensive_results['error'] = str(e)
        
        return comprehensive_results
    
    def _generate_simple_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simple test summary"""
        summary = {
            'overall_status': 'UNKNOWN',
            'engines_operational': False,
            'basic_operations_successful': False,
            'consciousness_detection_functional': False,
            'physics_compliance_validated': False,
            'overall_compliance_rate': 0.0,
            'zetetic_validation_passed': False
        }
        
        try:
            # Check engine availability
            summary['engines_operational'] = len(self.engines) > 0
            
            # Analyze basic operations
            basic_results = results['test_results'].get('basic_operations', {})
            if basic_results:
                successful_engines = sum(
                    1 for result in basic_results.values()
                    if isinstance(result, dict) and 'error' not in result
                )
                summary['basic_operations_successful'] = successful_engines > 0
            
            # Analyze consciousness detection
            consciousness_results = results['test_results'].get('consciousness_detection', {})
            if consciousness_results:
                functional_detections = sum(
                    1 for result in consciousness_results.values()
                    if isinstance(result, dict) and 'error' not in result
                )
                summary['consciousness_detection_functional'] = functional_detections > 0
            
            # Analyze physics compliance
            validation_results = results['test_results'].get('physics_validation', {})
            if validation_results:
                compliance_rates = [
                    result.get('compliance_rate', 0.0)
                    for result in validation_results.values()
                    if isinstance(result, dict)
                ]
                if compliance_rates:
                    summary['overall_compliance_rate'] = statistics.mean(compliance_rates)
                    summary['physics_compliance_validated'] = summary['overall_compliance_rate'] > 0.8
            
            # Overall zetetic validation
            summary['zetetic_validation_passed'] = (
                summary['engines_operational'] and
                summary['basic_operations_successful'] and
                summary['physics_compliance_validated']
            )
            
            # Overall status
            if summary['zetetic_validation_passed']:
                if summary['consciousness_detection_functional']:
                    summary['overall_status'] = 'REVOLUTIONARY_SUCCESS'
                else:
                    summary['overall_status'] = 'ZETETIC_SUCCESS'
            elif summary['engines_operational']:
                summary['overall_status'] = 'PARTIALLY_FUNCTIONAL'
            else:
                summary['overall_status'] = 'SYSTEM_FAILURE'
        
        except Exception as e:
            logger.warning(f"Simple summary generation failed: {e}")
        
        return summary
    
    def save_simple_results(self, results: Dict[str, Any]) -> str:
        """Save simple test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_working_zetetic_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {filename}")
        return filename


def main():
    """Run simple working zetetic test"""
    tester = SimpleWorkingZeteticTester()
    
    try:
        # Run comprehensive test
        results = tester.run_simple_comprehensive_test()
        
        # Save results
        filename = tester.save_simple_results(results)
        
        # Print summary
        print("\n" + "=" * 80)
        print("🚀 SIMPLE WORKING ZETETIC TEST COMPLETE")
        print("=" * 80)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Overall Status: {summary['overall_status']}")
            print(f"Engines Operational: {'✅' if summary['engines_operational'] else '❌'}")
            print(f"Basic Operations: {'✅ SUCCESSFUL' if summary['basic_operations_successful'] else '❌ FAILED'}")
            print(f"Consciousness Detection: {'✅ FUNCTIONAL' if summary['consciousness_detection_functional'] else '❌ NON-FUNCTIONAL'}")
            print(f"Physics Compliance: {'✅ VALIDATED' if summary['physics_compliance_validated'] else '❌ FAILED'}")
            print(f"Overall Compliance Rate: {summary['overall_compliance_rate']:.1%}")
            print(f"Zetetic Validation: {'✅ PASSED' if summary['zetetic_validation_passed'] else '❌ FAILED'}")
        
        print(f"Detailed results: {filename}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Simple working test failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main() 