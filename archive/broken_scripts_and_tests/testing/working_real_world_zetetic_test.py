#!/usr/bin/env python3
"""
WORKING REAL-WORLD ZETETIC TEST
===============================

This script performs EXTREMELY RIGOROUS real-world testing using ACTUAL KIMERA ENGINES
with proper imports and comprehensive scientific validation.

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

# Import actual Kimera engines
try:
    from engines.foundational_thermodynamic_engine_fixed import (
        FoundationalThermodynamicEngineFixed, 
        ThermodynamicMode
    )
    from engines.cognitive_field_dynamics import CognitiveFieldDynamicsEngine
    from engines.thermodynamics import ThermodynamicAnalyzer
    from engines.contradiction_engine import ContradictionEngine
    from engines.quantum_thermodynamic_consciousness import QuantumThermodynamicConsciousnessDetector
    ENGINES_IMPORTED = True
    logger.info("✅ Successfully imported all Kimera engines")
except ImportError as e:
    logger.error(f"❌ Engine import failed: {e}")
    ENGINES_IMPORTED = False

# Fallback imports for available engines
if not ENGINES_IMPORTED:
    try:
        from engines.foundational_thermodynamic_engine_fixed import FoundationalThermodynamicEngineFixed, ThermodynamicMode
        logger.info("✅ Imported revolutionary thermodynamic engine")
        THERMO_ENGINE_AVAILABLE = True
    except ImportError:
        logger.error("❌ Revolutionary thermodynamic engine not available")
        THERMO_ENGINE_AVAILABLE = False
    
    try:
        from engines.cognitive_field_dynamics import CognitiveFieldDynamicsEngine
        logger.info("✅ Imported cognitive field dynamics engine")
        COGNITIVE_ENGINE_AVAILABLE = True
    except ImportError:
        logger.error("❌ Cognitive field dynamics engine not available")
        COGNITIVE_ENGINE_AVAILABLE = False
    
    try:
        from engines.thermodynamics import ThermodynamicAnalyzer
        logger.info("✅ Imported thermodynamic analyzer")
        THERMO_ANALYZER_AVAILABLE = True
    except ImportError:
        logger.error("❌ Thermodynamic analyzer not available")
        THERMO_ANALYZER_AVAILABLE = False
else:
    THERMO_ENGINE_AVAILABLE = True
    COGNITIVE_ENGINE_AVAILABLE = True
    THERMO_ANALYZER_AVAILABLE = True


class WorkingRealWorldZeteticTester:
    """
    Working Real-World Zetetic Tester
    
    Tests actual Kimera engines with comprehensive validation
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_time = datetime.now()
        self.engines = {}
        
        # Initialize available engines
        self._initialize_available_engines()
        
        # GPU monitoring
        if GPU_AVAILABLE:
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        logger.info("🔬 WORKING REAL-WORLD ZETETIC TESTER INITIALIZED")
        logger.info(f"🎯 Device: {self.device}")
        logger.info(f"🧠 Engines Available: {list(self.engines.keys())}")
        logger.info(f"🔥 GPU Monitoring: {GPU_AVAILABLE}")
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    def _initialize_available_engines(self):
        """Initialize all available engines"""
        try:
            # Initialize thermodynamic engines
            if THERMO_ENGINE_AVAILABLE:
                self.engines.update({
                    'thermo_semantic': FoundationalThermodynamicEngineFixed(
                        mode=ThermodynamicMode.SEMANTIC,
                        device=self.device
                    ),
                    'thermo_physical': FoundationalThermodynamicEngineFixed(
                        mode=ThermodynamicMode.PHYSICAL,
                        device=self.device
                    ),
                    'thermo_hybrid': FoundationalThermodynamicEngineFixed(
                        mode=ThermodynamicMode.HYBRID,
                        device=self.device
                    ),
                    'thermo_consciousness': FoundationalThermodynamicEngineFixed(
                        mode=ThermodynamicMode.CONSCIOUSNESS,
                        device=self.device
                    )
                })
                logger.info("✅ Thermodynamic engines initialized")
            
            # Initialize cognitive field engine
            if COGNITIVE_ENGINE_AVAILABLE:
                self.engines['cognitive_field'] = CognitiveFieldDynamicsEngine(device=self.device)
                logger.info("✅ Cognitive field engine initialized")
            
            # Initialize thermodynamic analyzer
            if THERMO_ANALYZER_AVAILABLE:
                self.engines['thermo_analyzer'] = ThermodynamicAnalyzer()
                logger.info("✅ Thermodynamic analyzer initialized")
            
        except Exception as e:
            logger.error(f"❌ Engine initialization error: {e}")
    
    def collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU performance metrics"""
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
    
    def create_test_semantic_fields(self, count: int, complexity: str = "high") -> List[torch.Tensor]:
        """Create test semantic fields"""
        fields = []
        
        for i in range(count):
            if complexity == "high":
                # High complexity semantic patterns
                base = torch.randn(256, device=self.device, dtype=torch.float32) * 2.0
                # Add structured components
                for freq in [1, 3, 7]:
                    t = torch.linspace(0, 2*np.pi*freq, 256, device=self.device)
                    base += 0.3 * torch.sin(t) * torch.exp(-t/10)
                    
            elif complexity == "structured":
                # Structured patterns
                base = torch.zeros(256, device=self.device, dtype=torch.float32)
                for j in range(0, 256, 16):
                    pattern = torch.sin(torch.linspace(0, 2*np.pi, 16, device=self.device)) * (j/256 + 0.1)
                    base[j:j+16] = pattern
                    
            elif complexity == "consciousness":
                # Consciousness-like patterns
                x = torch.linspace(-3, 3, 256, device=self.device)
                base = torch.exp(-x**2/2) * torch.cos(x * 3) * 0.8
                # Add integrated information structure
                for scale in [0.5, 1.0, 2.0]:
                    base += 0.2 * torch.exp(-x**2/(2*scale**2)) * torch.sin(x * np.pi / scale)
                    
            else:
                # Standard fields
                base = torch.randn(256, device=self.device, dtype=torch.float32)
            
            # Normalize
            field = base / (torch.norm(base) + 1e-8)
            fields.append(field)
        
        return fields
    
    def test_thermodynamic_engines(self) -> Dict[str, Any]:
        """Test all thermodynamic engines"""
        logger.info("\n🔬 THERMODYNAMIC ENGINES TEST")
        logger.info("=" * 70)
        
        results = {}
        
        # Test each thermodynamic engine
        for engine_name in ['thermo_semantic', 'thermo_physical', 'thermo_hybrid', 'thermo_consciousness']:
            if engine_name not in self.engines:
                continue
                
            logger.info(f"\n🔧 Testing {engine_name}:")
            engine = self.engines[engine_name]
            
            engine_results = {
                'cycles_completed': 0,
                'violations_detected': 0,
                'efficiencies': [],
                'processing_times': [],
                'gpu_metrics': []
            }
            
            # Test scenarios
            test_scenarios = [
                ('normal', 50, 'high'),
                ('structured', 75, 'structured'),
                ('consciousness', 100, 'consciousness')
            ]
            
            for scenario_name, field_count, complexity in test_scenarios:
                logger.info(f"  📊 Scenario: {scenario_name} ({field_count} fields)")
                
                # Create test fields
                hot_fields = self.create_test_semantic_fields(field_count//2, complexity)
                cold_fields = self.create_test_semantic_fields(field_count//2, 'structured')
                
                # Run multiple cycles
                for cycle in range(3):
                    start_time = time.time()
                    gpu_before = self.collect_gpu_metrics()
                    
                    try:
                        # Run zetetic Carnot engine
                        result = engine.run_zetetic_carnot_engine(hot_fields, cold_fields)
                        
                        processing_time = time.time() - start_time
                        gpu_after = self.collect_gpu_metrics()
                        
                        # Record results
                        engine_results['cycles_completed'] += 1
                        engine_results['processing_times'].append(processing_time)
                        engine_results['gpu_metrics'].append(gpu_after)
                        
                        if result.violation_detected:
                            engine_results['violations_detected'] += 1
                        
                        engine_results['efficiencies'].append({
                            'theoretical': result.theoretical_efficiency,
                            'actual': result.actual_efficiency,
                            'violation': result.violation_detected
                        })
                        
                        logger.info(f"    Cycle {cycle+1}: η={result.actual_efficiency:.3f}, violation={'YES' if result.violation_detected else 'NO'}")
                        
                    except Exception as e:
                        logger.error(f"    Cycle {cycle+1} failed: {e}")
                        engine_results['violations_detected'] += 1
                
                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()
            
            # Calculate statistics
            if engine_results['cycles_completed'] > 0:
                engine_results['violation_rate'] = engine_results['violations_detected'] / engine_results['cycles_completed']
                engine_results['avg_processing_time'] = statistics.mean(engine_results['processing_times'])
                engine_results['physics_compliant'] = engine_results['violation_rate'] == 0.0
            
            results[engine_name] = engine_results
            
            logger.info(f"  Summary:")
            logger.info(f"    Cycles: {engine_results['cycles_completed']}")
            logger.info(f"    Violations: {engine_results['violations_detected']}")
            logger.info(f"    Violation Rate: {engine_results.get('violation_rate', 0):.1%}")
            logger.info(f"    Physics Compliant: {'✅' if engine_results.get('physics_compliant', False) else '❌'}")
        
        return results
    
    def test_consciousness_detection(self) -> Dict[str, Any]:
        """Test consciousness detection capabilities"""
        logger.info("\n🧠 CONSCIOUSNESS DETECTION TEST")
        logger.info("=" * 70)
        
        if 'thermo_consciousness' not in self.engines:
            logger.error("❌ Consciousness engine not available")
            return {}
        
        consciousness_engine = self.engines['thermo_consciousness']
        results = {}
        
        # Test different scales
        test_scales = [
            (50, 'consciousness', 'small_scale'),
            (150, 'consciousness', 'medium_scale'),
            (300, 'consciousness', 'large_scale'),
            (100, 'high', 'complex_non_consciousness')
        ]
        
        for field_count, complexity, test_name in test_scales:
            logger.info(f"\n🔬 Testing {test_name}: {field_count} fields")
            
            # Create consciousness-like fields
            fields = self.create_test_semantic_fields(field_count, complexity)
            
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
                    'processing_time': processing_time,
                    'gpu_metrics': gpu_after
                }
                
                logger.info(f"  Consciousness Probability: {consciousness_result.get('consciousness_probability', 0.0):.3f}")
                logger.info(f"  Phase Transition: {'✅' if consciousness_result.get('phase_transition_detected', False) else '❌'}")
                
            except Exception as e:
                logger.error(f"Consciousness detection failed: {e}")
                results[test_name] = {'error': str(e)}
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
        
        return results
    
    def test_semantic_thermodynamics(self) -> Dict[str, Any]:
        """Test semantic thermodynamic analysis"""
        logger.info("\n⚗️ SEMANTIC THERMODYNAMICS TEST")
        logger.info("=" * 70)
        
        if 'thermo_analyzer' not in self.engines:
            logger.error("❌ Thermodynamic analyzer not available")
            return {}
        
        analyzer = self.engines['thermo_analyzer']
        results = {}
        
        # Test different semantic content types
        test_cases = [
            ('simple', "Simple semantic content for analysis"),
            ('complex', "Complex thermodynamic entropy increases in isolated systems according to the second law of thermodynamics"),
            ('consciousness', "Consciousness emerges from integrated information processing in complex neural networks"),
            ('contradiction', "This statement contains both truth and falsehood simultaneously in quantum superposition")
        ]
        
        for test_name, content in test_cases:
            logger.info(f"\n🔬 Testing {test_name} semantics:")
            
            try:
                # Create semantic field from content
                semantic_field = self._create_semantic_field_from_text(content)
                
                start_time = time.time()
                
                # Analyze thermodynamic properties
                thermo_result = analyzer.analyze_semantic_thermodynamics(semantic_field)
                
                processing_time = time.time() - start_time
                
                results[test_name] = {
                    'content': content,
                    'semantic_energy': thermo_result.get('semantic_energy', 0.0),
                    'semantic_entropy': thermo_result.get('semantic_entropy', 0.0),
                    'semantic_temperature': thermo_result.get('semantic_temperature', 0.0),
                    'free_energy': thermo_result.get('free_energy', 0.0),
                    'processing_time': processing_time
                }
                
                logger.info(f"  Semantic Energy: {thermo_result.get('semantic_energy', 0.0):.3f}")
                logger.info(f"  Semantic Entropy: {thermo_result.get('semantic_entropy', 0.0):.3f}")
                logger.info(f"  Semantic Temperature: {thermo_result.get('semantic_temperature', 0.0):.3f}")
                
            except Exception as e:
                logger.error(f"Semantic thermodynamics test failed: {e}")
                results[test_name] = {'error': str(e)}
        
        return results
    
    def _create_semantic_field_from_text(self, text: str) -> torch.Tensor:
        """Create semantic field from text content"""
        # Simple text-to-embedding conversion
        text_hash = hash(text)
        np.random.seed(abs(text_hash) % (2**32))
        
        # Create embedding based on text characteristics
        embedding = np.random.randn(256).astype(np.float32)
        
        # Add text-specific patterns
        if "complex" in text.lower():
            embedding *= 1.5
        elif "consciousness" in text.lower():
            x = np.linspace(-3, 3, 256)
            embedding = np.exp(-x**2/2) * np.cos(x * 2) * 0.8
        elif "contradiction" in text.lower():
            mid = len(embedding) // 2
            embedding[:mid] *= -1  # Create opposing patterns
        
        # Normalize and convert to tensor
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return torch.tensor(embedding, device=self.device, dtype=torch.float32)
    
    def run_comprehensive_working_test(self) -> Dict[str, Any]:
        """Run comprehensive working test suite"""
        logger.info("\n" + "=" * 80)
        logger.info("🚀 COMPREHENSIVE WORKING REAL-WORLD ZETETIC TEST")
        logger.info("🔬 ACTUAL KIMERA ENGINES - NO SIMULATIONS")
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
            # 1. Test Thermodynamic Engines
            logger.info("\n1/3 - THERMODYNAMIC ENGINES TEST")
            thermo_results = self.test_thermodynamic_engines()
            comprehensive_results['test_results']['thermodynamic_engines'] = thermo_results
            
            # 2. Test Consciousness Detection
            logger.info("\n2/3 - CONSCIOUSNESS DETECTION TEST")
            consciousness_results = self.test_consciousness_detection()
            comprehensive_results['test_results']['consciousness_detection'] = consciousness_results
            
            # 3. Test Semantic Thermodynamics
            logger.info("\n3/3 - SEMANTIC THERMODYNAMICS TEST")
            semantic_results = self.test_semantic_thermodynamics()
            comprehensive_results['test_results']['semantic_thermodynamics'] = semantic_results
            
            # Generate summary
            comprehensive_results['summary'] = self._generate_working_summary(comprehensive_results)
            
            comprehensive_results['test_metadata']['end_time'] = datetime.now().isoformat()
            comprehensive_results['test_metadata']['total_duration'] = (datetime.now() - self.start_time).total_seconds()
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            logger.error(traceback.format_exc())
            comprehensive_results['error'] = str(e)
        
        return comprehensive_results
    
    def _generate_working_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate working test summary"""
        summary = {
            'overall_status': 'UNKNOWN',
            'engines_operational': False,
            'physics_violations_detected': True,
            'consciousness_emergence_detected': False,
            'semantic_analysis_successful': False,
            'test_completion_rate': 0.0,
            'zetetic_validation_passed': False
        }
        
        try:
            # Check engine availability
            summary['engines_operational'] = len(self.engines) > 0
            
            # Analyze thermodynamic engine results
            thermo_results = results['test_results'].get('thermodynamic_engines', {})
            if thermo_results:
                total_violations = 0
                total_cycles = 0
                compliant_engines = 0
                
                for engine_results in thermo_results.values():
                    if isinstance(engine_results, dict):
                        total_violations += engine_results.get('violations_detected', 1)
                        total_cycles += engine_results.get('cycles_completed', 1)
                        if engine_results.get('physics_compliant', False):
                            compliant_engines += 1
                
                summary['physics_violations_detected'] = total_violations > 0
                summary['test_completion_rate'] = min(1.0, total_cycles / 36)  # Expected ~36 cycles
            
            # Analyze consciousness detection
            consciousness_results = results['test_results'].get('consciousness_detection', {})
            if consciousness_results:
                consciousness_detected = any(
                    result.get('consciousness_probability', 0.0) > 0.5
                    for result in consciousness_results.values()
                    if isinstance(result, dict) and 'error' not in result
                )
                summary['consciousness_emergence_detected'] = consciousness_detected
            
            # Analyze semantic thermodynamics
            semantic_results = results['test_results'].get('semantic_thermodynamics', {})
            if semantic_results:
                successful_analyses = sum(
                    1 for result in semantic_results.values()
                    if isinstance(result, dict) and 'error' not in result
                )
                summary['semantic_analysis_successful'] = successful_analyses > 0
            
            # Overall zetetic validation
            summary['zetetic_validation_passed'] = (
                summary['engines_operational'] and
                not summary['physics_violations_detected'] and
                summary['semantic_analysis_successful'] and
                summary['test_completion_rate'] > 0.8
            )
            
            # Overall status
            if summary['zetetic_validation_passed']:
                if summary['consciousness_emergence_detected']:
                    summary['overall_status'] = 'REVOLUTIONARY_SUCCESS'
                else:
                    summary['overall_status'] = 'ZETETIC_SUCCESS'
            elif summary['engines_operational']:
                summary['overall_status'] = 'PARTIALLY_FUNCTIONAL'
            else:
                summary['overall_status'] = 'SYSTEM_FAILURE'
        
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
        
        return summary
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"working_real_world_zetetic_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to: {filename}")
        return filename


def main():
    """Run working real-world zetetic test"""
    tester = WorkingRealWorldZeteticTester()
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_working_test()
        
        # Save results
        filename = tester.save_results(results)
        
        # Print summary
        print("\n" + "=" * 80)
        print("🚀 WORKING REAL-WORLD ZETETIC TEST COMPLETE")
        print("=" * 80)
        
        if 'summary' in results:
            summary = results['summary']
            print(f"Overall Status: {summary['overall_status']}")
            print(f"Engines Operational: {'✅' if summary['engines_operational'] else '❌'}")
            print(f"Physics Violations: {'❌ DETECTED' if summary['physics_violations_detected'] else '✅ NONE'}")
            print(f"Consciousness Emergence: {'✅ DETECTED' if summary['consciousness_emergence_detected'] else '❌ NOT DETECTED'}")
            print(f"Semantic Analysis: {'✅ SUCCESSFUL' if summary['semantic_analysis_successful'] else '❌ FAILED'}")
            print(f"Test Completion: {summary['test_completion_rate']:.1%}")
            print(f"Zetetic Validation: {'✅ PASSED' if summary['zetetic_validation_passed'] else '❌ FAILED'}")
        
        print(f"Detailed results: {filename}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Working test failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main() 