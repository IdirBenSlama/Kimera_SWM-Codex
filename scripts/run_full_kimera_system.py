#!/usr/bin/env python3
"""
Full KIMERA SWM System Instance Runner
=====================================

Runs a complete KIMERA SWM system instance with all components
and tests functionality in a real operating environment.
"""

import asyncio
import time
import logging
import json
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime

# KIMERA Core System Components
from src.engines.vortex_energy_storage import VortexEnergyStorage, ResonancePattern
from src.engines.gyroscopic_universal_translator import GyroscopicUniversalTranslator, TranslationModality, TranslationRequest
from src.utils.memory_manager import memory_manager, MemoryContext
from src.utils.dependency_manager import dependency_manager  
from src.utils.processing_optimizer import processing_optimizer
from src.utils.gpu_optimizer import gpu_optimizer
from src.optimization.ai_system_optimizer import ai_optimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraSystemInstance:
    """Full KIMERA SWM System Instance"""
    
    def __init__(self):
        self.components = {}
        self.performance_metrics = {}
        self.running = False
        self.start_time = None
        
    async def initialize_system(self):
        """Initialize all KIMERA system components"""
        
        logger.info("🚀 Initializing Full KIMERA SWM System...")
        
        try:
            # Initialize core components
            self.components['memory_manager'] = memory_manager
            self.components['dependency_manager'] = dependency_manager
            self.components['processing_optimizer'] = processing_optimizer
            self.components['gpu_optimizer'] = gpu_optimizer
            self.components['ai_optimizer'] = ai_optimizer
            
            # Initialize vortex energy storage
            self.components['vortex_storage'] = VortexEnergyStorage()
            logger.info("✅ Vortex Energy Storage initialized")
            
            # Initialize universal translator
            self.components['translator'] = GyroscopicUniversalTranslator()
            logger.info("✅ Universal Translator initialized")
            
            # Initialize memory monitoring
            memory_manager.start_monitoring()
            logger.info("✅ Memory monitoring started")
            
            logger.info("🎉 KIMERA SWM System fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            traceback.print_exc()
            return False
    
    async def run_operational_tests(self):
        """Run comprehensive operational tests on the live system"""
        
        logger.info("🔬 Running operational system tests...")
        
        test_results = {
            'energy_operations': await self._test_energy_operations(),
            'translation_operations': await self._test_translation_operations(),
            'memory_operations': await self._test_memory_operations(),
            'system_integration': await self._test_system_integration(),
            'performance_metrics': await self._collect_performance_metrics()
        }
        
        return test_results
    
    async def _test_energy_operations(self):
        """Test vortex energy storage operations"""
        
        logger.info("⚡ Testing energy operations...")
        
        try:
            vs = self.components['vortex_storage']
            
            # Test energy storage cycle
            stored = vs.store_energy(2000.0)
            initial_status = vs.get_system_status()
            
            # Test resonance activation
            resonance_results = {}
            for pattern in ResonancePattern:
                activated = vs.activate_resonance(pattern)
                resonance_results[pattern.value] = activated
                
            # Test energy retrieval
            retrieved = vs.retrieve_energy(1000.0)
            final_status = vs.get_system_status()
            
            return {
                'status': 'SUCCESS',
                'energy_stored': stored,
                'energy_retrieved': retrieved,
                'resonance_patterns': resonance_results,
                'initial_capacity': initial_status.get('capacity_utilization', 0.0),
                'final_capacity': final_status.get('capacity_utilization', 0.0),
                'resonance_strength': final_status.get('resonance_strength', 0.0),
                'quantum_efficiency': final_status.get('quantum_efficiency', 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ Energy operations test failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    async def _test_translation_operations(self):
        """Test universal translation operations"""
        
        logger.info("🌐 Testing translation operations...")
        
        try:
            translator = self.components['translator']
            
            # Test basic translation
            request = TranslationRequest(
                source_modality=TranslationModality.NATURAL_LANGUAGE,
                target_modality=TranslationModality.NATURAL_LANGUAGE,
                content="Test KIMERA system operational translation"
            )
            
            result = await translator.translate(request)
            
            # Test different modality translations
            modality_tests = {}
            for modality in [TranslationModality.NATURAL_LANGUAGE, TranslationModality.MATHEMATICAL, TranslationModality.QUANTUM_ACTIONS]:
                try:
                    test_request = TranslationRequest(
                        source_modality=TranslationModality.NATURAL_LANGUAGE,
                        target_modality=modality,
                        content="KIMERA operational test"
                    )
                    test_result = await translator.translate(test_request)
                    modality_tests[modality.value] = {
                        'confidence': test_result.confidence_score,
                        'processing_time': test_result.processing_time,
                        'security_validated': test_result.security_validated
                    }
                except Exception as e:
                    modality_tests[modality.value] = {'error': str(e)}
            
            return {
                'status': 'SUCCESS',
                'primary_translation': {
                    'confidence': result.confidence_score,
                    'processing_time': result.processing_time,
                    'gyroscopic_stability': result.gyroscopic_stability,
                    'quantum_coherence': result.quantum_coherence
                },
                'modality_tests': modality_tests
            }
            
        except Exception as e:
            logger.error(f"❌ Translation operations test failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    async def _test_memory_operations(self):
        """Test memory management operations"""
        
        logger.info("🧠 Testing memory operations...")
        
        try:
            mm = self.components['memory_manager']
            
            # Test memory context operations
            with mm.get_context('test_pool') as ctx:
                # Simulate memory-intensive operations
                test_tensor = ctx.allocate_tensor((100, 100)) if hasattr(ctx, 'allocate_tensor') else None
                
            # Get memory report
            memory_report = mm.get_memory_report()
            
            return {
                'status': 'SUCCESS',
                'memory_report': memory_report,
                'context_operations': 'WORKING',
                'monitoring_active': memory_report.get('monitoring_active', False)
            }
            
        except Exception as e:
            logger.error(f"❌ Memory operations test failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    async def _test_system_integration(self):
        """Test integration between system components"""
        
        logger.info("🔗 Testing system integration...")
        
        try:
            # Test energy + translation integration
            vs = self.components['vortex_storage']
            translator = self.components['translator']
            
            # Store energy and activate resonance
            vs.store_energy(1500.0)
            vs.activate_resonance(ResonancePattern.GOLDEN_RATIO)
            
            # Get system status
            energy_status = vs.get_system_status()
            
            # Perform translation with enhanced energy
            request = TranslationRequest(
                source_modality=TranslationModality.NATURAL_LANGUAGE,
                target_modality=TranslationModality.QUANTUM_ACTIONS,
                content="Integrate vortex energy with quantum translation"
            )
            
            translation_result = await translator.translate(request)
            
            # Test AI optimization integration
            ai_report = ai_optimizer.get_optimization_report()
            
            return {
                'status': 'SUCCESS',
                'energy_translation_sync': {
                    'energy_level': energy_status.get('total_energy', 0.0),
                    'resonance_strength': energy_status.get('resonance_strength', 0.0),
                    'translation_confidence': translation_result.confidence_score,
                    'quantum_coherence': translation_result.quantum_coherence
                },
                'ai_optimization': {
                    'active': ai_report.get('optimization_active', False),
                    'suggestions': len(ai_report.get('suggestions', []))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ System integration test failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    async def _collect_performance_metrics(self):
        """Collect comprehensive performance metrics"""
        
        logger.info("📊 Collecting performance metrics...")
        
        try:
            vs = self.components['vortex_storage']
            translator = self.components['translator']
            mm = self.components['memory_manager']
            
            # Vortex system metrics
            vortex_status = vs.get_system_status()
            
            # Translation system metrics
            translator_status = translator.get_translator_status()
            
            # Memory system metrics
            memory_report = mm.get_memory_report()
            
            # AI optimization metrics
            ai_report = ai_optimizer.get_optimization_report()
            
            return {
                'vortex_metrics': {
                    'total_energy': vortex_status.get('total_energy', 0.0),
                    'capacity_utilization': vortex_status.get('capacity_utilization', 0.0),
                    'resonance_strength': vortex_status.get('resonance_strength', 0.0),
                    'quantum_efficiency': vortex_status.get('quantum_efficiency', 0.0),
                    'system_stability': vortex_status.get('system_stability', 0.0)
                },
                'translation_metrics': {
                    'total_translations': translator_status.get('stats', {}).get('total_translations', 0),
                    'successful_translations': translator_status.get('stats', {}).get('successful_translations', 0),
                    'average_confidence': translator_status.get('stats', {}).get('average_confidence', 0.0),
                    'equilibrium_maintained': translator_status.get('stats', {}).get('equilibrium_maintained', 0)
                },
                'memory_metrics': {
                    'tracked_objects': memory_report.get('tracked_objects', 0),
                    'memory_pools': memory_report.get('memory_pools', {}),
                    'monitoring_active': memory_report.get('monitoring_active', False)
                },
                'ai_metrics': {
                    'optimization_active': ai_report.get('optimization_active', False),
                    'cognitive_fidelity_score': ai_report.get('cognitive_fidelity_score', 0.0),
                    'suggestions_count': len(ai_report.get('suggestions', []))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Performance metrics collection failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def calculate_system_health(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health from test results"""
        
        health_scores = {}
        
        # Energy system health
        energy_ops = test_results.get('energy_operations', {})
        if energy_ops.get('status') == 'SUCCESS':
            energy_health = (
                (1.0 if energy_ops.get('energy_stored', False) else 0.0) * 0.2 +
                min(1.0, energy_ops.get('energy_retrieved', 0.0) / 500.0) * 0.2 +
                (energy_ops.get('final_capacity', 0.0)) * 0.2 +
                min(1.0, energy_ops.get('resonance_strength', 0.0) / 1000.0) * 0.2 +
                energy_ops.get('quantum_efficiency', 0.0) * 0.2
            )
        else:
            energy_health = 0.0
        
        health_scores['energy_health'] = energy_health
        
        # Translation system health
        trans_ops = test_results.get('translation_operations', {})
        if trans_ops.get('status') == 'SUCCESS':
            primary = trans_ops.get('primary_translation', {})
            translation_health = (
                primary.get('confidence', 0.0) * 0.3 +
                (1.0 - min(1.0, primary.get('processing_time', 1.0))) * 0.2 +
                primary.get('gyroscopic_stability', 0.0) * 0.25 +
                primary.get('quantum_coherence', 0.0) * 0.25
            )
        else:
            translation_health = 0.0
        
        health_scores['translation_health'] = translation_health
        
        # Memory system health
        memory_ops = test_results.get('memory_operations', {})
        memory_health = 1.0 if memory_ops.get('status') == 'SUCCESS' else 0.0
        health_scores['memory_health'] = memory_health
        
        # Integration health
        integration = test_results.get('system_integration', {})
        if integration.get('status') == 'SUCCESS':
            sync_data = integration.get('energy_translation_sync', {})
            integration_health = (
                min(1.0, sync_data.get('energy_level', 0.0) / 1000.0) * 0.3 +
                min(1.0, sync_data.get('resonance_strength', 0.0) / 1000.0) * 0.3 +
                sync_data.get('translation_confidence', 0.0) * 0.4
            )
        else:
            integration_health = 0.0
        
        health_scores['integration_health'] = integration_health
        
        # Overall system health
        overall_health = (
            energy_health * 0.3 +
            translation_health * 0.3 +
            memory_health * 0.2 +
            integration_health * 0.2
        )
        
        health_scores['overall_health'] = overall_health
        
        return health_scores
    
    async def run_full_system_test(self):
        """Run complete system test suite"""
        
        self.start_time = time.time()
        logger.info("🎯 Starting Full KIMERA SWM System Test...")
        
        # Initialize system
        if not await self.initialize_system():
            return {"status": "INITIALIZATION_FAILED"}
        
        self.running = True
        
        try:
            # Run operational tests
            test_results = await self.run_operational_tests()
            
            # Calculate health scores
            health_scores = self.calculate_system_health(test_results)
            
            # Final system status
            duration = time.time() - self.start_time
            
            final_report = {
                "timestamp": datetime.now().isoformat(),
                "test_duration": duration,
                "system_status": "OPERATIONAL" if health_scores['overall_health'] > 0.5 else "DEGRADED",
                "overall_health": health_scores['overall_health'],
                "health_breakdown": health_scores,
                "test_results": test_results,
                "performance_summary": {
                    "initialization_time": "< 1s" if duration < 1 else f"{duration:.1f}s",
                    "operational_stability": "STABLE" if all(
                        result.get('status') == 'SUCCESS' 
                        for result in test_results.values() 
                        if isinstance(result, dict)
                    ) else "UNSTABLE"
                }
            }
            
            logger.info(f"🏁 System test completed in {duration:.2f}s")
            logger.info(f"📊 Overall Health: {health_scores['overall_health']:.2f}")
            
            return final_report
            
        except Exception as e:
            logger.error(f"❌ System test failed: {e}")
            traceback.print_exc()
            return {
                "status": "FAILED",
                "error": str(e),
                "duration": time.time() - self.start_time if self.start_time else 0
            }
        
        finally:
            self.running = False
            # Cleanup
            if 'vortex_storage' in self.components:
                self.components['vortex_storage'].shutdown()

async def main():
    """Main entry point"""
    
    logger.info("🚀 KIMERA SWM FULL SYSTEM INSTANCE TEST")
    logger.info("=" * 60)
    
    system = KimeraSystemInstance()
    results = await system.run_full_system_test()
    
    # Save results
    results_file = f"kimera_full_system_test_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 60)
    logger.info("📋 FULL SYSTEM TEST RESULTS")
    logger.info("=" * 60)
    
    if results.get('status') == 'FAILED':
        logger.info(f"❌ System test failed: {results.get('error', 'Unknown error')}")
    else:
        logger.info(f"🎯 System Status: {results.get('system_status', 'UNKNOWN')}")
        logger.info(f"📊 Overall Health: {results.get('overall_health', 0.0):.2f}")
        logger.info(f"⏱️ Test Duration: {results.get('test_duration', 0.0):.2f}s")
        
        health_breakdown = results.get('health_breakdown', {})
        logger.info("\n🩺 Health Breakdown:")
        for component, health in health_breakdown.items():
            if component != 'overall_health':
                status = "✅" if health > 0.7 else "⚠️" if health > 0.4 else "❌"
                logger.info(f"  {status} {component}: {health:.2f}")
    
    logger.info(f"\n📁 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main()) 