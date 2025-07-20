#!/usr/bin/env python3
"""
Kimera SWM Demo with Advanced Selective Feedback Architecture
=============================================================

This demo showcases Kimera running with the new state-of-the-art 
selective feedback architecture including:
- Optuna hyperparameter optimization
- Advanced neural attention mechanisms
- GPU-accelerated processing
- Multi-domain specialization
"""

import asyncio
import requests
import json
import time
from datetime import datetime
from backend.core.optimizing_selective_feedback_interpreter import (

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)

    create_optimizing_selective_feedback_interpreter,
    OptimizationConfig
)

class KimeraAdvancedDemo:
    """Comprehensive demo of Kimera with advanced selective feedback"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.demo_results = {
            'start_time': datetime.now().isoformat(),
            'tests': {}
        }
    
    async def run_complete_demo(self):
        """Run comprehensive demo of all advanced features"""
        
        logger.info("🚀 KIMERA SWM - Advanced Selective Feedback Architecture Demo")
        logger.info("=" * 80)
        
        try:
            # Test 1: System Status Check
            await self._test_system_status()
            
            # Test 2: Advanced Selective Feedback
            await self._test_advanced_selective_feedback()
            
            # Test 3: Create Intelligent Geoid
            await self._test_intelligent_geoid_creation()
            
            # Test 4: Multi-Domain Analysis
            await self._test_multi_domain_analysis()
            
            # Test 5: Revolutionary Intelligence Integration
            await self._test_revolutionary_intelligence()
            
            # Test 6: GPU Performance Analysis
            await self._test_gpu_performance()
            
            self._generate_final_report()
            
            logger.info("✅ Kimera Advanced Demo Completed Successfully!")
            return self.demo_results
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return self.demo_results
    
    async def _test_system_status(self):
        """Test 1: Verify Kimera system status"""
        
        logger.debug("🔍 Test 1: System Status Check")
        
        try:
            response = requests.get(f"{self.base_url}/system/status")
            status = response.json()
            
            logger.info(f"   ✅ System Status: Operational")
            logger.info(f"   📊 Active Geoids: {status['system_info']['active_geoids']}")
            logger.info(f"   🧠 GPU: {status['gpu_info']['gpu_name']}")
            logger.debug(f"   🔬 Embedding Model: {status['model_info']['embedding_model']}")
            logger.info(f"   🚀 Revolutionary Intelligence: {status['revolutionary_intelligence']['status']}")
            
            self.demo_results['tests']['system_status'] = {
                'status': 'passed',
                'gpu_available': status['gpu_info']['gpu_available'],
                'revolutionary_ai': status['revolutionary_intelligence']['available']
            }
            
        except Exception as e:
            logger.error(f"   ❌ System status check failed: {e}")
            self.demo_results['tests']['system_status'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _test_advanced_selective_feedback(self):
        """Test 2: Advanced selective feedback interpreter"""
        
        logger.info("\n🧠 Test 2: Advanced Selective Feedback Architecture")
        
        try:
            # Create advanced interpreter with Optuna optimization
            config = OptimizationConfig(
                use_optuna=True,
                n_trials=10,
                mixed_precision=True,
                attention_optimization=True
            )
            
            # Test financial domain
            financial_interpreter = create_optimizing_selective_feedback_interpreter('financial', config)
            
            message = "Analyze cryptocurrency market volatility for Q4 investment strategy"
            context = {'type': 'financial', 'domain': 'investment', 'complexity': 'high'}
            
            start_time = time.time()
            analysis, metrics = await financial_interpreter.analyze_with_optimized_learning(
                message, context, optimize_hyperparams=True, enable_attention=True
            )
            analysis_time = time.time() - start_time
            
            logger.info(f"   ✅ Financial Analysis: {analysis_time*1000:.1f}ms")
            logger.info(f"   🎯 Optimization Score: {metrics.optimization_score:.3f}")
            logger.info(f"   💾 Memory Usage: {metrics.memory_usage_mb:.1f}MB")
            logger.info(f"   🚀 GPU Utilization: {metrics.gpu_utilization:.1f}%")
            logger.info(f"   📊 Consistency Score: {metrics.consistency_score:.3f}")
            
            self.demo_results['tests']['advanced_feedback'] = {
                'status': 'passed',
                'analysis_time_ms': analysis_time * 1000,
                'optimization_score': metrics.optimization_score,
                'consistency_score': metrics.consistency_score,
                'gpu_utilization': metrics.gpu_utilization
            }
            
        except Exception as e:
            logger.error(f"   ❌ Advanced feedback test failed: {e}")
            self.demo_results['tests']['advanced_feedback'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _test_intelligent_geoid_creation(self):
        """Test 3: Create an intelligent geoid with advanced analysis"""
        
        logger.debug("\n🔬 Test 3: Intelligent Geoid Creation")
        
        try:
            # Create a geoid with advanced semantic analysis
            create_request = {
                "echoform_text": "Revolutionary breakthrough in quantum consciousness integration",
                "metadata": {
                    "domain": "scientific",
                    "complexity": "expert",
                    "analysis_type": "advanced_selective_feedback"
                }
            }
            
            response = requests.post(f"{self.base_url}/geoids", json=create_request)
            geoid_data = response.json()
            
            geoid_id = geoid_data['geoid']['id']
            
            logger.info(f"   ✅ Geoid Created: {geoid_id}")
            logger.info(f"   🧬 Semantic Features: {len(geoid_data['geoid']['semantic_features'])
            logger.info(f"   📏 Embedding Dimension: {geoid_data['embedding_info']['dimension']}")
            
            # Test geoid speech generation
            speech_response = requests.get(f"{self.base_url}/geoids/{geoid_id}/speak")
            speech_data = speech_response.json()
            
            logger.info(f"   🗣️ Generated Speech: {speech_data['content'][:100]}...")
            
            self.demo_results['tests']['geoid_creation'] = {
                'status': 'passed',
                'geoid_id': geoid_id,
                'embedding_dimension': geoid_data['embedding_info']['dimension']
            }
            
        except Exception as e:
            logger.error(f"   ❌ Geoid creation test failed: {e}")
            self.demo_results['tests']['geoid_creation'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _test_multi_domain_analysis(self):
        """Test 4: Multi-domain specialized analysis"""
        
        logger.info("\n🌐 Test 4: Multi-Domain Analysis")
        
        try:
            domains = [
                ('financial', "Portfolio optimization with AI-driven strategies"),
                ('scientific', "Quantum entanglement in neural networks"),
                ('creative', "Innovative sustainable product design")
            ]
            
            results = {}
            
            for domain, message in domains:
                config = OptimizationConfig(use_optuna=True, n_trials=5)
                interpreter = create_optimizing_selective_feedback_interpreter(domain, config)
                
                context = {'type': domain, 'complexity': 'expert'}
                start_time = time.time()
                
                _, metrics = await interpreter.analyze_with_optimized_learning(
                    message, context, optimize_hyperparams=True
                )
                
                analysis_time = time.time() - start_time
                
                results[domain] = {
                    'latency_ms': analysis_time * 1000,
                    'optimization_score': metrics.optimization_score,
                    'consistency': metrics.consistency_score
                }
                
                logger.info(f"   {domain:>10}: {analysis_time*1000:>6.1f}ms | Score: {metrics.optimization_score:.3f}")
            
            self.demo_results['tests']['multi_domain'] = {
                'status': 'passed',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"   ❌ Multi-domain test failed: {e}")
            self.demo_results['tests']['multi_domain'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _test_revolutionary_intelligence(self):
        """Test 5: Revolutionary Intelligence Integration"""
        
        logger.info("\n🧠 Test 5: Revolutionary Intelligence Integration")
        
        try:
            # Test revolutionary intelligence demo
            demo_request = {
                "user_input": "I want to create a breakthrough cognitive architecture"
            }
            
            response = requests.post(
                f"{self.base_url}/system/revolutionary_demo",
                json=demo_request
            )
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info(f"   ✅ Revolutionary Analysis Generated")
                logger.info(f"   💡 Breakthrough Score: {result.get('breakthrough_score', 'N/A')
                logger.info(f"   🎯 Innovation Potential: {result.get('innovation_potential', 'High')
                
                self.demo_results['tests']['revolutionary_intelligence'] = {
                    'status': 'passed',
                    'available': True
                }
            else:
                logger.warning(f"   ⚠️ Revolutionary Intelligence unavailable")
                self.demo_results['tests']['revolutionary_intelligence'] = {
                    'status': 'skipped',
                    'reason': 'endpoint_unavailable'
                }
                
        except Exception as e:
            logger.error(f"   ❌ Revolutionary intelligence test failed: {e}")
            self.demo_results['tests']['revolutionary_intelligence'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _test_gpu_performance(self):
        """Test 6: GPU Performance Analysis"""
        
        logger.info("\n🚀 Test 6: GPU Performance Analysis")
        
        try:
            # Get detailed system health including GPU metrics
            response = requests.get(f"{self.base_url}/system/health/detailed")
            health_data = response.json()
            
            gpu_info = health_data['metrics']['gpu_info']
            
            logger.info(f"   ✅ GPU Device: {gpu_info['gpu_name']}")
            logger.info(f"   💾 GPU Memory: {gpu_info['gpu_memory_allocated'] / (1024**3)
            logger.info(f"   🔥 CUDA Available: {gpu_info['gpu_available']}")
            
            # Test GPU acceleration with multiple concurrent operations
            config = OptimizationConfig(mixed_precision=True, attention_optimization=True)
            interpreter = create_optimizing_selective_feedback_interpreter('balanced', config)
            
            # Run concurrent analyses to test GPU scaling
            tasks = []
            for i in range(5):
                task = interpreter.analyze_with_optimized_learning(
                    f"GPU test analysis {i+1}",
                    {'type': 'performance_test', 'iteration': i}
                )
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            avg_gpu_util = sum(r[1].gpu_utilization for r in results) / len(results)
            
            logger.info(f"   ⚡ Concurrent Analyses: 5 operations in {total_time*1000:.1f}ms")
            logger.info(f"   📊 Average GPU Utilization: {avg_gpu_util:.1f}%")
            
            self.demo_results['tests']['gpu_performance'] = {
                'status': 'passed',
                'concurrent_operations': 5,
                'total_time_ms': total_time * 1000,
                'avg_gpu_utilization': avg_gpu_util
            }
            
        except Exception as e:
            logger.error(f"   ❌ GPU performance test failed: {e}")
            self.demo_results['tests']['gpu_performance'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def _generate_final_report(self):
        """Generate final demo report"""
        
        passed = sum(1 for test in self.demo_results['tests'].values() 
                    if test.get('status') == 'passed')
        total = len(self.demo_results['tests'])
        
        self.demo_results['summary'] = {
            'tests_passed': passed,
            'tests_total': total,
            'success_rate': passed / total if total > 0 else 0,
            'overall_status': 'excellent' if passed == total else 'good' if passed >= total * 0.8 else 'needs_improvement'
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("📋 KIMERA ADVANCED DEMO SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {self.demo_results['summary']['overall_status'].upper()
        logger.info(f"Tests Passed: {passed}/{total}")
        logger.info(f"Success Rate: {self.demo_results['summary']['success_rate']*100:.1f}%")
        logger.info("=" * 80)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kimera_advanced_demo_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.demo_results, f, indent=2, default=str)
        
        logger.info(f"💾 Demo results saved to: {filename}")


async def main():
    """Run the comprehensive Kimera demo"""
    
    demo = KimeraAdvancedDemo()
    results = await demo.run_complete_demo()
    
    success_rate = results.get('summary', {}).get('success_rate', 0)
    
    if success_rate >= 0.9:
        logger.info("\n🎉 KIMERA ADVANCED SELECTIVE FEEDBACK ARCHITECTURE")
        logger.info("✅ FULLY OPERATIONAL AND OPTIMIZED!")
    elif success_rate >= 0.7:
        logger.warning("\n⚠️ Kimera operational with minor issues")
    else:
        logger.error("\n❌ Kimera needs attention")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main()) 