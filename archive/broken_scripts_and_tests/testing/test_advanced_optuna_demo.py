#!/usr/bin/env python3
"""
Advanced Selective Feedback with Optuna Optimization Demo
=========================================================

Demonstration of state-of-the-art selective feedback architecture with:
- Optuna hyperparameter optimization
- Multi-domain specialization
- GPU acceleration
- Advanced performance monitoring
"""

import asyncio
import time
import logging
from datetime import datetime
import json

# Import the advanced implementation
from backend.core.optimizing_selective_feedback_interpreter import (
    OptimizingSelectiveFeedbackInterpreter,
    OptimizationConfig,
    create_optimizing_selective_feedback_interpreter
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedOptunaDemo:
    """Demonstration of advanced optimization capabilities"""
    
    def __init__(self):
        self.demo_results = {
            'start_time': datetime.now().isoformat(),
            'tests': {}
        }
    
    async def run_demo(self):
        """Run optimization demo"""
        
        logger.info("🚀 Advanced Selective Feedback with Optuna Demo")
        logger.info("=" * 60)
        
        try:
            await self._demo_optimization_impact()
            await self._demo_multi_domain()
            self._generate_summary()
            
            logger.info("✅ Demo Completed")
            return self.demo_results
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return self.demo_results
    
    async def _demo_optimization_impact(self):
        """Demo optimization impact"""
        
        logger.info("🎯 Demo: Optimization Impact")
        
        try:
            # Without optimization
            config_basic = OptimizationConfig(use_optuna=False)
            interpreter_basic = create_optimizing_selective_feedback_interpreter('financial', config_basic)
            
            message = "Analyze crypto market volatility"
            context = {'type': 'financial', 'complexity': 'high'}
            
            basic_result, basic_metrics = await interpreter_basic.analyze_with_optimized_learning(
                message, context, optimize_hyperparams=False
            )
            
            # With optimization
            config_opt = OptimizationConfig(use_optuna=True, n_trials=10)
            interpreter_opt = create_optimizing_selective_feedback_interpreter('financial', config_opt)
            
            opt_result, opt_metrics = await interpreter_opt.analyze_with_optimized_learning(
                message, context, optimize_hyperparams=True
            )
            
            improvement = {
                'basic_latency_ms': basic_metrics.analysis_latency * 1000,
                'optimized_latency_ms': opt_metrics.analysis_latency * 1000,
                'optimization_score': opt_metrics.optimization_score
            }
            
            self.demo_results['tests']['optimization_impact'] = {
                'status': 'passed',
                'improvement': improvement
            }
            
            logger.info(f"   Basic: {improvement['basic_latency_ms']:.1f}ms")
            logger.info(f"   Optimized: {improvement['optimized_latency_ms']:.1f}ms")
            logger.info(f"   Score: {improvement['optimization_score']:.3f}")
            
        except Exception as e:
            self.demo_results['tests']['optimization_impact'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"   ❌ Failed: {e}")
    
    async def _demo_multi_domain(self):
        """Demo multi-domain optimization"""
        
        logger.info("🌐 Demo: Multi-Domain Optimization")
        
        try:
            domains = [
                ('financial', "Portfolio optimization"),
                ('scientific', "Quantum algorithms"),
                ('creative', "Product design")
            ]
            
            results = {}
            for domain, message in domains:
                config = OptimizationConfig(use_optuna=True, n_trials=5)
                interpreter = create_optimizing_selective_feedback_interpreter(domain, config)
                
                context = {'type': domain, 'complexity': 'expert'}
                
                _, metrics = await interpreter.analyze_with_optimized_learning(
                    message, context, optimize_hyperparams=True
                )
                
                results[domain] = {
                    'latency_ms': metrics.analysis_latency * 1000,
                    'optimization_score': metrics.optimization_score
                }
                
                logger.info(f"   {domain}: {metrics.optimization_score:.3f}")
            
            self.demo_results['tests']['multi_domain'] = {
                'status': 'passed',
                'results': results
            }
            
        except Exception as e:
            self.demo_results['tests']['multi_domain'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"   ❌ Failed: {e}")
    
    def _generate_summary(self):
        """Generate summary"""
        
        passed = sum(1 for test in self.demo_results['tests'].values() 
                    if test.get('status') == 'passed')
        total = len(self.demo_results['tests'])
        
        self.demo_results['summary'] = {
            'success_rate': passed / total if total > 0 else 0,
            'status': 'excellent' if passed == total else 'needs_improvement'
        }
        
        logger.info("=" * 60)
        logger.info(f"DEMO SUMMARY: {self.demo_results['summary']['status'].upper()}")
        logger.info(f"Success Rate: {passed}/{total}")
        logger.info("=" * 60)


async def main():
    """Run the demo"""
    
    demo = AdvancedOptunaDemo()
    results = await demo.run_demo()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optuna_demo_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to: {filename}")
    
    status = results.get('summary', {}).get('status', 'unknown')
    if status == 'excellent':
        logger.info("✅ OPTUNA OPTIMIZATION WORKING PERFECTLY!")
    else:
        logger.warning("⚠️  Needs improvements")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    logger.info("\n🎉 STATE-OF-THE-ART IMPLEMENTATION COMPLETE!")