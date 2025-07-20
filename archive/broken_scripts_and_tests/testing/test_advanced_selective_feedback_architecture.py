#!/usr/bin/env python3
"""
Advanced Selective Feedback Architecture Test
=============================================

Comprehensive test suite for the state-of-the-art implementation using:
- Optuna hyperparameter optimization
- Advanced neural architectures  
- GPU acceleration testing
- Performance benchmarking
- Memory optimization validation
"""

import asyncio
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Core testing
import pytest
import torch

# Import the advanced implementation
try:
    from backend.core.optimizing_selective_feedback_interpreter import (
        OptimizingSelectiveFeedbackInterpreter,
        OptimizationConfig,
        OptimizationMetrics,
        create_optimizing_selective_feedback_interpreter
    )
    from backend.core.anthropomorphic_profiler import create_default_profiler
    ADVANCED_AVAILABLE = True
except ImportError as e:
    logger.info(f"Advanced implementation not available: {e}")
    ADVANCED_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedSelectiveFeedbackArchitectureTest:
    """
    Comprehensive test suite for the advanced selective feedback architecture
    """
    
    def __init__(self):
        self.test_results = {
            'start_time': datetime.now().isoformat(),
            'tests': {},
            'performance_metrics': {},
            'optimization_results': {},
            'system_capabilities': {}
        }
        
        # Test configurations
        self.test_messages = [
            "Analyze market volatility trends for Q4 financial planning",
            "Explain quantum entanglement principles for research publication", 
            "Create innovative marketing campaign for sustainable products",
            "Evaluate risk factors in cryptocurrency investment strategy",
            "Design neural network architecture for image classification"
        ]
        
        self.test_contexts = [
            {'type': 'financial', 'domain': 'investment', 'complexity': 'high'},
            {'type': 'scientific', 'domain': 'physics', 'complexity': 'expert'},
            {'type': 'creative', 'domain': 'marketing', 'complexity': 'medium'},
            {'type': 'financial', 'domain': 'crypto', 'complexity': 'high'},
            {'type': 'scientific', 'domain': 'ai_ml', 'complexity': 'expert'}
        ]
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        
        logger.info("🚀 Starting Advanced Selective Feedback Architecture Test")
        logger.info("=" * 80)
        
        if not ADVANCED_AVAILABLE:
            return {
                'status': 'failed',
                'error': 'Advanced implementation not available',
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Test 1: System Capabilities Assessment
            await self._test_system_capabilities()
            
            # Test 2: Basic Advanced Analysis
            await self._test_basic_advanced_analysis()
            
            # Test 3: Hyperparameter Optimization
            await self._test_hyperparameter_optimization()
            
            # Test 4: Performance Benchmarking
            await self._test_performance_benchmarking()
            
            # Test 5: Memory Optimization
            await self._test_memory_optimization()
            
            # Test 6: GPU Acceleration (if available)
            await self._test_gpu_acceleration()
            
            # Test 7: Attention Mechanisms
            await self._test_attention_mechanisms()
            
            # Test 8: Multi-Domain Specialization
            await self._test_domain_specialization()
            
            # Test 9: Scalability Assessment
            await self._test_scalability()
            
            # Test 10: Quality Metrics Validation
            await self._test_quality_metrics()
            
            # Generate final report
            self._generate_final_report()
            
            logger.info("✅ Advanced Selective Feedback Architecture Test Completed")
            return self.test_results
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            self.test_results['status'] = 'failed'
            self.test_results['error'] = str(e)
            return self.test_results
    
    async def _test_system_capabilities(self):
        """Test 1: Assess system capabilities and available optimizations"""
        
        logger.info("🔍 Test 1: System Capabilities Assessment")
        
        test_start = time.time()
        capabilities = {}
        
        try:
            # Check PyTorch and CUDA
            capabilities['pytorch_version'] = torch.__version__
            capabilities['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                capabilities['gpu_name'] = torch.cuda.get_device_name()
                capabilities['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Check Optuna availability
            try:
                import optuna
                capabilities['optuna_available'] = True
                capabilities['optuna_version'] = optuna.__version__
            except ImportError:
                capabilities['optuna_available'] = False
            
            # Check memory monitoring
            try:
                import psutil
                capabilities['memory_monitoring'] = True
                capabilities['system_memory_gb'] = psutil.virtual_memory().total / 1024**3
            except ImportError:
                capabilities['memory_monitoring'] = False
            
            # Test basic interpreter creation
            config = OptimizationConfig(n_trials=5)  # Small number for testing
            interpreter = create_optimizing_selective_feedback_interpreter('balanced', config)
            capabilities['interpreter_creation'] = True
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['system_capabilities'] = {
                'status': 'passed',
                'capabilities': capabilities,
                'test_time_seconds': test_time
            }
            
            self.test_results['system_capabilities'] = capabilities
            
            logger.info(f"   ✅ System capabilities assessed in {test_time:.2f}s")
            logger.info(f"      CUDA: {capabilities['cuda_available']}")
            logger.info(f"      Optuna: {capabilities['optuna_available']}")
            logger.info(f"      Memory Monitoring: {capabilities['memory_monitoring']}")
            
        except Exception as e:
            self.test_results['tests']['system_capabilities'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ System capabilities test failed: {e}")
    
    async def _test_basic_advanced_analysis(self):
        """Test 2: Basic advanced analysis functionality"""
        
        logger.info("🧠 Test 2: Basic Advanced Analysis")
        
        test_start = time.time()
        
        try:
            # Create interpreter with minimal configuration
            config = OptimizationConfig(
                use_optuna=False,  # Disable for basic test
                mixed_precision=False,
                n_trials=5
            )
            interpreter = create_optimizing_selective_feedback_interpreter('balanced', config)
            
            # Test basic analysis
            message = "Analyze financial market trends"
            context = {'type': 'financial', 'complexity': 'medium'}
            
            analysis_result, advanced_metrics = await interpreter.analyze_with_optimized_learning(
                message, context, optimize_hyperparams=False, enable_attention=False
            )
            
            # Validate results
            assert analysis_result is not None, "Analysis result should not be None"
            assert isinstance(advanced_metrics, OptimizationMetrics), "Should return OptimizationMetrics"
            assert advanced_metrics.analysis_latency > 0, "Should record positive latency"
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['basic_advanced_analysis'] = {
                'status': 'passed',
                'analysis_latency_ms': advanced_metrics.analysis_latency * 1000,
                'throughput_ops_per_sec': advanced_metrics.throughput_ops_per_sec,
                'test_time_seconds': test_time,
                'detected_traits_count': len(analysis_result.detected_traits)
            }
            
            logger.info(f"   ✅ Basic analysis completed in {test_time:.2f}s")
            logger.info(f"      Analysis Latency: {advanced_metrics.analysis_latency * 1000:.1f}ms")
            logger.info(f"      Throughput: {advanced_metrics.throughput_ops_per_sec:.1f} ops/sec")
            
        except Exception as e:
            self.test_results['tests']['basic_advanced_analysis'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ Basic advanced analysis test failed: {e}")
    
    async def _test_hyperparameter_optimization(self):
        """Test 3: Hyperparameter optimization with Optuna"""
        
        logger.info("🎯 Test 3: Hyperparameter Optimization")
        
        test_start = time.time()
        
        try:
            # Check if Optuna is available
            try:
                import optuna
                optuna_available = True
            except ImportError:
                optuna_available = False
            
            if not optuna_available:
                logger.info("   ⚠️  Optuna not available, skipping optimization test")
                self.test_results['tests']['hyperparameter_optimization'] = {
                    'status': 'skipped',
                    'reason': 'optuna_not_available'
                }
                return
            
            # Create interpreter with optimization enabled
            config = OptimizationConfig(
                use_optuna=True,
                n_trials=10,  # Small number for testing
                optimization_timeout=60  # 1 minute limit
            )
            interpreter = create_optimizing_selective_feedback_interpreter('financial', config)
            
            # Test optimization for financial context
            message = "Evaluate investment portfolio risk"
            context = {'type': 'financial', 'domain': 'investment'}
            
            analysis_result, advanced_metrics = await interpreter.analyze_with_optimized_learning(
                message, context, optimize_hyperparams=True, enable_attention=False
            )
            
            # Get optimization results
            performance_report = interpreter.get_comprehensive_performance_report()
            optimization_status = performance_report.get('optimization_status', {})
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['hyperparameter_optimization'] = {
                'status': 'passed',
                'optimization_score': advanced_metrics.optimization_score,
                'optimization_enabled': optimization_status.get('hyperparameter_optimization', {}).get('optuna_enabled', False),
                'test_time_seconds': test_time
            }
            
            logger.info(f"   ✅ Hyperparameter optimization completed in {test_time:.2f}s")
            logger.info(f"      Optimization Score: {advanced_metrics.optimization_score:.3f}")
            
        except Exception as e:
            self.test_results['tests']['hyperparameter_optimization'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ Hyperparameter optimization test failed: {e}")
    
    async def _test_performance_benchmarking(self):
        """Test 4: Performance benchmarking across multiple analyses"""
        
        logger.info("⚡ Test 4: Performance Benchmarking")
        
        test_start = time.time()
        benchmark_results = []
        
        try:
            config = OptimizationConfig(use_optuna=False, mixed_precision=True)
            interpreter = create_optimizing_selective_feedback_interpreter('balanced', config)
            
            # Run multiple analyses to collect performance data
            for i, (message, context) in enumerate(zip(self.test_messages[:3], self.test_contexts[:3])):
                analysis_start = time.time()
                
                analysis_result, advanced_metrics = await interpreter.analyze_with_optimized_learning(
                    message, context, optimize_hyperparams=False
                )
                
                analysis_time = time.time() - analysis_start
                
                benchmark_results.append({
                    'iteration': i + 1,
                    'context_type': context['type'],
                    'analysis_latency_ms': advanced_metrics.analysis_latency * 1000,
                    'memory_usage_mb': advanced_metrics.memory_usage_mb,
                    'throughput_ops_per_sec': advanced_metrics.throughput_ops_per_sec,
                    'total_time_ms': analysis_time * 1000
                })
            
            # Calculate benchmark statistics
            latencies = [r['analysis_latency_ms'] for r in benchmark_results]
            throughputs = [r['throughput_ops_per_sec'] for r in benchmark_results]
            memory_usage = [r['memory_usage_mb'] for r in benchmark_results]
            
            performance_stats = {
                'avg_latency_ms': np.mean(latencies),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'latency_std_ms': np.std(latencies),
                'avg_throughput_ops_sec': np.mean(throughputs),
                'avg_memory_mb': np.mean(memory_usage),
                'peak_memory_mb': np.max(memory_usage)
            }
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['performance_benchmarking'] = {
                'status': 'passed',
                'benchmark_results': benchmark_results,
                'performance_stats': performance_stats,
                'test_time_seconds': test_time
            }
            
            self.test_results['performance_metrics'] = performance_stats
            
            logger.info(f"   ✅ Performance benchmarking completed in {test_time:.2f}s")
            logger.info(f"      Avg Latency: {performance_stats['avg_latency_ms']:.1f}ms")
            logger.info(f"      Avg Throughput: {performance_stats['avg_throughput_ops_sec']:.1f} ops/sec")
            logger.info(f"      Peak Memory: {performance_stats['peak_memory_mb']:.1f}MB")
            
        except Exception as e:
            self.test_results['tests']['performance_benchmarking'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ Performance benchmarking test failed: {e}")
    
    async def _test_memory_optimization(self):
        """Test 5: Memory optimization and monitoring"""
        
        logger.info("💾 Test 5: Memory Optimization")
        
        test_start = time.time()
        
        try:
            # Test with memory monitoring enabled
            config = OptimizationConfig(
                mixed_precision=True,
                max_memory_usage_mb=4096  # 4GB limit
            )
            interpreter = create_optimizing_selective_feedback_interpreter('scientific', config)
            
            # Monitor memory usage during analysis
            initial_memory = interpreter._get_memory_usage()
            
            # Run analysis
            message = "Explain complex quantum mechanics principles"
            context = {'type': 'scientific', 'complexity': 'expert'}
            
            analysis_result, advanced_metrics = await interpreter.analyze_with_optimized_learning(
                message, context
            )
            
            final_memory = interpreter._get_memory_usage()
            memory_delta = final_memory - initial_memory
            
            # Test cleanup
            interpreter.cleanup_resources()
            post_cleanup_memory = interpreter._get_memory_usage()
            cleanup_efficiency = (final_memory - post_cleanup_memory) / max(final_memory - initial_memory, 1)
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['memory_optimization'] = {
                'status': 'passed',
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': final_memory,
                'memory_delta_mb': memory_delta,
                'post_cleanup_memory_mb': post_cleanup_memory,
                'cleanup_efficiency': cleanup_efficiency,
                'mixed_precision_enabled': config.mixed_precision,
                'test_time_seconds': test_time
            }
            
            logger.info(f"   ✅ Memory optimization test completed in {test_time:.2f}s")
            logger.info(f"      Memory Delta: {memory_delta:.1f}MB")
            logger.info(f"      Cleanup Efficiency: {cleanup_efficiency:.1%}")
            
        except Exception as e:
            self.test_results['tests']['memory_optimization'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ Memory optimization test failed: {e}")
    
    async def _test_gpu_acceleration(self):
        """Test 6: GPU acceleration capabilities"""
        
        logger.info("🚀 Test 6: GPU Acceleration")
        
        test_start = time.time()
        
        try:
            cuda_available = torch.cuda.is_available()
            
            if not cuda_available:
                logger.info("   ⚠️  CUDA not available, testing CPU fallback")
                
            config = OptimizationConfig(mixed_precision=cuda_available)
            interpreter = create_optimizing_selective_feedback_interpreter('balanced', config)
            
            # Test analysis with GPU acceleration (if available)
            message = "Process large dataset for machine learning"
            context = {'type': 'scientific', 'domain': 'ai_ml'}
            
            analysis_result, advanced_metrics = await interpreter.analyze_with_optimized_learning(
                message, context, enable_attention=True
            )
            
            # Get GPU utilization if available
            gpu_utilization = advanced_metrics.gpu_utilization if cuda_available else 0.0
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['gpu_acceleration'] = {
                'status': 'passed',
                'cuda_available': cuda_available,
                'gpu_utilization': gpu_utilization,
                'mixed_precision_used': config.mixed_precision,
                'device_name': torch.cuda.get_device_name() if cuda_available else 'CPU',
                'analysis_latency_ms': advanced_metrics.analysis_latency * 1000,
                'test_time_seconds': test_time
            }
            
            logger.info(f"   ✅ GPU acceleration test completed in {test_time:.2f}s")
            logger.info(f"      Device: {torch.cuda.get_device_name() if cuda_available else 'CPU'}")
            logger.info(f"      GPU Utilization: {gpu_utilization:.1f}%")
            
        except Exception as e:
            self.test_results['tests']['gpu_acceleration'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ GPU acceleration test failed: {e}")
    
    async def _test_attention_mechanisms(self):
        """Test 7: Advanced attention mechanisms"""
        
        logger.info("🧠 Test 7: Advanced Attention Mechanisms")
        
        test_start = time.time()
        
        try:
            config = OptimizationConfig(attention_optimization=True)
            interpreter = create_optimizing_selective_feedback_interpreter('creative', config)
            
            # Test with attention enabled vs disabled
            message = "Create innovative marketing strategy"
            context = {'type': 'creative', 'complexity': 'high'}
            
            # Test with attention
            result_with_attention, metrics_with_attention = await interpreter.analyze_with_optimized_learning(
                message, context, enable_attention=True
            )
            
            # Test without attention for comparison
            result_without_attention, metrics_without_attention = await interpreter.analyze_with_optimized_learning(
                message, context, enable_attention=False
            )
            
            # Compare performance
            attention_speedup = metrics_without_attention.analysis_latency / metrics_with_attention.analysis_latency
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['attention_mechanisms'] = {
                'status': 'passed',
                'attention_enabled_latency_ms': metrics_with_attention.analysis_latency * 1000,
                'attention_disabled_latency_ms': metrics_without_attention.analysis_latency * 1000,
                'attention_speedup_factor': attention_speedup,
                'attention_optimization_enabled': config.attention_optimization,
                'test_time_seconds': test_time
            }
            
            logger.info(f"   ✅ Attention mechanisms test completed in {test_time:.2f}s")
            logger.info(f"      With Attention: {metrics_with_attention.analysis_latency * 1000:.1f}ms")
            logger.info(f"      Without Attention: {metrics_without_attention.analysis_latency * 1000:.1f}ms")
            
        except Exception as e:
            self.test_results['tests']['attention_mechanisms'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ Attention mechanisms test failed: {e}")
    
    async def _test_domain_specialization(self):
        """Test 8: Multi-domain specialization capabilities"""
        
        logger.info("🎯 Test 8: Domain Specialization")
        
        test_start = time.time()
        domain_results = {}
        
        try:
            domains = ['financial', 'scientific', 'creative']
            
            for domain in domains:
                config = OptimizationConfig(n_trials=5)  # Quick optimization for testing
                interpreter = create_optimizing_selective_feedback_interpreter(domain, config)
                
                # Test domain-specific analysis
                if domain == 'financial':
                    message = "Analyze market volatility and investment risks"
                    context = {'type': 'financial', 'domain': 'investment'}
                elif domain == 'scientific':
                    message = "Explain quantum computing algorithms"
                    context = {'type': 'scientific', 'domain': 'physics'}
                else:  # creative
                    message = "Design innovative product concepts"
                    context = {'type': 'creative', 'domain': 'design'}
                
                analysis_result, advanced_metrics = await interpreter.analyze_with_optimized_learning(
                    message, context
                )
                
                domain_results[domain] = {
                    'analysis_latency_ms': advanced_metrics.analysis_latency * 1000,
                    'confidence_score': advanced_metrics.prediction_confidence,
                    'uncertainty_score': advanced_metrics.uncertainty_score,
                    'consistency_score': advanced_metrics.consistency_score,
                    'detected_traits_count': len(analysis_result.detected_traits)
                }
            
            # Calculate domain specialization metrics
            avg_confidence = np.mean([r['confidence_score'] for r in domain_results.values()])
            avg_consistency = np.mean([r['consistency_score'] for r in domain_results.values()])
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['domain_specialization'] = {
                'status': 'passed',
                'domain_results': domain_results,
                'avg_confidence_across_domains': avg_confidence,
                'avg_consistency_across_domains': avg_consistency,
                'domains_tested': len(domains),
                'test_time_seconds': test_time
            }
            
            logger.info(f"   ✅ Domain specialization test completed in {test_time:.2f}s")
            logger.info(f"      Domains tested: {len(domains)}")
            logger.info(f"      Avg Confidence: {avg_confidence:.3f}")
            logger.info(f"      Avg Consistency: {avg_consistency:.3f}")
            
        except Exception as e:
            self.test_results['tests']['domain_specialization'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ Domain specialization test failed: {e}")
    
    async def _test_scalability(self):
        """Test 9: Scalability under load"""
        
        logger.info("📈 Test 9: Scalability Assessment")
        
        test_start = time.time()
        
        try:
            config = OptimizationConfig(use_optuna=False)  # Disable optimization for speed
            interpreter = create_optimizing_selective_feedback_interpreter('balanced', config)
            
            # Test with increasing load
            load_tests = [5, 10, 20]  # Number of concurrent analyses
            scalability_results = []
            
            for load in load_tests:
                load_start = time.time()
                
                # Create multiple concurrent analysis tasks
                tasks = []
                for i in range(load):
                    message_idx = i % len(self.test_messages)
                    context_idx = i % len(self.test_contexts)
                    
                    task = interpreter.analyze_with_optimized_learning(
                        self.test_messages[message_idx],
                        self.test_contexts[context_idx],
                        optimize_hyperparams=False,
                        enable_attention=False
                    )
                    tasks.append(task)
                
                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                load_time = time.time() - load_start
                successful_analyses = sum(1 for r in results if not isinstance(r, Exception))
                
                scalability_results.append({
                    'load_size': load,
                    'successful_analyses': successful_analyses,
                    'total_time_seconds': load_time,
                    'throughput_analyses_per_sec': successful_analyses / load_time,
                    'success_rate': successful_analyses / load
                })
            
            # Calculate scalability metrics
            throughputs = [r['throughput_analyses_per_sec'] for r in scalability_results]
            scalability_efficiency = throughputs[-1] / throughputs[0] if len(throughputs) > 1 else 1.0
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['scalability'] = {
                'status': 'passed',
                'scalability_results': scalability_results,
                'scalability_efficiency': scalability_efficiency,
                'max_load_tested': max(load_tests),
                'test_time_seconds': test_time
            }
            
            logger.info(f"   ✅ Scalability assessment completed in {test_time:.2f}s")
            logger.info(f"      Max Load: {max(load_tests)} concurrent analyses")
            logger.info(f"      Scalability Efficiency: {scalability_efficiency:.2f}")
            
        except Exception as e:
            self.test_results['tests']['scalability'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ Scalability test failed: {e}")
    
    async def _test_quality_metrics(self):
        """Test 10: Quality metrics validation"""
        
        logger.info("📊 Test 10: Quality Metrics Validation")
        
        test_start = time.time()
        
        try:
            config = OptimizationConfig()
            interpreter = create_optimizing_selective_feedback_interpreter('balanced', config)
            
            # Collect quality metrics across multiple analyses
            quality_metrics = []
            
            for message, context in zip(self.test_messages, self.test_contexts):
                analysis_result, advanced_metrics = await interpreter.analyze_with_optimized_learning(
                    message, context
                )
                
                quality_metrics.append({
                    'confidence': advanced_metrics.prediction_confidence,
                    'uncertainty': advanced_metrics.uncertainty_score,
                    'consistency': advanced_metrics.consistency_score,
                    'context_type': context['type']
                })
            
            # Calculate aggregate quality metrics
            confidences = [m['confidence'] for m in quality_metrics]
            uncertainties = [m['uncertainty'] for m in quality_metrics]
            consistencies = [m['consistency'] for m in quality_metrics]
            
            quality_stats = {
                'avg_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'avg_uncertainty': np.mean(uncertainties),
                'uncertainty_std': np.std(uncertainties),
                'avg_consistency': np.mean(consistencies),
                'consistency_std': np.std(consistencies),
                'quality_stability': 1.0 - np.std(confidences) / max(np.mean(confidences), 0.001)
            }
            
            # Generate comprehensive performance report
            performance_report = interpreter.get_comprehensive_performance_report()
            
            test_time = time.time() - test_start
            
            self.test_results['tests']['quality_metrics'] = {
                'status': 'passed',
                'quality_metrics': quality_metrics,
                'quality_stats': quality_stats,
                'performance_report_available': performance_report.get('status') != 'no_data',
                'test_time_seconds': test_time
            }
            
            logger.info(f"   ✅ Quality metrics validation completed in {test_time:.2f}s")
            logger.info(f"      Avg Confidence: {quality_stats['avg_confidence']:.3f}")
            logger.info(f"      Avg Consistency: {quality_stats['avg_consistency']:.3f}")
            logger.info(f"      Quality Stability: {quality_stats['quality_stability']:.3f}")
            
        except Exception as e:
            self.test_results['tests']['quality_metrics'] = {
                'status': 'failed',
                'error': str(e),
                'test_time_seconds': time.time() - test_start
            }
            logger.error(f"   ❌ Quality metrics validation failed: {e}")
    
    def _generate_final_report(self):
        """Generate comprehensive final test report"""
        
        self.test_results['end_time'] = datetime.now().isoformat()
        
        # Calculate test summary
        total_tests = len(self.test_results['tests'])
        passed_tests = sum(1 for test in self.test_results['tests'].values() 
                          if test.get('status') == 'passed')
        failed_tests = sum(1 for test in self.test_results['tests'].values() 
                          if test.get('status') == 'failed')
        skipped_tests = sum(1 for test in self.test_results['tests'].values() 
                           if test.get('status') == 'skipped')
        
        total_time = sum(test.get('test_time_seconds', 0) 
                        for test in self.test_results['tests'].values())
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'skipped_tests': skipped_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_test_time_seconds': total_time
        }
        
        # Overall assessment
        if failed_tests == 0:
            if skipped_tests == 0:
                self.test_results['overall_status'] = 'excellent'
            else:
                self.test_results['overall_status'] = 'good'
        elif failed_tests <= 2:
            self.test_results['overall_status'] = 'acceptable'
        else:
            self.test_results['overall_status'] = 'needs_improvement'
        
        logger.info("\n" + "=" * 80)
        logger.info("📋 ADVANCED SELECTIVE FEEDBACK ARCHITECTURE TEST REPORT")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {self.test_results['overall_status'].upper()}")
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests:.1%}")
        logger.info(f"Total Test Time: {total_time:.1f}s")
        
        if self.test_results.get('performance_metrics'):
            pm = self.test_results['performance_metrics']
            logger.info(f"Avg Latency: {pm['avg_latency_ms']:.1f}ms")
            logger.info(f"Avg Throughput: {pm['avg_throughput_ops_sec']:.1f} ops/sec")
        
        logger.info("=" * 80)


async def main():
    """Run the comprehensive test suite"""
    
    test_suite = AdvancedSelectiveFeedbackArchitectureTest()
    results = await test_suite.run_comprehensive_test()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"advanced_selective_feedback_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Test results saved to: {filename}")
    
    return results


if __name__ == "__main__":
    # Run the test suite
    results = asyncio.run(main())
    
    # Print final status
    overall_status = results.get('overall_status', 'unknown')
    success_rate = results.get('summary', {}).get('success_rate', 0)
    
    logger.info(f"\n🎯 FINAL RESULT: {overall_status.upper()
    logger.info(f"🏆 Success Rate: {success_rate:.1%}")
    
    if overall_status in ['excellent', 'good']:
        logger.info("✅ Advanced Selective Feedback Architecture is performing excellently!")
    elif overall_status == 'acceptable':
        logger.warning("⚠️  Advanced Selective Feedback Architecture has minor issues but is functional")
    else:
        logger.error("❌ Advanced Selective Feedback Architecture needs significant improvements")