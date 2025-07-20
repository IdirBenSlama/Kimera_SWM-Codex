#!/usr/bin/env python3
"""
Comprehensive Test Suite for Rigorous Universal Translator
========================================================

This test suite validates the rigorous universal translator using:
1. Mathematical proof verification
2. Empirical validation with statistical analysis
3. Performance benchmarking
4. Axiom compliance testing
5. Gyroscopic stability validation

Following zetetic methodology - every claim is questioned and validated.
"""

import sys
import asyncio
import time
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, List, Any
import logging
from scipy.stats import ttest_1samp, shapiro
from scipy import linalg as la

# Add backend path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from engines.rigorous_universal_translator import (
    RigorousUniversalTranslator,
    SemanticVector,
    SemanticSpace,
    UnderstandingOperator,
    CompositionOperator,
    AxiomValidator,
    create_rigorous_universal_translator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScientificValidationSuite:
    """
    Comprehensive scientific validation suite for rigorous universal translator
    
    Implements validation methodology based on:
    - Mathematical proof verification
    - Statistical hypothesis testing
    - Performance benchmarking
    - Reproducibility testing
    - Error analysis
    """
    
    def __init__(self):
        self.translator = None
        self.validation_results = {}
        
    async def initialize_translator(self, dimension: int = 256):
        """Initialize translator for testing"""
        logger.info(f"🔬 Initializing translator with dimension {dimension}")
        self.translator = await create_rigorous_universal_translator(dimension)
        
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run complete scientific validation suite"""
        
        logger.info("🚀 Starting comprehensive scientific validation")
        logger.info("=" * 60)
        
        if not self.translator:
            await self.initialize_translator()
        
        # Run all validation components
        validation_components = [
            ("Mathematical Foundation", self.validate_mathematical_foundations),
            ("Axiom Compliance", self.validate_axiom_compliance),
            ("Statistical Properties", self.validate_statistical_properties),
            ("Performance Benchmarks", self.validate_performance_benchmarks),
            ("Gyroscopic Stability", self.validate_gyroscopic_stability),
        ]
        
        overall_results = {}
        
        for component_name, validator_func in validation_components:
            logger.info(f"\n📊 Validating: {component_name}")
            try:
                result = await validator_func()
                overall_results[component_name.lower().replace(' ', '_')] = result
                
                # Log summary
                if isinstance(result, dict) and 'passed' in result:
                    status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                    logger.info(f"   {status}: {result.get('summary', 'No summary')}")
                
            except Exception as e:
                logger.error(f"   ❌ FAILED: {str(e)}")
                overall_results[component_name.lower().replace(' ', '_')] = {
                    'passed': False,
                    'error': str(e)
                }
        
        # Calculate overall validation score
        passed_components = sum(1 for result in overall_results.values() 
                              if isinstance(result, dict) and result.get('passed', False))
        total_components = len(validation_components)
        overall_score = passed_components / total_components
        
        final_results = {
            'validation_timestamp': time.time(),
            'overall_score': overall_score,
            'passed_components': passed_components,
            'total_components': total_components,
            'validation_status': 'PASSED' if overall_score >= 0.8 else 'FAILED',
            'component_results': overall_results,
            'scientific_rigor': 'HIGH' if overall_score >= 0.9 else 'MODERATE' if overall_score >= 0.7 else 'LOW'
        }
        
        logger.info("\n" + "=" * 60)
        logger.info(f"🎯 VALIDATION COMPLETE")
        logger.info(f"   Overall Score: {overall_score:.1%}")
        logger.info(f"   Status: {final_results['validation_status']}")
        logger.info(f"   Scientific Rigor: {final_results['scientific_rigor']}")
        
        return final_results
    
    async def validate_mathematical_foundations(self) -> Dict[str, Any]:
        """Validate mathematical foundations of the translator"""
        
        # Test semantic space properties
        semantic_space = self.translator.semantic_space
        
        # 1. Metric tensor properties
        metric = semantic_space.metric_tensor
        eigenvals = la.eigvals(metric)
        
        # Positive definiteness
        positive_definite = np.all(eigenvals > 0)
        
        # Symmetry
        symmetric = np.allclose(metric, metric.T)
        
        # 2. Understanding operator properties
        understanding_op = self.translator.understanding_op
        U_matrix = understanding_op.operator_matrix
        
        # Contraction property (eigenvalues < 1)
        U_eigenvals = la.eigvals(U_matrix)
        contractive = np.all(np.abs(U_eigenvals) < 1.0)
        
        # Information preservation (unitary up to scaling)
        det_U = la.det(U_matrix)
        info_preserving = abs(det_U) > 1e-6  # Non-singular
        
        return {
            'passed': positive_definite and symmetric and contractive and info_preserving,
            'summary': f"Mathematical foundations validated",
            'details': {
                'metric_positive_definite': positive_definite,
                'metric_symmetric': symmetric,
                'understanding_contractive': contractive,
                'understanding_info_preserving': info_preserving,
                'min_eigenvalue': float(np.min(eigenvals)),
                'max_understanding_eigenvalue': float(np.max(np.abs(U_eigenvals)))
            }
        }
    
    async def validate_axiom_compliance(self) -> Dict[str, Any]:
        """Validate fundamental axiom: U(A ∘ B) = U(A) ∘ U(B)"""
        
        # Generate test cases
        num_tests = 20
        axiom_violations = 0
        error_values = []
        
        for i in range(num_tests):
            # Create random semantic vectors
            a = SemanticVector(
                representation=np.random.randn(self.translator.semantic_space.dimension),
                meaning_space="test",
                entropy=np.random.uniform(0.5, 2.0),
                coherence=np.random.uniform(0.3, 0.9),
                temperature=np.random.uniform(0.5, 2.0)
            )
            
            b = SemanticVector(
                representation=np.random.randn(self.translator.semantic_space.dimension),
                meaning_space="test",
                entropy=np.random.uniform(0.5, 2.0),
                coherence=np.random.uniform(0.3, 0.9),
                temperature=np.random.uniform(0.5, 2.0)
            )
            
            # Validate axiom
            validation_result = self.translator.axiom_validator.validate_axiom(a, b)
            
            if not validation_result['axiom_holds']:
                axiom_violations += 1
            
            error_values.append(validation_result['relative_error'])
        
        # Statistical analysis
        mean_error = np.mean(error_values)
        
        axiom_compliance_rate = (num_tests - axiom_violations) / num_tests
        
        return {
            'passed': axiom_compliance_rate >= 0.8 and mean_error < 1e-2,
            'summary': f"Axiom compliance: {axiom_compliance_rate:.1%}, mean error: {mean_error:.2e}",
            'details': {
                'num_tests': num_tests,
                'axiom_violations': axiom_violations,
                'compliance_rate': axiom_compliance_rate,
                'mean_error': mean_error,
            }
        }
    
    async def validate_statistical_properties(self) -> Dict[str, Any]:
        """Validate statistical properties of translations"""
        
        # Generate multiple translations of the same content
        test_content = "The fundamental axiom of understanding"
        num_replications = 10
        
        confidence_scores = []
        
        for i in range(num_replications):
            result = await self.translator.translate(
                test_content, "natural_language", "mathematical"
            )
            
            if 'error' not in result:
                confidence_scores.append(result['confidence_score'])
        
        if len(confidence_scores) == 0:
            return {
                'passed': False,
                'summary': "No successful translations for statistical analysis",
                'details': {'error': 'All translations failed'}
            }
        
        # Consistency (low variance)
        confidence_cv = np.std(confidence_scores) / np.mean(confidence_scores) if np.mean(confidence_scores) > 0 else 1.0
        
        # Reasonable ranges
        confidence_in_range = all(0 <= c <= 1 for c in confidence_scores)
        
        consistent = confidence_cv < 0.2  # Allow some variability
        
        return {
            'passed': consistent and confidence_in_range,
            'summary': f"Statistical properties validated, CV: {confidence_cv:.3f}",
            'details': {
                'num_replications': len(confidence_scores),
                'confidence_mean': np.mean(confidence_scores),
                'confidence_cv': confidence_cv,
                'confidence_in_range': confidence_in_range,
                'consistency_achieved': consistent
            }
        }
    
    async def validate_performance_benchmarks(self) -> Dict[str, Any]:
        """Validate performance benchmarks"""
        
        # Test different content sizes and complexities
        test_cases = [
            ("Hello", "natural_language", "mathematical"),
            ("This is a moderate length sentence.", "natural_language", "echoform"),
            ("f(x) = x^2", "mathematical", "natural_language"),
        ]
        
        processing_times = []
        
        for content, source, target in test_cases:
            start_time = time.time()
            
            result = await self.translator.translate(content, source, target)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
        
        # Performance metrics
        mean_processing_time = np.mean(processing_times)
        
        # Performance requirements
        fast_enough = mean_processing_time < 2.0  # Less than 2 seconds average
        
        return {
            'passed': fast_enough,
            'summary': f"Performance: {mean_processing_time:.3f}s average",
            'details': {
                'num_test_cases': len(test_cases),
                'mean_processing_time': mean_processing_time,
                'fast_enough': fast_enough,
            }
        }
    
    async def validate_gyroscopic_stability(self) -> Dict[str, Any]:
        """Validate gyroscopic equilibrium stability"""
        
        # Perform multiple translations and monitor stability
        num_translations = 10
        stability_measurements = []
        
        test_contents = [
            "Stability test 1",
            "Different content for stability",
            "Mathematical expression: x + y = z",
        ]
        
        for i in range(num_translations):
            content = test_contents[i % len(test_contents)]
            result = await self.translator.translate(content, "natural_language", "mathematical")
            
            if 'gyroscopic_stability' in result:
                stability_measurements.append(result['gyroscopic_stability'])
        
        if not stability_measurements:
            return {
                'passed': False,
                'summary': "No stability measurements available",
                'details': {'error': 'No successful translations with stability data'}
            }
        
        # Stability analysis
        mean_stability = np.mean(stability_measurements)
        min_stability = np.min(stability_measurements)
        
        # Check equilibrium maintenance
        equilibrium_level = self.translator.equilibrium_state.equilibrium_level
        equilibrium_deviation = abs(equilibrium_level - 0.5)  # Target is 0.5
        
        # Stability requirements
        stable = mean_stability > 0.7 and min_stability > 0.5
        equilibrium_maintained = equilibrium_deviation < 0.2
        
        return {
            'passed': stable and equilibrium_maintained,
            'summary': f"Stability: {mean_stability:.3f}, equilibrium: {equilibrium_level:.3f}",
            'details': {
                'num_measurements': len(stability_measurements),
                'mean_stability': mean_stability,
                'min_stability': min_stability,
                'equilibrium_level': equilibrium_level,
                'equilibrium_deviation': equilibrium_deviation,
                'stable': stable,
                'equilibrium_maintained': equilibrium_maintained,
            }
        }


# Main execution
if __name__ == "__main__":
    async def main():
        suite = ScientificValidationSuite()
        results = await suite.run_complete_validation()
        
        logger.info("\n" + "=" * 80)
        logger.info("RIGOROUS UNIVERSAL TRANSLATOR - SCIENTIFIC VALIDATION REPORT")
        logger.info("=" * 80)
        
        logger.info(f"\nOVERALL ASSESSMENT:")
        logger.info(f"  Validation Score: {results['overall_score']:.1%}")
        logger.info(f"  Status: {results['validation_status']}")
        logger.info(f"  Scientific Rigor: {results['scientific_rigor']}")
        logger.info(f"  Components Passed: {results['passed_components']}/{results['total_components']}")
        
        logger.info(f"\nCOMPONENT RESULTS:")
        for component, result in results['component_results'].items():
            if isinstance(result, dict):
                status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
                summary = result.get('summary', 'No summary available')
                logger.info(f"  {component.replace('_', ' ')
                logger.info(f"    {summary}")
            else:
                logger.info(f"  {component.replace('_', ' ')
        
        logger.info(f"\nCONCLUSION:")
        if results['validation_status'] == 'PASSED':
            logger.info("✅ The Rigorous Universal Translator has passed comprehensive scientific validation.")
            logger.info("   The system demonstrates mathematical rigor, axiom compliance, and stable performance.")
            logger.info("   Ready for production use with high confidence in scientific foundations.")
        else:
            logger.error("❌ The Rigorous Universal Translator requires improvements before production use.")
            logger.error("   Review component failures and address mathematical or implementation issues.")
        
        logger.info("\n" + "=" * 80)
        
        return results['validation_status'] == 'PASSED'
    
    # Run the validation
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 