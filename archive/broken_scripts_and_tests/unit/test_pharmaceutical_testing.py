"""
Comprehensive Test Suite for Pharmaceutical Testing Engine

Tests all aspects of KCl extended-release capsule testing and validation
including GPU optimization, error handling, and performance monitoring.
"""

import unittest
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from typing import Dict, Any
import time

from backend.pharmaceutical.core.kcl_testing_engine import (
    KClTestingEngine, 
    PharmaceuticalTestingException,
    RawMaterialSpec,
    FlowabilityResult,
    DissolutionProfile,
    FormulationPrototype,
    PerformanceMetrics
)
from backend.pharmaceutical.protocols.usp_protocols import USPProtocolEngine
from backend.pharmaceutical.analysis.dissolution_analyzer import DissolutionAnalyzer
from backend.pharmaceutical.validation.pharmaceutical_validator import PharmaceuticalValidator
from backend.utils.kimera_exceptions import KimeraException


class TestKClTestingEngine(unittest.TestCase):
    """Test the KCl Testing Engine core functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.engine = KClTestingEngine(use_gpu=False)  # Use CPU for consistent testing
        self.valid_material_batch = {
            'name': 'Potassium Chloride USP',
            'grade': 'USP',
            'purity_percent': 99.5,
            'moisture_content': 0.8,
            'particle_size_d50': 150.0,
            'bulk_density': 0.85,
            'tapped_density': 1.05,
            'potassium_confirmed': True,
            'chloride_confirmed': True,
            'heavy_metals_ppm': 5.0,
            'sodium_percent': 0.1,
            'bromide_ppm': 100.0
        }
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.usp_standards)
        self.assertEqual(self.engine.device.type, 'cpu')
        self.assertFalse(self.engine.gpu_optimization_enabled)
    
    def test_gpu_initialization(self):
        """Test GPU initialization when available."""
        if torch.cuda.is_available():
            gpu_engine = KClTestingEngine(use_gpu=True)
            self.assertEqual(gpu_engine.device.type, 'cuda')
            self.assertTrue(gpu_engine.gpu_optimization_enabled)
    
    def test_valid_raw_material_characterization(self):
        """Test characterization of valid raw materials."""
        result = self.engine.characterize_raw_materials(self.valid_material_batch)
        
        self.assertIsInstance(result, RawMaterialSpec)
        self.assertEqual(result.purity_percent, 99.5)
        self.assertEqual(result.moisture_content, 0.8)
        self.assertTrue(result.identification_tests['potassium_test'])
        self.assertTrue(result.identification_tests['chloride_test'])
        
    def test_invalid_purity_characterization(self):
        """Test characterization with invalid purity."""
        invalid_batch = self.valid_material_batch.copy()
        invalid_batch['purity_percent'] = 95.0  # Below USP limit
        
        with self.assertRaises(PharmaceuticalTestingException):
            self.engine.characterize_raw_materials(invalid_batch)
    
    def test_high_moisture_characterization(self):
        """Test characterization with high moisture content."""
        invalid_batch = self.valid_material_batch.copy()
        invalid_batch['moisture_content'] = 1.5  # Above USP limit
        
        with self.assertRaises(PharmaceuticalTestingException):
            self.engine.characterize_raw_materials(invalid_batch)
    
    def test_missing_required_fields(self):
        """Test characterization with missing required fields."""
        incomplete_batch = {'name': 'KCl'}  # Missing required fields
        
        with self.assertRaises(PharmaceuticalTestingException):
            self.engine.characterize_raw_materials(incomplete_batch)
    
    def test_invalid_input_type(self):
        """Test characterization with invalid input type."""
        with self.assertRaises(PharmaceuticalTestingException):
            self.engine.characterize_raw_materials("invalid_input")
    
    def test_powder_flowability_analysis(self):
        """Test powder flowability analysis."""
        result = self.engine.analyze_powder_flowability(0.85, 1.05, 35.0)
        
        self.assertIsInstance(result, FlowabilityResult)
        self.assertAlmostEqual(result.carr_index, 19.05, places=1)
        self.assertAlmostEqual(result.hausner_ratio, 1.24, places=2)
        self.assertIn(result.flow_character, ['Excellent', 'Good', 'Fair', 'Passable', 'Poor'])
    
    def test_formulation_prototype_creation(self):
        """Test formulation prototype creation."""
        coating_thickness = 12.0
        polymer_ratios = {'ethylcellulose': 0.8, 'hpc': 0.2}
        process_params = {'temperature': 60.0, 'spray_rate': 1.0}
        
        prototype = self.engine.create_formulation_prototype(
            coating_thickness, polymer_ratios, process_params
        )
        
        self.assertIsInstance(prototype, FormulationPrototype)
        self.assertEqual(prototype.coating_thickness_percent, 12.0)
        self.assertGreater(prototype.encapsulation_efficiency, 0.85)
        self.assertLess(prototype.encapsulation_efficiency, 1.0)
    
    def test_dissolution_test_performance(self):
        """Test dissolution test with performance tracking."""
        # Create prototype first
        prototype = self.engine.create_formulation_prototype(
            12.0, {'ethylcellulose': 0.8, 'hpc': 0.2}, {'temperature': 60.0}
        )
        
        test_conditions = {
            'apparatus': 1,
            'medium': 'water',
            'volume_ml': 900,
            'temperature_c': 37.0,
            'rotation_rpm': 100
        }
        
        profile = self.engine.perform_dissolution_test(prototype, test_conditions)
        
        self.assertIsInstance(profile, DissolutionProfile)
        self.assertEqual(len(profile.time_points), 4)  # Default USP time points
        self.assertEqual(len(profile.release_percentages), 4)
        
        # Check that all release percentages are reasonable
        for release in profile.release_percentages:
            self.assertGreaterEqual(release, 0)
            self.assertLessEqual(release, 100)
    
    def test_f2_similarity_calculation(self):
        """Test f2 similarity calculation."""
        profile1 = DissolutionProfile([1, 2, 4, 6], [30, 55, 80, 95], {})
        profile2 = DissolutionProfile([1, 2, 4, 6], [28, 53, 78, 93], {})
        
        f2 = self.engine._calculate_f2_similarity(profile1, profile2)
        
        self.assertGreater(f2, 50.0)  # Should be similar profiles
        self.assertLessEqual(f2, 100.0)
    
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        # Perform some operations to generate metrics
        self.engine.characterize_raw_materials(self.valid_material_batch)
        
        # Check that performance was tracked
        self.assertGreater(len(self.engine.performance_metrics), 0)
        
        # Get performance report
        report = self.engine.get_performance_report()
        
        self.assertIn('operation_statistics', report)
        self.assertIn('system_health', report)
        self.assertIn('gpu_info', report)
        self.assertIn('recommendations', report)
    
    def test_gpu_cpu_encapsulation_consistency(self):
        """Test that GPU and CPU encapsulation methods give consistent results."""
        coating_thickness = 12.0
        polymer_ratios = {'ethylcellulose': 0.8, 'hpc': 0.2}
        process_params = {'temperature': 60.0, 'spray_rate': 1.0}
        
        cpu_result = self.engine._simulate_encapsulation_cpu(
            coating_thickness, polymer_ratios, process_params
        )
        
        # Mock GPU unavailable to test fallback
        with patch('torch.cuda.is_available', return_value=False):
            gpu_result = self.engine._simulate_encapsulation_gpu(
                coating_thickness, polymer_ratios, process_params
            )
        
        # Results should be reasonably close (within 5% due to different implementations)
        self.assertAlmostEqual(cpu_result, gpu_result, delta=0.05)


class TestUSPProtocolEngine(unittest.TestCase):
    """Test USP Protocol Engine functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.engine = USPProtocolEngine()
    
    def test_content_uniformity_passing(self):
        """Test content uniformity with passing values."""
        sample_measurements = [98.5, 101.2, 99.8, 100.5, 99.1, 102.0, 98.9, 100.8, 99.6, 101.1]
        result = self.engine.perform_content_uniformity_905(sample_measurements, 100.0)
        
        self.assertEqual(result.status, "PASSED")
        self.assertIsNotNone(result.acceptance_value)
    
    def test_content_uniformity_failing(self):
        """Test content uniformity with failing values."""
        sample_measurements = [85.0, 115.0, 90.0, 110.0, 88.0]  # High variability
        result = self.engine.perform_content_uniformity_905(sample_measurements, 100.0)
        
        self.assertEqual(result.status, "FAILED")
    
    def test_assay_test_valid(self):
        """Test assay test with valid concentrations."""
        result = self.engine.perform_assay_test(95.2, 100.0, 750.0)
        
        self.assertEqual(result.status, "PASSED")
        self.assertAlmostEqual(result.result_value, 95.2, places=1)


class TestDissolutionAnalyzer(unittest.TestCase):
    """Test Dissolution Analyzer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = DissolutionAnalyzer(use_gpu=False)
        self.sample_data = {
            'time_points': [1.0, 2.0, 4.0, 6.0],
            'release_percentages': [25.0, 50.0, 75.0, 90.0]
        }
    
    def test_kinetics_analysis(self):
        """Test dissolution kinetics analysis."""
        result = self.analyzer.analyze_dissolution_kinetics(
            self.sample_data['time_points'],
            self.sample_data['release_percentages']
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('zero_order', result)
        self.assertIn('first_order', result)
        self.assertIn('higuchi', result)
    
    def test_profile_comparison(self):
        """Test dissolution profile comparison."""
        profile1_times = [1.0, 2.0, 4.0, 6.0]
        profile1_releases = [25.0, 50.0, 75.0, 90.0]
        profile2_times = [1.0, 2.0, 4.0, 6.0]
        profile2_releases = [28.0, 52.0, 78.0, 92.0]
        
        comparison = self.analyzer.compare_dissolution_profiles(
            profile1_times, profile1_releases,
            profile2_times, profile2_releases
        )
        
        self.assertGreater(comparison.f2_similarity, 50.0)  # Should be similar
        self.assertEqual(comparison.similarity_assessment, 'SIMILAR')


class TestPharmaceuticalValidator(unittest.TestCase):
    """Test Pharmaceutical Validator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.validator = PharmaceuticalValidator(use_gpu=False)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator.kcl_engine)
        self.assertIsNotNone(self.validator.usp_engine)
        self.assertIsNotNone(self.validator.dissolution_analyzer)
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation workflow."""
        validation_data = {
            'raw_materials': {
                'name': 'KCl USP',
                'purity_percent': 99.5,
                'moisture_content': 0.8,
                'potassium_confirmed': True,
                'chloride_confirmed': True
            },
            'formulation': {
                'coating_thickness_percent': 12.0,
                'polymer_ratios': {'ethylcellulose': 0.8, 'hpc': 0.2}
            },
            'finished_product': {
                'dissolution_profile': [25, 50, 75, 90],
                'content_uniformity': [99.5, 100.5, 99.8, 100.2]
            }
        }
        
        result = self.validator.validate_comprehensive_pharmaceutical_development(validation_data)
        
        self.assertIn('validation_id', result.__dict__)
        self.assertIn('status', result.__dict__)
        self.assertIn('confidence_score', result.__dict__)


class TestPerformanceAndStress(unittest.TestCase):
    """Test performance and stress scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.engine = KClTestingEngine(use_gpu=False)
    
    def test_batch_processing_performance(self):
        """Test performance with batch processing."""
        batch_size = 100
        start_time = time.perf_counter()
        
        for i in range(batch_size):
            material_batch = {
                'name': f'KCl_Batch_{i}',
                'purity_percent': 99.0 + np.random.uniform(-0.3, 0.3),
                'moisture_content': 0.5 + np.random.uniform(0, 0.4),
                'particle_size_d50': 150.0,
                'bulk_density': 0.85,
                'tapped_density': 1.05,
                'potassium_confirmed': True,
                'chloride_confirmed': True
            }
            
            try:
                self.engine.characterize_raw_materials(material_batch)
            except PharmaceuticalTestingException:
                pass  # Expected for some out-of-spec batches
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Performance should be reasonable (less than 1 second per batch)
        self.assertLess(total_time / batch_size, 1.0)
        
        # Check performance report
        report = self.engine.get_performance_report()
        self.assertGreater(len(self.engine.performance_metrics), 0)
    
    def test_error_handling_robustness(self):
        """Test robust error handling with various invalid inputs."""
        invalid_inputs = [
            None,
            "",
            [],
            {"purity_percent": "invalid"},
            {"purity_percent": -1},
            {"moisture_content": 10.0},  # Too high
            {"purity_percent": 50.0, "moisture_content": 0.5},  # Too low purity
        ]
        
        for invalid_input in invalid_inputs:
            with self.assertRaises((PharmaceuticalTestingException, TypeError, ValueError)):
                self.engine.characterize_raw_materials(invalid_input)
    
    def test_memory_efficiency(self):
        """Test memory efficiency during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(50):
            material_batch = {
                'name': f'KCl_Memory_Test_{i}',
                'purity_percent': 99.5,
                'moisture_content': 0.8,
                'particle_size_d50': 150.0,
                'bulk_density': 0.85,
                'tapped_density': 1.05,
                'potassium_confirmed': True,
                'chloride_confirmed': True
            }
            
            self.engine.characterize_raw_materials(material_batch)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for 50 operations)
        self.assertLess(memory_increase, 50 * 1024 * 1024)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2) 