"""
Comprehensive Test Suite for KCl Testing Engine

Tests all aspects of KCl extended-release capsule testing including
GPU optimization, error handling, and performance monitoring.
"""

import unittest
import numpy as np
import torch
from unittest.mock import patch
import time
import sys
import os

# Add the backend to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.pharmaceutical.core.kcl_testing_engine import (
    KClTestingEngine, 
    PharmaceuticalTestingException,
    RawMaterialSpec,
    FlowabilityResult,
    DissolutionProfile,
    FormulationPrototype,
    PerformanceMetrics
)


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
        self.assertGreaterEqual(prototype.encapsulation_efficiency, 0.85)
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
    
    def test_batch_processing_performance(self):
        """Test performance with batch processing."""
        batch_size = 10  # Smaller batch for testing
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


if __name__ == '__main__':
    unittest.main(verbosity=2) 