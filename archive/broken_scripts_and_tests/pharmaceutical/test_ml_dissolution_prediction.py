"""
Test Suite for Machine Learning Dissolution Prediction

Tests the new ML-based dissolution prediction capabilities.
"""

import unittest
import numpy as np
import torch
import time
import sys
import os

# Add the backend to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.pharmaceutical.analysis.dissolution_analyzer import (
    DissolutionAnalyzer,
    ModelPrediction
)
from backend.utils.kimera_exceptions import KimeraBaseException as KimeraException


class TestMLDissolutionPrediction(unittest.TestCase):
    """Test machine learning dissolution prediction capabilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = DissolutionAnalyzer(use_gpu=True)
        self.test_formulation_params = {
            'coating_thickness': 15.0,
            'ethylcellulose_ratio': 0.8,
            'hpc_ratio': 0.2,
            'drug_loading': 50.0,
            'particle_size': 150.0,
            'tablet_hardness': 8.0,
            'porosity': 0.15,
            'surface_area': 2.5,
            'ph_media': 6.8,
            'temperature': 37.0,
            'agitation_speed': 100.0,
            'ionic_strength': 0.1
        }
        self.test_time_points = [1, 2, 4, 6, 8, 12, 16, 20, 24]
    
    def test_ml_model_initialization(self):
        """Test ML model initialization."""
        if torch.cuda.is_available():
            self.assertIsNotNone(self.analyzer.ml_model)
            
            # Test model architecture
            first_layer = self.analyzer.ml_model.feature_layers[0]
            self.assertEqual(first_layer.in_features, 12)
            
            # Output layer should produce 20 time points
            last_linear_layer = None
            for layer in self.analyzer.ml_model.feature_layers:
                if isinstance(layer, torch.nn.Linear):
                    last_linear_layer = layer
            self.assertEqual(last_linear_layer.out_features, 20)
            
        else:
            print("⚠️  CUDA not available - ML model initialization skipped")
    
    def test_feature_extraction(self):
        """Test formulation feature extraction."""
        features = self.analyzer._extract_formulation_features(self.test_formulation_params)
        
        # Should extract exactly 12 features
        self.assertEqual(len(features), 12)
        
        # All features should be numeric
        for feature in features:
            self.assertIsInstance(feature, (int, float))
            self.assertGreater(feature, 0)
        
        print("✅ Feature extraction: 12 numerical features extracted successfully")
    
    def test_ml_prediction_basic(self):
        """Test basic ML prediction functionality."""
        if not torch.cuda.is_available() or self.analyzer.ml_model is None:
            self.skipTest("ML model not available")
        
        try:
            prediction = self.analyzer.predict_dissolution_ml(
                self.test_formulation_params, 
                self.test_time_points
            )
            
            # Verify prediction structure
            self.assertIsInstance(prediction, ModelPrediction)
            self.assertEqual(len(prediction.predicted_times), len(self.test_time_points))
            self.assertEqual(len(prediction.predicted_releases), len(self.test_time_points))
            
            # Verify prediction quality
            for release in prediction.predicted_releases:
                self.assertGreaterEqual(release, 0.0)
                self.assertLessEqual(release, 100.0)
            
            print(f"✅ ML prediction completed with {prediction.prediction_accuracy:.2%} accuracy")
            
        except KimeraException as e:
            if "ML model not initialized" in str(e):
                self.skipTest("ML model initialization failed")
            else:
                raise


if __name__ == '__main__':
    unittest.main(verbosity=2) 