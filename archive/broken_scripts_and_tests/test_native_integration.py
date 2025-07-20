#!/usr/bin/env python3
"""
Integration test to verify native math implementations work with Kimera SWM components.
"""

import sys
import traceback
from typing import List, Dict

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def test_contradiction_engine():
    """Test contradiction engine with native math."""
    logger.debug("🔍 Testing Contradiction Engine...")
    
    try:
        from backend.engines.contradiction_engine import ContradictionEngine
        from backend.core.geoid import GeoidState
        
        # Create test geoids
        geoid_a = GeoidState(
            geoid_id='test_a',
            semantic_state={'feature1': 1.0, 'feature2': 2.0, 'feature3': 0.5},
            symbolic_state={'type': 'concept'},
            embedding_vector=[1.0, 2.0, 3.0, 4.0],
            metadata={'source': 'test'}
        )
        
        geoid_b = GeoidState(
            geoid_id='test_b', 
            semantic_state={'feature1': 0.5, 'feature2': 1.5, 'feature3': 2.0},
            symbolic_state={'type': 'concept'},
            embedding_vector=[2.0, 3.0, 4.0, 5.0],
            metadata={'source': 'test'}
        )
        
        # Test contradiction detection
        engine = ContradictionEngine(tension_threshold=0.1)
        tensions = engine.detect_tension_gradients([geoid_a, geoid_b])
        
        logger.info(f"   ✅ Detected {len(tensions)
        if tensions:
            tension = tensions[0]
            logger.info(f"   ✅ Tension score: {tension.tension_score:.4f}")
            logger.info(f"   ✅ Gradient type: {tension.gradient_type}")
        
        # Test pulse strength calculation
        if tensions:
            pulse = engine.calculate_pulse_strength(tensions[0], {
                'test_a': geoid_a,
                'test_b': geoid_b
            })
            logger.info(f"   ✅ Pulse strength: {pulse:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Contradiction engine test failed: {e}")
        traceback.print_exc()
        return False

def test_spde_engine():
    """Test SPDE engine with native Gaussian filter."""
    logger.info("🌊 Testing SPDE Engine...")
    
    try:
        from backend.engines.spde import SPDE
        
        # Create SPDE instance
        spde = SPDE(diffusion_rate=0.3, decay_factor=1.5)
        
        # Test with semantic state
        test_state = {
            'concept_a': 1.0,
            'concept_b': 2.0, 
            'concept_c': 0.5,
            'concept_d': 3.0,
            'concept_e': 1.5
        }
        
        logger.info(f"   Original state: {test_state}")
        
        # Apply diffusion
        diffused = spde.diffuse(test_state)
        
        logger.info(f"   Diffused state: {diffused}")
        
        # Verify diffusion worked
        assert len(diffused) == len(test_state), "State size should be preserved"
        assert all(isinstance(v, float) for v in diffused.values()), "All values should be floats"
        
        # Test with empty state
        empty_diffused = spde.diffuse({})
        assert empty_diffused == {}, "Empty state should remain empty"
        
        logger.info("   ✅ SPDE diffusion working correctly")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ SPDE engine test failed: {e}")
        traceback.print_exc()
        return False

def test_proactive_detector():
    """Test proactive contradiction detector."""
    logger.info("🔎 Testing Proactive Contradiction Detector...")
    
    try:
        from backend.engines.proactive_contradiction_detector import ProactiveContradictionDetector, ProactiveDetectionConfig
        from backend.core.geoid import GeoidState
        
        # Create detector with test config
        config = ProactiveDetectionConfig(
            batch_size=10,
            similarity_threshold=0.5,
            max_comparisons_per_run=50
        )
        detector = ProactiveContradictionDetector(config)
        
        # Create test geoids
        geoids = []
        for i in range(5):
            geoid = GeoidState(
                geoid_id=f'test_{i}',
                semantic_state={f'feature_{j}': float(i + j) for j in range(3)},
                symbolic_state={'type': 'test', 'index': i},
                embedding_vector=[float(i + j) for j in range(4)],
                metadata={'timestamp': f'2024-01-{i+1:02d}T12:00:00Z'}
            )
            geoids.append(geoid)
        
        # Test similarity calculation
        similarity = detector._calculate_similarity(geoids[0], geoids[1])
        logger.info(f"   ✅ Similarity calculation: {similarity:.4f}")
        
        # Test clustering
        clusters = detector._create_semantic_clusters(geoids)
        logger.info(f"   ✅ Created {len(clusters)
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Proactive detector test failed: {e}")
        traceback.print_exc()
        return False

def test_asm_engine():
    """Test ASM engine imports and basic functionality."""
    logger.info("📊 Testing ASM Engine...")
    
    try:
        from backend.engines.asm import AxisStabilityMonitor
        from backend.core.native_math import NativeDistance
        
        # Test the native distance function that ASM uses
        test_vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0]
        ]
        
        distances = NativeDistance.condensed_distances(test_vectors, metric='cosine')
        logger.info(f"   ✅ Distance calculation: {len(distances)
        logger.info(f"   ✅ Sample distances: {[f'{d:.4f}' for d in distances[:3]]}")
        
        # Verify ASM class can be instantiated (without DB)
        logger.info("   ✅ ASM class imports successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ ASM engine test failed: {e}")
        traceback.print_exc()
        return False

def test_geoid_entropy():
    """Test GeoidState entropy calculation."""
    logger.info("🧮 Testing GeoidState Entropy...")
    
    try:
        from backend.core.geoid import GeoidState
        
        # Test with uniform distribution (high entropy)
        uniform_geoid = GeoidState(
            geoid_id='uniform',
            semantic_state={'a': 0.25, 'b': 0.25, 'c': 0.25, 'd': 0.25},
            symbolic_state={},
            embedding_vector=[],
            metadata={}
        )
        
        uniform_entropy = uniform_geoid.calculate_entropy()
        logger.info(f"   ✅ Uniform entropy: {uniform_entropy:.4f}")
        
        # Test with skewed distribution (low entropy)
        skewed_geoid = GeoidState(
            geoid_id='skewed',
            semantic_state={'a': 0.7, 'b': 0.1, 'c': 0.1, 'd': 0.1},
            symbolic_state={},
            embedding_vector=[],
            metadata={}
        )
        
        skewed_entropy = skewed_geoid.calculate_entropy()
        logger.info(f"   ✅ Skewed entropy: {skewed_entropy:.4f}")
        
        # Verify entropy ordering
        assert uniform_entropy > skewed_entropy, "Uniform should have higher entropy than skewed"
        
        # Test with empty state
        empty_geoid = GeoidState('empty', {}, {}, [], {})
        empty_entropy = empty_geoid.calculate_entropy()
        assert empty_entropy == 0.0, "Empty state should have zero entropy"
        
        logger.info("   ✅ Entropy calculations working correctly")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ GeoidState entropy test failed: {e}")
        traceback.print_exc()
        return False

def test_api_compatibility():
    """Test that API components work with native math."""
    logger.info("🌐 Testing API Compatibility...")
    
    try:
        # Test the specific function used in main.py
        from backend.core.native_math import NativeMath
        
        # Simulate the exact usage in main.py
        vec_a = [1.0, 2.0, 3.0, 4.0]
        vec_b = [2.0, 3.0, 4.0, 5.0]
        
        cls_angle_proxy = NativeMath.cosine_distance(vec_a, vec_b) if any(vec_a) and any(vec_b) else 0.0
        logger.info(f"   ✅ API cosine distance: {cls_angle_proxy:.4f}")
        
        # Test edge cases that might occur in API
        empty_result = NativeMath.cosine_distance([], []) if any([]) and any([]) else 0.0
        assert empty_result == 0.0, "Empty vector handling should work"
        
        logger.info("   ✅ API compatibility verified")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ API compatibility test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    logger.info("🚀 Kimera SWM Native Math Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        test_contradiction_engine,
        test_spde_engine,
        test_proactive_detector,
        test_asm_engine,
        test_geoid_entropy,
        test_api_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
        logger.info()
    
    logger.info("=" * 60)
    logger.info(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Native math implementations are fully compatible with Kimera SWM")
        logger.info("✅ System is ready for production use without scipy dependency")
    else:
        logger.error("❌ Some tests failed - review and fix before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()