"""
Unit Tests for CLIP Service Integration
=======================================

Tests the CLIP (Contrastive Language-Image Pre-training) service
with security checks and resource management.
"""

import io
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from PIL import Image

from src.core.services.clip_service_integration import (
    CLIPEmbedding,
    CLIPServiceIntegration,
    EmbeddingCache,
    ResourceMonitor,
    SecurityChecker,
    get_clip_service,
)


class TestSecurityChecker(unittest.TestCase):
    """Test the security checker component"""

    def test_torch_version_check_safe(self):
        """Test PyTorch version check with safe version"""
        with patch("src.core.services.clip_service_integration.torch") as mock_torch:
            mock_torch.__version__ = "2.6.0"

            is_safe, version_info = SecurityChecker.is_torch_safe()

            self.assertTrue(is_safe)
            self.assertIn("safe", version_info)

    def test_torch_version_check_vulnerable(self):
        """Test PyTorch version check with vulnerable version"""
        with patch("src.core.services.clip_service_integration.torch") as mock_torch:
            mock_torch.__version__ = "2.4.0"

            is_safe, version_info = SecurityChecker.is_torch_safe()

            self.assertFalse(is_safe)
            self.assertIn("vulnerable", version_info)

    def test_model_integrity_check(self):
        """Test model integrity verification"""
        # Test with existing path
        with patch("os.path.exists", return_value=True):
            self.assertTrue(SecurityChecker.check_model_integrity("/path/to/model"))

        # Test with non-existing path
        with patch("os.path.exists", return_value=False):
            self.assertFalse(SecurityChecker.check_model_integrity("/invalid/path"))


class TestResourceMonitor(unittest.TestCase):
    """Test the resource monitor component"""

    def test_initialization(self):
        """Test resource monitor initialization"""
        monitor = ResourceMonitor()

        self.assertEqual(monitor.total_embeddings, 0)
        self.assertEqual(monitor.cache_hits, 0)
        self.assertEqual(monitor.cache_misses, 0)
        self.assertEqual(monitor.total_processing_time, 0.0)

    def test_record_embedding(self):
        """Test recording embedding metrics"""
        monitor = ResourceMonitor()

        # Record cache hit
        monitor.record_embedding(0.1, cache_hit=True)

        # Record cache miss
        monitor.record_embedding(0.2, cache_hit=False)

        metrics = monitor.get_metrics()

        self.assertEqual(metrics["total_embeddings"], 2)
        self.assertEqual(metrics["cache_hit_rate"], 50.0)
        self.assertAlmostEqual(metrics["average_processing_time"], 0.15)


class TestEmbeddingCache(unittest.TestCase):
    """Test the embedding cache component"""

    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = EmbeddingCache(max_size=100, ttl_hours=12)

        self.assertEqual(cache.max_size, 100)
        self.assertEqual(cache.ttl, timedelta(hours=12))
        self.assertEqual(len(cache.cache), 0)

    def test_cache_put_and_get(self):
        """Test putting and getting from cache"""
        cache = EmbeddingCache()

        # Create test embedding
        embedding = CLIPEmbedding(
            embedding=np.array([1.0, 2.0, 3.0]),
            modality="text",
            source_hash="test_hash",
            timestamp=datetime.now(timezone.utc),
            model_version="test_model",
            device="cpu",
        )

        # Put in cache
        cache.put("test_key", embedding)

        # Get from cache
        retrieved = cache.get("test_key")

        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved.embedding, embedding.embedding)

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration"""
        cache = EmbeddingCache(ttl_hours=0)  # Immediate expiration

        embedding = CLIPEmbedding(
            embedding=np.array([1.0, 2.0, 3.0]),
            modality="text",
            source_hash="test_hash",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),  # Old timestamp
            model_version="test_model",
            device="cpu",
        )

        cache.put("expired_key", embedding)

        # Should return None due to expiration
        retrieved = cache.get("expired_key")
        self.assertIsNone(retrieved)

    def test_cache_lru_eviction(self):
        """Test LRU eviction"""
        cache = EmbeddingCache(max_size=2)

        # Add 3 items to cache with max_size=2
        for i in range(3):
            embedding = CLIPEmbedding(
                embedding=np.array([float(i)]),
                modality="text",
                source_hash=f"hash_{i}",
                timestamp=datetime.now(timezone.utc),
                model_version="test",
                device="cpu",
            )
            cache.put(f"key_{i}", embedding)
            time.sleep(0.01)  # Ensure different access times

        # First item should be evicted
        self.assertIsNone(cache.get("key_0"))
        self.assertIsNotNone(cache.get("key_1"))
        self.assertIsNotNone(cache.get("key_2"))


class TestCLIPServiceIntegration(unittest.TestCase):
    """Test the CLIP service integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Patch the model loading
        self.patcher_model = patch(
            "src.core.services.clip_service_integration.CLIPModel"
        )
        self.patcher_processor = patch(
            "src.core.services.clip_service_integration.CLIPProcessor"
        )
        self.patcher_torch = patch("src.core.services.clip_service_integration.torch")

        self.mock_clip_model = self.patcher_model.start()
        self.mock_clip_processor = self.patcher_processor.start()
        self.mock_torch = self.patcher_torch.start()

        # Configure mocks
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()

        self.mock_clip_model.from_pretrained.return_value = self.mock_model
        self.mock_clip_processor.from_pretrained.return_value = self.mock_processor
        self.mock_torch.cuda.is_available.return_value = False

        # Create service
        self.service = CLIPServiceIntegration(lightweight_mode=False)

        # Create test data
        self.test_image = Image.new("RGB", (224, 224), color="red")
        self.test_text = "A red square image"

    def tearDown(self):
        """Clean up patches"""
        self.patcher_model.stop()
        self.patcher_processor.stop()
        self.patcher_torch.stop()

    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsNotNone(self.service)
        self.assertFalse(self.service.lightweight_mode)
        self.assertIsNotNone(self.service.resource_monitor)
        self.assertIsNotNone(self.service.cache)
        self.assertEqual(self.service.device, "cpu")

    def test_lightweight_mode_initialization(self):
        """Test initialization in lightweight mode"""
        service = CLIPServiceIntegration(lightweight_mode=True)

        self.assertTrue(service.lightweight_mode)
        self.assertIsNone(service.model)
        self.assertIsNone(service.processor)

    def test_get_image_embedding(self):
        """Test image embedding generation"""
        # Mock model output
        mock_features = MagicMock()
        mock_features.cpu.return_value.numpy.return_value = np.array([[1.0, 2.0, 3.0]])
        mock_features.norm.return_value = mock_features
        mock_features.__truediv__.return_value = mock_features

        self.mock_model.get_image_features.return_value = mock_features
        self.mock_processor.return_value = {"pixel_values": MagicMock()}

        # Get embedding
        embedding = self.service.get_image_embedding(self.test_image)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (3,))

    def test_get_text_embedding(self):
        """Test text embedding generation"""
        # Mock model output
        mock_features = MagicMock()
        mock_features.cpu.return_value.numpy.return_value = np.array([[1.0, 2.0, 3.0]])
        mock_features.norm.return_value = mock_features
        mock_features.__truediv__.return_value = mock_features

        self.mock_model.get_text_features.return_value = mock_features
        self.mock_processor.return_value = {"input_ids": MagicMock()}

        # Get embedding
        embedding = self.service.get_text_embedding(self.test_text)

        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (3,))

    def test_caching(self):
        """Test caching functionality"""
        # Mock model output
        mock_features = MagicMock()
        mock_features.cpu.return_value.numpy.return_value = np.array([[1.0, 2.0, 3.0]])
        mock_features.norm.return_value = mock_features
        mock_features.__truediv__.return_value = mock_features

        self.mock_model.get_text_features.return_value = mock_features
        self.mock_processor.return_value = {"input_ids": MagicMock()}

        # First call
        embedding1 = self.service.get_text_embedding(self.test_text)

        # Second call (should use cache)
        embedding2 = self.service.get_text_embedding(self.test_text)

        # Should be the same
        np.testing.assert_array_equal(embedding1, embedding2)

        # Model should only be called once
        self.mock_model.get_text_features.assert_called_once()

    def test_compute_similarity(self):
        """Test similarity computation"""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        embedding3 = np.array([1.0, 0.0, 0.0])

        # Orthogonal vectors
        sim_12 = self.service.compute_similarity(embedding1, embedding2)
        self.assertAlmostEqual(sim_12, 0.0, places=5)

        # Identical vectors
        sim_13 = self.service.compute_similarity(embedding1, embedding3)
        self.assertAlmostEqual(sim_13, 1.0, places=5)

    def test_find_best_match(self):
        """Test finding best matching embedding"""
        query = np.array([1.0, 0.0, 0.0])
        candidates = {
            "orthogonal": np.array([0.0, 1.0, 0.0]),
            "similar": np.array([0.9, 0.1, 0.0]),
            "identical": np.array([1.0, 0.0, 0.0]),
        }

        best_label, best_sim = self.service.find_best_match(query, candidates)

        self.assertEqual(best_label, "identical")
        self.assertAlmostEqual(best_sim, 1.0, places=5)

    def test_is_available(self):
        """Test availability check"""
        self.assertTrue(self.service.is_available())

        # Test with no model
        self.service.model = None
        self.assertFalse(self.service.is_available())

    def test_get_status(self):
        """Test status reporting"""
        status = self.service.get_status()

        self.assertIn("available", status)
        self.assertIn("lightweight_mode", status)
        self.assertIn("device", status)
        self.assertIn("model_name", status)
        self.assertIn("torch_safe", status)
        self.assertIn("embedding_dimension", status)
        self.assertIn("cache_enabled", status)
        self.assertIn("metrics", status)

    def test_clear_cache(self):
        """Test cache clearing"""
        # Add something to cache
        embedding = CLIPEmbedding(
            embedding=np.array([1.0, 2.0, 3.0]),
            modality="text",
            source_hash="test",
            timestamp=datetime.now(timezone.utc),
            model_version="test",
            device="cpu",
        )
        self.service.cache.put("test_key", embedding)

        # Clear cache
        self.service.clear_cache()

        # Cache should be empty
        self.assertEqual(len(self.service.cache.cache), 0)

    def test_shutdown(self):
        """Test service shutdown"""
        self.service.shutdown()

        # Model should be None
        self.assertIsNone(self.service.model)
        self.assertIsNone(self.service.processor)

        # Cache should be cleared
        self.assertEqual(len(self.service.cache.cache), 0)

    def test_singleton_pattern(self):
        """Test singleton pattern"""
        service1 = get_clip_service()
        service2 = get_clip_service()

        self.assertIs(service1, service2)


class TestCLIPServiceEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        """Set up test fixtures"""
        # Create service in lightweight mode to avoid model loading
        self.service = CLIPServiceIntegration(lightweight_mode=True)

    def test_empty_text_embedding(self):
        """Test handling of empty text"""
        embedding = self.service.get_text_embedding("")

        # Should return zero embedding in lightweight mode
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (512,))
        self.assertAlmostEqual(np.sum(embedding), 0.0)

    def test_invalid_image_bytes(self):
        """Test handling of invalid image bytes"""
        with self.assertRaises(Exception):
            self.service.get_image_embedding(b"invalid image data")

    def test_zero_norm_vectors(self):
        """Test similarity with zero vectors"""
        zero_vec = np.zeros(3)
        normal_vec = np.array([1.0, 0.0, 0.0])

        similarity = self.service.compute_similarity(zero_vec, normal_vec)
        self.assertEqual(similarity, 0.0)

    def test_find_best_match_no_candidates(self):
        """Test finding best match with no candidates"""
        query = np.array([1.0, 0.0, 0.0])
        result = self.service.find_best_match(query, {})

        self.assertIsNone(result)

    def test_find_best_match_below_threshold(self):
        """Test finding best match with high threshold"""
        query = np.array([1.0, 0.0, 0.0])
        candidates = {"orthogonal": np.array([0.0, 1.0, 0.0])}

        result = self.service.find_best_match(query, candidates, threshold=0.5)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
