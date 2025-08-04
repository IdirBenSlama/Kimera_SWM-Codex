
import asyncio
import logging
import unittest

from src.core.geometric_optimization.integration import GeometricOptimizationIntegrator

class TestGeometricOptimizationIntegration(unittest.TestCase):

    def test_integration_initialization(self):
        """Test that the GeometricOptimizationIntegrator initializes correctly."""
        try:
            integrator = GeometricOptimizationIntegrator()
            self.assertIsNotNone(integrator.portal_engine)
            self.assertIsNotNone(integrator.optimizer)
            logging.info("GeometricOptimizationIntegrator initialized successfully.")
        except Exception as e:
            self.fail(f"GeometricOptimizationIntegrator failed to initialize: {e}")

    async def test_create_optimized_mirror_portal(self):
        """Test the creation of an optimized mirror portal."""
        integrator = GeometricOptimizationIntegrator()
        semantic_content = {"meaning": 0.9, "beauty": 0.8}
        symbolic_content = {"form": "phyllotaxis", "pattern": "spiral"}

        result = await integrator.create_optimized_mirror_portal(semantic_content, symbolic_content)

        self.assertIn("portal_id", result)
        self.assertIn("optimized_coherence", result)
        self.assertTrue(0 <= result["optimized_coherence"] <= 1)
        logging.info(f"Optimized mirror portal created with coherence: {result['optimized_coherence']}")

    def test_get_global_optimization_metrics(self):
        """Test the retrieval of global optimization metrics."""
        integrator = GeometricOptimizationIntegrator()
        metrics = integrator.get_global_optimization_metrics()

        self.assertIn("vortex_efficiency", metrics)
        self.assertIn("portal_coherence", metrics)
        self.assertIn("diffusion_alignment", metrics)
        self.assertIn("ecoform_naturalness", metrics)
        logging.info(f"Global optimization metrics: {metrics}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
