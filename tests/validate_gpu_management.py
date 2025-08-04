
import asyncio
import logging
import unittest

from src.core.gpu_management.integration import GPUManagementIntegrator

class TestGpuManagementIntegration(unittest.TestCase):

    def test_integration_initialization(self):
        """Test that the GPUManagementIntegrator initializes correctly.""" 
        try:
            integrator = GPUManagementIntegrator()
            self.assertIsNotNone(integrator.memory_pool)
            self.assertIsNotNone(integrator.signal_memory_manager)
            self.assertIsNotNone(integrator.thermodynamic_integrator)
            logging.info("GPUManagementIntegrator initialized successfully.")
        except Exception as e:
            self.fail(f"GPUManagementIntegrator failed to initialize: {e}")

    def test_get_gpu_thermodynamic_state(self):
        """Test the retrieval of the GPU thermodynamic state."""
        integrator = GPUManagementIntegrator()
        thermo_state = integrator.get_gpu_thermodynamic_state([], 0)

        self.assertTrue(hasattr(thermo_state, "thermal_entropy"))
        self.assertTrue(hasattr(thermo_state, "computational_work"))
        self.assertTrue(hasattr(thermo_state, "power_efficiency"))
        logging.info(f"GPU thermodynamic state: {thermo_state}")

    def test_get_memory_pool_stats(self):
        """Test the retrieval of memory pool stats."""
        integrator = GPUManagementIntegrator()
        stats = integrator.get_memory_pool_stats()

        self.assertIn("total_blocks", stats)
        self.assertIn("total_pooled_memory_mb", stats)
        logging.info(f"Memory pool stats: {stats}")

    def test_get_signal_memory_stats(self):
        """Test the retrieval of signal memory stats."""
        integrator = GPUManagementIntegrator()
        stats = integrator.get_signal_memory_stats()

        self.assertIn("managed_fields_count", stats)
        self.assertIn("total_memory_overhead_bytes", stats)
        logging.info(f"Signal memory stats: {stats}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
