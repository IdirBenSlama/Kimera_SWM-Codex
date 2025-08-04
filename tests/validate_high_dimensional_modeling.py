import asyncio
import logging
import unittest

import torch

from src.core.high_dimensional_modeling.integration import (
    HighDimensionalModelingIntegrator,
)


class TestHighDimensionalModelingIntegration(unittest.TestCase):

    def test_integration_initialization(self):
        """Test that the HighDimensionalModelingIntegrator initializes correctly."""
        try:
            integrator = HighDimensionalModelingIntegrator()
            self.assertIsNotNone(integrator.bgm_engine)
            self.assertIsNotNone(integrator.homomorphic_processor)
            logging.info("HighDimensionalModelingIntegrator initialized successfully.")
        except Exception as e:
            self.fail(f"HighDimensionalModelingIntegrator failed to initialize: {e}")

    def test_generate_secure_market_scenarios(self):
        """Test the generation of secure market scenarios."""
        integrator = HighDimensionalModelingIntegrator()
        initial_prices = torch.ones(128) * 100
        encrypted_scenarios = integrator.generate_secure_market_scenarios(
            initial_prices, 10
        )

        self.assertIsNotNone(encrypted_scenarios)
        self.assertIn("ciphertext", encrypted_scenarios.__dict__)
        logging.info(f"Secure market scenarios generated successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
