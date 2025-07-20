import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from backend.engines.foundational_thermodynamic_engine_fixed import FoundationalThermodynamicEngineFixed, ThermodynamicMode

class TestFoundationalThermodynamicEngine(unittest.TestCase):
    """
    Unit tests for the FoundationalThermodynamicEngineFixed.
    
    This test suite ensures that the core thermodynamic calculations adhere to
    fundamental physical laws, specifically the Second Law of Thermodynamics as
    it applies to Carnot efficiency.
    """

    def setUp(self):
        """Set up the test environment"""
        self.engine = FoundationalThermodynamicEngineFixed()
        self.hot_fields = [MagicMock(embedding=np.random.rand(768)) for _ in range(10)]
        self.cold_fields = [np.random.normal(4, 1) for _ in range(50)]

    @patch('backend.engines.foundational_thermodynamic_engine_fixed.logger')
    def test_zetetic_carnot_engine_physics_compliance(self, mock_logger):
        """
        ZETETIC VERDICT: Ensures the Carnot engine does not violate the 2nd Law of Thermodynamics.
        
        This test validates that the calculated actual efficiency of the engine
        is always less than or equal to the theoretical Carnot efficiency.
        This serves as a permanent safeguard against the physics violation
        identified in the ZETETIC AUDIT SCIENTIFIC REPORT.
        """
        # --- EXECUTE ---
        # Run the self-validating Zetetic Carnot engine cycle.
        result = self.engine.run_zetetic_carnot_engine(self.hot_fields, self.cold_fields)

        # --- VERIFY ---
        # 1. Primary Assertion: The cycle must be reported as physics-compliant.
        self.assertTrue(
            result.physics_compliant,
            "CRITICAL FAILURE: The engine reported a physics violation."
        )

        # 2. Secondary Assertion: The actual efficiency must NOT exceed the theoretical limit.
        self.assertLessEqual(
            result.actual_efficiency,
            result.theoretical_efficiency,
            f"CRITICAL PHYSICS VIOLATION: Actual efficiency ({result.actual_efficiency:.4f}) "
            f"exceeded theoretical Carnot limit ({result.theoretical_efficiency:.4f})."
        )
        
        # 3. Sanity Check: Ensure efficiency is within a reasonable range [0, 1].
        self.assertGreaterEqual(result.actual_efficiency, 0, "Efficiency cannot be negative.")
        self.assertLessEqual(result.actual_efficiency, 1, "Efficiency cannot be greater than 1.")

if __name__ == '__main__':
    unittest.main() 