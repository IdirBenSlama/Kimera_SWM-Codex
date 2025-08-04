"""
Geometric and Aesthetic Optimization Integration
==============================================

This module integrates the GeoidMirrorPortalEngine and the GoldenRatioOptimizer
to create a unified system for geometric and aesthetic optimization.

Key Responsibilities:
- Provide a single interface for accessing geometric optimization functions.
- Manage the interaction between the portal engine and the optimizer.
- Ensure that all geometric operations are aligned with the golden ratio.
"""

import asyncio
import logging
from typing import Any, Dict

from .geoid_mirror_portal_engine import GeoidMirrorPortalEngine
from .golden_ratio_optimizer import GoldenRatioOptimizer, OptimizationDomain

logger = logging.getLogger(__name__)


class GeometricOptimizationIntegrator:
    """
    Integrates geometric and aesthetic optimization engines.
    """

    def __init__(self):
        self.portal_engine = GeoidMirrorPortalEngine()
        self.optimizer = GoldenRatioOptimizer()
        logger.info("🌀 Geometric Optimization Integrator initialized")

    async def create_optimized_mirror_portal(
        self, semantic_content: Dict[str, float], symbolic_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Creates a mirror portal that is optimized using the golden ratio.
        """
        # Create the portal
        semantic_geoid, symbolic_geoid, portal = (
            await self.portal_engine.create_dual_state_geoid(
                semantic_content, symbolic_content
            )
        )

        # Optimize the portal's coherence matrix
        coherence_matrix = torch.rand(
            (144, 144)
        )  # Placeholder for actual coherence matrix
        optimized_coherence = self.optimizer.optimize_portal_coherence(coherence_matrix)

        # Apply the optimized coherence to the portal (conceptual)
        portal.coherence_strength = optimized_coherence.mean().item()

        logger.info(f"Portal {portal.portal_id} optimized with golden ratio.")

        return {
            "portal_id": portal.portal_id,
            "semantic_geoid_id": semantic_geoid.geoid_id,
            "symbolic_geoid_id": symbolic_geoid.geoid_id,
            "optimized_coherence": portal.coherence_strength,
        }

    def get_global_optimization_metrics(self) -> Dict[str, float]:
        """
        Returns a dictionary of global optimization metrics.
        """
        return self.optimizer.get_optimization_metrics()


async def main():
    """
    Demonstration of the integrated geometric optimization system.
    """
    logging.basicConfig(level=logging.INFO)
    integrator = GeometricOptimizationIntegrator()

    semantic_content = {"meaning": 0.9, "beauty": 0.8}
    symbolic_content = {"form": "phyllotaxis", "pattern": "spiral"}

    result = await integrator.create_optimized_mirror_portal(
        semantic_content, symbolic_content
    )
    logger.info("Optimized Mirror Portal Created:")
    logger.info(result)

    metrics = integrator.get_global_optimization_metrics()
    logger.info("\nGlobal Optimization Metrics:")
    logger.info(metrics)


if __name__ == "__main__":
    import torch

    asyncio.run(main())
