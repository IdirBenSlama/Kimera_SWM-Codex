# This file makes the 'routers' directory a Python package and exports all routers

# Import all router modules to make them available for import
from . import (
    contradiction_router,
    core_actions_router,
    geoid_scar_router,
    insight_router,
    omnidimensional_router,
    output_analysis_router,
    statistics_router,
    system_router,
    thermodynamic_router,
    vault_router,
)

# Make the router objects directly accessible
__all__ = [
    "geoid_scar_router",
    "system_router",
    "contradiction_router",
    "vault_router",
    "insight_router",
    "statistics_router",
    "output_analysis_router",
    "core_actions_router",
    "thermodynamic_router",
    "omnidimensional_router",
]
