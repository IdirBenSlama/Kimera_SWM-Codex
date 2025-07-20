# This file makes the 'routers' directory a Python package and exports all routers

# Import all router modules to make them available for import
from . import geoid_scar_router
from . import system_router
from . import contradiction_router
from . import vault_router
from . import insight_router
from . import statistics_router
from . import output_analysis_router
from . import core_actions_router
from . import thermodynamic_router
from . import omnidimensional_router

# Make the router objects directly accessible
__all__ = [
    'geoid_scar_router',
    'system_router', 
    'contradiction_router',
    'vault_router',
    'insight_router',
    'statistics_router',
    'output_analysis_router',
    'core_actions_router',
    'thermodynamic_router',
    'omnidimensional_router'
] 