"""
KIMERA SWM Core Module - Organized Architecture
==============================================

Aerospace-grade organization of core components following functional categorization.

Directory Structure:
- Architecture: src/core/architecture/
- Async Operations: src/core/async_operations/
- Cognitive: src/core/cognitive/
- Context: src/core/context/
- Data: src/core/data/
- Ethics: src/core/ethics/
- Output: src/core/output/
- Primitives: src/core/primitives/
- Processing: src/core/processing/
- Security: src/core/security/
- System: src/core/system/
- Uncategorized: src/core/uncategorized/
- Universal: src/core/universal/
- Vault: src/core/vault/

Author: Kimera SWM Autonomous Architect
Date: 2025-08-04
Version: 3.1.0 (Organized Architecture)
"""

__version__ = "3.1.0"
__status__ = "Production"

# Architecture components
from .architecture.interfaces import *
# Context management
# from .context.context_supremacy import ContextSupremacyEngine  # Temporarily disabled due to syntax issues
# Processing foundations
# from .processing.quality_control import *  # Temporarily disabled due to syntax issues
# Organized module imports for backward compatibility
# Core system components
from .system.kimera_system_clean import KimeraSystem, get_kimera_system, kimera_singleton
# Universal systems
# from .universal.universal_compassion import UniversalCompassionEngine  # Temporarily disabled due to syntax errors

# Export key components
__all__ = [
    "KimeraSystem",
    "get_kimera_system",
    "kimera_singleton",
    # "ContextSupremacyEngine",
    # "UniversalCompassionEngine",
]
