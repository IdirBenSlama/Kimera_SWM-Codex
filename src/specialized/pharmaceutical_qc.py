"""
DEPRECATED: Use backend.core.quality_control.QualityControlSystem instead.
This file is retained for backward compatibility and will be removed in a future release.
"""

try:
    from core.quality_control import QualityControlSystem
except ImportError:
    # Create placeholders for core.quality_control
class QualityControlSystem:
    """Auto-generated class."""
    pass
        pass
