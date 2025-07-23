"""
Layer 2 Governance Module
Provides monitoring and security capabilities for Kimera SWM
"""

# Re-export monitoring and security modules from their actual locations
import sys
from pathlib import Path

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import and re-export modules
from src import monitoring, security

# Make monitoring and security available as submodules
sys.modules['src.layer_2_governance.monitoring'] = monitoring
sys.modules['src.layer_2_governance.security'] = security

__all__ = ['monitoring', 'security'] 