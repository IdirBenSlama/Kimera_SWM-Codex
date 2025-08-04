"""
Unified Master Cognitive Architecture Fix
=========================================
Patches the initialization issues in the unified architecture.
"""

def patch_unified_architecture():
    """Patch the unified architecture to fix initialization"""
    try:
        import sys
        from pathlib import Path
        
        # Patch the UnifiedMasterCognitiveArchitecture class
        src_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(src_path))
        
        try:
            from src.core.unified_master_cognitive_architecture import UnifiedMasterCognitiveArchitecture
import logging
logger = logging.getLogger(__name__)
        except ImportError as e:
            logger.info(f"Cannot import UnifiedMasterCognitiveArchitecture: {e}")
            return False
        
        # Save original __init__
        original_init = UnifiedMasterCognitiveArchitecture.__init__
        
        def patched_init(self, mode="progressive", **kwargs):
            # Remove the problematic enable_experimental parameter
            if 'enable_experimental' in kwargs:
                kwargs.pop('enable_experimental')
            
            # Call original with fixed parameters
            try:
                original_init(self, mode=mode, **kwargs)
            except TypeError:
                # If still fails, try without mode
                original_init(self, **kwargs)
        
        # Apply patch
        UnifiedMasterCognitiveArchitecture.__init__ = patched_init
        
        logger.info("Patched UnifiedMasterCognitiveArchitecture initialization")
        return True
        
    except Exception as e:
        logger.info(f"Failed to patch architecture: {e}")
        return False

# Auto-patch on import
patch_unified_architecture()
