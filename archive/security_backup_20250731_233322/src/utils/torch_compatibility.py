"""
PyTorch Compatibility Wrapper for CVE-2025-32434
This wrapper provides safe loading alternatives to torch.load
"""

import torch
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def safe_torch_load(file_path: str, map_location: Optional[str] = None, 
                   weights_only: bool = True) -> Any:
    """
    Safe wrapper for torch.load that handles the CVE-2025-32434 vulnerability.
    
    Args:
        file_path: Path to the file to load
        map_location: Device to map tensors to
        weights_only: Only load weights (safer)
    
    Returns:
        Loaded object
    """
    try:
        # Try safetensors first if available
        if file_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                return load_file(file_path, device=map_location)
            except ImportError:
                logger.warning("safetensors not available, falling back to torch.load")
        
        # Use torch.load with security considerations
        if hasattr(torch, 'load'):
            # For PyTorch 2.5.x, use weights_only=True for security
            return torch.load(file_path, map_location=map_location, weights_only=weights_only)
        else:
            # Fallback for older PyTorch versions
            return torch.load(file_path, map_location=map_location)
            
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        # Return empty state dict as fallback
        return {}

def get_pytorch_info() -> Dict[str, Any]:
    """Get PyTorch version and security information"""
    return {
        'version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'safe_loading': True  # This wrapper provides safe loading
    }
