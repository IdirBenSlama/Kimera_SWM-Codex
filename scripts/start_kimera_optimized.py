#!/usr/bin/env python3
"""
Optimized Kimera Startup Script
===============================
Starts Kimera with all fixes and optimizations applied.
"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ['KIMERA_MODE'] = 'progressive'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and run
if __name__ == "__main__":
    logger.info("Starting Kimera SWM (Optimized)")
    logger.info("=" * 50)
    
    # Import fixes
    try:
        from src.core.unified_master_cognitive_architecture_fix import patch_unified_architecture
        logger.info("Architecture patches loaded")
    except Exception as e:
        logger.info(f"Architecture patches not found: {e}")
    
    try:
        from src.core.gpu.gpu_optimizer import gpu_optimizer
        logger.info(f"GPU optimizer loaded: {gpu_optimizer.device}")
    except Exception as e:
        logger.info(f"GPU optimizer not found: {e}")
    
    # Start Kimera
    from src.main import main
import logging
logger = logging.getLogger(__name__)
    main()
