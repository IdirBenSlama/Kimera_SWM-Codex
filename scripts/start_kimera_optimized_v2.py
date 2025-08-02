"""
Kimera SWM Optimized Startup v2
===============================
Enhanced startup script with all optimizations.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup environment
os.environ['KIMERA_MODE'] = 'optimized'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async GPU execution

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def pre_startup_checks():
    """Run pre-startup system checks"""
    logger.info("Running pre-startup checks...")
    
    # Check database
    db_path = project_root / "data" / "database" / "kimera.db"
    if not db_path.exists():
        logger.warning("Database not found - will be created on startup")
        
    # Check GPU
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"✅ GPU available: {device_count} device(s)")
    except:
        logger.warning("⚠️ GPU not available - using CPU mode")
        
    # Check memory
    import psutil
    mem = psutil.virtual_memory()
    if mem.available < 4 * (1024**3):  # Less than 4GB
        logger.warning(f"⚠️ Low memory: {mem.available / (1024**3):.1f}GB available")
        
def apply_runtime_patches():
    """Apply runtime patches and optimizations"""
    logger.info("Applying runtime optimizations...")
    
    # Import patches
    try:
        from src.core.unified_master_cognitive_architecture_fix import patch_unified_architecture
        patch_unified_architecture()
        logger.info("✅ Architecture patches applied")
    except Exception as e:
        logger.warning(f"Architecture patch failed: {e}")
        
    # Import optimizers
    try:
        from src.utils.memory_optimizer import memory_optimizer
        memory_optimizer.optimize_memory()
        logger.info("✅ Memory optimizer initialized")
    except Exception as e:
        logger.warning(f"Memory optimizer failed: {e}")
        
    try:
        from src.engines.thermodynamic_efficiency_optimizer import efficiency_optimizer
        logger.info("✅ Efficiency optimizer loaded")
    except Exception as e:
        logger.warning(f"Efficiency optimizer failed: {e}")
        
def start_kimera():
    """Start Kimera with optimizations"""
    logger.info("="*60)
    logger.info("🚀 STARTING KIMERA SWM (OPTIMIZED v2)")
    logger.info("="*60)
    
    # Run checks
    pre_startup_checks()
    
    # Apply patches
    apply_runtime_patches()
    
    # Start main application
    logger.info("\n🌟 Launching Kimera...")
    start_time = time.time()
    
    try:
        from src.main import main
        main()
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    elapsed = time.time() - start_time
    logger.info(f"✅ Kimera started in {elapsed:.2f} seconds")
    

if __name__ == "__main__":
    start_kimera()
