import subprocess
import sys
import os

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


logger.info("Running KIMERA Mirror Portal Test...")
logger.info("="*60)

# Set environment
env = os.environ.copy()
env['PYTHONPATH'] = os.getcwd()

# Run the test
try:
    result = subprocess.run(
        [sys.executable, 'test_kimera_portal.py'],
        capture_output=True,
        text=True,
        env=env
    )
    
    logger.info(result.stdout)
    if result.stderr:
        logger.error("\nErrors:")
        logger.info(result.stderr)
        
except Exception as e:
    logger.error(f"Error: {e}")