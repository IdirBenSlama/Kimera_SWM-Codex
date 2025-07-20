#!/usr/bin/env python3
"""
Run the Mirror Portal Integration Test with proper error handling
"""

import subprocess
import sys
import os
from datetime import datetime

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def run_test():
    """Run the test and capture output"""
    logger.info("="*80)
    logger.info("🚀 KIMERA MIRROR PORTAL INTEGRATION TEST")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()
    logger.info()
    
    # Set Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + os.pathsep + env.get('PYTHONPATH', '')
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, 'test_mirror_portal_integration.py'],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Print output
        logger.info("STDOUT:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.info("\nSTDERR:")
            logger.info(result.stderr)
        
        logger.info(f"\nReturn code: {result.returncode}")
        
        # Check for log files
        import glob
        log_files = glob.glob("mirror_portal_test_*.log")
        json_files = glob.glob("mirror_portal_test_results_*.json")
        
        if log_files:
            logger.info(f"\n📄 Log files created: {', '.join(log_files)
        if json_files:
            logger.info(f"📊 Result files created: {', '.join(json_files)
            
    except Exception as e:
        logger.error(f"❌ Error running test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()