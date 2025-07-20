#!/usr/bin/env python3
"""
Execute the server test with proper output handling
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def run_server_test():
    """Run the server test and capture output"""
    logger.info("="*80)
    logger.info("🚀 KIMERA SERVER TEST EXECUTION")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now()
    logger.info()
    
    # Set environment
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    # Log file
    log_file = f"server_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger.info(f"📝 Logging to: {log_file}")
    logger.info("🔄 Running test_server_startup.py...")
    logger.info()
    
    try:
        # Run the test
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                [sys.executable, 'test_server_startup.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    logger.info(line.rstrip()
                    log.write(line)
                    log.flush()
            
            # Wait for completion
            process.wait()
            
            if process.returncode == 0:
                logger.info("\n✅ Server test completed successfully!")
            else:
                logger.warning(f"\n⚠️ Server test exited with code: {process.returncode}")
                
    except KeyboardInterrupt:
        logger.info("\n\n⏹️ Test interrupted by user")
        if 'process' in locals():
            process.terminate()
            process.wait()
    except Exception as e:
        logger.error(f"\n❌ Error running test: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"\n📄 Full log saved to: {log_file}")
    
    # Show summary
    logger.info("\n" + "="*80)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*80)
    logger.info("The test attempted to:")
    logger.info("  1. Start KIMERA server on port 8001")
    logger.info("  2. Test health endpoint")
    logger.info("  3. Create a test geoid")
    logger.info("  4. Check system status")
    logger.info("  5. Test Mirror Portal integration")
    logger.info("\nCheck the log file for detailed results.")

if __name__ == "__main__":
    run_server_test()