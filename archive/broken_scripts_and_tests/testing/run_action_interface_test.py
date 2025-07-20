#!/usr/bin/env python3
"""
Simple runner script for KIMERA Action Interface Test

This script runs the action interface test with proper environment setup.
"""

import sys
import os
import asyncio
import subprocess

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def main():
    """Run the action interface test."""
    logger.info("🚀 KIMERA Action Interface Test Runner")
    logger.info("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('backend'):
        logger.error("❌ Error: 'backend' directory not found.")
        logger.info("💡 Please run this script from the project root directory.")
        return 1
    
    # Check if the action interface module exists
    action_interface_path = os.path.join('backend', 'trading', 'execution', 'kimera_action_interface.py')
    if not os.path.exists(action_interface_path):
        logger.error(f"❌ Error: Action interface module not found at {action_interface_path}")
        return 1
    
    logger.info("✅ Environment check passed")
    logger.info("🎯 Starting action interface test...")
    logger.info()
    
    try:
        # Run the test
        result = subprocess.run([
            sys.executable, 
            'test_action_interface_runner.py'
        ], capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            logger.info(result.stdout)
        
        if result.stderr:
            logger.info("STDERR:", result.stderr)
        
        return result.returncode
        
    except Exception as e:
        logger.error(f"❌ Failed to run test: {str(e)
        return 1

if __name__ == "__main__":
    exit_code = main()
    logger.info(f"\n🏁 Test completed with exit code: {exit_code}")
    sys.exit(exit_code) 