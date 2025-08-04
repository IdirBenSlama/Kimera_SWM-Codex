#!/usr/bin/env python3
"""
KIMERA AUTONOMOUS TRADER - AUTO LAUNCH
=====================================

Automatically launches and confirms the autonomous Kimera trader
without requiring manual input.
"""

import subprocess
import sys
import os
import logging
logger = logging.getLogger(__name__)

def main():
    """Auto-launch Kimera autonomous trader"""
    logger.info("🧠 AUTO-LAUNCHING KIMERA AUTONOMOUS TRADER")
    logger.info("   Mode: FULLY AUTONOMOUS")
    logger.info("   Safety Limits: NONE")
    logger.info("   Auto-Confirmation: ENABLED")
    logger.info()
    
    # Create the confirmation input
    confirmation_input = "UNLEASH KIMERA\n"
    
    try:
        # Launch the autonomous trader with auto-confirmation
        process = subprocess.Popen(
            [sys.executable, "start_autonomous_kimera.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Send the confirmation automatically
        logger.info("🚀 Sending auto-confirmation: 'UNLEASH KIMERA'")
        process.stdin.write(confirmation_input)
        process.stdin.flush()
        
        # Stream the output
        logger.info("📊 KIMERA AUTONOMOUS TRADER OUTPUT:")
        logger.info("=" * 60)
        
        for line in iter(process.stdout.readline, ''):
            logger.info(line.rstrip())
            
            # Check for completion or error
            if "TARGET REACHED" in line:
                logger.info("\n🎉 MISSION ACCOMPLISHED!")
                break
            elif "ERROR" in line or "FAILED" in line:
                logger.info(f"\n❌ Error detected: {line.strip()}")
                break
        
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Auto-launch interrupted by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        logger.info(f"\n❌ Auto-launch failed: {e}")

if __name__ == "__main__":
    main() 