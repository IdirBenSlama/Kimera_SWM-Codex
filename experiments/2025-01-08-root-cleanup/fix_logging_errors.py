#!/usr/bin/env python3
"""
Fix logging permission errors in Kimera system
"""
import os
import shutil
import time
from pathlib import Path

def fix_logging_errors():
    """Fix logging permission errors by cleaning up locked log files"""
    
    logger.info("🔧 Fixing Kimera logging permission errors...")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir.mkdir(exist_ok=True)
        logger.info("✅ Created logs directory")
        return
    
    # List of problematic log files
    problematic_logs = [
        "kimera.log",
        "kimera_database.log",
        "kimera_trading.log",
        "kimera_api.log"
    ]
    
    for log_file in problematic_logs:
        log_path = logs_dir / log_file
        if log_path.exists():
            try:
                # Try to rename the file to see if it's locked
                temp_path = logs_dir / f"{log_file}.temp"
                shutil.move(str(log_path), str(temp_path))
                shutil.move(str(temp_path), str(log_path))
                logger.info(f"✅ {log_file} - accessible")
            except PermissionError:
                logger.info(f"❌ {log_file} - locked by another process")
                logger.info(f"   Solution: Stop all Kimera processes and restart")
            except Exception as e:
                logger.info(f"⚠️ {log_file} - {e}")
    
    # Create backup log rotation script
    rotation_script = """
# Log rotation script for Kimera
import logging.handlers
import os
from pathlib import Path

def setup_safe_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Use TimedRotatingFileHandler with delay to prevent file locking
    handler = logging.handlers.TimedRotatingFileHandler(
        logs_dir / "kimera.log",
        when='midnight',
        interval=1,
        backupCount=7,
        delay=True  # Don't open file until first log
    )
    
    return handler
"""
    
    with open("safe_logging.py", "w") as f:
        f.write(rotation_script)
    
    logger.info("✅ Created safe logging configuration")
    logger.info("\n🚨 IMPORTANT: Stop all Kimera processes before live trading")
    logger.info("   - Close all terminals running Kimera")
    logger.info("   - Kill any background processes")
    logger.info("   - Restart system cleanly")

if __name__ == "__main__":
    fix_logging_errors() 