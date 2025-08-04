#!/usr/bin/env python3
"""
Restart KIMERA Server
"""

import os
import sys
import time
import subprocess
import psutil
import signal
import logging
logger = logging.getLogger(__name__)

def find_kimera_processes():
    """Find all KIMERA-related processes"""
    kimera_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('kimera' in str(arg).lower() for arg in cmdline):
                kimera_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return kimera_processes

def stop_kimera():
    """Stop all KIMERA processes"""
    logger.info("🛑 Stopping KIMERA processes...")
    
    processes = find_kimera_processes()
    if not processes:
        logger.info("   No KIMERA processes found")
        return
    
    for proc in processes:
        try:
            logger.info(f"   Terminating PID {proc.pid}: {proc.name()}")
            proc.terminate()
        except Exception as e:
            logger.info(f"   Error terminating {proc.pid}: {e}")
    
    # Wait for processes to terminate
    time.sleep(2)
    
    # Force kill if still running
    for proc in processes:
        try:
            if proc.is_running():
                logger.info(f"   Force killing PID {proc.pid}")
                proc.kill()
        except Exception as e:
            logger.error(f"Error in restart_kimera.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling

def start_kimera():
    """Start KIMERA server"""
    logger.info("\n🚀 Starting KIMERA server...")
    
    # Use the Python 3.13 virtual environment
    venv_python = os.path.join("venv_py313", "Scripts", "python.exe")
    
    if not os.path.exists(venv_python):
        logger.info("❌ Virtual environment not found!")
        return False
    
    # Start KIMERA in a new process
    try:
        subprocess.Popen([venv_python, "kimera.py"], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
        logger.info("✅ KIMERA server started in new console")
        return True
    except Exception as e:
        logger.info(f"❌ Failed to start KIMERA: {e}")
        return False

def main():
    logger.info("""
╔════════════════════════════════════════════════════��═════════════╗
║                    KIMERA SERVER RESTART                          ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Stop existing processes
    stop_kimera()
    
    # Wait a moment
    time.sleep(1)
    
    # Start new instance
    if start_kimera():
        logger.info("\n✅ KIMERA restart complete!")
        logger.info("   The server is starting in a new console window")
        logger.info("   Wait a few seconds for it to fully initialize")
    else:
        logger.info("\n❌ KIMERA restart failed!")

if __name__ == "__main__":
    main()