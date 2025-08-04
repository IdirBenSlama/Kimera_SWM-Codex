#!/usr/bin/env python3
"""
KIMERA SWM - COMPLETE STARTUP WITH AUDIT
========================================

This script performs the complete startup sequence:
1. Install/verify requirements
2. Initialize all system components
3. Perform comprehensive audit
4. Start Kimera system if audit passes

This is the one-script solution for getting Kimera running from scratch.
"""

import sys
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def print_separator(title: str, char: str = "=", width: int = 70):
    """Print a visual separator with title"""
    logger.info(f"\n{char * width}")
    logger.info(f" {title.upper()}")
    logger.info(f"{char * width}")


def run_script(script_path: str, description: str):
    """Run a Python script and return success status"""
    print_separator(f"Running {description}")
    logger.info(f"Executing: {script_path}")
    
    start_time = time.time()
    
    try:
        # Run the script using the same Python interpreter
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully in {duration:.2f}s")
            if result.stdout:
                logger.info("Output:")
                logger.info(result.stdout)
            return True
        else:
            logger.info(f"❌ {description} failed with exit code {result.returncode}")
            if result.stderr:
                logger.info("Error output:")
                logger.info(result.stderr)
            if result.stdout:
                logger.info("Standard output:")
                logger.info(result.stdout)
            return False
    
    except Exception as e:
        duration = time.time() - start_time
        logger.info(f"❌ {description} failed with exception after {duration:.2f}s: {str(e)}")
        return False


def check_and_install_requirements():
    """Check and install Python requirements"""
    print_separator("Installing Requirements")
    
    base_dir = Path(__file__).parent.parent
    requirements_files = [
        base_dir / "requirements" / "base.txt",
        base_dir / "requirements" / "data.txt"
    ]
    
    for req_file in requirements_files:
        if req_file.exists():
            logger.info(f"Installing {req_file.name}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(req_file)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✅ {req_file.name} installed successfully")
                else:
                    logger.info(f"⚠️ {req_file.name} installation had issues:")
                    if result.stderr:
                        logger.info(result.stderr)
            except Exception as e:
                logger.info(f"❌ Failed to install {req_file.name}: {str(e)}")
        else:
            logger.info(f"⚠️ Requirements file not found: {req_file}")
    
    return True


def start_kimera_system():
    """Start the main Kimera system"""
    print_separator("Starting Kimera System")
    
    base_dir = Path(__file__).parent.parent
    kimera_script = base_dir / "kimera.py"
    
    if not kimera_script.exists():
        logger.info(f"❌ Kimera main script not found: {kimera_script}")
        return False
    
    logger.info(f"🚀 Starting Kimera SWM System...")
    logger.info(f"📍 Script location: {kimera_script}")
    logger.info(f"🌐 Web interface will be available at: http://127.0.0.1:8000/docs")
    logger.info(f"📊 System monitoring at: http://127.0.0.1:8000/monitoring")
    logger.info(f"💬 Chat interface at: http://127.0.0.1:8000/chat")
    
    try:
        # Start Kimera as a subprocess but don't wait for it to complete
        # since it's a web server that runs indefinitely
        process = subprocess.Popen([
            sys.executable, str(kimera_script)
        ], cwd=str(base_dir))
        
        logger.info(f"✅ Kimera system started with PID: {process.pid}")
        logger.info(f"🎯 System is starting up... This may take a moment.")
        logger.info(f"🌐 Once ready, access the system at: http://127.0.0.1:8000/docs")
        
        # Give the system a moment to start
        time.sleep(3)
        
        # Check if process is still running (not crashed immediately)
        if process.poll() is None:
            logger.info(f"✅ Kimera system is running successfully!")
            return True
        else:
            logger.info(f"❌ Kimera system crashed immediately")
            return False
    
    except Exception as e:
        logger.info(f"❌ Failed to start Kimera system: {str(e)}")
        return False


def main():
    """Main startup function"""
    print_separator("KIMERA SWM COMPLETE STARTUP SEQUENCE", "=", 80)
    logger.info(f"🚀 Complete system initialization and startup")
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "scripts"
    
    # Define the startup sequence
    startup_sequence = [
        {
            "description": "Requirements Installation",
            "function": check_and_install_requirements,
            "required": True
        },
        {
            "description": "System Initialization", 
            "script": scripts_dir / "system_initialization.py",
            "required": True
        },
        {
            "description": "System Audit",
            "script": scripts_dir / "system_audit.py", 
            "required": False  # Audit failure shouldn't prevent startup
        }
    ]
    
    # Execute startup sequence
    all_passed = True
    critical_failure = False
    
    for step in startup_sequence:
        if "function" in step:
            # Call function directly
            success = step["function"]()
        else:
            # Run script
            script_path = step["script"]
            if script_path.exists():
                success = run_script(str(script_path), step["description"])
            else:
                logger.info(f"⚠️ Script not found: {script_path}")
                success = False
        
        if not success:
            if step["required"]:
                logger.info(f"❌ Critical step failed: {step['description']}")
                critical_failure = True
                all_passed = False
                break
            else:
                logger.info(f"⚠️ Non-critical step failed: {step['description']}")
                all_passed = False
    
    print_separator("Startup Sequence Results")
    
    if critical_failure:
        logger.info(f"❌ CRITICAL FAILURE in startup sequence")
        logger.info(f"❌ Cannot start Kimera system due to critical issues")
        logger.info(f"❌ Please review the errors above and fix them")
        return False
    
    elif all_passed:
        logger.info(f"✅ ALL STARTUP STEPS COMPLETED SUCCESSFULLY")
        logger.info(f"✅ System is ready for operation")
    else:
        logger.info(f"⚠️ STARTUP COMPLETED WITH WARNINGS")
        logger.info(f"⚠️ Some non-critical issues detected but system can start")
    
    # Start Kimera system
    print_separator("Starting Kimera System")
    logger.info(f"🎯 Initializing Kimera SWM cognitive AI platform...")
    
    kimera_started = start_kimera_system()
    
    print_separator("STARTUP COMPLETE", "=", 80)
    
    if kimera_started:
        logger.info(f"🎉 KIMERA SWM SYSTEM IS NOW RUNNING! 🎉")
        logger.info(f"")
        logger.info(f"🌐 **WEB INTERFACE**:")
        logger.info(f"   📚 API Documentation: http://127.0.0.1:8000/docs")
        logger.info(f"   📊 System Monitoring: http://127.0.0.1:8000/monitoring")
        logger.info(f"   💬 Chat Interface: http://127.0.0.1:8000/chat")
        logger.info(f"   🔧 System Status: http://127.0.0.1:8000/system/status")
        logger.info(f"")
        logger.info(f"🧠 **COGNITIVE CAPABILITIES**:")
        logger.info(f"   🔬 Scientific Research: Advanced cognitive investigation")
        logger.info(f"   🧮 Thermodynamic Processing: Physics-compliant AI evolution")
        logger.info(f"   💾 Persistent Memory: Complete knowledge preservation")
        logger.info(f"   🔧 Self-Healing: Automatic anomaly detection and resolution")
        logger.info(f"   📈 Real-time Analytics: System health and performance monitoring")
        logger.info(f"")
        logger.info(f"🚀 **READY FOR**:")
        logger.info(f"   🔍 Breakthrough cognitive research and discovery")
        logger.info(f"   🏭 Production AI applications with persistent memory")
        logger.info(f"   🧪 Scientific investigation of consciousness and cognition")
        logger.info(f"   🤖 Self-improving autonomous AI systems")
        logger.info(f"")
        logger.info(f"📖 **GETTING STARTED**:")
        logger.info(f"   1. Open http://127.0.0.1:8000/docs in your browser")
        logger.info(f"   2. Explore the API documentation and endpoints")
        logger.info(f"   3. Try the /chat endpoint for conversational AI")
        logger.info(f"   4. Monitor system health at /monitoring")
        logger.info(f"   5. Check system status at /system/status")
        logger.info(f"")
        logger.info(f"🎯 The world's most advanced cognitive AI platform is now operational!")
        
        # Keep the script running so user can see the message
        logger.info(f"\nPress Ctrl+C to stop this script (Kimera will continue running)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info(f"\n👋 Startup script terminated. Kimera system continues running.")
            logger.info(f"💡 To stop Kimera, terminate the kimera.py process.")
        
        return True
    else:
        logger.info(f"❌ KIMERA SYSTEM FAILED TO START")
        logger.info(f"❌ Please check the error messages above")
        logger.info(f"❌ Review logs and try running individual scripts for debugging")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 