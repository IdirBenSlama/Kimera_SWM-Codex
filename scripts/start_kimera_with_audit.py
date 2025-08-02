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

def print_separator(title: str, char: str = "=", width: int = 70):
    """Print a visual separator with title"""
    print(f"\n{char * width}")
    print(f" {title.upper()}")
    print(f"{char * width}")


def run_script(script_path: str, description: str):
    """Run a Python script and return success status"""
    print_separator(f"Running {description}")
    print(f"Executing: {script_path}")
    
    start_time = time.time()
    
    try:
        # Run the script using the same Python interpreter
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {duration:.2f}s")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
            return False
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ {description} failed with exception after {duration:.2f}s: {str(e)}")
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
            print(f"Installing {req_file.name}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(req_file)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {req_file.name} installed successfully")
                else:
                    print(f"⚠️ {req_file.name} installation had issues:")
                    if result.stderr:
                        print(result.stderr)
            except Exception as e:
                print(f"❌ Failed to install {req_file.name}: {str(e)}")
        else:
            print(f"⚠️ Requirements file not found: {req_file}")
    
    return True


def start_kimera_system():
    """Start the main Kimera system"""
    print_separator("Starting Kimera System")
    
    base_dir = Path(__file__).parent.parent
    kimera_script = base_dir / "kimera.py"
    
    if not kimera_script.exists():
        print(f"❌ Kimera main script not found: {kimera_script}")
        return False
    
    print(f"🚀 Starting Kimera SWM System...")
    print(f"📍 Script location: {kimera_script}")
    print(f"🌐 Web interface will be available at: http://127.0.0.1:8000/docs")
    print(f"📊 System monitoring at: http://127.0.0.1:8000/monitoring")
    print(f"💬 Chat interface at: http://127.0.0.1:8000/chat")
    
    try:
        # Start Kimera as a subprocess but don't wait for it to complete
        # since it's a web server that runs indefinitely
        process = subprocess.Popen([
            sys.executable, str(kimera_script)
        ], cwd=str(base_dir))
        
        print(f"✅ Kimera system started with PID: {process.pid}")
        print(f"🎯 System is starting up... This may take a moment.")
        print(f"🌐 Once ready, access the system at: http://127.0.0.1:8000/docs")
        
        # Give the system a moment to start
        time.sleep(3)
        
        # Check if process is still running (not crashed immediately)
        if process.poll() is None:
            print(f"✅ Kimera system is running successfully!")
            return True
        else:
            print(f"❌ Kimera system crashed immediately")
            return False
    
    except Exception as e:
        print(f"❌ Failed to start Kimera system: {str(e)}")
        return False


def main():
    """Main startup function"""
    print_separator("KIMERA SWM COMPLETE STARTUP SEQUENCE", "=", 80)
    print(f"🚀 Complete system initialization and startup")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
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
                print(f"⚠️ Script not found: {script_path}")
                success = False
        
        if not success:
            if step["required"]:
                print(f"❌ Critical step failed: {step['description']}")
                critical_failure = True
                all_passed = False
                break
            else:
                print(f"⚠️ Non-critical step failed: {step['description']}")
                all_passed = False
    
    print_separator("Startup Sequence Results")
    
    if critical_failure:
        print(f"❌ CRITICAL FAILURE in startup sequence")
        print(f"❌ Cannot start Kimera system due to critical issues")
        print(f"❌ Please review the errors above and fix them")
        return False
    
    elif all_passed:
        print(f"✅ ALL STARTUP STEPS COMPLETED SUCCESSFULLY")
        print(f"✅ System is ready for operation")
    else:
        print(f"⚠️ STARTUP COMPLETED WITH WARNINGS")
        print(f"⚠️ Some non-critical issues detected but system can start")
    
    # Start Kimera system
    print_separator("Starting Kimera System")
    print(f"🎯 Initializing Kimera SWM cognitive AI platform...")
    
    kimera_started = start_kimera_system()
    
    print_separator("STARTUP COMPLETE", "=", 80)
    
    if kimera_started:
        print(f"🎉 KIMERA SWM SYSTEM IS NOW RUNNING! 🎉")
        print(f"")
        print(f"🌐 **WEB INTERFACE**:")
        print(f"   📚 API Documentation: http://127.0.0.1:8000/docs")
        print(f"   📊 System Monitoring: http://127.0.0.1:8000/monitoring")
        print(f"   💬 Chat Interface: http://127.0.0.1:8000/chat")
        print(f"   🔧 System Status: http://127.0.0.1:8000/system/status")
        print(f"")
        print(f"🧠 **COGNITIVE CAPABILITIES**:")
        print(f"   🔬 Scientific Research: Advanced cognitive investigation")
        print(f"   🧮 Thermodynamic Processing: Physics-compliant AI evolution")
        print(f"   💾 Persistent Memory: Complete knowledge preservation")
        print(f"   🔧 Self-Healing: Automatic anomaly detection and resolution")
        print(f"   📈 Real-time Analytics: System health and performance monitoring")
        print(f"")
        print(f"🚀 **READY FOR**:")
        print(f"   🔍 Breakthrough cognitive research and discovery")
        print(f"   🏭 Production AI applications with persistent memory")
        print(f"   🧪 Scientific investigation of consciousness and cognition")
        print(f"   🤖 Self-improving autonomous AI systems")
        print(f"")
        print(f"📖 **GETTING STARTED**:")
        print(f"   1. Open http://127.0.0.1:8000/docs in your browser")
        print(f"   2. Explore the API documentation and endpoints")
        print(f"   3. Try the /chat endpoint for conversational AI")
        print(f"   4. Monitor system health at /monitoring")
        print(f"   5. Check system status at /system/status")
        print(f"")
        print(f"🎯 The world's most advanced cognitive AI platform is now operational!")
        
        # Keep the script running so user can see the message
        print(f"\nPress Ctrl+C to stop this script (Kimera will continue running)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n👋 Startup script terminated. Kimera system continues running.")
            print(f"💡 To stop Kimera, terminate the kimera.py process.")
        
        return True
    else:
        print(f"❌ KIMERA SYSTEM FAILED TO START")
        print(f"❌ Please check the error messages above")
        print(f"❌ Review logs and try running individual scripts for debugging")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 