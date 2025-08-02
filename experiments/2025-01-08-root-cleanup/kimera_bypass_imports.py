#!/usr/bin/env python3
"""
KIMERA SWM - Bypass Import Issues and Launch
============================================

This script modifies problematic imports to get Kimera running.
"""

import os
import sys
from pathlib import Path

def patch_monitoring_core():
    """Patch the monitoring core to remove problematic imports"""
    monitoring_file = Path("backend/monitoring/kimera_monitoring_core.py")
    
    if monitoring_file.exists():
        content = monitoring_file.read_text(encoding='utf-8')
        
        # Comment out problematic imports
        replacements = [
            ("from memory_profiler import profile as memory_profile", "# from memory_profiler import profile as memory_profile\nmemory_profile = lambda x: x"),
            ("import GPUtil", "# import GPUtil\nGPUtil = None"),
            ("from py3nvml import py3nvml", "# from py3nvml import py3nvml\npy3nvml = None"),
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        # Create backup
        backup_file = monitoring_file.with_suffix('.py.bak')
        monitoring_file.rename(backup_file)
        
        # Write patched version
        monitoring_file.write_text(content, encoding='utf-8')
        print(f"✓ Patched {monitoring_file}")

def patch_distributed_tracing():
    """Patch distributed tracing to handle missing imports"""
    tracing_file = Path("backend/monitoring/distributed_tracing.py")
    
    if tracing_file.exists():
        content = tracing_file.read_text(encoding='utf-8')
        
        # Add try-except blocks around imports
        if "from opentelemetry.instrumentation.httpx" not in content.split('\n')[0]:
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                if line.strip().startswith("from opentelemetry.instrumentation.httpx"):
                    new_lines.append("try:")
                    new_lines.append(f"    {line}")
                    new_lines.append("except ImportError:")
                    new_lines.append("    HTTPXClientInstrumentor = None")
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
            
            # Create backup
            backup_file = tracing_file.with_suffix('.py.bak')
            if not backup_file.exists():
                tracing_file.rename(backup_file)
            
            # Write patched version
            tracing_file.write_text(content, encoding='utf-8')
            print(f"✓ Patched {tracing_file}")

def install_critical_missing():
    """Install only the most critical missing packages"""
    import subprocess
    
    venv_pip = Path("venv_py313/Scripts/pip")
    
    critical_packages = [
        "memory-profiler",
        "py-cpuinfo",
        "nvidia-ml-py3",
    ]
    
    for package in critical_packages:
        try:
            subprocess.run([str(venv_pip), "install", package], 
                         capture_output=True, check=False)
        except Exception as e:
            logger.error(f"Error in kimera_bypass_imports.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           KIMERA SWM - BYPASS IMPORTS AND LAUNCH                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Install critical packages
    print("📦 Installing critical packages...")
    install_critical_missing()
    
    # Patch problematic files
    print("\n🔧 Patching import issues...")
    patch_monitoring_core()
    patch_distributed_tracing()
    
    # Now try to run Kimera
    print("\n🚀 Starting Kimera...")
    print("=" * 60)
    
    import subprocess
    venv_python = Path("venv_py313/Scripts/python.exe")
    
    try:
        subprocess.run([str(venv_python), "kimera.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Kimera server stopped.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()