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

def main():
    """Auto-launch Kimera autonomous trader"""
    print("🧠 AUTO-LAUNCHING KIMERA AUTONOMOUS TRADER")
    print("   Mode: FULLY AUTONOMOUS")
    print("   Safety Limits: NONE")
    print("   Auto-Confirmation: ENABLED")
    print()
    
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
        print("🚀 Sending auto-confirmation: 'UNLEASH KIMERA'")
        process.stdin.write(confirmation_input)
        process.stdin.flush()
        
        # Stream the output
        print("📊 KIMERA AUTONOMOUS TRADER OUTPUT:")
        print("=" * 60)
        
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            
            # Check for completion or error
            if "TARGET REACHED" in line:
                print("\n🎉 MISSION ACCOMPLISHED!")
                break
            elif "ERROR" in line or "FAILED" in line:
                print(f"\n❌ Error detected: {line.strip()}")
                break
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Auto-launch interrupted by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"\n❌ Auto-launch failed: {e}")

if __name__ == "__main__":
    main() 