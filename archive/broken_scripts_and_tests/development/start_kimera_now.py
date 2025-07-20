#!/usr/bin/env python3
"""
Start Kimera System - Simple Direct Launch
"""

import subprocess
import sys
import os

print("="*80)
print("🚀 STARTING KIMERA SYSTEM")
print("="*80)

# Try different ways to start Kimera
options = [
    # Try the main FastAPI server
    ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8001"],
    
    # Try the simple test
    ["python", "simple_kimera_test.py"],
    
    # Try the final demo
    ["python", "final_kimera_demo.py"],
    
    # Try the web chat
    ["python", "kimera_web_chat.py"],
    
    # Try direct backend import
    ["python", "-c", "from backend.api.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8001)"],
]

for i, cmd in enumerate(options):
    print(f"\nAttempt {i+1}: Running {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        break  # If successful, stop trying
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
    except FileNotFoundError:
        print(f"❌ Command not found")
    except KeyboardInterrupt:
        print("\n✋ Stopped by user")
        break
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*80)
print("If all attempts failed, try:")
print("1. cd to the Kimera directory")
print("2. Activate virtual environment: .venv\\Scripts\\activate")
print("3. Install dependencies: pip install -r requirements.txt")
print("4. Run: python -m backend.api.main")
print("="*80)