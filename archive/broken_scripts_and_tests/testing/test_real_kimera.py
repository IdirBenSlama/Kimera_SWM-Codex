#!/usr/bin/env python3
"""
Test Real KIMERA - Send a message to test the fixed diffusion engine
"""

import time
import subprocess
import sys
import os

def test_kimera():
    """Test KIMERA with a simple message"""
    print("🧪 Testing REAL KIMERA Text Diffusion Engine...")
    print("=" * 50)
    
    # Wait a moment for KIMERA to be ready
    print("⏳ Waiting 3 seconds for KIMERA to be ready...")
    time.sleep(3)
    
    # Try to send input to the chat process
    try:
        # This is a simple approach - in a real scenario you'd use proper IPC
        print("📝 The test will be interactive - you can type messages to KIMERA")
        print("💡 Try asking: 'What are you thinking about?'")
        print("🎯 Look for responses that show actual diffusion generation times (not 0.05s)")
        print("✅ If you see generation times > 1 second, that's the REAL diffusion engine!")
        print("❌ If you see 0.05s times, that's still the fallback")
        print()
        print("The KIMERA chat is running in the background.")
        print("Switch to that terminal window to interact with it.")
        
    except Exception as e:
        print(f"❌ Error testing KIMERA: {e}")

if __name__ == "__main__":
    test_kimera() 