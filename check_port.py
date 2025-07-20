#!/usr/bin/env python3
"""
Port Availability Checker for Kimera SWM
----------------------------------------
This script checks if the default port 8000 is available and suggests alternatives.
"""

import socket
import sys
from typing import List

def check_port(port: int) -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        if check_port(port):
            return port
    return None

def main():
    """Main function to check port availability."""
    print("🔍 Checking port availability for Kimera SWM...")
    
    # Check default port
    default_port = 8000
    if check_port(default_port):
        print(f"✅ Port {default_port} is available")
        print(f"🚀 You can start Kimera with: python start_kimera.py")
        return
    
    print(f"❌ Port {default_port} is already in use")
    
    # Find alternative ports
    alternative_ports = [8001, 8002, 8003, 8004, 8005, 8080, 8081, 8082, 9000, 9001]
    available_ports = []
    
    for port in alternative_ports:
        if check_port(port):
            available_ports.append(port)
    
    if available_ports:
        print(f"✅ Available alternative ports: {available_ports}")
        recommended_port = available_ports[0]
        print(f"🚀 Recommended: Use port {recommended_port}")
        print(f"   Start with: export PORT={recommended_port} && python start_kimera.py")
    else:
        print("❌ No common ports are available")
        print("💡 Try stopping other services or manually specify a port")
        print("   Example: export PORT=9999 && python start_kimera.py")

if __name__ == "__main__":
    main() 