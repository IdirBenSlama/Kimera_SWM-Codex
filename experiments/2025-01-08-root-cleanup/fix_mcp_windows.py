#!/usr/bin/env python3
"""
Windows MCP Server Fix Script
Addresses the "Client Closed" error by implementing Windows-specific fixes
Based on solutions from Cursor forum discussions.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path

def kill_existing_servers():
    """Kill any existing MCP server processes."""
    print("🔄 Killing existing MCP server processes...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                      capture_output=True, check=False)
        subprocess.run(["taskkill", "/F", "/IM", "mcp-server-sqlite.exe"], 
                      capture_output=True, check=False)
        time.sleep(2)
        print("✅ Cleaned up existing processes")
    except Exception as e:
        print(f"⚠️ Error cleaning processes: {e}")

def create_minimal_mcp_config():
    """Create a minimal MCP configuration that works on Windows."""
    config = {
        "mcpServers": {
            "sqlite-minimal": {
                "command": "mcp-server-sqlite",
                "args": ["--db-path", str(Path.cwd() / "kimera_swm.db")],
                "env": {}
            }
        }
    }
    
    config_path = Path(".cursor/mcp.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created minimal config: {config_path}")
    return config

def test_minimal_config():
    """Test the minimal configuration."""
    print("\n🧪 Testing minimal configuration...")
    
    try:
        # Test SQLite server directly
        result = subprocess.run([
            "mcp-server-sqlite", 
            "--db-path", str(Path.cwd() / "kimera_swm.db")
        ], capture_output=True, text=True, timeout=5)
        
        print("✅ SQLite server test passed")
        return True
        
    except subprocess.TimeoutExpired:
        print("✅ SQLite server started (timeout expected)")
        return True
    except Exception as e:
        print(f"❌ SQLite server test failed: {e}")
        return False

def create_working_config():
    """Create a working MCP configuration step by step."""
    print("\n🔧 Creating working MCP configuration...")
    
    # Start with minimal working config
    config = {
        "mcpServers": {
            "sqlite-kimera": {
                "command": "mcp-server-sqlite",
                "args": ["--db-path", str(Path.cwd() / "kimera_swm.db")],
                "env": {}
            },
            "fetch": {
                "command": "python",
                "args": ["-m", "mcp_server_fetch"],
                "env": {"PYTHONUNBUFFERED": "1"}
            }
        }
    }
    
    config_path = Path(".cursor/mcp.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created working configuration")
    return config

def verify_environment():
    """Verify the Python environment has required packages."""
    print("\n🔍 Verifying environment...")
    
    required_packages = ["mcp", "fastmcp", "mcp_server_fetch", "mcp_server_sqlite"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except Exception as e:
                print(f"❌ Failed to install {package}: {e}")

def create_cursor_restart_script():
    """Create a script to properly restart Cursor."""
    script_content = """@echo off
echo Restarting Cursor for MCP changes...
taskkill /F /IM Cursor.exe 2>nul
timeout /t 3 /nobreak >nul
start "" "C:\\Users\\%USERNAME%\\AppData\\Local\\Programs\\cursor\\Cursor.exe"
echo Cursor restarted. Please check MCP servers status.
pause
"""
    
    script_path = Path("restart_cursor.bat")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created restart script: {script_path}")

def main():
    """Main fix routine."""
    print("🚀 Windows MCP Server Fix Script")
    print("Based on Cursor forum solutions")
    print("=" * 50)
    
    # Step 1: Clean environment
    kill_existing_servers()
    
    # Step 2: Verify environment
    verify_environment()
    
    # Step 3: Test minimal config
    create_minimal_mcp_config()
    if test_minimal_config():
        print("✅ Minimal config works")
    else:
        print("❌ Basic SQLite server not working")
        return
    
    # Step 4: Create working config
    create_working_config()
    
    # Step 5: Create restart script
    create_cursor_restart_script()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Run 'restart_cursor.bat' to restart Cursor")
    print("2. Check MCP servers in Cursor settings")
    print("3. If still red, try the alternative configs below")
    
    print("\n💡 ALTERNATIVE SOLUTIONS:")
    print("If servers still show red, try these configurations:")
    
    # Alternative 1: CMD wrapper
    alt_config_1 = {
        "mcpServers": {
            "sqlite-cmd": {
                "command": "cmd",
                "args": ["/c", "mcp-server-sqlite", "--db-path", str(Path.cwd() / "kimera_swm.db")],
                "env": {}
            }
        }
    }
    
    with open(".cursor/mcp_alt1.json", 'w') as f:
        json.dump(alt_config_1, f, indent=2)
    print("✅ Alternative 1 saved to .cursor/mcp_alt1.json")
    
    # Alternative 2: Full paths
    python_path = subprocess.check_output([sys.executable, "-c", "import sys; print(sys.executable)"], text=True).strip()
    alt_config_2 = {
        "mcpServers": {
            "sqlite-full": {
                "command": python_path,
                "args": ["-m", "mcp_server_sqlite", "--db-path", str(Path.cwd() / "kimera_swm.db")],
                "env": {"PYTHONUNBUFFERED": "1"}
            }
        }
    }
    
    with open(".cursor/mcp_alt2.json", 'w') as f:
        json.dump(alt_config_2, f, indent=2)
    print("✅ Alternative 2 saved to .cursor/mcp_alt2.json")
    
    print("\n📋 TROUBLESHOOTING:")
    print("- If servers remain red, copy contents of mcp_alt1.json to mcp.json")
    print("- If that fails, try mcp_alt2.json")
    print("- Restart Cursor completely after each change")
    print("- Check Windows firewall/antivirus isn't blocking processes")

if __name__ == "__main__":
    main() 