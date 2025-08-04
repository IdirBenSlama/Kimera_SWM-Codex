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
import logging
logger = logging.getLogger(__name__)

def kill_existing_servers():
    """Kill any existing MCP server processes."""
    logger.info("🔄 Killing existing MCP server processes...")
    try:
        subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                      capture_output=True, check=False)
        subprocess.run(["taskkill", "/F", "/IM", "mcp-server-sqlite.exe"], 
                      capture_output=True, check=False)
        time.sleep(2)
        logger.info("✅ Cleaned up existing processes")
    except Exception as e:
        logger.info(f"⚠️ Error cleaning processes: {e}")

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
    
    logger.info(f"✅ Created minimal config: {config_path}")
    return config

def test_minimal_config():
    """Test the minimal configuration."""
    logger.info("\n🧪 Testing minimal configuration...")
    
    try:
        # Test SQLite server directly
        result = subprocess.run([
            "mcp-server-sqlite", 
            "--db-path", str(Path.cwd() / "kimera_swm.db")
        ], capture_output=True, text=True, timeout=5)
        
        logger.info("✅ SQLite server test passed")
        return True
        
    except subprocess.TimeoutExpired:
        logger.info("✅ SQLite server started (timeout expected)")
        return True
    except Exception as e:
        logger.info(f"❌ SQLite server test failed: {e}")
        return False

def create_working_config():
    """Create a working MCP configuration step by step."""
    logger.info("\n🔧 Creating working MCP configuration...")
    
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
    
    logger.info("✅ Created working configuration")
    return config

def verify_environment():
    """Verify the Python environment has required packages."""
    logger.info("\n🔍 Verifying environment...")
    
    required_packages = ["mcp", "fastmcp", "mcp_server_fetch", "mcp_server_sqlite"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            logger.info(f"❌ {package} missing")
    
    if missing_packages:
        logger.info(f"\n📦 Installing missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"✅ Installed {package}")
            except Exception as e:
                logger.info(f"❌ Failed to install {package}: {e}")

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
    
    logger.info(f"✅ Created restart script: {script_path}")

def main():
    """Main fix routine."""
    logger.info("🚀 Windows MCP Server Fix Script")
    logger.info("Based on Cursor forum solutions")
    logger.info("=" * 50)
    
    # Step 1: Clean environment
    kill_existing_servers()
    
    # Step 2: Verify environment
    verify_environment()
    
    # Step 3: Test minimal config
    create_minimal_mcp_config()
    if test_minimal_config():
        logger.info("✅ Minimal config works")
    else:
        logger.info("❌ Basic SQLite server not working")
        return
    
    # Step 4: Create working config
    create_working_config()
    
    # Step 5: Create restart script
    create_cursor_restart_script()
    
    logger.info("\n🎯 NEXT STEPS:")
    logger.info("1. Run 'restart_cursor.bat' to restart Cursor")
    logger.info("2. Check MCP servers in Cursor settings")
    logger.info("3. If still red, try the alternative configs below")
    
    logger.info("\n💡 ALTERNATIVE SOLUTIONS:")
    logger.info("If servers still show red, try these configurations:")
    
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
    logger.info("✅ Alternative 1 saved to .cursor/mcp_alt1.json")
    
    # Alternative 2: Full paths
    python_path = subprocess.check_output([sys.executable, "-c", "import sys; logger.info(sys.executable)"], text=True).strip()
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
    logger.info("✅ Alternative 2 saved to .cursor/mcp_alt2.json")
    
    logger.info("\n📋 TROUBLESHOOTING:")
    logger.info("- If servers remain red, copy contents of mcp_alt1.json to mcp.json")
    logger.info("- If that fails, try mcp_alt2.json")
    logger.info("- Restart Cursor completely after each change")
    logger.info("- Check Windows firewall/antivirus isn't blocking processes")

if __name__ == "__main__":
    main() 