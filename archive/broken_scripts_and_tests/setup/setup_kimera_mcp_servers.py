#!/usr/bin/env python3
"""
Kimera MCP Servers Setup & Verification Script

This script sets up and verifies all the most suitable MCP servers for Kimera development.
Following Kimera's core principles: Cognitive Fidelity, Zero-Debugging, Atomic Task Breakdown.
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kimera_mcp_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KimeraMCPSetup:
    """Comprehensive MCP server setup for Kimera development"""
    
    def __init__(self):
        self.setup_results = {}
        self.config_files = {}
        self.workspace_root = Path.cwd()
        
        # MCP Servers optimized for Kimera (all FREE)
        self.mcp_servers = {
            # Phase 1: Core Infrastructure
            "github": {
                "package": "@modelcontextprotocol/server-github",
                "description": "GitHub integration for repository management",
                "priority": 1
            },
            "sequential-thinking": {
                "package": "@modelcontextprotocol/server-sequential-thinking", 
                "description": "Sequential thinking for atomic task breakdown",
                "priority": 1
            },
            "memory": {
                "package": "@modelcontextprotocol/server-memory",
                "description": "Memory bank for cognitive field dynamics",
                "priority": 1
            },
            
            # Phase 2: Development Tools
            "playwright": {
                "package": "@modelcontextprotocol/server-playwright",
                "description": "Browser automation for web interface testing",
                "priority": 2
            },
            "puppeteer": {
                "package": "@modelcontextprotocol/server-puppeteer",
                "description": "Browser automation with Puppeteer",
                "priority": 2
            },
            
            # Phase 3: Research & Documentation
            "context7": {
                "package": "@upstash/context7-mcp",
                "description": "Documentation and library context",
                "priority": 3
            }
        }
        
        logger.info("🚀 Kimera MCP Setup initialized")
    
    async def check_prerequisites(self) -> bool:
        """Check if Node.js and npm are available"""
        logger.info("🔍 Checking prerequisites...")
        
        try:
            # Check Node.js
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ Node.js not found. Please install Node.js 18+")
                return False
            logger.info(f"✅ Node.js: {result.stdout.strip()}")
            
            # Check npm
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ npm not found. Please install npm")
                return False
            logger.info(f"✅ npm: {result.stdout.strip()}")
            
            return True
            
        except FileNotFoundError:
            logger.error("❌ Node.js/npm not found. Please install Node.js 18+")
            return False
    
    async def install_mcp_server(self, name: str, config: Dict[str, Any]) -> bool:
        """Install a single MCP server"""
        logger.info(f"📦 Installing {name} MCP server...")
        
        try:
            cmd = ["npm", "install", "-g", config["package"]]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ {name} installed successfully")
                self.setup_results[name] = {"status": "success", "package": config["package"]}
                return True
            else:
                logger.error(f"❌ Failed to install {name}: {result.stderr}")
                self.setup_results[name] = {"status": "failed", "error": result.stderr}
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing {name}: {e}")
            self.setup_results[name] = {"status": "error", "error": str(e)}
            return False
    
    def create_cursor_config(self) -> Dict[str, Any]:
        """Create Cursor MCP configuration"""
        config = {"mcpServers": {}}
        
        for name, server_config in self.mcp_servers.items():
            if self.setup_results.get(name, {}).get("status") == "success":
                config["mcpServers"][name] = {
                    "command": "npx",
                    "args": ["-y", server_config["package"]]
                }
        
        return config
    
    def create_claude_desktop_config(self) -> Dict[str, Any]:
        """Create Claude Desktop MCP configuration"""
        config = {"mcpServers": {}}
        
        for name, server_config in self.mcp_servers.items():
            if self.setup_results.get(name, {}).get("status") == "success":
                config["mcpServers"][name.title()] = {
                    "command": "npx",
                    "args": ["-y", server_config["package"]]
                }
        
        return config
    
    async def save_configurations(self):
        """Save all MCP configurations"""
        logger.info("💾 Saving MCP configurations...")
        
        # Create .cursor directory
        cursor_dir = self.workspace_root / ".cursor"
        cursor_dir.mkdir(exist_ok=True)
        
        # Cursor configuration
        cursor_config = self.create_cursor_config()
        cursor_file = cursor_dir / "mcp.json"
        with open(cursor_file, 'w') as f:
            json.dump(cursor_config, f, indent=2)
        logger.info(f"✅ Cursor config: {cursor_file}")
        
        # Claude Desktop configuration
        claude_config = self.create_claude_desktop_config()
        claude_file = self.workspace_root / "claude_desktop_config.json"
        with open(claude_file, 'w') as f:
            json.dump(claude_config, f, indent=2)
        logger.info(f"✅ Claude Desktop config: {claude_file}")
    
    async def run_setup(self):
        """Run the complete setup process"""
        logger.info("🌟 Starting Kimera MCP Servers Setup")
        
        # Check prerequisites
        if not await self.check_prerequisites():
            return False
        
        # Install servers by priority
        phases = {}
        for name, config in self.mcp_servers.items():
            priority = config["priority"]
            if priority not in phases:
                phases[priority] = []
            phases[priority].append((name, config))
        
        total_success = 0
        for phase in sorted(phases.keys()):
            logger.info(f"\n🔄 Phase {phase} Installation:")
            
            for name, config in phases[phase]:
                success = await self.install_mcp_server(name, config)
                if success:
                    total_success += 1
        
        # Save configurations
        await self.save_configurations()
        
        # Summary
        total = len(self.mcp_servers)
        logger.info(f"\n🎉 Setup Complete! {total_success}/{total} servers installed")
        
        if total_success > 0:
            logger.info("✅ MCP servers ready for Kimera development!")
            return True
        else:
            logger.error("❌ No servers installed successfully")
            return False

async def main():
    """Main setup function"""
    setup = KimeraMCPSetup()
    success = await setup.run_setup()
    
    if success:
        print("\n🎯 Kimera MCP Servers are ready!")
        print("   Restart your IDE and start building!")
    else:
        print("\n❌ Setup failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 