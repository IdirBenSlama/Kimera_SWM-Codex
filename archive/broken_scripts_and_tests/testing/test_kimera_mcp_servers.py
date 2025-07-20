#!/usr/bin/env python3
"""
Kimera MCP Servers Testing & Verification Script

Tests all installed MCP servers for Kimera development workflows.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kimera_mcp_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KimeraMCPTester:
    """Testing suite for Kimera MCP servers"""
    
    def __init__(self):
        self.test_results = {}
        self.workspace_root = Path.cwd()
        
        # Expected MCP servers for Kimera
        self.expected_servers = {
            "github": "@modelcontextprotocol/server-github",
            "sequential-thinking": "@modelcontextprotocol/server-sequential-thinking",
            "memory": "@modelcontextprotocol/server-memory",
            "playwright": "@modelcontextprotocol/server-playwright",
            "puppeteer": "@modelcontextprotocol/server-puppeteer",
            "context7": "@upstash/context7-mcp"
        }
        
        logger.info("🧪 Kimera MCP Testing Suite initialized")
    
    async def test_server_installation(self, name: str, package: str) -> Dict[str, Any]:
        """Test if an MCP server is properly installed"""
        logger.info(f"🔍 Testing {name} installation...")
        
        try:
            result = subprocess.run(
                ["npm", "list", "-g", package], 
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {name} is installed")
                return {"installed": True, "error": None}
            else:
                logger.error(f"❌ {name} not found")
                return {"installed": False, "error": result.stderr}
                
        except Exception as e:
            logger.error(f"❌ Error checking {name}: {e}")
            return {"installed": False, "error": str(e)}
    
    async def test_configuration_files(self) -> Dict[str, Any]:
        """Test if MCP configuration files exist"""
        logger.info("📁 Testing configuration files...")
        
        configs = {
            "cursor": self.workspace_root / ".cursor" / "mcp.json",
            "claude": self.workspace_root / "claude_desktop_config.json"
        }
        
        results = {}
        for name, path in configs.items():
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    server_count = len(data.get("mcpServers", {}))
                    logger.info(f"✅ {name} config: {server_count} servers")
                    results[name] = {"exists": True, "servers": server_count}
                except Exception as e:
                    logger.error(f"❌ {name} config invalid: {e}")
                    results[name] = {"exists": False, "error": str(e)}
            else:
                logger.error(f"❌ {name} config not found")
                results[name] = {"exists": False, "error": "File not found"}
        
        return results
    
    def calculate_score(self) -> float:
        """Calculate overall performance score"""
        total = len(self.expected_servers)
        installed = sum(1 for result in self.test_results.values() 
                       if isinstance(result, dict) and result.get("installed", False))
        return (installed / total) * 100 if total > 0 else 0
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Kimera MCP Tests")
        
        # Test each server
        for name, package in self.expected_servers.items():
            result = await self.test_server_installation(name, package)
            self.test_results[name] = result
        
        # Test configurations
        config_results = await self.test_configuration_files()
        self.test_results["configurations"] = config_results
        
        # Calculate score
        score = self.calculate_score()
        
        # Summary
        logger.info(f"\n🎯 Test Complete! Score: {score:.1f}%")
        
        installed_count = sum(1 for result in self.test_results.values() 
                            if isinstance(result, dict) and result.get("installed", False))
        
        if score >= 80:
            logger.info("🌟 Excellent! All systems ready for Kimera development")
            return True
        elif score >= 60:
            logger.info("✅ Good! Most systems ready")
            return True
        else:
            logger.warning("⚠️  Some issues detected")
            return False

async def main():
    """Main testing function"""
    tester = KimeraMCPTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎯 Kimera MCP Servers are ready!")
    else:
        print("\n⚠️  Some issues detected. Check logs.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 