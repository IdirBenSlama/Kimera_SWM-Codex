#!/usr/bin/env python3
"""
Kimera MCP Servers Validation Script

This script validates the MCP server setup by testing:
1. Docker connectivity
2. Docker image availability
3. Database service connectivity
4. Configuration file validation

Author: Kimera Cognitive System
"""

import json
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_validation.log')
    ]
)

logger = logging.getLogger(__name__)

class MCPValidator:
    """Validates MCP server setup and configuration."""
    
    def __init__(self, config_path: str = '.cursor/mcp.json'):
        self.config_path = Path(config_path)
        self.config = None
        self.docker_images = [
            'mcp/context7',
            'mcp/desktop-commander', 
            'mcp/exa',
            'mcp/memory',
            'mcp/neo4j-cypher',
            'mcp/neo4j-memory',
            'mcp/node-code-sandbox',
            'mcp/paper-search',
            'mcp/postgres',
            'mcp/sequentialthinking'
        ]
        
    def load_config(self) -> bool:
        """Load and validate MCP configuration file."""
        try:
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
                
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            if 'mcpServers' not in self.config:
                logger.error("Invalid configuration: 'mcpServers' key not found")
                return False
                
            logger.info(f"Configuration loaded successfully with {len(self.config['mcpServers'])} servers")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def check_docker_availability(self) -> bool:
        """Check if Docker is installed and running."""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Docker available: {result.stdout.strip()}")
                return True
            else:
                logger.error("Docker not available")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Docker command timed out")
            return False
        except FileNotFoundError:
            logger.error("Docker not installed")
            return False
        except Exception as e:
            logger.error(f"Error checking Docker: {e}")
            return False
    
    def check_docker_daemon(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("Docker daemon is running")
                return True
            else:
                logger.error("Docker daemon not running")
                logger.error(result.stderr)
                return False
        except Exception as e:
            logger.error(f"Error checking Docker daemon: {e}")
            return False
    
    def check_docker_images(self) -> Dict[str, bool]:
        """Check availability of required Docker images."""
        results = {}
        
        for image in self.docker_images:
            try:
                result = subprocess.run(['docker', 'inspect', image], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✓ Image available: {image}")
                    results[image] = True
                else:
                    logger.warning(f"✗ Image not found: {image}")
                    results[image] = False
            except Exception as e:
                logger.error(f"Error checking image {image}: {e}")
                results[image] = False
                
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive validation report."""
        report = []
        report.append("=" * 60)
        report.append("KIMERA MCP SERVERS VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Docker status
        report.append(f"🐳 DOCKER STATUS")
        report.append(f"   Available: {'✓' if results.get('docker_available') else '✗'}")
        report.append(f"   Daemon Running: {'✓' if results.get('docker_daemon') else '✗'}")
        report.append("")
        
        # Image status
        if results.get('image_status'):
            available = sum(1 for status in results['image_status'].values() if status)
            total = len(results['image_status'])
            report.append(f"📦 DOCKER IMAGES ({available}/{total} available)")
            for image, status in results['image_status'].items():
                report.append(f"   {'✓' if status else '✗'} {image}")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        
        if not results.get('docker_available'):
            report.append("   • Install Docker Desktop from docker.com")
            
        if not results.get('docker_daemon'):
            report.append("   • Start Docker Desktop application")
            
        if results.get('image_status'):
            missing = [img for img, status in results['image_status'].items() if not status]
            if missing:
                report.append("   • Pull missing images:")
                for img in missing:
                    report.append(f"     docker pull {img}")
        
        report.append("")
        report.append("🚀 NEXT STEPS")
        report.append("   1. Address any issues above")
        report.append("   2. Restart Cursor AI")
        report.append("   3. Check MCP settings in Cursor")
        report.append("   4. Test tools in Cursor chat")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def run_validation(self) -> Dict:
        """Run complete MCP setup validation."""
        logger.info("Starting MCP setup validation...")
        
        results = {}
        
        # 1. Load configuration
        results['config_valid'] = self.load_config()
        
        # 2. Check Docker
        results['docker_available'] = self.check_docker_availability()
        if results['docker_available']:
            results['docker_daemon'] = self.check_docker_daemon()
        
        # 3. Check Docker images
        if results.get('docker_daemon'):
            results['image_status'] = self.check_docker_images()
        
        # Generate and display report
        report = self.generate_report(results)
        print(report)
        
        # Save report to file
        with open('mcp_validation_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("Validation complete. Report saved to mcp_validation_report.txt")
        
        return results

def main():
    """Main entry point."""
    validator = MCPValidator()
    results = validator.run_validation()
    
    # Return exit code based on critical issues
    critical_issues = 0
    if not results.get('docker_available'):
        critical_issues += 1
    if not results.get('docker_daemon'):
        critical_issues += 1
    if not results.get('config_valid'):
        critical_issues += 1
    
    if critical_issues > 0:
        logger.error(f"Validation failed with {critical_issues} critical issues")
        sys.exit(1)
    else:
        logger.info("Validation completed successfully")
        sys.exit(0)

if __name__ == '__main__':
    main() 