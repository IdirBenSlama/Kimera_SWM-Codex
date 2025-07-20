#!/usr/bin/env python3
"""
Neo4j Test Setup Helper

This script helps set up Neo4j for testing the integration.
It provides Docker commands and environment setup.
"""

import os
import subprocess
import time

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Docker found: {result.stdout.strip()
            return True
        else:
            logger.error("❌ Docker not found")
            return False
    except FileNotFoundError:
        logger.error("❌ Docker not installed")
        return False

def start_neo4j_container():
    """Start Neo4j container for testing"""
    container_name = "kimera-neo4j-test"
    
    # Check if container already exists
    try:
        result = subprocess.run(['docker', 'ps', '-a', '--filter', f'name={container_name}', '--format', '{{.Names}}'], 
                              capture_output=True, text=True)
        if container_name in result.stdout:
            logger.info(f"📦 Container {container_name} already exists")
            # Start if stopped
            subprocess.run(['docker', 'start', container_name], capture_output=True)
            logger.info(f"🚀 Started existing container {container_name}")
        else:
            # Create new container
            cmd = [
                'docker', 'run', '-d',
                '--name', container_name,
                '-p', '7687:7687',
                '-p', '7474:7474',
                '-e', 'NEO4J_AUTH=neo4j/testpassword',
                '-e', 'NEO4J_PLUGINS=["apoc"]',
                'neo4j:5'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Created and started Neo4j container: {container_name}")
            else:
                logger.error(f"❌ Failed to start container: {result.stderr}")
                return False
        
        # Wait for Neo4j to be ready
        logger.info("⏳ Waiting for Neo4j to be ready...")
        time.sleep(10)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error managing Docker container: {e}")
        return False

def set_environment_variables():
    """Set environment variables for Neo4j connection"""
    env_vars = {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USER': 'neo4j',
        'NEO4J_PASS': 'testpassword'
    }
    
    logger.debug("\n🔧 Setting environment variables:")
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"   {key}={value}")
    
    # Also create a .env file for persistence
    with open('.env', 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    logger.info("✅ Environment variables set and saved to .env file")

def test_connection():
    """Test Neo4j connection"""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from backend.graph.session import driver_liveness_check
        
        if driver_liveness_check():
            logger.info("✅ Neo4j connection test successful")
            return True
        else:
            logger.error("❌ Neo4j connection test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Connection test error: {e}")
        return False

def show_neo4j_info():
    """Show Neo4j access information"""
    logger.info("\n���� Neo4j Access Information:")
    logger.info("   Web Interface: http://localhost:7474")
    logger.info("   Bolt URI: bolt://localhost:7687")
    logger.info("   Username: neo4j")
    logger.info("   Password: testpassword")
    logger.debug("\n🔍 Useful Cypher queries to run in Neo4j Browser:")
    logger.info("   MATCH (n)
    logger.info("   MATCH (g:Geoid)
    logger.info("   MATCH (s:Scar)

def main():
    logger.info("🚀 KIMERA Neo4j Test Setup")
    logger.info("=" * 40)
    
    # Check Docker
    if not check_docker():
        logger.error("\n❌ Docker is required for this setup. Please install Docker first.")
        return 1
    
    # Start Neo4j container
    if not start_neo4j_container():
        logger.error("\n❌ Failed to start Neo4j container")
        return 1
    
    # Set environment variables
    set_environment_variables()
    
    # Test connection
    logger.info("\n🔌 Testing Neo4j connection...")
    if test_connection():
        logger.info("\n🎉 Neo4j setup complete!")
        show_neo4j_info()
        
        logger.info("\n▶️  Next steps:")
        logger.info("   1. Run: python test_neo4j_integration.py")
        logger.info("   2. Or start KIMERA with Neo4j integration enabled")
        
        return 0
    else:
        logger.warning("\n⚠️  Setup completed but connection test failed.")
        logger.info("   Neo4j might still be starting up. Wait a moment and try:")
        logger.info("   python test_neo4j_integration.py")
        return 1

if __name__ == "__main__":
    exit(main())