#!/usr/bin/env python3

def get_secure_demo_key(key_name: str) -> str:
    """Get secure demo key or raise error for production"""
    if os.getenv("KIMERA_ENV") == "production":
        raise ValueError(f"Demo key {key_name} cannot be used in production. Set environment variable.")
    
    # Generate deterministic demo key (NOT for production)
    demo_keys = {
        "PHEMEX_API_KEY": "demo_key_" + hashlib.sha256(f"phemex_api_{key_name}".encode()).hexdigest()[:16],
        "PHEMEX_API_SECRET": "demo_secret_" + hashlib.sha256(f"phemex_secret_{key_name}".encode()).hexdigest()[:32],
    }
    
    return demo_keys.get(key_name, f"demo_{key_name.lower()}")

def get_secure_password(env_var: str, default: str) -> str:
    """Get secure password from environment or generate one"""
    password = os.getenv(env_var)
    if not password:
        logger.warning(f"Using default password for {env_var}. Set environment variable for production.")
        return default
    return password

"""
Kimera MCP Docker Setup Script

This script pulls all required Docker images for MCP servers.

Author: Kimera Cognitive System
"""

import subprocess
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

DOCKER_IMAGES = [
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

def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"Docker available: {result.stdout.strip()}")
            return True
        else:
            logger.error("Docker not available")
            return False
    except Exception as e:
        logger.error(f"Error checking Docker: {e}")
        return False

def pull_image(image):
    """Pull a specific Docker image."""
    try:
        logger.info(f"Pulling {image}...")
        result = subprocess.run(['docker', 'pull', image], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"✓ Successfully pulled {image}")
            return True
        else:
            logger.error(f"✗ Failed to pull {image}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Timeout pulling {image}")
        return False
    except Exception as e:
        logger.error(f"✗ Error pulling {image}: {e}")
        return False

def setup_database_services():
    """Setup optional database services."""
    logger.info("Setting up optional database services...")
    
    # Neo4j
    try:
        logger.info("Starting Neo4j service...")
        result = subprocess.run([
            'docker', 'run', '-d', '--name', 'kimera-neo4j',
            '-p', '7474:7474', '-p', '7687:7687',
            '-e', f'NEO4J_AUTH=neo4j/{get_secure_password("NEO4J_PASSWORD", "neo4j_default_2024")}',
            'neo4j:latest'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✓ Neo4j service started")
        else:
            logger.warning(f"Neo4j service may already be running: {result.stderr}")
    except Exception as e:
        logger.warning(f"Could not start Neo4j: {e}")
    
    # PostgreSQL
    try:
        logger.info("Starting PostgreSQL service...")
        result = subprocess.run([
            'docker', 'run', '-d', '--name', 'kimera-postgres',
            '-p', '5432:5432',
            '-e', 'POSTGRES_DB=mydb',
            '-e', f'POSTGRES_PASSWORD={get_secure_password("POSTGRES_PASSWORD", "postgres_default_2024")}',
            'postgres:latest'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✓ PostgreSQL service started")
        else:
            logger.warning(f"PostgreSQL service may already be running: {result.stderr}")
    except Exception as e:
        logger.warning(f"Could not start PostgreSQL: {e}")

def main():
    """Main setup function."""
    logger.info("Starting Kimera MCP Docker setup...")
    
    # Check Docker availability
    if not check_docker():
        logger.error("Docker is not available. Please install Docker Desktop first.")
        sys.exit(1)
    
    # Pull all images
    success_count = 0
    total_count = len(DOCKER_IMAGES)
    
    for image in DOCKER_IMAGES:
        if pull_image(image):
            success_count += 1
    
    # Setup database services
    setup_database_services()
    
    # Report results
    logger.info(f"Setup complete: {success_count}/{total_count} images pulled successfully")
    
    if success_count == total_count:
        logger.info("🎉 All MCP Docker images are ready!")
        logger.info("Next steps:")
        logger.info("1. Restart Cursor AI")
        logger.info("2. Check MCP settings in Cursor")
        logger.info("3. Test the new tools in chat")
    else:
        failed_count = total_count - success_count
        logger.warning(f"⚠️ {failed_count} images failed to pull")
        logger.warning("Some MCP servers may not work properly")
    
    return 0 if success_count == total_count else 1

if __name__ == '__main__':
    sys.exit(main()) 