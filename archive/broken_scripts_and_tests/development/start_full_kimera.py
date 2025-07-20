#!/usr/bin/env python3
"""
Start Full KIMERA SWM System
Launches the complete system with all dependencies
"""

import os
import uvicorn
import logging

# Configure environment for full system
os.environ['DATABASE_URL'] = 'postgresql://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm'
os.environ['REDIS_HOST'] = 'localhost'
os.environ['REDIS_PORT'] = '6379'
os.environ['NEO4J_URI'] = 'bolt://localhost:7687'
os.environ['NEO4J_USER'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'password'
os.environ['API_PORT'] = '8003'
os.environ['ENVIRONMENT'] = 'production'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Start the full KIMERA SWM system"""
    
    print("🚀 STARTING FULL KIMERA SWM SYSTEM")
    print("=" * 50)
    print("🔗 Database: PostgreSQL (kimera_swm)")
    print("🔗 Cache: Redis (localhost:6379)")
    print("🔗 Graph: Neo4j (localhost:7687)")
    print("🌐 API Port: 8003")
    print("=" * 50)
    
    try:
        # Import the full application
        from backend.api.main import app
        
        logger.info("✅ Full KIMERA SWM application imported successfully")
        logger.info("🚀 Starting server on http://localhost:8003")
        
        # Start the server
        uvicorn.run(
            app, 
            host='0.0.0.0', 
            port=8003, 
            log_level='info',
            access_log=True,
            reload=False
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start full KIMERA SWM system: {e}")
        raise

if __name__ == "__main__":
    main() 