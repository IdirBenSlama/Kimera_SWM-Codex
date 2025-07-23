#!/usr/bin/env python3
"""
Database Initialization Script for KIMERA
=========================================

Creates all necessary database tables with aerospace-grade schema design.
Supports both SQLite and PostgreSQL with automatic detection.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize database with all required tables."""
    logger.info("=" * 70)
    logger.info("KIMERA Database Initialization")
    logger.info("=" * 70)
    
    try:
        # Import database components
        from sqlalchemy import create_engine, text
        from src.vault.database import Base, get_engine
        from src.config.kimera_config import get_config
        
        # Get configuration
        config = get_config()
        db_url = config.database.url
        
        logger.info(f"Database URL: {db_url.split('@')[0]}...")
        
        # Create engine
        engine = get_engine()
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✓ Database connection successful")
        
        # Create all tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables created successfully")
        
        # Verify tables
        with engine.connect() as conn:
            if 'sqlite' in db_url:
                # SQLite
                result = conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ))
            else:
                # PostgreSQL
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
                ))
            
            tables = [row[0] for row in result]
            logger.info(f"✓ Created {len(tables)} tables:")
            for table in sorted(tables):
                logger.info(f"  - {table}")
        
        # Create indexes for performance
        logger.info("Creating indexes...")
        create_indexes(engine)
        logger.info("✓ Indexes created successfully")
        
        # Initialize with sample data (optional)
        if os.environ.get('KIMERA_INIT_SAMPLE_DATA', 'false').lower() == 'true':
            logger.info("Initializing sample data...")
            initialize_sample_data(engine)
            logger.info("✓ Sample data initialized")
        
        logger.info("\n" + "=" * 70)
        logger.info("Database initialization completed successfully!")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        return False

def create_indexes(engine):
    """Create performance indexes."""
    index_statements = [
        # Geoid indexes
        "CREATE INDEX IF NOT EXISTS idx_geoid_created_at ON geoids(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_geoid_vault_id ON geoids(vault_id)",
        
        # SCAR indexes
        "CREATE INDEX IF NOT EXISTS idx_scar_created_at ON scars(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_scar_resolution ON scars(resolution_strategy)",
        
        # Insight indexes
        "CREATE INDEX IF NOT EXISTS idx_insight_created_at ON insights(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_insight_status ON insights(status)",
        
        # Conversation indexes
        "CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversations(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_conversation_timestamp ON conversations(timestamp)",
    ]
    
    with engine.connect() as conn:
        for stmt in index_statements:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception as e:
                # Index might already exist
                logger.debug(f"Index creation note: {e}")

def initialize_sample_data(engine):
    """Initialize sample data for testing."""
    from src.vault.database import GeoidDB, ScarDB, InsightDB
    from sqlalchemy.orm import Session
    from datetime import datetime
    import numpy as np
    
    with Session(engine) as session:
        # Create sample geoids
        sample_geoids = [
            GeoidDB(
                geoid_id="sample_geoid_1",
                semantic_features={"concept": "consciousness", "domain": "philosophy"},
                symbolic_content={"type": "abstract", "complexity": "high"},
                echoform_text="What is the nature of consciousness?",
                embedding=np.random.rand(768).tolist(),
                vault_id="default",
                metadata={"source": "sample_data"}
            ),
            GeoidDB(
                geoid_id="sample_geoid_2",
                semantic_features={"concept": "intelligence", "domain": "ai"},
                symbolic_content={"type": "technical", "complexity": "medium"},
                echoform_text="How does artificial intelligence work?",
                embedding=np.random.rand(768).tolist(),
                vault_id="default",
                metadata={"source": "sample_data"}
            )
        ]
        
        for geoid in sample_geoids:
            session.add(geoid)
        
        # Create sample SCAR
        sample_scar = ScarDB(
            scar_id="sample_scar_1",
            geoid_ids=["sample_geoid_1", "sample_geoid_2"],
            tension_gradient={"score": 0.75, "type": "conceptual"},
            resolution_strategy="synthesis",
            embedding=np.random.rand(768).tolist(),
            metadata={"source": "sample_data"}
        )
        session.add(sample_scar)
        
        # Create sample insight
        sample_insight = InsightDB(
            insight_id="sample_insight_1",
            echoform_repr="Consciousness and intelligence are interconnected phenomena",
            geoid_associations=["sample_geoid_1", "sample_geoid_2"],
            scar_associations=["sample_scar_1"],
            confidence_score=0.85,
            status="active",
            metadata={"source": "sample_data"}
        )
        session.add(sample_insight)
        
        session.commit()
        logger.info(f"✓ Created {len(sample_geoids)} sample geoids, 1 SCAR, and 1 insight")

def verify_installation():
    """Verify the database installation."""
    try:
        from src.vault.vault_manager import VaultManager
        
        # Initialize VaultManager
        vault = VaultManager()
        
        # Test basic operations
        stats = vault.get_statistics()
        logger.info("\nDatabase Statistics:")
        logger.info(f"  Total Geoids: {stats.get('total_geoids', 0)}")
        logger.info(f"  Total SCARs: {stats.get('total_scars', 0)}")
        logger.info(f"  Total Insights: {stats.get('total_insights', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Verify DATABASE_URL is set
    if 'DATABASE_URL' not in os.environ:
        logger.error("DATABASE_URL not set. Please configure .env file")
        sys.exit(1)
    
    db_url = os.environ['DATABASE_URL']
    if not db_url.startswith('postgresql'):
        logger.error(f"PostgreSQL is required. Current DATABASE_URL: {db_url.split('://')[0]}://...")
        sys.exit(1)
    
    logger.info(f"Using PostgreSQL database: {db_url.split('@')[1] if '@' in db_url else 'localhost'}")
    
    # Run initialization
    success = initialize_database()
    
    if success:
        # Verify installation
        verify_installation()
        sys.exit(0)
    else:
        sys.exit(1)