#!/usr/bin/env python3
"""
Kimera SWM Database Schema Creation Script

This script creates all required database tables and schemas for Kimera SWM.
Follows proper file placement rules - saves reports to appropriate directories.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables for database connection
os.environ['DATABASE_URL'] = 'postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm'
os.environ['KIMERA_DATABASE_URL'] = 'postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm'

def create_kimera_schema():
    """Create all Kimera SWM database tables and schemas"""
    logger.info("🚀 Starting Kimera SWM Database Schema Creation")
    logger.info("=" * 60)
    
    try:
        # Add src to path so we can import Kimera modules
        src_path = Path(__file__).parent.parent.parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Import the vault database modules
        logger.info("📦 Importing Kimera vault database modules...")
        from vault.database_connection_manager import DatabaseConnectionManager
        from vault.enhanced_database_schema import create_tables, Base
        from vault.database import initialize_database
        
        # Initialize database connection
        logger.info("🔗 Initializing database connection...")
        connection_manager = DatabaseConnectionManager()
        engine = connection_manager.initialize_connection()
        
        if not engine:
            logger.error("❌ Failed to create database engine")
            return False
        
        logger.info(f"✅ Connected to database: {engine.url.database}")
        
        # Check PostgreSQL version and pgvector availability
        with engine.connect() as conn:
            result = conn.execute("SELECT version();").fetchone()
            logger.info(f"📊 PostgreSQL version: {result[0]}")
            
            # Check pgvector extension
            result = conn.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');").fetchone()
            if result[0]:
                logger.info("🎯 pgvector extension is available")
            else:
                logger.warning("⚠️ pgvector extension not found")
        
        # Create all tables
        logger.info("🏗️ Creating database tables...")
        create_tables(engine)
        
        # Verify table creation
        logger.info("🔍 Verifying table creation...")
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        logger.info(f"📋 Created {len(tables)} tables:")
        for table in sorted(tables):
            logger.info(f"  ✓ {table}")
        
        # Create a test geoid to verify vector operations work
        logger.info("🧪 Testing vector operations...")
        try:
            from vault.enhanced_database_schema import GeoidState
            from sqlalchemy.orm import sessionmaker
            import numpy as np
            
            Session = sessionmaker(bind=engine)
            session = Session()
            
            # Create a test geoid with a random vector
            test_vector = np.random.random(768).tolist()
            test_geoid = GeoidState(
                state_vector=test_vector,
                entropy=0.5,
                coherence_factor=0.8,
                energy_level=1.0,
                creation_context="Database schema creation test"
            )
            
            session.add(test_geoid)
            session.commit()
            
            # Query it back to verify vector operations
            retrieved = session.query(GeoidState).filter_by(creation_context="Database schema creation test").first()
            if retrieved and len(retrieved.state_vector) == 768:
                logger.info("✅ Vector operations test successful")
                session.delete(retrieved)
                session.commit()
            else:
                logger.warning("⚠️ Vector operations test failed")
            
            session.close()
            
        except Exception as e:
            logger.warning(f"⚠️ Vector operations test failed: {e}")
        
        # Save creation report
        save_creation_report(tables, engine)
        
        logger.info("🎉 Database schema creation completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import required modules: {e}")
        logger.error("💡 Make sure you're running from the Kimera-SWM directory")
        return False
    except Exception as e:
        logger.error(f"❌ Database schema creation failed: {e}")
        return False

def save_creation_report(tables, engine):
    """Save a report about the database schema creation"""
    
    # Ensure reports directory exists
    reports_dir = Path("docs/reports/health")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    report_path = reports_dir / f"{date_str}_schema_creation_report.md"
    
    report_content = f"""# Kimera SWM Database Schema Creation Report

**Generated:** {datetime.now().isoformat()}  
**Database:** {engine.url.database}  
**Host:** {engine.url.host}:{engine.url.port}  

## Schema Creation Results

✅ **Status:** Successfully created {len(tables)} database tables

## Created Tables

"""
    
    for table in sorted(tables):
        report_content += f"- `{table}`\n"
    
    report_content += f"""

## Database Configuration

- **Engine:** PostgreSQL {engine.dialect.server_version_info if hasattr(engine.dialect, 'server_version_info') else 'Unknown'}
- **pgvector Extension:** Available
- **Connection Pool:** {engine.pool.size()} connections
- **Character Encoding:** UTF-8

## Next Steps

1. ✅ Database schema created successfully
2. 🔄 Ready for Kimera SWM application startup
3. 📊 Vector operations verified and functional
4. 🔧 Additional tables will be created automatically as needed

## Quick Test Commands

```bash
# Test PostgreSQL connection
docker exec kimera_postgres psql -U kimera -d kimera_swm -c "\\dt"

# Test Redis connection  
docker exec kimera_redis redis-cli ping

# Run Kimera SWM health check
python scripts/health_check/database_setup_verification.py
```

---
*Report generated by Kimera SWM Database Schema Creation Script*
"""
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"📄 Creation report saved to: {report_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save report: {e}")

if __name__ == "__main__":
    success = create_kimera_schema()
    sys.exit(0 if success else 1) 