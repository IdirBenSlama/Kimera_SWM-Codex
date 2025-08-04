#!/usr/bin/env python3
"""
Kimera PostgreSQL Migration Script
==================================
Migrates Kimera from SQLite to PostgreSQL with proper configuration.
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostgreSQLMigrator:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        
    def create_postgresql_env(self):
        """Create PostgreSQL environment configuration"""
        logger.info("Creating PostgreSQL environment configuration...")
        
        env_content = """# PostgreSQL Configuration for Kimera
DATABASE_URL=postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm
KIMERA_DATABASE_URL=postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm

# API Keys
OPENAI_API_KEY=your-openai-api-key-here

# Environment
KIMERA_ENV=development

# PostgreSQL specific settings
POSTGRES_USER=kimera
POSTGRES_PASSWORD=kimera_secure_pass_2025
POSTGRES_DB=kimera_swm
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Performance settings
KIMERA_DB_POOL_SIZE=20
KIMERA_DB_MAX_OVERFLOW=40
KIMERA_DB_POOL_TIMEOUT=30
KIMERA_DB_POOL_RECYCLE=3600
KIMERA_DB_POOL_PRE_PING=true

# Quantum settings
KIMERA_QUANTUM_ENABLED=true
KIMERA_QUANTUM_DIMENSION=10
KIMERA_QUANTUM_COHERENCE_THRESHOLD=0.7

# Thermodynamic settings
KIMERA_ENTROPY_PRECISION=high
KIMERA_THERMODYNAMIC_MODE=quantum
KIMERA_CARNOT_TOLERANCE=0.01
KIMERA_MAX_EFFICIENCY=0.99

# Diffusion settings
KIMERA_DIFFUSION_COEFFICIENT=0.1
KIMERA_SPDE_TIMESTEP=0.001
KIMERA_SPDE_SPATIAL_RESOLUTION=128

# Portal/Vortex settings
KIMERA_MAX_PORTALS=100
KIMERA_PORTAL_STABILITY_THRESHOLD=0.2
KIMERA_VORTEX_VISCOSITY=0.01
KIMERA_VORTEX_GRID_SIZE=128

# Semantic settings
KIMERA_EMBEDDING_DIM=1024
KIMERA_SEMANTIC_MODEL=BAAI/bge-m3
KIMERA_USE_FLAG_EMBEDDING=1

# Performance optimization
KIMERA_GPU_MEMORY_FRACTION=0.8
KIMERA_BATCH_SIZE=1024
KIMERA_CACHE_SIZE=10000
"""
        
        env_file = self.project_root / ".env.postgresql"
        try:
            env_file.write_text(env_content)
            logger.info(f"✓ Created PostgreSQL environment file: {env_file}")
        except Exception as e:
            logger.error(f"Failed to create environment file: {e}")
            
    def create_docker_compose(self):
        """Create docker-compose.yml for PostgreSQL with pgvector"""
        logger.info("Creating docker-compose configuration...")
        
        compose_content = """version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: kimera_postgres
    environment:
      POSTGRES_USER: kimera
      POSTGRES_PASSWORD: kimera_secure_pass_2025
      POSTGRES_DB: kimera_swm
    ports:
      - "5432:5432"
    volumes:
      - kimera_postgres_data:/var/lib/postgresql/data
      - ./init_db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U kimera -d kimera_swm"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  kimera_postgres_data:
"""
        
        compose_file = self.project_root / "docker-compose.yml"
        try:
            compose_file.write_text(compose_content)
            logger.info(f"✓ Created docker-compose.yml: {compose_file}")
        except Exception as e:
            logger.error(f"Failed to create docker-compose.yml: {e}")
            
    def create_init_sql(self):
        """Create PostgreSQL initialization script"""
        logger.info("Creating PostgreSQL initialization script...")
        
        init_sql = """-- Kimera PostgreSQL Initialization Script

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create optimized indexes function
CREATE OR REPLACE FUNCTION create_optimized_indexes() RETURNS void AS $$
BEGIN
    -- Create GIN indexes for JSON fields
    CREATE INDEX IF NOT EXISTS idx_geoids_symbolic_state_gin ON geoids USING gin (symbolic_state);
    CREATE INDEX IF NOT EXISTS idx_geoids_metadata_gin ON geoids USING gin (metadata_json);
    CREATE INDEX IF NOT EXISTS idx_scars_geoids_gin ON scars USING gin (geoids);
    
    -- Create vector indexes for similarity search
    CREATE INDEX IF NOT EXISTS idx_geoids_semantic_vector ON geoids USING ivfflat (semantic_vector vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS idx_scars_vector ON scars USING ivfflat (scar_vector vector_cosine_ops) WITH (lists = 100);
    
    -- Create B-tree indexes for frequently queried columns
    CREATE INDEX IF NOT EXISTS idx_scars_timestamp ON scars (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_scars_vault_id ON scars (vault_id);
    CREATE INDEX IF NOT EXISTS idx_insights_created_at ON insights (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_insights_type ON insights (insight_type);
    
    -- Create partial indexes for performance
    CREATE INDEX IF NOT EXISTS idx_scars_unresolved ON scars (scar_id) WHERE resolved_by IS NULL;
    CREATE INDEX IF NOT EXISTS idx_insights_provisional ON insights (insight_id) WHERE status = 'provisional';
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE kimera_swm TO kimera;
GRANT CREATE ON SCHEMA public TO kimera;

-- Performance settings
ALTER DATABASE kimera_swm SET shared_buffers = '256MB';
ALTER DATABASE kimera_swm SET effective_cache_size = '1GB';
ALTER DATABASE kimera_swm SET maintenance_work_mem = '64MB';
ALTER DATABASE kimera_swm SET work_mem = '16MB';

-- Connection settings
ALTER DATABASE kimera_swm SET max_connections = 100;
ALTER DATABASE kimera_swm SET idle_in_transaction_session_timeout = '30min';

-- Logging settings for development
ALTER DATABASE kimera_swm SET log_statement = 'mod';
ALTER DATABASE kimera_swm SET log_duration = on;
"""
        
        init_file = self.project_root / "init_db.sql"
        try:
            init_file.write_text(init_sql)
            logger.info(f"✓ Created PostgreSQL init script: {init_file}")
        except Exception as e:
            logger.error(f"Failed to create init script: {e}")
            
    def create_migration_script(self):
        """Create script to migrate data from SQLite to PostgreSQL"""
        logger.info("Creating data migration script...")
        
        migration_script = '''#!/usr/bin/env python3
"""
Migrate data from SQLite to PostgreSQL
"""

import sqlite3
import psycopg2
from psycopg2.extras import execute_values
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_data():
    # Connect to SQLite
    sqlite_conn = sqlite3.connect('kimera_swm.db')
    sqlite_cursor = sqlite_conn.cursor()
    
    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="kimera_swm",
        user="kimera",
        password="kimera_secure_pass_2025"
    )
    pg_cursor = pg_conn.cursor()
    
    try:
        # Migrate geoids
        logger.info("Migrating geoids...")
        sqlite_cursor.execute("SELECT * FROM geoids")
        geoids = sqlite_cursor.fetchall()
        
        if geoids:
            execute_values(
                pg_cursor,
                """
                INSERT INTO geoids (geoid_id, symbolic_state, metadata_json, 
                                   semantic_state_json, semantic_vector)
                VALUES %s
                ON CONFLICT (geoid_id) DO NOTHING
                """,
                geoids
            )
            logger.info(f"Migrated {len(geoids)} geoids")
        
        # Migrate scars
        logger.info("Migrating scars...")
        sqlite_cursor.execute("SELECT * FROM scars")
        scars = sqlite_cursor.fetchall()
        
        if scars:
            execute_values(
                pg_cursor,
                """
                INSERT INTO scars (scar_id, geoids, reason, timestamp, resolved_by,
                                  pre_entropy, post_entropy, delta_entropy, cls_angle,
                                  semantic_polarity, mutation_frequency, weight,
                                  last_accessed, vault_id, scar_vector)
                VALUES %s
                ON CONFLICT (scar_id) DO NOTHING
                """,
                scars
            )
            logger.info(f"Migrated {len(scars)} scars")
        
        # Migrate insights
        logger.info("Migrating insights...")
        sqlite_cursor.execute("SELECT * FROM insights")
        insights = sqlite_cursor.fetchall()
        
        if insights:
            execute_values(
                pg_cursor,
                """
                INSERT INTO insights (insight_id, insight_type, source_resonance_id,
                                     echoform_repr, application_domains, confidence,
                                     entropy_reduction, utility_score, status,
                                     created_at, last_reinforced_cycle)
                VALUES %s
                ON CONFLICT (insight_id) DO NOTHING
                """,
                insights
            )
            logger.info(f"Migrated {len(insights)} insights")
        
        # Commit changes
        pg_conn.commit()
        logger.info("✓ Migration completed successfully")
        
    except Exception as e:
        pg_conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    migrate_data()
'''
        
        migration_file = self.project_root / "migrate_to_postgresql.py"
        try:
            migration_file.write_text(migration_script)
            migration_file.chmod(0o755)
            logger.info(f"✓ Created migration script: {migration_file}")
        except Exception as e:
            logger.error(f"Failed to create migration script: {e}")
            
    def update_database_py(self):
        """Update database.py to remove SQLite fallback"""
        logger.info("Updating database.py configuration...")
        
        db_file = self.project_root / "backend" / "vault" / "database.py"
        if db_file.exists():
            try:
                content = db_file.read_text()
                
                # Update the DATABASE_URL line to use PostgreSQL by default
                old_line = 'DATABASE_URL = os.getenv("DATABASE_URL", os.getenv("KIMERA_DATABASE_URL", "sqlite:///kimera_swm.db"))'
                new_line = 'DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm")'
                
                content = content.replace(old_line, new_line)
                
                db_file.write_text(content)
                logger.info("✓ Updated database.py to use PostgreSQL")
            except Exception as e:
                logger.error(f"Failed to update database.py: {e}")
                
    def print_instructions(self):
        """Print setup instructions"""
        logger.info("\n" + "="*60)
        logger.info("POSTGRESQL MIGRATION INSTRUCTIONS")
        logger.info("="*60)
        logger.info("\n1. Start PostgreSQL with pgvector:")
        logger.info("   docker-compose up -d")
        logger.info("\n2. Wait for PostgreSQL to be ready:")
        logger.info("   docker-compose logs -f postgres")
        logger.info("\n3. Create database schema:")
        logger.info("   python -c \"from src.vault.database import create_tables; create_tables()\"")
        logger.info("\n4. Run optimizations:")
        logger.info("   docker exec -it kimera_postgres psql -U kimera -d kimera_swm -c \"SELECT create_optimized_indexes();\"")
        logger.info("\n5. (Optional) Migrate existing SQLite data:")
        logger.info("   python migrate_to_postgresql.py")
        logger.info("\n6. Update your .env file:")
        logger.info("   cp .env.postgresql .env")
        logger.info("\n7. Restart Kimera:")
        logger.info("   python kimera.py")
        logger.info("="*60)
        
    def run_migration(self):
        """Run the complete migration process"""
        logger.info("Starting PostgreSQL migration...")
        
        self.create_postgresql_env()
        self.create_docker_compose()
        self.create_init_sql()
        self.create_migration_script()
        self.update_database_py()
        self.print_instructions()
        
        logger.info("\n✓ PostgreSQL migration files created successfully!")


if __name__ == "__main__":
    migrator = PostgreSQLMigrator()
    migrator.run_migration() 