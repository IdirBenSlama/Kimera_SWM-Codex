#!/usr/bin/env python3
"""
Kimera SWM Database Initialization
"""

import sqlite3
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_sqlite_database():
    """Create SQLite database with proper schema"""
    project_root = Path(__file__).parent.parent
    db_dir = project_root / "data" / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = db_dir / "kimera_system.db"
    
    # SQLite-compatible schema
    schema_sql = """
    CREATE TABLE IF NOT EXISTS geoid_states (
        id TEXT PRIMARY KEY,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        state_vector TEXT NOT NULL,
        meta_data TEXT DEFAULT '{}',
        entropy REAL NOT NULL,
        coherence_factor REAL NOT NULL,
        energy_level REAL DEFAULT 1.0,
        creation_context TEXT,
        tags TEXT
    );
    
    CREATE TABLE IF NOT EXISTS cognitive_transitions (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        transition_energy REAL NOT NULL,
        conservation_error REAL NOT NULL,
        transition_type TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        meta_data TEXT DEFAULT '{}',
        FOREIGN KEY (source_id) REFERENCES geoid_states (id),
        FOREIGN KEY (target_id) REFERENCES geoid_states (id)
    );
    
    CREATE TABLE IF NOT EXISTS semantic_embeddings (
        id TEXT PRIMARY KEY,
        text_content TEXT NOT NULL,
        embedding TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source TEXT,
        meta_data TEXT DEFAULT '{}'
    );
    
    CREATE TABLE IF NOT EXISTS scar_records (
        id TEXT PRIMARY KEY,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        anomaly_type TEXT NOT NULL,
        severity_level TEXT NOT NULL,
        description TEXT,
        context_data TEXT DEFAULT '{}',
        resolution_status TEXT DEFAULT 'pending'
    );
    
    CREATE TABLE IF NOT EXISTS system_metrics (
        id TEXT PRIMARY KEY,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        metric_name TEXT NOT NULL,
        metric_value REAL,
        metric_data TEXT DEFAULT '{}'
    );
    
    CREATE TABLE IF NOT EXISTS vault_entries (
        id TEXT PRIMARY KEY,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        entry_type TEXT NOT NULL,
        entry_data TEXT NOT NULL,
        encryption_status TEXT DEFAULT 'none'
    );
    """
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Execute schema
        for statement in schema_sql.split(';'):
            if statement.strip():
                cursor.execute(statement)
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ SQLite database created: {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database creation failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_sqlite_database()
