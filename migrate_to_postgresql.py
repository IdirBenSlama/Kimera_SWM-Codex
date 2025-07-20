#!/usr/bin/env python3
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
        logger.info("Migration completed successfully")
        
    except Exception as e:
        pg_conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    migrate_data()
