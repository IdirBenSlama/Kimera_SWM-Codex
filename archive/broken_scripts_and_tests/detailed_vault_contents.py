#!/usr/bin/env python3
"""
Detailed Vault Contents Inspector
=================================

This script provides detailed inspection of individual scars and their contents.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Any

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

def find_database() -> str:
    """Find the KIMERA database file"""
    possible_paths = [
        "kimera_swm.db",
        "data/kimera_swm.db", 
        "backend/kimera_swm.db"
    ]
    
    for path in possible_paths:
        full_path = os.path.join(ROOT_DIR, path)
        if os.path.exists(full_path):
            return full_path
    
    return os.path.join(ROOT_DIR, "kimera_swm.db")

def inspect_vault_contents():
    """Inspect detailed vault contents"""
    db_path = find_database()
    
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found at {db_path}")
        return
    
    logger.debug("🔍 DETAILED VAULT CONTENTS INSPECTION")
    logger.info("=" * 60)
    logger.info(f"📁 Database: {db_path}")
    logger.info(f"📊 Database Size: {os.path.getsize(db_path)
    logger.info(f"🕐 Inspection Time: {datetime.now()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"\n🗂️  Available Tables: {', '.join(tables)
        
        # Inspect scars table
        if "scars" in tables:
            logger.info(f"\n🏛️  SCARS TABLE ANALYSIS")
            logger.info("-" * 40)
            
            # Get table schema
            cursor.execute("PRAGMA table_info(scars);")
            columns = cursor.fetchall()
            logger.info(f"📋 Columns: {', '.join([col[1] for col in columns])
            
            # Get all scars
            cursor.execute("SELECT * FROM scars ORDER BY timestamp DESC;")
            scars = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            logger.info(f"\n📊 Total Scars: {len(scars)
            
            if scars:
                logger.debug(f"\n🔍 DETAILED SCAR CONTENTS:")
                logger.info("=" * 60)
                
                for i, scar_data in enumerate(scars, 1):
                    scar_dict = dict(zip(column_names, scar_data))
                    
                    logger.info(f"\n🏷️  SCAR #{i}")
                    logger.info("-" * 20)
                    logger.info(f"ID: {scar_dict.get('scar_id', 'N/A')
                    logger.info(f"Vault: {scar_dict.get('vault_id', 'N/A')
                    logger.info(f"Reason: {scar_dict.get('reason', 'N/A')
                    logger.info(f"Weight: {scar_dict.get('weight', 'N/A')
                    logger.info(f"Timestamp: {scar_dict.get('timestamp', 'N/A')
                    logger.info(f"Resolved By: {scar_dict.get('resolved_by', 'N/A')
                    
                    # Parse geoids if available
                    geoids_str = scar_dict.get('geoids', '')
                    if geoids_str:
                        try:
                            geoids = json.loads(geoids_str)
                            logger.info(f"Geoids: {geoids}")
                        except:
                            logger.info(f"Geoids (raw)
                    
                    # Show entropy metrics
                    pre_entropy = scar_dict.get('pre_entropy', 'N/A')
                    post_entropy = scar_dict.get('post_entropy', 'N/A')
                    delta_entropy = scar_dict.get('delta_entropy', 'N/A')
                    
                    logger.info(f"Entropy - Pre: {pre_entropy}, Post: {post_entropy}, Delta: {delta_entropy}")
                    
                    # Show additional metrics
                    cls_angle = scar_dict.get('cls_angle', 'N/A')
                    semantic_polarity = scar_dict.get('semantic_polarity', 'N/A')
                    mutation_frequency = scar_dict.get('mutation_frequency', 'N/A')
                    
                    logger.info(f"CLS Angle: {cls_angle}")
                    logger.info(f"Semantic Polarity: {semantic_polarity}")
                    logger.info(f"Mutation Frequency: {mutation_frequency}")
                    
                    # Show scar vector if available (truncated)
                    scar_vector = scar_dict.get('scar_vector', '')
                    if scar_vector:
                        try:
                            vector = json.loads(scar_vector)
                            if isinstance(vector, list) and len(vector) > 0:
                                logger.info(f"Scar Vector: [{vector[0]:.4f}, {vector[1]:.4f}, ..., {vector[-1]:.4f}] (length: {len(vector)
                        except:
                            logger.info(f"Scar Vector: {scar_vector[:100]}..." if len(scar_vector)
        
        # Inspect geoids table
        if "geoids" in tables:
            logger.info(f"\n🧠 GEOIDS TABLE ANALYSIS")
            logger.info("-" * 40)
            
            cursor.execute("SELECT COUNT(*) FROM geoids;")
            geoid_count = cursor.fetchone()[0]
            logger.info(f"📊 Total Geoids: {geoid_count}")
            
            if geoid_count > 0:
                # Get recent geoids
                cursor.execute("SELECT geoid_id, semantic_state, symbolic_state FROM geoids ORDER BY rowid DESC LIMIT 5;")
                recent_geoids = cursor.fetchall()
                
                logger.debug(f"\n🔍 RECENT GEOIDS (Last 5)
                for i, (geoid_id, semantic_state, symbolic_state) in enumerate(recent_geoids, 1):
                    logger.info(f"\n🏷️  GEOID #{i}")
                    logger.info(f"ID: {geoid_id}")
                    
                    # Parse semantic state
                    try:
                        semantic = json.loads(semantic_state) if semantic_state else {}
                        logger.info(f"Semantic Features: {list(semantic.keys()
                    except:
                        logger.info(f"Semantic State (raw)
                    
                    # Parse symbolic state
                    try:
                        symbolic = json.loads(symbolic_state) if symbolic_state else {}
                        logger.info(f"Symbolic Content: {symbolic}")
                    except:
                        logger.info(f"Symbolic State (raw)
        
        # Vault distribution analysis
        if "scars" in tables:
            logger.info(f"\n📊 VAULT DISTRIBUTION ANALYSIS")
            logger.info("-" * 40)
            
            cursor.execute("SELECT vault_id, COUNT(*), SUM(weight), AVG(weight) FROM scars GROUP BY vault_id;")
            vault_stats = cursor.fetchall()
            
            for vault_id, count, total_weight, avg_weight in vault_stats:
                logger.info(f"{vault_id.upper()
                logger.info(f"  Scars: {count}")
                logger.info(f"  Total Weight: {total_weight:.2f}")
                logger.info(f"  Average Weight: {avg_weight:.4f}")
            
            # Calculate balance
            if len(vault_stats) >= 2:
                vault_a_count = next((count for vault_id, count, _, _ in vault_stats if vault_id == "vault_a"), 0)
                vault_b_count = next((count for vault_id, count, _, _ in vault_stats if vault_id == "vault_b"), 0)
                imbalance = abs(vault_a_count - vault_b_count)
                
                logger.info(f"\n⚖️  BALANCE ASSESSMENT:")
                logger.info(f"Imbalance: {imbalance} scars")
                
                if imbalance == 0:
                    balance_quality = "PERFECT"
                elif imbalance <= 1:
                    balance_quality = "EXCELLENT"
                elif imbalance <= 5:
                    balance_quality = "GOOD"
                else:
                    balance_quality = "POOR"
                
                logger.info(f"Quality: {balance_quality}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_vault_contents()