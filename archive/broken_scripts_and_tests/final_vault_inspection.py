#!/usr/bin/env python3
"""
Final Comprehensive Vault Inspection
====================================

Complete vault system inspection with proper error handling.
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

def inspect_vault_system():
    """Complete vault system inspection"""
    
    logger.info("🏛️  KIMERA VAULT SYSTEM - FINAL INSPECTION")
    logger.info("=" * 60)
    
    # Find database
    db_path = os.path.join(ROOT_DIR, "kimera_swm.db")
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found at {db_path}")
        return
    
    logger.info(f"📁 Database: {db_path}")
    logger.info(f"📊 Size: {os.path.getsize(db_path)
    logger.info(f"🕐 Inspection: {datetime.now()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table schemas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"\n🗂️  Tables: {', '.join(tables)
        
        # Inspect scars table
        if "scars" in tables:
            logger.info(f"\n🏛️  VAULT SCARS ANALYSIS")
            logger.info("=" * 40)
            
            # Get scar statistics
            cursor.execute("SELECT COUNT(*) FROM scars;")
            total_scars = cursor.fetchone()[0]
            
            cursor.execute("SELECT vault_id, COUNT(*), SUM(weight), AVG(weight) FROM scars GROUP BY vault_id;")
            vault_stats = cursor.fetchall()
            
            logger.info(f"📊 Total Scars: {total_scars}")
            
            vault_summary = {}
            for vault_id, count, total_weight, avg_weight in vault_stats:
                vault_summary[vault_id] = {
                    "count": count,
                    "total_weight": total_weight or 0,
                    "avg_weight": avg_weight or 0
                }
                logger.info(f"\n🏛️  {vault_id.upper()
                logger.info(f"   Scars: {count}")
                logger.info(f"   Total Weight: {total_weight:.2f}")
                logger.info(f"   Average Weight: {avg_weight:.4f}")
            
            # Balance analysis
            vault_a_count = vault_summary.get("vault_a", {}).get("count", 0)
            vault_b_count = vault_summary.get("vault_b", {}).get("count", 0)
            imbalance = abs(vault_a_count - vault_b_count)
            
            logger.info(f"\n⚖️  BALANCE ANALYSIS:")
            logger.info(f"   Vault A: {vault_a_count} scars")
            logger.info(f"   Vault B: {vault_b_count} scars")
            logger.info(f"   Imbalance: {imbalance} scars")
            
            if imbalance == 0:
                balance_status = "🟢 PERFECT"
            elif imbalance <= 1:
                balance_status = "🟢 EXCELLENT"
            elif imbalance <= 5:
                balance_status = "🟡 GOOD"
            else:
                balance_status = "🔴 POOR"
            
            logger.info(f"   Status: {balance_status}")
            
            # Recent scars
            logger.info(f"\n📋 RECENT SCARS:")
            cursor.execute("SELECT scar_id, vault_id, reason, weight, timestamp FROM scars ORDER BY timestamp DESC LIMIT 5;")
            recent_scars = cursor.fetchall()
            
            for i, (scar_id, vault_id, reason, weight, timestamp) in enumerate(recent_scars, 1):
                logger.info(f"   {i}. {scar_id} ({vault_id})
                logger.info(f"      {timestamp}")
            
            # Scar reasons distribution
            cursor.execute("SELECT reason, COUNT(*) FROM scars GROUP BY reason ORDER BY COUNT(*) DESC;")
            reasons = cursor.fetchall()
            
            logger.info(f"\n📊 SCAR REASONS:")
            for reason, count in reasons:
                logger.info(f"   {reason}: {count}")
            
            # Temporal analysis
            cursor.execute("SELECT DATE(timestamp) as date, COUNT(*) FROM scars GROUP BY DATE(timestamp) ORDER BY date DESC LIMIT 7;")
            daily_counts = cursor.fetchall()
            
            logger.info(f"\n📅 DAILY SCAR CREATION (Last 7 days)
            for date, count in daily_counts:
                logger.info(f"   {date}: {count} scars")
        
        # Inspect geoids table
        if "geoids" in tables:
            logger.info(f"\n🧠 GEOIDS ANALYSIS")
            logger.info("=" * 40)
            
            # Get geoid table schema
            cursor.execute("PRAGMA table_info(geoids);")
            geoid_columns = [col[1] for col in cursor.fetchall()]
            logger.info(f"📋 Columns: {', '.join(geoid_columns)
            
            cursor.execute("SELECT COUNT(*) FROM geoids;")
            total_geoids = cursor.fetchone()[0]
            logger.info(f"📊 Total Geoids: {total_geoids}")
            
            if total_geoids > 0:
                # Get recent geoids with available columns
                available_columns = ["geoid_id"]
                if "semantic_state" in geoid_columns:
                    available_columns.append("semantic_state")
                elif "semantic_features" in geoid_columns:
                    available_columns.append("semantic_features")
                
                if "symbolic_state" in geoid_columns:
                    available_columns.append("symbolic_state")
                elif "symbolic_content" in geoid_columns:
                    available_columns.append("symbolic_content")
                
                query = f"SELECT {', '.join(available_columns)} FROM geoids ORDER BY rowid DESC LIMIT 3;"
                cursor.execute(query)
                recent_geoids = cursor.fetchall()
                
                logger.info(f"\n📋 RECENT GEOIDS (Last 3)
                for i, geoid_data in enumerate(recent_geoids, 1):
                    geoid_id = geoid_data[0]
                    logger.info(f"   {i}. {geoid_id}")
                    
                    # Show semantic features if available
                    if len(geoid_data) > 1 and geoid_data[1]:
                        try:
                            semantic = json.loads(geoid_data[1])
                            feature_count = len(semantic) if isinstance(semantic, dict) else 0
                            logger.info(f"      Semantic features: {feature_count}")
                        except:
                            logger.info(f"      Semantic data: {str(geoid_data[1])
                    
                    # Show symbolic content if available
                    if len(geoid_data) > 2 and geoid_data[2]:
                        try:
                            symbolic = json.loads(geoid_data[2])
                            logger.info(f"      Symbolic: {symbolic}")
                        except:
                            logger.info(f"      Symbolic data: {str(geoid_data[2])
        
        # System health assessment
        logger.info(f"\n🏥 SYSTEM HEALTH ASSESSMENT")
        logger.info("=" * 40)
        
        health_score = 100
        issues = []
        recommendations = []
        
        # Check vault balance
        if total_scars > 0:
            if imbalance == 0:
                logger.info("✅ Vault balance: PERFECT")
            elif imbalance <= 1:
                logger.info("✅ Vault balance: EXCELLENT")
            elif imbalance <= 5:
                logger.warning("⚠️  Vault balance: GOOD (minor imbalance)
                health_score -= 10
                recommendations.append("Consider vault rebalancing")
            else:
                logger.error("❌ Vault balance: POOR (significant imbalance)
                health_score -= 30
                issues.append(f"Vault imbalance: {imbalance} scars")
                recommendations.append("Immediate vault rebalancing required")
        else:
            logger.info("ℹ️  Vault balance: N/A (no scars)
        
        # Check data integrity
        cursor.execute("SELECT COUNT(*) FROM scars WHERE scar_id IS NULL OR vault_id IS NULL;")
        null_data = cursor.fetchone()[0]
        
        if null_data == 0:
            logger.info("✅ Data integrity: GOOD")
        else:
            logger.error(f"❌ Data integrity: ISSUES ({null_data} records with null data)
            health_score -= 20
            issues.append(f"{null_data} scars with missing data")
        
        # Overall health
        if health_score >= 90:
            health_status = "🟢 EXCELLENT"
        elif health_score >= 70:
            health_status = "🟡 GOOD"
        elif health_score >= 50:
            health_status = "🟠 FAIR"
        else:
            health_status = "🔴 POOR"
        
        logger.info(f"\n🎯 OVERALL HEALTH: {health_status} ({health_score}/100)
        
        if issues:
            logger.warning(f"\n⚠️  ISSUES FOUND:")
            for issue in issues:
                logger.info(f"   - {issue}")
        
        if recommendations:
            logger.info(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                logger.info(f"   - {rec}")
        
        # Summary statistics
        logger.info(f"\n📊 SUMMARY STATISTICS")
        logger.info("=" * 40)
        logger.info(f"Database Size: {os.path.getsize(db_path)
        logger.info(f"Total Tables: {len(tables)
        logger.info(f"Total Scars: {total_scars}")
        logger.info(f"Total Geoids: {total_geoids if 'geoids' in tables else 'N/A'}")
        logger.info(f"Vault Balance: {imbalance} scar difference")
        logger.info(f"Health Score: {health_score}/100")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_vault_system()