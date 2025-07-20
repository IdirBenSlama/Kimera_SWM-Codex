#!/usr/bin/env python3
"""
Comprehensive Vault Inspection Tool
===================================

This script provides detailed inspection of the KIMERA vault system,
including scar analysis, distribution metrics, and health assessment.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
import sqlite3
from pathlib import Path

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

try:
    from backend.vault.vault_manager import VaultManager
    from backend.vault.database import SessionLocal, ScarDB, GeoidDB
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    logger.warning("⚠️  Backend modules not available, using direct database access")

class VaultInspector:
    def __init__(self):
        self.db_path = self.find_database()
        self.vault_manager = None
        
        if BACKEND_AVAILABLE:
            try:
                self.vault_manager = VaultManager()
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize VaultManager: {e}")
    
    def find_database(self) -> str:
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
        
        # Look for any .db file
        for db_file in Path(ROOT_DIR).rglob("*.db"):
            if "kimera" in db_file.name.lower():
                return str(db_file)
        
        return os.path.join(ROOT_DIR, "kimera_swm.db")
    
    def inspect_database_direct(self) -> Dict[str, Any]:
        """Direct database inspection using SQLite"""
        if not os.path.exists(self.db_path):
            return {"error": f"Database not found at {self.db_path}"}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            result = {
                "database_path": self.db_path,
                "database_size": os.path.getsize(self.db_path),
                "tables": tables,
                "inspection_time": datetime.now().isoformat()
            }
            
            # Inspect scars table if it exists
            if "scars" in tables:
                cursor.execute("SELECT COUNT(*) FROM scars;")
                total_scars = cursor.fetchone()[0]
                
                cursor.execute("SELECT vault_id, COUNT(*) FROM scars GROUP BY vault_id;")
                vault_distribution = dict(cursor.fetchall())
                
                cursor.execute("SELECT vault_id, SUM(weight) FROM scars GROUP BY vault_id;")
                vault_weights = dict(cursor.fetchall())
                
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM scars;")
                time_range = cursor.fetchone()
                
                result["scars"] = {
                    "total_count": total_scars,
                    "vault_distribution": vault_distribution,
                    "vault_weights": vault_weights,
                    "time_range": {
                        "earliest": time_range[0],
                        "latest": time_range[1]
                    }
                }
                
                # Get sample scars
                cursor.execute("SELECT * FROM scars LIMIT 5;")
                columns = [description[0] for description in cursor.description]
                sample_scars = []
                for row in cursor.fetchall():
                    sample_scars.append(dict(zip(columns, row)))
                result["scars"]["samples"] = sample_scars
            
            # Inspect geoids table if it exists
            if "geoids" in tables:
                cursor.execute("SELECT COUNT(*) FROM geoids;")
                total_geoids = cursor.fetchone()[0]
                
                cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM geoids;")
                geoid_time_range = cursor.fetchone()
                
                result["geoids"] = {
                    "total_count": total_geoids,
                    "time_range": {
                        "earliest": geoid_time_range[0],
                        "latest": geoid_time_range[1]
                    }
                }
            
            conn.close()
            return result
            
        except Exception as e:
            return {"error": f"Database inspection failed: {e}"}
    
    def inspect_vault_manager(self) -> Dict[str, Any]:
        """Inspect vault using VaultManager if available"""
        if not self.vault_manager:
            return {"error": "VaultManager not available"}
        
        try:
            result = {
                "vault_a": {
                    "scar_count": self.vault_manager.get_total_scar_count("vault_a"),
                    "total_weight": self.vault_manager.get_total_scar_weight("vault_a")
                },
                "vault_b": {
                    "scar_count": self.vault_manager.get_total_scar_count("vault_b"),
                    "total_weight": self.vault_manager.get_total_scar_weight("vault_b")
                }
            }
            
            # Calculate balance metrics
            total_scars = result["vault_a"]["scar_count"] + result["vault_b"]["scar_count"]
            total_weight = result["vault_a"]["total_weight"] + result["vault_b"]["total_weight"]
            
            if total_scars > 0:
                scar_imbalance = abs(result["vault_a"]["scar_count"] - result["vault_b"]["scar_count"])
                weight_imbalance = abs(result["vault_a"]["total_weight"] - result["vault_b"]["total_weight"])
                
                result["balance_metrics"] = {
                    "total_scars": total_scars,
                    "total_weight": total_weight,
                    "scar_imbalance": scar_imbalance,
                    "weight_imbalance": weight_imbalance,
                    "scar_balance_percentage": ((total_scars - scar_imbalance) / total_scars) * 100,
                    "weight_balance_percentage": ((total_weight - weight_imbalance) / total_weight) * 100 if total_weight > 0 else 100
                }
                
                # Balance quality assessment
                if scar_imbalance <= 1:
                    balance_quality = "EXCELLENT"
                elif scar_imbalance <= 5:
                    balance_quality = "GOOD"
                elif scar_imbalance <= 10:
                    balance_quality = "FAIR"
                else:
                    balance_quality = "POOR"
                
                result["balance_metrics"]["quality"] = balance_quality
            
            return result
            
        except Exception as e:
            return {"error": f"VaultManager inspection failed: {e}"}
    
    def analyze_scar_distribution(self) -> Dict[str, Any]:
        """Analyze scar distribution patterns"""
        if not os.path.exists(self.db_path):
            return {"error": "Database not found"}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if scars table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scars';")
            if not cursor.fetchone():
                return {"error": "Scars table not found"}
            
            # Analyze scar patterns
            cursor.execute("""
                SELECT 
                    vault_id,
                    COUNT(*) as count,
                    AVG(weight) as avg_weight,
                    MIN(weight) as min_weight,
                    MAX(weight) as max_weight,
                    SUM(weight) as total_weight
                FROM scars 
                GROUP BY vault_id
            """)
            
            vault_stats = {}
            for row in cursor.fetchall():
                vault_id, count, avg_weight, min_weight, max_weight, total_weight = row
                vault_stats[vault_id] = {
                    "count": count,
                    "avg_weight": round(avg_weight, 4) if avg_weight else 0,
                    "min_weight": min_weight,
                    "max_weight": max_weight,
                    "total_weight": total_weight
                }
            
            # Analyze scar reasons
            cursor.execute("SELECT reason, COUNT(*) FROM scars GROUP BY reason ORDER BY COUNT(*) DESC;")
            reason_distribution = dict(cursor.fetchall())
            
            # Analyze temporal patterns
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as daily_count
                FROM scars 
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 10
            """)
            daily_patterns = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "vault_statistics": vault_stats,
                "reason_distribution": reason_distribution,
                "daily_patterns": daily_patterns
            }
            
        except Exception as e:
            return {"error": f"Scar analysis failed: {e}"}
    
    def generate_vault_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive vault health report"""
        report = {
            "inspection_timestamp": datetime.now().isoformat(),
            "database_info": {},
            "vault_manager_info": {},
            "scar_analysis": {},
            "health_assessment": {}
        }
        
        # Database inspection
        db_info = self.inspect_database_direct()
        report["database_info"] = db_info
        
        # VaultManager inspection
        vm_info = self.inspect_vault_manager()
        report["vault_manager_info"] = vm_info
        
        # Scar analysis
        scar_analysis = self.analyze_scar_distribution()
        report["scar_analysis"] = scar_analysis
        
        # Health assessment
        health = {"status": "UNKNOWN", "issues": [], "recommendations": []}
        
        if "error" not in db_info and "scars" in db_info:
            total_scars = db_info["scars"]["total_count"]
            vault_dist = db_info["scars"]["vault_distribution"]
            
            if total_scars == 0:
                health["status"] = "EMPTY"
                health["recommendations"].append("System appears to be newly initialized")
            elif total_scars > 0:
                # Check vault balance
                vault_a_count = vault_dist.get("vault_a", 0)
                vault_b_count = vault_dist.get("vault_b", 0)
                imbalance = abs(vault_a_count - vault_b_count)
                
                if imbalance <= 1:
                    health["status"] = "EXCELLENT"
                elif imbalance <= 5:
                    health["status"] = "GOOD"
                    health["recommendations"].append("Consider vault rebalancing")
                else:
                    health["status"] = "POOR"
                    health["issues"].append(f"Significant vault imbalance: {imbalance} scars")
                    health["recommendations"].append("Immediate vault rebalancing recommended")
        
        report["health_assessment"] = health
        return report
    
    def print_vault_report(self):
        """Print formatted vault inspection report"""
        logger.info("🏛️  KIMERA VAULT SYSTEM INSPECTION")
        logger.info("=" * 60)
        
        report = self.generate_vault_health_report()
        
        # Database Information
        logger.info(f"\n📊 DATABASE INFORMATION")
        logger.info("-" * 30)
        db_info = report["database_info"]
        if "error" in db_info:
            logger.error(f"❌ Error: {db_info['error']}")
        else:
            logger.info(f"📁 Database Path: {db_info['database_path']}")
            logger.info(f"💾 Database Size: {db_info['database_size']:,} bytes")
            logger.info(f"🗂️  Tables: {', '.join(db_info['tables'])
        
        # Vault Statistics
        logger.info(f"\n🏛️  VAULT STATISTICS")
        logger.info("-" * 30)
        vm_info = report["vault_manager_info"]
        if "error" in vm_info:
            logger.error(f"❌ Error: {vm_info['error']}")
        else:
            logger.info(f"Vault A: {vm_info['vault_a']['scar_count']} scars (weight: {vm_info['vault_a']['total_weight']:.2f})
            logger.info(f"Vault B: {vm_info['vault_b']['scar_count']} scars (weight: {vm_info['vault_b']['total_weight']:.2f})
            
            if "balance_metrics" in vm_info:
                metrics = vm_info["balance_metrics"]
                logger.info(f"\n⚖️  BALANCE METRICS")
                logger.info(f"   Total Scars: {metrics['total_scars']}")
                logger.info(f"   Total Weight: {metrics['total_weight']:.2f}")
                logger.info(f"   Scar Imbalance: {metrics['scar_imbalance']}")
                logger.info(f"   Balance Quality: {metrics['quality']}")
                logger.info(f"   Balance Percentage: {metrics['scar_balance_percentage']:.1f}%")
        
        # Scar Analysis
        logger.debug(f"\n🔍 SCAR ANALYSIS")
        logger.info("-" * 30)
        scar_analysis = report["scar_analysis"]
        if "error" in scar_analysis:
            logger.error(f"❌ Error: {scar_analysis['error']}")
        else:
            if "vault_statistics" in scar_analysis:
                for vault_id, stats in scar_analysis["vault_statistics"].items():
                    logger.info(f"{vault_id.upper()
                    logger.info(f"   Count: {stats['count']}")
                    logger.info(f"   Avg Weight: {stats['avg_weight']:.4f}")
                    logger.info(f"   Weight Range: {stats['min_weight']:.2f} - {stats['max_weight']:.2f}")
            
            if "reason_distribution" in scar_analysis:
                logger.info(f"\n📋 SCAR REASONS:")
                for reason, count in list(scar_analysis["reason_distribution"].items())[:5]:
                    logger.info(f"   {reason}: {count}")
        
        # Health Assessment
        logger.info(f"\n🏥 HEALTH ASSESSMENT")
        logger.info("-" * 30)
        health = report["health_assessment"]
        status_emoji = {
            "EXCELLENT": "🟢",
            "GOOD": "🟡", 
            "FAIR": "🟠",
            "POOR": "🔴",
            "EMPTY": "⚪",
            "UNKNOWN": "❓"
        }
        logger.info(f"Status: {status_emoji.get(health['status'], '❓')
        
        if health["issues"]:
            logger.warning(f"\n⚠️  ISSUES:")
            for issue in health["issues"]:
                logger.info(f"   - {issue}")
        
        if health["recommendations"]:
            logger.info(f"\n💡 RECOMMENDATIONS:")
            for rec in health["recommendations"]:
                logger.info(f"   - {rec}")
        
        # Save report
        report_file = os.path.join(ROOT_DIR, "vault_inspection_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n💾 Detailed report saved to: {report_file}")

def main():
    """Main inspection function"""
    inspector = VaultInspector()
    inspector.print_vault_report()

if __name__ == "__main__":
    main()