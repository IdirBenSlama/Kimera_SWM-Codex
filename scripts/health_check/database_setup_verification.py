#!/usr/bin/env python3
# Fix import paths
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


"""
Database Setup Verification Script for Kimera SWM

This script verifies that all required databases are properly set up and accessible.
Follows Kimera SWM file placement rules - saves reports to appropriate directories.
"""

import os
import sys
import json
import logging
import psycopg2
import redis
import requests
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Ensure proper logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseSetupVerifier:
    """Verifies all Kimera SWM database requirements"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "databases": {},
            "overall_status": "unknown",
            "errors": [],
            "recommendations": []
        }
        
        # Ensure reports directory exists
        self.reports_dir = Path("docs/reports/health")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def verify_postgresql(self) -> Dict[str, Any]:
        """Verify PostgreSQL with pgvector extension"""
        logger.info("🔍 Checking PostgreSQL connection and pgvector extension...")
        
        status = {
            "service": "PostgreSQL",
            "status": "unknown",
            "version": None,
            "pgvector_available": False,
            "connection_string": None,
            "error": None
        }
        
        try:
            # Try different connection configurations
            connection_configs = [
                {
                    "host": "localhost",
                    "port": 5432,
                    "database": "kimera_swm",
                    "user": "kimera",
                    "password": "kimera_secure_pass_2025"
                },
                {
                    "host": "kimera_postgres",
                    "port": 5432,
                    "database": "kimera_swm", 
                    "user": "kimera",
                    "password": "kimera_secure_pass_2025"
                }
            ]
            
            for config in connection_configs:
                try:
                    conn = psycopg2.connect(**config)
                    cursor = conn.cursor()
                    
                    # Get PostgreSQL version
                    cursor.execute("SELECT version();")
                    status["version"] = cursor.fetchone()[0]
                    
                    # Check for pgvector extension
                    cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
                    status["pgvector_available"] = cursor.fetchone()[0]
                    
                    # Test vector operations if pgvector is available
                    if status["pgvector_available"]:
                        cursor.execute("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector;")
                        distance = cursor.fetchone()[0]
                        logger.info(f"✓ pgvector test successful - distance calculation: {distance}")
                    
                    status["status"] = "connected"
                    status["connection_string"] = f"postgresql://{config['user']}@{config['host']}:{config['port']}/{config['database']}"
                    
                    cursor.close()
                    conn.close()
                    break
                    
                except psycopg2.Error as e:
                    continue
                    
        except Exception as e:
            status["error"] = str(e)
            status["status"] = "failed"
            
        if status["status"] == "connected":
            logger.info(f"✓ PostgreSQL connected: {status['version']}")
            if status["pgvector_available"]:
                logger.info("✓ pgvector extension is available")
            else:
                logger.warning("⚠ pgvector extension not found")
                self.results["recommendations"].append("Install pgvector extension: CREATE EXTENSION vector;")
        else:
            logger.error(f"✗ PostgreSQL connection failed: {status.get('error', 'Unknown error')}")
            
        return status
    
    def verify_redis(self) -> Dict[str, Any]:
        """Verify Redis connection"""
        logger.info("🔍 Checking Redis connection...")
        
        status = {
            "service": "Redis",
            "status": "unknown",
            "version": None,
            "memory_info": None,
            "error": None
        }
        
        try:
            # Try different Redis configurations
            redis_configs = [
                {"host": "localhost", "port": 6379, "db": 0},
                {"host": "kimera_redis", "port": 6379, "db": 0}
            ]
            
            for config in redis_configs:
                try:
                    r = redis.Redis(**config)
                    
                    # Test connection
                    r.ping()
                    
                    # Get Redis info
                    info = r.info()
                    status["version"] = info.get("redis_version")
                    status["memory_info"] = {
                        "used_memory": info.get("used_memory"),
                        "used_memory_human": info.get("used_memory_human"),
                        "maxmemory": info.get("maxmemory")
                    }
                    
                    status["status"] = "connected"
                    break
                    
                except redis.RedisError:
                    continue
                    
        except Exception as e:
            status["error"] = str(e)
            status["status"] = "failed"
            
        if status["status"] == "connected":
            logger.info(f"✓ Redis connected: v{status['version']}")
            logger.info(f"✓ Memory usage: {status['memory_info']['used_memory_human']}")
        else:
            logger.error(f"✗ Redis connection failed: {status.get('error', 'Unknown error')}")
            
        return status
    
    def verify_questdb(self) -> Dict[str, Any]:
        """Verify QuestDB (optional for trading)"""
        logger.info("🔍 Checking QuestDB connection (optional)...")
        
        status = {
            "service": "QuestDB",
            "status": "unknown", 
            "version": None,
            "rest_api": False,
            "influx_protocol": False,
            "error": None
        }
        
        try:
            # Check QuestDB REST API
            response = requests.get("http://localhost:9000/", timeout=5)
            if response.status_code == 200:
                status["rest_api"] = True
                status["status"] = "connected"
                logger.info("✓ QuestDB REST API accessible")
        except Exception as e:
            logger.error(f"Error in database_setup_verification.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            status["status"] = "not_running"
            logger.info("ℹ QuestDB not running (optional for trading features)")
            
        return status
    
    def verify_prometheus(self) -> Dict[str, Any]:
        """Verify Prometheus (optional for monitoring)"""
        logger.info("🔍 Checking Prometheus connection (optional)...")
        
        status = {
            "service": "Prometheus",
            "status": "unknown",
            "version": None,
            "targets": [],
            "error": None
        }
        
        try:
            # Check Prometheus API
            response = requests.get("http://localhost:9090/api/v1/status/buildinfo", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status["version"] = data.get("data", {}).get("version")
                status["status"] = "connected"
                logger.info(f"✓ Prometheus connected: v{status['version']}")
        except Exception as e:
            logger.error(f"Error in database_setup_verification.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            status["status"] = "not_running"
            logger.info("ℹ Prometheus not running (optional for monitoring)")
            
        return status
    
    def check_docker_services(self) -> Dict[str, Any]:
        """Check Docker services status"""
        logger.info("🔍 Checking Docker services...")
        
        status = {
            "docker_available": False,
            "services": {},
            "compose_file_exists": False
        }
        
        try:
            import subprocess
            
            # Check if docker-compose.yml exists
            compose_path = Path("config/docker/docker-compose.yml")
            status["compose_file_exists"] = compose_path.exists()
            
            # Check Docker availability
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                status["docker_available"] = True
                logger.info(f"✓ Docker available: {result.stdout.strip()}")
            
        except FileNotFoundError:
            logger.warning("⚠ Docker not found in PATH")
        except Exception as e:
            logger.error(f"✗ Docker check failed: {e}")
            
        return status
    
    def run_full_verification(self) -> Dict[str, Any]:
        """Run complete database setup verification"""
        logger.info("🚀 Starting Kimera SWM Database Setup Verification")
        logger.info("=" * 60)
        
        # Check all database services
        self.results["databases"]["postgresql"] = self.verify_postgresql()
        self.results["databases"]["redis"] = self.verify_redis()
        self.results["databases"]["questdb"] = self.verify_questdb()
        self.results["databases"]["prometheus"] = self.verify_prometheus()
        self.results["databases"]["docker"] = self.check_docker_services()
        
        # Determine overall status
        critical_services = ["postgresql", "redis"]
        critical_failures = []
        
        for service in critical_services:
            if self.results["databases"][service]["status"] != "connected":
                critical_failures.append(service)
        
        if not critical_failures:
            self.results["overall_status"] = "healthy"
            logger.info("✅ All critical databases are operational")
        elif len(critical_failures) == 1:
            self.results["overall_status"] = "degraded"
            logger.warning(f"⚠ Critical service failure: {critical_failures[0]}")
        else:
            self.results["overall_status"] = "failed"
            logger.error(f"❌ Multiple critical services failed: {critical_failures}")
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.results
    
    def _generate_recommendations(self):
        """Generate setup recommendations based on verification results"""
        
        if self.results["databases"]["postgresql"]["status"] != "connected":
            self.results["recommendations"].extend([
                "Start PostgreSQL: docker-compose up postgres -d",
                "Verify PostgreSQL credentials in environment variables",
                "Check if port 5432 is available"
            ])
        
        if self.results["databases"]["redis"]["status"] != "connected":
            self.results["recommendations"].extend([
                "Start Redis: docker-compose up redis -d", 
                "Check if port 6379 is available"
            ])
        
        if not self.results["databases"]["postgresql"]["pgvector_available"]:
            self.results["recommendations"].append(
                "Install pgvector: Run init_db.sql or CREATE EXTENSION vector;"
            )
        
        if self.results["databases"]["docker"]["compose_file_exists"] and self.results["overall_status"] != "healthy":
            self.results["recommendations"].insert(0, 
                "Quick fix: Run 'docker-compose up -d' to start all services"
            )
    
    def save_report(self):
        """Save verification report to appropriate directory"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        report_path = self.reports_dir / f"{date_str}_database_setup_verification.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📄 Report saved to: {report_path}")
        
        # Also create a summary markdown report
        md_report_path = self.reports_dir / f"{date_str}_database_setup_summary.md"
        self._create_markdown_summary(md_report_path)
        
        return report_path
    
    def _create_markdown_summary(self, path: Path):
        """Create a human-readable markdown summary"""
        
        md_content = f"""# Kimera SWM Database Setup Verification Report

**Generated:** {self.results['timestamp']}  
**Overall Status:** {self.results['overall_status'].upper()}

## Database Services Status

| Service | Status | Version | Notes |
|---------|--------|---------|-------|
"""
        
        for service, data in self.results["databases"].items():
             status = data.get("status", "unknown")
             status_emoji = "✅" if status == "connected" else "❌" if status == "failed" else "⚠"
             version = data.get("version", "N/A")
             notes = "pgvector available" if service == "postgresql" and data.get("pgvector_available") else ""
             
             md_content += f"| {service} | {status_emoji} {status} | {version} | {notes} |\n"
        
        if self.results["recommendations"]:
            md_content += "\n## Recommendations\n\n"
            for i, rec in enumerate(self.results["recommendations"], 1):
                md_content += f"{i}. {rec}\n"
        
        md_content += f"\n## Quick Setup Commands\n\n"
        md_content += f"```bash\n"
        md_content += f"# Navigate to Kimera-SWM directory\n"
        md_content += f"cd Kimera-SWM\n\n"
        md_content += f"# Start all database services\n"
        md_content += f"docker-compose -f config/docker/docker-compose.yml up -d\n\n"
        md_content += f"# Check service status\n"
        md_content += f"docker-compose -f config/docker/docker-compose.yml ps\n\n"
        md_content += f"# View logs if needed\n"
        md_content += f"docker-compose -f config/docker/docker-compose.yml logs\n"
        md_content += f"```\n"
        
        with open(path, 'w') as f:
            f.write(md_content)


def main():
    """Main verification function"""
    verifier = DatabaseSetupVerifier()
    results = verifier.run_full_verification()
    report_path = verifier.save_report()
    
    logger.info("\n" + "=" * 60)
    logger.info("KIMERA SWM DATABASE SETUP VERIFICATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Overall Status: {results['overall_status'].upper()}")
    logger.info(f"Report saved to: {report_path}")
    
    if results["recommendations"]:
        logger.info("\nRecommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            logger.info(f"  {i}. {rec}")
    
    # Exit with appropriate code
    if results["overall_status"] == "healthy":
        sys.exit(0)
    elif results["overall_status"] == "degraded":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main() 