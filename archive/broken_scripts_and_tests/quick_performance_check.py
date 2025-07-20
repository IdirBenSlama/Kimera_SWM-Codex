#!/usr/bin/env python3
"""
Quick Performance Check for Kimera SWM
Verifies all systems are operational with correct configuration
"""

import os
import sys
import time
import requests
from sqlalchemy import create_engine, text

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://kimera:kimera_secure_pass_2025@localhost:5432/kimera_swm")
API_BASE_URL = "http://localhost:8001"  # Correct port

def quick_check():
    logger.info("🚀 Kimera SWM Quick Performance Check")
    logger.info("=" * 40)
    
    issues = []
    
    # 1. Database Check
    logger.info("\n📊 Database Performance:")
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # JSONB Query
        start = time.time()
        conn.execute(text("SELECT COUNT(*) FROM geoids WHERE symbolic_state ? 'symbols'")).scalar()
        jsonb_time = time.time() - start
        logger.info(f"   JSONB query: {jsonb_time*1000:.1f}ms", "✅" if jsonb_time < 0.01 else "⚠️")
        
        # Vector Query
        start = time.time()
        conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT s1.scar_id FROM scars s1, scars s2
                WHERE s1.scar_vector IS NOT NULL AND s2.scar_vector IS NOT NULL
                AND s1.scar_id < s2.scar_id
                ORDER BY s1.scar_vector <=> s2.scar_vector
                LIMIT 5
            ) t
        """)).scalar()
        vector_time = time.time() - start
        logger.info(f"   Vector query: {vector_time:.2f}s", "✅" if vector_time < 2 else "⚠️")
        if vector_time >= 2:
            issues.append("Vector queries are slower than expected")
    
    # 2. API Check
    logger.info("\n🌐 API Performance:")
    api_ok = True
    for endpoint in ['/system/health', '/system/status', '/monitoring/status']:
        try:
            start = time.time()
            resp = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
            api_time = time.time() - start
            logger.info(f"   {endpoint}: {resp.status_code} ({api_time*1000:.0f}ms)")
            if resp.status_code != 200:
                api_ok = False
                issues.append(f"API endpoint {endpoint} returned {resp.status_code}")
        except Exception as e:
            logger.error(f"   {endpoint}: ERROR - {str(e)}")
            api_ok = False
            issues.append(f"API endpoint {endpoint} failed")
    
    # 3. Overall Assessment
    logger.info("\n📈 Overall Assessment:")
    
    if not issues:
        logger.info("   🎉 EXCELLENT - All systems operational!")
        logger.info("   ✅ Database queries optimized")
        logger.info("   ✅ API endpoints responding")
        logger.info("   ✅ Vector search functional")
        return 0
    else:
        logger.warning("   ⚠️  ISSUES DETECTED:")
        for issue in issues:
            logger.info(f"      - {issue}")
        return 1

if __name__ == "__main__":
    exit(quick_check())