#!/usr/bin/env python3
"""
KIMERA MEMORY LEAK DETECTION VALIDATION
======================================

Simple validation script for the advanced memory leak detection system.
"""

import time
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_leak_detection():
    """Test the memory leak detection system"""
    logger.info("🧪 Testing Kimera Memory Leak Detection System")
    
    try:
        # Test 1: Check if files exist
        backend_dir = Path("backend/analysis")
        guardian_file = backend_dir / "kimera_memory_leak_guardian.py"
        
        if guardian_file.exists():
            logger.info("✅ Memory leak guardian file exists")
            
            # Test 2: Try to import
            try:
                sys.path.append(str(Path("backend").absolute()))
                from analysis.kimera_memory_leak_guardian import get_memory_leak_guardian
                
                guardian = get_memory_leak_guardian()
                logger.info("✅ Memory leak guardian imported successfully")
                
                # Test 3: Basic functionality
                if hasattr(guardian, 'start_monitoring'):
                    guardian.start_monitoring()
                    time.sleep(1)
                    guardian.stop_monitoring()
                    logger.info("✅ Monitoring functionality works")
                
                # Test 4: Generate report
                report = guardian.generate_comprehensive_report()
                if 'analysis_summary' in report:
                    logger.info("✅ Report generation works")
                    logger.info(f"   Functions analyzed: {report['analysis_summary']['total_functions_analyzed']}")
                    logger.info(f"   Active allocations: {report['analysis_summary']['active_allocations']}")
                
                return True
                
            except ImportError as e:
                logger.warning(f"⚠️ Could not import memory leak guardian: {e}")
                return False
        else:
            logger.error("❌ Memory leak guardian file not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting Memory Leak Detection Validation")
    
    success = test_memory_leak_detection()
    
    if success:
        logger.info("🎉 Memory Leak Detection System: VALIDATION SUCCESSFUL")
        return 0
    else:
        logger.error("❌ Memory Leak Detection System: VALIDATION FAILED")
        return 1

if __name__ == "__main__":
    exit(main()) 