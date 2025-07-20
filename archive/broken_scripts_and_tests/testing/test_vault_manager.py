#!/usr/bin/env python3
"""Test vault manager functionality directly"""

import os
import sys
from dotenv import load_dotenv

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vault_manager():
    """Test vault manager methods"""
    logger.info("Testing UnderstandingVaultManager...")
    
    try:
        from backend.vault.understanding_vault_manager import UnderstandingVaultManager
        
        # Create instance
        vault_manager = UnderstandingVaultManager()
        logger.info("✅ Created UnderstandingVaultManager instance")
        
        # Test get_all_geoids
        try:
            geoids = vault_manager.get_all_geoids()
            logger.info(f"✅ get_all_geoids()
        except Exception as e:
            logger.error(f"❌ get_all_geoids()
        
        # Test get_total_scar_count
        try:
            count_a = vault_manager.get_total_scar_count("vault_a")
            count_b = vault_manager.get_total_scar_count("vault_b")
            logger.info(f"✅ get_total_scar_count()
        except Exception as e:
            logger.error(f"❌ get_total_scar_count()
        
        # Test get_understanding_metrics
        try:
            metrics = vault_manager.get_understanding_metrics()
            logger.info(f"✅ get_understanding_metrics()
            logger.info(f"   Components: {metrics['understanding_components']}")
        except Exception as e:
            logger.error(f"❌ get_understanding_metrics()
            
    except Exception as e:
        logger.error(f"❌ Failed to import/create UnderstandingVaultManager: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vault_manager()