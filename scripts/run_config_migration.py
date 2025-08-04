"""
Run Configuration Migration on KIMERA Codebase
Phase 2, Week 6-7: Configuration Management Implementation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.config_migration import migrate_configuration
import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Running KIMERA Configuration Migration...")
    logger.info("=" * 60)
    
    # Run migration analysis
    migrate_configuration(project_root)
    
    logger.info("\n" + "=" * 60)
    logger.info("Migration analysis complete!")
    logger.info("\nNext steps:")
    logger.info("1. Review the generated config_migration_report.md")
    logger.info("2. Add suggested environment variables to .env file")
    logger.info("3. Update code to use configuration system instead of hardcoded values")
    logger.info("4. Test thoroughly in development before deploying")