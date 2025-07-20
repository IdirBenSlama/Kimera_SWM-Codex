"""
Run Configuration Migration on KIMERA Codebase
Phase 2, Week 6-7: Configuration Management Implementation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.config.config_migration import migrate_configuration

if __name__ == "__main__":
    print("Running KIMERA Configuration Migration...")
    print("=" * 60)
    
    # Run migration analysis
    migrate_configuration(project_root)
    
    print("\n" + "=" * 60)
    print("Migration analysis complete!")
    print("\nNext steps:")
    print("1. Review the generated config_migration_report.md")
    print("2. Add suggested environment variables to .env file")
    print("3. Update code to use configuration system instead of hardcoded values")
    print("4. Test thoroughly in development before deploying")