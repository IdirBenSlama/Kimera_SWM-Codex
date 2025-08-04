#!/usr/bin/env python3
"""
Duplicate Migration Script for Kimera SWM
Protocol Version: 3.0
Principle: No deletions, only archival with documentation
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import hashlib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DuplicateMigrator:
    """Migrate duplicate files following Protocol v3.0"""
    
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.archive_dir = Path(f"archive/{datetime.now():%Y-%m-%d}_duplicate_cleanup")
        self.migration_log = []
        
    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file content"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def prepare_migrations(self):
        """Prepare the list of migrations based on duplicate analysis"""
        
        # Define the backend duplicates to migrate
        migrations = [
            # Governance duplicates
            {
                "keep": "backend/governance/self_tuning.py",
                "archive": ["backend/layer_2_governance/governance/self_tuning.py"],
                "reason": "Consolidating to simpler path structure"
            },
            
            # Monitoring duplicates
            {
                "keep": "backend/monitoring/benchmarking_suite.py",
                "archive": ["backend/layer_2_governance/monitoring/benchmarking_suite.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/distributed_tracing.py",
                "archive": ["backend/layer_2_governance/monitoring/distributed_tracing.py.bak"],
                "reason": "Removing backup file duplicate"
            },
            {
                "keep": "backend/monitoring/enhanced_entropy_monitor.py",
                "archive": ["backend/layer_2_governance/monitoring/enhanced_entropy_monitor.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/kimera_dashboard.py",
                "archive": ["backend/layer_2_governance/monitoring/kimera_dashboard.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/kimera_monitoring_core.py",
                "archive": ["backend/layer_2_governance/monitoring/kimera_monitoring_core.py.bak"],
                "reason": "Removing backup file duplicate"
            },
            {
                "keep": "backend/monitoring/metrics_and_alerting.py",
                "archive": ["backend/layer_2_governance/monitoring/metrics_and_alerting.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/metrics_integration.py",
                "archive": ["backend/layer_2_governance/monitoring/metrics_integration.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/monitoring_integration.py",
                "archive": ["backend/layer_2_governance/monitoring/monitoring_integration.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/psychiatric_stability_monitor.py",
                "archive": ["backend/layer_2_governance/monitoring/psychiatric_stability_monitor.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/revolutionary_thermodynamic_monitor.py",
                "archive": ["backend/layer_2_governance/monitoring/revolutionary_thermodynamic_monitor.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/semantic_metrics.py",
                "archive": ["backend/layer_2_governance/monitoring/semantic_metrics.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/structured_logging.py",
                "archive": ["backend/layer_2_governance/monitoring/structured_logging.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/system_observer.py",
                "archive": ["backend/layer_2_governance/monitoring/system_observer.py"],
                "reason": "Consolidating monitoring modules"
            },
            {
                "keep": "backend/monitoring/telemetry.py",
                "archive": ["backend/layer_2_governance/monitoring/telemetry.py"],
                "reason": "Consolidating monitoring modules"
            },
            
            # Security duplicates
            {
                "keep": "backend/security/cognitive_firewall.py",
                "archive": ["backend/layer_2_governance/security/cognitive_firewall.py"],
                "reason": "Consolidating security modules"
            },
            {
                "keep": "backend/security/security_integration.py",
                "archive": ["backend/layer_2_governance/security/security_integration.py"],
                "reason": "Consolidating security modules"
            },
            {
                "keep": "backend/security/sql_injection_prevention.py",
                "archive": ["backend/layer_2_governance/security/sql_injection_prevention.py"],
                "reason": "Consolidating security modules"
            }
        ]
        
        return migrations
    
    def verify_migration(self, keep_path, archive_paths):
        """Verify that files are indeed duplicates before migration"""
        keep_file = Path(keep_path)
        
        if not keep_file.exists():
            logger.error(f"Keep file does not exist: {keep_path}")
            return False
        
        keep_hash = self.calculate_file_hash(keep_file)
        
        for archive_path in archive_paths:
            archive_file = Path(archive_path)
            if not archive_file.exists():
                logger.warning(f"Archive file does not exist: {archive_path}")
                continue
                
            archive_hash = self.calculate_file_hash(archive_file)
            if keep_hash != archive_hash:
                logger.error(f"Files are not identical: {keep_path} vs {archive_path}")
                return False
        
        return True
    
    def execute_migration(self, migration):
        """Execute a single migration"""
        keep_path = Path(migration['keep'])
        
        for archive_path in migration['archive']:
            archive_file = Path(archive_path)
            
            if not archive_file.exists():
                logger.warning(f"Skipping non-existent file: {archive_path}")
                continue
            
            # Create archive destination
            relative_path = archive_file.relative_to('.')
            dest_path = self.archive_dir / relative_path
            
            # Log the migration
            migration_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'archive',
                'source': str(archive_file),
                'destination': str(dest_path),
                'kept_file': str(keep_path),
                'reason': migration['reason'],
                'file_hash': self.calculate_file_hash(archive_file)
            }
            
            if self.dry_run:
                logger.info(f"[DRY RUN] Would archive: {archive_file} -> {dest_path}")
            else:
                # Create destination directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move the file
                shutil.move(str(archive_file), str(dest_path))
                logger.info(f"Archived: {archive_file} -> {dest_path}")
            
            self.migration_log.append(migration_entry)
    
    def create_archive_documentation(self):
        """Create ARCHIVED.md documentation"""
        doc_content = f"""# Duplicate Files Archive
**Date**: {datetime.now():%Y-%m-%d}  
**Protocol Version**: 3.0  
**Type**: Duplicate Cleanup  

## Summary

This archive contains duplicate files that were consolidated as part of the codebase cleanup process.
All files have been verified to be exact duplicates before archival.

## Migration Details

"""
        
        # Group by category
        monitoring_migrations = [m for m in self.migration_log if 'monitoring' in m['source']]
        security_migrations = [m for m in self.migration_log if 'security' in m['source']]
        governance_migrations = [m for m in self.migration_log if 'governance' in m['source']]
        
        if monitoring_migrations:
            doc_content += "### Monitoring Module Consolidation\n\n"
            for m in monitoring_migrations:
                doc_content += f"- **{Path(m['source']).name}**\n"
                doc_content += f"  - Kept: `{m['kept_file']}`\n"
                doc_content += f"  - Hash: `{m['file_hash'][:16]}...`\n"
                doc_content += f"  - Reason: {m['reason']}\n\n"
        
        if security_migrations:
            doc_content += "### Security Module Consolidation\n\n"
            for m in security_migrations:
                doc_content += f"- **{Path(m['source']).name}**\n"
                doc_content += f"  - Kept: `{m['kept_file']}`\n"
                doc_content += f"  - Hash: `{m['file_hash'][:16]}...`\n"
                doc_content += f"  - Reason: {m['reason']}\n\n"
        
        if governance_migrations:
            doc_content += "### Governance Module Consolidation\n\n"
            for m in governance_migrations:
                doc_content += f"- **{Path(m['source']).name}**\n"
                doc_content += f"  - Kept: `{m['kept_file']}`\n"
                doc_content += f"  - Hash: `{m['file_hash'][:16]}...`\n"
                doc_content += f"  - Reason: {m['reason']}\n\n"
        
        doc_content += """
## Verification

All archived files were verified to be exact duplicates using SHA256 hashing.
The original files remain in their primary locations under `backend/`.

## Rollback

To rollback this migration:
1. Copy files from this archive back to their original locations
2. Remove the kept files if desired (though this is not recommended)

---
*Generated by Kimera SWM Protocol v3.0 Migration System*
"""
        
        return doc_content
    
    def run(self):
        """Execute the migration process"""
        logger.info(f"Starting duplicate migration ({'DRY RUN' if self.dry_run else 'LIVE'})")
        
        # Prepare migrations
        migrations = self.prepare_migrations()
        logger.info(f"Prepared {len(migrations)} migrations")
        
        # Verify all migrations
        logger.info("Verifying migrations...")
        for migration in migrations:
            if not self.verify_migration(migration['keep'], migration['archive']):
                logger.error("Migration verification failed! Aborting.")
                return False
        
        logger.info("All migrations verified successfully")
        
        # Execute migrations
        for migration in migrations:
            self.execute_migration(migration)
        
        # Create documentation
        if not self.dry_run and self.migration_log:
            doc_path = self.archive_dir / "ARCHIVED.md"
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(self.create_archive_documentation())
            logger.info(f"Created archive documentation: {doc_path}")
        
        # Save migration log
        log_path = f"migration_log_{self.timestamp}.json"
        with open(log_path, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'dry_run': self.dry_run,
                'migrations': self.migration_log,
                'summary': {
                    'total_files_migrated': len(self.migration_log),
                    'archive_location': str(self.archive_dir)
                }
            }, f, indent=2)
        
        logger.info(f"Migration log saved to: {log_path}")
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"MIGRATION SUMMARY {'(DRY RUN)' if self.dry_run else ''}")
        logger.info(f"{'='*60}")
        logger.info(f"Total files to migrate: {len(self.migration_log)}")
        logger.info(f"Archive location: {self.archive_dir}")
        logger.info(f"Migration log: {log_path}")
        
        if self.dry_run:
            logger.info("\n⚠️  This was a DRY RUN. No files were actually moved.")
            logger.info("Run with --execute to perform the actual migration.")
        
        return True


def main():
    """Main execution"""
    # Check for --execute flag
    dry_run = "--execute" not in sys.argv
    
    migrator = DuplicateMigrator(dry_run=dry_run)
    success = migrator.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main()) 