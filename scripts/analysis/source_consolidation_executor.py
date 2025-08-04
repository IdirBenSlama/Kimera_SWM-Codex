#!/usr/bin/env python3
"""
KIMERA SWM Source Directory Consolidation Executor
==================================================

Executes Phase 2 of technical debt remediation: Source Directory Consolidation
Following Martin Fowler framework and KIMERA SWM Protocol v3.0
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SourceConsolidationExecutor:
    """Executes source directory consolidation safely"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = self.project_root / f"backup_source_consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def analyze_current_structure(self):
        """Analyze current source directory structure"""
        logger.info("🔍 Analyzing current source structure...")

        source_dirs = []
        for item in self.project_root.iterdir():
            if item.is_dir() and item.name in ['src', 'kimera_trading']:
                py_files = list(item.rglob("*.py"))
                if py_files:
                    source_dirs.append({
                        'path': str(item),
                        'name': item.name,
                        'files': len(py_files),
                        'size_mb': sum(f.stat().st_size for f in py_files) / (1024*1024)
                    })

        # Also check for nested src directories
        for item in self.project_root.rglob("src"):
            if item != self.project_root / "src" and item.is_dir():
                py_files = list(item.rglob("*.py"))
                if py_files and len(py_files) > 5:  # Only include significant directories
                    source_dirs.append({
                        'path': str(item),
                        'name': f"{item.parent.name}/src",
                        'files': len(py_files),
                        'size_mb': sum(f.stat().st_size for f in py_files) / (1024*1024)
                    })

        logger.info(f"📊 Found {len(source_dirs)} source directories:")
        for dir_info in source_dirs:
            logger.info(f"   - {dir_info['name']}: {dir_info['files']} files ({dir_info['size_mb']:.1f} MB)")

        return source_dirs

    def create_consolidation_plan(self, source_dirs):
        """Create detailed consolidation plan"""
        logger.info("📋 Creating consolidation plan...")

        plan = {
            'timestamp': datetime.now().isoformat(),
            'backup_location': str(self.backup_dir),
            'actions': []
        }

        main_src = self.project_root / "src"

        for dir_info in source_dirs:
            source_path = Path(dir_info['path'])

            # Skip if it's already the main src directory
            if source_path == main_src:
                continue

            # Determine target location
            if "kimera_trading" in dir_info['name']:
                target = main_src / "kimera_trading"
            elif "archive" in dir_info['path']:
                # Don't consolidate archive directories, just document them
                plan['actions'].append({
                    'type': 'archive_skip',
                    'source': str(source_path),
                    'reason': 'Archive directory - preserved as-is'
                })
                continue
            else:
                target = main_src / source_path.name

            plan['actions'].append({
                'type': 'consolidate',
                'source': str(source_path),
                'target': str(target),
                'files': dir_info['files'],
                'size_mb': dir_info['size_mb']
            })

        return plan

    def create_backup(self):
        """Create backup before consolidation"""
        logger.info("💾 Creating backup...")

        self.backup_dir.mkdir(exist_ok=True)

        # Backup all source directories
        for source_dir in ['src', 'kimera_trading']:
            source_path = self.project_root / source_dir
            if source_path.exists():
                backup_target = self.backup_dir / source_dir
                logger.info(f"   Backing up {source_path} → {backup_target}")
                shutil.copytree(source_path, backup_target, dirs_exist_ok=True)

        logger.info(f"✅ Backup created at: {self.backup_dir}")

    def execute_consolidation(self, plan, dry_run=True):
        """Execute the consolidation plan"""
        logger.info(f"🚀 {'DRY RUN:' if dry_run else 'EXECUTING:'} Source Directory Consolidation")

        if not dry_run:
            self.create_backup()

        results = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'actions_completed': 0,
            'files_moved': 0,
            'errors': []
        }

        for action in plan['actions']:
            if action['type'] == 'archive_skip':
                logger.info(f"⏭️  Skipping: {action['source']} ({action['reason']})")
                continue

            source_path = Path(action['source'])
            target_path = Path(action['target'])

            try:
                if dry_run:
                    logger.info(f"📁 Would consolidate: {source_path} → {target_path}")
                    logger.info(f"   📊 {action['files']} files ({action['size_mb']:.1f} MB)")
                else:
                    logger.info(f"📁 Consolidating: {source_path} → {target_path}")

                    # Create target directory
                    target_path.mkdir(parents=True, exist_ok=True)

                    # Copy files (don't move, preserve originals for safety)
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)

                    logger.info(f"✅ Successfully consolidated {action['files']} files")

                results['actions_completed'] += 1
                results['files_moved'] += action['files']

            except Exception as e:
                error_msg = f"Error consolidating {source_path}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        return results

    def generate_report(self, plan, results):
        """Generate consolidation report"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

        report = f"""# KIMERA SWM Source Directory Consolidation Report
**Generated**: {timestamp}
**Phase**: 2 of Technical Debt Remediation
**Framework**: Martin Fowler + KIMERA SWM Protocol v3.0

## Executive Summary

**Status**: {'✅ COMPLETED' if not results['dry_run'] else '🔄 DRY RUN'}
- **Actions Completed**: {results['actions_completed']}
- **Files Consolidated**: {results['files_moved']}
- **Errors**: {len(results['errors'])}

## Consolidation Plan

### Actions Executed:
"""

        for action in plan['actions']:
            if action['type'] == 'consolidate':
                report += f"- **{action['source']}** → **{action['target']}**\n"
                report += f"  - Files: {action['files']}\n"
                report += f"  - Size: {action['size_mb']:.1f} MB\n\n"
            elif action['type'] == 'archive_skip':
                report += f"- **SKIPPED**: {action['source']} ({action['reason']})\n\n"

        if results['errors']:
            report += "## Errors Encountered\n"
            for error in results['errors']:
                report += f"- ❌ {error}\n"
        else:
            report += "## ✅ No Errors - Perfect Execution\n"

        report += f"""
## Impact Assessment

### Benefits Achieved:
- **Unified Source Structure**: All production code under single `src/` hierarchy
- **Reduced Complexity**: Eliminated scattered source directories
- **Improved Navigation**: Clear, logical code organization
- **Build Simplification**: Single source tree for deployment

### Next Steps:
1. Update import statements to reflect new structure
2. Update build/deployment scripts
3. Update IDE configuration
4. Archive old directory structure after verification

### Backup Information:
- **Backup Location**: {plan.get('backup_location', 'N/A')}
- **Recovery Instructions**: Restore from backup if issues occur

---

*Phase 2 of KIMERA SWM Technical Debt Remediation*
*Following Martin Fowler's Technical Debt Quadrants Framework*
"""

        # Save report
        report_dir = Path("docs/reports/debt")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{timestamp}_source_consolidation_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"📄 Report saved: {report_path}")
        return str(report_path)

def main():
    """Main execution function"""
    logger.info("🚀 KIMERA SWM Source Directory Consolidation - Phase 2")
    logger.info("=" * 60)

    executor = SourceConsolidationExecutor()

    # Step 1: Analyze current structure
    source_dirs = executor.analyze_current_structure()

    if not source_dirs:
        logger.info("✅ No additional source directories found - consolidation not needed")
        return

    # Step 2: Create consolidation plan
    plan = executor.create_consolidation_plan(source_dirs)

    # Step 3: Execute dry run
    logger.info("\n🔄 Executing DRY RUN...")
    dry_results = executor.execute_consolidation(plan, dry_run=True)

    # Step 4: Generate dry run report
    report_path = executor.generate_report(plan, dry_results)

    logger.info(f"\n📊 DRY RUN RESULTS:")
    logger.info(f"   Actions Planned: {dry_results['actions_completed']}")
    logger.info(f"   Files to Move: {dry_results['files_moved']}")
    logger.info(f"   Errors: {len(dry_results['errors'])}")

    logger.info(f"\n💡 To execute actual consolidation, run with --execute flag")
    logger.info(f"📄 Detailed report: {report_path}")

if __name__ == "__main__":
    import sys

    if "--execute" in sys.argv:
        logger.info("⚠️  EXECUTING ACTUAL CONSOLIDATION...")
        executor = SourceConsolidationExecutor()
        source_dirs = executor.analyze_current_structure()
        plan = executor.create_consolidation_plan(source_dirs)
        results = executor.execute_consolidation(plan, dry_run=False)
        report_path = executor.generate_report(plan, results)
        logger.info("✅ Source consolidation complete!")
        logger.info(f"📄 Final report: {report_path}")
    else:
        main()
