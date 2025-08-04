#!/usr/bin/env python3
"""
KIMERA SWM Technical Debt Action Plan Generator
==============================================

Generates actionable remediation steps based on technical debt analysis.
Follows the Kimera SWM Autonomous Architect Protocol v3.0.

Usage:
    python scripts/utils/debt_action_plan.py
    python scripts/utils/debt_action_plan.py --execute --phase 1
    python scripts/utils/debt_action_plan.py --dry-run
"""

import os
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import logging
logger = logging.getLogger(__name__)

class KimeraDebtActionPlan:
    """
    Generates and executes technical debt remediation action plans.

    Follows aerospace-grade safety principles:
    - Defense in depth (backup before changes)
    - Positive confirmation (verify each step)
    - Conservative decision making (dry-run first)
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "archive" / f"debt_remediation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.actions_performed = []

    def generate_phase1_plan(self) -> List[Dict]:
        """
        Phase 1: Critical Structural Debt Remediation
        Priority: P0 - Foundation must be solid before other improvements
        """

        actions = [
            {
                "id": "P1.1",
                "title": "Create Backup Archive",
                "description": "Create safety backup before any structural changes",
                "type": "safety",
                "commands": [
                    f"mkdir -p {self.backup_dir}",
                    f"cp -r . {self.backup_dir}/pre_remediation_backup"
                ],
                "validation": lambda: self.backup_dir.exists(),
                "risk": "LOW"
            },
            {
                "id": "P1.2",
                "title": "Consolidate Misplaced Src Directories",
                "description": "Move srccore* directories to proper locations under src/",
                "type": "structural",
                "commands": self._generate_src_consolidation_commands(),
                "validation": lambda: self._validate_src_structure(),
                "risk": "MEDIUM"
            },
            {
                "id": "P1.3",
                "title": "Move Reports to Docs Structure",
                "description": "Organize scattered reports into proper docs/reports/ structure",
                "type": "organizational",
                "commands": self._generate_report_organization_commands(),
                "validation": lambda: self._validate_docs_structure(),
                "risk": "LOW"
            },
            {
                "id": "P1.4",
                "title": "Consolidate Configuration Files",
                "description": "Move config files to single configs/ directory",
                "type": "configuration",
                "commands": self._generate_config_consolidation_commands(),
                "validation": lambda: self._validate_config_structure(),
                "risk": "MEDIUM"
            },
            {
                "id": "P1.5",
                "title": "Clean Root Directory",
                "description": "Move temporary and build files from root to appropriate locations",
                "type": "cleanup",
                "commands": self._generate_root_cleanup_commands(),
                "validation": lambda: self._validate_root_cleanliness(),
                "risk": "LOW"
            }
        ]

        return actions

    def _generate_src_consolidation_commands(self) -> List[str]:
        """Generate commands to consolidate misplaced src directories."""
        commands = []

        # Find misplaced src directories
        misplaced_dirs = [d for d in self.project_root.iterdir()
                         if d.is_dir() and d.name.startswith('srccore')]

        for misplaced_dir in misplaced_dirs:
            # Extract the actual module name from srccore*
            if misplaced_dir.name.startswith('srccorehigh_dimensional_modeling'):
                target = "src/core/high_dimensional_modeling"
            elif misplaced_dir.name.startswith('srccoregpu_management'):
                target = "src/core/gpu_management"
            elif misplaced_dir.name.startswith('srccoregeometric_optimization'):
                target = "src/core/geometric_optimization"
            else:
                continue

            commands.extend([
                f"mkdir -p {target}",
                f"cp -r {misplaced_dir}/* {target}/",
                f"# Archive original: mv {misplaced_dir} archive/structural_cleanup_{datetime.now().strftime('%Y%m%d')}/",
            ])

        return commands

    def _generate_report_organization_commands(self) -> List[str]:
        """Generate commands to organize reports properly."""
        commands = [
            "mkdir -p docs/reports/status",
            "mkdir -p docs/reports/roadmaps",
            "mkdir -p docs/reports/completion",
            "mkdir -p docs/reports/health",
        ]

        # Move various report files
        report_mappings = {
            "*ROADMAP*.md": "docs/reports/roadmaps/",
            "*STATUS*.md": "docs/reports/status/",
            "*COMPLETION*.md": "docs/reports/completion/",
            "*REPORT*.md": "docs/reports/health/",
            "audit_report_*.json": "docs/reports/health/",
            "test_report_*.json": "docs/reports/health/"
        }

        for pattern, target in report_mappings.items():
            commands.append(f"find . -maxdepth 1 -name '{pattern}' -exec mv {{}} {target} \\;")

        return commands

    def _generate_config_consolidation_commands(self) -> List[str]:
        """Generate commands to consolidate configuration files."""
        return [
            "mkdir -p configs/environments",
            "mkdir -p configs/tools",

            # Move scattered config files
            "find . -maxdepth 1 -name '*.toml' -not -name 'pyproject.toml' -exec mv {} configs/tools/ \\;",
            "find . -maxdepth 1 -name '.env*' -exec mv {} configs/environments/ \\;",

            # Consolidate multiple config directories
            "if [ -d config ]; then cp -r config/* configs/ && mv config archive/; fi",
            "if [ -d configs_consolidated ]; then cp -r configs_consolidated/* configs/ && mv configs_consolidated archive/; fi",
        ]

    def _generate_root_cleanup_commands(self) -> List[str]:
        """Generate commands to clean up root directory."""
        return [
            "mkdir -p tmp/build_artifacts",
            "mkdir -p archive/temporary_files",

            # Move temporary files
            "find . -maxdepth 1 -name '*.py' -not -name 'kimera.py' -not -name 'run_tests.py' -exec mv {} scripts/utils/ \\;",
            "find . -maxdepth 1 -name 'nul' -exec mv {} tmp/ \\;",
            "find . -maxdepth 1 -name '*.log' -exec mv {} logs/ \\;",
        ]

    def _validate_src_structure(self) -> bool:
        """Validate that src structure is properly organized."""
        required_dirs = ["src/core", "src/engines", "src/utils"]
        return all((self.project_root / d).exists() for d in required_dirs)

    def _validate_docs_structure(self) -> bool:
        """Validate that docs structure is properly organized."""
        required_dirs = ["docs/reports/status", "docs/reports/roadmaps", "docs/reports/completion"]
        return all((self.project_root / d).exists() for d in required_dirs)

    def _validate_config_structure(self) -> bool:
        """Validate that configuration is consolidated."""
        return (self.project_root / "configs").exists()

    def _validate_root_cleanliness(self) -> bool:
        """Validate that root directory is clean."""
        # Count non-essential files in root
        root_files = [f for f in self.project_root.iterdir()
                     if f.is_file() and not f.name.startswith('.')
                     and f.name not in ['README.md', 'LICENSE', 'CHANGELOG.md', 'pyproject.toml', 'requirements.txt']]
        return len(root_files) <= 3  # Allow a few essential files

    def execute_action(self, action: Dict, dry_run: bool = True) -> bool:
        """
        Execute a single action with aerospace-grade safety.

        Args:
            action: Action dictionary with commands and validation
            dry_run: If True, only print what would be done

        Returns:
            bool: True if action succeeded, False otherwise
        """

        logger.info(f"\n🎯 Executing Action {action['id']}: {action['title']}")
        logger.info(f"📋 Description: {action['description']}")
        logger.info(f"⚠️  Risk Level: {action['risk']}")

        if dry_run:
            logger.info("🔍 DRY RUN - Commands that would be executed:")
            for cmd in action['commands']:
                logger.info(f"   $ {cmd}")
            return True

        # Create backup if not safety action
        if action['type'] != 'safety':
            self._create_incremental_backup(action['id'])

        # Execute commands
        try:
            for cmd in action['commands']:
                if cmd.startswith('#'):  # Skip comments
                    continue

                logger.info(f"   🚀 {cmd}")
                result = os.system(cmd)
                if result != 0:
                    logger.info(f"   ❌ Command failed with exit code {result}")
                    return False

            # Validate action completion
            if action['validation']():
                logger.info(f"   ✅ Action {action['id']} completed successfully")
                self.actions_performed.append(action['id'])
                return True
            else:
                logger.info(f"   ❌ Action {action['id']} validation failed")
                return False

        except Exception as e:
            logger.info(f"   ❌ Action {action['id']} failed: {e}")
            return False

    def _create_incremental_backup(self, action_id: str):
        """Create incremental backup before risky operations."""
        backup_path = self.backup_dir / f"before_{action_id}"
        backup_path.mkdir(parents=True, exist_ok=True)

        # Backup critical files
        critical_paths = ['src', 'docs', 'configs', 'scripts']
        for path in critical_paths:
            if (self.project_root / path).exists():
                shutil.copytree(self.project_root / path, backup_path / path, dirs_exist_ok=True)

    def execute_phase(self, phase: int, dry_run: bool = True) -> bool:
        """Execute a complete phase of debt remediation."""

        if phase == 1:
            actions = self.generate_phase1_plan()
        else:
            logger.info(f"❌ Phase {phase} not yet implemented")
            return False

        logger.info(f"\n🚀 KIMERA SWM DEBT REMEDIATION - PHASE {phase}")
        logger.info("=" * 60)
        logger.info(f"📋 Total Actions: {len(actions)}")
        logger.info(f"🔍 Mode: {'DRY RUN' if dry_run else 'LIVE EXECUTION'}")

        if not dry_run:
            response = input(f"\n⚠️  This will modify the codebase. Continue? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("🛑 Execution cancelled by user")
                return False

        # Execute actions in sequence
        success_count = 0
        for action in actions:
            if self.execute_action(action, dry_run):
                success_count += 1
            else:
                logger.info(f"\n🛑 Phase {phase} halted due to action failure")
                break

        logger.info(f"\n📊 PHASE {phase} SUMMARY:")
        logger.info(f"   ✅ Successful Actions: {success_count}/{len(actions)}")
        logger.info(f"   📋 Actions Performed: {', '.join(self.actions_performed)}")

        if success_count == len(actions):
            logger.info(f"   🎉 Phase {phase} completed successfully!")
            return True
        else:
            logger.info(f"   ⚠️  Phase {phase} completed with issues")
            return False

def main():
    parser = argparse.ArgumentParser(description="Kimera SWM Technical Debt Action Plan")
    parser.add_argument("--execute", action="store_true", help="Execute the action plan")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Show what would be done (default)")
    parser.add_argument("--phase", type=int, default=1, help="Phase to execute (1-4)")
    parser.add_argument("--project-root", default=".", help="Project root directory")

    args = parser.parse_args()

    # Handle argument logic
    if args.execute:
        dry_run = False
    else:
        dry_run = True

    action_plan = KimeraDebtActionPlan(args.project_root)

    if args.phase:
        action_plan.execute_phase(args.phase, dry_run)
    else:
        # Show available phases
        logger.info("🎯 KIMERA SWM TECHNICAL DEBT REMEDIATION PHASES")
        logger.info("=" * 50)
        logger.info("Phase 1: Critical Structural Debt (Directory organization)")
        logger.info("Phase 2: Import System Standardization (Coming soon)")
        logger.info("Phase 3: Test Infrastructure Consolidation (Coming soon)")
        logger.info("Phase 4: Dependency Optimization (Coming soon)")
        logger.info("\nUse --phase N to execute a specific phase")

if __name__ == "__main__":
    main()
