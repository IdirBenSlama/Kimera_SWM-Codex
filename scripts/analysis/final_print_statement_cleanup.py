#!/usr/bin/env python3
"""Final Print Statement Cleanup - Foundation Completion
===================================================

Completes the zero-debugging protocol by eliminating any remaining
print statements in active source code (excluding archives).

Purpose: Achieve 100% zero-debugging compliance for foundation completion
Strategy: Target only active source, skip archived and legitimate uses
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class FinalPrintCleanup:
    """Final cleanup of print statements for foundation completion"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.active_source_dirs = ["src", "scripts", "tests"]
        self.skip_patterns = [
            "archive",
            "backup_",
            ".venv",
            "__pycache__",
            ".git",
            ".mypy_cache",
            ".pytest_cache",
            "node_modules",
        ]

    def run_final_cleanup(self) -> dict:
        """Execute final print statement cleanup"""
        logger.info("🧹 Starting final print statement cleanup...")
        logger.info("🎯 Goal: Complete zero-debugging protocol")
        logger.info("=" * 60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "files_scanned": 0,
            "violations_found": 0,
            "violations_fixed": 0,
            "files_modified": 0,
            "legitimate_uses": 0,
            "details": [],
        }

        # Scan active source directories only
        for source_dir in self.active_source_dirs:
            dir_path = self.project_root / source_dir
            if dir_path.exists():
                logger.info(f"🔍 Scanning {source_dir}/ directory...")
                self._process_directory(dir_path, results)

        # Generate summary
        self._generate_summary(results)

        return results

    def _process_directory(self, directory: Path, results: dict) -> None:
        """Process a directory for print statement cleanup"""
        for py_file in directory.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            results["files_scanned"] += 1

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                violations = self._find_print_violations(py_file, content)
                if violations:
                    results["violations_found"] += len(violations)

                    # Fix violations
                    fixed_content, fixes_made = self._fix_print_statements(
                        content, violations,
                    )

                    if fixes_made > 0:
                        # Write back the fixed content
                        with open(py_file, "w", encoding="utf-8") as f:
                            f.write(fixed_content)

                        results["violations_fixed"] += fixes_made
                        results["files_modified"] += 1

                        relative_path = py_file.relative_to(self.project_root)
                        results["details"].append(
                            {
                                "file": str(relative_path),
                                "violations_fixed": fixes_made,
                                "action": "Fixed print statements with logging",
                            },
                        )

                        logger.info(
                            f"✅ Fixed {fixes_made} violations in {relative_path}",
                        )

            except Exception as e:
                logger.warning(f"⚠️ Could not process {py_file}: {e}")

    def _find_print_violations(
        self, file_path: Path, content: str,
    ) -> List[Tuple[int, str]]:
        """Find print statement violations in content"""
        violations = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip comments
            if stripped.startswith("#"):
                continue

            # Skip if already has logger
            if "logger" in line:
                continue

            # Skip if inside docstrings or triple quotes
            if '"""' in line or "'''" in line:
                continue

            # Look for print statements
            if self._is_print_violation(stripped):
                violations.append((i, line))

        return violations

    def _is_print_violation(self, line: str) -> bool:
        """Check if line contains a print statement violation"""
        # Patterns that indicate a print statement
        print_patterns = [
            r"^\s*print\s*\(",  # print at start of line
            r"[^a-zA-Z_]print\s*\(",  # print not preceded by identifier
        ]

        for pattern in print_patterns:
            if re.search(pattern, line):
                # Additional checks for legitimate uses
                if any(
                    legitimate in line.lower()
                    for legitimate in ["pprint", "blueprint", "footprint", "imprint"]
                ):
                    return False

                # Skip if it's in a string
                if line.count('"') >= 2 or line.count("'") >= 2:
                    continue

                return True

        return False

    def _fix_print_statements(
        self, content: str, violations: List[Tuple[int, str]],
    ) -> Tuple[str, int]:
        """Fix print statements by replacing with logger"""
        lines = content.split("\n")
        fixes_made = 0

        # Add logger import if not present
        has_logging_import = "import logging" in content
        has_logger_definition = "logger = logging.getLogger" in content

        if not has_logging_import:
            # Find a good place to add the import
            import_index = self._find_import_insertion_point(lines)
            lines.insert(import_index, "import logging")

            # Adjust line numbers for violations
            violations = [
                (line_num + 1 if line_num > import_index else line_num, line)
                for line_num, line in violations
            ]

        if not has_logger_definition:
            # Add logger definition after imports
            logger_index = self._find_logger_insertion_point(lines)
            lines.insert(logger_index, "logger = logging.getLogger(__name__)")

            # Adjust line numbers for violations
            violations = [
                (line_num + 1 if line_num > logger_index else line_num, line)
                for line_num, line in violations
            ]

        # Fix each violation
        for line_num, original_line in violations:
            # Adjust for 0-based indexing
            line_index = line_num - 1

            if 0 <= line_index < len(lines):
                old_line = lines[line_index]
                new_line = self._convert_print_to_logger(old_line)

                if new_line != old_line:
                    lines[line_index] = new_line
                    fixes_made += 1

        return "\n".join(lines), fixes_made

    def _find_import_insertion_point(self, lines: List[str]) -> int:
        """Find the best place to insert logging import"""
        # Look for existing imports
        for i, line in enumerate(lines):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                continue
            return max(0, i)

        # If no imports found, insert at the beginning (after docstring if any)
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith("#") and '"""' not in line:
                return i

        return 0

    def _find_logger_insertion_point(self, lines: List[str]) -> int:
        """Find the best place to insert logger definition"""
        # Insert after all imports
        import_end = 0
        for i, line in enumerate(lines):
            if (
                line.strip().startswith("import ")
                or line.strip().startswith("from ")
                or line.strip() == ""
            ):
                import_end = i + 1
            else:
                break

        return import_end

    def _convert_print_to_logger(self, line: str) -> str:
        """Convert a print statement to logger.info"""
        # Extract indentation
        indent = len(line) - len(line.lstrip())
        indentation = line[:indent]

        # Replace print with logger.info
        # Handle various print statement formats
        patterns = [
            (r"print\s*\((.*)\)", r"logger.info(\1)"),
        ]

        converted = line
        for pattern, replacement in patterns:
            converted = re.sub(pattern, replacement, converted)

        return converted

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        path_str = str(file_path).lower()

        # Skip files in archive or backup directories
        for pattern in self.skip_patterns:
            if pattern in path_str:
                return True

        # Skip files that are legitimate to have print statements
        filename = file_path.name
        if any(
            skip_name in filename for skip_name in ["test_", "__init__.py", "setup.py"]
        ):
            # Allow some print statements in test files and setup
            return True

        return False

    def _generate_summary(self, results: dict) -> None:
        """Generate cleanup summary"""
        logger.info("\n🎉 FINAL PRINT CLEANUP COMPLETE!")
        logger.info("=" * 50)
        logger.info("📊 CLEANUP RESULTS:")
        logger.info(f"   Files Scanned: {results['files_scanned']}")
        logger.info(f"   Violations Found: {results['violations_found']}")
        logger.info(f"   Violations Fixed: {results['violations_fixed']}")
        logger.info(f"   Files Modified: {results['files_modified']}")

        if results["violations_fixed"] > 0:
            logger.info("\n✅ FILES IMPROVED:")
            for detail in results["details"]:
                logger.info(
                    f"   📄 {detail['file']}: {detail['violations_fixed']} fixes",
                )

        if results["violations_found"] == results["violations_fixed"]:
            logger.info("\n🏆 ZERO-DEBUGGING PROTOCOL: FULLY IMPLEMENTED!")
        else:
            remaining = results["violations_found"] - results["violations_fixed"]
            logger.info(f"\n⚠️ {remaining} violations may need manual review")


def main():
    """Execute final print cleanup"""
    logger.info("🧹 KIMERA SWM Final Print Statement Cleanup")
    logger.info("🎯 Goal: Complete zero-debugging protocol")
    logger.info("=" * 50)

    cleanup = FinalPrintCleanup()
    results = cleanup.run_final_cleanup()

    return results


if __name__ == "__main__":
    results = main()
