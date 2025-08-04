"""
Configuration Migration Tool for KIMERA System
Helps migrate from hardcoded values to configuration-based approach
Phase 2, Week 6-7: Configuration Management Implementation
"""

import ast
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HardcodedValue:
    """Represents a hardcoded value found in code"""

    file_path: Path
    line_number: int
    value: str
    value_type: str
    context: str
    suggested_config_key: str


class ConfigurationMigrator:
    """
    Scans codebase for hardcoded values and suggests configuration replacements
    """

    # Patterns to identify hardcoded values
    PATTERNS = {
        "api_key": re.compile(r'["\']([a-zA-Z0-9]{32,})["\']'),
        "url": re.compile(r'["\']https?://[^"\']+["\']'),
        "path": re.compile(r'["\'](/[^"\']+|[A-Z]:\\[^"\']+|\.\.?/[^"\']+)["\']'),
        "port": re.compile(r"port\s*=\s*(\d{2,5})"),
        "host": re.compile(r'host\s*=\s*["\']([^"\']+)["\']'),
        "database": re.compile(r'["\'](?:sqlite|postgresql|mysql)://[^"\']+["\']'),
        "timeout": re.compile(r"timeout\s*=\s*(\d+(?:\.\d+)?)"),
        "max_workers": re.compile(r"max_workers\s*=\s*(\d+)"),
        "batch_size": re.compile(r"batch_size\s*=\s*(\d+)"),
    }

    # Files/directories to skip
    SKIP_PATTERNS = {
        "__pycache__",
        ".git",
        ".env",
        "venv",
        "node_modules",
        "*.pyc",
        "*.log",
        "tests",
        "test_*.py",
        "*_test.py",
    }

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.hardcoded_values: List[HardcodedValue] = []
        self.processed_files: Set[Path] = set()

    def scan(self, directories: Optional[List[str]] = None) -> List[HardcodedValue]:
        """
        Scan directories for hardcoded values

        Args:
            directories: Specific directories to scan (default: all)

        Returns:
            List of found hardcoded values
        """
        scan_dirs = directories or ["backend", "kimera.py"]

        for dir_name in scan_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.is_file():
                self._scan_file(dir_path)
            elif dir_path.is_dir():
                self._scan_directory(dir_path)

        logger.info(f"Scanned {len(self.processed_files)} files")
        logger.info(f"Found {len(self.hardcoded_values)} potential hardcoded values")

        return self.hardcoded_values

    def _scan_directory(self, directory: Path) -> None:
        """Recursively scan directory for Python files"""
        for path in directory.rglob("*.py"):
            if not any(pattern in str(path) for pattern in self.SKIP_PATTERNS):
                self._scan_file(path)

    def _scan_file(self, file_path: Path) -> None:
        """Scan a single file for hardcoded values"""
        if file_path in self.processed_files:
            return

        self.processed_files.add(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()

            # Check for specific hardcoded values
            self._check_ast_literals(file_path, content)

            # Check patterns
            for line_num, line in enumerate(lines, 1):
                for value_type, pattern in self.PATTERNS.items():
                    matches = pattern.findall(line)
                    for match in matches:
                        if self._should_report_value(match, value_type):
                            self.hardcoded_values.append(
                                HardcodedValue(
                                    file_path=file_path,
                                    line_number=line_num,
                                    value=match,
                                    value_type=value_type,
                                    context=line.strip(),
                                    suggested_config_key=self._suggest_config_key(
                                        file_path, value_type, match
                                    ),
                                )
                            )

        except Exception as e:
            logger.warning(f"Failed to scan {file_path}: {e}")

    def _check_ast_literals(self, file_path: Path, content: str) -> None:
        """Use AST to find hardcoded values more accurately"""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                # Check for hardcoded numbers in specific contexts
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Check for configuration-like variable names
                            if any(
                                keyword in target.id.lower()
                                for keyword in [
                                    "timeout",
                                    "port",
                                    "size",
                                    "limit",
                                    "max",
                                    "min",
                                ]
                            ):
                                if isinstance(node.value, ast.Constant):
                                    if isinstance(node.value.value, (int, float)):
                                        line_num = node.lineno
                                        self.hardcoded_values.append(
                                            HardcodedValue(
                                                file_path=file_path,
                                                line_number=line_num,
                                                value=str(node.value.value),
                                                value_type="numeric_constant",
                                                context=f"{target.id} = {node.value.value}",
                                                suggested_config_key=self._suggest_config_key(
                                                    file_path, "numeric", target.id
                                                ),
                                            )
                                        )

        except Exception as e:
            logger.debug(f"AST parsing failed for {file_path}: {e}")

    def _should_report_value(self, value: str, value_type: str) -> bool:
        """Determine if a value should be reported as hardcoded"""
        # Skip common non-configuration values
        skip_values = {
            "localhost",
            "127.0.0.1",
            "0.0.0.0",  # Common hosts
            "http://localhost",
            "https://localhost",  # Local URLs
            "__init__",
            "__main__",
            "__file__",  # Python internals
            "utf-8",
            "ascii",  # Common encodings
        }

        if value in skip_values:
            return False

        # Skip test values
        if "test" in value.lower() or "example" in value.lower():
            return False

        # API keys should always be reported
        if value_type == "api_key" and len(value) >= 32:
            return True

        # Paths starting with ./ or ../ are usually relative and OK
        if value_type == "path" and value.startswith(("./", "../")):
            return False

        return True

    def _suggest_config_key(
        self, file_path: Path, value_type: str, context: str
    ) -> str:
        """Suggest a configuration key for the hardcoded value"""
        # Extract module name
        relative_path = file_path.relative_to(self.project_root)
        module_parts = relative_path.parts[:-1]  # Exclude filename

        # Create base key
        if module_parts:
            base_key = "_".join(module_parts).upper()
        else:
            base_key = "KIMERA"

        # Add context-specific suffix
        if value_type == "api_key":
            if "openai" in context.lower():
                return "OPENAI_API_KEY"
            elif "cryptopanic" in context.lower():
                return "CRYPTOPANIC_API_KEY"
            else:
                return f"{base_key}_API_KEY"

        elif value_type == "database":
            return "KIMERA_DATABASE_URL"

        elif value_type == "port":
            return f"{base_key}_PORT"

        elif value_type == "host":
            return f"{base_key}_HOST"

        elif value_type == "timeout":
            return f"{base_key}_TIMEOUT"

        else:
            # Use context to create meaningful key
            context_clean = re.sub(r"[^a-zA-Z0-9_]", "_", context).upper()
            return f"{base_key}_{context_clean}"[:50]  # Limit length

    def generate_migration_report(self) -> str:
        """Generate a migration report with suggestions"""
        if not self.hardcoded_values:
            return "No hardcoded values found."

        # Group by file
        by_file = defaultdict(list)
        for hv in self.hardcoded_values:
            by_file[hv.file_path].append(hv)

        report_lines = [
            "# KIMERA Configuration Migration Report",
            f"Found {len(self.hardcoded_values)} hardcoded values in {len(by_file)} files",
            "",
            "## Summary by Type",
        ]

        # Summary by type
        by_type = defaultdict(int)
        for hv in self.hardcoded_values:
            by_type[hv.value_type] += 1

        for value_type, count in sorted(by_type.items()):
            report_lines.append(f"- {value_type}: {count}")

        report_lines.extend(["", "## Detailed Findings", ""])

        # Detailed findings
        for file_path, values in sorted(by_file.items()):
            report_lines.append(f"### {file_path.relative_to(self.project_root)}")
            report_lines.append("")

            for hv in sorted(values, key=lambda x: x.line_number):
                report_lines.extend(
                    [
                        f"**Line {hv.line_number}** ({hv.value_type})",
                        f"```python",
                        hv.context,
                        f"```",
                        f"- Value: `{hv.value}`",
                        f"- Suggested config key: `{hv.suggested_config_key}`",
                        f"- Replacement:",
                        f"```python",
                        f"from src.config import get_settings",
                        f"settings = get_settings()",
                        f"# Use: settings.{self._config_path_from_key(hv.suggested_config_key)}",
                        f"```",
                        "",
                    ]
                )

        return "\n".join(report_lines)

    def _config_path_from_key(self, key: str) -> str:
        """Convert environment key to settings path"""
        if key.startswith("OPENAI_"):
            return "api_keys.openai_api_key"
        elif key.startswith("CRYPTOPANIC_"):
            return "api_keys.cryptopanic_api_key"
        elif "DATABASE" in key:
            return "database.url"
        elif "PORT" in key:
            return "server.port"
        elif "HOST" in key:
            return "server.host"
        elif "TIMEOUT" in key:
            return "performance.request_timeout"
        else:
            return "get_custom_setting()"

    def generate_env_entries(self) -> List[str]:
        """Generate .env entries for found hardcoded values"""
        env_entries = []
        seen_keys = set()

        for hv in self.hardcoded_values:
            if hv.suggested_config_key not in seen_keys:
                seen_keys.add(hv.suggested_config_key)

                # Generate appropriate value
                if hv.value_type == "api_key":
                    value = ""  # Don't include actual API keys
                    comment = " # Add your API key here"
                else:
                    value = hv.value
                    comment = f" # Found in {hv.file_path.name}:{hv.line_number}"

                env_entries.append(f"{hv.suggested_config_key}={value}{comment}")

        return sorted(env_entries)


def migrate_configuration(project_root: Path = Path.cwd()) -> None:
    """
    Run configuration migration analysis

    Args:
        project_root: Root directory of the project
    """
    migrator = ConfigurationMigrator(project_root)

    # Scan for hardcoded values
    hardcoded_values = migrator.scan()

    # Generate reports
    migration_report = migrator.generate_migration_report()
    env_entries = migrator.generate_env_entries()

    # Save migration report
    report_path = project_root / "config_migration_report.md"
    with open(report_path, "w") as f:
        f.write(migration_report)

    logger.info(f"Migration report saved to: {report_path}")

    # Save suggested .env entries
    if env_entries:
        env_path = project_root / "suggested_env_entries.txt"
        with open(env_path, "w") as f:
            f.write("# Suggested environment variables based on scan\n")
            f.write("# Add these to your .env file\n\n")
            f.write("\n".join(env_entries))

        logger.info(f"Suggested .env entries saved to: {env_path}")

    # Print summary
    logger.info(f"\nConfiguration Migration Summary:")
    logger.info(f"- Files scanned: {len(migrator.processed_files)}")
    logger.info(f"- Hardcoded values found: {len(hardcoded_values)}")
    logger.info(f"- Migration report: {report_path}")
    if env_entries:
        logger.info(f"- Suggested .env entries: {env_path}")


if __name__ == "__main__":
    migrate_configuration()
