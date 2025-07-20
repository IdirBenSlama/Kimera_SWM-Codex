#!/usr/bin/env python
"""
Credential Security Scanner for Kimera SWM

This script scans the codebase for potentially hardcoded credentials, API keys,
and other sensitive information that should be stored in environment variables.

Usage:
    python scripts/check_credentials_security.py [--fix]
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Patterns to detect potential hardcoded credentials
CREDENTIAL_PATTERNS = [
    r'api_key\s*=\s*["\']([^"\']+)["\']',
    r'apikey\s*=\s*["\']([^"\']+)["\']',
    r'API_KEY\s*=\s*["\']([^"\']+)["\']',
    r'password\s*=\s*["\']([^"\']+)["\']',
    r'PASSWORD\s*=\s*["\']([^"\']+)["\']',
    r'secret\s*=\s*["\']([^"\']+)["\']',
    r'SECRET\s*=\s*["\']([^"\']+)["\']',
    r'token\s*=\s*["\']([^"\']+)["\']',
    r'TOKEN\s*=\s*["\']([^"\']+)["\']',
    r'auth\s*=\s*["\']([^"\']+)["\']',
    r'AUTH\s*=\s*["\']([^"\']+)["\']',
]

# Files and directories to exclude
EXCLUDE_DIRS = [
    '.git',
    '.venv',
    '__pycache__',
    'node_modules',
    'archive',
    'tests',
]

# Files to exclude
EXCLUDE_FILES = [
    '.env.template',
    'check_credentials_security.py',
    'CREDENTIALS_CONFIGURATION_GUIDE.md',
]

# Safe patterns that should not be flagged
SAFE_PATTERNS = [
    r'api_key\s*=\s*["\']YOUR_API_KEY["\']',
    r'api_key\s*=\s*["\']REPLACE_WITH_YOUR_API_KEY["\']',
    r'api_key\s*=\s*os\.getenv',
    r'api_key\s*=\s*ConfigManager\.get_api_key',
    r'api_key\s*=\s*None',
    r'api_key\s*=\s*""',
    r'api_key\s*=\s*\'\'',
]

def is_safe_pattern(line: str) -> bool:
    """Check if a line matches any safe pattern"""
    for pattern in SAFE_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False

def scan_file(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Scan a file for potential hardcoded credentials
    
    Args:
        file_path: Path to the file to scan
        
    Returns:
        List of tuples containing (line_number, pattern, line)
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                # Skip comments
                if line.strip().startswith('#') or line.strip().startswith('//'):
                    continue
                    
                # Check if line contains a credential pattern
                for pattern in CREDENTIAL_PATTERNS:
                    match = re.search(pattern, line)
                    if match and not is_safe_pattern(line):
                        issues.append((i, pattern, line.strip()))
                        break
    except Exception as e:
        logger.warning(f"Error scanning {file_path}: {e}")
    
    return issues

def scan_directory(directory: Path) -> Dict[str, List[Tuple[int, str, str]]]:
    """
    Recursively scan a directory for potential hardcoded credentials
    
    Args:
        directory: Directory to scan
        
    Returns:
        Dictionary mapping file paths to lists of issues
    """
    results = {}
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            # Only scan Python, JavaScript, and configuration files
            if not file.endswith(('.py', '.js', '.ts', '.json', '.yaml', '.yml')):
                continue
                
            # Skip excluded files
            if file in EXCLUDE_FILES:
                continue
                
            file_path = Path(root) / file
            issues = scan_file(file_path)
            
            if issues:
                results[str(file_path)] = issues
    
    return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Scan for hardcoded credentials')
    parser.add_argument('--fix', action='store_true', help='Suggest fixes for issues')
    args = parser.parse_args()
    
    logger.info("Scanning for hardcoded credentials...")
    
    # Get project root directory (parent of the scripts directory)
    project_root = Path(__file__).parent.parent
    
    # Scan the project
    results = scan_directory(project_root)
    
    # Report results
    if not results:
        logger.info("✅ No hardcoded credentials found!")
        return
    
    total_issues = sum(len(issues) for issues in results.values())
    logger.warning(f"⚠️ Found {total_issues} potential hardcoded credentials in {len(results)} files")
    
    for file_path, issues in results.items():
        print(f"\n{file_path}:")
        for line_num, pattern, line in issues:
            print(f"  Line {line_num}: {line}")
            
            if args.fix:
                print("  Suggestion: Use environment variables instead:")
                print("    from backend.config.config_manager import ConfigManager")
                print("    api_key = ConfigManager.get_api_key('service_name')")
    
    print("\nRecommendation:")
    print("1. Replace hardcoded credentials with environment variables")
    print("2. Use the ConfigManager to access credentials")
    print("3. See docs/CREDENTIALS_CONFIGURATION_GUIDE.md for more information")

if __name__ == "__main__":
    main() 