#!/usr/bin/env python3
"""
Automated Security Scanner for Kimera System
Continuously monitors for credential exposure and security issues
"""

import re
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

class SecurityScanner:
    """Automated security scanner for credential exposure"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scan_results = []
        
        # Patterns for credential detection
        self.credential_patterns = [
            r'(?i)api_key\s*=\s*["\']([^"\']+)["\']',
            r'(?i)secret\s*=\s*["\']([^"\']+)["\']',
            r'(?i)password\s*=\s*["\']([^"\']+)["\']',
            r'(?i)token\s*=\s*["\']([^"\']+)["\']',
            r'sk-[a-zA-Z0-9]{32,}',  # OpenAI API keys
            r'pk-[a-zA-Z0-9]{32,}',  # Other API keys
            r'ghp_[a-zA-Z0-9]{36}',  # GitHub tokens
            r'hf_[a-zA-Z0-9]{34}',   # HuggingFace tokens
        ]
        
        # Safe patterns that should not be flagged
        self.safe_patterns = [
            r'api_key\s*=\s*["\']#',  # Comments
            r'password\s*=\s*["\']#',  # Comments
            r'os\.getenv\(',  # Environment variables
            r'get_secure_demo_key\(',  # Our secure functions
            r'hashlib\.sha256\(',  # Generated hashes
            r'test_.*=.*test_',  # Test credentials
        ]
    
    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single file for credentials"""
        findings = []
        
        if not file_path.exists() or file_path.suffix != '.py':
            return findings
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                # Skip safe patterns
                if any(re.search(pattern, line) for pattern in self.safe_patterns):
                    continue
                
                # Check for credential patterns
                for pattern in self.credential_patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        findings.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'pattern': pattern,
                            'match': match.group(0),
                            'severity': 'HIGH' if any(x in line.lower() for x in ['production', 'prod', 'live']) else 'MEDIUM'
                        })
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return findings
    
    def scan_project(self) -> Dict[str, Any]:
        """Scan entire project for credentials"""
        all_findings = []
        
        # Scan Python files
        for py_file in self.project_root.rglob('*.py'):
            # Skip virtual environments and build directories
            if any(part in str(py_file) for part in ['venv', 'env', '__pycache__', 'build', 'dist']):
                continue
            
            findings = self.scan_file(py_file)
            all_findings.extend(findings)
        
        # Scan configuration files
        for config_file in self.project_root.rglob('*.json'):
            findings = self.scan_file(config_file)
            all_findings.extend(findings)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_findings': len(all_findings),
            'findings': all_findings,
            'summary': {
                'high_severity': len([f for f in all_findings if f['severity'] == 'HIGH']),
                'medium_severity': len([f for f in all_findings if f['severity'] == 'MEDIUM']),
                'files_scanned': len(set(f['file'] for f in all_findings))
            }
        }
    
    def generate_report(self) -> str:
        """Generate security report"""
        results = self.scan_project()
        
        report = f"""
# Security Scan Report
Generated: {results['timestamp']}

## Summary
- Total findings: {results['total_findings']}
- High severity: {results['summary']['high_severity']}
- Medium severity: {results['summary']['medium_severity']}
- Files with issues: {results['summary']['files_scanned']}

## Findings
"""
        
        for finding in results['findings']:
            report += f"""
### {finding['severity']} - {finding['file']}:{finding['line']}
```
{finding['match']}
```
Pattern: {finding['pattern']}
"""
        
        return report

if __name__ == "__main__":
    scanner = SecurityScanner(Path.cwd())
    report = scanner.generate_report()
    print(report)
    
    # Save report
    with open("security_scan_report.md", "w") as f:
        f.write(report)
