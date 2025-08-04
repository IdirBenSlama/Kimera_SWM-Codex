#!/usr/bin/env python3
"""
Analyze Archive Directory Structure
Protocol Version: 3.0
Purpose: Categorize and document broken scripts in archive
"""

import os
import ast
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class ArchiveAnalyzer:
    """Analyze and categorize files in the archive directory"""
    
    def __init__(self, archive_path='archive'):
        self.archive_path = Path(archive_path)
        self.categories = defaultdict(list)
        self.syntax_errors = defaultdict(list)
        self.file_stats = {
            'total_files': 0,
            'python_files': 0,
            'broken_python': 0,
            'documentation': 0,
            'other': 0
        }
        
    def analyze_python_file(self, file_path):
        """Analyze a Python file for syntax errors"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the file
            ast.parse(content, filename=str(file_path))
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Other error: {str(e)}"
    
    def categorize_file(self, file_path):
        """Categorize a file based on its path and content"""
        relative_path = file_path.relative_to(self.archive_path)
        path_parts = relative_path.parts
        
        # Determine category based on path
        if 'broken_scripts_and_tests' in path_parts:
            if 'testing' in path_parts:
                return 'broken_tests'
            elif 'development' in path_parts:
                return 'broken_development'
            elif 'demonstration' in path_parts:
                return 'broken_demos'
            elif 'analysis' in path_parts:
                return 'broken_analysis'
            elif 'utilities' in path_parts:
                return 'broken_utilities'
            else:
                return 'broken_other'
        elif 'documentation' in path_parts:
            return 'old_documentation'
        else:
            return 'archived_code'
    
    def scan_archive(self):
        """Scan the entire archive directory"""
        logger.info(f"Scanning archive directory: {self.archive_path}")
        
        for file_path in self.archive_path.rglob('*'):
            if file_path.is_file():
                self.file_stats['total_files'] += 1
                
                # Categorize the file
                category = self.categorize_file(file_path)
                self.categories[category].append(str(file_path))
                
                # Analyze Python files
                if file_path.suffix == '.py':
                    self.file_stats['python_files'] += 1
                    
                    is_valid, error = self.analyze_python_file(file_path)
                    if not is_valid:
                        self.file_stats['broken_python'] += 1
                        self.syntax_errors[category].append({
                            'file': str(file_path),
                            'error': error
                        })
                elif file_path.suffix in ['.md', '.txt', '.rst']:
                    self.file_stats['documentation'] += 1
                else:
                    self.file_stats['other'] += 1
    
    def generate_report(self):
        """Generate a comprehensive archive analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.file_stats,
            'categories': {k: len(v) for k, v in self.categories.items()},
            'broken_files_by_category': {
                k: len(v) for k, v in self.syntax_errors.items()
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if self.syntax_errors.get('broken_tests', []):
            report['recommendations'].append({
                'priority': 'LOW',
                'category': 'Broken Tests',
                'count': len(self.syntax_errors['broken_tests']),
                'action': 'Review for potential recovery of test patterns'
            })
        
        if self.syntax_errors.get('broken_development', []):
            report['recommendations'].append({
                'priority': 'MEDIUM',
                'category': 'Broken Development Scripts',
                'count': len(self.syntax_errors['broken_development']),
                'action': 'Check for valuable algorithms or approaches'
            })
        
        if self.syntax_errors.get('broken_utilities', []):
            report['recommendations'].append({
                'priority': 'HIGH',
                'category': 'Broken Utilities',
                'count': len(self.syntax_errors['broken_utilities']),
                'action': 'Some utilities might be worth fixing and moving to scripts/'
            })
        
        return report
    
    def create_archive_summary(self):
        """Create a summary document for the archive"""
        summary = f"""# Archive Directory Summary
**Generated**: {datetime.now():%Y-%m-%d %H:%M}  
**Protocol Version**: 3.0  

## Overview

The archive directory contains historical code, broken scripts, and deprecated functionality.
This summary provides an analysis of the archive contents.

## Statistics

- **Total Files**: {self.file_stats['total_files']}
- **Python Files**: {self.file_stats['python_files']}
- **Broken Python Files**: {self.file_stats['broken_python']}
- **Documentation Files**: {self.file_stats['documentation']}
- **Other Files**: {self.file_stats['other']}

## Categories

"""
        # Add category breakdown
        for category, files in sorted(self.categories.items(), key=lambda x: len(x[1]), reverse=True):
            summary += f"### {category.replace('_', ' ').title()}\n"
            summary += f"- **File Count**: {len(files)}\n"
            
            if category in self.syntax_errors:
                summary += f"- **Broken Files**: {len(self.syntax_errors[category])}\n"
                
                # Show first 3 errors as examples
                summary += f"- **Example Errors**:\n"
                for error_info in self.syntax_errors[category][:3]:
                    file_name = Path(error_info['file']).name
                    error_msg = error_info['error'].split('\n')[0]  # First line only
                    summary += f"  - `{file_name}`: {error_msg}\n"
            
            summary += "\n"
        
        summary += """## Recommendations

1. **Broken Utilities**: Some utility scripts might contain useful functions that could be 
   extracted and moved to the main codebase after fixing.

2. **Test Patterns**: Broken test files might contain valuable test scenarios that could 
   be rewritten for the current codebase.

3. **Documentation**: Old documentation should be reviewed to ensure all relevant 
   information has been transferred to current docs.

4. **Gradual Cleanup**: Consider periodic reviews to permanently remove truly obsolete code.

---
*Generated by Kimera SWM Archive Analyzer*
"""
        
        return summary


def main():
    """Main execution"""
    analyzer = ArchiveAnalyzer()
    
    # Scan the archive
    analyzer.scan_archive()
    
    # Generate report
    report = analyzer.generate_report()
    
    # Save detailed report
    with open('archive_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create and save summary
    summary = analyzer.create_archive_summary()
    summary_path = Path('archive/ARCHIVE_SUMMARY.md')
    summary_path.write_text(summary)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("ARCHIVE ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"\nTotal files in archive: {report['stats']['total_files']}")
    logger.info(f"Broken Python files: {report['stats']['broken_python']}")
    
    logger.info("\nCategories:")
    for category, count in sorted(report['categories'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {category}: {count} files")
    
    logger.info("\nBroken files by category:")
    for category, count in report['broken_files_by_category'].items():
        logger.info(f"  - {category}: {count} broken files")
    
    logger.info(f"\nDetailed report: archive_analysis_report.json")
    logger.info(f"Summary document: {summary_path}")
    
    # Show recommendations
    if report['recommendations']:
        logger.info("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            logger.info(f"\n[{rec['priority']}] {rec['category']} ({rec['count']} files)")
            logger.info(f"  → {rec['action']}")


if __name__ == '__main__':
    main() 