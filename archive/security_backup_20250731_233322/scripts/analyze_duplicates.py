#!/usr/bin/env python3
"""
Duplicate File Analysis for Kimera SWM
Protocol Version: 3.0
Purpose: Identify and analyze duplicate files for consolidation
"""

import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DuplicateAnalyzer:
    """Analyze duplicate files in the codebase"""
    
    def __init__(self, root_path='.'):
        self.root_path = Path(root_path)
        self.file_hashes = defaultdict(list)
        self.ast_signatures = defaultdict(list)
        self.ignore_patterns = [
            '.git', '__pycache__', '.pytest_cache', 
            'node_modules', '.venv', 'venv', '.env',
            '.pyc', '.pyo', '.pyd', '.so', '.dll'
        ]
        
    def should_ignore(self, path):
        """Check if path should be ignored"""
        path_str = str(path)
        return any(pattern in path_str for pattern in self.ignore_patterns)
    
    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file content"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def calculate_ast_signature(self, tree):
        """Generate a signature from AST for semantic comparison"""
        # Simplified AST signature - counts of different node types
        node_counts = defaultdict(int)
        for node in ast.walk(tree):
            node_counts[type(node).__name__] += 1
        
        # Create a stable signature
        signature = "|".join(f"{k}:{v}" for k, v in sorted(node_counts.items()))
        return hashlib.md5(signature.encode()).hexdigest()
    
    def analyze_python_file(self, file_path):
        """Analyze a Python file for AST similarity"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            return self.calculate_ast_signature(tree)
        except Exception as e:
            logger.debug(f"Could not parse {file_path}: {e}")
            return None
    
    def scan_files(self):
        """Scan all files and identify duplicates"""
        logger.info("Scanning for duplicate files...")
        
        total_files = 0
        python_files = 0
        
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and not self.should_ignore(file_path):
                total_files += 1
                
                # Calculate content hash
                try:
                    content_hash = self.calculate_file_hash(file_path)
                    self.file_hashes[content_hash].append(file_path)
                    
                    # For Python files, also check AST similarity
                    if file_path.suffix == '.py':
                        python_files += 1
                        ast_sig = self.analyze_python_file(file_path)
                        if ast_sig:
                            self.ast_signatures[ast_sig].append(file_path)
                            
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(f"Scanned {total_files} files ({python_files} Python files)")
    
    def find_duplicates(self):
        """Find all duplicate files"""
        exact_duplicates = []
        similar_python_files = []
        
        # Find exact duplicates
        for file_hash, files in self.file_hashes.items():
            if len(files) > 1:
                exact_duplicates.append({
                    'hash': file_hash,
                    'files': [str(f) for f in files],
                    'count': len(files),
                    'size': files[0].stat().st_size
                })
        
        # Find similar Python files (same AST signature but different content)
        for ast_sig, files in self.ast_signatures.items():
            if len(files) > 1:
                # Check if they're not already exact duplicates
                file_strs = [str(f) for f in files]
                is_exact = any(
                    set(file_strs).issubset(set(dup['files'])) 
                    for dup in exact_duplicates
                )
                
                if not is_exact:
                    similar_python_files.append({
                        'ast_signature': ast_sig,
                        'files': file_strs,
                        'count': len(files)
                    })
        
        return exact_duplicates, similar_python_files
    
    def generate_report(self, exact_duplicates, similar_files):
        """Generate a detailed duplicate report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_duplicate_groups': len(exact_duplicates),
                'total_duplicate_files': sum(d['count'] for d in exact_duplicates),
                'total_wasted_space_mb': sum(d['size'] * (d['count'] - 1) for d in exact_duplicates) / (1024 * 1024),
                'similar_python_groups': len(similar_files)
            },
            'exact_duplicates': exact_duplicates,
            'similar_python_files': similar_files,
            'recommendations': self.generate_recommendations(exact_duplicates, similar_files)
        }
        
        return report
    
    def generate_recommendations(self, exact_duplicates, similar_files):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Group duplicates by directory patterns
        archive_duplicates = 0
        test_duplicates = 0
        backend_duplicates = 0
        
        for dup_group in exact_duplicates:
            files = dup_group['files']
            if any('archive' in f for f in files):
                archive_duplicates += 1
            if any('test' in f for f in files):
                test_duplicates += 1
            if any('backend' in f for f in files):
                backend_duplicates += 1
        
        if archive_duplicates > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Archive Cleanup',
                'description': f"Found {archive_duplicates} duplicate groups involving archive/ directory",
                'action': 'Consolidate or remove duplicates from archive/'
            })
        
        if backend_duplicates > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Production Code',
                'description': f"Found {backend_duplicates} duplicate groups in backend/ directory",
                'action': 'Review and consolidate duplicate production code immediately'
            })
        
        if test_duplicates > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Test Code',
                'description': f"Found {test_duplicates} duplicate groups in test files",
                'action': 'Consolidate duplicate test code to improve maintainability'
            })
        
        if len(similar_files) > 0:
            recommendations.append({
                'priority': 'LOW',
                'category': 'Code Similarity',
                'description': f"Found {len(similar_files)} groups of semantically similar Python files",
                'action': 'Review for potential refactoring opportunities'
            })
        
        return recommendations


def main():
    """Main execution"""
    analyzer = DuplicateAnalyzer()
    
    # Scan files
    analyzer.scan_files()
    
    # Find duplicates
    exact_dups, similar_files = analyzer.find_duplicates()
    
    # Generate report
    report = analyzer.generate_report(exact_dups, similar_files)
    
    # Save detailed report
    with open('duplicate_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DUPLICATE FILE ANALYSIS REPORT")
    logger.info("="*60)
    logger.info(f"\nTimestamp: {report['timestamp']}")
    logger.info(f"\nSUMMARY:")
    logger.info(f"- Duplicate groups: {report['summary']['total_duplicate_groups']}")
    logger.info(f"- Total duplicate files: {report['summary']['total_duplicate_files']}")
    logger.info(f"- Wasted space: {report['summary']['total_wasted_space_mb']:.2f} MB")
    logger.info(f"- Similar Python file groups: {report['summary']['similar_python_groups']}")
    
    logger.info(f"\nRECOMMENDATIONS:")
    for rec in report['recommendations']:
        logger.info(f"\n[{rec['priority']}] {rec['category']}")
        logger.info(f"  {rec['description']}")
        logger.info(f"  â†’ {rec['action']}")
    
    logger.info(f"\nDetailed report saved to: duplicate_analysis_report.json")
    
    # Show first 5 duplicate groups as examples
    if exact_dups:
        logger.info(f"\nEXAMPLE DUPLICATES (first 5):")
        for i, dup in enumerate(exact_dups[:5]):
            logger.info(f"\nGroup {i+1} ({dup['count']} files, {dup['size']/1024:.1f} KB each):")
            for file in dup['files'][:3]:  # Show max 3 files per group
                logger.info(f"  - {file}")
            if len(dup['files']) > 3:
                logger.info(f"  ... and {len(dup['files']) - 3} more")


if __name__ == '__main__':
    main() 