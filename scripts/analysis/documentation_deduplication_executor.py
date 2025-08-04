#!/usr/bin/env python3
"""
KIMERA SWM Documentation Deduplication Executor
===============================================

Executes Phase 3 of technical debt remediation: Documentation Deduplication
Following Martin Fowler framework and KIMERA SWM Protocol v3.0

Implements:
- Duplicate content detection using content hashing
- Similarity analysis using fuzzy matching
- Single source of truth establishment
- Redundant file archival with explanation
"""

import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
import json
import shutil
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentationDeduplicator:
    """Analyzes and deduplicates documentation files"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.doc_extensions = {'.md', '.txt', '.rst', '.adoc', '.wiki'}
        self.backup_dir = self.project_root / f"backup_docs_dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def analyze_documentation(self):
        """Analyze all documentation files for duplicates and redundancy"""
        logger.info("🔍 Analyzing documentation for duplicates...")

        doc_files = []

        # Find all documentation files
        for ext in self.doc_extensions:
            doc_files.extend(list(self.project_root.rglob(f"*{ext}")))

        # Filter out certain directories
        skip_patterns = {'.venv', '__pycache__', '.git', '.mypy_cache', 'node_modules'}
        doc_files = [f for f in doc_files if not any(pattern in str(f) for pattern in skip_patterns)]

        logger.info(f"📄 Found {len(doc_files)} documentation files")

        # Analyze each file
        file_analysis = {}
        content_hashes = {}

        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Calculate content hash
                content_hash = hashlib.md5(content.encode()).hexdigest()

                # Basic metrics
                word_count = len(content.split())
                line_count = len(content.splitlines())
                size_kb = doc_file.stat().st_size / 1024

                file_info = {
                    'path': str(doc_file),
                    'relative_path': str(doc_file.relative_to(self.project_root)),
                    'content_hash': content_hash,
                    'word_count': word_count,
                    'line_count': line_count,
                    'size_kb': size_kb,
                    'content': content,
                    'extension': doc_file.suffix
                }

                file_analysis[str(doc_file)] = file_info

                # Track content hashes for exact duplicates
                if content_hash not in content_hashes:
                    content_hashes[content_hash] = []
                content_hashes[content_hash].append(str(doc_file))

            except Exception as e:
                logger.warning(f"Error analyzing {doc_file}: {e}")

        return file_analysis, content_hashes

    def find_exact_duplicates(self, content_hashes: Dict[str, List[str]]):
        """Find files with identical content"""
        exact_duplicates = []

        for content_hash, files in content_hashes.items():
            if len(files) > 1:
                exact_duplicates.append({
                    'hash': content_hash,
                    'files': files,
                    'count': len(files)
                })

        logger.info(f"🔍 Found {len(exact_duplicates)} groups of exact duplicates")
        return exact_duplicates

    def find_similar_documents(self, file_analysis: Dict, similarity_threshold: float = 0.8):
        """Find documents with similar content using fuzzy matching"""
        logger.info("🔍 Analyzing content similarity...")

        similar_groups = []
        processed_files = set()

        files = list(file_analysis.keys())

        for i, file1 in enumerate(files):
            if file1 in processed_files:
                continue

            similar_to_file1 = [file1]

            for file2 in files[i+1:]:
                if file2 in processed_files:
                    continue

                # Calculate similarity
                content1 = file_analysis[file1]['content']
                content2 = file_analysis[file2]['content']

                similarity = SequenceMatcher(None, content1, content2).ratio()

                if similarity >= similarity_threshold:
                    similar_to_file1.append(file2)
                    processed_files.add(file2)

            if len(similar_to_file1) > 1:
                similar_groups.append({
                    'primary': file1,
                    'similar_files': similar_to_file1,
                    'count': len(similar_to_file1)
                })
                processed_files.add(file1)

        logger.info(f"🔍 Found {len(similar_groups)} groups of similar documents")
        return similar_groups

    def analyze_redundancy_patterns(self, file_analysis: Dict):
        """Analyze patterns of redundancy"""
        logger.info("🔍 Analyzing redundancy patterns...")

        patterns = {
            'readme_files': [],
            'changelog_files': [],
            'todo_files': [],
            'status_reports': [],
            'roadmap_files': [],
            'summary_files': [],
            'analysis_files': [],
            'completion_reports': []
        }

        for file_path, info in file_analysis.items():
            filename = Path(file_path).name.lower()

            if 'readme' in filename:
                patterns['readme_files'].append(file_path)
            elif 'changelog' in filename or 'change' in filename:
                patterns['changelog_files'].append(file_path)
            elif 'todo' in filename:
                patterns['todo_files'].append(file_path)
            elif 'status' in filename or 'report' in filename:
                patterns['status_reports'].append(file_path)
            elif 'roadmap' in filename:
                patterns['roadmap_files'].append(file_path)
            elif 'summary' in filename:
                patterns['summary_files'].append(file_path)
            elif 'analysis' in filename:
                patterns['analysis_files'].append(file_path)
            elif 'completion' in filename or 'complete' in filename:
                patterns['completion_reports'].append(file_path)

        # Log patterns found
        for pattern_type, files in patterns.items():
            if len(files) > 1:
                logger.info(f"   📄 {pattern_type}: {len(files)} files")

        return patterns

    def create_deduplication_plan(self, file_analysis: Dict, exact_duplicates: List,
                                similar_groups: List, patterns: Dict):
        """Create comprehensive deduplication plan"""
        logger.info("📋 Creating deduplication plan...")

        plan = {
            'timestamp': datetime.now().isoformat(),
            'backup_location': str(self.backup_dir),
            'actions': [],
            'statistics': {
                'total_files': len(file_analysis),
                'exact_duplicates': sum(len(group['files']) - 1 for group in exact_duplicates),
                'similar_documents': sum(len(group['similar_files']) - 1 for group in similar_groups),
                'pattern_redundancy': sum(max(0, len(files) - 1) for files in patterns.values())
            }
        }

        # Handle exact duplicates
        for duplicate_group in exact_duplicates:
            files = duplicate_group['files']
            # Keep the first file (alphabetically) as primary
            primary_file = sorted(files)[0]
            duplicates_to_remove = [f for f in files if f != primary_file]

            for duplicate_file in duplicates_to_remove:
                plan['actions'].append({
                    'type': 'remove_exact_duplicate',
                    'file': duplicate_file,
                    'primary': primary_file,
                    'reason': f"Exact duplicate of {Path(primary_file).name}"
                })

        # Handle similar documents
        for similar_group in similar_groups:
            primary = similar_group['primary']
            similar_files = [f for f in similar_group['similar_files'] if f != primary]

            # Choose the most comprehensive file as primary
            # (largest file size usually indicates more complete content)
            all_files = similar_group['similar_files']
            primary = max(all_files, key=lambda f: file_analysis[f]['size_kb'])
            similar_files = [f for f in all_files if f != primary]

            for similar_file in similar_files:
                plan['actions'].append({
                    'type': 'archive_similar',
                    'file': similar_file,
                    'primary': primary,
                    'reason': f"Similar content to {Path(primary).name}"
                })

        # Handle pattern redundancy
        for pattern_type, files in patterns.items():
            if len(files) > 1:
                # Keep the most recent file
                files_with_time = []
                for file_path in files:
                    try:
                        mtime = Path(file_path).stat().st_mtime
                        files_with_time.append((file_path, mtime))
                    except:
                        files_with_time.append((file_path, 0))

                # Sort by modification time (newest first)
                files_with_time.sort(key=lambda x: x[1], reverse=True)
                primary_file = files_with_time[0][0]
                redundant_files = [f[0] for f in files_with_time[1:]]

                for redundant_file in redundant_files:
                    plan['actions'].append({
                        'type': 'archive_redundant_pattern',
                        'file': redundant_file,
                        'primary': primary_file,
                        'pattern': pattern_type,
                        'reason': f"Redundant {pattern_type.replace('_', ' ')} - superseded by {Path(primary_file).name}"
                    })

        return plan

    def create_backup(self):
        """Create backup before deduplication"""
        logger.info("💾 Creating documentation backup...")

        self.backup_dir.mkdir(exist_ok=True)

        # Find all documentation files
        doc_files = []
        for ext in self.doc_extensions:
            doc_files.extend(list(self.project_root.rglob(f"*{ext}")))

        backup_count = 0
        for doc_file in doc_files:
            try:
                # Calculate relative path
                rel_path = doc_file.relative_to(self.project_root)
                backup_path = self.backup_dir / rel_path

                # Create directory if needed
                backup_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(doc_file, backup_path)
                backup_count += 1

            except Exception as e:
                logger.warning(f"Error backing up {doc_file}: {e}")

        logger.info(f"✅ Backed up {backup_count} documentation files to: {self.backup_dir}")

    def execute_deduplication(self, plan: Dict, dry_run: bool = True):
        """Execute the deduplication plan"""
        logger.info(f"🚀 {'DRY RUN:' if dry_run else 'EXECUTING:'} Documentation Deduplication")

        if not dry_run:
            self.create_backup()

        results = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'actions_completed': 0,
            'files_removed': 0,
            'files_archived': 0,
            'errors': []
        }

        # Create archive directory for redundant files
        archive_dir = self.project_root / "archive" / f"documentation_dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for action in plan['actions']:
            try:
                file_path = Path(action['file'])

                if dry_run:
                    logger.info(f"📄 Would {action['type']}: {file_path.name}")
                    logger.info(f"   Reason: {action['reason']}")
                else:
                    if action['type'] == 'remove_exact_duplicate':
                        # Remove exact duplicates
                        file_path.unlink()
                        logger.info(f"🗑️  Removed exact duplicate: {file_path.name}")
                        results['files_removed'] += 1

                    elif action['type'] in ['archive_similar', 'archive_redundant_pattern']:
                        # Archive similar/redundant files
                        archive_target = archive_dir / file_path.relative_to(self.project_root)
                        archive_target.parent.mkdir(parents=True, exist_ok=True)

                        # Create explanation file
                        explanation_file = archive_target.with_suffix(archive_target.suffix + '.ARCHIVED')
                        with open(explanation_file, 'w') as f:
                            f.write(f"ARCHIVED: {datetime.now().isoformat()}\n")
                            f.write(f"REASON: {action['reason']}\n")
                            f.write(f"PRIMARY: {action['primary']}\n")

                        # Move file to archive
                        shutil.move(str(file_path), str(archive_target))
                        logger.info(f"📦 Archived: {file_path.name} → archive/")
                        results['files_archived'] += 1

                results['actions_completed'] += 1

            except Exception as e:
                error_msg = f"Error processing {action['file']}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)

        return results

    def generate_report(self, plan: Dict, results: Dict, file_analysis: Dict):
        """Generate comprehensive deduplication report"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

        total_size_before = sum(info['size_kb'] for info in file_analysis.values())
        estimated_size_after = total_size_before

        # Estimate size reduction
        for action in plan['actions']:
            if action['file'] in file_analysis:
                estimated_size_after -= file_analysis[action['file']]['size_kb']

        size_reduction = total_size_before - estimated_size_after
        reduction_percentage = (size_reduction / total_size_before) * 100 if total_size_before > 0 else 0

        report = f"""# KIMERA SWM Documentation Deduplication Report
**Generated**: {timestamp}
**Phase**: 3 of Technical Debt Remediation
**Framework**: Martin Fowler + KIMERA SWM Protocol v3.0

## Executive Summary

**Status**: {'✅ COMPLETED' if not results['dry_run'] else '🔄 DRY RUN'}
- **Files Analyzed**: {len(file_analysis)}
- **Actions Completed**: {results['actions_completed']}
- **Files Removed**: {results['files_removed']}
- **Files Archived**: {results['files_archived']}
- **Errors**: {len(results['errors'])}

## Impact Analysis

### Size Reduction
- **Total Size Before**: {total_size_before:.1f} KB
- **Estimated Size After**: {estimated_size_after:.1f} KB
- **Size Reduction**: {size_reduction:.1f} KB ({reduction_percentage:.1f}%)

### Redundancy Statistics
- **Exact Duplicates**: {plan['statistics']['exact_duplicates']} files
- **Similar Documents**: {plan['statistics']['similar_documents']} files
- **Pattern Redundancy**: {plan['statistics']['pattern_redundancy']} files

## Actions Performed

### Exact Duplicates Removed
"""

        exact_duplicate_actions = [a for a in plan['actions'] if a['type'] == 'remove_exact_duplicate']
        for action in exact_duplicate_actions:
            report += f"- ❌ **{Path(action['file']).name}** (duplicate of {Path(action['primary']).name})\n"

        report += "\n### Similar Documents Archived\n"
        similar_actions = [a for a in plan['actions'] if a['type'] == 'archive_similar']
        for action in similar_actions:
            report += f"- 📦 **{Path(action['file']).name}** (similar to {Path(action['primary']).name})\n"

        report += "\n### Redundant Patterns Archived\n"
        pattern_actions = [a for a in plan['actions'] if a['type'] == 'archive_redundant_pattern']
        for action in pattern_actions:
            report += f"- 📦 **{Path(action['file']).name}** ({action['pattern'].replace('_', ' ')})\n"

        if results['errors']:
            report += "\n## Errors Encountered\n"
            for error in results['errors']:
                report += f"- ❌ {error}\n"
        else:
            report += "\n## ✅ No Errors - Perfect Execution\n"

        report += f"""
## Benefits Achieved

### Documentation Quality
- **Single Source of Truth**: Eliminated conflicting documentation
- **Reduced Maintenance**: Fewer files to keep updated
- **Improved Navigation**: Clearer documentation structure
- **Storage Efficiency**: {reduction_percentage:.1f}% reduction in documentation storage

### Developer Experience
- **Faster Onboarding**: Clear, non-redundant documentation
- **Reduced Confusion**: No conflicting information sources
- **Easier Updates**: Single files to maintain per topic

### Next Steps
1. Review archived files to ensure no critical information lost
2. Update internal links to point to primary documents
3. Establish documentation standards to prevent future redundancy
4. Implement automated duplicate detection in CI/CD

### Backup Information
- **Backup Location**: {plan.get('backup_location', 'N/A')}
- **Recovery Instructions**: Restore from backup if critical information missing

---

*Phase 3 of KIMERA SWM Technical Debt Remediation*
*Following Martin Fowler's Technical Debt Quadrants Framework*
"""

        # Save report
        report_dir = Path("docs/reports/debt")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{timestamp}_documentation_deduplication_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"📄 Report saved: {report_path}")
        return str(report_path)

def main():
    """Main execution function"""
    logger.info("🚀 KIMERA SWM Documentation Deduplication - Phase 3")
    logger.info("=" * 60)

    deduplicator = DocumentationDeduplicator()

    # Step 1: Analyze documentation
    file_analysis, content_hashes = deduplicator.analyze_documentation()

    if not file_analysis:
        logger.info("✅ No documentation files found - deduplication not needed")
        return

    # Step 2: Find duplicates and patterns
    exact_duplicates = deduplicator.find_exact_duplicates(content_hashes)
    similar_groups = deduplicator.find_similar_documents(file_analysis)
    patterns = deduplicator.analyze_redundancy_patterns(file_analysis)

    # Step 3: Create deduplication plan
    plan = deduplicator.create_deduplication_plan(file_analysis, exact_duplicates, similar_groups, patterns)

    # Step 4: Execute dry run
    logger.info("\n🔄 Executing DRY RUN...")
    dry_results = deduplicator.execute_deduplication(plan, dry_run=True)

    # Step 5: Generate dry run report
    report_path = deduplicator.generate_report(plan, dry_results, file_analysis)

    logger.info(f"\n📊 DRY RUN RESULTS:")
    logger.info(f"   Actions Planned: {dry_results['actions_completed']}")
    logger.info(f"   Files to Remove: {plan['statistics']['exact_duplicates']}")
    logger.info(f"   Files to Archive: {plan['statistics']['similar_documents'] + plan['statistics']['pattern_redundancy']}")
    logger.info(f"   Errors: {len(dry_results['errors'])}")

    logger.info(f"\n💡 To execute actual deduplication, run with --execute flag")
    logger.info(f"📄 Detailed report: {report_path}")

if __name__ == "__main__":
    import sys

    if "--execute" in sys.argv:
        logger.info("⚠️  EXECUTING ACTUAL DEDUPLICATION...")
        deduplicator = DocumentationDeduplicator()
        file_analysis, content_hashes = deduplicator.analyze_documentation()
        exact_duplicates = deduplicator.find_exact_duplicates(content_hashes)
        similar_groups = deduplicator.find_similar_documents(file_analysis)
        patterns = deduplicator.analyze_redundancy_patterns(file_analysis)
        plan = deduplicator.create_deduplication_plan(file_analysis, exact_duplicates, similar_groups, patterns)
        results = deduplicator.execute_deduplication(plan, dry_run=False)
        report_path = deduplicator.generate_report(plan, results, file_analysis)
        logger.info("✅ Documentation deduplication complete!")
        logger.info(f"📄 Final report: {report_path}")
    else:
        main()
