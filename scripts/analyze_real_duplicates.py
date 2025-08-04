#!/usr/bin/env python3
"""
Analyze real duplicates - excluding empty files and __init__.py
Protocol Version: 3.0
"""

import json
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


def analyze_real_duplicates():
    """Analyze duplicate report focusing on meaningful duplicates"""
    
    # Load the duplicate report
    with open('duplicate_analysis_report.json', 'r') as f:
        report = json.load(f)
    
    # Filter out empty files and categorize
    real_duplicates = []
    init_file_duplicates = []
    backend_duplicates = []
    test_duplicates = []
    archive_duplicates = []
    
    for dup_group in report['exact_duplicates']:
        # Skip empty files
        if dup_group['size'] == 0:
            # Check if they're __init__.py files
            if any('__init__.py' in f for f in dup_group['files']):
                init_file_duplicates.append(dup_group)
            continue
        
        # Categorize real duplicates
        files = dup_group['files']
        
        # Check categories
        has_backend = any('backend' in f for f in files)
        has_test = any('test' in f.lower() for f in files)
        has_archive = any('archive' in f for f in files)
        
        if has_backend and not has_archive:
            backend_duplicates.append(dup_group)
        elif has_test:
            test_duplicates.append(dup_group)
        elif has_archive:
            archive_duplicates.append(dup_group)
        else:
            real_duplicates.append(dup_group)
    
    # Print analysis
    logger.info("\n" + "="*60)
    logger.info("REAL DUPLICATE ANALYSIS (Excluding Empty Files)")
    logger.info("="*60)
    
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"- Empty __init__.py duplicates: {len(init_file_duplicates)} groups (NORMAL - ignore)")
    logger.info(f"- Backend duplicates: {len(backend_duplicates)} groups (CRITICAL)")
    logger.info(f"- Test duplicates: {len(test_duplicates)} groups (MEDIUM)")
    logger.info(f"- Archive duplicates: {len(archive_duplicates)} groups (LOW)")
    logger.info(f"- Other duplicates: {len(real_duplicates)} groups")
    
    # Show critical backend duplicates
    if backend_duplicates:
        logger.info(f"\n🔴 CRITICAL: Backend Duplicates")
        logger.info("-" * 40)
        for i, dup in enumerate(backend_duplicates, 1):
            size_kb = dup['size'] / 1024
            logger.info(f"\nDuplicate Group {i} ({size_kb:.1f} KB):")
            for file in dup['files']:
                logger.info(f"  - {file}")
    
    # Show other concerning duplicates
    if real_duplicates:
        logger.info(f"\n🟡 OTHER Duplicates (Non-archive)")
        logger.info("-" * 40)
        for i, dup in enumerate(real_duplicates[:5], 1):  # Show first 5
            size_kb = dup['size'] / 1024
            logger.info(f"\nDuplicate Group {i} ({size_kb:.1f} KB):")
            for file in dup['files'][:5]:  # Show max 5 files
                logger.info(f"  - {file}")
            if len(dup['files']) > 5:
                logger.info(f"  ... and {len(dup['files']) - 5} more")
    
    # Generate migration recommendations
    logger.info(f"\n📋 RECOMMENDED ACTIONS:")
    logger.info("-" * 40)
    
    if backend_duplicates:
        logger.info("\n1. BACKEND DUPLICATES (Immediate Action Required):")
        for dup in backend_duplicates[:3]:  # Show first 3
            files = [f for f in dup['files'] if 'archive' not in f]
            if len(files) > 1:
                logger.info(f"\n   Keep: {files[0]}")
                for f in files[1:]:
                    logger.info(f"   Archive: {f}")
    
    return {
        'backend_duplicates': backend_duplicates,
        'test_duplicates': test_duplicates,
        'archive_duplicates': archive_duplicates,
        'other_duplicates': real_duplicates
    }


if __name__ == '__main__':
    analyze_real_duplicates() 