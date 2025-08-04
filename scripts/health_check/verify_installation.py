#!/usr/bin/env python3
# Fix import paths
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


"""
KIMERA SWM Installation Verification Script
Following KIMERA Protocol v3.0 - Empirical Verification
"""

import sys
import importlib
from datetime import datetime
from typing import List, Tuple

def verify_package_imports() -> Tuple[List[str], List[str]]:
    """Verify critical package imports following zero-trust principles"""

    critical_packages = [
        ('fastapi', 'FastAPI web framework'),
        ('torch', 'PyTorch ML framework'),
        ('numpy', 'Scientific computing'),
        ('scipy', 'Advanced scientific computing'),
        ('qiskit', 'Quantum computing'),
        ('sklearn', 'Machine learning'),
        ('pandas', 'Data manipulation'),
        ('transformers', 'Transformer models'),
        ('sqlalchemy', 'Database ORM'),
        ('redis', 'In-memory database'),
        ('neo4j', 'Graph database'),
        ('pytest', 'Testing framework'),
        ('black', 'Code formatter'),
        ('mypy', 'Type checker'),
        ('yaml', 'YAML parser'),
        ('loguru', 'Advanced logging'),
        ('requests', 'HTTP library'),
        ('dotenv', 'Environment variables')
    ]

    successful = []
    failed = []

    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting import verification...")

    for package, description in critical_packages:
        try:
            importlib.import_module(package)
            successful.append(f"✅ {package} - {description}")
            logger.info(f"✅ {package}")
        except ImportError as e:
            failed.append(f"❌ {package} - {description} - Error: {str(e)}")
            logger.info(f"❌ {package} - {str(e)}")
        except Exception as e:
            failed.append(f"⚠️ {package} - {description} - Unexpected error: {str(e)}")
            logger.info(f"⚠️ {package} - {str(e)}")

    return successful, failed

def test_basic_functionality():
    """Test basic functionality of key packages"""
    logger.info(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing basic functionality...")

    try:
        import numpy as np
        import torch
        import pandas as pd
import logging
logger = logging.getLogger(__name__)

        # Test numpy
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
        logger.info("✅ NumPy basic operations")

        # Test torch
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.sum().item() == 6.0
        logger.info("✅ PyTorch basic operations")

        # Test pandas
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        logger.info("✅ Pandas basic operations")

        return True

    except Exception as e:
        logger.info(f"❌ Functionality test failed: {str(e)}")
        return False

def main():
    """Main verification process"""
    logger.info("="*60)
    logger.info("KIMERA SWM Installation Verification")
    logger.info("Following KIMERA Protocol v3.0 - Zero Trust Verification")
    logger.info("="*60)

    # Step 1: Import verification
    successful, failed = verify_package_imports()

    # Step 2: Functionality testing
    functionality_ok = test_basic_functionality()

    # Step 3: Generate summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Successful imports: {len(successful)}")
    logger.info(f"❌ Failed imports: {len(failed)}")
    logger.info(f"🔧 Functionality test: {'PASSED' if functionality_ok else 'FAILED'}")

    if failed:
        logger.info("\nFAILED IMPORTS:")
        for failure in failed:
            logger.info(f"  {failure}")

    # Step 4: Overall assessment
    if len(failed) == 0 and functionality_ok:
        logger.info("\n🎉 ALL VERIFICATIONS PASSED - KIMERA SWM is ready for operation!")
        return 0
    else:
        logger.info(f"\n⚠️ PARTIAL SUCCESS - {len(failed)} packages failed, functionality {'failed' if not functionality_ok else 'passed'}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
