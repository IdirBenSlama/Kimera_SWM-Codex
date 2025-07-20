#!/usr/bin/env python3
"""
Comprehensive Library Audit for KIMERA Phase 1 and Scientific Validation
========================================================================

Checks all required libraries for:
1. Phase 1 GPU Foundation 
2. Scientific Validation Framework
3. Benchmark and Testing Libraries
"""

import sys
import subprocess
import importlib.util

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


def check_library(lib_name, import_name=None):
    """Check if a library is installed."""
    if import_name is None:
        import_name = lib_name.replace('-', '_')
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            return True, "installed"
        else:
            return False, "not found"
    except ImportError as e:
        return False, f"import error: {e}"

def main():
    logger.info("=" * 80)
    logger.info("KIMERA COMPREHENSIVE LIBRARY AUDIT")
    logger.info("=" * 80)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Executable: {sys.executable}")
    logger.info()

    # Phase 1 GPU Foundation Libraries
    logger.info("=== PHASE 1 GPU FOUNDATION LIBRARIES ===")
    gpu_libs = [
        ('cupy-cuda11x', 'cupy'),
        ('rapids-cudf', 'cudf'), 
        ('triton', 'triton'),
        ('qiskit-aer-gpu', 'qiskit_aer'),
        ('numba', 'numba'),
        ('torch', 'torch'),
        ('tensorflow-gpu', 'tensorflow')
    ]
    
    gpu_missing = []
    gpu_installed = []
    
    for lib_name, import_name in gpu_libs:
        installed, status = check_library(lib_name, import_name)
        if installed:
            gpu_installed.append(lib_name)
            logger.info(f"✅ {lib_name}")
        else:
            gpu_missing.append(lib_name)
            logger.error(f"❌ {lib_name} - {status}")
    
    # Scientific Validation Libraries
    logger.info("\n=== SCIENTIFIC VALIDATION LIBRARIES ===")
    sci_libs = [
        ('pytest', 'pytest'),
        ('hypothesis', 'hypothesis'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('networkx', 'networkx'),
        ('sympy', 'sympy'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('jupyter', 'jupyter')
    ]
    
    sci_missing = []
    sci_installed = []
    
    for lib_name, import_name in sci_libs:
        installed, status = check_library(lib_name, import_name)
        if installed:
            sci_installed.append(lib_name)
            logger.info(f"✅ {lib_name}")
        else:
            sci_missing.append(lib_name)
            logger.error(f"❌ {lib_name} - {status}")
    
    # Benchmark and Performance Libraries
    logger.info("\n=== BENCHMARK & PERFORMANCE LIBRARIES ===")
    bench_libs = [
        ('memory-profiler', 'memory_profiler'),
        ('psutil', 'psutil'),
        ('line-profiler', 'line_profiler'),
        ('py-spy', 'py_spy'),
        ('cProfile', 'cProfile'),
        ('timeit', 'timeit'),
        ('concurrent.futures', 'concurrent.futures')
    ]
    
    bench_missing = []
    bench_installed = []
    
    for lib_name, import_name in bench_libs:
        installed, status = check_library(lib_name, import_name)
        if installed:
            bench_installed.append(lib_name)
            logger.info(f"✅ {lib_name}")
        else:
            bench_missing.append(lib_name)
            logger.error(f"❌ {lib_name} - {status}")
    
    # AI/ML Specific Libraries
    logger.info("\n=== AI/ML SPECIFIC LIBRARIES ===")
    ai_libs = [
        ('transformers', 'transformers'),
        ('sentence-transformers', 'sentence_transformers'),
        ('openai', 'openai'),
        ('anthropic', 'anthropic'),
        ('langchain', 'langchain'),
        ('faiss-cpu', 'faiss'),
        ('chromadb', 'chromadb')
    ]
    
    ai_missing = []
    ai_installed = []
    
    for lib_name, import_name in ai_libs:
        installed, status = check_library(lib_name, import_name)
        if installed:
            ai_installed.append(lib_name)
            logger.info(f"✅ {lib_name}")
        else:
            ai_missing.append(lib_name)
            logger.error(f"❌ {lib_name} - {status}")
    
    # Summary
    total_missing = len(gpu_missing) + len(sci_missing) + len(bench_missing) + len(ai_missing)
    total_installed = len(gpu_installed) + len(sci_installed) + len(bench_installed) + len(ai_installed)
    
    logger.info("\n" + "=" * 80)
    logger.info("LIBRARY AUDIT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"📊 Total Libraries Checked: {total_installed + total_missing}")
    logger.info(f"✅ Installed: {total_installed}")
    logger.error(f"❌ Missing: {total_missing}")
    logger.info(f"📈 Installation Rate: {total_installed/(total_installed + total_missing)
    
    logger.debug(f"\n🔧 CRITICAL MISSING LIBRARIES:")
    critical_missing = []
    
    # Phase 1 GPU libraries are critical
    for lib in gpu_missing:
        if lib in ['cupy-cuda11x', 'triton', 'qiskit-aer-gpu']:
            critical_missing.append(lib)
            logger.info(f"  🚨 {lib} (Phase 1 GPU Foundation)
    
    # Scientific validation libraries are critical  
    for lib in sci_missing:
        if lib in ['pytest', 'hypothesis', 'scipy', 'scikit-learn']:
            critical_missing.append(lib)
            logger.info(f"  🚨 {lib} (Scientific Validation)
    
    if critical_missing:
        logger.warning(f"\n⚠️ WARNING: {len(critical_missing)
        logger.info("Phase 1 Week 1 GPU Foundation is INCOMPLETE without these libraries.")
        logger.info("Scientific validation framework cannot run without these libraries.")
    else:
        logger.info(f"\n🎉 All critical libraries are installed!")
    
    # Generate installation commands
    all_missing = gpu_missing + sci_missing + bench_missing + ai_missing
    if all_missing:
        logger.info(f"\n📝 INSTALLATION COMMANDS:")
        logger.info("Run these commands to install missing libraries:")
        logger.info()
        
        # GPU libraries (special handling)
        gpu_install_cmds = []
        if 'cupy-cuda11x' in gpu_missing:
            gpu_install_cmds.append("pip install cupy-cuda11x")
        if 'rapids-cudf' in gpu_missing:
            gpu_install_cmds.append("pip install rapids-cudf")
        if 'triton' in gpu_missing:
            gpu_install_cmds.append("pip install triton")
        if 'qiskit-aer-gpu' in gpu_missing:
            gpu_install_cmds.append("pip install qiskit-aer-gpu")
        
        # Regular libraries
        regular_missing = [lib for lib in all_missing if lib not in ['cupy-cuda11x', 'rapids-cudf', 'triton', 'qiskit-aer-gpu']]
        
        if gpu_install_cmds:
            logger.info("# GPU Libraries (install separately)
            for cmd in gpu_install_cmds:
                logger.info(f"  {cmd}")
            logger.info()
        
        if regular_missing:
            logger.info("# Regular Libraries (can install together)
            logger.info(f"  pip install {' '.join(regular_missing)
    
    return total_missing == 0

if __name__ == "__main__":
    success = main()
    if not success:
        logger.debug(f"\n🔧 Action Required: Install missing libraries to complete Phase 1 and enable scientific validation.")
    else:
        logger.info(f"\n✅ All libraries installed - Phase 1 GPU Foundation and Scientific Validation ready!")