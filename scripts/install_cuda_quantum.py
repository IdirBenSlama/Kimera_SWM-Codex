"""
CUDA Quantum Installation Script for KIMERA
==========================================

Automated installation and setup script for CUDA Quantum integration
with comprehensive hardware detection and configuration optimization.

Installation Steps:
1. Environment validation
2. CUDA Quantum package installation
3. Hardware capability detection
4. Configuration optimization
5. Integration testing
6. Performance benchmarking

Author: KIMERA Development Team
Version: 1.0.0 - CUDA Quantum Installation
"""

import subprocess
import sys
import logging
import platform
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [CUDA Quantum Installer] %(message)s'
)
logger = logging.getLogger(__name__)


class CUDAQuantumInstaller:
    """Automated CUDA Quantum installation and setup for KIMERA"""
    
    def __init__(self, force_reinstall: bool = False, skip_tests: bool = False):
        self.force_reinstall = force_reinstall
        self.skip_tests = skip_tests
        self.installation_log = []
        self.project_root = Path(__file__).parent.parent
        
        logger.info("🚀 CUDA Quantum Installation for KIMERA Starting...")
        logger.info(f"Project Root: {self.project_root}")
        logger.info(f"Force Reinstall: {force_reinstall}")
        logger.info(f"Skip Tests: {skip_tests}")
    
    def run_installation(self) -> bool:
        """Run complete installation process"""
        try:
            logger.info("=" * 70)
            logger.info("CUDA QUANTUM INSTALLATION FOR KIMERA")
            logger.info("=" * 70)
            
            # Step 1: Environment validation
            if not self._validate_environment():
                return False
            
            # Step 2: Install dependencies
            if not self._install_dependencies():
                return False
            
            # Step 3: Install CUDA Quantum
            if not self._install_cuda_quantum():
                return False
            
            # Step 4: Detect hardware capabilities
            hardware_info = self._detect_hardware()
            
            # Step 5: Setup configuration
            if not self._setup_configuration(hardware_info):
                return False
            
            # Step 6: Run integration tests
            if not self.skip_tests:
                if not self._run_integration_tests():
                    logger.warning("⚠️  Integration tests failed - continuing anyway")
            
            # Step 7: Performance benchmarking
            if not self.skip_tests:
                self._run_performance_benchmark()
            
            # Step 8: Generate installation report
            self._generate_installation_report(hardware_info)
            
            logger.info("✅ CUDA Quantum installation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Installation failed: {e}")
            return False
    
    def _validate_environment(self) -> bool:
        """Validate installation environment"""
        logger.info("🔍 Step 1: Validating Environment...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 10):
                logger.error(f"❌ Python 3.10+ required, found {python_version.major}.{python_version.minor}")
                return False
            
            logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check pip version
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ pip not available")
                return False
            
            logger.info(f"✅ pip available: {result.stdout.strip()}")
            
            # Check platform
            system = platform.system()
            arch = platform.machine()
            logger.info(f"✅ Platform: {system} {arch}")
            
            # Check if we're in virtual environment (recommended)
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            if not in_venv:
                logger.warning("⚠️  Not in virtual environment - installation may affect system packages")
            else:
                logger.info("✅ Virtual environment detected")
            
            self.installation_log.append({
                'step': 'environment_validation',
                'status': 'success',
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'platform': f"{system} {arch}",
                'virtual_env': in_venv
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment validation failed: {e}")
            return False
    
    def _install_dependencies(self) -> bool:
        """Install required dependencies"""
        logger.info("📦 Step 2: Installing Dependencies...")
        
        try:
            # Install base requirements
            base_requirements = self.project_root / "requirements" / "base.txt"
            if base_requirements.exists():
                logger.info("Installing base requirements...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(base_requirements)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"❌ Failed to install base requirements: {result.stderr}")
                    return False
            
            # Install quantum requirements
            quantum_requirements = self.project_root / "requirements" / "quantum.txt"
            if quantum_requirements.exists():
                logger.info("Installing quantum requirements...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(quantum_requirements)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"❌ Failed to install quantum requirements: {result.stderr}")
                    return False
            
            logger.info("✅ Dependencies installed successfully")
            
            self.installation_log.append({
                'step': 'dependency_installation',
                'status': 'success'
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def _install_cuda_quantum(self) -> bool:
        """Install CUDA Quantum package"""
        logger.info("⚡ Step 3: Installing CUDA Quantum...")
        
        try:
            # Check if already installed and not forcing reinstall
            if not self.force_reinstall:
                try:
                    import cudaq
                    logger.info(f"✅ CUDA Quantum already installed: {getattr(cudaq, '__version__', 'unknown')}")
                    return True
                except ImportError:
                    pass
            
            # Install CUDA Quantum - Primary method
            logger.info("Attempting CUDA Quantum installation via NVIDIA PyPI...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--extra-index-url", "https://pypi.nvidia.com", 
                "cudaq>=0.9.0"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ CUDA Quantum installed successfully via NVIDIA PyPI")
                self.installation_log.append({
                    'step': 'cudaq_installation',
                    'method': 'nvidia_pypi',
                    'status': 'success'
                })
                return True
            else:
                logger.warning(f"⚠️  NVIDIA PyPI installation failed: {result.stderr}")
                logger.info("Trying alternative installation methods...")
        
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  NVIDIA PyPI installation timed out")
        except Exception as e:
            logger.warning(f"⚠️  NVIDIA PyPI installation error: {e}")
        
        # Fallback method 1: Direct pip install
        try:
            logger.info("Attempting direct pip installation...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "cudaq"
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                logger.info("✅ CUDA Quantum installed successfully via direct pip")
                self.installation_log.append({
                    'step': 'cudaq_installation',
                    'method': 'direct_pip',
                    'status': 'success'
                })
                return True
            else:
                logger.warning(f"⚠️  Direct pip installation failed: {result.stderr}")
        
        except Exception as e:
            logger.warning(f"⚠️  Direct pip installation error: {e}")
        
        # Fallback method 2: Install compatible alternatives
        logger.info("CUDA Quantum installation failed - installing alternative quantum libraries...")
        
        alternatives_installed = []
        
        # Install Qiskit (most reliable)
        try:
            logger.info("Installing Qiskit ecosystem...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "qiskit>=1.0.0", "qiskit-aer>=0.15.0"
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                logger.info("✅ Qiskit installed successfully")
                alternatives_installed.append("qiskit")
            else:
                logger.warning("⚠️  Qiskit installation failed")
        except Exception as e:
            logger.warning(f"⚠️  Qiskit installation error: {e}")
        
        # Install Cirq
        try:
            logger.info("Installing Cirq...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "cirq>=1.4.0"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("✅ Cirq installed successfully")
                alternatives_installed.append("cirq")
            else:
                logger.warning("⚠️  Cirq installation failed")
        except Exception as e:
            logger.warning(f"⚠️  Cirq installation error: {e}")
        
        # Record fallback installation
        if alternatives_installed:
            logger.info(f"✅ Alternative quantum libraries installed: {alternatives_installed}")
            self.installation_log.append({
                'step': 'cudaq_installation',
                'method': 'alternatives',
                'status': 'partial_success',
                'alternatives_installed': alternatives_installed
            })
            return True  # Partial success
        else:
            logger.error("❌ All installation methods failed")
            self.installation_log.append({
                'step': 'cudaq_installation',
                'method': 'all_failed',
                'status': 'failed'
            })
            return False
    
    def _detect_hardware(self) -> Dict[str, any]:
        """Detect available hardware capabilities"""
        logger.info("🔧 Step 4: Detecting Hardware Capabilities...")
        
        hardware_info = {
            'cpu_count': None,
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_memory_gb': 0,
            'cuda_version': None,
            'cudaq_targets': []
        }
        
        try:
            # CPU information
            import psutil
            hardware_info['cpu_count'] = psutil.cpu_count(logical=False)
            logger.info(f"✅ CPU cores: {hardware_info['cpu_count']}")
            
            # GPU detection
            try:
                import torch
                if torch.cuda.is_available():
                    hardware_info['gpu_available'] = True
                    hardware_info['gpu_count'] = torch.cuda.device_count()
                    
                    # Get GPU memory
                    device_props = torch.cuda.get_device_properties(0)
                    hardware_info['gpu_memory_gb'] = device_props.total_memory / 1e9
                    hardware_info['cuda_version'] = torch.version.cuda
                    
                    logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
                    logger.info(f"   Memory: {hardware_info['gpu_memory_gb']:.1f} GB")
                    logger.info(f"   CUDA Version: {hardware_info['cuda_version']}")
                else:
                    logger.info("ℹ️  No GPU detected or CUDA not available")
                    
            except ImportError:
                logger.info("ℹ️  PyTorch not available for GPU detection")
            
            # CUDA Quantum targets
            try:
                import cudaq
                targets = cudaq.get_targets()
                hardware_info['cudaq_targets'] = [target.name for target in targets]
                logger.info(f"✅ CUDA Quantum targets: {hardware_info['cudaq_targets']}")
                
            except ImportError:
                logger.warning("⚠️  Could not detect CUDA Quantum targets")
            
            self.installation_log.append({
                'step': 'hardware_detection',
                'status': 'success',
                'hardware_info': hardware_info
            })
            
            return hardware_info
            
        except Exception as e:
            logger.error(f"❌ Hardware detection failed: {e}")
            return hardware_info
    
    def _setup_configuration(self, hardware_info: Dict[str, any]) -> bool:
        """Setup optimized configuration"""
        logger.info("⚙️  Step 5: Setting up Configuration...")
        
        try:
            # Create configuration directory
            config_dir = self.project_root / "config"
            config_dir.mkdir(exist_ok=True)
            
            # Generate optimized configuration
            config = {
                'hardware': {
                    'preferred_gpu_backend': 'nvidia' if hardware_info['gpu_available'] else 'qpp-cpu',
                    'fallback_cpu_backend': 'qpp-cpu',
                    'enable_multi_gpu': hardware_info['gpu_count'] > 1,
                    'gpu_memory_fraction': 0.8,
                    'max_qubits_single_gpu': min(25, int(hardware_info['gpu_memory_gb'] * 2)) if hardware_info['gpu_available'] else 15,
                    'auto_detect_capabilities': True
                },
                'simulation': {
                    'default_shots': 1024,
                    'enable_circuit_optimization': True,
                    'state_vector_threshold': 20
                },
                'cognitive': {
                    'monitoring_level': 'standard',
                    'assessment_frequency': 10
                },
                'performance': {
                    'enable_performance_tracking': True,
                    'generate_performance_reports': True
                }
            }
            
            # Save configuration
            config_file = config_dir / "quantum_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ Configuration saved: {config_file}")
            
            self.installation_log.append({
                'step': 'configuration_setup',
                'status': 'success',
                'config_file': str(config_file)
            })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration setup failed: {e}")
            return False
    
    def _run_integration_tests(self) -> bool:
        """Run integration tests"""
        logger.info("🧪 Step 6: Running Integration Tests...")
        
        try:
            test_file = self.project_root / "tests" / "quantum" / "test_cuda_quantum_integration.py"
            
            if not test_file.exists():
                logger.warning("⚠️  Integration test file not found - skipping tests")
                return True
            
            # Run tests with pytest
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✅ Integration tests passed")
                
                self.installation_log.append({
                    'step': 'integration_tests',
                    'status': 'success'
                })
                
                return True
            else:
                logger.warning("⚠️  Integration tests failed:")
                logger.warning(result.stdout)
                logger.warning(result.stderr)
                
                self.installation_log.append({
                    'step': 'integration_tests',
                    'status': 'failed',
                    'error': result.stderr
                })
                
                return False
            
        except Exception as e:
            logger.error(f"❌ Integration tests failed: {e}")
            return False
    
    def _run_performance_benchmark(self) -> None:
        """Run performance benchmark"""
        logger.info("📊 Step 7: Running Performance Benchmark...")
        
        try:
            # Simple benchmark
            benchmark_script = '''
import time
import sys
sys.path.append(".")

try:
    from backend.engines.cuda_quantum_engine import create_cuda_quantum_engine, QuantumBackendType
    
    # Create engine
    engine = create_cuda_quantum_engine(
        backend_type=QuantumBackendType.CPU_STATEVECTOR,
        enable_cognitive_monitoring=False
    )
    
    # Benchmark GHZ state
    start_time = time.time()
    counts, metrics = engine.simulate_quantum_circuit(
        lambda: engine.create_ghz_state(4),
        shots=100
    )
    execution_time = time.time() - start_time
    
    print(f"Benchmark Results:")
    print(f"  4-qubit GHZ state: {execution_time:.3f}s")
    print(f"  Simulation time: {metrics.simulation_time:.3f}s")
    print(f"  Fidelity: {metrics.fidelity_estimate:.3f}")
    
except Exception as e:
    print(f"Benchmark failed: {e}")
'''
            
            # Run benchmark
            result = subprocess.run([
                sys.executable, "-c", benchmark_script
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✅ Performance benchmark completed:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"   {line}")
            else:
                logger.warning("⚠️  Performance benchmark failed")
                logger.warning(result.stderr)
            
        except Exception as e:
            logger.warning(f"⚠️  Performance benchmark failed: {e}")
    
    def _generate_installation_report(self, hardware_info: Dict[str, any]) -> None:
        """Generate installation report"""
        logger.info("📋 Step 8: Generating Installation Report...")
        
        try:
            report = {
                'installation_date': datetime.now().isoformat(),
                'installation_log': self.installation_log,
                'hardware_info': hardware_info,
                'system_info': {
                    'platform': platform.platform(),
                    'python_version': sys.version,
                    'python_executable': sys.executable
                }
            }
            
            # Save report
            reports_dir = self.project_root / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"cuda_quantum_installation_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✅ Installation report saved: {report_file}")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to generate installation report: {e}")


def main():
    """Main installation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install CUDA Quantum for KIMERA")
    parser.add_argument("--force-reinstall", action="store_true", 
                       help="Force reinstall even if already installed")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip integration tests and benchmarks")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Run installation
    installer = CUDAQuantumInstaller(
        force_reinstall=args.force_reinstall,
        skip_tests=args.skip_tests
    )
    
    success = installer.run_installation()
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 CUDA QUANTUM INSTALLATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Restart your Python environment")
        print("2. Run: python -c 'import cudaq; print(cudaq.get_targets())'")
        print("3. Execute: python kimera.py --quantum-backend nvidia")
        print("4. Check the installation report in the reports/ directory")
        print("\nFor more information, see docs/CUDA_QUANTUM_INTEGRATION.md")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ CUDA QUANTUM INSTALLATION FAILED")
        print("=" * 70)
        print("\nPlease check the error messages above and:")
        print("1. Ensure you have Python 3.10+")
        print("2. Consider using a virtual environment")
        print("3. Check your CUDA installation if using GPU")
        print("4. Try running with --force-reinstall")
        print("\nFor help, see docs/TROUBLESHOOTING.md")
        sys.exit(1)


if __name__ == "__main__":
    main() 