#!/usr/bin/env python3
"""
KIMERA SWM - GPU ACCELERATION SETUP SCRIPT
==========================================

Comprehensive GPU setup and configuration script for Kimera SWM.
Handles CUDA detection, dependency installation, and GPU optimization.

Features:
- Automatic CUDA detection and configuration
- GPU dependency installation
- Hardware capability assessment
- Performance optimization
- Troubleshooting and diagnostics
"""

import os
import sys
import subprocess
import logging
import platform
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import shutil
import urllib.request
import warnings

# Suppress warnings during setup
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUSetupManager:
    """GPU acceleration setup and configuration manager"""
    
    def __init__(self):
        self.system_info = {}
        self.gpu_info = {}
        self.cuda_info = {}
        self.requirements_installed = False
        
        # Supported CUDA versions
        self.supported_cuda_versions = ['11.8', '12.0', '12.1', '12.2', '12.3']
        
        # GPU requirements
        self.min_gpu_memory_gb = 4
        self.recommended_gpu_memory_gb = 8
        
    def run_setup(self) -> bool:
        """Run complete GPU setup process"""
        logger.info("🚀 Starting Kimera SWM GPU Acceleration Setup")
        logger.info("=" * 60)
        
        try:
            # Step 1: System detection
            logger.info("📊 Step 1: System Detection")
            self._detect_system_info()
            self._print_system_info()
            
            # Step 2: GPU detection
            logger.info("\n🔍 Step 2: GPU Hardware Detection")
            self._detect_gpu_hardware()
            
            # Step 3: CUDA detection
            logger.info("\n⚙️ Step 3: CUDA Environment Detection")
            self._detect_cuda_environment()
            
            # Step 4: Assess compatibility
            logger.info("\n🔬 Step 4: Compatibility Assessment")
            compatibility = self._assess_compatibility()
            
            if not compatibility['gpu_suitable']:
                logger.warning("⚠️ GPU hardware not suitable for acceleration")
                return self._setup_cpu_fallback()
            
            # Step 5: Install dependencies
            logger.info("\n📦 Step 5: Installing GPU Dependencies")
            if not self._install_gpu_dependencies():
                logger.error("❌ Failed to install GPU dependencies")
                return False
            
            # Step 6: Configure environment
            logger.info("\n🔧 Step 6: Environment Configuration")
            self._configure_gpu_environment()
            
            # Step 7: Verify installation
            logger.info("\n✅ Step 7: Installation Verification")
            verification = self._verify_gpu_setup()
            
            # Step 8: Performance optimization
            if verification['success']:
                logger.info("\n⚡ Step 8: Performance Optimization")
                self._optimize_gpu_performance()
            
            # Step 9: Generate report
            logger.info("\n📋 Step 9: Setup Report Generation")
            self._generate_setup_report(verification)
            
            logger.info("\n🎉 GPU Acceleration Setup Complete!")
            return verification['success']
            
        except Exception as e:
            logger.error(f"❌ GPU setup failed: {e}")
            return False
    
    def _detect_system_info(self) -> None:
        """Detect system information"""
        self.system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': sys.version,
            'python_executable': sys.executable
        }
        
        # Detect package manager
        if shutil.which('conda'):
            self.system_info['package_manager'] = 'conda'
        elif shutil.which('pip'):
            self.system_info['package_manager'] = 'pip'
        else:
            self.system_info['package_manager'] = 'unknown'
    
    def _print_system_info(self) -> None:
        """Print system information"""
        logger.info(f"   Operating System: {self.system_info['platform']}")
        logger.info(f"   Architecture: {self.system_info['architecture']}")
        logger.info(f"   Python Version: {sys.version.split()[0]}")
        logger.info(f"   Package Manager: {self.system_info['package_manager']}")
    
    def _detect_gpu_hardware(self) -> None:
        """Detect GPU hardware information"""
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                self.gpu_info['gpus'] = []
                
                for i, line in enumerate(lines):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpu = {
                            'id': i,
                            'name': parts[0],
                            'memory_mb': int(parts[1]),
                            'memory_gb': round(int(parts[1]) / 1024, 1),
                            'driver_version': parts[2],
                            'compute_capability': parts[3]
                        }
                        self.gpu_info['gpus'].append(gpu)
                
                self.gpu_info['nvidia_driver_available'] = True
                logger.info(f"✅ Found {len(self.gpu_info['gpus'])} NVIDIA GPU(s)")
                
                for gpu in self.gpu_info['gpus']:
                    logger.info(f"   GPU {gpu['id']}: {gpu['name']}")
                    logger.info(f"      Memory: {gpu['memory_gb']}GB")
                    logger.info(f"      Driver: {gpu['driver_version']}")
                    logger.info(f"      Compute: {gpu['compute_capability']}")
            else:
                self.gpu_info['nvidia_driver_available'] = False
                logger.warning("⚠️ nvidia-smi not available or failed")
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            self.gpu_info['nvidia_driver_available'] = False
            logger.warning("⚠️ NVIDIA drivers not detected")
        
        # Alternative GPU detection methods
        if not self.gpu_info.get('nvidia_driver_available'):
            self._detect_gpu_alternative_methods()
    
    def _detect_gpu_alternative_methods(self) -> None:
        """Try alternative GPU detection methods"""
        logger.info("🔍 Trying alternative GPU detection methods...")
        
        # Try Windows WMIC
        if platform.system() == "Windows":
            try:
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    gpu_names = [line.strip() for line in lines[1:] if line.strip()]
                    
                    if any('NVIDIA' in name for name in gpu_names):
                        logger.info("✅ NVIDIA GPU detected via Windows WMIC")
                        self.gpu_info['gpus'] = [{'name': name, 'detected_via': 'wmic'} for name in gpu_names if 'NVIDIA' in name]
                    else:
                        logger.info("ℹ️ No NVIDIA GPUs found via Windows WMIC")
            except Exception:
                pass
        
        # Try lspci on Linux
        elif platform.system() == "Linux":
            try:
                result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.lower()
                    if 'nvidia' in lines:
                        logger.info("✅ NVIDIA GPU detected via lspci")
                        self.gpu_info['detected_via_lspci'] = True
                    else:
                        logger.info("ℹ️ No NVIDIA GPUs found via lspci")
            except Exception:
                pass
    
    def _detect_cuda_environment(self) -> None:
        """Detect CUDA environment and installation"""
        self.cuda_info = {
            'cuda_available': False,
            'cuda_version': None,
            'cuda_path': None,
            'cudnn_available': False
        }
        
        # Check for CUDA_PATH environment variable
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and os.path.exists(cuda_path):
            self.cuda_info['cuda_path'] = cuda_path
            logger.info(f"✅ CUDA_PATH found: {cuda_path}")
        
        # Try nvcc command
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                output = result.stdout
                # Extract CUDA version
                for line in output.split('\n'):
                    if 'release' in line.lower():
                        import re
                        version_match = re.search(r'V(\d+\.\d+)', line)
                        if version_match:
                            self.cuda_info['cuda_version'] = version_match.group(1)
                            self.cuda_info['cuda_available'] = True
                            logger.info(f"✅ CUDA {self.cuda_info['cuda_version']} detected via nvcc")
                            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ nvcc command not found")
        
        # Check for common CUDA installation paths
        if not self.cuda_info['cuda_available']:
            self._check_cuda_installation_paths()
        
        # Check for cuDNN
        self._check_cudnn_availability()
    
    def _check_cuda_installation_paths(self) -> None:
        """Check common CUDA installation paths"""
        common_paths = []
        
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                r"C:\tools\cuda"
            ]
        else:
            common_paths = [
                "/usr/local/cuda",
                "/opt/cuda",
                "/usr/cuda"
            ]
        
        for base_path in common_paths:
            if os.path.exists(base_path):
                # Look for version directories
                try:
                    items = os.listdir(base_path)
                    version_dirs = [item for item in items if item.startswith('v') or item.replace('.', '').isdigit()]
                    
                    if version_dirs:
                        latest_version = sorted(version_dirs)[-1]
                        cuda_path = os.path.join(base_path, latest_version)
                        
                        if os.path.exists(os.path.join(cuda_path, 'bin')):
                            self.cuda_info['cuda_path'] = cuda_path
                            self.cuda_info['cuda_available'] = True
                            
                            # Extract version from directory name
                            version = latest_version.replace('v', '').replace('_', '.')
                            self.cuda_info['cuda_version'] = version
                            
                            logger.info(f"✅ CUDA found at: {cuda_path}")
                            break
                except Exception:
                    continue
    
    def _check_cudnn_availability(self) -> None:
        """Check for cuDNN availability"""
        if self.cuda_info['cuda_path']:
            cudnn_header = os.path.join(self.cuda_info['cuda_path'], 'include', 'cudnn.h')
            if os.path.exists(cudnn_header):
                self.cuda_info['cudnn_available'] = True
                logger.info("✅ cuDNN detected")
            else:
                logger.info("ℹ️ cuDNN not found")
    
    def _assess_compatibility(self) -> Dict[str, Any]:
        """Assess GPU and CUDA compatibility"""
        compatibility = {
            'gpu_suitable': False,
            'cuda_suitable': False,
            'memory_sufficient': False,
            'compute_capability_ok': False,
            'driver_compatible': False,
            'overall_compatible': False,
            'recommendations': []
        }
        
        # Check GPU suitability
        if self.gpu_info.get('gpus'):
            best_gpu = max(self.gpu_info['gpus'], key=lambda x: x.get('memory_gb', 0))
            
            # Memory check
            if best_gpu.get('memory_gb', 0) >= self.min_gpu_memory_gb:
                compatibility['memory_sufficient'] = True
                if best_gpu.get('memory_gb', 0) >= self.recommended_gpu_memory_gb:
                    logger.info(f"✅ GPU memory sufficient: {best_gpu['memory_gb']}GB")
                else:
                    logger.info(f"⚠️ GPU memory meets minimum but below recommended: {best_gpu['memory_gb']}GB")
                    compatibility['recommendations'].append("Consider upgrading to GPU with 8GB+ memory for optimal performance")
            else:
                logger.warning(f"❌ Insufficient GPU memory: {best_gpu.get('memory_gb', 0)}GB < {self.min_gpu_memory_gb}GB required")
                compatibility['recommendations'].append("Upgrade to GPU with at least 4GB memory")
            
            # Compute capability check
            compute_cap = best_gpu.get('compute_capability', '0.0')
            if compute_cap:
                major_version = float(compute_cap.split('.')[0]) if '.' in compute_cap else 0
                if major_version >= 6.0:
                    compatibility['compute_capability_ok'] = True
                    logger.info(f"✅ Compute capability sufficient: {compute_cap}")
                else:
                    logger.warning(f"❌ Compute capability too low: {compute_cap} < 6.0 required")
                    compatibility['recommendations'].append("Upgrade to GPU with compute capability 6.0+")
            
            compatibility['gpu_suitable'] = compatibility['memory_sufficient'] and compatibility['compute_capability_ok']
        else:
            logger.warning("❌ No suitable NVIDIA GPU detected")
            compatibility['recommendations'].append("Install NVIDIA GPU with compute capability 6.0+ and 4GB+ memory")
        
        # Check CUDA compatibility
        if self.cuda_info['cuda_available']:
            cuda_version = self.cuda_info['cuda_version']
            if cuda_version in self.supported_cuda_versions or any(cuda_version.startswith(v) for v in self.supported_cuda_versions):
                compatibility['cuda_suitable'] = True
                logger.info(f"✅ CUDA version compatible: {cuda_version}")
            else:
                logger.warning(f"⚠️ CUDA version may not be optimal: {cuda_version}")
                compatibility['recommendations'].append(f"Consider upgrading to CUDA {self.supported_cuda_versions[-1]}")
                compatibility['cuda_suitable'] = True  # Still try to proceed
        else:
            logger.warning("❌ CUDA not detected")
            compatibility['recommendations'].append("Install CUDA Toolkit from NVIDIA developer website")
        
        # Driver compatibility
        if self.gpu_info.get('nvidia_driver_available'):
            compatibility['driver_compatible'] = True
            logger.info("✅ NVIDIA drivers available")
        else:
            logger.warning("❌ NVIDIA drivers not available")
            compatibility['recommendations'].append("Install latest NVIDIA drivers")
        
        compatibility['overall_compatible'] = (
            compatibility['gpu_suitable'] and 
            compatibility['driver_compatible']
        )
        
        return compatibility
    
    def _install_gpu_dependencies(self) -> bool:
        """Install GPU dependencies"""
        try:
            # Get current directory
            current_dir = Path(__file__).parent.parent
            gpu_requirements = current_dir / "requirements" / "gpu.txt"
            
            if not gpu_requirements.exists():
                logger.error(f"❌ GPU requirements file not found: {gpu_requirements}")
                return False
            
            logger.info(f"📦 Installing dependencies from {gpu_requirements}")
            
            # Install GPU requirements
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(gpu_requirements)]
            
            logger.info("🔄 Installing PyTorch with CUDA support...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                logger.info("✅ GPU dependencies installed successfully")
                self.requirements_installed = True
                return True
            else:
                logger.error(f"❌ Failed to install GPU dependencies:")
                logger.error(result.stderr)
                
                # Try alternative PyTorch installation
                return self._install_pytorch_alternative()
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Installation timeout - trying alternative approach")
            return self._install_pytorch_alternative()
        except Exception as e:
            logger.error(f"❌ Installation failed: {e}")
            return False
    
    def _install_pytorch_alternative(self) -> bool:
        """Try alternative PyTorch installation"""
        logger.info("🔄 Trying alternative PyTorch installation...")
        
        # Determine appropriate PyTorch installation command
        cuda_version = self.cuda_info.get('cuda_version', '12.1')
        
        if cuda_version.startswith('11.8'):
            index_url = "https://download.pytorch.org/whl/cu118"
        elif cuda_version.startswith('12.1'):
            index_url = "https://download.pytorch.org/whl/cu121"
        else:
            # Default to CPU version if CUDA version unknown
            index_url = "https://download.pytorch.org/whl/cpu"
            logger.warning("⚠️ Unknown CUDA version, installing CPU-only PyTorch")
        
        try:
            cmd = [
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'torchaudio',
                '--index-url', index_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✅ PyTorch installed successfully with alternative method")
                
                # Install other GPU dependencies
                other_deps = ['cupy-cuda12x>=13.0.0', 'cuda-python>=12.0.0', 'GPUtil>=1.4.0']
                for dep in other_deps:
                    try:
                        subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                                     capture_output=True, timeout=120)
                    except Exception:
                        logger.warning(f"⚠️ Failed to install {dep}")
                
                return True
            else:
                logger.error("❌ Alternative PyTorch installation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Alternative installation failed: {e}")
            return False
    
    def _configure_gpu_environment(self) -> None:
        """Configure GPU environment variables"""
        env_vars = {}
        
        # Set CUDA_PATH if detected
        if self.cuda_info.get('cuda_path'):
            env_vars['CUDA_PATH'] = self.cuda_info['cuda_path']
        
        # Set CUDA device order
        env_vars['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        # Set PyTorch CUDA memory management
        env_vars['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Apply environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"🔧 Set {key}={value}")
        
        # Save environment configuration
        self._save_environment_config(env_vars)
    
    def _save_environment_config(self, env_vars: Dict[str, str]) -> None:
        """Save environment configuration to file"""
        config_dir = Path(__file__).parent.parent / "config"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "gpu_environment.json"
        
        config = {
            'gpu_environment_variables': env_vars,
            'setup_timestamp': time.time(),
            'gpu_info': self.gpu_info,
            'cuda_info': self.cuda_info
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"💾 Environment configuration saved to {config_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save environment config: {e}")
    
    def _verify_gpu_setup(self) -> Dict[str, Any]:
        """Verify GPU setup and installation"""
        verification = {
            'success': False,
            'pytorch_cuda': False,
            'cupy_available': False,
            'gpu_detected': False,
            'memory_available': 0,
            'compute_capability': None,
            'errors': []
        }
        
        try:
            # Test PyTorch CUDA
            import torch
            verification['pytorch_cuda'] = torch.cuda.is_available()
            
            if verification['pytorch_cuda']:
                verification['gpu_detected'] = True
                verification['memory_available'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                verification['compute_capability'] = torch.cuda.get_device_capability(0)
                logger.info(f"✅ PyTorch CUDA available")
                logger.info(f"   Device: {torch.cuda.get_device_name(0)}")
                logger.info(f"   Memory: {verification['memory_available']:.1f}GB")
                logger.info(f"   Compute: {verification['compute_capability']}")
            else:
                verification['errors'].append("PyTorch CUDA not available")
                logger.warning("❌ PyTorch CUDA not available")
                
        except ImportError as e:
            verification['errors'].append(f"PyTorch import failed: {e}")
            logger.error(f"❌ PyTorch import failed: {e}")
        
        try:
            # Test CuPy
            import cupy
            verification['cupy_available'] = True
            logger.info(f"✅ CuPy available: {cupy.__version__}")
        except ImportError as e:
            verification['errors'].append(f"CuPy import failed: {e}")
            logger.warning(f"⚠️ CuPy not available: {e}")
        
        verification['success'] = (
            verification['pytorch_cuda'] and 
            verification['gpu_detected'] and
            verification['memory_available'] >= self.min_gpu_memory_gb
        )
        
        return verification
    
    def _optimize_gpu_performance(self) -> None:
        """Optimize GPU performance settings"""
        logger.info("⚡ Applying GPU performance optimizations...")
        
        # Set optimal environment variables
        optimizations = {
            'TORCH_CUDNN_V8_API_ENABLED': '1',  # Enable cuDNN v8 API
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',  # Configure cuBLAS workspace
        }
        
        for key, value in optimizations.items():
            os.environ[key] = value
            logger.info(f"🔧 Applied optimization: {key}={value}")
        
        # Test performance
        self._benchmark_gpu_performance()
    
    def _benchmark_gpu_performance(self) -> None:
        """Run basic GPU performance benchmark"""
        try:
            import torch
            import time
            
            if not torch.cuda.is_available():
                return
            
            logger.info("🏃 Running GPU performance benchmark...")
            
            # Warm up
            device = torch.device('cuda')
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark matrix multiplication
            sizes = [1000, 2000, 4000]
            for size in sizes:
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                
                start_time = time.time()
                result = torch.matmul(a, b)
                torch.cuda.synchronize()
                end_time = time.time()
                
                gflops = (2 * size**3) / (end_time - start_time) / 1e9
                logger.info(f"   Matrix {size}x{size}: {gflops:.1f} GFLOPS")
            
        except Exception as e:
            logger.warning(f"⚠️ Benchmark failed: {e}")
    
    def _setup_cpu_fallback(self) -> bool:
        """Setup CPU fallback configuration"""
        logger.info("🔄 Setting up CPU fallback mode...")
        
        try:
            # Install CPU-only versions
            cmd = [
                sys.executable, '-m', 'pip', 'install',
                'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ CPU fallback mode configured")
                return True
            else:
                logger.error("❌ CPU fallback setup failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ CPU fallback setup failed: {e}")
            return False
    
    def _generate_setup_report(self, verification: Dict[str, Any]) -> None:
        """Generate comprehensive setup report"""
        report_dir = Path(__file__).parent.parent / "docs" / "reports" / "gpu"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        report_file = report_dir / f"{timestamp}_gpu_setup_report.md"
        
        report_content = f"""# KIMERA SWM GPU SETUP REPORT
**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Setup Status**: {'✅ SUCCESS' if verification['success'] else '❌ FAILED'}

## SYSTEM INFORMATION
- **Operating System**: {self.system_info.get('platform', 'Unknown')}
- **Architecture**: {self.system_info.get('architecture', 'Unknown')}
- **Python Version**: {sys.version.split()[0]}
- **Package Manager**: {self.system_info.get('package_manager', 'Unknown')}

## GPU HARDWARE DETECTION
{'✅ GPU DETECTED' if self.gpu_info.get('gpus') else '❌ NO GPU DETECTED'}

"""
        
        if self.gpu_info.get('gpus'):
            for gpu in self.gpu_info['gpus']:
                report_content += f"""### GPU {gpu.get('id', 'Unknown')}
- **Name**: {gpu.get('name', 'Unknown')}
- **Memory**: {gpu.get('memory_gb', 0)}GB
- **Driver Version**: {gpu.get('driver_version', 'Unknown')}
- **Compute Capability**: {gpu.get('compute_capability', 'Unknown')}

"""
        
        report_content += f"""## CUDA ENVIRONMENT
- **CUDA Available**: {'✅ YES' if self.cuda_info.get('cuda_available') else '❌ NO'}
- **CUDA Version**: {self.cuda_info.get('cuda_version', 'Not detected')}
- **CUDA Path**: {self.cuda_info.get('cuda_path', 'Not found')}
- **cuDNN Available**: {'✅ YES' if self.cuda_info.get('cudnn_available') else '❌ NO'}

## VERIFICATION RESULTS
- **PyTorch CUDA**: {'✅ AVAILABLE' if verification.get('pytorch_cuda') else '❌ NOT AVAILABLE'}
- **CuPy**: {'✅ AVAILABLE' if verification.get('cupy_available') else '❌ NOT AVAILABLE'}
- **GPU Memory**: {verification.get('memory_available', 0):.1f}GB
- **Compute Capability**: {verification.get('compute_capability', 'Unknown')}

## INSTALLATION STATUS
- **Requirements Installed**: {'✅ YES' if self.requirements_installed else '❌ NO'}
- **GPU Dependencies**: {'✅ SUCCESS' if verification['success'] else '❌ FAILED'}

## RECOMMENDATIONS
"""
        
        if verification['success']:
            report_content += """✅ **GPU acceleration is ready for use!**

### Next Steps:
1. Restart your Python environment to ensure all changes take effect
2. Test GPU acceleration with Kimera SWM
3. Monitor GPU utilization during operation
4. Consider enabling mixed precision training for better performance

### Optimization Tips:
- Use batch processing for better GPU utilization
- Monitor GPU memory usage to avoid out-of-memory errors
- Enable GPU monitoring in Kimera SWM dashboard
"""
        else:
            report_content += """❌ **GPU acceleration setup incomplete**

### Issues Found:
"""
            for error in verification.get('errors', []):
                report_content += f"- {error}\n"
            
            report_content += """
### Recommended Actions:
1. Install or update NVIDIA drivers
2. Install CUDA Toolkit from NVIDIA developer website
3. Verify GPU hardware compatibility
4. Check system requirements and dependencies
5. Consider CPU fallback mode if GPU not available
"""
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"📋 Setup report saved to: {report_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save setup report: {e}")


def main():
    """Main setup function"""
    import time
    
    try:
        setup_manager = GPUSetupManager()
        success = setup_manager.run_setup()
        
        logger.info("\n" + "="*60)
        if success:
            logger.info("🎉 KIMERA SWM GPU ACCELERATION SETUP COMPLETE!")
            logger.info("✅ GPU acceleration is ready for use")
            logger.info("\n🚀 Next steps:")
            logger.info("   1. Restart your Python environment")
            logger.info("   2. Run: python scripts/test_gpu_setup.py")
            logger.info("   3. Start Kimera SWM with GPU acceleration enabled")
        else:
            logger.info("❌ GPU ACCELERATION SETUP FAILED")
            logger.info("💡 Check the setup report for detailed troubleshooting steps")
            logger.info("🔄 Falling back to CPU mode")
        
        logger.info("="*60)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Setup interrupted by user")
        return 1
    except Exception as e:
        logger.info(f"\n❌ Setup failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 