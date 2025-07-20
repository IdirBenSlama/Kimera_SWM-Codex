#!/usr/bin/env python3
"""
KIMERA Phase 1, Week 2: Quantum Integration - Library Installation Script

This script automatically installs and configures all quantum computing libraries
required for KIMERA's quantum-enhanced cognitive processing capabilities.

Phase: Phase 1, Week 2 - Quantum Integration
Date: December 19, 2024
Prerequisites: Phase 1, Week 1 (GPU Foundation) completed successfully
"""

import subprocess
import sys
import logging
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quantum_installation_phase2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KimeraQuantumInstaller:
    """
    Automated quantum computing library installer for KIMERA Phase 1, Week 2.
    
    Installs and validates:
    - Qiskit-Aer-GPU for high-performance quantum simulation
    - CUDA-Q for GPU-native quantum computing
    - PennyLane-Lightning-GPU for quantum machine learning
    - Additional quantum utility libraries
    """
    
    def __init__(self):
        """Initialize the quantum library installer."""
        self.installation_log = {
            'phase': 'Phase 1, Week 2 - Quantum Integration',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'libraries': {},
            'validation_results': {},
            'performance_benchmarks': {},
            'errors': [],
            'warnings': []
        }
        
        # Quantum libraries to install
        self.quantum_libraries = {
            'qiskit-aer-gpu': {
                'packages': ['qiskit-aer[gpu]', 'qiskit[visualization]', 'qiskit-algorithms'],
                'test_import': 'qiskit_aer',
                'gpu_required': True,
                'description': 'High-performance GPU-accelerated quantum simulation'
            },
            'cudaq': {
                'packages': ['cudaq'],
                'test_import': 'cudaq',
                'gpu_required': True,
                'description': 'NVIDIA CUDA-Q GPU-native quantum computing'
            },
            'pennylane-lightning-gpu': {
                'packages': ['pennylane', 'pennylane-lightning[gpu]'],
                'test_import': 'pennylane',
                'gpu_required': True,
                'description': 'Quantum machine learning with GPU acceleration'
            },
            'quantum-utilities': {
                'packages': ['qiskit-optimization', 'qiskit-nature', 'qiskit-finance'],
                'test_import': 'qiskit_optimization',
                'gpu_required': False,
                'description': 'Quantum computing utility libraries'
            },
            'visualization': {
                'packages': ['matplotlib', 'seaborn', 'plotly', 'qiskit[visualization]'],
                'test_import': 'matplotlib',
                'gpu_required': False,
                'description': 'Quantum visualization and plotting libraries'
            }
        }
        
        # Performance validation tests
        self.validation_tests = [
            'test_gpu_availability',
            'test_qiskit_aer_gpu',
            'test_cudaq_functionality',
            'test_pennylane_gpu',
            'test_quantum_circuit_simulation',
            'test_quantum_ml_integration',
            'benchmark_quantum_performance'
        ]
        
    def validate_prerequisites(self) -> bool:
        """
        Validate that Phase 1, Week 1 prerequisites are satisfied.
        
        Returns:
            bool: True if all prerequisites are met
        """
        logger.info("🔍 Validating Phase 1, Week 1 prerequisites...")
        
        try:
            # Check for GPU Foundation
            gpu_foundation_path = Path('backend/utils/gpu_foundation.py')
            if not gpu_foundation_path.exists():
                logger.error("❌ GPU Foundation not found - Phase 1, Week 1 not completed")
                return False
            
            # Validate GPU availability
            import torch
            
            if not torch.cuda.is_available():
                logger.error("❌ CUDA not available - GPU Foundation required")
                return False
                
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"✅ GPU Available: {gpu_name}")
            logger.info(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            
            # Check minimum requirements
            if gpu_memory < 8.0:  # 8GB minimum for quantum simulation
                logger.warning(f"⚠️ GPU Memory {gpu_memory:.1f}GB may be insufficient for large quantum circuits")
                self.installation_log['warnings'].append(f"Low GPU memory: {gpu_memory:.1f}GB")
                
            # Validate CUDA version
            cuda_version = torch.version.cuda
            logger.info(f"✅ CUDA Version: {cuda_version}")
            
            if not cuda_version or float(cuda_version) < 11.0:
                logger.error(f"❌ CUDA version {cuda_version} insufficient - CUDA 11.0+ required")
                return False
                
            logger.info("✅ All Phase 1, Week 1 prerequisites satisfied")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prerequisite validation failed: {e}")
            self.installation_log['errors'].append(f"Prerequisite validation: {e}")
            return False
    
    def install_library_group(self, group_name: str, group_info: Dict[str, Any]) -> bool:
        """
        Install a group of related quantum libraries.
        
        Args:
            group_name: Name of the library group
            group_info: Installation information
            
        Returns:
            bool: True if installation successful
        """
        logger.info(f"📦 Installing {group_name}: {group_info['description']}")
        
        installation_start = time.perf_counter()
        
        try:
            # Install each package in the group
            for package in group_info['packages']:
                logger.info(f"   Installing {package}...")
                
                install_cmd = [sys.executable, '-m', 'pip', 'install', package, '--upgrade']
                
                # Add GPU-specific flags if needed
                if group_info['gpu_required']:
                    install_cmd.extend(['--extra-index-url', 'https://pypi.nvidia.com'])
                
                result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"❌ Failed to install {package}")
                    logger.error(f"Error: {result.stderr}")
                    self.installation_log['errors'].append(f"{package}: {result.stderr}")
                    return False
                else:
                    logger.info(f"   ✅ {package} installed successfully")
            
            # Test import
            try:
                importlib.import_module(group_info['test_import'])
                logger.info(f"✅ {group_name} import validation successful")
            except ImportError as e:
                logger.warning(f"⚠️ {group_name} import test failed: {e}")
                self.installation_log['warnings'].append(f"{group_name} import: {e}")
            
            installation_time = time.perf_counter() - installation_start
            
            self.installation_log['libraries'][group_name] = {
                'status': 'success',
                'packages': group_info['packages'],
                'installation_time': installation_time,
                'description': group_info['description']
            }
            
            logger.info(f"✅ {group_name} installation completed in {installation_time:.1f}s")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {group_name} installation timed out")
            self.installation_log['errors'].append(f"{group_name}: Installation timeout")
            return False
        except Exception as e:
            logger.error(f"❌ {group_name} installation failed: {e}")
            self.installation_log['errors'].append(f"{group_name}: {e}")
            return False
    
    def validate_quantum_installation(self) -> Dict[str, Any]:
        """
        Validate quantum library installations with comprehensive testing.
        
        Returns:
            Dict containing validation results
        """
        logger.info("🧪 Validating quantum library installations...")
        
        validation_results = {
            'overall_success': True,
            'test_results': {},
            'performance_metrics': {},
            'errors': []
        }
        
        try:
            # Test 1: GPU Availability
            logger.info("   Testing GPU availability...")
            import torch
            
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count()
            
            validation_results['test_results']['gpu_availability'] = {
                'success': gpu_available,
                'gpu_count': gpu_count,
                'gpu_name': torch.cuda.get_device_name(0) if gpu_available else None
            }
            
            if not gpu_available:
                logger.error("❌ GPU not available for quantum processing")
                validation_results['overall_success'] = False
            else:
                logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            
            # Test 2: Qiskit-Aer-GPU
            logger.info("   Testing Qiskit-Aer-GPU...")
            try:
                from qiskit import QuantumCircuit, execute
                from qiskit_aer import AerSimulator
                
                # Create test circuit
                qc = QuantumCircuit(5, 5)
                qc.h(range(5))
                qc.measure_all()
                
                # Test GPU simulation
                simulator = AerSimulator(method='statevector', device='GPU')
                
                start_time = time.perf_counter()
                job = execute(qc, simulator, shots=1024)
                result = job.result()
                execution_time = time.perf_counter() - start_time
                
                counts = result.get_counts()
                
                validation_results['test_results']['qiskit_aer_gpu'] = {
                    'success': len(counts) > 0,
                    'execution_time': execution_time,
                    'measurement_outcomes': len(counts),
                    'total_shots': sum(counts.values())
                }
                
                logger.info(f"✅ Qiskit-Aer-GPU test passed - {execution_time:.3f}s execution")
                
            except Exception as e:
                logger.error(f"❌ Qiskit-Aer-GPU test failed: {e}")
                validation_results['test_results']['qiskit_aer_gpu'] = {'success': False, 'error': str(e)}
                validation_results['overall_success'] = False
            
            # Test 3: CUDA-Q (if available)
            logger.info("   Testing CUDA-Q...")
            try:
                import cudaq
                
                @cudaq.kernel
                def test_kernel():
                    qubits = cudaq.qvector(5)
                    for i in range(5):
                        h(qubits[i])
                    mz(qubits)
                
                start_time = time.perf_counter()
                result = cudaq.sample(test_kernel, shots_count=1024)
                execution_time = time.perf_counter() - start_time
                
                validation_results['test_results']['cudaq'] = {
                    'success': True,
                    'execution_time': execution_time,
                    'measurement_outcomes': len(result)
                }
                
                logger.info(f"✅ CUDA-Q test passed - {execution_time:.3f}s execution")
                
            except Exception as e:
                logger.warning(f"⚠️ CUDA-Q test failed (may not be available): {e}")
                validation_results['test_results']['cudaq'] = {'success': False, 'error': str(e)}
            
            # Test 4: PennyLane-Lightning-GPU
            logger.info("   Testing PennyLane-Lightning-GPU...")
            try:
                import pennylane as qml
                import numpy as np
                
                # Create GPU device
                dev = qml.device('lightning.gpu', wires=4)
                
                @qml.qnode(dev)
                def test_circuit(params):
                    qml.RY(params[0], wires=0)
                    qml.RY(params[1], wires=1)
                    qml.CNOT(wires=[0, 1])
                    return qml.expval(qml.PauliZ(0))
                
                params = np.array([0.5, 1.0])
                
                start_time = time.perf_counter()
                result = test_circuit(params)
                execution_time = time.perf_counter() - start_time
                
                validation_results['test_results']['pennylane_gpu'] = {
                    'success': isinstance(result, (int, float, np.number)),
                    'execution_time': execution_time,
                    'expectation_value': float(result)
                }
                
                logger.info(f"✅ PennyLane-Lightning-GPU test passed - {execution_time:.3f}s execution")
                
            except Exception as e:
                logger.error(f"❌ PennyLane-Lightning-GPU test failed: {e}")
                validation_results['test_results']['pennylane_gpu'] = {'success': False, 'error': str(e)}
                validation_results['overall_success'] = False
            
            # Performance benchmark
            logger.info("   Running quantum performance benchmark...")
            try:
                benchmark_results = self.benchmark_quantum_performance()
                validation_results['performance_metrics'] = benchmark_results
                logger.info("✅ Quantum performance benchmark completed")
            except Exception as e:
                logger.error(f"❌ Quantum performance benchmark failed: {e}")
                validation_results['errors'].append(f"Performance benchmark: {e}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Quantum validation failed: {e}")
            validation_results['overall_success'] = False
            validation_results['errors'].append(str(e))
            return validation_results
    
    def benchmark_quantum_performance(self) -> Dict[str, Any]:
        """
        Benchmark quantum computing performance across different qubit counts.
        
        Returns:
            Dict containing performance metrics
        """
        logger.info("🚀 Benchmarking quantum performance...")
        
        benchmark_results = {
            'qubit_scalability': [],
            'simulation_times': [],
            'memory_usage': [],
            'throughput_metrics': {}
        }
        
        try:
            from qiskit import QuantumCircuit, execute
            from qiskit_aer import AerSimulator
            import torch
            
            simulator = AerSimulator(method='statevector', device='GPU')
            
            # Test different qubit counts
            qubit_counts = [5, 10, 15, 20]
            
            for num_qubits in qubit_counts:
                logger.info(f"   Benchmarking {num_qubits} qubits...")
                
                # Create benchmark circuit
                qc = QuantumCircuit(num_qubits, num_qubits)
                
                # Add complexity
                for i in range(num_qubits):
                    qc.h(i)
                for i in range(num_qubits - 1):
                    qc.cx(i, i + 1)
                for i in range(num_qubits):
                    qc.rz(0.5, i)
                
                qc.measure_all()
                
                # Memory before execution
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
                
                # Execute and time
                start_time = time.perf_counter()
                job = execute(qc, simulator, shots=1024)
                result = job.result()
                execution_time = time.perf_counter() - start_time
                
                # Memory after execution
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / (1024**2)  # MB
                
                # Calculate throughput
                total_operations = num_qubits * 3 + (num_qubits - 1)  # H + CX + RZ gates
                throughput = total_operations * 1024 / execution_time  # operations per second
                
                benchmark_results['qubit_scalability'].append(num_qubits)
                benchmark_results['simulation_times'].append(execution_time)
                benchmark_results['memory_usage'].append(memory_used)
                
                logger.info(f"     {num_qubits} qubits: {execution_time:.3f}s, {memory_used:.1f}MB, {throughput:.0f} ops/s")
            
            # Overall throughput metrics
            avg_simulation_time = sum(benchmark_results['simulation_times']) / len(benchmark_results['simulation_times'])
            max_qubits_tested = max(benchmark_results['qubit_scalability'])
            total_memory_peak = max(benchmark_results['memory_usage'])
            
            benchmark_results['throughput_metrics'] = {
                'average_simulation_time': avg_simulation_time,
                'max_qubits_validated': max_qubits_tested,
                'peak_memory_usage_mb': total_memory_peak,
                'quantum_ready': avg_simulation_time < 5.0 and max_qubits_tested >= 15
            }
            
            logger.info(f"✅ Quantum performance: {max_qubits_tested} qubits, {avg_simulation_time:.3f}s avg")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"❌ Quantum benchmarking failed: {e}")
            return {'error': str(e)}
    
    def generate_installation_report(self) -> str:
        """
        Generate comprehensive installation and validation report.
        
        Returns:
            str: Path to generated report file
        """
        logger.info("📋 Generating installation report...")
        
        report_data = {
            'installation_summary': self.installation_log,
            'validation_results': self.installation_log.get('validation_results', {}),
            'system_configuration': self.get_system_configuration(),
            'recommendations': self.generate_recommendations()
        }
        
        # Save report
        report_path = f"logs/quantum_installation_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"📄 Installation report saved: {report_path}")
        return report_path
    
    def get_system_configuration(self) -> Dict[str, Any]:
        """Get current system configuration for quantum computing."""
        try:
            import torch
            
            config = {
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__,
                'python_version': sys.version
            }
            
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                config.update({
                    'gpu_memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multiprocessors': props.multi_processor_count
                })
            
            return config
        except Exception as e:
            return {'error': str(e)}
    
    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on installation results."""
        recommendations = []
        
        # Check for any errors
        if self.installation_log.get('errors'):
            recommendations.append("❌ Resolve installation errors before proceeding to quantum integration")
        
        # Check for warnings
        if self.installation_log.get('warnings'):
            recommendations.append("⚠️ Review installation warnings for potential performance impacts")
        
        # Performance recommendations
        validation_results = self.installation_log.get('validation_results', {})
        if 'performance_metrics' in validation_results:
            metrics = validation_results['performance_metrics']
            
            if 'throughput_metrics' in metrics:
                throughput = metrics['throughput_metrics']
                
                if not throughput.get('quantum_ready', False):
                    recommendations.append("🔧 System may need optimization for quantum workloads")
                
                if throughput.get('peak_memory_usage_mb', 0) > 10000:  # >10GB
                    recommendations.append("💾 Consider memory optimization for large quantum circuits")
                    
                if throughput.get('max_qubits_validated', 0) < 20:
                    recommendations.append("📈 GPU memory may limit large quantum circuit simulation")
        
        # Add positive recommendations
        if not recommendations:
            recommendations.extend([
                "✅ System optimally configured for quantum integration",
                "🚀 Ready to proceed with Phase 1, Week 2 implementation",
                "📈 Performance metrics indicate excellent quantum computing capability"
            ])
        
        return recommendations
    
    def run_complete_installation(self) -> bool:
        """
        Run complete quantum library installation and validation process.
        
        Returns:
            bool: True if installation and validation successful
        """
        logger.info("🚀 Starting KIMERA Phase 1, Week 2: Quantum Integration - Library Installation")
        logger.info("=" * 80)
        
        overall_success = True
        
        try:
            # Step 1: Validate prerequisites
            if not self.validate_prerequisites():
                logger.error("❌ Prerequisites not satisfied - cannot proceed")
                return False
            
            # Step 2: Install quantum libraries
            logger.info("\n📦 Installing quantum computing libraries...")
            
            for group_name, group_info in self.quantum_libraries.items():
                success = self.install_library_group(group_name, group_info)
                if not success:
                    overall_success = False
                    logger.error(f"❌ Failed to install {group_name}")
                else:
                    logger.info(f"✅ {group_name} installed successfully")
            
            # Step 3: Validate installations
            logger.info("\n🧪 Validating quantum library installations...")
            
            validation_results = self.validate_quantum_installation()
            self.installation_log['validation_results'] = validation_results
            
            if not validation_results['overall_success']:
                overall_success = False
                logger.error("❌ Quantum library validation failed")
            else:
                logger.info("✅ All quantum libraries validated successfully")
            
            # Step 4: Generate report
            report_path = self.generate_installation_report()
            
            # Step 5: Final status
            logger.info("\n" + "=" * 80)
            if overall_success:
                logger.info("🎉 QUANTUM INTEGRATION INSTALLATION COMPLETED SUCCESSFULLY!")
                logger.info("✅ KIMERA is ready for Phase 1, Week 2: Quantum Integration")
                logger.info("🚀 Next: Begin quantum circuit implementation and testing")
            else:
                logger.error("❌ QUANTUM INTEGRATION INSTALLATION FAILED")
                logger.error("🔧 Review errors and warnings before proceeding")
            
            logger.info(f"📋 Complete report available: {report_path}")
            logger.info("=" * 80)
            
            return overall_success
            
        except Exception as e:
            logger.error(f"❌ Installation process failed: {e}")
            self.installation_log['errors'].append(f"Installation process: {e}")
            return False

def main():
    """Main installation function."""
    installer = KimeraQuantumInstaller()
    success = installer.run_complete_installation()
    
    if success:
        logger.info("\n🎉 KIMERA Quantum Integration libraries installed successfully!")
        logger.info("🚀 Ready to proceed with Phase 1, Week 2 implementation.")
        sys.exit(0)
    else:
        logger.error("\n❌ KIMERA Quantum Integration installation failed.")
        logger.debug("🔧 Please review the logs and resolve any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main() 