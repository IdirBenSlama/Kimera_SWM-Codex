"""
System Profiler - Deep System Analysis
======================================

Implements comprehensive system profiling based on:
- Linux perf tools patterns
- Windows Performance Toolkit
- Medical device profiling standards
"""

import os
import sys
import platform
import psutil
import logging
import json
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SystemProfile:
    """Complete system profile snapshot."""
    timestamp: datetime
    
    # Hardware
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    disk_info: List[Dict[str, Any]]
    gpu_info: Optional[Dict[str, Any]]
    
    # Software
    os_info: Dict[str, Any]
    python_info: Dict[str, Any]
    installed_packages: List[Dict[str, str]]
    
    # Runtime
    process_info: Dict[str, Any]
    environment_vars: Dict[str, str]
    
    # Performance
    performance_metrics: Dict[str, Any]

class SystemProfiler:
    """
    Comprehensive system profiling tool.
    
    Features:
    - Hardware inventory
    - Software stack analysis
    - Performance baseline
    - Compatibility checking
    """
    
    def __init__(self):
        self.profiles: List[SystemProfile] = []
        self.baseline_profile: Optional[SystemProfile] = None
        
        logger.info("SystemProfiler initialized")
    
    def capture_profile(self) -> SystemProfile:
        """Capture complete system profile."""
        logger.info("Capturing system profile...")
        
        profile = SystemProfile(
            timestamp=datetime.now(),
            cpu_info=self._get_cpu_info(),
            memory_info=self._get_memory_info(),
            disk_info=self._get_disk_info(),
            gpu_info=self._get_gpu_info(),
            os_info=self._get_os_info(),
            python_info=self._get_python_info(),
            installed_packages=self._get_installed_packages(),
            process_info=self._get_process_info(),
            environment_vars=self._get_environment_vars(),
            performance_metrics=self._get_performance_metrics()
        )
        
        self.profiles.append(profile)
        
        # Keep only last 10 profiles
        if len(self.profiles) > 10:
            self.profiles = self.profiles[-10:]
        
        logger.info("System profile captured")
        return profile
    
    def set_baseline(self, profile: Optional[SystemProfile] = None):
        """Set baseline profile for comparison."""
        if profile:
            self.baseline_profile = profile
        else:
            self.baseline_profile = self.capture_profile()
        
        logger.info("Baseline profile set")
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        info = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'current_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'per_cpu_percent': psutil.cpu_percent(interval=1, percpu=True)
        }
        
        # Platform-specific CPU info
        if platform.system() == 'Linux':
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            info['model'] = line.split(':')[1].strip()
                            break
            except Exception as e:
                logger.error(f"Error in system_profiler.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
        elif platform.system() == 'Windows':
            try:
                import wmi
                c = wmi.WMI()
                for processor in c.Win32_Processor():
                    info['model'] = processor.Name
                    info['manufacturer'] = processor.Manufacturer
                    break
            except Exception as e:
                logger.error(f"Error in system_profiler.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
        
        return info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': mem.total,
            'available': mem.available,
            'used': mem.used,
            'percent': mem.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_percent': swap.percent
        }
    
    def _get_disk_info(self) -> List[Dict[str, Any]]:
        """Get disk information."""
        disks = []
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent
                })
            except PermissionError:
                continue
        
        return disks
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information."""
        gpu_info = {}
        
        # Try PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['device_count'] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info[f'gpu_{i}'] = {
                        'name': props.name,
                        'total_memory': props.total_memory,
                        'major': props.major,
                        'minor': props.minor,
                        'multi_processor_count': props.multi_processor_count
                    }
            else:
                gpu_info['cuda_available'] = False
        except ImportError:
            gpu_info['cuda_available'] = False
        
        # Try nvidia-smi
        if platform.system() in ['Linux', 'Windows']:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,temperature.gpu', '--format=csv,noheader'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_info[f'nvidia_gpu_{i}'] = {
                                'name': parts[0],
                                'memory_total': parts[1],
                                'memory_used': parts[2],
                                'temperature': parts[3]
                            }
            except Exception as e:
                logger.error(f"Error in system_profiler.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
        
        return gpu_info if gpu_info else None
    
    def _get_os_info(self) -> Dict[str, Any]:
        """Get operating system information."""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'platform': platform.platform(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
    
    def _get_python_info(self) -> Dict[str, Any]:
        """Get Python environment information."""
        return {
            'version': sys.version,
            'version_info': {
                'major': sys.version_info.major,
                'minor': sys.version_info.minor,
                'micro': sys.version_info.micro
            },
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler(),
            'executable': sys.executable,
            'prefix': sys.prefix,
            'path': sys.path[:5]  # First 5 paths
        }
    
    def _get_installed_packages(self) -> List[Dict[str, str]]:
        """Get list of installed Python packages."""
        packages = []
        
        try:
            import pkg_resources
            for dist in pkg_resources.working_set:
                packages.append({
                    'name': dist.project_name,
                    'version': dist.version
                })
        except Exception as e:
            logger.error(f"Error in system_profiler.py: {e}", exc_info=True)
            raise  # Re-raise for proper error handling
            # Fallback to pip list
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'list', '--format=json'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    packages = json.loads(result.stdout)
            except Exception as e:
                logger.error(f"Error in system_profiler.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
        
        # Sort by name
        packages.sort(key=lambda x: x['name'].lower())
        
        # Return only key packages for KIMERA
        key_packages = [
            'torch', 'transformers', 'fastapi', 'uvicorn',
            'sqlalchemy', 'numpy', 'scipy', 'pydantic',
            'psutil', 'aiofiles', 'httpx', 'prometheus-client'
        ]
        
        return [p for p in packages if p['name'] in key_packages]
    
    def _get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        process = psutil.Process()
        
        with process.oneshot():
            info = {
                'pid': process.pid,
                'name': process.name(),
                'status': process.status(),
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat(),
                'cpu_percent': process.cpu_percent(),
                'memory_info': {
                    'rss': process.memory_info().rss,
                    'vms': process.memory_info().vms,
                    'percent': process.memory_percent()
                },
                'num_threads': process.num_threads(),
                'num_fds': len(process.open_files()) if hasattr(process, 'open_files') else None
            }
        
        return info
    
    def _get_environment_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        relevant_vars = [
            'PYTHONPATH', 'PATH', 'CUDA_VISIBLE_DEVICES',
            'DATABASE_URL', 'KIMERA_PROFILE', 'KIMERA_HOME'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                # Truncate long values
                value = os.environ[var]
                if len(value) > 100:
                    value = value[:100] + '...'
                env_vars[var] = value
        
        return env_vars
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        # CPU times
        cpu_times = psutil.cpu_times()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        return {
            'cpu_times': {
                'user': cpu_times.user,
                'system': cpu_times.system,
                'idle': cpu_times.idle
            },
            'disk_io': {
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            } if disk_io else None,
            'network_io': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            } if net_io else None
        }
    
    def compare_profiles(
        self,
        profile1: SystemProfile,
        profile2: SystemProfile
    ) -> Dict[str, Any]:
        """Compare two system profiles."""
        differences = {
            'hardware_changes': [],
            'software_changes': [],
            'performance_changes': []
        }
        
        # Compare CPU
        if profile1.cpu_info['logical_cores'] != profile2.cpu_info['logical_cores']:
            differences['hardware_changes'].append(
                f"CPU cores changed: {profile1.cpu_info['logical_cores']} -> "
                f"{profile2.cpu_info['logical_cores']}"
            )
        
        # Compare memory
        mem_change = (
            (profile2.memory_info['total'] - profile1.memory_info['total']) / 
            profile1.memory_info['total'] * 100
        )
        if abs(mem_change) > 5:  # 5% change
            differences['hardware_changes'].append(
                f"Memory changed by {mem_change:.1f}%"
            )
        
        # Compare packages
        pkgs1 = {p['name']: p['version'] for p in profile1.installed_packages}
        pkgs2 = {p['name']: p['version'] for p in profile2.installed_packages}
        
        for name, version in pkgs2.items():
            if name not in pkgs1:
                differences['software_changes'].append(f"Added: {name} {version}")
            elif pkgs1[name] != version:
                differences['software_changes'].append(
                    f"Updated: {name} {pkgs1[name]} -> {version}"
                )
        
        for name in pkgs1:
            if name not in pkgs2:
                differences['software_changes'].append(f"Removed: {name}")
        
        return differences
    
    def export_profile(self, profile: SystemProfile, filepath: str):
        """Export profile to file."""
        data = asdict(profile)
        
        # Convert datetime objects
        data['timestamp'] = data['timestamp'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Profile exported to {filepath}")
    
    def check_compatibility(self) -> Dict[str, List[str]]:
        """Check system compatibility for KIMERA."""
        issues = {
            'critical': [],
            'warnings': [],
            'recommendations': []
        }
        
        profile = self.capture_profile()
        
        # Check Python version
        if profile.python_info['version_info']['major'] < 3:
            issues['critical'].append("Python 3.x required")
        elif profile.python_info['version_info']['minor'] < 11:
            issues['warnings'].append("Python 3.11+ recommended")
        
        # Check memory
        if profile.memory_info['total'] < 8 * 1024**3:  # 8GB
            issues['warnings'].append(
                f"Low memory: {profile.memory_info['total'] / 1024**3:.1f}GB "
                f"(8GB+ recommended)"
            )
        
        # Check disk space
        for disk in profile.disk_info:
            if disk['mountpoint'] == '/' and disk['percent'] > 90:
                issues['critical'].append(
                    f"Low disk space: {disk['percent']:.1f}% used"
                )
        
        # Check GPU
        if not profile.gpu_info or not profile.gpu_info.get('cuda_available'):
            issues['recommendations'].append(
                "No CUDA GPU detected - performance will be limited"
            )
        
        return issues

# Global profiler instance
_system_profiler: Optional[SystemProfiler] = None

def get_system_profiler() -> SystemProfiler:
    """Get global system profiler instance."""
    global _system_profiler
    if _system_profiler is None:
        _system_profiler = SystemProfiler()
    return _system_profiler