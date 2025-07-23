#!/usr/bin/env python3
"""
KIMERA SWM System Requirements and Health Check
Follows the Kimera SWM Autonomous Architect Protocol v3.0
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraHealthChecker:
    """Comprehensive health checker for Kimera SWM system"""
    
    def __init__(self):
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {},
            "requirements": {},
            "directory_structure": {},
            "dependencies": {},
            "configuration": {},
            "performance": {},
            "security": {},
            "recommendations": []
        }
        
    def check_python_version(self):
        """Check Python version requirements"""
        logger.info("Checking Python version...")
        version_info = {
            "version": sys.version,
            "version_info": list(sys.version_info),
            "executable": sys.executable,
            "platform": sys.platform
        }
        
        meets_requirement = sys.version_info >= (3, 10)
        self.report["requirements"]["python"] = {
            "required": "3.10+",
            "current": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "meets_requirement": meets_requirement,
            "details": version_info
        }
        
        if meets_requirement:
            logger.info("✅ Python 3.10+ requirement satisfied")
        else:
            logger.error("❌ Python 3.10+ required")
            self.report["recommendations"].append("Upgrade Python to version 3.10 or higher")
            
    def check_gpu_availability(self):
        """Check GPU and CUDA availability"""
        logger.info("Checking GPU availability...")
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            gpu_info = {
                "cuda_available": gpu_available,
                "device_count": torch.cuda.device_count() if gpu_available else 0,
                "current_device": torch.cuda.current_device() if gpu_available else None,
                "device_name": torch.cuda.get_device_name() if gpu_available else None
            }
            
            if gpu_available:
                gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
                gpu_info["memory_allocated"] = torch.cuda.memory_allocated()
                gpu_info["memory_cached"] = torch.cuda.memory_reserved()
                logger.info(f"✅ CUDA GPU detected: {gpu_info['device_name']}")
            else:
                logger.warning("⚠️ No CUDA GPU detected - will use CPU")
                self.report["recommendations"].append("Consider GPU acceleration for optimal performance")
                
        except ImportError:
            gpu_info = {"error": "PyTorch not available"}
            logger.error("❌ PyTorch not installed")
            self.report["recommendations"].append("Install PyTorch for GPU acceleration")
            
        self.report["requirements"]["gpu"] = gpu_info
        
    def check_virtual_environment(self):
        """Check virtual environment status"""
        logger.info("Checking virtual environment...")
        
        venv_active = os.environ.get('VIRTUAL_ENV') is not None
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        
        env_info = {
            "virtual_env_active": venv_active,
            "virtual_env_path": os.environ.get('VIRTUAL_ENV'),
            "conda_env": conda_env,
            "python_path": sys.executable
        }
        
        self.report["requirements"]["environment"] = env_info
        
        if venv_active or conda_env:
            logger.info("✅ Virtual environment detected")
        else:
            logger.warning("⚠️ No virtual environment detected")
            self.report["recommendations"].append("Use a virtual environment for dependency isolation")
            
    def check_directory_structure(self):
        """Verify required directory structure exists"""
        logger.info("Checking directory structure...")
        
        required_dirs = [
            'src', 'tests', 'docs', 'scripts', 'configs', 'requirements',
            'scripts/health_check', 'scripts/migration', 'scripts/utils', 'scripts/analysis',
            'docs/reports/health', 'docs/reports/analysis', 'docs/reports/debt', 
            'docs/reports/performance', 'experiments', 'archive', 'cache', 'tmp'
        ]
        
        structure_status = {}
        missing_dirs = []
        
        for dir_path in required_dirs:
            exists = Path(dir_path).exists()
            structure_status[dir_path] = exists
            if not exists:
                missing_dirs.append(dir_path)
                
        self.report["directory_structure"] = {
            "status": structure_status,
            "missing_directories": missing_dirs,
            "compliance": len(missing_dirs) == 0
        }
        
        if missing_dirs:
            logger.warning(f"⚠️ Missing directories: {missing_dirs}")
            self.report["recommendations"].append(f"Create missing directories: {missing_dirs}")
        else:
            logger.info("✅ Directory structure compliant")
            
    def check_configuration_files(self):
        """Check for required configuration files"""
        logger.info("Checking configuration files...")
        
        config_files = {
            '.env': Path('.env').exists(),
            'pyproject.toml': Path('pyproject.toml').exists(),
            'requirements.txt': Path('requirements.txt').exists(),
            'src/main.py': Path('src/main.py').exists(),
            'docs/kimera_ai_reference.md': Path('docs/kimera_ai_reference.md').exists()
        }
        
        missing_configs = [f for f, exists in config_files.items() if not exists]
        
        self.report["configuration"] = {
            "files": config_files,
            "missing_files": missing_configs,
            "compliance": len(missing_configs) == 0
        }
        
        if missing_configs:
            logger.warning(f"⚠️ Missing configuration files: {missing_configs}")
            self.report["recommendations"].append(f"Create missing configuration files: {missing_configs}")
        else:
            logger.info("✅ Configuration files present")
            
    def check_dependencies(self):
        """Check critical dependencies"""
        logger.info("Checking critical dependencies...")
        
        critical_deps = [
            'fastapi', 'uvicorn', 'torch', 'sqlalchemy', 'pydantic',
            'aiohttp', 'python-dotenv', 'rich', 'loguru'
        ]
        
        dep_status = {}
        missing_deps = []
        
        for dep in critical_deps:
            try:
                spec = importlib.util.find_spec(dep)
                if spec is None:
                    dep_status[dep] = {"installed": False, "version": None}
                    missing_deps.append(dep)
                else:
                    try:
                        module = importlib.import_module(dep)
                        version = getattr(module, '__version__', 'Unknown')
                        dep_status[dep] = {"installed": True, "version": version}
                    except:
                        dep_status[dep] = {"installed": True, "version": "Unknown"}
            except Exception as e:
                dep_status[dep] = {"installed": False, "error": str(e)}
                missing_deps.append(dep)
                
        self.report["dependencies"] = {
            "status": dep_status,
            "missing_dependencies": missing_deps,
            "compliance": len(missing_deps) == 0
        }
        
        if missing_deps:
            logger.warning(f"⚠️ Missing dependencies: {missing_deps}")
            self.report["recommendations"].append(f"Install missing dependencies: {missing_deps}")
        else:
            logger.info("✅ Critical dependencies available")
            
    def create_missing_directories(self):
        """Create missing directories if needed"""
        logger.info("Creating missing directories...")
        
        if "missing_directories" in self.report["directory_structure"]:
            for dir_path in self.report["directory_structure"]["missing_directories"]:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {dir_path}: {e}")
                    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        logger.info("Generating comprehensive health report...")
        
        # Calculate overall health score
        checks = [
            self.report["requirements"].get("python", {}).get("meets_requirement", False),
            self.report["directory_structure"].get("compliance", False),
            self.report["configuration"].get("compliance", False),
            self.report["dependencies"].get("compliance", False)
        ]
        
        health_score = sum(checks) / len(checks) * 100
        self.report["overall_health"] = {
            "score": health_score,
            "status": "Excellent" if health_score >= 90 else 
                     "Good" if health_score >= 75 else 
                     "Fair" if health_score >= 60 else "Poor"
        }
        
        # Save report
        report_date = datetime.now().strftime('%Y-%m-%d')
        report_path = f"docs/reports/health/{report_date}_health_report.json"
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2)
            
        logger.info(f"Health report saved to: {report_path}")
        
        # Generate markdown summary
        self.generate_markdown_summary(report_path.replace('.json', '.md'))
        
        return self.report
        
    def generate_markdown_summary(self, output_path):
        """Generate human-readable markdown summary"""
        
        md_content = f"""# KIMERA SWM Health Check Report
**Generated**: {self.report['timestamp']}
**Overall Health**: {self.report['overall_health']['score']:.1f}% ({self.report['overall_health']['status']})

## System Requirements Status

### Python Version
- **Required**: 3.10+
- **Current**: {self.report['requirements']['python']['current']}
- **Status**: {'✅ PASS' if self.report['requirements']['python']['meets_requirement'] else '❌ FAIL'}

### GPU Acceleration
- **CUDA Available**: {self.report['requirements']['gpu'].get('cuda_available', 'Unknown')}
- **Device**: {self.report['requirements']['gpu'].get('device_name', 'None')}

### Environment
- **Virtual Environment**: {'✅ Active' if self.report['requirements']['environment']['virtual_env_active'] else '⚠️ Not Active'}

## Directory Structure
- **Compliance**: {'✅ PASS' if self.report['directory_structure']['compliance'] else '❌ FAIL'}
- **Missing Directories**: {len(self.report['directory_structure']['missing_directories'])}

## Configuration Files
- **Compliance**: {'✅ PASS' if self.report['configuration']['compliance'] else '❌ FAIL'}
- **Missing Files**: {len(self.report['configuration']['missing_files'])}

## Dependencies
- **Compliance**: {'✅ PASS' if self.report['dependencies']['compliance'] else '❌ FAIL'}
- **Missing Dependencies**: {len(self.report['dependencies']['missing_dependencies'])}

## Recommendations
"""
        
        for i, rec in enumerate(self.report['recommendations'], 1):
            md_content += f"{i}. {rec}\n"
            
        md_content += f"""
## Next Steps
1. Address any failing requirements
2. Install missing dependencies: `pip install -r requirements/base.txt`
3. Create .env file from template
4. Run system startup: `python src/main.py`

---
*Report generated by Kimera SWM Autonomous Architect Protocol v3.0*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
            
        logger.info(f"Markdown summary saved to: {output_path}")
        
    def run_complete_health_check(self):
        """Run all health checks"""
        logger.info("🔍 Starting Kimera SWM Health Check...")
        
        self.check_python_version()
        self.check_gpu_availability()
        self.check_virtual_environment()
        self.check_directory_structure()
        self.check_configuration_files()
        self.check_dependencies()
        
        # Create missing directories
        self.create_missing_directories()
        
        # Generate report
        report = self.generate_health_report()
        
        logger.info("✅ Health check complete!")
        return report

def main():
    """Main entry point"""
    checker = KimeraHealthChecker()
    report = checker.run_complete_health_check()
    
    print(f"\n🏥 KIMERA SWM HEALTH CHECK COMPLETE")
    print(f"Overall Health: {report['overall_health']['score']:.1f}% ({report['overall_health']['status']})")
    print(f"Recommendations: {len(report['recommendations'])}")
    
    if report['recommendations']:
        print("\n📋 Action Items:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return report

if __name__ == "__main__":
    main() 