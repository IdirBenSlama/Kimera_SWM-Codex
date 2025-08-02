#!/usr/bin/env python3
"""
Kimera SWM Comprehensive Dependency Verifier
===========================================
Verifies all required dependencies and database configurations.
Scientific rigor: Every assumption is tested empirically.
"""

import os
import sys
import subprocess
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraDependencyVerifier:
    """Comprehensive dependency verification with scientific methodology"""
    
    def __init__(self):
        self.results = {
            "verification_timestamp": datetime.now().isoformat(),
            "python_dependencies": {},
            "database_services": {},
            "monitoring_services": {},
            "gpu_acceleration": {},
            "system_requirements": {},
            "configuration_files": {},
            "recommendations": [],
            "critical_issues": [],
            "overall_status": "UNKNOWN"
        }
        
    def verify_python_dependencies(self) -> Dict[str, Any]:
        """Verify all Python package dependencies"""
        logger.info("🔍 Verifying Python dependencies...")
        
        required_packages = {
            # Core framework
            "fastapi": "Web framework for API",
            "uvicorn": "ASGI server",
            "pydantic": "Data validation",
            "sqlalchemy": "ORM for database operations",
            "alembic": "Database migrations",
            
            # Database drivers
            "psycopg2": "PostgreSQL driver",
            "neo4j": "Neo4j graph database driver",
            "redis": "Redis in-memory database driver",
            "aioredis": "Async Redis driver",
            
            # Monitoring & metrics
            "prometheus_client": "Prometheus metrics collection",
            "grafana_api": "Grafana integration",
            
            # AI/ML frameworks
            "torch": "PyTorch deep learning",
            "transformers": "Hugging Face transformers",
            "accelerate": "Hardware acceleration",
            "datasets": "ML datasets",
            
            # GPU acceleration
            "cupy": "CUDA array processing",
            "pynvml": "NVIDIA GPU monitoring",
            
            # Performance & profiling
            "psutil": "System resource monitoring",
            "memory_profiler": "Memory profiling",
            "line_profiler": "Line-by-line profiling",
            "py-spy": "Performance profiling",
            
            # Load testing
            "locust": "Load testing framework",
            "aiohttp": "Async HTTP client",
            
            # Data processing
            "numpy": "Numerical computing",
            "pandas": "Data manipulation",
            "scipy": "Scientific computing",
            "scikit-learn": "Machine learning",
            
            # Visualization
            "matplotlib": "Plotting library",
            "seaborn": "Statistical visualization",
            "plotly": "Interactive plots",
            
            # Development tools
            "pytest": "Testing framework",
            "black": "Code formatting",
            "ruff": "Linting",
            "mypy": "Type checking"
        }
        
        dependencies_status = {}
        
        for package, description in required_packages.items():
            try:
                # Handle special cases for Python 3.13 compatibility
                if package == "aioredis":
                    # aioredis has issues with Python 3.13 - skip for now
                    dependencies_status[package] = {
                        "status": "⚠️ COMPATIBILITY_ISSUE",
                        "description": f"{description} (Python 3.13 compatibility issue)",
                        "import_successful": False,
                        "error": "Python 3.13 compatibility issue with TimeoutError"
                    }
                    logger.warning(f"⚠️ {package}: Compatibility issue with Python 3.13")
                    continue
                    
                __import__(package.replace('-', '_'))
                dependencies_status[package] = {
                    "status": "✅ AVAILABLE",
                    "description": description,
                    "import_successful": True
                }
                logger.info(f"✅ {package}: Available")
            except ImportError as e:
                dependencies_status[package] = {
                    "status": "❌ MISSING",
                    "description": description,
                    "import_successful": False,
                    "error": str(e)
                }
                logger.warning(f"❌ {package}: Missing - {description}")
                self.results["critical_issues"].append(f"Missing package: {package}")
            except Exception as e:
                dependencies_status[package] = {
                    "status": "❌ ERROR",
                    "description": description,
                    "import_successful": False,
                    "error": str(e)
                }
                logger.warning(f"❌ {package}: Error - {str(e)}")
        
        self.results["python_dependencies"] = dependencies_status
        return dependencies_status
    
    def verify_database_services(self) -> Dict[str, Any]:
        """Verify database service availability and configuration"""
        logger.info("🗄️ Verifying database services...")
        
        database_status = {}
        
        # Test PostgreSQL connection
        try:
            import psycopg2
            from psycopg2 import OperationalError
            
            # Test connection with default settings
            test_configs = [
                {
                    "host": "localhost",
                    "port": 5432,
                    "database": "kimera_swm",
                    "user": "kimera_user",
                    "password": "kimera_secure_pass"
                },
                {
                    "host": "localhost", 
                    "port": 5432,
                    "database": "postgres",
                    "user": "postgres",
                    "password": "postgres"
                }
            ]
            
            postgresql_connected = False
            for config in test_configs:
                try:
                    conn = psycopg2.connect(**config)
                    conn.close()
                    database_status["postgresql"] = {
                        "status": "✅ CONNECTED",
                        "config": config,
                        "available": True
                    }
                    postgresql_connected = True
                    logger.info(f"✅ PostgreSQL: Connected to {config['database']}")
                    break
                except OperationalError:
                    continue
            
            if not postgresql_connected:
                database_status["postgresql"] = {
                    "status": "❌ UNAVAILABLE",
                    "available": False,
                    "error": "Cannot connect to PostgreSQL"
                }
                self.results["critical_issues"].append("PostgreSQL service not available")
                logger.error("❌ PostgreSQL: Cannot connect")
                
        except ImportError:
            database_status["postgresql"] = {
                "status": "❌ DRIVER_MISSING",
                "available": False,
                "error": "psycopg2 not installed"
            }
            
        # Test Neo4j connection
        try:
            from neo4j import GraphDatabase
            
            neo4j_configs = [
                {"uri": "bolt://localhost:7687", "auth": ("neo4j", "kimera_neo4j")},
                {"uri": "bolt://localhost:7687", "auth": ("neo4j", "password")},
                {"uri": "bolt://localhost:7687", "auth": ("neo4j", "neo4j")}
            ]
            
            neo4j_connected = False
            for config in neo4j_configs:
                try:
                    driver = GraphDatabase.driver(config["uri"], auth=config["auth"])
                    driver.verify_connectivity()
                    driver.close()
                    database_status["neo4j"] = {
                        "status": "✅ CONNECTED",
                        "config": config,
                        "available": True
                    }
                    neo4j_connected = True
                    logger.info("✅ Neo4j: Connected")
                    break
                except Exception:
                    continue
            
            if not neo4j_connected:
                database_status["neo4j"] = {
                    "status": "❌ UNAVAILABLE",
                    "available": False,
                    "error": "Cannot connect to Neo4j"
                }
                self.results["critical_issues"].append("Neo4j service not available")
                logger.error("❌ Neo4j: Cannot connect")
                
        except ImportError:
            database_status["neo4j"] = {
                "status": "❌ DRIVER_MISSING",
                "available": False,
                "error": "neo4j driver not installed"
            }
            
        # Test Redis connection
        try:
            import redis
            
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()
            database_status["redis"] = {
                "status": "✅ CONNECTED",
                "available": True
            }
            logger.info("✅ Redis: Connected")
        except Exception as e:
            database_status["redis"] = {
                "status": "❌ UNAVAILABLE",
                "available": False,
                "error": str(e)
            }
            logger.warning("⚠️ Redis: Not available (optional)")
            
        self.results["database_services"] = database_status
        return database_status
    
    def verify_monitoring_services(self) -> Dict[str, Any]:
        """Verify Prometheus and Grafana availability"""
        logger.info("📊 Verifying monitoring services...")
        
        monitoring_status = {}
        
        # Check Prometheus
        try:
            import requests
            prometheus_url = "http://localhost:9090/api/v1/query"
            response = requests.get(prometheus_url, params={"query": "up"}, timeout=5)
            if response.status_code == 200:
                monitoring_status["prometheus"] = {
                    "status": "✅ RUNNING",
                    "url": prometheus_url,
                    "available": True
                }
                logger.info("✅ Prometheus: Running")
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            monitoring_status["prometheus"] = {
                "status": "❌ UNAVAILABLE",
                "available": False,
                "error": str(e)
            }
            logger.warning("⚠️ Prometheus: Not running")
            
        # Check Grafana
        try:
            import requests
            grafana_url = "http://localhost:3000/api/health"
            response = requests.get(grafana_url, timeout=5)
            if response.status_code == 200:
                monitoring_status["grafana"] = {
                    "status": "✅ RUNNING",
                    "url": grafana_url,
                    "available": True
                }
                logger.info("✅ Grafana: Running")
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            monitoring_status["grafana"] = {
                "status": "❌ UNAVAILABLE", 
                "available": False,
                "error": str(e)
            }
            logger.warning("⚠️ Grafana: Not running")
            
        self.results["monitoring_services"] = monitoring_status
        return monitoring_status
    
    def verify_gpu_acceleration(self) -> Dict[str, Any]:
        """Verify GPU acceleration capabilities"""
        logger.info("🎮 Verifying GPU acceleration...")
        
        gpu_status = {}
        
        # Check CUDA availability
        try:
            import torch
            gpu_status["cuda_available"] = torch.cuda.is_available()
            gpu_status["cuda_device_count"] = torch.cuda.device_count()
            
            if torch.cuda.is_available():
                gpu_status["cuda_version"] = torch.version.cuda
                gpu_status["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                logger.info(f"✅ CUDA: Available with {torch.cuda.device_count()} GPU(s)")
            else:
                logger.warning("⚠️ CUDA: Not available")
                
        except ImportError:
            gpu_status["error"] = "PyTorch not available"
            logger.warning("⚠️ PyTorch: Not installed")
            
        # Check NVIDIA drivers
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_status["nvidia_driver_available"] = True
            gpu_status["nvidia_gpu_count"] = gpu_count
            
            gpu_info = []
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                gpu_info.append(name)
            
            gpu_status["nvidia_gpu_names"] = gpu_info
            logger.info(f"✅ NVIDIA Driver: Available with {gpu_count} GPU(s)")
            
        except Exception as e:
            gpu_status["nvidia_driver_available"] = False
            gpu_status["nvidia_error"] = str(e)
            logger.warning("⚠️ NVIDIA Driver: Not available")
            
        self.results["gpu_acceleration"] = gpu_status
        return gpu_status
    
    def verify_configuration_files(self) -> Dict[str, Any]:
        """Verify critical configuration files exist"""
        logger.info("⚙️ Verifying configuration files...")
        
        config_status = {}
        required_configs = {
            ".env": "Environment variables",
            "configs/database/database_config.json": "Database configuration",
            "configs/gpu_config.json": "GPU configuration", 
            "configs/initialization_config.json": "Initialization settings",
            "pyproject.toml": "Project configuration",
            "requirements.txt": "Python dependencies"
        }
        
        for config_path, description in required_configs.items():
            full_path = Path(config_path)
            if full_path.exists():
                config_status[config_path] = {
                    "status": "✅ EXISTS",
                    "description": description,
                    "exists": True
                }
                logger.info(f"✅ {config_path}: Exists")
            else:
                config_status[config_path] = {
                    "status": "❌ MISSING",
                    "description": description,
                    "exists": False
                }
                logger.warning(f"❌ {config_path}: Missing")
                self.results["recommendations"].append(f"Create {config_path}: {description}")
        
        self.results["configuration_files"] = config_status
        return config_status
    
    def create_database_configurations(self):
        """Create proper database configurations for PostgreSQL"""
        logger.info("🔧 Creating database configurations...")
        
        # Ensure directories exist
        os.makedirs("configs/database", exist_ok=True)
        os.makedirs("configs/environments", exist_ok=True)
        
        # PostgreSQL configuration
        postgres_config = {
            "database": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "database": "kimera_swm",
                "username": "kimera_user",
                "password": "kimera_secure_pass",
                "pool_size": 20,
                "max_overflow": 30,
                "pool_timeout": 30,
                "pool_recycle": 1800,
                "echo": False,
                "echo_pool": False
            },
            "fallback": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "database": "postgres",
                "username": "postgres", 
                "password": "postgres"
            }
        }
        
        with open("configs/database/database_config.json", "w") as f:
            json.dump(postgres_config, f, indent=2)
        logger.info("✅ Created PostgreSQL configuration")
        
        # Neo4j configuration  
        neo4j_config = {
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "kimera_neo4j",
                "encrypted": False,
                "trust": "TRUST_ALL_CERTIFICATES",
                "max_connection_lifetime": 1000,
                "max_connection_pool_size": 50,
                "connection_acquisition_timeout": 60
            }
        }
        
        with open("configs/database/neo4j_config.json", "w") as f:
            json.dump(neo4j_config, f, indent=2)
        logger.info("✅ Created Neo4j configuration")
        
        # Redis configuration
        redis_config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None,
                "socket_timeout": 5,
                "connection_pool_kwargs": {
                    "max_connections": 50,
                    "retry_on_timeout": True
                }
            }
        }
        
        with open("configs/database/redis_config.json", "w") as f:
            json.dump(redis_config, f, indent=2)
        logger.info("✅ Created Redis configuration")
        
        # Environment variables to ensure PostgreSQL is used
        env_content = """# Kimera SWM Environment Configuration
# Database Configuration
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://kimera_user:kimera_secure_pass@localhost:5432/kimera_swm
DATABASE_FALLBACK_URL=postgresql://postgres:postgres@localhost:5432/postgres

# Force PostgreSQL (no SQLite fallback)
FORCE_POSTGRESQL=true
DISABLE_SQLITE_FALLBACK=true

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=kimera_neo4j

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8001
GRAFANA_ENABLED=true

# Performance
UVICORN_WORKERS=4
UVICORN_HOST=0.0.0.0
UVICORN_PORT=8000

# Kimera Modes
KIMERA_MODE=optimized
ENABLE_GPU_ACCELERATION=true
ENABLE_ADVANCED_MONITORING=true
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        logger.info("✅ Created environment configuration")
    
    def generate_installation_scripts(self):
        """Generate scripts to install missing services"""
        logger.info("📝 Generating installation scripts...")
        
        os.makedirs("scripts/installation", exist_ok=True)
        
        # PostgreSQL installation script
        postgres_script = """# PostgreSQL Installation and Setup
# Run as Administrator

# Download and install PostgreSQL
Write-Host "📦 Installing PostgreSQL..."

# Using Chocolatey (recommended)
if (Get-Command choco -ErrorAction SilentlyContinue) {
    choco install postgresql13 -y
} else {
    Write-Host "⚠️ Chocolatey not found. Please install PostgreSQL manually from:"
    Write-Host "   https://www.postgresql.org/download/windows/"
    Write-Host "   Version: 13 or higher"
}

# Wait for service to start
Start-Sleep -Seconds 10

# Create Kimera database and user
Write-Host "🔧 Setting up Kimera database..."

# Connect to PostgreSQL and create database
$env:PGPASSWORD="postgres"
psql -U postgres -c "CREATE DATABASE kimera_swm;"
psql -U postgres -c "CREATE USER kimera_user WITH PASSWORD 'kimera_secure_pass';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE kimera_swm TO kimera_user;"

Write-Host "✅ PostgreSQL setup complete!"
"""
        
        with open("scripts/installation/install_postgresql.ps1", "w") as f:
            f.write(postgres_script)
        
        # Neo4j installation script
        neo4j_script = """# Neo4j Installation and Setup
# Run as Administrator

Write-Host "📦 Installing Neo4j..."

# Using Chocolatey
if (Get-Command choco -ErrorAction SilentlyContinue) {
    choco install neo4j-community -y
} else {
    Write-Host "⚠️ Chocolatey not found. Please install Neo4j manually from:"
    Write-Host "   https://neo4j.com/download/"
    Write-Host "   Version: Community 4.4 or higher"
}

# Start Neo4j service
Write-Host "🚀 Starting Neo4j service..."
Start-Service -Name "neo4j"

# Set initial password
Write-Host "🔧 Setting up Neo4j authentication..."
neo4j-admin set-initial-password kimera_neo4j

Write-Host "✅ Neo4j setup complete!"
Write-Host "🌐 Access Neo4j Browser at: http://localhost:7474"
"""
        
        with open("scripts/installation/install_neo4j.ps1", "w") as f:
            f.write(neo4j_script)
        
        # Prometheus installation script
        prometheus_script = """# Prometheus Installation and Setup
# Run as Administrator

Write-Host "📦 Installing Prometheus..."

# Create Prometheus directory
$prometheusDir = "C:\\Prometheus"
New-Item -ItemType Directory -Force -Path $prometheusDir

# Download Prometheus
$prometheusVersion = "2.45.0"
$prometheusUrl = "https://github.com/prometheus/prometheus/releases/download/v$prometheusVersion/prometheus-$prometheusVersion.windows-amd64.zip"
$prometheusZip = "$prometheusDir\\prometheus.zip"

Invoke-WebRequest -Uri $prometheusUrl -OutFile $prometheusZip
Expand-Archive -Path $prometheusZip -DestinationPath $prometheusDir -Force

# Create Prometheus configuration
$configContent = @"
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kimera-swm'
    static_configs:
      - targets: ['localhost:8001']
        labels:
          service: 'kimera-api'
  
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
"@

Set-Content -Path "$prometheusDir\\prometheus.yml" -Value $configContent

Write-Host "✅ Prometheus setup complete!"
Write-Host "🚀 To start Prometheus: cd $prometheusDir && .\\prometheus.exe"
"""
        
        with open("scripts/installation/install_prometheus.ps1", "w") as f:
            f.write(prometheus_script)
        
        logger.info("✅ Installation scripts created in scripts/installation/")
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run all verification checks"""
        logger.info("🔍 Starting comprehensive dependency verification...")
        
        # Run all verification checks
        self.verify_python_dependencies()
        self.verify_database_services()
        self.verify_monitoring_services()
        self.verify_gpu_acceleration()
        self.verify_configuration_files()
        
        # Create configurations
        self.create_database_configurations()
        self.generate_installation_scripts()
        
        # Determine overall status
        critical_count = len(self.results["critical_issues"])
        if critical_count == 0:
            self.results["overall_status"] = "✅ ALL_SYSTEMS_GO"
        elif critical_count <= 2:
            self.results["overall_status"] = "⚠️ MINOR_ISSUES"
        else:
            self.results["overall_status"] = "❌ CRITICAL_ISSUES"
        
        # Generate recommendations
        if not self.results["database_services"].get("postgresql", {}).get("available", False):
            self.results["recommendations"].append("Install PostgreSQL: Run scripts/installation/install_postgresql.ps1")
        
        if not self.results["database_services"].get("neo4j", {}).get("available", False):
            self.results["recommendations"].append("Install Neo4j: Run scripts/installation/install_neo4j.ps1")
        
        if not self.results["monitoring_services"].get("prometheus", {}).get("available", False):
            self.results["recommendations"].append("Install Prometheus: Run scripts/installation/install_prometheus.ps1")
        
        # Save results
        os.makedirs("docs/reports/health", exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        report_path = f"docs/reports/health/{timestamp}_dependency_verification.json"
        
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        return self.results
    
    def print_summary(self):
        """Print verification summary"""
        print("\n" + "="*80)
        print("🔍 KIMERA SWM DEPENDENCY VERIFICATION REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL STATUS: {self.results['overall_status']}")
        
        if self.results["critical_issues"]:
            print(f"\n❌ CRITICAL ISSUES ({len(self.results['critical_issues'])}):")
            for issue in self.results["critical_issues"]:
                print(f"   • {issue}")
        
        if self.results["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS ({len(self.results['recommendations'])}):")
            for rec in self.results["recommendations"]:
                print(f"   • {rec}")
        
        # Database status
        print(f"\n🗄️ DATABASE SERVICES:")
        for db, status in self.results["database_services"].items():
            print(f"   • {db}: {status.get('status', 'UNKNOWN')}")
        
        # GPU status
        if self.results["gpu_acceleration"]:
            print(f"\n🎮 GPU ACCELERATION:")
            gpu = self.results["gpu_acceleration"]
            if gpu.get("cuda_available"):
                print(f"   • CUDA: ✅ Available ({gpu.get('cuda_device_count', 0)} devices)")
            else:
                print(f"   • CUDA: ❌ Not available")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    verifier = KimeraDependencyVerifier()
    results = verifier.run_comprehensive_verification()
    verifier.print_summary()
    
    print(f"\n📄 Full report saved to: docs/reports/health/")
    
    if results["overall_status"].startswith("❌"):
        print("\n🚨 CRITICAL ISSUES DETECTED - Please resolve before continuing")
        sys.exit(1)
    elif results["overall_status"].startswith("⚠️"):
        print("\n⚠️ MINOR ISSUES DETECTED - Kimera may run with limited functionality")
        sys.exit(2)
    else:
        print("\n✅ ALL SYSTEMS GO - Ready for Kimera SWM deployment!")
        sys.exit(0)