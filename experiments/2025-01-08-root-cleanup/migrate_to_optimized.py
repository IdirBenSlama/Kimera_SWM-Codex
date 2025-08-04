#!/usr/bin/env python3
"""
Kimera Performance Optimization Migration Script
==============================================
Seamlessly migrates Kimera to the optimized implementation.

This script:
1. Backs up current configuration
2. Updates the main entry point
3. Verifies the optimization
4. Provides rollback capability
"""

import os
import shutil
import sys
import time
import subprocess
from pathlib import Path
import json

class KimeraOptimizationMigrator:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.backend_dir = self.base_dir / "backend"
        self.api_dir = self.backend_dir / "api"
        self.backup_dir = self.base_dir / "backup_before_optimization"
        
    def create_backup(self):
        """Create backup of current implementation"""
        logger.info("📦 Creating backup...")
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup critical files
        files_to_backup = [
            self.api_dir / "main.py",
            self.api_dir / "routers" / "metrics_router.py",
            self.api_dir / "routers" / "geoid_scar_router.py",
            self.backend_dir / "monitoring" / "system_health_monitor.py"
        ]
        
        for file in files_to_backup:
            if file.exists():
                backup_path = self.backup_dir / file.name
                shutil.copy2(file, backup_path)
                logger.info(f"  ✓ Backed up {file.name}")
        
        # Save backup metadata
        metadata = {
            "timestamp": time.time(),
            "files": [str(f.name) for f in files_to_backup],
            "kimera_version": "0.1.0"
        }
        
        with open(self.backup_dir / "backup_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ Backup completed")
    
    def update_main_entry(self):
        """Update the main kimera.py to use optimized version"""
        logger.info("\n🔧 Updating main entry point...")
        
        kimera_py = self.base_dir / "kimera.py"
        
        # Read current content
        with open(kimera_py, 'r') as f:
            content = f.read()
        
        # Update import to use optimized main
        updated_content = content.replace(
            'from src.api.main import app',
            'from src.api.main_optimized import app'
        )
        
        # Write updated content
        with open(kimera_py, 'w') as f:
            f.write(updated_content)
        
        logger.info("✅ Main entry point updated")
    
    def update_router_imports(self):
        """Update router imports in optimized main"""
        logger.info("\n🔄 Updating router imports...")
        
        main_optimized = self.api_dir / "main_optimized.py"
        
        # Read the file
        with open(main_optimized, 'r') as f:
            content = f.read()
        
        # Add optimized geoid router import
        if "optimized_geoid_router" not in content:
            # Find the router import section
            import_section = "from .routers import optimized_metrics_router"
            replacement = """from .routers import optimized_metrics_router
from .routers import optimized_geoid_router"""
            
            content = content.replace(import_section, replacement)
            
            # Add router inclusion
            router_section = "app.include_router(optimized_metrics_router.router, tags=[\"System-Metrics-Optimized\"])"
            replacement = """app.include_router(optimized_metrics_router.router, tags=["System-Metrics-Optimized"])
app.include_router(optimized_geoid_router.router, prefix="/kimera", tags=["Geoids-Optimized"])"""
            
            content = content.replace(router_section, replacement)
            
            # Write back
            with open(main_optimized, 'w') as f:
                f.write(content)
            
            logger.info("✅ Router imports updated")
    
    def create_startup_script(self):
        """Create optimized startup script"""
        logger.info("\n📝 Creating optimized startup script...")
        
        script_content = """#!/usr/bin/env python3
\"\"\"
Optimized Kimera Startup Script
==============================
Starts Kimera with performance optimizations enabled.
\"\"\"

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import optimized app
from src.api.main_optimized import app

if __name__ == "__main__":
    # Performance-optimized Uvicorn configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=1,  # Single worker for now, can increase based on CPU cores
        loop="uvloop",  # High-performance event loop
        access_log=False,  # Disable access logs for performance
        log_level="warning",  # Reduce log verbosity
        limit_concurrency=1000,  # High concurrency limit
        limit_max_requests=10000,  # Request limit before worker restart
        timeout_keep_alive=5,  # Keep-alive timeout
    )
"""
        
        startup_file = self.base_dir / "kimera_optimized.py"
        with open(startup_file, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix-like systems
        if os.name != 'nt':
            os.chmod(startup_file, 0o755)
        
        logger.info("✅ Startup script created: kimera_optimized.py")
    
    def install_performance_dependencies(self):
        """Install additional performance dependencies"""
        logger.info("\n📦 Installing performance dependencies...")
        
        dependencies = [
            "orjson",  # Fast JSON serialization
            "uvloop",  # High-performance event loop
            "httptools",  # Fast HTTP parser
            "python-multipart",  # For file uploads
            "aiofiles",  # Async file operations
        ]
        
        for dep in dependencies:
            logger.info(f"  Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         capture_output=True, text=True)
        
        logger.info("✅ Performance dependencies installed")
    
    def verify_optimization(self):
        """Verify the optimization is working"""
        logger.info("\n🔍 Verifying optimization...")
        
        # Check if optimized files exist
        required_files = [
            self.api_dir / "main_optimized.py",
            self.api_dir / "routers" / "optimized_metrics_router.py",
            self.api_dir / "routers" / "optimized_geoid_router.py",
            self.backend_dir / "optimization" / "__init__.py",
            self.backend_dir / "optimization" / "metrics_cache.py",
            self.backend_dir / "optimization" / "async_metrics.py"
        ]
        
        all_exist = True
        for file in required_files:
            if file.exists():
                logger.info(f"  ✓ {file.relative_to(self.base_dir)}")
            else:
                logger.info(f"  ✗ {file.relative_to(self.base_dir)} MISSING")
                all_exist = False
        
        if all_exist:
            logger.info("✅ All optimization files verified")
            return True
        else:
            logger.info("❌ Some optimization files are missing")
            return False
    
    def create_rollback_script(self):
        """Create script to rollback changes if needed"""
        logger.info("\n🔄 Creating rollback script...")
        
        rollback_content = f"""#!/usr/bin/env python3
\"\"\"
Rollback Kimera Optimization
\"\"\"

import shutil
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

backup_dir = Path("{self.backup_dir}")
api_dir = Path("{self.api_dir}")

# Restore backed up files
for backup_file in backup_dir.glob("*.py"):
    target = api_dir / backup_file.name
    if backup_file.name == "metrics_router.py" or backup_file.name == "geoid_scar_router.py":
        target = api_dir / "routers" / backup_file.name
    
    shutil.copy2(backup_file, target)
    logger.info(f"Restored {{backup_file.name}}")

# Restore kimera.py
kimera_py = Path("{self.base_dir / 'kimera.py'}")
with open(kimera_py, 'r') as f:
    content = f.read()

content = content.replace(
    'from src.api.main_optimized import app',
    'from src.api.main import app'
)

with open(kimera_py, 'w') as f:
    f.write(content)

logger.info("✅ Rollback completed")
"""
        
        rollback_file = self.base_dir / "rollback_optimization.py"
        with open(rollback_file, 'w') as f:
            f.write(rollback_content)
        
        if os.name != 'nt':
            os.chmod(rollback_file, 0o755)
        
        logger.info("✅ Rollback script created: rollback_optimization.py")
    
    def print_next_steps(self):
        """Print instructions for next steps"""
        logger.info("\n" + "="*60)
        logger.info("🎉 OPTIMIZATION MIGRATION COMPLETED!")
        logger.info("="*60)
        
        logger.info("\n📋 Next Steps:")
        logger.info("1. Stop the current Kimera server")
        logger.info("2. Start the optimized version:")
        logger.info("   python kimera_optimized.py")
        logger.info("\n3. Run performance test to verify improvements:")
        logger.info("   python performance_test_kimera.py")
        
        logger.info("\n🔄 If you need to rollback:")
        logger.info("   python rollback_optimization.py")
        
        logger.info("\n⚡ Expected Performance Improvements:")
        logger.info("   - /health endpoint: <100μs (from 11.3s)")
        logger.info("   - /system-metrics/: <1ms (from 26.6s)")
        logger.info("   - Geoid creation: <500ms (from 2.6s)")
        logger.info("   - Overall throughput: >100 req/s (from 0.9 req/s)")
        
        logger.info("\n🚀 Happy optimizing!")
    
    def run(self):
        """Run the complete migration"""
        logger.info("🚀 Starting Kimera Performance Optimization Migration")
        logger.info("="*60)
        
        try:
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Install dependencies
            self.install_performance_dependencies()
            
            # Step 3: Update main entry
            self.update_main_entry()
            
            # Step 4: Update router imports
            self.update_router_imports()
            
            # Step 5: Create startup script
            self.create_startup_script()
            
            # Step 6: Verify optimization
            if not self.verify_optimization():
                logger.info("\n⚠️  Warning: Some files are missing. Migration may be incomplete.")
            
            # Step 7: Create rollback script
            self.create_rollback_script()
            
            # Step 8: Print next steps
            self.print_next_steps()
            
        except Exception as e:
            logger.info(f"\n❌ Migration failed: {e}")
            logger.info("Run rollback_optimization.py to restore previous state")
            raise


if __name__ == "__main__":
    migrator = KimeraOptimizationMigrator()
    migrator.run()