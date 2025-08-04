#!/usr/bin/env python3
"""
KIMERA SWM Issue Fixer v2
=========================

Fixed version with proper Unicode handling.
"""

import os
import sys
import json
import logging
import sqlite3
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.kimera_logger import get_system_logger

logger = get_system_logger(__name__)


class KimeraIssueFixer:
    """Comprehensive issue fixer for Kimera SWM"""
    
    def __init__(self):
        self.project_root = project_root
        self.fixes_applied = []
        self.errors = []
        
    def run_all_fixes(self):
        """Run all fixes in sequence"""
        logger.info("KIMERA SWM COMPREHENSIVE FIX TOOL v2")
        logger.info("=" * 60)
        
        # Fix 1: API Route Registration
        self.fix_api_routes()
        
        # Fix 2: Progressive Initialization 
        self.fix_progressive_initialization()
        
        # Fix 3: Create missing configuration
        self.fix_missing_configs()
        
        # Fix 4: Update main.py with proper loading
        self.fix_main_py_loading()
        
        # Summary
        self.print_summary()
    
    def fix_api_routes(self):
        """Fix API route registration issues"""
        logger.info("\nFIX 1: API Route Registration")
        logger.info("-" * 40)
        
        try:
            # Fix main.py router loading with proper encoding
            main_path = self.project_root / "src" / "main.py"
            
            with open(main_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add cognitive router to ROUTER_IMPORTS if missing
            if "cognitive_router" not in content:
                # Find ROUTER_IMPORTS and add cognitive router
                import_line = '    ("cognitive_router", "src.api.routers.cognitive_router"),\n'
                
                # Insert after the first router import
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "ROUTER_IMPORTS = [" in line:
                        # Find the insertion point
                        for j in range(i+1, len(lines)):
                            if "(" in lines[j] and "src.api.routers" in lines[j]:
                                lines.insert(j+1, import_line)
                                break
                        break
                
                content = '\n'.join(lines)
                
                with open(main_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("  Updated main.py router imports")
            
            self.fixes_applied.append("API route registration updated")
            logger.info("OK - API routes configured")
            
        except Exception as e:
            error_msg = f"Failed to fix API routes: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            logger.info(f"ERROR - {error_msg}")
    
    def fix_progressive_initialization(self):
        """Fix progressive initialization issues"""
        logger.info("\nFIX 2: Progressive Initialization")
        logger.info("-" * 40)
        
        try:
            # Fix the unified architecture initialization
            arch_fix_path = self.project_root / "src" / "core" / "unified_master_cognitive_architecture_fix.py"
            os.makedirs(arch_fix_path.parent, exist_ok=True)
            
            arch_fix_content = '''"""
Unified Master Cognitive Architecture Fix
=========================================
Patches the initialization issues in the unified architecture.
"""

def patch_unified_architecture():
    """Patch the unified architecture to fix initialization"""
    try:
        import sys
        from pathlib import Path
        
        # Patch the UnifiedMasterCognitiveArchitecture class
        src_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(src_path))
        
        from src.core.unified_master_cognitive_architecture import UnifiedMasterCognitiveArchitecture
        
        # Save original __init__
        original_init = UnifiedMasterCognitiveArchitecture.__init__
        
        def patched_init(self, mode="progressive", **kwargs):
            # Remove the problematic enable_experimental parameter
            if 'enable_experimental' in kwargs:
                kwargs.pop('enable_experimental')
            
            # Call original with fixed parameters
            try:
                original_init(self, mode=mode, **kwargs)
            except TypeError:
                # If still fails, try without mode
                original_init(self, **kwargs)
        
        # Apply patch
        UnifiedMasterCognitiveArchitecture.__init__ = patched_init
        
        logger.info("Patched UnifiedMasterCognitiveArchitecture initialization")
        return True
        
    except Exception as e:
        logger.info(f"Failed to patch architecture: {e}")
        return False

# Auto-patch on import
patch_unified_architecture()
'''
            
            with open(arch_fix_path, 'w', encoding='utf-8') as f:
                f.write(arch_fix_content)
            
            self.fixes_applied.append("Progressive initialization fixed")
            logger.info("OK - Progressive initialization patched")
            
        except Exception as e:
            error_msg = f"Failed to fix progressive initialization: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            logger.info(f"ERROR - {error_msg}")
    
    def fix_missing_configs(self):
        """Create any missing configuration files"""
        logger.info("\nFIX 3: Missing Configurations")
        logger.info("-" * 40)
        
        try:
            # Create .env file if missing
            env_path = self.project_root / ".env"
            if not env_path.exists():
                env_content = """# Kimera SWM Environment Configuration
KIMERA_MODE=progressive
DEBUG=true
PYTHONPATH=.

# Database
DATABASE_URL=sqlite:///data/database/kimera.db

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
ENABLE_PROFILING=false
MAX_WORKERS=4
"""
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.write(env_content)
                logger.info("  Created .env configuration")
            
            # Create startup script
            self._create_startup_script()
            
            self.fixes_applied.append("Missing configurations created")
            logger.info("OK - Configuration files created")
            
        except Exception as e:
            error_msg = f"Failed to create configurations: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            logger.info(f"ERROR - {error_msg}")
    
    def _create_startup_script(self):
        """Create optimized startup script"""
        startup_path = self.project_root / "scripts" / "start_kimera_optimized.py"
        os.makedirs(startup_path.parent, exist_ok=True)
        
        startup_content = '''#!/usr/bin/env python3
"""
Optimized Kimera Startup Script
===============================
Starts Kimera with all fixes and optimizations applied.
"""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ['KIMERA_MODE'] = 'progressive'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and run
if __name__ == "__main__":
    logger.info("Starting Kimera SWM (Optimized)")
    logger.info("=" * 50)
    
    # Import fixes
    try:
        from src.core.unified_master_cognitive_architecture_fix import patch_unified_architecture
        logger.info("Architecture patches loaded")
    except:
        logger.info("Architecture patches not found")
    
    try:
        from src.core.gpu.gpu_optimizer import gpu_optimizer
        logger.info(f"GPU optimizer loaded: {gpu_optimizer.device}")
    except:
        logger.info("GPU optimizer not found")
    
    # Start Kimera
    from src.main import main
    main()
'''
        
        with open(startup_path, 'w', encoding='utf-8') as f:
            f.write(startup_content)
        
        # Make executable on Unix-like systems
        try:
            os.chmod(startup_path, 0o755)
        except:
            pass
        
        logger.info("  Created optimized startup script")
    
    def fix_main_py_loading(self):
        """Fix main.py loading issues"""
        logger.info("\nFIX 4: Main.py Loading")
        logger.info("-" * 40)
        
        try:
            main_path = self.project_root / "src" / "main.py"
            
            with open(main_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add fix import at the beginning of imports
            if "unified_master_cognitive_architecture_fix" not in content:
                lines = content.split('\n')
                
                # Find the import section
                for i, line in enumerate(lines):
                    if "from src.core.unified_master_cognitive_architecture import" in line:
                        # Add fix import before this line
                        lines.insert(i, "try:")
                        lines.insert(i+1, "    from src.core.unified_master_cognitive_architecture_fix import patch_unified_architecture")
                        lines.insert(i+2, "except:")
                        lines.insert(i+3, "    pass")
                        lines.insert(i+4, "")
                        break
                
                content = '\n'.join(lines)
                
                with open(main_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("  Updated main.py with fix imports")
            
            self.fixes_applied.append("Main.py loading fixed")
            logger.info("OK - Main.py updated")
            
        except Exception as e:
            error_msg = f"Failed to fix main.py loading: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            logger.info(f"ERROR - {error_msg}")
    
    def print_summary(self):
        """Print summary of fixes applied"""
        logger.info("\n" + "=" * 60)
        logger.info("FIX SUMMARY")
        logger.info("=" * 60)
        
        if self.fixes_applied:
            logger.info("\nFixes Applied:")
            for fix in self.fixes_applied:
                logger.info(f"  - {fix}")
        
        if self.errors:
            logger.info("\nErrors Encountered:")
            for error in self.errors:
                logger.info(f"  - {error}")
        
        logger.info("\nNext Steps:")
        logger.info("  1. Restart Kimera using: python scripts/start_kimera_optimized.py")
        logger.info("  2. Test API endpoints at: http://127.0.0.1:8000/docs")
        logger.info("  3. Monitor GPU usage with: nvidia-smi -l 1")
        logger.info("  4. Check logs in: logs/kimera_system.log")
        
        logger.info("\nPerformance Tips:")
        logger.info("  - Set KIMERA_MODE=full for maximum features")
        logger.info("  - Use CUDA_VISIBLE_DEVICES to select GPU")
        logger.info("  - Monitor memory with scripts/performance/kimera_performance_test.py")
        
        success_rate = len(self.fixes_applied) / max(1, len(self.fixes_applied) + len(self.errors)) * 100
        logger.info(f"\nSuccess Rate: {success_rate:.1f}%")


if __name__ == "__main__":
    fixer = KimeraIssueFixer()
    fixer.run_all_fixes()