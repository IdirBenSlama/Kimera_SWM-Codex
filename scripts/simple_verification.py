#!/usr/bin/env python3
"""
Simple Kimera SWM System Verification
"""

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

def main():
    print('🔍 KIMERA SWM SYSTEM VERIFICATION')
    print('=' * 50)

    # Test critical imports
    critical_components = [
        ('KimeraSystem Core', 'src.core.kimera_system'),
        ('GPU Manager', 'src.core.gpu.gpu_manager'),
        ('Vault Manager', 'src.vault.vault_manager'),
        ('Database Schema', 'src.vault.enhanced_database_schema'),
        ('Orchestrator', 'src.orchestration.kimera_orchestrator'),
        ('GPU Router', 'src.api.routers.gpu_router')
    ]

    success_count = 0
    for name, module in critical_components:
        try:
            __import__(module)
            print(f'✅ {name}')
            success_count += 1
        except Exception as e:
            print(f'❌ {name}: {str(e)[:100]}')

    print(f'\nCore Components: {success_count}/{len(critical_components)} available')

    # Test database
    try:
        import sqlite3
        db_path = project_root / 'data/database/kimera_system.db'
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            print(f'✅ Database: {len(tables)} tables')
        else:
            print('⚠️ Database: File not found')
    except Exception as e:
        print(f'❌ Database: {e}')

    # Test GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        else:
            print('⚠️ GPU: CUDA not available')
    except Exception as e:
        print(f'❌ GPU: {e}')

    # Test core integration
    try:
        from src.core.kimera_system import get_kimera_system
        system = get_kimera_system()
        
        print('✅ Core System: Importable')
        
        # Initialize to check status
        system.initialize()
        state = system.get_system_state()
        
        print(f'✅ System State: {state.get("state", "unknown")}')
        print(f'✅ GPU Enabled: {state.get("gpu_acceleration_enabled", False)}')
        print(f'✅ Device: {state.get("device", "unknown")}')
        
    except Exception as e:
        print(f'❌ Core Integration: {e}')

    print('\n🏁 Verification Complete')
    
    return 0 if success_count == len(critical_components) else 1

if __name__ == "__main__":
    sys.exit(main()) 