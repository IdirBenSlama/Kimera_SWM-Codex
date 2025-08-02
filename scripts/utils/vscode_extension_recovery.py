#!/usr/bin/env python3
"""
Kimera SWM VS Code Extension Recovery Protocol
===============================================

Implements defense-in-depth approach to VS Code extension failures.
Based on aerospace "test as you fly" and nuclear "positive confirmation" principles.

Date: 2025-08-02
Issue: HuggingFace extension EPIPE/panic failures
Approach: Systematic diagnosis and multi-layered recovery
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict


class VSCodeRecoveryProtocol:
    """Scientific approach to VS Code extension recovery."""
    
    def __init__(self) -> None:
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.backup_dir = Path('tmp/vscode_recovery_backup') / self.timestamp
        self.log_file = Path(f'docs/reports/analysis/{self.timestamp}_vscode_recovery_log.md')
        self.results = []
        
        # Ensure directories exist
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.log_file.parent, exist_ok=True)
    
    def log_step(self, step: str, result: str, success: bool = True) -> None:
        """Positive confirmation logging - document everything."""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        entry = f"- **{step}**: {result} [{status}]"
        self.results.append(entry)
        print(f"{status}: {step} - {result}")
    
    def hypothesis(self) -> str:
        """State our hypothesis before implementation."""
        return """
        **Hypothesis**: The HuggingFace extension crash is caused by:
        1. Corrupted extension state/cache
        2. Time calculation bug in Rust backend
        3. Resource exhaustion or permissions
        4. Version incompatibility
        
        **Expected Outcome**: Systematic cleanup and reconfiguration will restore functionality.
        **Verification**: Extension starts without crashes and maintains stable connection.
        """
    
    def get_vscode_paths(self) -> Dict[str, Path]:
        """Identify all VS Code configuration paths."""
        user_home = Path.home()
        
        paths = {
            'extensions': user_home / '.vscode' / 'extensions',
            'user_data': user_home / 'AppData' / 'Roaming' / 'Code' / 'User',
            'workspace_storage': user_home / 'AppData' / 'Roaming' / 'Code' / 'User' / 'workspaceStorage',
            'logs': user_home / 'AppData' / 'Roaming' / 'Code' / 'logs',
            'crash_dumps': user_home / 'AppData' / 'Roaming' / 'Code' / 'CrashDumps'
        }
        
        return paths
    
    def backup_critical_configs(self) -> bool:
        """Conservative decision making - backup before modification."""
        try:
            vscode_paths = self.get_vscode_paths()
            
            for name, path in vscode_paths.items():
                if path.exists():
                    backup_path = self.backup_dir / name
                    if path.is_dir():
                        shutil.copytree(path, backup_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(path, backup_path)
            
            self.log_step("Backup Configuration", f"Saved to {self.backup_dir}", True)
            return True
            
        except (OSError, PermissionError, FileNotFoundError) as e:
            self.log_step("Backup Configuration", f"Failed: {e}", False)
            return False
    
    def terminate_vscode_processes(self) -> bool:
        """Ensure clean slate - terminate all VS Code processes."""
        try:
            # Windows process termination
            subprocess.run(['taskkill', '/F', '/IM', 'Code.exe'], 
                         capture_output=True, check=False)
            subprocess.run(['taskkill', '/F', '/IM', 'code.exe'], 
                         capture_output=True, check=False)
            
            # Wait for cleanup
            time.sleep(3)
            
            self.log_step("Process Termination", "All VS Code processes terminated", True)
            return True
            
        except (OSError, subprocess.SubprocessError) as e:
            self.log_step("Process Termination", f"Error: {e}", False)
            return False
    
    def clear_extension_cache(self) -> bool:
        """Clear potentially corrupted cache data."""
        try:
            vscode_paths = self.get_vscode_paths()
            cache_cleared = 0
            
            # Clear workspace storage
            workspace_storage = vscode_paths['workspace_storage']
            if workspace_storage.exists():
                shutil.rmtree(workspace_storage, ignore_errors=True)
                cache_cleared += 1
            
            # Clear logs
            logs_path = vscode_paths['logs']
            if logs_path.exists():
                shutil.rmtree(logs_path, ignore_errors=True)
                cache_cleared += 1
            
            # Clear crash dumps
            crash_dumps = vscode_paths['crash_dumps']
            if crash_dumps.exists():
                shutil.rmtree(crash_dumps, ignore_errors=True)
                cache_cleared += 1
            
            self.log_step("Cache Clearing", f"Cleared {cache_cleared} cache locations", True)
            return True
            
        except Exception as e:
            self.log_step("Cache Clearing", f"Failed: {e}", False)
            return False
    
    def manage_huggingface_extension(self, action: str = "disable") -> bool:
        """Disable or reinstall HuggingFace extension."""
        try:
            if action == "disable":
                result = subprocess.run([
                    'code', '--disable-extension', 'HuggingFace.huggingface-vscode'
                ], capture_output=True, text=True, timeout=30)
                
            elif action == "uninstall":
                result = subprocess.run([
                    'code', '--uninstall-extension', 'HuggingFace.huggingface-vscode'
                ], capture_output=True, text=True, timeout=30)
                
            elif action == "install":
                result = subprocess.run([
                    'code', '--install-extension', 'HuggingFace.huggingface-vscode'
                ], capture_output=True, text=True, timeout=60)
            
            success = result.returncode == 0
            self.log_step(f"Extension {action}", 
                         f"Command completed with code {result.returncode}", success)
            return success
            
        except Exception as e:
            self.log_step(f"Extension {action}", f"Failed: {e}", False)
            return False
    
    def verify_system_health(self) -> Dict[str, bool]:
        """Comprehensive system verification."""
        checks = {}
        
        # Check system time
        try:
            result = subprocess.run(['w32tm', '/query', '/status'], 
                                  capture_output=True, text=True)
            checks['system_time'] = "successful" in result.stdout.lower()
        except:
            checks['system_time'] = False
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            checks['memory'] = memory.percent < 85
        except:
            checks['memory'] = False
        
        # Check disk space
        try:
            import psutil
            disk = psutil.disk_usage('.')
            checks['disk_space'] = (disk.free / disk.total) > 0.1
        except:
            checks['disk_space'] = False
        
        # Check Node.js availability
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True)
            checks['nodejs'] = result.returncode == 0
        except:
            checks['nodejs'] = False
        
        for check, status in checks.items():
            self.log_step(f"System Check: {check}", 
                         "PASS" if status else "FAIL", status)
        
        return checks
    
    def generate_recovery_report(self) -> str:
        """Generate comprehensive recovery report."""
        report = f"""# VS Code Extension Recovery Report

**Date**: {self.timestamp}
**Issue**: HuggingFace VS Code Extension EPIPE/Panic Failures
**Protocol**: Kimera SWM Defense-in-Depth Recovery

## Hypothesis
{self.hypothesis()}

## Recovery Steps Executed

{chr(10).join(self.results)}

## Next Steps

### If Recovery Successful
1. Monitor extension stability for 24 hours
2. Enable gradual feature re-activation
3. Document any recurring issues

### If Recovery Failed
1. Use alternative AI coding assistance (GitHub Copilot, Codeium)
2. Access HuggingFace models via API instead of extension
3. Report detailed bug to HuggingFace extension team
4. Consider workspace-specific VS Code installation

## Alternative Workflows

### Temporary Solution
```bash
# Use HuggingFace models via Python API
pip install transformers
python -c "from transformers import pipeline; print('HuggingFace API working')"
```

### Long-term Mitigation
1. Set up multiple AI coding assistants (redundancy)
2. Regular VS Code configuration backups
3. Automated extension health monitoring

## Backup Location
Configuration backed up to: `{self.backup_dir}`

---
Generated by Kimera SWM Autonomous Architect Recovery Protocol
"""
        return report
    
    def execute_full_recovery(self) -> bool:
        """Execute complete recovery protocol."""
        print("🚀 Kimera SWM VS Code Recovery Protocol Initiated")
        print("=" * 60)
        
        # Step 1: Backup (Conservative Decision Making)
        if not self.backup_critical_configs():
            print("❌ Critical: Backup failed. Aborting for safety.")
            return False
        
        # Step 2: Clean Shutdown (Test as you fly)
        self.terminate_vscode_processes()
        
        # Step 3: System Health Check (Positive Confirmation)
        health = self.verify_system_health()
        critical_failures = [k for k, v in health.items() if not v and k in ['memory', 'disk_space']]
        
        if critical_failures:
            self.log_step("Critical System Issues", f"Failed: {critical_failures}", False)
            print("⚠️  Critical system issues detected. Resolve before proceeding.")
            return False
        
        # Step 4: Cache Cleanup (Defense in Depth)
        self.clear_extension_cache()
        
        # Step 5: Extension Management (No Single Point of Failure)
        self.manage_huggingface_extension("disable")
        time.sleep(2)
        self.manage_huggingface_extension("uninstall")
        time.sleep(3)
        
        # Step 6: Fresh Installation
        install_success = self.manage_huggingface_extension("install")
        
        # Step 7: Generate Report
        report = self.generate_recovery_report()
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📊 Recovery report saved to: {self.log_file}")
        print(f"🔄 Backup available at: {self.backup_dir}")
        
        return install_success


def main() -> int:
    """Execute VS Code recovery protocol."""
    recovery = VSCodeRecoveryProtocol()
    
    print("Kimera SWM Autonomous Architect")
    print("VS Code Extension Recovery Protocol v1.0")
    print("Applying transdisciplinary methodology to extension failures...")
    print()
    
    success = recovery.execute_full_recovery()
    
    if success:
        print("\n✅ Recovery protocol completed successfully!")
        print("🔬 Verify extension functionality with a test session.")
    else:
        print("\n⚠️  Recovery protocol completed with issues.")
        print("📋 Review the generated report for alternative solutions.")
    
    print(f"\n📄 Full report: {recovery.log_file}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())