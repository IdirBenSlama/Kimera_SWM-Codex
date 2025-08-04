#!/usr/bin/env python3
"""
KIMERA SWM Proper Startup Script
Following KIMERA Protocol v3.0 - Aerospace-Grade System Launch

This script provides a robust startup procedure with comprehensive error handling
and system verification following zero-trust principles.
"""

import os
import sys
import time
import socket
import subprocess
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class KimeraSWMLauncher:
    """
    Aerospace-grade KIMERA SWM launcher with comprehensive pre-flight checks
    and automated issue resolution.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.report_path = f'docs/reports/health/{self.date_str}_system_startup.md'

        # Ensure reports directory exists
        os.makedirs('docs/reports/health', exist_ok=True)

        self.startup_issues = []
        self.startup_successes = []

    def log_action(self, message: str, level: str = "INFO"):
        """Log with timestamp following KIMERA documentation standards"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"[{timestamp}] {level}: {message}")
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)

    def setup_python_environment(self) -> bool:
        """Setup Python environment for proper module imports"""
        self.log_action("Setting up Python environment...")

        try:
            # Add project root to Python path
            project_root_str = str(self.project_root.absolute())
            if project_root_str not in sys.path:
                sys.path.insert(0, project_root_str)

            # Set PYTHONPATH environment variable
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            if project_root_str not in current_pythonpath:
                if current_pythonpath:
                    os.environ['PYTHONPATH'] = f"{project_root_str}{os.pathsep}{current_pythonpath}"
                else:
                    os.environ['PYTHONPATH'] = project_root_str

            self.log_action(f"✅ Python path configured: {project_root_str}")
            self.startup_successes.append("Python environment setup")
            return True

        except Exception as e:
            self.log_action(f"❌ Failed to setup Python environment: {e}", "ERROR")
            self.startup_issues.append(f"Python environment setup failed: {e}")
            return False

    def verify_database_systems(self) -> bool:
        """Verify database systems are operational"""
        self.log_action("Verifying database systems...")

        try:
            # Check if our databases exist
            db_path = self.project_root / 'data' / 'databases' / 'kimera_swm.db'
            if not db_path.exists():
                self.log_action("❌ KIMERA database not found - running initialization...", "WARNING")
                return self.initialize_databases()

            self.log_action("✅ Database systems verified")
            self.startup_successes.append("Database verification")
            return True

        except Exception as e:
            self.log_action(f"❌ Database verification failed: {e}", "ERROR")
            self.startup_issues.append(f"Database verification failed: {e}")
            return False

    def initialize_databases(self) -> bool:
        """Initialize databases if not present"""
        self.log_action("Initializing database systems...")

        try:
            # Import and run database initialization
            from scripts.database_setup.initialize_kimera_databases import KimeraDatabaseInitializer

            initializer = KimeraDatabaseInitializer()
            success = initializer.run_initialization()

            if success:
                self.log_action("✅ Database initialization completed")
                self.startup_successes.append("Database initialization")
                return True
            else:
                self.log_action("⚠️ Database initialization completed with warnings", "WARNING")
                self.startup_issues.append("Database initialization had warnings")
                return True  # Continue anyway with warnings

        except Exception as e:
            self.log_action(f"❌ Database initialization failed: {e}", "ERROR")
            self.startup_issues.append(f"Database initialization failed: {e}")
            return False

    def check_port_availability(self, start_port: int = 8000, end_port: int = 8010) -> int:
        """Find an available port for the server"""
        self.log_action(f"Finding available port in range {start_port}-{end_port}...")

        for port in range(start_port, end_port + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('127.0.0.1', port))
                    self.log_action(f"✅ Port {port} is available")
                    return port
            except OSError:
                continue

        # If no port in range, let system choose
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
            self.log_action(f"✅ System assigned port {port}")
            return port

    def create_startup_environment(self) -> dict:
        """Create environment variables for startup"""
        self.log_action("Creating startup environment...")

        env = os.environ.copy()

        # Core environment variables
        env.update({
            'KIMERA_MODE': 'production',
            'DEBUG_MODE': 'false',
            'LOG_LEVEL': 'INFO',
            'PYTHONPATH': str(self.project_root.absolute()),
            'KIMERA_PROJECT_ROOT': str(self.project_root.absolute())
        })

        self.log_action("✅ Startup environment configured")
        self.startup_successes.append("Environment configuration")
        return env

    def start_kimera_system(self, port: int) -> subprocess.Popen:
        """Start the KIMERA SWM system"""
        self.log_action("Starting KIMERA SWM system...")

        try:
            # Prepare environment
            env = self.create_startup_environment()
            env['KIMERA_PORT'] = str(port)

            # Change to project directory
            os.chdir(self.project_root)

            # Start the system using the proper module path
            cmd = [sys.executable, '-m', 'src.main']

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            self.log_action(f"✅ KIMERA SWM started with PID {process.pid}")
            self.log_action(f"🌐 Server will be available at: http://127.0.0.1:{port}")
            self.log_action(f"📚 API Documentation: http://127.0.0.1:{port}/docs")

            return process

        except Exception as e:
            self.log_action(f"❌ Failed to start KIMERA system: {e}", "ERROR")
            self.startup_issues.append(f"System startup failed: {e}")
            raise

    def monitor_startup(self, process: subprocess.Popen, port: int, timeout: int = 60) -> bool:
        """Monitor system startup and verify it's responding"""
        self.log_action("Monitoring system startup...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process is still running
            if process.poll() is not None:
                self.log_action("❌ Process terminated unexpectedly", "ERROR")
                return False

            # Check if port is responding
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    if sock.connect_ex(('127.0.0.1', port)) == 0:
                        self.log_action("✅ System is responding on the network")
                        self.startup_successes.append("Network connectivity")
                        return True
            except:
                pass

            time.sleep(2)

        self.log_action("⚠️ System startup timeout - may still be initializing", "WARNING")
        self.startup_issues.append("Startup timeout reached")
        return False

    def generate_startup_report(self, success: bool, port: int = None):
        """Generate startup report"""

        report_content = f"""# KIMERA SWM System Startup Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Startup Script: scripts/startup/launch_kimera_swm.py

## Executive Summary
- **Startup Status**: {'✅ SUCCESS' if success else '❌ FAILED'}
- **Successful Steps**: {len(self.startup_successes)}
- **Issues Encountered**: {len(self.startup_issues)}
- **Server Port**: {port if port else 'N/A'}

## Startup Sequence Results

### ✅ Successful Steps
{self._format_list(self.startup_successes) if self.startup_successes else '*None*'}

### ⚠️ Issues Encountered
{self._format_list(self.startup_issues) if self.startup_issues else '*None*'}

## System Access Information

{f'''### 🌐 Web Interface
- **Main URL**: http://127.0.0.1:{port}
- **API Documentation**: http://127.0.0.1:{port}/docs
- **Health Check**: http://127.0.0.1:{port}/health
- **Interactive API**: http://127.0.0.1:{port}/redoc''' if port else '**System not accessible** - startup failed'}

## Next Steps

### If System is Running
1. **Access Web Interface**: Open browser to main URL
2. **Test API Endpoints**: Use the interactive documentation
3. **Monitor Logs**: Watch for any ongoing issues
4. **Run Health Checks**: Verify all subsystems are operational

### If System Failed to Start
1. **Review Issues**: Check the issues list above
2. **Check Dependencies**: Ensure all required packages are installed
3. **Verify Databases**: Run database initialization script
4. **Check Logs**: Look for detailed error messages

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure PYTHONPATH is set correctly
- **Port Conflicts**: System will automatically find available ports
- **Database Issues**: Run `python scripts/database_setup/initialize_kimera_databases.py`
- **Permission Issues**: Ensure write access to data directories

### Support Commands
```bash
# Check system health
python scripts/health_check/verify_installation.py

# Reinitialize databases
python scripts/database_setup/initialize_kimera_databases.py

# Force restart
python scripts/startup/launch_kimera_swm.py
```

---
*Generated by KIMERA SWM Autonomous Architect v3.0*
*Following Protocol: Extreme Rigor + Breakthrough Creativity*
"""

        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        self.log_action(f"✅ Startup report saved to: {self.report_path}")

    def _format_list(self, items: list) -> str:
        """Format list for markdown"""
        return '\n'.join([f"- {item}" for item in items])

    def launch(self) -> bool:
        """Execute the complete launch sequence"""
        self.log_action("="*80)
        self.log_action("KIMERA SWM System Launch")
        self.log_action("Following KIMERA Protocol v3.0 - Aerospace-Grade Startup")
        self.log_action("="*80)

        try:
            # Phase 1: Environment setup
            if not self.setup_python_environment():
                self.generate_startup_report(False)
                return False

            # Phase 2: Database verification
            if not self.verify_database_systems():
                self.generate_startup_report(False)
                return False

            # Phase 3: Port allocation
            port = self.check_port_availability()

            # Phase 4: System startup
            process = self.start_kimera_system(port)

            # Phase 5: Startup monitoring
            startup_success = self.monitor_startup(process, port)

            # Phase 6: Report generation
            self.generate_startup_report(startup_success, port)

            if startup_success:
                self.log_action("="*80)
                self.log_action("🎉 KIMERA SWM SYSTEM SUCCESSFULLY LAUNCHED!")
                self.log_action(f"🌐 Access your system at: http://127.0.0.1:{port}")
                self.log_action(f"📚 API Documentation: http://127.0.0.1:{port}/docs")
                self.log_action("="*80)

                # Keep the process running
                try:
                    while True:
                        if process.poll() is not None:
                            self.log_action("System process terminated", "WARNING")
                            break
                        time.sleep(5)
                except KeyboardInterrupt:
                    self.log_action("Shutdown requested by user")
                    process.terminate()
                    process.wait()

                return True
            else:
                self.log_action("⚠️ System started but may have limited functionality", "WARNING")
                return False

        except Exception as e:
            self.log_action(f"❌ Launch sequence failed: {e}", "ERROR")
            self.startup_issues.append(f"Launch sequence failure: {e}")
            self.generate_startup_report(False)
            return False

def main():
    """Main launcher function"""
    launcher = KimeraSWMLauncher()
    success = launcher.launch()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
