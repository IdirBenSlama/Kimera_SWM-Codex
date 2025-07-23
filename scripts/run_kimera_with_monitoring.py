#!/usr/bin/env python3
"""
Kimera SWM System Runner with Real-Time Vault Monitoring
========================================================

This script starts the complete Kimera system and activates comprehensive
real-time vault monitoring. It provides:

1. System initialization and health checks
2. Real-time vault activity monitoring
3. Cognitive state tracking
4. Performance analytics
5. Anomaly detection
6. Comprehensive logging and console output

Usage:
    python run_kimera_with_monitoring.py [options]
    
Options:
    --port PORT          FastAPI server port (default: 8000)
    --host HOST          FastAPI server host (default: 0.0.0.0)
    --log-level LEVEL    Logging level (default: INFO)
    --no-monitoring      Skip automatic vault monitoring startup
    --monitoring-interval N   Monitoring check interval in seconds (default: 5)
"""

import asyncio
import logging
import signal
import sys
import time
import threading
import argparse
from datetime import datetime
from typing import Optional
import json

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("KIMERA_LAUNCHER")

# Import FastAPI and uvicorn after logging setup
try:
    import uvicorn
    from fastapi import FastAPI
    import requests
    import torch
except ImportError as e:
    logger.error(f"Required dependencies not installed: {e}")
    logger.error("Please run: pip install fastapi uvicorn requests torch")
    sys.exit(1)

class KimeraSystemRunner:
    """
    Comprehensive Kimera system runner with real-time monitoring capabilities.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000, 
                 enable_monitoring: bool = True, monitoring_interval: int = 5):
        self.host = host
        self.port = port
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self.base_url = f"http://localhost:{port}"
        
        # System state
        self.server_process = None
        self.monitoring_thread = None
        self.is_running = False
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Kimera System Runner initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self._shutdown()
    
    def start_system(self):
        """Start the complete Kimera system"""
        try:
            logger.info("=" * 80)
            logger.info("🌟 KIMERA SWM ALPHA PROTOTYPE SYSTEM STARTUP")
            logger.info("=" * 80)
            
            # System information
            self._display_system_info()
            
            # Start FastAPI server in a separate thread
            logger.info("🔧 Starting FastAPI server...")
            self._start_fastapi_server()
            
            # Wait for server to be ready
            self._wait_for_server_ready()
            
            # Initialize vault monitoring if enabled
            if self.enable_monitoring:
                self._initialize_vault_monitoring()
            
            # Display system status
            self._display_system_status()
            
            # Start main monitoring loop
            self._start_monitoring_loop()
            
        except Exception as e:
            logger.error(f"❌ Failed to start Kimera system: {e}", exc_info=True)
            self._shutdown()
    
    def _display_system_info(self):
        """Display system information"""
        logger.info("📊 System Information:")
        logger.info(f"   🐍 Python: {sys.version.split()[0]}")
        logger.info(f"   🔥 PyTorch: {torch.__version__}")
        logger.info(f"   🚀 GPU Available: {'✅' if torch.cuda.is_available() else '❌'}")
        if torch.cuda.is_available():
            logger.info(f"   🎮 GPU: {torch.cuda.get_device_name()}")
        logger.info(f"   🌐 Server: {self.host}:{self.port}")
        logger.info(f"   📊 Monitoring: {'✅' if self.enable_monitoring else '❌'}")
        logger.info("")
    
    def _start_fastapi_server(self):
        """Start the FastAPI server in a background thread"""
        def run_server():
            try:
                # Import the FastAPI app
                from src.main import app
                
                # Configure uvicorn
                config = uvicorn.Config(
                    app=app,
                    host=self.host,
                    port=self.port,
                    log_level="info",
                    reload=False,
                    workers=1
                )
                
                server = uvicorn.Server(config)
                asyncio.run(server.serve())
                
            except Exception as e:
                logger.error(f"❌ FastAPI server error: {e}")
                self.is_running = False
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        logger.info("🔄 FastAPI server thread started")
    
    def _wait_for_server_ready(self, timeout: int = 60):
        """Wait for the FastAPI server to be ready"""
        logger.info("⏳ Waiting for server to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Server is ready!")
                    self.is_running = True
                    return
            except Exception:
                pass
            
            time.sleep(2)
        
        raise Exception(f"Server failed to start within {timeout} seconds")
    
    def _initialize_vault_monitoring(self):
        """Initialize the vault monitoring system"""
        try:
            logger.info("🔍 Initializing vault monitoring...")
            
            # Check monitoring status
            response = requests.get(f"{self.base_url}/kimera/vault/monitoring/status")
            if response.status_code == 200:
                status = response.json()
                logger.info(f"📊 Monitoring system status: {status.get('status')}")
                
                # Start monitoring if not already running
                if not status.get('is_monitoring', False):
                    start_response = requests.post(f"{self.base_url}/kimera/vault/monitoring/start")
                    if start_response.status_code == 200:
                        result = start_response.json()
                        logger.info("✅ Vault monitoring started successfully")
                        logger.info(f"   📈 Monitoring interval: {result.get('monitoring_interval')}s")
                        logger.info(f"   🏥 Health check interval: {result.get('health_check_interval')}s")
                    else:
                        logger.warning(f"⚠️ Failed to start monitoring: {start_response.text}")
                else:
                    logger.info("📊 Vault monitoring already active")
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize vault monitoring: {e}")
    
    def _display_system_status(self):
        """Display comprehensive system status"""
        try:
            logger.info("=" * 80)
            logger.info("📊 KIMERA SYSTEM STATUS")
            logger.info("=" * 80)
            
            # Get system health
            health_response = requests.get(f"{self.base_url}/health")
            if health_response.status_code == 200:
                health = health_response.json()
                logger.info(f"🏥 System Health: {health.get('status')}")
                logger.info(f"🔧 Kimera Status: {health.get('kimera_status')}")
                logger.info(f"🎮 GPU Available: {'✅' if health.get('gpu_available') else '❌'}")
                if health.get('gpu_name'):
                    logger.info(f"   GPU: {health.get('gpu_name')}")
            
            # Get monitoring health
            monitoring_response = requests.get(f"{self.base_url}/monitoring/health")
            if monitoring_response.status_code == 200:
                monitoring = monitoring_response.json()
                logger.info(f"📊 Monitoring Health: {monitoring.get('status')}")
                components = monitoring.get('components', {})
                
                # Count operational engines
                engine_components = {k: v for k, v in components.items() if k.startswith('engine_')}
                operational = sum(1 for v in engine_components.values() if v == 'healthy')
                total = len(engine_components)
                logger.info(f"⚙️ Engines: {operational}/{total} operational")
                
                if operational < total:
                    logger.info("   Non-operational engines:")
                    for engine, status in engine_components.items():
                        if status != 'healthy':
                            logger.info(f"     {engine}: {status}")
            
            # Get vault status
            vault_response = requests.get(f"{self.base_url}/kimera/vault/status")
            if vault_response.status_code == 200:
                vault = vault_response.json()
                logger.info(f"🗄️ Vault Status: {vault.get('database_status', 'Unknown')}")
            
            # Get vault monitoring health
            vault_health_response = requests.get(f"{self.base_url}/kimera/vault/monitoring/health")
            if vault_health_response.status_code == 200:
                vault_health = vault_health_response.json()
                if vault_health.get('status') == 'health_data_available':
                    metrics = vault_health.get('health_metrics', {})
                    logger.info("🧠 Vault Cognitive State:")
                    logger.info(f"   State: {metrics.get('cognitive_state')}")
                    logger.info(f"   Geoids: {metrics.get('total_geoids', 0)}")
                    logger.info(f"   Scars: {metrics.get('total_scars', 0)}")
                    logger.info(f"   Insights: {metrics.get('total_insights', 0)}")
                    logger.info(f"   Activity Rate: {metrics.get('recent_activity_rate', 0):.2f}/min")
                    logger.info(f"   DB Connected: {'✅' if metrics.get('database_connected') else '❌'}")
                    logger.info(f"   DB Latency: {metrics.get('database_latency_ms', 0):.1f}ms")
            
            logger.info("=" * 80)
            logger.info("🎯 System is operational! Access points:")
            logger.info(f"   📱 Web UI: http://localhost:{self.port}")
            logger.info(f"   📚 API Docs: http://localhost:{self.port}/docs")
            logger.info(f"   🔍 Monitoring: http://localhost:{self.port}/monitoring/health")
            logger.info(f"   🗄️ Vault Health: http://localhost:{self.port}/kimera/vault/monitoring/health")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Failed to display system status: {e}")
    
    def _start_monitoring_loop(self):
        """Start the main monitoring loop"""
        logger.info("🔄 Starting monitoring loop...")
        logger.info("📝 Press Ctrl+C to shutdown gracefully")
        logger.info("")
        
        try:
            while self.is_running and not self.shutdown_requested:
                try:
                    # Periodic status check
                    self._periodic_status_check()
                    time.sleep(self.monitoring_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"❌ Error in monitoring loop: {e}")
                    time.sleep(self.monitoring_interval)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()
    
    def _periodic_status_check(self):
        """Perform periodic status checks"""
        try:
            # Get vault health
            vault_health_response = requests.get(
                f"{self.base_url}/kimera/vault/monitoring/health", 
                timeout=5
            )
            
            if vault_health_response.status_code == 200:
                vault_health = vault_health_response.json()
                if vault_health.get('status') == 'health_data_available':
                    metrics = vault_health.get('health_metrics', {})
                    
                    # Log significant changes or alerts
                    cognitive_state = metrics.get('cognitive_state')
                    activity_rate = metrics.get('recent_activity_rate', 0)
                    anomalies = metrics.get('anomalies_detected', 0)
                    
                    # Log activity if detected
                    if activity_rate > 0:
                        logger.info(f"🧠 Cognitive Activity: {cognitive_state} | "
                                   f"Rate: {activity_rate:.2f}/min | "
                                   f"G:{metrics.get('total_geoids', 0)} "
                                   f"S:{metrics.get('total_scars', 0)} "
                                   f"I:{metrics.get('total_insights', 0)}")
                    
                    # Log anomalies
                    if anomalies > 0:
                        logger.warning(f"⚠️ {anomalies} anomalies detected in vault")
                        
                    # Check for new activities
                    activities_response = requests.get(
                        f"{self.base_url}/kimera/vault/monitoring/activities?limit=5"
                    )
                    if activities_response.status_code == 200:
                        activities_data = activities_response.json()
                        activities = activities_data.get('activities', [])
                        
                        # Log recent activities
                        for activity in activities[-3:]:  # Last 3 activities
                            activity_type = activity.get('activity_type', 'unknown')
                            metadata = activity.get('metadata', {})
                            timestamp = activity.get('timestamp', '')
                            
                            if activity_type == 'geoid_creation':
                                logger.info(f"🧠 New geoid formations: +{metadata.get('new_count', 0)} "
                                           f"(total: {metadata.get('total', 0)})")
                            elif activity_type == 'scar_formation':
                                logger.info(f"⚡ New scar formations: +{metadata.get('new_count', 0)} "
                                           f"(total: {metadata.get('total', 0)})")
                            elif activity_type == 'insight_generation':
                                logger.info(f"💡 New insights: +{metadata.get('new_count', 0)} "
                                           f"(total: {metadata.get('total', 0)})")
                                
        except Exception as e:
            logger.debug(f"Periodic check error: {e}")
    
    def _shutdown(self):
        """Shutdown the system gracefully"""
        if self.shutdown_requested:
            return
            
        self.shutdown_requested = True
        logger.info("🛑 Initiating graceful shutdown...")
        
        try:
            # Stop vault monitoring
            if self.enable_monitoring:
                try:
                    requests.post(f"{self.base_url}/kimera/vault/monitoring/stop", timeout=5)
                    logger.info("📊 Vault monitoring stopped")
                except Exception:
                    pass
            
            # Stop system monitoring
            try:
                requests.post(f"{self.base_url}/monitoring/stop", timeout=5)
                logger.info("🔍 System monitoring stopped")
            except Exception:
                pass
            
            self.is_running = False
            logger.info("✅ Kimera system shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
        
        finally:
            sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Kimera SWM System Runner with Monitoring")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="FastAPI server host")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--no-monitoring", action="store_true", 
                       help="Skip automatic vault monitoring startup")
    parser.add_argument("--monitoring-interval", type=int, default=5,
                       help="Monitoring check interval in seconds")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and start the system runner
    runner = KimeraSystemRunner(
        host=args.host,
        port=args.port,
        enable_monitoring=not args.no_monitoring,
        monitoring_interval=args.monitoring_interval
    )
    
    runner.start_system()

if __name__ == "__main__":
    main() 