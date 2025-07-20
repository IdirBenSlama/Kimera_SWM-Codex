#!/usr/bin/env python3
"""
Kimera SWM - Complete Monitoring Integration Demo
================================================

Comprehensive demonstration of the state-of-the-art monitoring system
fully integrated with Kimera architecture, showcasing all capabilities.
"""

import asyncio
import time
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any
import threading

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# FastAPI and monitoring imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# Core monitoring system
from backend.monitoring.kimera_monitoring_core import (
    initialize_monitoring,
    MonitoringLevel,
    start_monitoring,
    stop_monitoring,
    get_monitoring_core
)

# Metrics integration
from backend.monitoring.metrics_integration import (
    initialize_metrics_integration,
    MetricIntegrationConfig,
    get_integration_manager
)

# Monitoring routes
from backend.monitoring.monitoring_routes import router as monitoring_router

# Dashboard (optional)
try:
    from backend.monitoring.kimera_dashboard import create_dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


class KimeraMonitoringIntegrationDemo:
    """
    Complete demonstration of Kimera's state-of-the-art monitoring system
    
    Features demonstrated:
    - Real-time system monitoring (CPU, memory, disk, network)
    - GPU and AI workload tracking
    - Kimera-specific cognitive metrics
    - Prometheus metrics export
    - OpenTelemetry distributed tracing
    - Alert generation and management
    - Performance profiling
    - Interactive monitoring dashboard
    - API endpoints for monitoring data
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Kimera SWM - State-of-the-Art Monitoring System",
            description="Comprehensive monitoring for Kimera's cognitive architecture",
            version="1.0.0"
        )
        
        self.monitoring_core = None
        self.integration_manager = None
        self.dashboard_thread = None
        
        logger.info("🧠 Kimera Monitoring Integration Demo initialized")
    
    async def initialize_monitoring_system(self):
        """Initialize the complete monitoring system"""
        
        logger.info("🚀 Initializing state-of-the-art monitoring system...")
        
        # Initialize monitoring core with extreme detail level
        self.monitoring_core = initialize_monitoring(
            monitoring_level=MonitoringLevel.EXTREME,
            enable_tracing=True,
            enable_profiling=True,
            enable_anomaly_detection=True
        )
        
        # Initialize metrics integration
        config = MetricIntegrationConfig(
            enable_request_metrics=True,
            enable_performance_metrics=True,
            track_geoid_operations=True,
            track_scar_operations=True,
            track_contradiction_events=True,
            track_selective_feedback=True,
            track_revolutionary_insights=True
        )
        
        self.integration_manager = initialize_metrics_integration(config)
        
        # Integrate with FastAPI
        self.integration_manager.integrate_with_fastapi(self.app, prometheus_port=9090)
        
        # Add monitoring routes
        self.app.include_router(monitoring_router)
        
        # Add demo routes
        self._add_demo_routes()
        
        logger.info("✅ Monitoring system initialized successfully")
        logger.info("   📊 Monitoring Level: EXTREME")
        logger.debug("   🔬 Tracing: Enabled")
        logger.info("   📈 Profiling: Enabled")
        logger.info("   🚨 Anomaly Detection: Enabled")
        logger.info("   🌐 FastAPI Integration: Complete")
        logger.info("   📊 Prometheus Export: Port 9090")
    
    def _add_demo_routes(self):
        """Add demonstration routes to showcase monitoring"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Main dashboard page"""
            return """
            <html>
            <head>
                <title>Kimera SWM - Monitoring System</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                    .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }
                    .section { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .endpoint { background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 4px; }
                    .status { color: #27ae60; font-weight: bold; }
                    .metric { display: inline-block; margin: 10px; padding: 10px; background: #3498db; color: white; border-radius: 4px; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🧠 Kimera SWM - State-of-the-Art Monitoring System</h1>
                    <p>Comprehensive monitoring with extreme detail tracking for cognitive architecture</p>
                </div>
                
                <div class="section">
                    <h2>📊 System Status</h2>
                    <p class="status">🟢 All monitoring systems operational</p>
                    <div class="metric">System Monitoring</div>
                    <div class="metric">GPU Tracking</div>
                    <div class="metric">Kimera Cognitive Metrics</div>
                    <div class="metric">Performance Profiling</div>
                    <div class="metric">Anomaly Detection</div>
                </div>
                
                <div class="section">
                    <h2>🔗 Monitoring Endpoints</h2>
                    <div class="endpoint"><strong>GET /monitoring/health</strong> - Health check</div>
                    <div class="endpoint"><strong>GET /monitoring/status</strong> - System status</div>
                    <div class="endpoint"><strong>GET /monitoring/metrics/summary</strong> - Metrics summary</div>
                    <div class="endpoint"><strong>GET /monitoring/metrics/system</strong> - System metrics</div>
                    <div class="endpoint"><strong>GET /monitoring/metrics/kimera</strong> - Kimera metrics</div>
                    <div class="endpoint"><strong>GET /monitoring/metrics/gpu</strong> - GPU metrics</div>
                    <div class="endpoint"><strong>GET /monitoring/alerts</strong> - Alert management</div>
                    <div class="endpoint"><strong>GET /monitoring/performance</strong> - Performance data</div>
                    <div class="endpoint"><strong>GET /metrics</strong> - Prometheus export</div>
                </div>
                
                <div class="section">
                    <h2>🧪 Demo Endpoints</h2>
                    <div class="endpoint"><strong>GET /demo/geoid-operation</strong> - Simulate geoid operation</div>
                    <div class="endpoint"><strong>GET /demo/scar-formation</strong> - Simulate scar formation</div>
                    <div class="endpoint"><strong>GET /demo/contradiction-event</strong> - Simulate contradiction</div>
                    <div class="endpoint"><strong>GET /demo/selective-feedback</strong> - Simulate selective feedback</div>
                    <div class="endpoint"><strong>GET /demo/revolutionary-insight</strong> - Simulate insight generation</div>
                    <div class="endpoint"><strong>GET /demo/stress-test</strong> - Performance stress test</div>
                </div>
                
                <div class="section">
                    <h2>⚡ Quick Actions</h2>
                    <p><a href="/monitoring/health">🏥 Health Check</a></p>
                    <p><a href="/monitoring/status">📊 System Status</a></p>
                    <p><a href="/monitoring/dashboard/data">📈 Dashboard Data</a></p>
                    <p><a href="/metrics">🎯 Prometheus Metrics</a></p>
                    <p><a href="/demo/stress-test">🚀 Run Stress Test</a></p>
                </div>
            </body>
            </html>
            """
        
        @self.app.get("/demo/geoid-operation")
        async def demo_geoid_operation():
            """Demonstrate geoid operation monitoring"""
            
            from backend.monitoring.metrics_integration import track_geoid_operation
            
            async with track_geoid_operation("creation", "vault_alpha"):
                # Simulate geoid operation
                await asyncio.sleep(0.1)
                operation_time = time.time()
            
            # Update geoid count
            self.integration_manager.component_integration.update_component_counts(geoid_count=42)
            
            return {
                "operation": "geoid_creation",
                "vault_id": "vault_alpha",
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "message": "Geoid operation completed and tracked"
            }
        
        @self.app.get("/demo/scar-formation")
        async def demo_scar_formation():
            """Demonstrate scar formation monitoring"""
            
            from backend.monitoring.metrics_integration import track_scar_operation
            
            async with track_scar_operation("formation", "cognitive_scar"):
                # Simulate scar formation
                await asyncio.sleep(0.2)
            
            # Update scar count
            self.integration_manager.component_integration.update_component_counts(scar_count=15)
            
            return {
                "operation": "scar_formation",
                "scar_type": "cognitive_scar",
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "message": "Scar formation completed and tracked"
            }
        
        @self.app.get("/demo/contradiction-event")
        async def demo_contradiction_event():
            """Demonstrate contradiction event monitoring"""
            
            from backend.monitoring.metrics_integration import track_contradiction_event
            
            async with track_contradiction_event("logical_inconsistency", severity=7.5):
                # Simulate contradiction processing
                await asyncio.sleep(0.15)
            
            return {
                "event": "contradiction_detection",
                "source": "logical_inconsistency",
                "severity": 7.5,
                "timestamp": datetime.now().isoformat(),
                "status": "resolved",
                "message": "Contradiction event processed and tracked"
            }
        
        @self.app.get("/demo/selective-feedback")
        async def demo_selective_feedback():
            """Demonstrate selective feedback monitoring"""
            
            from backend.monitoring.metrics_integration import track_selective_feedback
            
            async with track_selective_feedback("financial"):
                # Simulate selective feedback operation
                await asyncio.sleep(0.3)
                
                # Record accuracy
                accuracy = 0.924
                if hasattr(self.monitoring_core, 'kimera_prometheus_metrics'):
                    accuracy_metric = self.monitoring_core.kimera_prometheus_metrics.get('selective_feedback_accuracy')
                    if accuracy_metric:
                        accuracy_metric.labels(domain="financial").set(accuracy)
            
            return {
                "operation": "selective_feedback",
                "domain": "financial",
                "accuracy": 0.924,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "message": "Selective feedback operation completed and tracked"
            }
        
        @self.app.get("/demo/revolutionary-insight")
        async def demo_revolutionary_insight():
            """Demonstrate revolutionary insight monitoring"""
            
            # Record revolutionary insight
            breakthrough_score = 8.7
            self.integration_manager.component_integration.record_revolutionary_insight(breakthrough_score)
            
            return {
                "insight": "revolutionary_breakthrough",
                "breakthrough_score": breakthrough_score,
                "timestamp": datetime.now().isoformat(),
                "status": "generated",
                "message": "Revolutionary insight generated and tracked"
            }
        
        @self.app.get("/demo/stress-test")
        async def demo_stress_test():
            """Demonstrate system under load with monitoring"""
            
            logger.info("🚀 Starting monitoring stress test...")
            
            # Simulate various operations concurrently
            tasks = []
            
            # Create multiple concurrent operations
            for i in range(10):
                tasks.extend([
                    self.demo_geoid_operation(),
                    self.demo_scar_formation(),
                    self.demo_contradiction_event(),
                    self.demo_selective_feedback(),
                ])
            
            # Execute all operations
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            # Count successes and failures
            successes = sum(1 for r in results if not isinstance(r, Exception))
            failures = len(results) - successes
            
            return {
                "test": "monitoring_stress_test",
                "total_operations": len(results),
                "successful_operations": successes,
                "failed_operations": failures,
                "duration_seconds": round(duration, 3),
                "operations_per_second": round(len(results) / duration, 1),
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "message": "Stress test completed with full monitoring"
            }
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        
        logger.info("🔄 Starting monitoring system...")
        
        # Start background monitoring
        await start_monitoring()
        
        logger.info("✅ Monitoring system started")
        logger.info("   📊 Real-time metrics collection active")
        logger.debug("   🔍 System resource monitoring active")
        logger.info("   🔥 GPU monitoring active")
        logger.info("   🧠 Kimera cognitive monitoring active")
        logger.info("   🚨 Anomaly detection active")
    
    async def start_dashboard(self):
        """Start the monitoring dashboard (if available)"""
        
        if not DASHBOARD_AVAILABLE:
            logger.warning("⚠️ Dashboard not available - install dependencies")
            return
        
        try:
            def run_dashboard():
                dashboard = create_dashboard("dash", port=8050)
                dashboard.run_dashboard()
            
            self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            self.dashboard_thread.start()
            
            logger.info("🖥️ Monitoring dashboard started on http://localhost:8050")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to start dashboard: {e}")
    
    def run_api_server(self, host="0.0.0.0", port=8000):
        """Run the FastAPI server with monitoring"""
        
        logger.info(f"🌐 Starting Kimera Monitoring API on {host}:{port}")
        logger.info(f"📊 Prometheus metrics available on port 9090")
        logger.info(f"🖥️ Dashboard (if available)
        logger.info()
        logger.info("Available endpoints:")
        logger.info("  🏠 Main page: http://localhost:8000/")
        logger.info("  🏥 Health check: http://localhost:8000/monitoring/health")
        logger.info("  📊 System status: http://localhost:8000/monitoring/status")
        logger.info("  🎯 Prometheus: http://localhost:8000/metrics")
        logger.info("  🚀 Stress test: http://localhost:8000/demo/stress-test")
        logger.info()
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    
    async def run_complete_demo(self):
        """Run the complete monitoring system demonstration"""
        
        logger.info("🧠 Kimera SWM - Complete State-of-the-Art Monitoring Demo")
        logger.info("=" * 60)
        
        try:
            # Initialize monitoring
            await self.initialize_monitoring_system()
            
            # Start monitoring
            await self.start_monitoring()
            
            # Start dashboard (optional)
            await self.start_dashboard()
            
            # Give system time to initialize
            await asyncio.sleep(2)
            
            logger.info("\n🎉 Complete monitoring system is now running!")
            logger.info("=" * 60)
            
            # Run demo operations
            logger.info("🧪 Running demonstration operations...")
            
            # Simulate Kimera operations with monitoring
            operations = [
                ("Geoid Operation", self.demo_geoid_operation()),
                ("Scar Formation", self.demo_scar_formation()),
                ("Contradiction Event", self.demo_contradiction_event()),
                ("Selective Feedback", self.demo_selective_feedback()),
                ("Revolutionary Insight", self.demo_revolutionary_insight()),
            ]
            
            for name, operation in operations:
                logger.info(f"   🔄 {name}...")
                result = await operation
                logger.info(f"   ✅ {result['message']}")
                await asyncio.sleep(1)
            
            logger.info("\n🚀 Starting API server...")
            logger.info("   Use Ctrl+C to stop the server")
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Demo interrupted by user")
        except Exception as e:
            logger.error(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            try:
                await stop_monitoring()
                logger.info("🛑 Monitoring system stopped")
            except:
                pass


async def main():
    """Main demo execution"""
    
    # Create and run demo
    demo = KimeraMonitoringIntegrationDemo()
    
    # Check if we should run the complete demo or just the API
    if len(sys.argv) > 1 and sys.argv[1] == "--api-only":
        # Just initialize and run API
        await demo.initialize_monitoring_system()
        await demo.start_monitoring()
        await demo.start_dashboard()
        demo.run_api_server()
    else:
        # Run complete demo
        await demo.run_complete_demo()
        demo.run_api_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Demo terminated by user")
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 