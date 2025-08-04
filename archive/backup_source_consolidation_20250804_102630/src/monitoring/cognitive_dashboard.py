#!/usr/bin/env python3
"""
Kimera SWM Cognitive Monitoring Dashboard
=========================================

Real-time monitoring dashboard for the Kimera SWM Cognitive Architecture
providing system health, performance metrics, and operational insights.
"""

import asyncio
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Dashboard framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Real-time system metrics"""
    timestamp: str
    system_id: str
    state: str
    uptime: float
    
    # Performance metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_processing_time: float = 0.0
    requests_per_second: float = 0.0
    
    # Resource metrics
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    
    # Component metrics
    active_components: int = 0
    component_health: Dict[str, float] = field(default_factory=dict)
    
    # Cognitive metrics
    insights_generated: int = 0
    patterns_learned: int = 0
    consciousness_events: int = 0
    understanding_quality: float = 0.0


class CognitiveDashboard:
    """Real-time cognitive monitoring dashboard"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Kimera SWM Cognitive Dashboard",
            description="Real-time monitoring for Kimera SWM Cognitive Architecture",
            version="5.0.0"
        )
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Metrics storage
        self.current_metrics: Optional[SystemMetrics] = None
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1000
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup dashboard routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return self._get_dashboard_html()
        
        @self.app.get("/api/metrics")
        async def get_current_metrics():
            """Get current system metrics"""
            if self.current_metrics:
                return self.current_metrics
            return {"error": "No metrics available"}
        
        @self.app.get("/api/metrics/history")
        async def get_metrics_history(hours: int = 1):
            """Get metrics history"""
            # Filter metrics by time range
            cutoff_time = time.time() - (hours * 3600)
            recent_metrics = [
                m for m in self.metrics_history
                if datetime.fromisoformat(m.timestamp.replace('Z', '+00:00')).timestamp() > cutoff_time
            ]
            return {"metrics": recent_metrics, "count": len(recent_metrics)}
        
        @self.app.get("/api/status")
        async def dashboard_status():
            """Dashboard status"""
            return {
                "dashboard": "active",
                "connections": len(self.active_connections),
                "metrics_count": len(self.metrics_history),
                "last_update": self.current_metrics.timestamp if self.current_metrics else None
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect_websocket(websocket)
    
    async def connect_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time updates"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            # Send current metrics immediately
            if self.current_metrics:
                await websocket.send_json({
                    "type": "metrics_update",
                    "data": self.current_metrics.__dict__
                })
            
            # Keep connection alive and handle messages
            while True:
                try:
                    # Wait for messages (ping/pong for keep-alive)
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    if message == "ping":
                        await websocket.send_text("pong")
                    
                except asyncio.TimeoutError:
                    # Send periodic updates
                    if self.current_metrics:
                        await websocket.send_json({
                            "type": "heartbeat",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def update_metrics(self, metrics: SystemMetrics):
        """Update system metrics and broadcast to connected clients"""
        self.current_metrics = metrics
        
        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        # Broadcast to WebSocket clients
        if self.active_connections:
            message = {
                "type": "metrics_update",
                "data": metrics.__dict__
            }
            
            # Send to all connected clients
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error in cognitive_dashboard.py: {e}", exc_info=True)
                    raise  # Re-raise for proper error handling
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Kimera SWM Cognitive Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            min-height: 100vh;
        }
        
        .dashboard-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .dashboard-title {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .dashboard-subtitle {
            font-size: 1.2em;
            opacity: 0.8;
            margin: 10px 0;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #4fc3f7;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .metric-label {
            font-size: 0.9em;
            opacity: 0.7;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-ready { background-color: #4caf50; }
        .status-processing { background-color: #ff9800; }
        .status-error { background-color: #f44336; }
        
        .component-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .component-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .health-bar {
            width: 100px;
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .health-fill {
            height: 100%;
            background: linear-gradient(90deg, #f44336, #ff9800, #4caf50);
            transition: width 0.3s ease;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            background: rgba(0, 0, 0, 0.7);
            font-size: 0.9em;
        }
        
        .connected { color: #4caf50; }
        .disconnected { color: #f44336; }
        
        .last-update {
            text-align: center;
            margin-top: 20px;
            opacity: 0.7;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
        <span class="disconnected">● Connecting...</span>
    </div>

    <div class="dashboard-header">
        <h1 class="dashboard-title">🧠 Kimera SWM</h1>
        <p class="dashboard-subtitle">Cognitive Architecture Dashboard</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">System Status</div>
            <div class="metric-value" id="systemState">
                <span class="status-indicator status-ready"></span>Initializing
            </div>
            <div class="metric-label">Current State</div>
        </div>

        <div class="metric-card">
            <div class="metric-title">Total Operations</div>
            <div class="metric-value" id="totalOps">0</div>
            <div class="metric-label">Processed Requests</div>
        </div>

        <div class="metric-card">
            <div class="metric-title">Success Rate</div>
            <div class="metric-value" id="successRate">0%</div>
            <div class="metric-label">Operation Success</div>
        </div>

        <div class="metric-card">
            <div class="metric-title">Avg Processing Time</div>
            <div class="metric-value" id="avgTime">0ms</div>
            <div class="metric-label">Response Time</div>
        </div>

        <div class="metric-card">
            <div class="metric-title">Active Components</div>
            <div class="metric-value" id="activeComponents">0</div>
            <div class="metric-label">System Components</div>
        </div>

        <div class="metric-card">
            <div class="metric-title">Cognitive Insights</div>
            <div class="metric-value" id="insights">0</div>
            <div class="metric-label">Generated Insights</div>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-title">Component Health</div>
            <div class="component-list" id="componentHealth">
                <div class="component-item">
                    <span>No components</span>
                    <div class="health-bar">
                        <div class="health-fill" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-title">System Performance</div>
            <div style="margin: 10px 0;">
                <div>Uptime: <span id="uptime">0s</span></div>
                <div>Memory: <span id="memory">0%</span></div>
                <div>GPU: <span id="gpu">0%</span></div>
            </div>
        </div>
    </div>

    <div class="last-update" id="lastUpdate">
        Last Update: Never
    </div>

    <script>
        class CognitiveDashboard {
            constructor() {
                this.ws = null;
                this.reconnectInterval = 5000;
                this.connect();
            }

            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('Connected to dashboard');
                    this.updateConnectionStatus(true);
                };
                
                this.ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    if (message.type === 'metrics_update') {
                        this.updateMetrics(message.data);
                    }
                };
                
                this.ws.onclose = () => {
                    console.log('Disconnected from dashboard');
                    this.updateConnectionStatus(false);
                    setTimeout(() => this.connect(), this.reconnectInterval);
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }

            updateConnectionStatus(connected) {
                const status = document.getElementById('connectionStatus');
                if (connected) {
                    status.innerHTML = '<span class="connected">● Connected</span>';
                } else {
                    status.innerHTML = '<span class="disconnected">● Disconnected</span>';
                }
            }

            updateMetrics(metrics) {
                // Update basic metrics
                document.getElementById('systemState').innerHTML = 
                    `<span class="status-indicator status-${metrics.state}"></span>${metrics.state}`;
                document.getElementById('totalOps').textContent = metrics.total_operations || 0;
                
                const successRate = metrics.total_operations > 0 
                    ? (metrics.successful_operations / metrics.total_operations * 100).toFixed(1)
                    : 0;
                document.getElementById('successRate').textContent = successRate + '%';
                
                document.getElementById('avgTime').textContent = 
                    ((metrics.average_processing_time || 0) * 1000).toFixed(0) + 'ms';
                document.getElementById('activeComponents').textContent = metrics.active_components || 0;
                document.getElementById('insights').textContent = metrics.insights_generated || 0;
                
                // Update performance metrics
                document.getElementById('uptime').textContent = this.formatUptime(metrics.uptime || 0);
                document.getElementById('memory').textContent = (metrics.memory_usage || 0).toFixed(1) + '%';
                document.getElementById('gpu').textContent = (metrics.gpu_utilization || 0).toFixed(1) + '%';
                
                // Update component health
                this.updateComponentHealth(metrics.component_health || {});
                
                // Update timestamp
                const now = new Date().toLocaleTimeString();
                document.getElementById('lastUpdate').textContent = `Last Update: ${now}`;
            }

            updateComponentHealth(health) {
                const container = document.getElementById('componentHealth');
                container.innerHTML = '';
                
                if (Object.keys(health).length === 0) {
                    container.innerHTML = '<div class="component-item"><span>No components</span></div>';
                    return;
                }
                
                for (const [component, healthValue] of Object.entries(health)) {
                    const item = document.createElement('div');
                    item.className = 'component-item';
                    item.innerHTML = `
                        <span>${component}</span>
                        <div class="health-bar">
                            <div class="health-fill" style="width: ${(healthValue * 100).toFixed(0)}%"></div>
                        </div>
                    `;
                    container.appendChild(item);
                }
            }

            formatUptime(seconds) {
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                const secs = Math.floor(seconds % 60);
                
                if (hours > 0) {
                    return `${hours}h ${minutes}m`;
                } else if (minutes > 0) {
                    return `${minutes}m ${secs}s`;
                } else {
                    return `${secs}s`;
                }
            }
        }

        // Initialize dashboard
        window.addEventListener('load', () => {
            new CognitiveDashboard();
        });
    </script>
</body>
</html>
        """

# Global dashboard instance
cognitive_dashboard = CognitiveDashboard()

# Convenience function to start dashboard
def start_dashboard(host: str = "0.0.0.0", port: int = 8001):
    """Start the cognitive dashboard server"""
    uvicorn.run(
        cognitive_dashboard.app,
        host=host,
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    logger.info("🖥️  Starting Kimera SWM Cognitive Dashboard...")
    logger.info("📊 Dashboard will be available at: http://localhost:8001")
    start_dashboard()