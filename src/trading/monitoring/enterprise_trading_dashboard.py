"""
Enterprise Trading Dashboard and Monitoring System

Provides enterprise-level monitoring, control, and intervention capabilities
for Kimera's autonomous trading operations.

Features:
- Real-time trading dashboard
- Risk monitoring and alerts
- Performance analytics
- Manual intervention controls
- Compliance monitoring
- System health tracking
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Web framework for dashboard
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Data visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Metrics and monitoring
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from src.trading.core.advanced_autonomous_engine import AdvancedAutonomousEngine, AutonomyLevel, DecisionSpeed
from src.utils.kimera_logger import get_logger, LogCategory

logger = get_logger(__name__, category=LogCategory.TRADING)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class InterventionType(Enum):
    """Types of manual interventions"""
    PAUSE_TRADING = "pause_trading"
    RESUME_TRADING = "resume_trading"
    EMERGENCY_STOP = "emergency_stop"
    CLOSE_POSITION = "close_position"
    CANCEL_ORDER = "cancel_order"
    ADJUST_RISK_LIMITS = "adjust_risk_limits"
    CHANGE_AUTONOMY_LEVEL = "change_autonomy_level"
    APPROVE_DECISION = "approve_decision"
    REJECT_DECISION = "reject_decision"


@dataclass
class TradingAlert:
    """Trading system alert"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    component: str
    data: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class ManualIntervention:
    """Manual intervention record"""
    intervention_id: str
    intervention_type: InterventionType
    user_id: str
    timestamp: datetime
    parameters: Dict[str, Any]
    reason: str
    status: str
    result: Optional[Dict[str, Any]] = None


class EnterpriseTradingDashboard:
    """
    Enterprise-level trading dashboard and monitoring system
    
    Provides comprehensive monitoring, control, and intervention capabilities
    for autonomous trading operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Enterprise Trading Dashboard
        
        Args:
            config: Dashboard configuration including:
                - port: Web server port
                - host: Host address
                - auth_enabled: Enable authentication
                - alert_thresholds: Alert configuration
                - update_intervals: Data update frequencies
        """
        self.config = config
        self.host = config.get("host", "0.0.0.0")
        self.port = config.get("port", 8080)
        self.auth_enabled = config.get("auth_enabled", True)
        
        # Initialize web application
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Kimera Enterprise Trading Dashboard",
                description="Real-time monitoring and control for autonomous trading",
                version="1.0.0"
            )
            self._setup_routes()
        else:
            logger.error("FastAPI not available - dashboard disabled")
            self.app = None
        
        # Trading engine reference
        self.trading_engine: Optional[AdvancedAutonomousEngine] = None
        
        # Real-time data storage
        self.real_time_data = {
            "trading_metrics": deque(maxlen=1000),
            "performance_data": deque(maxlen=1000),
            "risk_metrics": deque(maxlen=1000),
            "system_health": deque(maxlen=1000),
            "cognitive_state": deque(maxlen=1000)
        }
        
        # Alert system
        self.alerts = deque(maxlen=10000)
        self.alert_history = deque(maxlen=100000)
        self.alert_thresholds = config.get("alert_thresholds", {})
        
        # Intervention tracking
        self.interventions = deque(maxlen=10000)
        self.pending_approvals = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        # Metrics collection
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
        
        # Control flags
        self.is_running = False
        self.dashboard_server = None
        
        logger.info("🖥️ Enterprise Trading Dashboard initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        if not self.app:
            return
        
        # Enable CORS for web interface
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Main dashboard route
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return self._generate_dashboard_html()
        
        # API routes
        @self.app.get("/api/status")
        async def get_system_status():
            return await self._get_system_status()
        
        @self.app.get("/api/trading-metrics")
        async def get_trading_metrics():
            return await self._get_trading_metrics()
        
        @self.app.get("/api/performance")
        async def get_performance_data():
            return await self._get_performance_data()
        
        @self.app.get("/api/risk-metrics")
        async def get_risk_metrics():
            return await self._get_risk_metrics()
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            return await self._get_alerts()
        
        @self.app.get("/api/opportunities")
        async def get_opportunities():
            return await self._get_opportunities()
        
        @self.app.get("/api/decisions")
        async def get_decisions():
            return await self._get_decisions()
        
        # Control routes
        @self.app.post("/api/intervention/{intervention_type}")
        async def manual_intervention(intervention_type: str, parameters: Dict[str, Any]):
            return await self._handle_manual_intervention(intervention_type, parameters)
        
        @self.app.post("/api/approve-decision/{decision_id}")
        async def approve_decision(decision_id: str):
            return await self._approve_decision(decision_id)
        
        @self.app.post("/api/reject-decision/{decision_id}")
        async def reject_decision(decision_id: str):
            return await self._reject_decision(decision_id)
        
        @self.app.post("/api/emergency-stop")
        async def emergency_stop():
            return await self._emergency_stop()
        
        # WebSocket for real-time updates
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket_connection(websocket)
        
        # Prometheus metrics endpoint
        if PROMETHEUS_AVAILABLE:
            @self.app.get("/metrics")
            async def metrics():
                return generate_latest()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.metrics = {
            "trades_total": Counter("kimera_trades_total", "Total number of trades", ["symbol", "side", "status"]),
            "pnl_total": Gauge("kimera_pnl_total", "Total P&L"),
            "active_positions": Gauge("kimera_active_positions", "Number of active positions"),
            "decision_latency": Histogram("kimera_decision_latency_seconds", "Decision making latency"),
            "execution_latency": Histogram("kimera_execution_latency_seconds", "Trade execution latency"),
            "risk_score": Gauge("kimera_risk_score", "Current risk score"),
            "cognitive_coherence": Gauge("kimera_cognitive_coherence", "Cognitive coherence level"),
            "alerts_total": Counter("kimera_alerts_total", "Total alerts", ["level", "component"]),
            "interventions_total": Counter("kimera_interventions_total", "Manual interventions", ["type"])
        }
    
    async def start_dashboard(self):
        """Start the dashboard server"""
        if not FASTAPI_AVAILABLE or not self.app:
            logger.error("❌ Cannot start dashboard - FastAPI not available")
            return
        
        logger.info(f"🖥️ Starting Enterprise Trading Dashboard on {self.host}:{self.port}")
        
        self.is_running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._data_collection_loop()),
            asyncio.create_task(self._alert_monitoring_loop()),
            asyncio.create_task(self._websocket_broadcast_loop()),
            asyncio.create_task(self._performance_analysis_loop())
        ]
        
        # Start web server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        try:
            logger.info(f"🌐 Dashboard available at: http://{self.host}:{self.port}")
            await server.serve()
        except Exception as e:
            logger.error(f"❌ Dashboard server error: {str(e)}")
        finally:
            self.is_running = False
            # Cancel background tasks
            for task in tasks:
                task.cancel()
    
    async def stop_dashboard(self):
        """Stop the dashboard server"""
        logger.info("🛑 Stopping Enterprise Trading Dashboard...")
        self.is_running = False
    
    def set_trading_engine(self, engine: AdvancedAutonomousEngine):
        """Set reference to trading engine"""
        self.trading_engine = engine
        logger.info("🔗 Trading engine connected to dashboard")
    
    async def _data_collection_loop(self):
        """Collect real-time data from trading engine"""
        while self.is_running:
            try:
                if not self.trading_engine:
                    await asyncio.sleep(1)
                    continue
                
                timestamp = datetime.now()
                
                # Collect trading metrics
                trading_metrics = {
                    "timestamp": timestamp,
                    "active_opportunities": len(self.trading_engine.active_opportunities),
                    "active_decisions": len(self.trading_engine.active_decisions),
                    "decision_queue_size": len(self.trading_engine.decision_queue),
                    "avg_decision_time": np.mean(list(self.trading_engine.execution_times)) if self.trading_engine.execution_times else 0
                }
                self.real_time_data["trading_metrics"].append(trading_metrics)
                
                # Collect performance data
                performance_data = {
                    "timestamp": timestamp,
                    **self.trading_engine.performance_state.get("metrics", {})
                }
                self.real_time_data["performance_data"].append(performance_data)
                
                # Collect risk metrics
                risk_data = {
                    "timestamp": timestamp,
                    **self.trading_engine.risk_state.get("metrics", {})
                }
                self.real_time_data["risk_metrics"].append(risk_data)
                
                # Collect system health
                health_data = {
                    "timestamp": timestamp,
                    **self.trading_engine.system_health
                }
                self.real_time_data["system_health"].append(health_data)
                
                # Collect cognitive state
                cognitive_data = {
                    "timestamp": timestamp,
                    **self.trading_engine.cognitive_state
                }
                self.real_time_data["cognitive_state"].append(cognitive_data)
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    self._update_prometheus_metrics(trading_metrics, performance_data, risk_data)
                
                await asyncio.sleep(1)  # Collect every second
                
            except Exception as e:
                logger.error(f"❌ Data collection error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _alert_monitoring_loop(self):
        """Monitor for alert conditions"""
        while self.is_running:
            try:
                if not self.trading_engine:
                    await asyncio.sleep(5)
                    continue
                
                # Check for alert conditions
                await self._check_trading_alerts()
                await self._check_risk_alerts()
                await self._check_performance_alerts()
                await self._check_system_alerts()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Alert monitoring error: {str(e)}")
                await asyncio.sleep(10)
    
    async def _websocket_broadcast_loop(self):
        """Broadcast real-time updates to connected clients"""
        while self.is_running:
            try:
                if not self.websocket_connections:
                    await asyncio.sleep(1)
                    continue
                
                # Prepare update data
                update_data = {
                    "timestamp": datetime.now().isoformat(),
                    "trading_metrics": list(self.real_time_data["trading_metrics"])[-1] if self.real_time_data["trading_metrics"] else {},
                    "performance_data": list(self.real_time_data["performance_data"])[-1] if self.real_time_data["performance_data"] else {},
                    "risk_metrics": list(self.real_time_data["risk_metrics"])[-1] if self.real_time_data["risk_metrics"] else {},
                    "system_health": list(self.real_time_data["system_health"])[-1] if self.real_time_data["system_health"] else {},
                    "alerts": [asdict(alert) for alert in list(self.alerts)[-10:]],  # Last 10 alerts
                    "opportunities": len(self.trading_engine.active_opportunities) if self.trading_engine else 0,
                    "decisions": len(self.trading_engine.active_decisions) if self.trading_engine else 0
                }
                
                # Broadcast to all connected clients
                disconnected = []
                for websocket in self.websocket_connections:
                    try:
                        await websocket.send_json(update_data)
                    except Exception as e:
                        logger.error(f"Error in enterprise_trading_dashboard.py: {e}", exc_info=True)
                        raise  # Re-raise for proper error handling
                        disconnected.append(websocket)
                
                # Remove disconnected clients
                for ws in disconnected:
                    self.websocket_connections.remove(ws)
                
                await asyncio.sleep(self.config.get('update_intervals', {}).get('websocket_broadcast', 1))
            except asyncio.CancelledError:
                logger.info("Websocket broadcast loop cancelled.")
                break
            except Exception as e:
                logger.error("Error in websocket broadcast loop", error=e, exc_info=True)
                await asyncio.sleep(5)
    
    async def _performance_analysis_loop(self):
        """Analyze performance and generate insights"""
        while self.is_running:
            try:
                # Perform periodic performance analysis
                await self._analyze_trading_performance()
                await self._analyze_risk_metrics()
                await self._analyze_cognitive_performance()
                
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                logger.error(f"❌ Performance analysis error: {str(e)}")
                await asyncio.sleep(120)
    
    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Kimera Enterprise Trading Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                .status-online { color: #10B981; }
                .status-warning { color: #F59E0B; }
                .status-critical { color: #EF4444; }
                .metric-card { transition: all 0.3s ease; }
                .metric-card:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body class="bg-gray-100">
            <div class="min-h-screen">
                <!-- Header -->
                <header class="bg-white shadow-sm border-b">
                    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <div class="flex justify-between h-16">
                            <div class="flex items-center">
                                <h1 class="text-2xl font-bold text-gray-900">🧠 Kimera Enterprise Trading Dashboard</h1>
                            </div>
                            <div class="flex items-center space-x-4">
                                <div id="system-status" class="flex items-center">
                                    <div class="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                                    <span class="text-sm font-medium">System Online</span>
                                </div>
                                <button id="emergency-stop" class="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md text-sm font-medium">
                                    🚨 Emergency Stop
                                </button>
                            </div>
                        </div>
                    </div>
                </header>

                <!-- Main Content -->
                <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                    <!-- Status Cards -->
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                        <div class="metric-card bg-white overflow-hidden shadow rounded-lg">
                            <div class="p-5">
                                <div class="flex items-center">
                                    <div class="flex-shrink-0">
                                        <div class="text-2xl">⚡</div>
                                    </div>
                                    <div class="ml-5 w-0 flex-1">
                                        <dl>
                                            <dt class="text-sm font-medium text-gray-500 truncate">Active Decisions</dt>
                                            <dd id="active-decisions" class="text-lg font-medium text-gray-900">0</dd>
                                        </dl>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="metric-card bg-white overflow-hidden shadow rounded-lg">
                            <div class="p-5">
                                <div class="flex items-center">
                                    <div class="flex-shrink-0">
                                        <div class="text-2xl">🎯</div>
                                    </div>
                                    <div class="ml-5 w-0 flex-1">
                                        <dl>
                                            <dt class="text-sm font-medium text-gray-500 truncate">Opportunities</dt>
                                            <dd id="opportunities" class="text-lg font-medium text-gray-900">0</dd>
                                        </dl>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="metric-card bg-white overflow-hidden shadow rounded-lg">
                            <div class="p-5">
                                <div class="flex items-center">
                                    <div class="flex-shrink-0">
                                        <div class="text-2xl">💰</div>
                                    </div>
                                    <div class="ml-5 w-0 flex-1">
                                        <dl>
                                            <dt class="text-sm font-medium text-gray-500 truncate">Daily P&L</dt>
                                            <dd id="daily-pnl" class="text-lg font-medium text-gray-900">$0.00</dd>
                                        </dl>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="metric-card bg-white overflow-hidden shadow rounded-lg">
                            <div class="p-5">
                                <div class="flex items-center">
                                    <div class="flex-shrink-0">
                                        <div class="text-2xl">🛡️</div>
                                    </div>
                                    <div class="ml-5 w-0 flex-1">
                                        <dl>
                                            <dt class="text-sm font-medium text-gray-500 truncate">Risk Score</dt>
                                            <dd id="risk-score" class="text-lg font-medium text-gray-900">0.00</dd>
                                        </dl>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Charts -->
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                        <div class="bg-white shadow rounded-lg p-6">
                            <h3 class="text-lg font-medium text-gray-900 mb-4">Performance Chart</h3>
                            <div id="performance-chart" style="height: 300px;"></div>
                        </div>
                        
                        <div class="bg-white shadow rounded-lg p-6">
                            <h3 class="text-lg font-medium text-gray-900 mb-4">Risk Metrics</h3>
                            <div id="risk-chart" style="height: 300px;"></div>
                        </div>
                    </div>

                    <!-- Alerts and Controls -->
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <div class="bg-white shadow rounded-lg">
                            <div class="px-6 py-4 border-b border-gray-200">
                                <h3 class="text-lg font-medium text-gray-900">Recent Alerts</h3>
                            </div>
                            <div class="p-6">
                                <div id="alerts-list" class="space-y-3">
                                    <!-- Alerts will be populated here -->
                                </div>
                            </div>
                        </div>

                        <div class="bg-white shadow rounded-lg">
                            <div class="px-6 py-4 border-b border-gray-200">
                                <h3 class="text-lg font-medium text-gray-900">Manual Controls</h3>
                            </div>
                            <div class="p-6">
                                <div class="space-y-4">
                                    <button id="pause-trading" class="w-full bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-md">
                                        ⏸️ Pause Trading
                                    </button>
                                    <button id="resume-trading" class="w-full bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md">
                                        ▶️ Resume Trading
                                    </button>
                                    <button id="close-positions" class="w-full bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded-md">
                                        📤 Close All Positions
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </main>
            </div>

            <script>
                // WebSocket connection for real-time updates
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };

                function updateDashboard(data) {
                    // Update metrics
                    document.getElementById('active-decisions').textContent = data.decisions || 0;
                    document.getElementById('opportunities').textContent = data.opportunities || 0;
                    
                    if (data.performance_data && data.performance_data.daily_return) {
                        const dailyPnl = (data.performance_data.daily_return * 10000).toFixed(2);
                        document.getElementById('daily-pnl').textContent = `$${dailyPnl}`;
                    }
                    
                    if (data.risk_metrics && data.risk_metrics.portfolio_risk) {
                        document.getElementById('risk-score').textContent = data.risk_metrics.portfolio_risk.toFixed(3);
                    }
                    
                    // Update alerts
                    const alertsList = document.getElementById('alerts-list');
                    alertsList.innerHTML = '';
                    
                    if (data.alerts && data.alerts.length > 0) {
                        data.alerts.forEach(alert => {
                            const alertDiv = document.createElement('div');
                            alertDiv.className = `p-3 rounded-md border-l-4 ${getAlertColor(alert.level)}`;
                            alertDiv.innerHTML = `
                                <div class="flex">
                                    <div class="flex-shrink-0">
                                        <div class="text-sm font-medium">${alert.title}</div>
                                        <div class="text-xs text-gray-500">${new Date(alert.timestamp).toLocaleTimeString()}</div>
                                    </div>
                                </div>
                            `;
                            alertsList.appendChild(alertDiv);
                        });
                    } else {
                        alertsList.innerHTML = '<div class="text-gray-500 text-sm">No recent alerts</div>';
                    }
                }

                function getAlertColor(level) {
                    switch(level) {
                        case 'critical': return 'border-red-400 bg-red-50';
                        case 'warning': return 'border-yellow-400 bg-yellow-50';
                        case 'emergency': return 'border-red-600 bg-red-100';
                        default: return 'border-blue-400 bg-blue-50';
                    }
                }

                // Control button handlers
                document.getElementById('emergency-stop').addEventListener('click', function() {
                    if (confirm('Are you sure you want to trigger an emergency stop?')) {
                        fetch('/api/emergency-stop', { method: 'POST' })
                            .then(response => response.json())
                            .then(data => alert('Emergency stop initiated'));
                    }
                });

                document.getElementById('pause-trading').addEventListener('click', function() {
                    fetch('/api/intervention/pause_trading', { 
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    })
                    .then(response => response.json())
                    .then(data => alert('Trading paused'));
                });

                document.getElementById('resume-trading').addEventListener('click', function() {
                    fetch('/api/intervention/resume_trading', { 
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    })
                    .then(response => response.json())
                    .then(data => alert('Trading resumed'));
                });

                document.getElementById('close-positions').addEventListener('click', function() {
                    if (confirm('Are you sure you want to close all positions?')) {
                        fetch('/api/intervention/close_position', { 
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ position: 'all' })
                        })
                        .then(response => response.json())
                        .then(data => alert('Closing all positions'));
                    }
                });
            </script>
        </body>
        </html>
        """
    
    # API endpoint implementations
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.trading_engine:
            return {"status": "disconnected", "message": "Trading engine not connected"}
        
        return self.trading_engine.get_status_summary()
    
    async def _get_trading_metrics(self) -> Dict[str, Any]:
        """Get current trading metrics"""
        return {
            "data": list(self.real_time_data["trading_metrics"]),
            "latest": list(self.real_time_data["trading_metrics"])[-1] if self.real_time_data["trading_metrics"] else {}
        }
    
    async def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data"""
        return {
            "data": list(self.real_time_data["performance_data"]),
            "latest": list(self.real_time_data["performance_data"])[-1] if self.real_time_data["performance_data"] else {}
        }
    
    async def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics"""
        return {
            "data": list(self.real_time_data["risk_metrics"]),
            "latest": list(self.real_time_data["risk_metrics"])[-1] if self.real_time_data["risk_metrics"] else {}
        }
    
    async def _get_alerts(self) -> Dict[str, Any]:
        """Get current alerts"""
        return {
            "alerts": [asdict(alert) for alert in self.alerts],
            "count": len(self.alerts)
        }
    
    async def _get_opportunities(self) -> Dict[str, Any]:
        """Get current trading opportunities"""
        if not self.trading_engine:
            return {"opportunities": [], "count": 0}
        
        opportunities = []
        for opp in self.trading_engine.active_opportunities.values():
            opportunities.append({
                "id": opp.opportunity_id,
                "symbol": opp.symbol,
                "type": opp.opportunity_type,
                "confidence": opp.confidence,
                "expected_return": opp.expected_return,
                "risk_score": opp.risk_score,
                "urgency": opp.urgency,
                "timestamp": opp.timestamp.isoformat()
            })
        
        return {"opportunities": opportunities, "count": len(opportunities)}
    
    async def _get_decisions(self) -> Dict[str, Any]:
        """Get current decisions"""
        if not self.trading_engine:
            return {"decisions": [], "count": 0}
        
        decisions = []
        for decision in self.trading_engine.active_decisions.values():
            decisions.append({
                "id": decision.decision_id,
                "autonomy_level": decision.autonomy_level.value,
                "requires_approval": decision.requires_approval,
                "opportunities_count": len(decision.opportunities),
                "risk_assessment": decision.risk_assessment
            })
        
        return {"decisions": decisions, "count": len(decisions)}
    
    async def _handle_manual_intervention(self, intervention_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle manual intervention"""
        intervention = ManualIntervention(
            intervention_id=f"intervention_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            intervention_type=InterventionType(intervention_type),
            user_id="dashboard_user",  # Would be actual user ID in production
            timestamp=datetime.now(),
            parameters=parameters,
            reason="Manual intervention from dashboard",
            status="initiated"
        )
        
        self.interventions.append(intervention)
        
        # Execute intervention
        result = await self._execute_intervention(intervention)
        intervention.result = result
        intervention.status = "completed" if result.get("success") else "failed"
        
        return {"success": True, "intervention_id": intervention.intervention_id, "result": result}
    
    async def _execute_intervention(self, intervention: ManualIntervention) -> Dict[str, Any]:
        """Execute manual intervention"""
        if not self.trading_engine:
            return {"success": False, "error": "Trading engine not available"}
        
        try:
            if intervention.intervention_type == InterventionType.PAUSE_TRADING:
                self.trading_engine.pause_trading = True
                return {"success": True, "message": "Trading paused"}
            
            elif intervention.intervention_type == InterventionType.RESUME_TRADING:
                self.trading_engine.pause_trading = False
                return {"success": True, "message": "Trading resumed"}
            
            elif intervention.intervention_type == InterventionType.EMERGENCY_STOP:
                await self.trading_engine.emergency_shutdown()
                return {"success": True, "message": "Emergency stop executed"}
            
            elif intervention.intervention_type == InterventionType.CLOSE_POSITION:
                await self.trading_engine._close_all_positions()
                return {"success": True, "message": "Positions closed"}
            
            else:
                return {"success": False, "error": "Unsupported intervention type"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _approve_decision(self, decision_id: str) -> Dict[str, Any]:
        """Approve a pending decision"""
        # Implementation would approve the decision
        return {"success": True, "message": f"Decision {decision_id} approved"}
    
    async def _reject_decision(self, decision_id: str) -> Dict[str, Any]:
        """Reject a pending decision"""
        # Implementation would reject the decision
        return {"success": True, "message": f"Decision {decision_id} rejected"}
    
    async def _emergency_stop(self) -> Dict[str, Any]:
        """Execute emergency stop"""
        if self.trading_engine:
            await self.trading_engine.emergency_shutdown()
            return {"success": True, "message": "Emergency stop executed"}
        return {"success": False, "error": "Trading engine not available"}
    
    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        self.websocket_connections.append(websocket)
        
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            self.websocket_connections.remove(websocket)
    
    # Alert checking methods
    async def _check_trading_alerts(self):
        """Check for trading-related alerts"""
        if not self.trading_engine:
            return
        
        # Check decision queue backlog
        if len(self.trading_engine.decision_queue) > 100:
            await self._create_alert(
                AlertLevel.WARNING,
                "Decision Queue Backlog",
                f"Decision queue has {len(self.trading_engine.decision_queue)} pending decisions",
                "trading",
                {"queue_size": len(self.trading_engine.decision_queue)}
            )
    
    async def _check_risk_alerts(self):
        """Check for risk-related alerts"""
        if not self.trading_engine or not self.trading_engine.risk_state:
            return
        
        risk_metrics = self.trading_engine.risk_state.get("metrics", {})
        
        # Check portfolio risk
        portfolio_risk = risk_metrics.get("portfolio_risk", 0)
        if portfolio_risk > 0.15:  # 15% risk threshold
            await self._create_alert(
                AlertLevel.CRITICAL,
                "High Portfolio Risk",
                f"Portfolio risk at {portfolio_risk:.1%}, exceeding 15% threshold",
                "risk",
                {"portfolio_risk": portfolio_risk}
            )
            
        # Check position size limits
        position_size = risk_metrics.get("position_size", 0)
        if position_size > 0.2:  # 20% of portfolio
            await self._create_alert(
                AlertLevel.CRITICAL,
                "Large Position Size",
                f"Position size at {position_size:.1%} of portfolio, exceeding 20% limit",
                "risk",
                {"position_size": position_size}
            )
            
        # Check leverage limits
        leverage = risk_metrics.get("leverage", 0)
        if leverage > 3.0:  # 3x leverage
            await self._create_alert(
                AlertLevel.EMERGENCY,
                "High Leverage",
                f"Leverage at {leverage:.1f}x, exceeding 3x limit",
                "risk",
                {"leverage": leverage}
            )
            
        # Check daily loss limits
        daily_loss = risk_metrics.get("daily_loss", 0)
        if daily_loss > 0.05:  # 5% daily loss
            await self._create_alert(
                AlertLevel.EMERGENCY,
                "Daily Loss Limit",
                f"Daily loss at {daily_loss:.1%}, exceeding 5% limit",
                "risk",
                {"daily_loss": daily_loss}
            )
    
    async def _check_performance_alerts(self):
        """Check for performance-related alerts"""
        if not self.trading_engine or not self.trading_engine.performance_state:
            return
        
        performance_metrics = self.trading_engine.performance_state.get("metrics", {})
        
        # Check drawdown
        max_drawdown = performance_metrics.get("max_drawdown", 0)
        if max_drawdown > 0.10:  # 10% drawdown threshold
            await self._create_alert(
                AlertLevel.WARNING,
                "High Drawdown",
                f"Maximum drawdown at {max_drawdown:.1%}",
                "performance",
                {"max_drawdown": max_drawdown}
            )
    
    async def _check_system_alerts(self):
        """Check for system health alerts"""
        if not self.trading_engine:
            return
        
        system_health = self.trading_engine.system_health
        
        # Check decision latency
        decision_latency = system_health.get("decision_latency", 0)
        if decision_latency > 1000:  # 1 second threshold
            await self._create_alert(
                AlertLevel.WARNING,
                "High Decision Latency",
                f"Decision latency at {decision_latency:.0f}ms",
                "system",
                {"decision_latency": decision_latency}
            )
    
    async def _create_alert(self, level: AlertLevel, title: str, message: str, component: str, data: Dict[str, Any]):
        """Create a new alert"""
        alert = TradingAlert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            component=component,
            data=data
        )
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.metrics["alerts_total"].labels(level=level.value, component=component).inc()
        
        logger.warning(f"🚨 {level.value.upper()} Alert: {title} - {message}")
    
    def _update_prometheus_metrics(self, trading_metrics: Dict, performance_data: Dict, risk_data: Dict):
        """Update Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # Update trading metrics
            self.metrics["active_positions"].set(trading_metrics.get("active_decisions", 0))
            
            # Update performance metrics
            if "daily_return" in performance_data:
                self.metrics["pnl_total"].set(performance_data["daily_return"] * 10000)  # Convert to dollar amount
            
            # Update risk metrics
            if "portfolio_risk" in risk_data:
                self.metrics["risk_score"].set(risk_data["portfolio_risk"])
            
            # Update cognitive metrics
            if "coherence" in self.trading_engine.cognitive_state:
                self.metrics["cognitive_coherence"].set(self.trading_engine.cognitive_state["coherence"])
                
        except Exception as e:
            logger.error(f"❌ Failed to update Prometheus metrics: {str(e)}")
    
    # Analysis methods
    async def _analyze_trading_performance(self):
        """Analyze trading performance"""
        # Implementation for performance analysis
        pass
    
    async def _analyze_risk_metrics(self):
        """Analyze risk metrics"""
        # Implementation for risk analysis
        pass
    
    async def _analyze_cognitive_performance(self):
        """Analyze cognitive performance"""
        # Implementation for cognitive analysis
        pass


def create_enterprise_dashboard(config: Dict[str, Any]) -> EnterpriseTradingDashboard:
    """Factory function to create Enterprise Trading Dashboard"""
    return EnterpriseTradingDashboard(config)


# Example usage
async def main():
    """Example of running the Enterprise Trading Dashboard"""
    config = {
        "host": "0.0.0.0",
        "port": 8080,
        "auth_enabled": False,
        "alert_thresholds": {
            "high_risk": 0.15,
            "high_drawdown": 0.10,
            "high_latency": 1000
        }
    }
    
    dashboard = create_enterprise_dashboard(config)
    
    try:
        await dashboard.start_dashboard()
    except KeyboardInterrupt:
        await dashboard.stop_dashboard()


if __name__ == "__main__":
    asyncio.run(main()) 