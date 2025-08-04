"""
Kimera SWM - Real-Time Monitoring Dashboard
==========================================

State-of-the-art monitoring dashboard using Dash, Plotly, and Streamlit
for comprehensive visualization of Kimera system metrics.
"""

import asyncio
import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Initialize structured logger
try:
    from utils.kimera_logger import get_system_logger
except ImportError:
    # Create placeholders for utils.kimera_logger
    def get_system_logger(*args, **kwargs):
        return None


logger = get_system_logger(__name__)


import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Streamlit for alternative dashboard
import streamlit as st
from dash import Input, Output, callback, dcc, html

# Panel for advanced interactive dashboards
try:
    import holoviews as hv
    import panel as pn

    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False

from .kimera_monitoring_core import AlertSeverity, MonitoringLevel, get_monitoring_core


class KimeraDashboard:
    """
    Real-time monitoring dashboard for Kimera SWM

    Features:
    - Real-time system metrics visualization
    - Kimera-specific cognitive architecture monitoring
    - GPU and AI workload tracking
    - Alert management and notification center
    - Anomaly detection visualization
    - Performance profiling dashboards
    - Custom metric exploration
    """

    def __init__(self, port: int = 8050, debug: bool = False):
        self.port = port
        self.debug = debug
        self.monitoring_core = get_monitoring_core()

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                "https://codepen.io/chriddyp/pen/bWLwgP.css",
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
            ],
        )

        self.app.title = "Kimera SWM - Monitoring Dashboard"

        # Setup dashboard layout
        self._setup_layout()
        self._setup_callbacks()

        logger.info("🖥️ Kimera Real-Time Dashboard initialized")
        logger.info(f"   🌐 Port: {port}")
        logger.debug(f"   🔧 Debug: {debug}")

    def _setup_layout(self):
        """Setup the dashboard layout"""

        self.app.layout = html.Div(
            [
                # Header
                html.Div(
                    [
                        html.H1(
                            [
                                html.I(
                                    className="fas fa-brain",
                                    style={"margin-right": "10px"},
                                ),
                                "Kimera SWM - State-of-the-Art Monitoring",
                            ],
                            style={
                                "text-align": "center",
                                "color": "#2c3e50",
                                "margin-bottom": "30px",
                            },
                        ),
                        # Status indicators
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4(
                                            "🟢 System Status",
                                            style={"color": "#27ae60"},
                                        ),
                                        html.P(
                                            "All systems operational",
                                            id="system-status",
                                        ),
                                    ],
                                    className="three columns",
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "📊 Active Monitoring",
                                            style={"color": "#3498db"},
                                        ),
                                        html.P("0 alerts", id="alert-count"),
                                    ],
                                    className="three columns",
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "🔥 GPU Status", style={"color": "#e74c3c"}
                                        ),
                                        html.P("RTX 4090 Ready", id="gpu-status"),
                                    ],
                                    className="three columns",
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "🧠 Kimera AI", style={"color": "#9b59b6"}
                                        ),
                                        html.P(
                                            "Revolutionary Mode", id="kimera-status"
                                        ),
                                    ],
                                    className="three columns",
                                ),
                            ],
                            className="row",
                            style={"margin-bottom": "30px"},
                        ),
                    ]
                ),
                # Main dashboard tabs
                dcc.Tabs(
                    id="main-tabs",
                    value="system-overview",
                    children=[
                        dcc.Tab(label="🖥️ System Overview", value="system-overview"),
                        dcc.Tab(label="🧠 Kimera Cognitive", value="kimera-cognitive"),
                        dcc.Tab(label="🔥 GPU & AI Workloads", value="gpu-ai"),
                        dcc.Tab(label="🚨 Alerts & Anomalies", value="alerts"),
                        dcc.Tab(label="📈 Performance Profiling", value="performance"),
                        dcc.Tab(label="🔍 Custom Metrics", value="custom-metrics"),
                    ],
                ),
                # Tab content
                html.Div(id="tab-content"),
                # Auto-refresh component
                dcc.Interval(
                    id="interval-component",
                    interval=5 * 1000,  # Update every 5 seconds
                    n_intervals=0,
                ),
                # Store for shared data
                dcc.Store(id="monitoring-data"),
            ],
            style={"padding": "20px"},
        )

    def _setup_callbacks(self):
        """Setup dashboard callbacks for interactivity"""

        @self.app.callback(
            Output("tab-content", "children"), Input("main-tabs", "value")
        )
        def render_tab_content(active_tab):
            if active_tab == "system-overview":
                return self._create_system_overview_tab()
            elif active_tab == "kimera-cognitive":
                return self._create_kimera_cognitive_tab()
            elif active_tab == "gpu-ai":
                return self._create_gpu_ai_tab()
            elif active_tab == "alerts":
                return self._create_alerts_tab()
            elif active_tab == "performance":
                return self._create_performance_tab()
            elif active_tab == "custom-metrics":
                return self._create_custom_metrics_tab()

        @self.app.callback(
            [
                Output("monitoring-data", "data"),
                Output("system-status", "children"),
                Output("alert-count", "children"),
                Output("gpu-status", "children"),
                Output("kimera-status", "children"),
            ],
            Input("interval-component", "n_intervals"),
        )
        def update_monitoring_data(n):
            # Get monitoring data
            status = self.monitoring_core.get_monitoring_status()
            metrics = self.monitoring_core.get_metrics_summary()

            # Update status indicators
            system_status = (
                "🟢 All systems operational"
                if status["is_running"]
                else "🔴 Monitoring stopped"
            )
            alert_count = f"{status['alerts_count']} alerts"
            gpu_status = (
                "🔥 RTX 4090 Active"
                if status["capabilities"]["nvidia_monitoring"]
                else "⚪ No GPU"
            )
            kimera_status = "🧠 Revolutionary Mode Active"

            # Combine data
            monitoring_data = {
                "status": status,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }

            return (
                monitoring_data,
                system_status,
                alert_count,
                gpu_status,
                kimera_status,
            )

    def _create_system_overview_tab(self):
        """Create system overview tab content"""

        return html.Div(
            [
                html.H3(
                    "🖥️ System Resource Monitoring", style={"margin-bottom": "20px"}
                ),
                # Real-time system charts
                html.Div(
                    [
                        # CPU Usage
                        html.Div(
                            [dcc.Graph(id="cpu-usage-chart")], className="six columns"
                        ),
                        # Memory Usage
                        html.Div(
                            [dcc.Graph(id="memory-usage-chart")],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        # Network I/O
                        html.Div(
                            [dcc.Graph(id="network-chart")], className="six columns"
                        ),
                        # Disk Usage
                        html.Div(
                            [dcc.Graph(id="disk-usage-chart")], className="six columns"
                        ),
                    ],
                    className="row",
                ),
                # System information table
                html.H4("📋 System Information", style={"margin-top": "30px"}),
                html.Div(id="system-info-table"),
            ]
        )

    def _create_kimera_cognitive_tab(self):
        """Create Kimera cognitive architecture monitoring tab"""

        return html.Div(
            [
                html.H3(
                    "🧠 Kimera Cognitive Architecture Monitoring",
                    style={"margin-bottom": "20px"},
                ),
                # Cognitive metrics
                html.Div(
                    [
                        # Geoid Evolution
                        html.Div(
                            [dcc.Graph(id="geoid-evolution-chart")],
                            className="six columns",
                        ),
                        # Scar Formation
                        html.Div(
                            [dcc.Graph(id="scar-formation-chart")],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        # Contradiction Events
                        html.Div(
                            [dcc.Graph(id="contradiction-events-chart")],
                            className="six columns",
                        ),
                        # Selective Feedback Analysis
                        html.Div(
                            [dcc.Graph(id="selective-feedback-chart")],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
                # Revolutionary Insights
                html.H4(
                    "💡 Revolutionary Insights Generation", style={"margin-top": "30px"}
                ),
                html.Div(
                    [dcc.Graph(id="revolutionary-insights-chart")],
                    className="twelve columns",
                ),
                # Cognitive Coherence Heatmap
                html.H4("🌐 Cognitive Coherence Matrix", style={"margin-top": "30px"}),
                html.Div(
                    [dcc.Graph(id="cognitive-coherence-heatmap")],
                    className="twelve columns",
                ),
            ]
        )

    def _create_gpu_ai_tab(self):
        """Create GPU and AI workload monitoring tab"""

        return html.Div(
            [
                html.H3(
                    "🔥 GPU & AI Workload Monitoring", style={"margin-bottom": "20px"}
                ),
                # GPU metrics
                html.Div(
                    [
                        # GPU Utilization
                        html.Div(
                            [dcc.Graph(id="gpu-utilization-chart")],
                            className="six columns",
                        ),
                        # GPU Memory
                        html.Div(
                            [dcc.Graph(id="gpu-memory-chart")], className="six columns"
                        ),
                    ],
                    className="row",
                ),
                html.Div(
                    [
                        # GPU Temperature
                        html.Div(
                            [dcc.Graph(id="gpu-temperature-chart")],
                            className="six columns",
                        ),
                        # GPU Power
                        html.Div(
                            [dcc.Graph(id="gpu-power-chart")], className="six columns"
                        ),
                    ],
                    className="row",
                ),
                # AI Workload metrics
                html.H4("🤖 AI Workload Analysis", style={"margin-top": "30px"}),
                html.Div(
                    [
                        # Embedding Generation Latency
                        html.Div(
                            [dcc.Graph(id="embedding-latency-chart")],
                            className="six columns",
                        ),
                        # Analysis Latency by Type
                        html.Div(
                            [dcc.Graph(id="analysis-latency-chart")],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
                # Optimization Scores
                html.H4("🎯 Optimization Performance", style={"margin-top": "30px"}),
                html.Div(
                    [dcc.Graph(id="optimization-scores-chart")],
                    className="twelve columns",
                ),
            ]
        )

    def _create_alerts_tab(self):
        """Create alerts and anomalies monitoring tab"""

        return html.Div(
            [
                html.H3(
                    "🚨 Alerts & Anomaly Detection", style={"margin-bottom": "20px"}
                ),
                # Alert summary
                html.Div(
                    [
                        html.Div(
                            [html.H4("📊 Alert Summary"), html.Div(id="alert-summary")],
                            className="four columns",
                        ),
                        html.Div(
                            [
                                html.H4("⚠️ Recent Alerts"),
                                html.Div(id="recent-alerts-list"),
                            ],
                            className="eight columns",
                        ),
                    ],
                    className="row",
                ),
                # Alert timeline
                html.H4("📈 Alert Timeline", style={"margin-top": "30px"}),
                html.Div(
                    [dcc.Graph(id="alert-timeline-chart")], className="twelve columns"
                ),
                # Anomaly detection
                html.H4("🔍 Anomaly Detection", style={"margin-top": "30px"}),
                html.Div(
                    [
                        # Anomaly scores
                        html.Div(
                            [dcc.Graph(id="anomaly-scores-chart")],
                            className="six columns",
                        ),
                        # Anomaly correlation
                        html.Div(
                            [dcc.Graph(id="anomaly-correlation-chart")],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
            ]
        )

    def _create_performance_tab(self):
        """Create performance profiling tab"""

        return html.Div(
            [
                html.H3(
                    "📈 Performance Profiling & Analysis",
                    style={"margin-bottom": "20px"},
                ),
                # Memory profiling
                html.Div(
                    [
                        # Memory usage over time
                        html.Div(
                            [dcc.Graph(id="memory-profiling-chart")],
                            className="six columns",
                        ),
                        # Memory allocation patterns
                        html.Div(
                            [dcc.Graph(id="memory-allocation-chart")],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
                # Performance metrics
                html.Div(
                    [
                        # Response time distribution
                        html.Div(
                            [dcc.Graph(id="response-time-distribution")],
                            className="six columns",
                        ),
                        # Throughput metrics
                        html.Div(
                            [dcc.Graph(id="throughput-metrics")],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
                # Profiling insights
                html.H4("🔬 Profiling Insights", style={"margin-top": "30px"}),
                html.Div(
                    [html.Div(id="profiling-insights")], className="twelve columns"
                ),
            ]
        )

    def _create_custom_metrics_tab(self):
        """Create custom metrics exploration tab"""

        return html.Div(
            [
                html.H3("🔍 Custom Metrics Explorer", style={"margin-bottom": "20px"}),
                # Metric selector
                html.Div(
                    [
                        html.Label("Select Metrics to Visualize:"),
                        dcc.Dropdown(
                            id="custom-metrics-dropdown",
                            multi=True,
                            placeholder="Choose metrics...",
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                # Time range selector
                html.Div(
                    [
                        html.Label("Time Range:"),
                        dcc.DatePickerRange(
                            id="custom-time-range",
                            start_date=datetime.now() - timedelta(hours=24),
                            end_date=datetime.now(),
                        ),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                # Custom visualization
                html.Div(
                    [dcc.Graph(id="custom-metrics-chart")], className="twelve columns"
                ),
                # Correlation analysis
                html.H4("🔗 Metric Correlation Analysis", style={"margin-top": "30px"}),
                html.Div(
                    [dcc.Graph(id="metrics-correlation-heatmap")],
                    className="twelve columns",
                ),
            ]
        )

    def run_dashboard(self):
        """Run the dashboard server"""

        logger.info(f"🚀 Starting Kimera Dashboard on port {self.port}")
        logger.info(f"🌐 Dashboard URL: http://localhost:{self.port}")

        self.app.run_server(debug=self.debug, port=self.port, host="0.0.0.0")


class StreamlitDashboard:
    """Alternative Streamlit-based monitoring dashboard"""

    def __init__(self):
        self.monitoring_core = get_monitoring_core()

    def run_streamlit_dashboard(self):
        """Run Streamlit monitoring dashboard"""

        st.set_page_config(
            page_title="Kimera SWM Monitoring",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("🧠 Kimera SWM - Real-Time Monitoring")

        # Sidebar controls
        st.sidebar.header("⚙️ Dashboard Controls")

        monitoring_level = st.sidebar.selectbox(
            "Monitoring Level",
            options=[level.value for level in MonitoringLevel],
            index=1,
        )

        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, 5)

        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

        # Main dashboard
        self._render_streamlit_content()

    def _render_streamlit_content(self):
        """Render Streamlit dashboard content"""

        # Get monitoring data
        status = self.monitoring_core.get_monitoring_status()
        metrics = self.monitoring_core.get_metrics_summary()

        # Status overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🖥️ System Status",
                "Online" if status["is_running"] else "Offline",
                delta="Monitoring Active",
            )

        with col2:
            st.metric("🚨 Alerts", status["alerts_count"], delta=f"Last 24h")

        with col3:
            st.metric(
                "⏱️ Uptime", f"{status['uptime_seconds']:.0f}s", delta="Since start"
            )

        with col4:
            st.metric("🔧 Tasks", status["background_tasks"], delta="Active monitoring")

        st.divider()

        # Metrics charts
        if metrics:
            self._render_streamlit_charts(metrics)

    def _render_streamlit_charts(self, metrics: Dict[str, Any]):
        """Render Streamlit charts"""

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💻 CPU Usage")
            if "cpu_usage" in metrics:
                cpu_data = metrics["cpu_usage"]
                st.metric(
                    "Current CPU",
                    f"{cpu_data['current']:.1f}%",
                    delta=f"Avg: {cpu_data['average']:.1f}%",
                )

        with col2:
            st.subheader("🧠 Memory Usage")
            if "memory_usage" in metrics:
                memory_data = metrics["memory_usage"]
                st.metric(
                    "Current Memory",
                    f"{memory_data['current']:.1f}%",
                    delta=f"Peak: {memory_data['max']:.1f}%",
                )


def create_dashboard(dashboard_type: str = "dash", port: int = 8050) -> Any:
    """Create and return appropriate dashboard instance"""

    if dashboard_type.lower() == "dash":
        return KimeraDashboard(port=port)
    elif dashboard_type.lower() == "streamlit":
        return StreamlitDashboard()
    else:
        raise ValueError(f"Unknown dashboard type: {dashboard_type}")


def run_monitoring_dashboard(
    dashboard_type: str = "dash", port: int = 8050, debug: bool = False
):
    """Run the monitoring dashboard"""

    dashboard = create_dashboard(dashboard_type, port)

    if dashboard_type.lower() == "dash":
        dashboard.run_dashboard()
    elif dashboard_type.lower() == "streamlit":
        dashboard.run_streamlit_dashboard()


if __name__ == "__main__":
    # Run dashboard
    run_monitoring_dashboard(dashboard_type="dash", port=8050, debug=True)
