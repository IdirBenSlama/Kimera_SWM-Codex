"""
KIMERA Semantic Trading Dashboard
=================================

Enterprise-grade monitoring and visualization dashboard for the semantic trading system.
Provides real-time insights into trading performance, contradiction detection, and system health.

Features:
- Real-time performance metrics
- Contradiction heatmaps
- P&L tracking
- Risk exposure monitoring
- System health indicators
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import dcc, html, Input, Output, State
    import dash_bootstrap_components as dbc
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logging.warning("Visualization libraries not available")
    VISUALIZATION_AVAILABLE = False

# Monitoring libraries
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logging.warning("Prometheus client not available")
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TradingMetrics:
    """Real-time trading metrics"""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    active_positions: int
    total_volume: float
    contradiction_count: int
    semantic_pressure: float


@dataclass
class SystemHealth:
    """System health indicators"""
    cpu_usage: float
    memory_usage: float
    latency_ms: float
    error_rate: float
    uptime_hours: float
    connected_exchanges: int
    data_feed_status: Dict[str, bool]


class SemanticTradingDashboard:
    """
    Comprehensive monitoring dashboard for Kimera's semantic trading system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dashboard
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        self.metrics_history = []
        self.contradiction_history = []
        self.trade_history = []
        self.system_health_history = []
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        
        # Initialize Dash app if visualization available
        if VISUALIZATION_AVAILABLE:
            self.app = self._create_dash_app()
        else:
            self.app = None
        
        logger.info("📊 Semantic Trading Dashboard initialized")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors"""
        # Trading metrics
        self.pnl_gauge = Gauge('kimera_trading_pnl', 'Current P&L')
        self.positions_gauge = Gauge('kimera_active_positions', 'Number of active positions')
        self.contradiction_counter = Counter('kimera_contradictions_detected', 'Total contradictions detected')
        self.trade_counter = Counter('kimera_trades_executed', 'Total trades executed')
        
        # Performance metrics
        self.win_rate_gauge = Gauge('kimera_win_rate', 'Trading win rate')
        self.sharpe_gauge = Gauge('kimera_sharpe_ratio', 'Sharpe ratio')
        self.execution_latency = Histogram('kimera_execution_latency_ms', 'Trade execution latency')
        
        # System metrics
        self.cpu_gauge = Gauge('kimera_cpu_usage', 'CPU usage percentage')
        self.memory_gauge = Gauge('kimera_memory_usage', 'Memory usage percentage')
        self.error_rate_gauge = Gauge('kimera_error_rate', 'System error rate')
    
    def _create_dash_app(self) -> 'dash.Dash':
        """Create the Dash application for web-based monitoring"""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Define the layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("KIMERA Semantic Trading Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Key Metrics Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Total P&L", className="card-title"),
                            html.H2(id="total-pnl", children="$0.00", className="text-success")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Win Rate", className="card-title"),
                            html.H2(id="win-rate", children="0.0%", className="text-info")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Active Positions", className="card-title"),
                            html.H2(id="active-positions", children="0", className="text-warning")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Contradictions/Hour", className="card-title"),
                            html.H2(id="contradiction-rate", children="0", className="text-danger")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Charts Row
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="pnl-chart"),
                    dcc.Interval(id="pnl-interval", interval=5000)  # Update every 5 seconds
                ], width=6),
                
                dbc.Col([
                    dcc.Graph(id="contradiction-heatmap"),
                    dcc.Interval(id="contradiction-interval", interval=10000)  # Update every 10 seconds
                ], width=6)
            ], className="mb-4"),
            
            # Performance Metrics Row
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="sharpe-chart")
                ], width=4),
                
                dbc.Col([
                    dcc.Graph(id="volume-chart")
                ], width=4),
                
                dbc.Col([
                    dcc.Graph(id="execution-latency-chart")
                ], width=4)
            ], className="mb-4"),
            
            # System Health Row
            dbc.Row([
                dbc.Col([
                    html.H3("System Health", className="mb-3"),
                    dbc.Progress(id="cpu-progress", value=0, max=100, striped=True, animated=True, className="mb-2"),
                    html.P("CPU Usage", className="text-muted"),
                    dbc.Progress(id="memory-progress", value=0, max=100, striped=True, animated=True, className="mb-2"),
                    html.P("Memory Usage", className="text-muted"),
                    dcc.Interval(id="health-interval", interval=2000)  # Update every 2 seconds
                ], width=12)
            ])
        ], fluid=True)
        
        # Define callbacks
        self._setup_callbacks(app)
        
        return app
    
    def _setup_callbacks(self, app: 'dash.Dash'):
        """Setup Dash callbacks for real-time updates"""
        
        @app.callback(
            [Output("total-pnl", "children"),
             Output("win-rate", "children"),
             Output("active-positions", "children"),
             Output("contradiction-rate", "children")],
            [Input("pnl-interval", "n_intervals")]
        )
        def update_key_metrics(n):
            """Update key metrics cards"""
            if self.metrics_history:
                latest = self.metrics_history[-1]
                pnl_text = f"${latest.total_pnl:,.2f}"
                pnl_class = "text-success" if latest.total_pnl >= 0 else "text-danger"
                
                return (
                    html.Span(pnl_text, className=pnl_class),
                    f"{latest.win_rate:.1%}",
                    str(latest.active_positions),
                    str(latest.contradiction_count)
                )
            return "$0.00", "0.0%", "0", "0"
        
        @app.callback(
            Output("pnl-chart", "figure"),
            [Input("pnl-interval", "n_intervals")]
        )
        def update_pnl_chart(n):
            """Update P&L chart"""
            if not self.metrics_history:
                return go.Figure()
            
            df = pd.DataFrame([
                {
                    'timestamp': m.timestamp,
                    'total_pnl': m.total_pnl,
                    'daily_pnl': m.daily_pnl
                } for m in self.metrics_history[-100:]  # Last 100 data points
            ])
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Cumulative P&L", "Daily P&L")
            )
            
            # Cumulative P&L
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['total_pnl'],
                    mode='lines',
                    name='Total P&L',
                    line=dict(color='green' if df['total_pnl'].iloc[-1] >= 0 else 'red')
                ),
                row=1, col=1
            )
            
            # Daily P&L
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['daily_pnl'],
                    name='Daily P&L',
                    marker_color=['green' if x >= 0 else 'red' for x in df['daily_pnl']]
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            return fig
        
        @app.callback(
            Output("contradiction-heatmap", "figure"),
            [Input("contradiction-interval", "n_intervals")]
        )
        def update_contradiction_heatmap(n):
            """Update contradiction heatmap"""
            if not self.contradiction_history:
                return go.Figure()
            
            # Create a matrix of contradictions by source
            sources = list(set([c['source_a'] for c in self.contradiction_history[-50:]] + 
                             [c['source_b'] for c in self.contradiction_history[-50:]]))
            
            matrix = np.zeros((len(sources), len(sources)))
            
            for contradiction in self.contradiction_history[-50:]:
                if contradiction['source_a'] in sources and contradiction['source_b'] in sources:
                    i = sources.index(contradiction['source_a'])
                    j = sources.index(contradiction['source_b'])
                    matrix[i][j] += contradiction['tension_score']
                    matrix[j][i] += contradiction['tension_score']
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=sources,
                y=sources,
                colorscale='RdYlBu_r',
                text=np.round(matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Contradiction Heatmap",
                height=500,
                xaxis_title="Data Source",
                yaxis_title="Data Source"
            )
            
            return fig
        
        @app.callback(
            [Output("cpu-progress", "value"),
             Output("memory-progress", "value")],
            [Input("health-interval", "n_intervals")]
        )
        def update_system_health(n):
            """Update system health indicators"""
            if self.system_health_history:
                latest = self.system_health_history[-1]
                return latest.cpu_usage, latest.memory_usage
            return 0, 0
    
    def update_metrics(self, metrics: TradingMetrics):
        """
        Update dashboard with new metrics
        
        Args:
            metrics: Latest trading metrics
        """
        self.metrics_history.append(metrics)
        
        # Keep only last 24 hours of data
        cutoff = datetime.now() - timedelta(hours=24)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff]
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.pnl_gauge.set(metrics.total_pnl)
            self.positions_gauge.set(metrics.active_positions)
            self.win_rate_gauge.set(metrics.win_rate)
            self.sharpe_gauge.set(metrics.sharpe_ratio)
    
    def log_contradiction(self, contradiction: Dict[str, Any]):
        """
        Log a detected contradiction
        
        Args:
            contradiction: Contradiction details
        """
        self.contradiction_history.append(contradiction)
        
        # Keep only last 1000 contradictions
        if len(self.contradiction_history) > 1000:
            self.contradiction_history = self.contradiction_history[-1000:]
        
        if PROMETHEUS_AVAILABLE:
            self.contradiction_counter.inc()
    
    def log_trade(self, trade: Dict[str, Any]):
        """
        Log an executed trade
        
        Args:
            trade: Trade details
        """
        self.trade_history.append(trade)
        
        # Keep only last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
        
        if PROMETHEUS_AVAILABLE:
            self.trade_counter.inc()
            if 'execution_time' in trade:
                self.execution_latency.observe(trade['execution_time'] * 1000)  # Convert to ms
    
    def update_system_health(self, health: SystemHealth):
        """
        Update system health metrics
        
        Args:
            health: System health indicators
        """
        self.system_health_history.append(health)
        
        # Keep only last hour of data
        cutoff = datetime.now() - timedelta(hours=1)
        self.system_health_history = [h for h in self.system_health_history 
                                    if hasattr(h, 'timestamp') and h.timestamp > cutoff]
        
        if PROMETHEUS_AVAILABLE:
            self.cpu_gauge.set(health.cpu_usage)
            self.memory_gauge.set(health.memory_usage)
            self.error_rate_gauge.set(health.error_rate)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        trades_df = pd.DataFrame(self.trade_history)
        
        summary = {
            'current_metrics': {
                'total_pnl': latest.total_pnl,
                'daily_pnl': latest.daily_pnl,
                'win_rate': latest.win_rate,
                'sharpe_ratio': latest.sharpe_ratio,
                'max_drawdown': latest.max_drawdown,
                'active_positions': latest.active_positions
            },
            'trade_statistics': {
                'total_trades': len(self.trade_history),
                'avg_trade_size': trades_df['size'].mean() if not trades_df.empty else 0,
                'avg_execution_time': trades_df['execution_time'].mean() if 'execution_time' in trades_df else 0
            },
            'contradiction_analysis': {
                'total_detected': len(self.contradiction_history),
                'avg_tension_score': np.mean([c['tension_score'] for c in self.contradiction_history]) if self.contradiction_history else 0,
                'top_contradiction_pairs': self._get_top_contradiction_pairs()
            },
            'system_health': {
                'avg_cpu_usage': np.mean([h.cpu_usage for h in self.system_health_history]) if self.system_health_history else 0,
                'avg_memory_usage': np.mean([h.memory_usage for h in self.system_health_history]) if self.system_health_history else 0,
                'uptime': self.system_health_history[-1].uptime_hours if self.system_health_history else 0
            }
        }
        
        return summary
    
    def _get_top_contradiction_pairs(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get the top contradiction pairs by frequency"""
        if not self.contradiction_history:
            return []
        
        pair_counts = {}
        for c in self.contradiction_history:
            pair = tuple(sorted([c['source_a'], c['source_b']]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'sources': list(pair),
                'count': count,
                'avg_tension': np.mean([
                    c['tension_score'] for c in self.contradiction_history
                    if set([c['source_a'], c['source_b']]) == set(pair)
                ])
            }
            for pair, count in sorted_pairs[:top_n]
        ]
    
    def run_dashboard(self, port: int = 8050):
        """
        Run the web dashboard
        
        Args:
            port: Port to run the dashboard on
        """
        if self.app:
            logger.info(f"Starting dashboard on port {port}")
            self.app.run_server(host='0.0.0.0', port=port, debug=False)
        else:
            logger.warning("Dashboard not available - visualization libraries not installed")


def create_semantic_trading_dashboard(config: Optional[Dict[str, Any]] = None) -> SemanticTradingDashboard:
    """
    Factory function to create a Semantic Trading Dashboard
    
    Args:
        config: Dashboard configuration
        
    Returns:
        SemanticTradingDashboard instance
    """
    if config is None:
        config = {}
    
    return SemanticTradingDashboard(config) 