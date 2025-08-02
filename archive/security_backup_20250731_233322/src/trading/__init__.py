"""
Kimera Semantic Trading Module

An advanced autonomous trading system that leverages Kimera's semantic thermodynamic
reactor to detect market contradictions and execute intelligent trading strategies.

This plug-and-play module interfaces directly with Kimera's core reactor to:
- Detect semantic contradictions across multiple data streams
- Execute trades with ultra-low latency
- Provide enterprise-grade monitoring and compliance
- Adapt strategies in real-time based on thermodynamic principles
"""

__version__ = "2.0.0"

# Main integration interface
from .kimera_trading_integration import (
    create_kimera_trading_system,
    process_trading_opportunity,
    KimeraTradingIntegration,
    KimeraTradingConfig
)

# Core components
from .core.semantic_trading_reactor import (
    SemanticTradingReactor,
    TradingRequest,
    TradingResult,
    create_semantic_trading_reactor
)

# Execution layer
from .execution.semantic_execution_bridge import (
    SemanticExecutionBridge,
    ExecutionRequest,
    ExecutionResult,
    OrderType,
    OrderStatus,
    create_semantic_execution_bridge
)

# Monitoring
from .monitoring.semantic_trading_dashboard import (
    SemanticTradingDashboard,
    TradingMetrics,
    SystemHealth,
    create_semantic_trading_dashboard
)

# API Connectors
from .connectors.cryptopanic_connector import (
    CryptoPanicConnector,
    create_cryptopanic_connector,
    CryptoPanicNews
)
from .connectors.taapi_connector import (
    TAAPIConnector,
    create_taapi_connector,
    Indicator,
    Timeframe
)

__all__ = [
    # Main interface
    "create_kimera_trading_system",
    "process_trading_opportunity",
    "KimeraTradingIntegration",
    "KimeraTradingConfig",
    
    # Core
    "SemanticTradingReactor",
    "TradingRequest",
    "TradingResult",
    "create_semantic_trading_reactor",
    
    # Execution
    "SemanticExecutionBridge",
    "ExecutionRequest",
    "ExecutionResult",
    "OrderType",
    "OrderStatus",
    "create_semantic_execution_bridge",
    
    # Monitoring
    "SemanticTradingDashboard",
    "TradingMetrics",
    "SystemHealth",
    "create_semantic_trading_dashboard",
    
    # API Connectors
    "CryptoPanicConnector",
    "create_cryptopanic_connector",
    "CryptoPanicNews",
    "TAAPIConnector",
    "create_taapi_connector",
    "Indicator",
    "Timeframe"
] 