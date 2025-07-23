from enum import Enum
from typing import Dict, Any

class CircuitBreakerType(Enum):
    """Types of circuit breakers available"""
    PORTFOLIO_RISK = "portfolio_risk"
    POSITION_SIZE = "position_size" 
    LEVERAGE = "leverage"
    DAILY_LOSS = "daily_loss"
    API_ERRORS = "api_errors"
    LATENCY = "latency"

# Default thresholds for circuit breakers
DEFAULT_CIRCUIT_BREAKERS: Dict[CircuitBreakerType, Dict[str, Any]] = {
    CircuitBreakerType.PORTFOLIO_RISK: {
        "warning_threshold": 0.15,  # 15%
        "trigger_threshold": 0.25,  # 25%
        "cooldown_period": 3600,  # 1 hour
    },
    CircuitBreakerType.POSITION_SIZE: {
        "warning_threshold": 0.20,  # 20%
        "trigger_threshold": 0.30,  # 30%
        "cooldown_period": 1800,  # 30 minutes
    },
    CircuitBreakerType.LEVERAGE: {
        "warning_threshold": 3.0,  # 3x
        "trigger_threshold": 5.0,  # 5x
        "cooldown_period": 7200,  # 2 hours
    },
    CircuitBreakerType.DAILY_LOSS: {
        "warning_threshold": 0.05,  # 5%
        "trigger_threshold": 0.10,  # 10%
        "cooldown_period": 86400,  # 24 hours
    },
    CircuitBreakerType.API_ERRORS: {
        "warning_threshold": 5,  # 5 errors
        "trigger_threshold": 10,  # 10 errors
        "time_window": 300,  # 5 minutes
        "cooldown_period": 900,  # 15 minutes
    },
    CircuitBreakerType.LATENCY: {
        "warning_threshold": 500,  # 500ms
        "trigger_threshold": 1000,  # 1000ms
        "time_window": 60,  # 1 minute
        "cooldown_period": 300,  # 5 minutes
    }
}