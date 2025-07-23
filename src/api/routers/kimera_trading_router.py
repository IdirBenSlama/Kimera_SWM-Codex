"""
Kimera Integrated Trading System API Router
===========================================

FastAPI router for the Kimera Integrated Trading System, providing
RESTful endpoints for trading operations, monitoring, and configuration.

This router integrates with Kimera's semantic engines and provides
secure, authenticated access to trading functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime
import json

# Kimera imports
from src.core.kimera_system import get_kimera_system
from src.utils.kimera_logger import get_cognitive_logger

# Trading system imports
try:
    from src.trading.kimera_integrated_trading_system import (
        create_kimera_integrated_trading_system,
        validate_kimera_integration,
        KimeraIntegratedTradingEngine
    )
    TRADING_AVAILABLE = True
except ImportError as e:
    logging.error(f"Kimera trading system not available: {e}")
    TRADING_AVAILABLE = False

# Initialize router and logger
router = APIRouter()
security = HTTPBearer()
logger = get_cognitive_logger(__name__)

# Global trading engine instance
trading_engine: Optional[KimeraIntegratedTradingEngine] = None

# ===================== PYDANTIC MODELS =====================

class TradingConfig(BaseModel):
    """Trading system configuration model"""
    starting_capital: float = Field(default=1000.0, ge=100.0, le=1000000.0)
    max_position_size: float = Field(default=0.25, ge=0.01, le=1.0)
    max_risk_per_trade: float = Field(default=0.02, ge=0.001, le=0.1)
    trading_symbols: List[str] = Field(default=["BTCUSDT", "ETHUSDT"])
    market_data_interval: int = Field(default=5, ge=1, le=60)
    semantic_analysis_interval: int = Field(default=10, ge=5, le=300)
    signal_generation_interval: int = Field(default=15, ge=5, le=300)
    enable_vault_protection: bool = Field(default=True)
    enable_thermodynamic_validation: bool = Field(default=True)
    enable_paper_trading: bool = Field(default=True)

class TradingStatus(BaseModel):
    """Trading system status response model"""
    system_status: str
    kimera_integration: Dict[str, Any]
    portfolio: Dict[str, float]
    positions: Dict[str, int]
    semantic_analysis: Dict[str, Any]
    performance: Dict[str, float]
    active_signals: int
    market_data_symbols: List[str]

class TradingSignal(BaseModel):
    """Trading signal model"""
    signal_id: str
    symbol: str
    action: str
    confidence: float
    strategy: str
    reasoning: List[str]
    entry_price: float
    stop_loss: Optional[float]
    profit_targets: List[float]
    timestamp: datetime

class ValidationResult(BaseModel):
    """Integration validation result model"""
    kimera_available: bool
    kimera_system: bool
    contradiction_engine: bool
    thermodynamics_engine: bool
    vault_manager: bool
    gpu_foundation: bool
    overall_status: str
    details: Dict[str, str]

# ===================== DEPENDENCY FUNCTIONS =====================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate authentication token (simplified for demo)"""
    # In production, implement proper JWT validation
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"user_id": "demo_user", "permissions": ["trading"]}

async def verify_trading_available():
    """Verify trading system is available"""
    if not TRADING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kimera trading system is not available"
        )

async def get_trading_engine():
    """Get or create trading engine instance"""
    global trading_engine
    
    if not trading_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trading engine not initialized. Please initialize first."
        )
    
    return trading_engine

# ===================== API ENDPOINTS =====================

@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint for the trading system"""
    return {
        "status": "healthy",
        "service": "kimera_trading",
        "timestamp": datetime.now().isoformat(),
        "trading_available": str(TRADING_AVAILABLE)
    }

@router.post("/validate", response_model=ValidationResult)
async def validate_integration(
    _: dict = Depends(verify_trading_available)
):
    """Validate Kimera integration and component availability"""
    try:
        validation = await validate_kimera_integration()
        
        # Determine overall status
        critical_components = ['kimera_system', 'contradiction_engine', 'thermodynamics_engine']
        critical_available = all(validation[comp] for comp in critical_components)
        
        overall_status = "fully_operational" if all(validation.values()) else \
                        "operational" if critical_available else \
                        "degraded"
        
        # Create detailed status
        details = {}
        for component, available in validation.items():
            details[component] = "available" if available else "unavailable"
        
        return ValidationResult(
            **validation,
            overall_status=overall_status,
            details=details
        )
        
    except Exception as e:
        logger.error(f"Error validating integration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )

@router.post("/initialize", response_model=Dict[str, str])
async def initialize_trading_system(
    config: TradingConfig,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    _: dict = Depends(verify_trading_available)
):
    """Initialize the Kimera trading system with configuration"""
    global trading_engine
    
    try:
        logger.info(f"Initializing trading system for user: {user['user_id']}")
        
        # Validate Kimera integration first
        validation = await validate_kimera_integration()
        if not validation['kimera_system']:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kimera system is not available"
            )
        
        # Create trading engine
        config_dict = config.dict()
        trading_engine = create_kimera_integrated_trading_system(config_dict)
        
        logger.info("Trading engine created successfully")
        
        return {
            "status": "initialized",
            "message": "Kimera trading system initialized successfully",
            "user_id": user['user_id'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error initializing trading system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Initialization failed: {str(e)}"
        )

@router.post("/start", response_model=Dict[str, str])
async def start_trading_system(
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    engine: KimeraIntegratedTradingEngine = Depends(get_trading_engine)
):
    """Start the trading system"""
    try:
        if engine.is_running:
            return {
                "status": "already_running",
                "message": "Trading system is already running",
                "timestamp": datetime.now().isoformat()
            }
        
        # Start trading engine in background
        background_tasks.add_task(engine.start)
        
        logger.info(f"Trading system started for user: {user['user_id']}")
        
        return {
            "status": "starting",
            "message": "Trading system is starting",
            "user_id": user['user_id'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting trading system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start trading system: {str(e)}"
        )

@router.post("/stop", response_model=Dict[str, str])
async def stop_trading_system(
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
    engine: KimeraIntegratedTradingEngine = Depends(get_trading_engine)
):
    """Stop the trading system"""
    try:
        if not engine.is_running:
            return {
                "status": "already_stopped",
                "message": "Trading system is already stopped",
                "timestamp": datetime.now().isoformat()
            }
        
        # Stop trading engine in background
        background_tasks.add_task(engine.stop)
        
        logger.info(f"Trading system stopped for user: {user['user_id']}")
        
        return {
            "status": "stopping",
            "message": "Trading system is stopping",
            "user_id": user['user_id'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping trading system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop trading system: {str(e)}"
        )

@router.get("/status", response_model=TradingStatus)
async def get_trading_status(
    user: dict = Depends(get_current_user),
    engine: KimeraIntegratedTradingEngine = Depends(get_trading_engine)
):
    """Get comprehensive trading system status"""
    try:
        status = engine.get_status()
        
        return TradingStatus(
            system_status=status['system_status'],
            kimera_integration=status['kimera_integration'],
            portfolio=status['portfolio'],
            positions=status['positions'],
            semantic_analysis=status['semantic_analysis'],
            performance=status['performance'],
            active_signals=status['active_signals'],
            market_data_symbols=status['market_data_symbols']
        )
        
    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )

@router.get("/signals", response_model=List[TradingSignal])
async def get_active_signals(
    user: dict = Depends(get_current_user),
    engine: KimeraIntegratedTradingEngine = Depends(get_trading_engine)
):
    """Get active trading signals"""
    try:
        signals = []
        
        for symbol, signal in engine.active_signals.items():
            signals.append(TradingSignal(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence,
                strategy=signal.strategy.value,
                reasoning=signal.reasoning,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                profit_targets=signal.profit_targets,
                timestamp=signal.timestamp
            ))
        
        return signals
        
    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get signals: {str(e)}"
        )

@router.get("/positions", response_model=List[Dict[str, Any]])
async def get_positions(
    user: dict = Depends(get_current_user),
    engine: KimeraIntegratedTradingEngine = Depends(get_trading_engine)
):
    """Get all trading positions"""
    try:
        positions = []
        
        for pos_id, position in engine.positions.items():
            positions.append({
                "position_id": position.position_id,
                "symbol": position.symbol,
                "side": position.side,
                "amount_base": position.amount_base,
                "amount_quote": position.amount_quote,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "strategy": position.strategy.value,
                "is_active": position.is_active,
                "entry_time": position.entry_time.isoformat(),
                "vault_protected": position.vault_protected,
                "thermodynamic_validation": position.thermodynamic_validation
            })
        
        return positions
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get positions: {str(e)}"
        )

@router.get("/contradictions", response_model=List[Dict[str, Any]])
async def get_semantic_contradictions(
    user: dict = Depends(get_current_user),
    engine: KimeraIntegratedTradingEngine = Depends(get_trading_engine)
):
    """Get detected semantic contradictions"""
    try:
        contradictions = []
        
        for contradiction in engine.semantic_contradictions:
            contradictions.append({
                "contradiction_id": contradiction.contradiction_id,
                "signal_type": contradiction.signal_type.value,
                "opportunity_type": contradiction.opportunity_type,
                "confidence": contradiction.confidence,
                "thermodynamic_pressure": contradiction.thermodynamic_pressure,
                "semantic_distance": contradiction.semantic_distance,
                "timestamp": contradiction.timestamp.isoformat(),
                "geoid_a_id": contradiction.geoid_a.geoid_id,
                "geoid_b_id": contradiction.geoid_b.geoid_id,
                "tension_score": contradiction.tension_gradient.tension_score
            })
        
        return contradictions
        
    except Exception as e:
        logger.error(f"Error getting contradictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get contradictions: {str(e)}"
        )

@router.get("/market-data/{symbol}", response_model=Dict[str, Any])
async def get_market_data(
    symbol: str,
    user: dict = Depends(get_current_user),
    engine: KimeraIntegratedTradingEngine = Depends(get_trading_engine)
):
    """Get market data for a specific symbol"""
    try:
        if symbol not in engine.market_data_cache:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Market data for {symbol} not found"
            )
        
        latest_data = engine.market_data_cache[symbol][-1]
        
        return {
            "symbol": latest_data.symbol,
            "price": latest_data.price,
            "volume": latest_data.volume,
            "change_24h": latest_data.change_24h,
            "change_pct_24h": latest_data.change_pct_24h,
            "high_24h": latest_data.high_24h,
            "low_24h": latest_data.low_24h,
            "bid": latest_data.bid,
            "ask": latest_data.ask,
            "spread": latest_data.spread,
            "timestamp": latest_data.timestamp.isoformat(),
            "semantic_temperature": latest_data.semantic_temperature,
            "thermodynamic_pressure": latest_data.thermodynamic_pressure,
            "cognitive_field_strength": latest_data.cognitive_field_strength,
            "contradiction_count": latest_data.contradiction_count,
            "volatility": latest_data.volatility,
            "momentum": latest_data.momentum,
            "rsi": latest_data.rsi,
            "macd": latest_data.macd
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market data: {str(e)}"
        )

@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    user: dict = Depends(get_current_user),
    engine: KimeraIntegratedTradingEngine = Depends(get_trading_engine)
):
    """Get detailed performance metrics"""
    try:
        status = engine.get_status()
        
        return {
            "portfolio_performance": status['portfolio'],
            "trading_performance": status['performance'],
            "semantic_performance": status['semantic_analysis'],
            "kimera_integration_stats": {
                "engine_calls": status['kimera_integration']['engine_calls'],
                "device": status['kimera_integration']['device'],
                "system_status": status['kimera_integration']['kimera_system_status']
            },
            "risk_metrics": {
                "active_positions": status['positions']['active_count'],
                "vault_protected": status['positions']['vault_protected'],
                "total_exposure": sum(p.amount_quote for p in engine.positions.values() if p.is_active)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

@router.post("/emergency-stop", response_model=Dict[str, str])
async def emergency_stop(
    user: dict = Depends(get_current_user),
    engine: KimeraIntegratedTradingEngine = Depends(get_trading_engine)
):
    """Emergency stop - immediately halt all trading operations"""
    try:
        logger.warning(f"Emergency stop triggered by user: {user['user_id']}")
        
        # Stop the engine
        await engine.stop()
        
        return {
            "status": "emergency_stopped",
            "message": "Trading system emergency stopped",
            "user_id": user['user_id'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emergency stop failed: {str(e)}"
        )

@router.get("/kimera-status", response_model=Dict[str, Any])
async def get_kimera_system_status(
    user: dict = Depends(get_current_user)
):
    """Get detailed Kimera system status"""
    try:
        kimera_system = get_kimera_system()
        
        if not kimera_system:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kimera system not available"
            )
        
        return {
            "system_state": kimera_system.state.name,
            "initialization_complete": kimera_system._initialization_complete,
            "device": kimera_system.get_device(),
            "components": {
                "contradiction_engine": kimera_system.get_contradiction_engine() is not None,
                "thermodynamics_engine": kimera_system.get_thermodynamic_engine() is not None,
                "vault_manager": kimera_system.get_vault_manager() is not None,
                "gpu_foundation": kimera_system.get_gpu_foundation() is not None
            },
            "status": kimera_system.get_status(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Kimera status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Kimera status: {str(e)}"
        )

# ===================== WEBSOCKET ENDPOINTS =====================

@router.websocket("/ws/status")
async def websocket_status_feed(websocket):
    """WebSocket endpoint for real-time status updates"""
    await websocket.accept()
    
    try:
        while True:
            if trading_engine and trading_engine.is_running:
                status = trading_engine.get_status()
                await websocket.send_json({
                    "type": "status_update",
                    "data": status,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                await websocket.send_json({
                    "type": "status_update",
                    "data": {"system_status": "stopped"},
                    "timestamp": datetime.now().isoformat()
                })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@router.websocket("/ws/signals")
async def websocket_signals_feed(websocket):
    """WebSocket endpoint for real-time trading signals"""
    await websocket.accept()
    
    try:
        last_signal_count = 0
        
        while True:
            if trading_engine:
                current_signal_count = len(trading_engine.active_signals)
                
                if current_signal_count != last_signal_count:
                    signals = []
                    for symbol, signal in trading_engine.active_signals.items():
                        signals.append({
                            "signal_id": signal.signal_id,
                            "symbol": signal.symbol,
                            "action": signal.action,
                            "confidence": signal.confidence,
                            "strategy": signal.strategy.value,
                            "timestamp": signal.timestamp.isoformat()
                        })
                    
                    await websocket.send_json({
                        "type": "signals_update",
                        "data": signals,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    last_signal_count = current_signal_count
            
            await asyncio.sleep(2)  # Check every 2 seconds
            
    except Exception as e:
        logger.error(f"WebSocket signals error: {e}")
        await websocket.close()

# ===================== UTILITY ENDPOINTS =====================

@router.get("/config/default", response_model=TradingConfig)
async def get_default_config():
    """Get default trading configuration"""
    return TradingConfig()

@router.get("/symbols", response_model=List[str])
async def get_supported_symbols():
    """Get list of supported trading symbols"""
    return [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT",
        "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "BNBUSDT"
    ]

@router.get("/strategies", response_model=List[str])
async def get_supported_strategies():
    """Get list of supported trading strategies"""
    return [
        "semantic_contradiction",
        "thermodynamic_equilibrium", 
        "cognitive_field_dynamics",
        "momentum_surfing",
        "mean_reversion",
        "breakout_hunter",
        "volatility_harvester",
        "trend_rider"
    ]

# Add router tags and metadata
router.tags = ["Kimera Trading"]
router.prefix = "/trading" 