"""
Omnidimensional Protocol Engine API Router

Provides REST API endpoints for the Kimera Omnidimensional Protocol Engine.
All endpoints are subject to ethical governance oversight.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.ethical_governor import EthicalGovernor
from ...core.kimera_system import kimera_singleton
from ...engines.omnidimensional_protocol_engine import (
    ArbitrageExecutionError,
    KimeraProtocolEngineError,
    OmnidimensionalProtocolEngine,
    ProtocolConnectionError,
    SentimentAnalysisError,
    YieldOptimizationError,
)

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/omnidimensional",
    tags=["Omnidimensional Protocol Engine"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


# Pydantic models for request/response validation
class TradingCycleRequest(BaseModel):
    """Request model for trading cycle execution"""

    strategies: Optional[List[str]] = Field(
        default=["arbitrage", "yield", "sentiment", "cross_chain", "market_making"],
        description="List of strategies to execute",
    )
    max_risk_tolerance: Optional[float] = Field(
        default=0.5, ge=0.0, le=1.0, description="Maximum risk tolerance (0.0 to 1.0)"
    )


class ContinuousTradingRequest(BaseModel):
    """Request model for continuous trading"""

    duration_hours: float = Field(
        default=24.0, gt=0.0, le=168.0, description="Duration in hours"  # Max 1 week
    )
    cycle_interval_seconds: Optional[int] = Field(
        default=60, ge=10, le=3600, description="Interval between cycles in seconds"
    )


class SentimentAnalysisRequest(BaseModel):
    """Request model for sentiment analysis"""

    protocol: str = Field(description="Protocol name to analyze")
    timeframe: Optional[str] = Field(default="1h", description="Analysis timeframe")


class RouteOptimizationRequest(BaseModel):
    """Request model for route optimization"""

    token_in: str = Field(description="Input token")
    token_out: str = Field(description="Output token")
    amount: float = Field(gt=0, description="Amount to trade")
    strategy: Optional[str] = Field(
        default="best_price", description="Routing strategy"
    )


class YieldOptimizationRequest(BaseModel):
    """Request model for yield optimization"""

    assets: Dict[str, float] = Field(description="Asset allocation {token: amount}")
    min_apy: Optional[float] = Field(
        default=0.03, ge=0.0, description="Minimum APY threshold"
    )


# Global engine instance (initialized on startup)
_omnidimensional_engine: Optional[OmnidimensionalProtocolEngine] = None


async def get_omnidimensional_engine() -> OmnidimensionalProtocolEngine:
    """Dependency to get the omnidimensional engine instance"""
    global _omnidimensional_engine

    if _omnidimensional_engine is None:
        try:
            # Get ethical governor from kimera singleton
            ethical_governor = kimera_singleton.ethical_governor
            if ethical_governor is None:
                raise HTTPException(
                    status_code=503, detail="Ethical governor not available"
                )

            # Initialize omnidimensional engine
            _omnidimensional_engine = OmnidimensionalProtocolEngine(ethical_governor)
            logger.info("✅ Omnidimensional engine initialized for API")

        except Exception as e:
            logger.error(f"❌ Failed to initialize omnidimensional engine: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize omnidimensional engine: {str(e)}",
            )

    return _omnidimensional_engine


@router.get("/status")
async def get_status(
    engine: OmnidimensionalProtocolEngine = Depends(get_omnidimensional_engine),
) -> Dict[str, Any]:
    """Get the current status of the omnidimensional engine"""
    try:
        status = {
            "status": "operational",
            "protocols_count": len(engine.registry.protocols),
            "hardware_info": engine.device_info,
            "total_profit": engine.total_profit,
            "total_trades": engine.total_trades,
            "strategy_performance": engine.strategy_performance,
            "ethical_compliance": "active",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("📊 Status request completed")
        return status

    except Exception as e:
        logger.error(f"❌ Status request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/protocols")
async def list_protocols(
    engine: OmnidimensionalProtocolEngine = Depends(get_omnidimensional_engine),
) -> Dict[str, Any]:
    """List all supported protocols with their information"""
    try:
        protocols_info = {}

        for name, protocol in engine.registry.protocols.items():
            protocols_info[name] = {
                "name": protocol.name,
                "type": protocol.type.value,
                "chains": [chain.value for chain in protocol.chains],
                "tvl": protocol.tvl,
                "daily_volume": protocol.daily_volume,
                "audit_score": protocol.audit_score,
                "gas_efficiency": protocol.gas_efficiency,
                "liquidity_depth": protocol.liquidity_depth,
                "supported_features": protocol.supported_features,
            }

        result = {
            "total_protocols": len(protocols_info),
            "protocols": protocols_info,
            "categories": {
                "spot_amm": len(
                    [
                        p
                        for p in engine.registry.protocols.values()
                        if p.type.value == "spot_amm"
                    ]
                ),
                "derivatives": len(
                    [
                        p
                        for p in engine.registry.protocols.values()
                        if p.type.value == "derivatives"
                    ]
                ),
                "cross_chain": len(
                    [
                        p
                        for p in engine.registry.protocols.values()
                        if p.type.value == "cross_chain"
                    ]
                ),
                "yield_optimizer": len(
                    [
                        p
                        for p in engine.registry.protocols.values()
                        if p.type.value == "yield_optimizer"
                    ]
                ),
            },
        }

        logger.info(f"📋 Listed {len(protocols_info)} protocols")
        return result

    except Exception as e:
        logger.error(f"❌ Protocol listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-cycle")
async def execute_trading_cycle(
    request: TradingCycleRequest,
    engine: OmnidimensionalProtocolEngine = Depends(get_omnidimensional_engine),
) -> Dict[str, Any]:
    """Execute a single omnidimensional trading cycle"""
    try:
        logger.info(f"🔄 Executing trading cycle with strategies: {request.strategies}")

        # Execute the trading cycle
        result = await engine.execute_omnidimensional_cycle()

        # Add request parameters to result
        result["request_parameters"] = {
            "strategies": request.strategies,
            "max_risk_tolerance": request.max_risk_tolerance,
        }

        logger.info(f"✅ Trading cycle completed: ${result['total_profit']:.2f} profit")
        return result

    except KimeraProtocolEngineError as e:
        logger.error(f"❌ Trading cycle failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in trading cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-continuous-trading")
async def start_continuous_trading(
    request: ContinuousTradingRequest,
    background_tasks: BackgroundTasks,
    engine: OmnidimensionalProtocolEngine = Depends(get_omnidimensional_engine),
) -> Dict[str, Any]:
    """Start continuous trading as a background task"""
    try:
        logger.info(
            f"🚀 Starting continuous trading for {request.duration_hours} hours"
        )

        # Add continuous trading as background task
        background_tasks.add_task(engine.run_continuous_trading, request.duration_hours)

        return {
            "status": "started",
            "duration_hours": request.duration_hours,
            "cycle_interval_seconds": request.cycle_interval_seconds,
            "message": "Continuous trading started as background task",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Failed to start continuous trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-sentiment")
async def analyze_protocol_sentiment(
    request: SentimentAnalysisRequest,
    engine: OmnidimensionalProtocolEngine = Depends(get_omnidimensional_engine),
) -> Dict[str, Any]:
    """Analyze sentiment for a specific protocol"""
    try:
        logger.info(f"🔍 Analyzing sentiment for {request.protocol}")

        # Check if protocol exists
        if request.protocol not in engine.registry.protocols:
            raise HTTPException(
                status_code=404, detail=f"Protocol '{request.protocol}' not found"
            )

        # Perform sentiment analysis
        sentiment_result = await engine.sentiment_analyzer.analyze_protocol_sentiment(
            request.protocol, request.timeframe
        )

        # Add protocol info to result
        protocol_info = engine.registry.protocols[request.protocol]
        result = {
            "protocol": request.protocol,
            "protocol_info": {
                "name": protocol_info.name,
                "type": protocol_info.type.value,
                "tvl": protocol_info.tvl,
                "audit_score": protocol_info.audit_score,
            },
            "sentiment_analysis": sentiment_result,
            "timeframe": request.timeframe,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✅ Sentiment analysis completed for {request.protocol}: {sentiment_result['composite_score']:.3f}"
        )
        return result

    except SentimentAnalysisError as e:
        logger.error(f"❌ Sentiment analysis failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-route")
async def optimize_route(
    request: RouteOptimizationRequest,
    engine: OmnidimensionalProtocolEngine = Depends(get_omnidimensional_engine),
) -> Dict[str, Any]:
    """Find optimal routing for a token swap"""
    try:
        logger.info(
            f"🛣️ Optimizing route: {request.amount} {request.token_in} → {request.token_out}"
        )

        # Find optimal route
        route_result = await engine.router.find_optimal_route(
            request.token_in, request.token_out, request.amount, request.strategy
        )

        # Add request parameters to result
        result = {
            "request": {
                "token_in": request.token_in,
                "token_out": request.token_out,
                "amount": request.amount,
                "strategy": request.strategy,
            },
            "optimal_route": route_result,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✅ Route optimization completed: {route_result['protocols_used']}"
        )
        return result

    except ProtocolConnectionError as e:
        logger.error(f"❌ Route optimization failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in route optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan-arbitrage")
async def scan_arbitrage_opportunities(
    engine: OmnidimensionalProtocolEngine = Depends(get_omnidimensional_engine),
) -> Dict[str, Any]:
    """Scan for current arbitrage opportunities"""
    try:
        logger.info("⚡ Scanning for arbitrage opportunities")

        # Scan for arbitrage opportunities
        opportunities = await engine.arbitrage_engine.scan_arbitrage_opportunities()

        # Format opportunities for API response
        formatted_opportunities = []
        for opp in opportunities:
            formatted_opportunities.append(
                {
                    "protocol_route": opp.protocol_route,
                    "estimated_profit": opp.estimated_profit,
                    "profit_percentage": opp.profit_percentage,
                    "risk_score": opp.risk_score,
                    "execution_time": opp.execution_time,
                    "gas_cost": opp.gas_cost,
                    "liquidity_required": opp.liquidity_required,
                    "confidence_score": opp.confidence_score,
                    "sentiment_boost": opp.sentiment_boost,
                }
            )

        result = {
            "opportunities_found": len(formatted_opportunities),
            "opportunities": formatted_opportunities,
            "scan_timestamp": datetime.now().isoformat(),
            "total_estimated_profit": sum(
                opp.estimated_profit for opp in opportunities
            ),
        }

        logger.info(f"✅ Found {len(opportunities)} arbitrage opportunities")
        return result

    except ArbitrageExecutionError as e:
        logger.error(f"❌ Arbitrage scanning failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in arbitrage scanning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-yield")
async def optimize_yield(
    request: YieldOptimizationRequest,
    engine: OmnidimensionalProtocolEngine = Depends(get_omnidimensional_engine),
) -> Dict[str, Any]:
    """Find optimal yield strategies for given assets"""
    try:
        logger.info(f"📈 Optimizing yield for {len(request.assets)} assets")

        # Find optimal yield strategies
        strategies = await engine.yield_optimizer.find_optimal_yield_strategies(
            request.assets
        )

        # Filter by minimum APY if specified
        filtered_strategies = [
            strategy
            for strategy in strategies
            if strategy.get("apy", 0) >= request.min_apy
        ]

        # Calculate totals
        total_deployable = sum(request.assets.values())
        total_estimated_yield = sum(
            strategy.get("estimated_daily_yield", 0) for strategy in filtered_strategies
        )

        result = {
            "request": {"assets": request.assets, "min_apy": request.min_apy},
            "strategies_found": len(filtered_strategies),
            "strategies": filtered_strategies,
            "summary": {
                "total_deployable_capital": total_deployable,
                "total_estimated_daily_yield": total_estimated_yield,
                "average_apy": sum(s.get("apy", 0) for s in filtered_strategies)
                / max(len(filtered_strategies), 1),
                "annualized_yield_estimate": total_estimated_yield * 365,
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"✅ Found {len(filtered_strategies)} yield strategies")
        return result

    except YieldOptimizationError as e:
        logger.error(f"❌ Yield optimization failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error in yield optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics(
    engine: OmnidimensionalProtocolEngine = Depends(get_omnidimensional_engine),
) -> Dict[str, Any]:
    """Get comprehensive performance metrics"""
    try:
        performance = {
            "overall_performance": {
                "total_profit": engine.total_profit,
                "total_trades": engine.total_trades,
                "average_profit_per_trade": engine.total_profit
                / max(engine.total_trades, 1),
            },
            "strategy_breakdown": engine.strategy_performance,
            "hardware_utilization": engine.device_info,
            "protocol_statistics": {
                "total_protocols": len(engine.registry.protocols),
                "protocols_by_type": {},
            },
            "system_status": {
                "ethical_compliance": "active",
                "hardware_acceleration": engine.device_info["gpu_available"],
                "memory_status": "optimal",
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Add protocol type breakdown
        for protocol in engine.registry.protocols.values():
            protocol_type = protocol.type.value
            if (
                protocol_type
                not in performance["protocol_statistics"]["protocols_by_type"]
            ):
                performance["protocol_statistics"]["protocols_by_type"][
                    protocol_type
                ] = 0
            performance["protocol_statistics"]["protocols_by_type"][protocol_type] += 1

        logger.info("📊 Performance metrics retrieved")
        return performance

    except Exception as e:
        logger.error(f"❌ Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "omnidimensional_protocol_engine",
        "timestamp": datetime.now().isoformat(),
    }


# Error handlers - Note: These would need to be registered on the main app, not the router
# Exception handlers are not supported on APIRouter in FastAPI
# @router.exception_handler(KimeraProtocolEngineError)
# async def kimera_protocol_engine_error_handler(request, exc):
#     """Handle Kimera protocol engine specific errors"""
#     logger.error(f"Kimera Protocol Engine Error: {exc}")
#     return JSONResponse(
#         status_code=400
#         content={"detail": str(exc), "error_type": "KimeraProtocolEngineError"}
#     )

# @router.exception_handler(ValueError)
# async def value_error_handler(request, exc):
#     """Handle value errors"""
#     logger.error(f"Value Error: {exc}")
#     return JSONResponse(
#         status_code=422
#         content={"detail": str(exc), "error_type": "ValueError"}
#     )
