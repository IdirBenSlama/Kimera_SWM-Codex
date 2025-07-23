"""
Multi-Exchange Trading Orchestrator

Coordinates trading across multiple exchanges (Binance, Phemex, etc.)
with Kimera's cognitive analysis for cross-exchange opportunities.
"""

import os
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.trading.core.trading_engine import KimeraTradingEngine, MarketState, TradingDecision
from src.trading.api.binance_connector import BinanceConnector
from src.trading.api.phemex_connector import PhemexConnector
from src.trading.monitoring.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class Exchange(Enum):
    """Supported exchanges"""
    BINANCE = "binance"
    PHEMEX = "phemex"


@dataclass
class CrossExchangeOpportunity:
    """Cross-exchange arbitrage or trading opportunity"""
    symbol: str
    exchange_buy: Exchange
    exchange_sell: Exchange
    price_buy: float
    price_sell: float
    spread: float
    spread_percent: float
    cognitive_score: float
    reasoning: List[str]


class MultiExchangeOrchestrator:
    """
    Orchestrates trading across multiple exchanges with Kimera's cognitive analysis.
    Identifies cross-exchange opportunities and manages unified portfolio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-exchange orchestrator.
        
        Args:
            config: Configuration including API keys for each exchange
        """
        self.config = config
        self.trading_engine = KimeraTradingEngine(config)
        self.performance_tracker = PerformanceTracker()
        
        # Initialize exchange connectors
        self.exchanges = {}
        
        if config.get("binance_enabled", False):
            self.exchanges[Exchange.BINANCE] = BinanceConnector(
                api_key=config.get("binance_api_key", ""),
                api_secret=config.get("binance_api_secret", ""),
                testnet=config.get("testnet", os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true')
            )
        
        if config.get("phemex_enabled", False):
            self.exchanges[Exchange.PHEMEX] = PhemexConnector(
                api_key=config.get("phemex_api_key", ""),
                api_secret=config.get("phemex_api_secret", ""),
                testnet=config.get("testnet", os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true')
            )
        
        # Trading state per exchange
        self.market_states: Dict[Exchange, Dict[str, MarketState]] = {
            ex: {} for ex in self.exchanges
        }
        self.positions: Dict[Exchange, Dict[str, Any]] = {
            ex: {} for ex in self.exchanges
        }
        
        # Cross-exchange analysis
        self.arbitrage_threshold = config.get("arbitrage_threshold", 0.001)  # 0.1%
        self.unified_symbols = self._map_unified_symbols()
        
        self.is_running = False
        
        logger.info(f"Multi-exchange orchestrator initialized with {len(self.exchanges)} exchanges")
    
    def _map_unified_symbols(self) -> Dict[str, Dict[Exchange, str]]:
        """Map unified symbols to exchange-specific symbols"""
        return {
            "BTC/USDT": {
                Exchange.BINANCE: "BTCUSDT",
                Exchange.PHEMEX: "BTCUSD"  # Phemex uses USD for perpetuals
            },
            "ETH/USDT": {
                Exchange.BINANCE: "ETHUSDT",
                Exchange.PHEMEX: "ETHUSD"
            },
            "BNB/USDT": {
                Exchange.BINANCE: "BNBUSDT",
                Exchange.PHEMEX: None  # Not available on Phemex
            }
        }
    
    async def start_trading(self) -> None:
        """Start multi-exchange trading system"""
        try:
            logger.info("Starting multi-exchange trading system...")
            
            # Connect to all exchanges
            for exchange in self.exchanges.values():
                await exchange.__aenter__()
            
            self.is_running = True
            
            # Create tasks for each exchange and cross-exchange analysis
            tasks = []
            
            # Per-exchange monitoring
            for exchange_type in self.exchanges:
                tasks.append(
                    asyncio.create_task(
                        self._exchange_monitor_loop(exchange_type)
                    )
                )
            
            # Cross-exchange analysis
            tasks.append(
                asyncio.create_task(self._cross_exchange_analysis_loop())
            )
            
            # Unified decision making
            tasks.append(
                asyncio.create_task(self._unified_decision_loop())
            )
            
            logger.info("Multi-exchange system started successfully")
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Failed to start multi-exchange trading: {str(e)}")
            await self.stop_trading()
            raise
    
    async def stop_trading(self) -> None:
        """Stop multi-exchange trading system"""
        logger.info("Stopping multi-exchange trading system...")
        
        self.is_running = False
        
        # Close all exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()
        
        logger.info("Multi-exchange system stopped")
    
    async def _exchange_monitor_loop(self, exchange_type: Exchange) -> None:
        """Monitor a specific exchange"""
        exchange = self.exchanges[exchange_type]
        
        while self.is_running:
            try:
                # Update market data for all symbols
                for unified_symbol, mapping in self.unified_symbols.items():
                    exchange_symbol = mapping.get(exchange_type)
                    
                    if exchange_symbol:
                        # Get market data
                        if exchange_type == Exchange.BINANCE:
                            market_data = await exchange.get_market_data(exchange_symbol)
                        else:  # Phemex
                            market_data = await exchange.get_market_data(exchange_symbol)
                        
                        # Analyze with Kimera
                        market_state = await self.trading_engine.analyze_market(
                            exchange_symbol, market_data
                        )
                        
                        # Store market state
                        if exchange_type not in self.market_states:
                            self.market_states[exchange_type] = {}
                        
                        self.market_states[exchange_type][unified_symbol] = market_state
                        
                        # Log significant changes
                        if market_state.cognitive_pressure > 0.8:
                            logger.warning(
                                f"{exchange_type.value} - {unified_symbol}: "
                                f"High cognitive pressure: {market_state.cognitive_pressure}"
                            )
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"{exchange_type.value} monitoring error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _cross_exchange_analysis_loop(self) -> None:
        """Analyze cross-exchange opportunities"""
        while self.is_running:
            try:
                opportunities = []
                
                # Check each unified symbol across exchanges
                for unified_symbol in self.unified_symbols:
                    states = {}
                    
                    # Gather states from all exchanges
                    for exchange_type in self.exchanges:
                        if (exchange_type in self.market_states and 
                            unified_symbol in self.market_states[exchange_type]):
                            states[exchange_type] = self.market_states[exchange_type][unified_symbol]
                    
                    # Need at least 2 exchanges for arbitrage
                    if len(states) >= 2:
                        opportunity = self._analyze_arbitrage(unified_symbol, states)
                        if opportunity:
                            opportunities.append(opportunity)
                
                # Log opportunities
                if opportunities:
                    for opp in opportunities:
                        logger.info(
                            f"🎯 Arbitrage Opportunity: {opp.symbol} - "
                            f"Buy on {opp.exchange_buy.value} @ ${opp.price_buy:,.2f}, "
                            f"Sell on {opp.exchange_sell.value} @ ${opp.price_sell:,.2f} "
                            f"({opp.spread_percent:.3f}%)"
                        )
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Cross-exchange analysis error: {str(e)}")
                await asyncio.sleep(2)
    
    def _analyze_arbitrage(
        self, 
        symbol: str, 
        states: Dict[Exchange, MarketState]
    ) -> Optional[CrossExchangeOpportunity]:
        """Analyze arbitrage opportunity between exchanges"""
        prices = {ex: state.price for ex, state in states.items()}
        
        # Find min and max prices
        min_exchange = min(prices, key=prices.get)
        max_exchange = max(prices, key=prices.get)
        
        min_price = prices[min_exchange]
        max_price = prices[max_exchange]
        
        spread = max_price - min_price
        spread_percent = (spread / min_price) if min_price > 0 else 0
        
        # Check if spread exceeds threshold
        if spread_percent > self.arbitrage_threshold:
            # Calculate cognitive score for the opportunity
            min_state = states[min_exchange]
            max_state = states[max_exchange]
            
            # Higher score if both markets show favorable conditions
            cognitive_score = (
                (1 - min_state.cognitive_pressure) * 0.3 +  # Low pressure good for buying
                (1 - max_state.cognitive_pressure) * 0.3 +  # Low pressure good for selling
                min_state.contradiction_level * 0.2 +        # Inefficiencies create opportunities
                max_state.contradiction_level * 0.2
            )
            
            reasoning = []
            if min_state.cognitive_pressure < 0.3:
                reasoning.append(f"Low buy pressure on {min_exchange.value}")
            if max_state.cognitive_pressure < 0.3:
                reasoning.append(f"Low sell pressure on {max_exchange.value}")
            if spread_percent > self.arbitrage_threshold * 2:
                reasoning.append("Significant price divergence")
            
            return CrossExchangeOpportunity(
                symbol=symbol,
                exchange_buy=min_exchange,
                exchange_sell=max_exchange,
                price_buy=min_price,
                price_sell=max_price,
                spread=spread,
                spread_percent=spread_percent,
                cognitive_score=cognitive_score,
                reasoning=reasoning
            )
        
        return None
    
    async def _unified_decision_loop(self) -> None:
        """Make unified trading decisions across exchanges"""
        while self.is_running:
            try:
                # Get portfolio state across all exchanges
                unified_portfolio = await self._get_unified_portfolio()
                
                # Make decisions for each symbol on each exchange
                for unified_symbol in self.unified_symbols:
                    for exchange_type in self.exchanges:
                        if (exchange_type in self.market_states and 
                            unified_symbol in self.market_states[exchange_type]):
                            
                            market_state = self.market_states[exchange_type][unified_symbol]
                            
                            # Adjust portfolio state for exchange
                            exchange_portfolio = {
                                "total_value": unified_portfolio["total_value"],
                                "free_balance": unified_portfolio["balances"].get(
                                    exchange_type, {}).get("free", 0
                                ),
                                "positions": self.positions.get(exchange_type, {}),
                                "daily_pnl": 0,
                                "margin_used": self._calculate_margin_used(exchange_type)
                            }
                            
                            # Make decision
                            decision = await self.trading_engine.make_trading_decision(
                                unified_symbol, market_state, exchange_portfolio
                            )
                            
                            # Log significant decisions
                            if decision.action != "HOLD" and decision.confidence > 0.5:
                                logger.info(
                                    f"📊 {exchange_type.value} - {unified_symbol}: "
                                    f"{decision.action} (confidence: {decision.confidence:.2f})"
                                )
                
                await asyncio.sleep(30)  # Decision interval
                
            except Exception as e:
                logger.error(f"Unified decision error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _get_unified_portfolio(self) -> Dict[str, Any]:
        """Get unified portfolio across all exchanges"""
        total_value = 0
        balances = {}
        
        for exchange_type, exchange in self.exchanges.items():
            try:
                if exchange_type == Exchange.BINANCE:
                    account = await exchange.get_account()
                    # Extract USDT balance
                    for balance in account.get("balances", []):
                        if balance["asset"] == "USDT":
                            usdt_balance = float(balance["free"]) + float(balance["locked"])
                            balances[exchange_type] = {
                                "free": float(balance["free"]),
                                "total": usdt_balance
                            }
                            total_value += usdt_balance
                            
                elif exchange_type == Exchange.PHEMEX:
                    balance = await exchange.get_wallet_balance("USDT")
                    balances[exchange_type] = {
                        "free": balance["available"],
                        "total": balance["balance"]
                    }
                    total_value += balance["balance"]
                    
            except Exception as e:
                logger.error(f"Failed to get {exchange_type.value} balance: {str(e)}")
        
        return {
            "total_value": total_value,
            "balances": balances,
            "positions": self.positions
        }
    
    def _calculate_margin_used(self, exchange_type: Exchange) -> float:
        """Calculate margin used on an exchange"""
        positions = self.positions.get(exchange_type, {})
        return sum(pos.get("value", 0) for pos in positions.values())
    
    async def execute_arbitrage(self, opportunity: CrossExchangeOpportunity) -> None:
        """Execute an arbitrage trade"""
        try:
            # Calculate position size based on available balance and opportunity
            buy_exchange = self.exchanges[opportunity.exchange_buy]
            sell_exchange = self.exchanges[opportunity.exchange_sell]
            
            # This would execute the actual trades
            # For safety, this is commented out in the demo
            
            logger.info(
                f"Would execute arbitrage: Buy {opportunity.symbol} on "
                f"{opportunity.exchange_buy.value} and sell on "
                f"{opportunity.exchange_sell.value}"
            )
            
        except Exception as e:
            logger.error(f"Arbitrage execution failed: {str(e)}") 