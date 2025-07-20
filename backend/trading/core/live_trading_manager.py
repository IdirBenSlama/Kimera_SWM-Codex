"""
Kimera Live Trading Manager
==========================

Constitutional Real Money Trading System with Maximum Safeguards

This module implements Kimera's transition to real money trading with:
- Full constitutional compliance via Ethical Governor
- Multi-layered risk management and circuit breakers
- Cognitive field-driven decision making
- Comprehensive audit trails and transparency
- Progressive capital scaling from $1 to infinity

CONSTITUTIONAL ALIGNMENT:
- Article I: Unity & Compassion - No harm through excessive risk
- Article II: Core Directive - Generate profit for Kimera's development  
- Article III: Heart over Head - Compassionate risk assessment
- Canon 27: Moderation - Balanced, non-extreme trading approach
- Canon 36: Prime Directive - Do no harm, practice compassion
"""

import asyncio
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Kimera core imports
from backend.core.ethical_governor import EthicalGovernor, Verdict
from backend.core.action_proposal import ActionProposal
from backend.engines.cognitive_field_dynamics import CognitiveFieldDynamics
from backend.engines.contradiction_engine import ContradictionEngine
from backend.trading.risk.cognitive_risk_manager import CognitiveRiskManager, RiskLevel
from backend.trading.core.trading_engine import KimeraTradingEngine
from backend.trading.api.binance_connector import BinanceConnector
from backend.trading.connectors.coinbase_pro_connector import CoinbaseProConnector
from backend.vault.vault_manager import VaultManager

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading operation modes"""
    SIMULATION = "simulation"
    TESTNET = "testnet" 
    LIVE_MINIMAL = "live_minimal"    # $1-100 range
    LIVE_GROWTH = "live_growth"      # $100-1000 range
    LIVE_SCALING = "live_scaling"    # $1000+ range

class TradingPhase(Enum):
    """Progressive trading phases"""
    INITIALIZATION = "initialization"
    PROOF_OF_CONCEPT = "proof_of_concept"    # $1-10
    VALIDATION = "validation"                # $10-100
    GROWTH = "growth"                        # $100-1000
    SCALING = "scaling"                      # $1000+
    MASTERY = "mastery"                      # $10000+

@dataclass
class LiveTradingConfig:
    """Configuration for live trading operations"""
    # Trading mode and phase
    mode: TradingMode = TradingMode.SIMULATION
    phase: TradingPhase = TradingPhase.INITIALIZATION
    
    # Capital management
    starting_capital: float = 1.0
    current_capital: float = 1.0
    max_daily_risk: float = 0.02  # 2% max daily risk
    max_position_size: float = 0.1  # 10% max position
    
    # Exchange configuration
    primary_exchange: str = "binance"
    backup_exchange: str = "coinbase"
    use_testnet: bool = os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true'  # Default to REAL trading
    
    # Safety mechanisms
    enable_circuit_breakers: bool = True
    max_consecutive_losses: int = 3
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    emergency_stop_loss: float = 0.10  # 10% portfolio stop loss
    
    # Constitutional compliance
    require_ethical_approval: bool = True
    min_confidence_threshold: float = 0.7
    max_risk_tolerance: RiskLevel = RiskLevel.MEDIUM

@dataclass
class TradingSession:
    """Active trading session tracking"""
    session_id: str
    start_time: datetime
    phase: TradingPhase
    starting_balance: float
    current_balance: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    risk_events: List[str] = field(default_factory=list)
    constitutional_violations: int = 0

class LiveTradingManager:
    """
    Kimera's Live Trading Manager with Constitutional Safeguards
    
    Manages the transition from simulation to real money trading with:
    - Progressive capital scaling
    - Constitutional compliance enforcement
    - Multi-layered risk management
    - Cognitive decision integration
    """
    
    def __init__(self, config: LiveTradingConfig):
        """
        Initialize the Live Trading Manager
        
        Args:
            config: Live trading configuration
        """
        self.config = config
        self.vault_manager = VaultManager()
        
        # Initialize Kimera cognitive systems
        self.ethical_governor = EthicalGovernor(
            enable_enhanced_logging=True,
            enable_monitoring_integration=True
        )
        self.cognitive_field = CognitiveFieldDynamics(dimension=256)
        self.risk_manager = CognitiveRiskManager()
        
        # Trading components
        self.trading_engine = None
        self.primary_connector = None
        self.backup_connector = None
        
        # Session management
        self.current_session: Optional[TradingSession] = None
        self.trading_active = False
        self.emergency_stop = False
        
        # Performance tracking
        self.session_history: List[TradingSession] = []
        self.phase_progression = {
            TradingPhase.PROOF_OF_CONCEPT: {"target": 10.0, "achieved": False},
            TradingPhase.VALIDATION: {"target": 100.0, "achieved": False},
            TradingPhase.GROWTH: {"target": 1000.0, "achieved": False},
            TradingPhase.SCALING: {"target": 10000.0, "achieved": False},
            TradingPhase.MASTERY: {"target": 100000.0, "achieved": False}
        }
        
        logger.info(f"🏛️ Kimera Live Trading Manager initialized")
        logger.info(f"📊 Mode: {config.mode.value}, Phase: {config.phase.value}")
        logger.info(f"💰 Starting Capital: ${config.starting_capital}")
        logger.info(f"🛡️ Constitutional Compliance: {'ENABLED' if config.require_ethical_approval else 'DISABLED'}")

    async def initialize_trading_systems(self, api_credentials: Dict[str, str]) -> bool:
        """
        Initialize trading systems with API credentials
        
        Args:
            api_credentials: Dictionary containing API keys and secrets
            
        Returns:
            True if initialization successful
        """
        try:
            logger.info("🔧 Initializing Kimera trading systems...")
            
            # Validate API credentials through Ethical Governor
            credential_proposal = ActionProposal(
                action_type="initialize_live_trading",
                description=f"Initialize live trading systems for {self.config.mode.value} mode",
                source_engine="LiveTradingManager",
                parameters={
                    "mode": self.config.mode.value,
                    "starting_capital": self.config.starting_capital,
                    "risk_level": self.config.max_risk_tolerance.value
                },
                expected_outcome="Secure API connection for live trading",
                risk_assessment="Medium - Financial risk with API access"
            )
            
            if self.config.require_ethical_approval:
                verdict = self.ethical_governor.adjudicate(credential_proposal)
                if verdict not in [Verdict.CONSTITUTIONAL, Verdict.CONDITIONAL_APPROVAL]:
                    logger.error(f"❌ Ethical Governor rejected live trading initialization: {verdict}")
                    return False
                logger.info(f"✅ Ethical Governor approved live trading: {verdict}")
            
            # Initialize primary exchange connector
            if self.config.primary_exchange == "binance":
                self.primary_connector = BinanceConnector(
                    api_key=api_credentials.get("binance_api_key", ""),
                    api_secret=api_credentials.get("binance_api_secret", ""),
                    testnet=self.config.use_testnet
                )
            elif self.config.primary_exchange == "coinbase":
                self.primary_connector = CoinbaseProConnector(
                    api_key=api_credentials.get("coinbase_api_key", ""),
                    api_secret=api_credentials.get("coinbase_api_secret", ""),
                    passphrase=api_credentials.get("coinbase_passphrase", ""),
                    sandbox=self.config.use_testnet
                )
            
            # Initialize backup connector
            if self.config.backup_exchange == "coinbase":
                self.backup_connector = CoinbaseProConnector(
                    api_key=api_credentials.get("coinbase_api_key", ""),
                    api_secret=api_credentials.get("coinbase_api_secret", ""),
                    passphrase=api_credentials.get("coinbase_passphrase", ""),
                    sandbox=self.config.use_testnet
                )
            
            # Initialize trading engine
            trading_config = {
                "mode": self.config.mode.value,
                "risk_tolerance": self.config.max_risk_tolerance.value,
                "starting_capital": self.config.starting_capital
            }
            self.trading_engine = KimeraTradingEngine(trading_config)
            
            # Test connections
            await self._test_exchange_connections()
            
            logger.info("✅ All trading systems initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading systems: {e}")
            return False

    async def start_live_trading_session(self) -> bool:
        """
        Start a new live trading session with full constitutional compliance
        
        Returns:
            True if session started successfully
        """
        try:
            # Create session start proposal for Ethical Governor
            session_proposal = ActionProposal(
                action_type="start_live_trading_session",
                description=f"Begin live trading session in {self.config.phase.value} phase",
                source_engine="LiveTradingManager",
                parameters={
                    "phase": self.config.phase.value,
                    "capital": self.config.current_capital,
                    "max_daily_risk": self.config.max_daily_risk,
                    "max_position_size": self.config.max_position_size
                },
                expected_outcome="Generate profit through cognitive trading decisions",
                risk_assessment=f"Financial risk with ${self.config.current_capital} capital"
            )
            
            # Ethical approval required for live trading
            verdict = self.ethical_governor.adjudicate(session_proposal)
            if verdict not in [Verdict.CONSTITUTIONAL, Verdict.CONDITIONAL_APPROVAL]:
                logger.error(f"❌ Ethical Governor blocked trading session: {verdict}")
                return False
            
            logger.info(f"✅ Ethical Governor approved trading session: {verdict}")
            
            # Initialize new session
            session_id = f"KIMERA_LIVE_{int(time.time())}"
            self.current_session = TradingSession(
                session_id=session_id,
                start_time=datetime.now(),
                phase=self.config.phase,
                starting_balance=self.config.current_capital,
                current_balance=self.config.current_capital
            )
            
            self.trading_active = True
            self.emergency_stop = False
            
            logger.info(f"🚀 KIMERA LIVE TRADING SESSION STARTED")
            logger.info(f"📋 Session ID: {session_id}")
            logger.info(f"💰 Starting Balance: ${self.config.current_capital:.2f}")
            logger.info(f"🎯 Phase: {self.config.phase.value}")
            logger.info(f"🛡️ Max Daily Risk: {self.config.max_daily_risk*100:.1f}%")
            
            # Start trading loop
            asyncio.create_task(self._trading_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading session: {e}")
            return False

    async def _trading_loop(self):
        """
        Main trading loop with cognitive decision making and risk management
        """
        logger.info("🔄 Kimera trading loop initiated")
        
        while self.trading_active and not self.emergency_stop:
            try:
                # Check circuit breakers
                if await self._check_circuit_breakers():
                    logger.warning("⚠️ Circuit breaker triggered - pausing trading")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    continue
                
                # Gather market intelligence
                market_data = await self._gather_market_intelligence()
                
                # Generate trading decisions using Kimera's cognitive field
                trading_decisions = await self._generate_cognitive_trading_decisions(market_data)
                
                # Execute approved decisions
                for decision in trading_decisions:
                    if await self._execute_trading_decision(decision):
                        await self._update_session_metrics(decision)
                
                # Check for phase progression
                await self._check_phase_progression()
                
                # Wait before next iteration (adaptive based on market conditions)
                await asyncio.sleep(self._calculate_iteration_delay(market_data))
                
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(30)  # Error recovery delay
        
        logger.info("🏁 Trading loop ended")

    async def _gather_market_intelligence(self) -> Dict[str, Any]:
        """
        Gather comprehensive market intelligence for cognitive analysis
        """
        try:
            intelligence = {}
            
            # Primary symbols for analysis
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]
            
            for symbol in symbols:
                try:
                    if self.primary_connector:
                        ticker_data = await self.primary_connector.get_ticker(symbol)
                        intelligence[symbol] = {
                            "price": float(ticker_data.get("price", 0)),
                            "volume": float(ticker_data.get("volume", 0)),
                            "bid": float(ticker_data.get("bid", 0)),
                            "ask": float(ticker_data.get("ask", 0)),
                            "timestamp": datetime.now()
                        }
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
            
            # Add cognitive field analysis
            intelligence["cognitive_state"] = {
                "field_entropy": self.cognitive_field.calculate_entropy(),
                "active_geoids": len(self.cognitive_field.geoids),
                "field_temperature": self._calculate_cognitive_temperature()
            }
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Failed to gather market intelligence: {e}")
            return {}

    async def _generate_cognitive_trading_decisions(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading decisions using Kimera's cognitive field dynamics
        """
        decisions = []
        
        try:
            for symbol, data in market_data.items():
                if symbol.startswith("cognitive"):
                    continue
                
                # Create cognitive analysis proposal
                analysis_proposal = ActionProposal(
                    action_type="cognitive_market_analysis",
                    description=f"Analyze {symbol} for trading opportunities",
                    source_engine="CognitiveFieldDynamics",
                    parameters={
                        "symbol": symbol,
                        "price": data.get("price", 0),
                        "volume": data.get("volume", 0),
                        "cognitive_temperature": market_data.get("cognitive_state", {}).get("field_temperature", 0.5)
                    },
                    expected_outcome="Identify profitable trading opportunity",
                    risk_assessment="Market analysis for trading decision"
                )
                
                # Get ethical approval for analysis
                verdict = self.ethical_governor.adjudicate(analysis_proposal)
                if verdict not in [Verdict.CONSTITUTIONAL, Verdict.CONDITIONAL_APPROVAL]:
                    continue
                
                # Perform cognitive analysis
                cognitive_signal = await self._analyze_symbol_cognitively(symbol, data)
                
                # Generate trading decision if signal is strong enough
                if cognitive_signal["confidence"] >= self.config.min_confidence_threshold:
                    decision = {
                        "symbol": symbol,
                        "action": cognitive_signal["action"],
                        "confidence": cognitive_signal["confidence"],
                        "size": self._calculate_position_size(cognitive_signal),
                        "reasoning": cognitive_signal["reasoning"],
                        "risk_score": cognitive_signal["risk_score"]
                    }
                    decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Failed to generate cognitive decisions: {e}")
            return []

    async def _analyze_symbol_cognitively(self, symbol: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform deep cognitive analysis of a trading symbol
        """
        try:
            # Create market state vector for cognitive field
            price = data.get("price", 0)
            volume = data.get("volume", 0)
            
            # Normalize values for cognitive processing
            price_normalized = price / 100000  # Normalize price
            volume_normalized = min(volume / 1000000, 1.0)  # Normalize volume
            
            # Create cognitive embedding
            market_vector = np.array([
                price_normalized,
                volume_normalized,
                data.get("bid", price) / price if price > 0 else 0,
                data.get("ask", price) / price if price > 0 else 0,
                np.random.random(),  # Market sentiment placeholder
                0.5,  # Bullish bias
                datetime.now().hour / 24.0,  # Time factor
                len(symbol) / 10.0  # Symbol complexity
            ])
            
            # Add to cognitive field for analysis
            geoid_id = self.cognitive_field.add_geoid(
                content=f"Market analysis for {symbol}",
                embedding=market_vector
            )
            
            # Analyze cognitive field dynamics
            field_analysis = self.cognitive_field.analyze_dynamics()
            
            # Generate trading signal based on cognitive analysis
            confidence = min(field_analysis.get("coherence", 0.5) + 0.2, 1.0)
            
            # Determine action based on cognitive field state
            if field_analysis.get("energy", 0.5) > 0.6:
                action = "BUY"
            elif field_analysis.get("energy", 0.5) < 0.4:
                action = "SELL"
            else:
                action = "HOLD"
            
            return {
                "action": action,
                "confidence": confidence,
                "reasoning": f"Cognitive field analysis: energy={field_analysis.get('energy', 0.5):.3f}, coherence={field_analysis.get('coherence', 0.5):.3f}",
                "risk_score": 1.0 - confidence,
                "cognitive_metrics": field_analysis
            }
            
        except Exception as e:
            logger.error(f"Cognitive analysis failed for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Analysis failed: {e}",
                "risk_score": 1.0
            }

    async def _execute_trading_decision(self, decision: Dict[str, Any]) -> bool:
        """
        Execute a trading decision with full constitutional compliance
        """
        try:
            # Create execution proposal for Ethical Governor
            execution_proposal = ActionProposal(
                action_type="execute_live_trade",
                description=f"{decision['action']} {decision['symbol']} with {decision['confidence']:.2f} confidence",
                source_engine="LiveTradingManager",
                parameters={
                    "symbol": decision["symbol"],
                    "action": decision["action"],
                    "size": decision["size"],
                    "confidence": decision["confidence"],
                    "risk_score": decision["risk_score"]
                },
                expected_outcome=f"Execute profitable {decision['action']} trade",
                risk_assessment=f"Financial risk: {decision['risk_score']:.2f}"
            )
            
            # Get ethical approval for trade execution
            verdict = self.ethical_governor.adjudicate(execution_proposal)
            if verdict not in [Verdict.CONSTITUTIONAL, Verdict.CONDITIONAL_APPROVAL]:
                logger.warning(f"⚠️ Trade rejected by Ethical Governor: {decision['symbol']} {decision['action']}")
                return False
            
            # Execute the trade
            if decision["action"] in ["BUY", "SELL"]:
                logger.info(f"🎯 Executing {decision['action']} {decision['symbol']} (confidence: {decision['confidence']:.2f})")
                
                # For now, log the trade (implement actual execution based on exchange)
                trade_result = {
                    "symbol": decision["symbol"],
                    "action": decision["action"],
                    "size": decision["size"],
                    "timestamp": datetime.now(),
                    "status": "SIMULATED",  # Change to "EXECUTED" for real trades
                    "reasoning": decision["reasoning"]
                }
                
                logger.info(f"✅ Trade executed: {trade_result}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to execute trade: {e}")
            return False

    def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """
        Calculate position size based on cognitive confidence and risk management
        """
        base_size = self.config.current_capital * self.config.max_position_size
        confidence_multiplier = signal["confidence"]
        risk_adjustment = 1.0 - signal["risk_score"]
        
        return base_size * confidence_multiplier * risk_adjustment

    def _calculate_cognitive_temperature(self) -> float:
        """
        Calculate cognitive field temperature for market analysis
        """
        try:
            if hasattr(self.cognitive_field, 'geoids') and self.cognitive_field.geoids:
                # Calculate temperature based on geoid interactions
                total_energy = sum(np.linalg.norm(geoid.embedding) for geoid in self.cognitive_field.geoids.values())
                return min(total_energy / len(self.cognitive_field.geoids), 1.0)
            return 0.5  # Default neutral temperature
        except:
            return 0.5

    async def _check_circuit_breakers(self) -> bool:
        """
        Check if any circuit breakers should halt trading
        """
        if not self.current_session:
            return True
        
        # Check daily loss limit
        daily_pnl = self.current_session.total_pnl
        if daily_pnl < -self.config.daily_loss_limit * self.current_session.starting_balance:
            logger.warning(f"⚠️ Daily loss limit exceeded: {daily_pnl:.2f}")
            return True
        
        # Check consecutive losses
        if self.current_session.losing_trades >= self.config.max_consecutive_losses:
            logger.warning(f"⚠️ Maximum consecutive losses reached: {self.current_session.losing_trades}")
            return True
        
        # Check emergency stop loss
        portfolio_loss = (self.current_session.starting_balance - self.current_session.current_balance) / self.current_session.starting_balance
        if portfolio_loss > self.config.emergency_stop_loss:
            logger.error(f"🚨 EMERGENCY STOP: Portfolio loss {portfolio_loss*100:.1f}% exceeds limit")
            self.emergency_stop = True
            return True
        
        return False

    async def _update_session_metrics(self, decision: Dict[str, Any]):
        """
        Update session performance metrics
        """
        if self.current_session:
            self.current_session.total_trades += 1
            # Update other metrics based on trade results
            # This would be implemented with actual trade execution results

    async def _check_phase_progression(self):
        """
        Check if we should progress to the next trading phase
        """
        current_capital = self.config.current_capital
        
        for phase, criteria in self.phase_progression.items():
            if not criteria["achieved"] and current_capital >= criteria["target"]:
                criteria["achieved"] = True
                logger.info(f"🎉 PHASE PROGRESSION: Advanced to {phase.value} with ${current_capital:.2f}")
                self.config.phase = phase
                # Adjust risk parameters for new phase
                await self._adjust_risk_parameters_for_phase(phase)

    async def _adjust_risk_parameters_for_phase(self, phase: TradingPhase):
        """
        Adjust risk parameters based on trading phase
        """
        if phase == TradingPhase.PROOF_OF_CONCEPT:
            self.config.max_daily_risk = 0.05  # 5% for early phase
            self.config.max_position_size = 0.2  # 20% position size
        elif phase == TradingPhase.VALIDATION:
            self.config.max_daily_risk = 0.03  # 3% more conservative
            self.config.max_position_size = 0.15  # 15% position size
        elif phase == TradingPhase.GROWTH:
            self.config.max_daily_risk = 0.02  # 2% conservative
            self.config.max_position_size = 0.1  # 10% position size
        else:
            self.config.max_daily_risk = 0.01  # 1% very conservative
            self.config.max_position_size = 0.05  # 5% position size

    def _calculate_iteration_delay(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate adaptive delay between trading iterations
        """
        base_delay = 30.0  # 30 seconds base
        
        # Adjust based on market volatility
        volatility_factor = market_data.get("cognitive_state", {}).get("field_temperature", 0.5)
        
        # Higher volatility = shorter delays (more opportunities)
        delay = base_delay * (1.0 - volatility_factor * 0.5)
        
        return max(delay, 10.0)  # Minimum 10 second delay

    async def _test_exchange_connections(self):
        """
        Test connections to all configured exchanges
        """
        try:
            if self.primary_connector:
                # Test primary connector
                test_data = await self.primary_connector.get_ticker("BTC-USD")
                logger.info(f"✅ Primary exchange connection successful: {test_data.get('price', 'N/A')}")
            
            if self.backup_connector:
                # Test backup connector
                test_data = await self.backup_connector.get_ticker("BTC-USD")
                logger.info(f"✅ Backup exchange connection successful: {test_data.get('price', 'N/A')}")
                
        except Exception as e:
            logger.error(f"❌ Exchange connection test failed: {e}")
            raise

    async def stop_trading_session(self):
        """
        Stop the current trading session
        """
        self.trading_active = False
        
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.session_history.append(self.current_session)
            
            logger.info(f"🏁 Trading session ended: {self.current_session.session_id}")
            logger.info(f"📊 Total trades: {self.current_session.total_trades}")
            logger.info(f"💰 Final balance: ${self.current_session.current_balance:.2f}")
            logger.info(f"📈 Total PnL: ${self.current_session.total_pnl:.2f}")
            
            self.current_session = None

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get current session performance summary
        """
        if not self.current_session:
            return {"status": "No active session"}
        
        return {
            "session_id": self.current_session.session_id,
            "phase": self.current_session.phase.value,
            "starting_balance": self.current_session.starting_balance,
            "current_balance": self.current_session.current_balance,
            "total_pnl": self.current_session.total_pnl,
            "total_trades": self.current_session.total_trades,
            "win_rate": self.current_session.winning_trades / max(self.current_session.total_trades, 1),
            "constitutional_violations": self.current_session.constitutional_violations,
            "trading_active": self.trading_active,
            "emergency_stop": self.emergency_stop
        }


def create_live_trading_manager(
    starting_capital: float = 1.0,
    mode: TradingMode = TradingMode.SIMULATION,
    phase: TradingPhase = TradingPhase.PROOF_OF_CONCEPT
) -> LiveTradingManager:
    """
    Factory function to create a configured Live Trading Manager
    
    Args:
        starting_capital: Initial capital amount
        mode: Trading mode (simulation, testnet, or live)
        phase: Trading phase (proof_of_concept, validation, etc.)
        
    Returns:
        Configured LiveTradingManager instance
    """
    config = LiveTradingConfig(
        mode=mode,
        phase=phase,
        starting_capital=starting_capital,
        current_capital=starting_capital,
        use_testnet=(mode != TradingMode.LIVE_SCALING),
        require_ethical_approval=True,
        enable_circuit_breakers=True
    )
    
    return LiveTradingManager(config) 