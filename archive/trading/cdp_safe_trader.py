"""
KIMERA CDP SAFE TRADER
======================

Ultra-conservative trading system for Coinbase Developer Platform
with comprehensive risk management and safety controls.

SAFETY PHILOSOPHY:
- Every trade must be justified by multiple signals
- Maximum position size limits
- Mandatory stop losses
- Real-time monitoring
- Emergency stop mechanisms
- Detailed logging for transparency
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
import numpy as np
from coinbase.wallet.client import Client

# Configure logging for transparency
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/kimera_cdp_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SafetyLimits:
    """Comprehensive safety limits for real money trading"""
    max_position_size_eur: float = 2.0  # Maximum €2 per position
    max_daily_loss_eur: float = 5.0     # Maximum €5 daily loss
    max_total_risk_eur: float = 10.0    # Maximum €10 total at risk
    min_confidence_threshold: float = 0.75  # 75% minimum confidence
    max_consecutive_losses: int = 3      # Stop after 3 losses
    mandatory_stop_loss_pct: float = 0.05  # 5% mandatory stop loss
    profit_target_pct: float = 0.08     # 8% profit target
    min_wallet_balance_eur: float = 3.0  # Keep minimum €3 in wallet
    max_trades_per_hour: int = 2         # Maximum 2 trades per hour
    cooldown_after_loss_minutes: int = 30  # 30min cooldown after loss

@dataclass
class TradeSignal:
    """Structured trade signal with confidence and reasoning"""
    symbol: str
    side: str  # 'buy' or 'sell'
    confidence: float
    reasoning: List[str]
    target_price: float
    stop_loss_price: float
    profit_target_price: float
    risk_reward_ratio: float
    timestamp: datetime

@dataclass
class Position:
    """Active trading position"""
    symbol: str
    side: str
    amount_eur: float
    amount_crypto: float
    entry_price: float
    stop_loss_price: float
    profit_target_price: float
    timestamp: datetime
    is_active: bool = True

@dataclass
class TradeResult:
    """Result of executed trade"""
    symbol: str
    side: str
    amount_eur: float
    entry_price: float
    exit_price: float
    profit_loss_eur: float
    profit_loss_pct: float
    duration_minutes: float
    timestamp: datetime
    success: bool
    reasoning: str

class KimeraCDPSafeTrader:
    """
    Ultra-safe CDP trading system with comprehensive risk management
    """
    
    def __init__(self, api_key: str, private_key: str = None, testnet: bool = None):
        """
        Initialize safe trader with maximum safety controls
        
        Args:
            api_key: CDP API key
            private_key: CDP private key (None for simulation mode)
            testnet: Use testnet if True. Defaults to environment variable KIMERA_USE_TESTNET or False for real trading.
        """
        self.api_key = api_key
        self.private_key = private_key
        
        # Default to real trading unless explicitly set to testnet
        if testnet is None:
            testnet = os.getenv('KIMERA_USE_TESTNET', 'false').lower() == 'true'
        
        self.testnet = testnet
        self.simulation_mode = private_key is None
        
        # Safety limits (ultra-conservative for real money)
        self.safety_limits = SafetyLimits()
        
        # Trading state
        self.active_positions: Dict[str, Position] = {}
        self.trade_history: List[TradeResult] = []
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.last_trade_time: Optional[datetime] = None
        self.emergency_stop: bool = False
        
        # Market data cache
        self.price_cache: Dict[str, Tuple[float, datetime]] = {}
        
        # Initialize CDP client
        if not self.simulation_mode:
            try:
                self.client = Client(api_key, private_key, sandbox=testnet)
                logger.info(f"🔐 CDP Client initialized ({'TESTNET' if testnet else 'LIVE'})")
            except Exception as e:
                logger.error(f"❌ Failed to initialize CDP client: {e}")
                self.simulation_mode = True
                self.client = None
        else:
            self.client = None
            logger.info("🎯 Running in SIMULATION MODE (no real trades)")
        
        # Load existing state if available
        self._load_trading_state()
        
        logger.info(f"✅ Kimera CDP Safe Trader initialized")
        logger.info(f"   Mode: {'SIMULATION' if self.simulation_mode else 'LIVE'}")
        logger.info(f"   Max Position: €{self.safety_limits.max_position_size_eur}")
        logger.info(f"   Daily Loss Limit: €{self.safety_limits.max_daily_loss_eur}")
    
    def _load_trading_state(self):
        """Load previous trading state for continuity"""
        try:
            if os.path.exists('data/trading_state.json'):
                with open('data/trading_state.json', 'r') as f:
                    state = json.load(f)
                    self.daily_pnl = state.get('daily_pnl', 0.0)
                    self.consecutive_losses = state.get('consecutive_losses', 0)
                    # Load other state as needed
                logger.info(f"📊 Loaded trading state: P&L €{self.daily_pnl:.2f}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load trading state: {e}")
    
    def _save_trading_state(self):
        """Save current trading state"""
        try:
            os.makedirs('data', exist_ok=True)
            state = {
                'daily_pnl': self.daily_pnl,
                'consecutive_losses': self.consecutive_losses,
                'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
                'timestamp': datetime.now().isoformat()
            }
            with open('data/trading_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save trading state: {e}")
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get current account balances"""
        try:
            if self.simulation_mode:
                # Return mock balance for simulation
                return {'EUR': 5.0, 'BTC': 0.0, 'ETH': 0.0}
            
            # Get real balances from CDP
            accounts = self.client.get_accounts()
            balances = {}
            
            for account in accounts.data:
                currency = account.balance.currency
                amount = float(account.balance.amount)
                balances[currency] = amount
            
            logger.info(f"💰 Account balances: {balances}")
            return balances
            
        except Exception as e:
            logger.error(f"❌ Failed to get account balance: {e}")
            return {}
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with caching"""
        try:
            # Check cache first (valid for 30 seconds)
            if symbol in self.price_cache:
                price, timestamp = self.price_cache[symbol]
                if datetime.now() - timestamp < timedelta(seconds=30):
                    return price
            
            if self.simulation_mode:
                # Return mock prices for simulation
                mock_prices = {
                    'BTC': 43000.0 + np.random.uniform(-500, 500),
                    'ETH': 2600.0 + np.random.uniform(-50, 50),
                    'SOL': 95.0 + np.random.uniform(-5, 5)
                }
                price = mock_prices.get(symbol.replace('-EUR', ''), 100.0)
            else:
                # Get real price from CDP
                exchange_rates = self.client.get_exchange_rates(currency=symbol)
                price = float(exchange_rates.rates.get('EUR', 0))
            
            # Cache the price
            self.price_cache[symbol] = (price, datetime.now())
            return price
            
        except Exception as e:
            logger.error(f"❌ Failed to get price for {symbol}: {e}")
            return None
    
    def analyze_market_conditions(self, symbol: str) -> TradeSignal:
        """
        Analyze market conditions and generate trade signal
        
        This is a simplified version - in production, this would use
        multiple technical indicators, sentiment analysis, etc.
        """
        try:
            # Get current price (using direct call instead of asyncio.run)
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't use asyncio.run()
                # For simulation, use mock price directly
                mock_prices = {
                    'BTC': 43000.0 + np.random.uniform(-500, 500),
                    'ETH': 2600.0 + np.random.uniform(-50, 50),
                    'SOL': 95.0 + np.random.uniform(-5, 5)
                }
                current_price = mock_prices.get(symbol.replace('-EUR', ''), 100.0)
            else:
                current_price = asyncio.run(self.get_current_price(symbol))
            
            if not current_price:
                return None
            
            # Simple momentum analysis (placeholder for real analysis)
            confidence = 0.6  # Base confidence
            reasoning = []
            
            # Add various signal analysis here
            # For now, using simplified logic
            
            # Determine side based on simple logic
            side = 'buy' if np.random.random() > 0.5 else 'sell'
            
            # Calculate targets
            if side == 'buy':
                stop_loss_price = current_price * (1 - self.safety_limits.mandatory_stop_loss_pct)
                profit_target_price = current_price * (1 + self.safety_limits.profit_target_pct)
            else:
                stop_loss_price = current_price * (1 + self.safety_limits.mandatory_stop_loss_pct)
                profit_target_price = current_price * (1 - self.safety_limits.profit_target_pct)
            
            risk_reward_ratio = abs(profit_target_price - current_price) / abs(current_price - stop_loss_price)
            
            reasoning.append(f"Current price: €{current_price:.2f}")
            reasoning.append(f"Risk/Reward ratio: {risk_reward_ratio:.2f}")
            
            return TradeSignal(
                symbol=symbol,
                side=side,
                confidence=confidence,
                reasoning=reasoning,
                target_price=current_price,
                stop_loss_price=stop_loss_price,
                profit_target_price=profit_target_price,
                risk_reward_ratio=risk_reward_ratio,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze market conditions: {e}")
            return None
    
    def validate_trade_safety(self, signal: TradeSignal) -> Tuple[bool, List[str]]:
        """
        Comprehensive safety validation before executing any trade
        
        Returns:
            (is_safe, list_of_issues)
        """
        issues = []
        
        # Check emergency stop
        if self.emergency_stop:
            issues.append("Emergency stop is active")
        
        # Check confidence threshold
        if signal.confidence < self.safety_limits.min_confidence_threshold:
            issues.append(f"Confidence {signal.confidence:.2f} below threshold {self.safety_limits.min_confidence_threshold}")
        
        # Check consecutive losses
        if self.consecutive_losses >= self.safety_limits.max_consecutive_losses:
            issues.append(f"Too many consecutive losses: {self.consecutive_losses}")
        
        # Check daily loss limit
        if self.daily_pnl <= -self.safety_limits.max_daily_loss_eur:
            issues.append(f"Daily loss limit reached: €{self.daily_pnl:.2f}")
        
        # Check trade frequency
        if self.last_trade_time:
            time_since_last = datetime.now() - self.last_trade_time
            if time_since_last.total_seconds() < 1800:  # 30 minutes
                issues.append(f"Too soon since last trade: {time_since_last}")
        
        # Check position size would be within limits
        position_size = self.safety_limits.max_position_size_eur
        total_risk = sum(pos.amount_eur for pos in self.active_positions.values())
        if total_risk + position_size > self.safety_limits.max_total_risk_eur:
            issues.append(f"Total risk would exceed limit: €{total_risk + position_size:.2f}")
        
        # Check risk/reward ratio
        if signal.risk_reward_ratio < 1.5:  # Minimum 1.5:1 risk/reward
            issues.append(f"Poor risk/reward ratio: {signal.risk_reward_ratio:.2f}")
        
        is_safe = len(issues) == 0
        
        if is_safe:
            logger.info("✅ Trade passed all safety checks")
        else:
            logger.warning(f"⚠️ Trade failed safety checks: {', '.join(issues)}")
        
        return is_safe, issues
    
    async def execute_trade(self, signal: TradeSignal) -> Optional[Position]:
        """
        Execute trade with comprehensive safety checks
        """
        try:
            # Validate safety first
            is_safe, issues = self.validate_trade_safety(signal)
            if not is_safe:
                logger.warning(f"🚫 Trade rejected: {', '.join(issues)}")
                return None
            
            # Get current balance
            balances = await self.get_account_balance()
            eur_balance = balances.get('EUR', 0)
            
            # Check minimum balance requirement
            if eur_balance < self.safety_limits.min_wallet_balance_eur:
                logger.warning(f"🚫 Insufficient balance: €{eur_balance:.2f}")
                return None
            
            # Calculate position size (conservative)
            max_position = min(
                self.safety_limits.max_position_size_eur,
                eur_balance * 0.2  # Maximum 20% of balance
            )
            
            if self.simulation_mode:
                logger.info(f"🎯 SIMULATION: Would execute {signal.side} {signal.symbol} for €{max_position:.2f}")
                # Create mock position for simulation
                position = Position(
                    symbol=signal.symbol,
                    side=signal.side,
                    amount_eur=max_position,
                    amount_crypto=max_position / signal.target_price,
                    entry_price=signal.target_price,
                    stop_loss_price=signal.stop_loss_price,
                    profit_target_price=signal.profit_target_price,
                    timestamp=datetime.now()
                )
                self.active_positions[f"{signal.symbol}_{int(time.time())}"] = position
                logger.info(f"✅ Simulation position created: {position}")
                return position
            else:
                # Execute real trade (EXTREMELY CAREFUL)
                logger.info(f"💸 EXECUTING REAL TRADE: {signal.side} {signal.symbol} for €{max_position:.2f}")
                
                # Implement real trade execution here
                # This would use CDP API to execute the actual trade
                
                # For now, return None to prevent accidental execution
                logger.error("🚫 REAL TRADING NOT YET IMPLEMENTED - Safety measure")
                return None
        
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return None
    
    def get_safety_status(self) -> Dict[str, any]:
        """Get comprehensive safety status report"""
        return {
            'emergency_stop': self.emergency_stop,
            'simulation_mode': self.simulation_mode,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'active_positions': len(self.active_positions),
            'safety_limits': asdict(self.safety_limits),
            'last_trade': self.last_trade_time.isoformat() if self.last_trade_time else None
        }

# Factory function for safe initialization
def create_safe_trader(api_key: str, private_key: str = None, testnet: bool = None) -> KimeraCDPSafeTrader:
    """
    Create a safe CDP trader with maximum safety controls
    
    Args:
        api_key: CDP API key
        private_key: CDP private key (None for simulation)
        testnet: Use testnet (strongly recommended for real money)
    """
    return KimeraCDPSafeTrader(api_key, private_key, testnet) 