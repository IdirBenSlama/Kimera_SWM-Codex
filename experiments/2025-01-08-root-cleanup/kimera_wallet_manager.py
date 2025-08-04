#!/usr/bin/env python3
"""
KIMERA COMPREHENSIVE WALLET MANAGEMENT SYSTEM
=============================================

This system manages your ENTIRE wallet portfolio, making intelligent decisions
about ALL assets, not just USDT. It can convert between assets, rebalance,
and execute trades using whatever is available.
"""

import os
import asyncio
import ccxt
import sys
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add backend to path for Kimera integration
sys.path.append('backend')

# Load environment variables
load_dotenv()

@dataclass
class AssetInfo:
    """Information about a single asset"""
    symbol: str
    balance: float
    usd_value: float
    percentage: float
    price_usd: float

@dataclass
class TradingOpportunity:
    """A trading opportunity identified by Kimera"""
    from_asset: str
    to_asset: str
    amount: float
    confidence: float
    strategy: str
    expected_profit: float

class KimeraWalletManager:
    """Comprehensive wallet management system"""
    
    def __init__(self):
        # Get API credentials
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Missing Binance API credentials")
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Portfolio data
        self.portfolio: Dict[str, AssetInfo] = {}
        self.total_portfolio_value = 0.0
        self.tradeable_pairs = []
        
        logger.info("🚀 Kimera Wallet Manager initialized")
    
    async def analyze_complete_portfolio(self) -> Dict[str, AssetInfo]:
        """Analyze the complete wallet portfolio"""
        logger.info("\n📊 ANALYZING COMPLETE PORTFOLIO...")
        logger.info("=" * 50)
        
        try:
            # Get all balances
            balance_data = self.exchange.fetch_balance()
            
            # Get all tickers for price data
            all_tickers = self.exchange.fetch_tickers()
            
            portfolio = {}
            total_value = 0.0
            
            # Analyze each asset with non-zero balance
            for asset, balance_info in balance_data.items():
                if asset in ['free', 'used', 'total', 'info']:
                    continue
                
                # Handle different balance data formats
                if isinstance(balance_info, dict):
                    free_balance = float(balance_info.get('free', 0))
                else:
                    try:
                        free_balance = float(balance_info) if float(balance_info) > 0 else 0
                    except (ValueError, TypeError):
                        free_balance = 0
                
                if free_balance > 0:
                    # Get USD value
                    usd_value = 0.0
                    price_usd = 0.0
                    
                    if asset == 'USDT':
                        usd_value = free_balance
                        price_usd = 1.0
                    else:
                        # Try different ticker combinations to get USD price
                        possible_pairs = [f"{asset}/USDT", f"{asset}/BUSD", f"{asset}/USD"]
                        
                        for pair in possible_pairs:
                            if pair in all_tickers:
                                price_usd = all_tickers[pair]['last']
                                usd_value = free_balance * price_usd
                                break
                        
                        # If no direct USD pair, try via BTC
                        if usd_value == 0 and f"{asset}/BTC" in all_tickers and "BTC/USDT" in all_tickers:
                            btc_price = all_tickers[f"{asset}/BTC"]['last']
                            btc_usd = all_tickers["BTC/USDT"]['last']
                            price_usd = btc_price * btc_usd
                            usd_value = free_balance * price_usd
                    
                    if usd_value > 0.01:  # Only include assets worth more than 1 cent
                        portfolio[asset] = AssetInfo(
                            symbol=asset,
                            balance=free_balance,
                            usd_value=usd_value,
                            percentage=0.0,  # Will calculate after total
                            price_usd=price_usd
                        )
                        total_value += usd_value
            
            # Calculate percentages
            for asset_info in portfolio.values():
                asset_info.percentage = (asset_info.usd_value / total_value) * 100
            
            # Sort by value
            portfolio = dict(sorted(portfolio.items(), key=lambda x: x[1].usd_value, reverse=True))
            
            # Display portfolio
            logger.info(f"💰 TOTAL PORTFOLIO VALUE: ${total_value:.2f}")
            logger.info("\n📈 ASSET BREAKDOWN:")
            logger.info("-" * 60)
            
            for asset, info in portfolio.items():
                logger.info(f"{asset:8} | {info.balance:12.6f} | ${info.usd_value:8.2f} | {info.percentage:5.1f}%")
            
            self.portfolio = portfolio
            self.total_portfolio_value = total_value
            
            return portfolio
            
        except Exception as e:
            logger.info(f"❌ Error analyzing portfolio: {e}")
            return {}
    
    def identify_trading_opportunities(self) -> List[TradingOpportunity]:
        """Identify trading opportunities using Kimera's intelligence"""
        logger.info("\n🧠 KIMERA INTELLIGENCE: Identifying Trading Opportunities...")
        logger.info("=" * 60)
        
        opportunities = []
        
        # Strategy 1: Concentration Risk Management
        # If any asset is >70% of portfolio, suggest diversification
        for asset, info in self.portfolio.items():
            if info.percentage > 70:
                opportunities.append(TradingOpportunity(
                    from_asset=asset,
                    to_asset="BTC",  # Diversify to BTC
                    amount=info.balance * 0.3,  # Trade 30%
                    confidence=0.8,
                    strategy="CONCENTRATION_RISK_REDUCTION",
                    expected_profit=0.05
                ))
        
        # Strategy 2: Small Balance Consolidation
        # Consolidate small balances (<5% of portfolio) into major assets
        small_assets = [asset for asset, info in self.portfolio.items() 
                       if info.percentage < 5 and asset not in ['USDT', 'BTC', 'ETH']]
        
        if len(small_assets) > 2:
            for asset in small_assets[:2]:  # Consolidate up to 2 small assets
                info = self.portfolio[asset]
                opportunities.append(TradingOpportunity(
                    from_asset=asset,
                    to_asset="USDT",  # Convert to stable coin
                    amount=info.balance,
                    confidence=0.7,
                    strategy="SMALL_BALANCE_CONSOLIDATION",
                    expected_profit=0.02
                ))
        
        # Strategy 3: Market Momentum Trading
        # Use TRX (your largest holding) for momentum trades
        if 'TRX' in self.portfolio:
            trx_info = self.portfolio['TRX']
            if trx_info.percentage > 50:  # If TRX is majority holding
                opportunities.append(TradingOpportunity(
                    from_asset="TRX",
                    to_asset="BTC",
                    amount=trx_info.balance * 0.1,  # Trade 10% for momentum
                    confidence=0.75,
                    strategy="MOMENTUM_DIVERSIFICATION",
                    expected_profit=0.08
                ))
        
        # Strategy 4: Stable Coin Opportunity
        # If we have very little USDT, convert some assets for trading flexibility
        usdt_percentage = self.portfolio.get('USDT', AssetInfo('USDT', 0, 0, 0, 1)).percentage
        if usdt_percentage < 10:  # Less than 10% in stable coins
            # Find best asset to partially convert
            best_asset = max([asset for asset in self.portfolio.keys() if asset != 'USDT'], 
                           key=lambda x: self.portfolio[x].usd_value, default=None)
            
            if best_asset:
                info = self.portfolio[best_asset]
                opportunities.append(TradingOpportunity(
                    from_asset=best_asset,
                    to_asset="USDT",
                    amount=min(info.balance * 0.15, info.usd_value * 0.15 / info.price_usd),
                    confidence=0.85,
                    strategy="STABLE_COIN_ALLOCATION",
                    expected_profit=0.03
                ))
        
        # Display opportunities
        if opportunities:
            logger.info(f"🎯 IDENTIFIED {len(opportunities)} TRADING OPPORTUNITIES:")
            logger.info("-" * 60)
            
            for i, opp in enumerate(opportunities, 1):
                logger.info(f"{i}. {opp.strategy}")
                logger.info(f"   Trade: {opp.amount:.6f} {opp.from_asset} → {opp.to_asset}")
                logger.info(f"   Confidence: {opp.confidence:.1%}")
                logger.info(f"   Expected Profit: {opp.expected_profit:.1%}")
                logger.info()
        else:
            logger.info("📊 Portfolio is well-balanced. No immediate opportunities identified.")
        
        return opportunities
    
    async def execute_opportunity(self, opportunity: TradingOpportunity) -> bool:
        """Execute a trading opportunity"""
        logger.info(f"\n🚀 EXECUTING: {opportunity.strategy}")
        logger.info(f"Trading {opportunity.amount:.6f} {opportunity.from_asset} → {opportunity.to_asset}")
        
        try:
            # Determine trading pair
            symbol = f"{opportunity.from_asset}/{opportunity.to_asset}"
            
            # Check if direct pair exists, otherwise use intermediate
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                trading_symbol = symbol
            except Exception as e:
                logger.error(f"Error in kimera_wallet_manager.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                # Use USDT as intermediate
                if opportunity.to_asset != 'USDT':
                    logger.info(f"   Using USDT as intermediate currency")
                    # First convert from_asset to USDT, then USDT to to_asset
                    symbol1 = f"{opportunity.from_asset}/USDT"
                    symbol2 = f"{opportunity.to_asset}/USDT"
                    
                    # Execute first trade
                    order1 = self.exchange.create_market_sell_order(symbol1, opportunity.amount)
                    logger.info(f"   ✅ Step 1: Sold {opportunity.amount:.6f} {opportunity.from_asset}")
                    
                    # Calculate amount for second trade
                    usdt_received = order1['cost']
                    ticker2 = self.exchange.fetch_ticker(symbol2)
                    to_amount = usdt_received / ticker2['last']
                    
                    # Execute second trade
                    order2 = self.exchange.create_market_buy_order(symbol2, to_amount)
                    logger.info(f"   ✅ Step 2: Bought {to_amount:.6f} {opportunity.to_asset}")
                    
                    return True
                else:
                    trading_symbol = f"{opportunity.from_asset}/USDT"
            
            # Execute direct trade
            if opportunity.to_asset == 'USDT':
                order = self.exchange.create_market_sell_order(trading_symbol, opportunity.amount)
                logger.info(f"   ✅ Sold {opportunity.amount:.6f} {opportunity.from_asset} for USDT")
            else:
                # Calculate USDT needed
                from_ticker = self.exchange.fetch_ticker(f"{opportunity.from_asset}/USDT")
                usdt_value = opportunity.amount * from_ticker['last']
                
                to_ticker = self.exchange.fetch_ticker(f"{opportunity.to_asset}/USDT")
                to_amount = usdt_value / to_ticker['last']
                
                order = self.exchange.create_market_buy_order(f"{opportunity.to_asset}/USDT", to_amount)
                logger.info(f"   ✅ Bought {to_amount:.6f} {opportunity.to_asset}")
            
            logger.info(f"   📋 Order ID: {order['id']}")
            logger.info(f"   💰 Value: ${order.get('cost', 0):.2f}")
            
            return True
            
        except Exception as e:
            logger.info(f"   ❌ Execution failed: {e}")
            return False
    
    async def run_intelligent_management(self):
        """Run the complete intelligent wallet management"""
        logger.info("🧠 KIMERA INTELLIGENT WALLET MANAGEMENT")
        logger.info("=" * 60)
        logger.info(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Analyze complete portfolio
        portfolio = await self.analyze_complete_portfolio()
        
        if not portfolio:
            logger.info("❌ Could not analyze portfolio")
            return
        
        # Step 2: Identify opportunities
        opportunities = self.identify_trading_opportunities()
        
        if not opportunities:
            logger.info("✅ Portfolio is optimally managed. No actions needed.")
            return
        
        # Step 3: Ask for permission to execute
        logger.info("\n" + "!" * 60)
        logger.info("🤖 KIMERA WANTS TO OPTIMIZE YOUR PORTFOLIO")
        logger.info("!" * 60)
        
        for i, opp in enumerate(opportunities, 1):
            logger.info(f"{i}. {opp.strategy}: Trade {opp.amount:.6f} {opp.from_asset} → {opp.to_asset}")
        
        response = input("\nExecute Kimera's recommendations? (yes/no): ")
        
        if response.lower() != 'yes':
            logger.info("🛑 Portfolio optimization cancelled by user")
            return
        
        # Step 4: Execute opportunities
        logger.info("\n🚀 EXECUTING KIMERA'S PORTFOLIO OPTIMIZATION...")
        
        executed = 0
        for opp in opportunities:
            if await self.execute_opportunity(opp):
                executed += 1
                await asyncio.sleep(1)  # Rate limiting
        
        # Step 5: Re-analyze portfolio
        logger.info(f"\n✅ EXECUTED {executed}/{len(opportunities)} OPTIMIZATIONS")
        logger.info("\n📊 UPDATED PORTFOLIO:")
        await self.analyze_complete_portfolio()
        
        logger.info("\n🎯 KIMERA PORTFOLIO OPTIMIZATION COMPLETE!")
        logger.info(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

async def main():
    try:
        kimera = KimeraWalletManager()
        await kimera.run_intelligent_management()
    except Exception as e:
        logger.info(f"❌ CRITICAL ERROR: {e}")
        import traceback
import logging
logger = logging.getLogger(__name__)
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 