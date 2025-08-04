#!/usr/bin/env python3
"""
KIMERA PORTFOLIO-AWARE TRADER
============================

Fixes the invalid sell amount problem by:
1. Only trading assets you actually own
2. Calculating proper sell amounts based on real balances
3. Executing profitable trades with your current holdings

Current Portfolio Analysis:
- USDT: $33.41 (tradeable)
- TRX: $43.00 (tradeable) 
- ADA: $290.59 (tradeable)
- Total: $369.51

Mission: Generate profit using your actual holdings
"""

import asyncio
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging

# Set API credentials
os.environ['BINANCE_API_KEY'] = os.getenv("BINANCE_API_KEY", "")
os.environ['BINANCE_API_SECRET'] = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - KIMERA_PORTFOLIO_AWARE - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KimeraPortfolioAwareTrader:
    """
    Portfolio-aware trader that only trades assets you actually own
    """
    
    def __init__(self):
        self.client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )
        
        # Your actual holdings (from previous analysis)
        self.actual_holdings = {
            'USDT': 33.41,
            'TRX': 143.20,  # Amount in TRX
            'ADA': 415.90,  # Amount in ADA
            'BNB': 0.00365494,  # Small amount
            'IQ': 5.10  # Small amount
        }
        
        # Tradeable pairs based on your holdings
        self.tradeable_pairs = [
            'TRXUSDT',  # You have TRX
            'ADAUSDT',  # You have ADA
            'BNBUSDT',  # You have BNB (small)
        ]
        
        self.session_active = False
        self.trades_executed = 0
        self.total_profit = 0.0
        self.start_balance = 0.0
        
        logger.info("🎯 KIMERA PORTFOLIO-AWARE TRADER INITIALIZED")
        logger.info(f"   Tradeable pairs: {self.tradeable_pairs}")
    
    async def get_actual_balances(self) -> Dict[str, float]:
        """Get your actual current balances from Binance"""
        try:
            account = self.client.get_account()
            balances = {}
            
            for balance in account['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                if free > 0:
                    balances[asset] = free
            
            logger.info("💰 ACTUAL BALANCES:")
            total_value = 0.0
            for asset, amount in balances.items():
                if asset == 'USDT':
                    value = amount
                else:
                    try:
                        ticker = self.client.get_avg_price(symbol=f"{asset}USDT")
                        price = float(ticker['price'])
                        value = amount * price
                    except Exception as e:
                        logger.error(f"Error in kimera_portfolio_aware_trader.py: {e}", exc_info=True)
                        raise  # Re-raise for proper error handling
                        value = 0.0
                
                total_value += value
                logger.info(f"   {asset}: {amount:.8f} = ${value:.2f}")
            
            logger.info(f"   TOTAL PORTFOLIO: ${total_value:.2f}")
            self.start_balance = total_value
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return {}
    
    async def analyze_profitable_opportunities(self, balances: Dict[str, float]) -> List[Dict]:
        """Analyze profitable opportunities with your actual holdings"""
        opportunities = []
        
        for pair in self.tradeable_pairs:
            base_asset = pair.replace('USDT', '')
            
            # Skip if you don't have this asset
            if base_asset not in balances or balances[base_asset] <= 0:
                continue
            
            try:
                # Get current price and market data
                ticker = self.client.get_24hr_ticker(symbol=pair)
                current_price = float(ticker['lastPrice'])
                price_change = float(ticker['priceChangePercent'])
                volume = float(ticker['volume'])
                
                # Calculate potential profit opportunities
                opportunity = await self.calculate_opportunity(
                    pair, base_asset, balances[base_asset], 
                    current_price, price_change, volume
                )
                
                if opportunity:
                    opportunities.append(opportunity)
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {pair}: {e}")
        
        # Sort by profit potential
        opportunities.sort(key=lambda x: x['profit_score'], reverse=True)
        return opportunities
    
    async def calculate_opportunity(self, pair: str, asset: str, amount: float, 
                                 price: float, price_change: float, volume: float) -> Optional[Dict]:
        """Calculate specific trading opportunity"""
        try:
            # Get orderbook for better analysis
            orderbook = self.client.get_orderbook(symbol=pair, limit=20)
            bid_price = float(orderbook['bids'][0][0])
            ask_price = float(orderbook['asks'][0][0])
            spread = (ask_price - bid_price) / bid_price * 100
            
            # Calculate values
            total_value = amount * price
            min_trade_value = 6.0  # Minimum $6 trade on Binance
            
            # Skip if position too small
            if total_value < min_trade_value:
                return None
            
            # Calculate sellable amount (leave some for fees)
            sellable_amount = amount * 0.99  # 1% buffer for fees
            sellable_value = sellable_amount * bid_price
            
            # Score the opportunity
            momentum_score = abs(price_change) / 10  # Higher volatility = higher score
            volume_score = min(volume / 1000000, 10) / 10  # Normalize volume
            spread_score = max(0, (0.5 - spread) / 0.5)  # Lower spread = higher score
            
            profit_score = (momentum_score + volume_score + spread_score) / 3
            
            # Determine action based on price movement
            if price_change > 2.0:  # Strong upward movement
                action = 'SELL'  # Take profits
                confidence = min(90, 50 + abs(price_change) * 2)
            elif price_change < -2.0:  # Strong downward movement
                action = 'HOLD'  # Wait for recovery
                confidence = 30
            else:
                action = 'HOLD'  # Neutral
                confidence = 40
            
            return {
                'pair': pair,
                'asset': asset,
                'action': action,
                'amount': sellable_amount,
                'current_price': price,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'total_value': total_value,
                'sellable_value': sellable_value,
                'price_change': price_change,
                'volume': volume,
                'spread': spread,
                'profit_score': profit_score,
                'confidence': confidence,
                'reason': f"Price {price_change:+.2f}%, Vol: {volume/1000000:.1f}M, Spread: {spread:.3f}%"
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate opportunity for {pair}: {e}")
            return None
    
    async def execute_profitable_trade(self, opportunity: Dict) -> bool:
        """Execute a profitable trade with proper amount validation"""
        try:
            pair = opportunity['pair']
            action = opportunity['action']
            amount = opportunity['amount']
            
            if action != 'SELL':
                logger.info(f"⏸️  {pair}: {action} - {opportunity['reason']}")
                return False
            
            # Validate sell amount
            if amount <= 0:
                logger.error(f"❌ Invalid sell amount: {amount}")
                return False
            
            # Check minimum notional value
            if opportunity['sellable_value'] < 6.0:
                logger.warning(f"⚠️  {pair}: Trade value ${opportunity['sellable_value']:.2f} below minimum $6")
                return False
            
            # Format amount properly
            formatted_amount = f"{amount:.8f}".rstrip('0').rstrip('.')
            
            logger.info(f"🔥 EXECUTING TRADE:")
            logger.info(f"   Pair: {pair}")
            logger.info(f"   Action: {action}")
            logger.info(f"   Amount: {formatted_amount} {opportunity['asset']}")
            logger.info(f"   Value: ${opportunity['sellable_value']:.2f}")
            logger.info(f"   Price: ${opportunity['current_price']:.6f}")
            logger.info(f"   Confidence: {opportunity['confidence']:.1f}%")
            logger.info(f"   Reason: {opportunity['reason']}")
            
            # Execute the trade
            order = self.client.order_market_sell(
                symbol=pair,
                quantity=formatted_amount
            )
            
            logger.info(f"✅ TRADE EXECUTED:")
            logger.info(f"   Order ID: {order['orderId']}")
            logger.info(f"   Status: {order['status']}")
            logger.info(f"   Executed Qty: {order['executedQty']}")
            
            # Update counters
            self.trades_executed += 1
            
            # Calculate profit (simplified)
            if 'fills' in order:
                total_received = sum(float(fill['qty']) * float(fill['price']) for fill in order['fills'])
                self.total_profit += total_received - opportunity['sellable_value']
            
            return True
            
        except BinanceAPIException as e:
            logger.error(f"❌ Binance API Error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return False
    
    async def run_profitable_session(self, duration_minutes: int = 30):
        """Run a profitable trading session"""
        try:
            logger.info(f"🚀 STARTING PROFITABLE SESSION - {duration_minutes} MINUTES")
            
            # Get actual balances
            balances = await self.get_actual_balances()
            if not balances:
                logger.error("❌ No balances found")
                return
            
            self.session_active = True
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            cycle_count = 0
            
            while self.session_active and datetime.now() < end_time:
                cycle_count += 1
                logger.info(f"🔄 CYCLE {cycle_count} - {(datetime.now() - start_time).total_seconds():.0f}s elapsed")
                
                # Analyze opportunities
                opportunities = await self.analyze_profitable_opportunities(balances)
                
                if opportunities:
                    logger.info(f"📊 Found {len(opportunities)} opportunities")
                    
                    # Execute best opportunity
                    best_opportunity = opportunities[0]
                    if best_opportunity['confidence'] > 60:  # Only high confidence trades
                        await self.execute_profitable_trade(best_opportunity)
                        
                        # Refresh balances after trade
                        balances = await self.get_actual_balances()
                else:
                    logger.info("📊 No profitable opportunities found")
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30 second cycles
            
            logger.info("🏁 PROFITABLE SESSION COMPLETED")
            await self.generate_session_report(start_time, datetime.now())
            
        except Exception as e:
            logger.error(f"❌ Session failed: {e}")
        finally:
            self.session_active = False
    
    async def generate_session_report(self, start_time: datetime, end_time: datetime):
        """Generate session report"""
        try:
            duration = (end_time - start_time).total_seconds()
            current_balance = 0.0
            
            # Get final balance
            final_balances = await self.get_actual_balances()
            for asset, amount in final_balances.items():
                if asset == 'USDT':
                    current_balance += amount
                else:
                    try:
                        ticker = self.client.get_avg_price(symbol=f"{asset}USDT")
                        price = float(ticker['price'])
                        current_balance += amount * price
                    except Exception as e:
                        logger.error(f"Error in kimera_portfolio_aware_trader.py: {e}", exc_info=True)
                        raise  # Re-raise for proper error handling
            
            profit = current_balance - self.start_balance
            profit_pct = (profit / self.start_balance * 100) if self.start_balance > 0 else 0
            
            report = {
                'session_type': 'PORTFOLIO_AWARE_TRADING',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_minutes': duration / 60,
                'start_balance': self.start_balance,
                'end_balance': current_balance,
                'profit_usd': profit,
                'profit_percentage': profit_pct,
                'trades_executed': self.trades_executed,
                'profit_per_trade': profit / self.trades_executed if self.trades_executed > 0 else 0
            }
            
            # Save report
            report_file = f'portfolio_aware_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("📋 SESSION REPORT:")
            logger.info(f"   Duration: {duration/60:.1f} minutes")
            logger.info(f"   Start Balance: ${self.start_balance:.2f}")
            logger.info(f"   End Balance: ${current_balance:.2f}")
            logger.info(f"   Profit: ${profit:.2f} ({profit_pct:+.2f}%)")
            logger.info(f"   Trades: {self.trades_executed}")
            logger.info(f"   Report: {report_file}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")

async def main():
    """Main execution"""
    try:
        logger.info("🎯 KIMERA PORTFOLIO-AWARE TRADER")
        logger.info("=" * 50)
        logger.info("✅ Fixes invalid sell amount problem")
        logger.info("💰 Only trades assets you actually own")
        logger.info("🔥 Calculates proper amounts based on real balances")
        logger.info("=" * 50)
        
        trader = KimeraPortfolioAwareTrader()
        
        # Run 30-minute session
        await trader.run_profitable_session(duration_minutes=30)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 