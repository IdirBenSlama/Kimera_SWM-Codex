#!/usr/bin/env python3
"""
KIMERA TRX AGGRESSIVE TRADER
Works with existing TRX balance - converts to USDT for aggressive trading
"""

import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import os

# Set credentials directly
os.environ['BINANCE_API_KEY'] = os.getenv("BINANCE_API_KEY", "")
os.environ['BINANCE_API_SECRET'] = 'qUn5JqSpYz1GDxFj2X3UF23TYgtxKrTsCbDZEoBMYCPbYZgP4siVLyspkB5HAPl7'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - TRX-TRADER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kimera_trx_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KimeraTRXTrader:
    """KIMERA TRX AGGRESSIVE TRADER"""
    
    def __init__(self):
        self.client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )
        
        self.session_duration = 600  # 10 minutes
        self.trading_session_start = None
        self.executed_trades = []
        self.initial_trx_balance = 0
        self.initial_usdt_balance = 0
        
        # AGGRESSIVE TRADING SETTINGS
        self.trade_amount_usdt = 50.0  # $50 per trade
        self.execution_interval = 3  # 3 seconds between trades
        
        # High-volume pairs for aggressive trading
        self.aggressive_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
            'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'MATICUSDT'
        ]
        
        logger.info("KIMERA TRX AGGRESSIVE TRADER INITIALIZED")
        logger.info("CONVERTING TRX TO USDT FOR AGGRESSIVE TRADING")
        logger.info(f"TARGET TRADE SIZE: ${self.trade_amount_usdt} per trade")
    
    def get_balances(self) -> Dict[str, float]:
        """Get current balances"""
        try:
            account = self.client.get_account()
            balances = {}
            
            for balance in account['balances']:
                free = float(balance['free'])
                if free > 0:
                    balances[balance['asset']] = free
            
            return balances
        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return {}
    
    async def convert_trx_to_usdt(self, trx_amount: float) -> Dict:
        """Convert TRX to USDT"""
        try:
            logger.info(f"Converting {trx_amount:.2f} TRX to USDT...")
            
            # Get TRX price
            ticker = self.client.get_symbol_ticker(symbol='TRXUSDT')
            trx_price = float(ticker['price'])
            expected_usdt = trx_amount * trx_price
            
            logger.info(f"TRX Price: ${trx_price:.6f} - Expected USDT: ${expected_usdt:.2f}")
            
            # Execute TRX to USDT conversion
            order = self.client.order_market_sell(
                symbol='TRXUSDT',
                quantity=trx_amount
            )
            
            conversion_result = {
                'status': 'EXECUTED',
                'type': 'TRX_TO_USDT_CONVERSION',
                'trx_amount': trx_amount,
                'trx_price': trx_price,
                'expected_usdt': expected_usdt,
                'order': order,
                'timestamp': datetime.now().isoformat()
            }
            
            self.executed_trades.append(conversion_result)
            logger.info(f"TRX CONVERSION EXECUTED: {trx_amount:.2f} TRX -> ~${expected_usdt:.2f} USDT")
            logger.info(f"Order ID: {order['orderId']}")
            
            return conversion_result
            
        except BinanceAPIException as e:
            logger.error(f"TRX conversion failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    async def aggressive_buy_trade(self, symbol: str, usdt_amount: float) -> Dict:
        """Execute aggressive buy trade"""
        try:
            order = self.client.order_market_buy(
                symbol=symbol,
                quoteOrderQty=usdt_amount
            )
            
            trade_result = {
                'status': 'EXECUTED',
                'type': 'AGGRESSIVE_BUY',
                'symbol': symbol,
                'usdt_amount': usdt_amount,
                'order': order,
                'timestamp': datetime.now().isoformat()
            }
            
            self.executed_trades.append(trade_result)
            logger.info(f"AGGRESSIVE BUY: {symbol} - ${usdt_amount:.2f}")
            
            return trade_result
            
        except BinanceAPIException as e:
            logger.error(f"Aggressive buy failed for {symbol}: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    async def aggressive_sell_trade(self, symbol: str) -> Dict:
        """Execute aggressive sell trade"""
        try:
            base_asset = symbol.replace('USDT', '')
            balances = self.get_balances()
            asset_balance = balances.get(base_asset, 0)
            
            if asset_balance <= 0:
                return {'status': 'NO_BALANCE', 'symbol': symbol}
            
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=asset_balance
            )
            
            trade_result = {
                'status': 'EXECUTED',
                'type': 'AGGRESSIVE_SELL',
                'symbol': symbol,
                'quantity': asset_balance,
                'order': order,
                'timestamp': datetime.now().isoformat()
            }
            
            self.executed_trades.append(trade_result)
            logger.info(f"AGGRESSIVE SELL: {symbol} - {asset_balance:.6f} units")
            
            return trade_result
            
        except BinanceAPIException as e:
            logger.error(f"Aggressive sell failed for {symbol}: {e}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def get_best_trading_opportunity(self) -> str:
        """Get best trading opportunity based on 1-minute price movement"""
        try:
            best_symbol = 'BTCUSDT'  # Default
            best_score = 0
            
            for symbol in self.aggressive_pairs:
                try:
                    # Get recent price data
                    klines = self.client.get_klines(symbol=symbol, interval='1m', limit=5)
                    closes = [float(k[4]) for k in klines]
                    
                    # Calculate price momentum
                    price_change = (closes[-1] - closes[0]) / closes[0] * 100
                    volatility = abs(price_change)
                    
                    # Score based on volatility (higher volatility = better opportunity)
                    score = volatility * 10
                    
                    if score > best_score:
                        best_score = score
                        best_symbol = symbol
                        
                except Exception:
                    continue
            
            logger.info(f"Best opportunity: {best_symbol} (Score: {best_score:.2f})")
            return best_symbol
            
        except Exception as e:
            logger.warning(f"Failed to get best opportunity: {e}")
            return 'BTCUSDT'  # Fallback
    
    async def rapid_scalping_session(self):
        """Execute rapid scalping with converted USDT"""
        scalp_count = 0
        
        while time.time() - self.trading_session_start < self.session_duration:
            elapsed = time.time() - self.trading_session_start
            remaining = self.session_duration - elapsed
            
            logger.info(f"RAPID SCALP #{scalp_count + 1} - Remaining: {remaining:.0f}s")
            
            # Get current USDT balance
            balances = self.get_balances()
            usdt_balance = balances.get('USDT', 0)
            
            if usdt_balance < self.trade_amount_usdt:
                logger.warning(f"Insufficient USDT for trade: ${usdt_balance:.2f}")
                break
            
            # Get best trading opportunity
            symbol = self.get_best_trading_opportunity()
            
            # 1. AGGRESSIVE BUY
            buy_result = await self.aggressive_buy_trade(symbol, self.trade_amount_usdt)
            
            if buy_result['status'] == 'EXECUTED':
                logger.info("SCALP BUY EXECUTED - Waiting 2 seconds for sell...")
                await asyncio.sleep(2)  # Wait 2 seconds
                
                # 2. AGGRESSIVE SELL
                sell_result = await self.aggressive_sell_trade(symbol)
                
                if sell_result['status'] == 'EXECUTED':
                    logger.info("SCALP SELL EXECUTED - Complete scalp cycle")
                    scalp_count += 1
                else:
                    logger.warning("SCALP SELL FAILED")
            else:
                logger.warning("SCALP BUY FAILED")
            
            # Wait before next scalp
            await asyncio.sleep(self.execution_interval)
        
        logger.info(f"RAPID SCALPING COMPLETED: {scalp_count} complete cycles")
    
    async def aggressive_trading_session(self):
        """Main aggressive trading session"""
        self.trading_session_start = time.time()
        logger.info("KIMERA TRX AGGRESSIVE TRADING SESSION STARTED")
        
        try:
            # Get initial balances
            initial_balances = self.get_balances()
            self.initial_trx_balance = initial_balances.get('TRX', 0)
            self.initial_usdt_balance = initial_balances.get('USDT', 0)
            
            logger.info(f"Initial TRX Balance: {self.initial_trx_balance:.2f}")
            logger.info(f"Initial USDT Balance: ${self.initial_usdt_balance:.2f}")
            
            # STEP 1: Convert portion of TRX to USDT for trading
            if self.initial_trx_balance > 100:  # Keep some TRX, convert some for trading
                trx_to_convert = min(500, self.initial_trx_balance * 0.5)  # Convert up to 500 TRX or 50%
                
                logger.info(f"Converting {trx_to_convert:.2f} TRX to USDT for aggressive trading...")
                conversion_result = await self.convert_trx_to_usdt(trx_to_convert)
                
                if conversion_result['status'] == 'EXECUTED':
                    # Wait a moment for balance to update
                    await asyncio.sleep(2)
                    
                    # STEP 2: Execute rapid scalping with converted USDT
                    logger.info("Starting rapid scalping with converted USDT...")
                    await self.rapid_scalping_session()
                else:
                    logger.error("TRX conversion failed - cannot proceed with aggressive trading")
            else:
                logger.error(f"Insufficient TRX balance for trading: {self.initial_trx_balance:.2f}")
            
            # Final summary
            await self.finalize_session()
            
        except Exception as e:
            logger.error(f"Aggressive trading session error: {e}")
            await self.emergency_shutdown()
    
    async def finalize_session(self):
        """Finalize trading session"""
        logger.info("KIMERA TRX AGGRESSIVE SESSION COMPLETED")
        
        # Get final balances
        final_balances = self.get_balances()
        final_trx = final_balances.get('TRX', 0)
        final_usdt = final_balances.get('USDT', 0)
        
        # Calculate performance
        session_summary = {
            'session_type': 'TRX_AGGRESSIVE',
            'session_duration': time.time() - self.trading_session_start,
            'initial_trx_balance': self.initial_trx_balance,
            'initial_usdt_balance': self.initial_usdt_balance,
            'final_trx_balance': final_trx,
            'final_usdt_balance': final_usdt,
            'total_trades': len(self.executed_trades),
            'executed_trades': self.executed_trades
        }
        
        # Save results
        filename = f"trx_aggressive_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_summary, f, indent=2, default=str)
        
        logger.info(f"TRX AGGRESSIVE TRADING SUMMARY:")
        logger.info(f"  Total Trades: {len(self.executed_trades)}")
        logger.info(f"  Initial TRX: {self.initial_trx_balance:.2f}")
        logger.info(f"  Final TRX: {final_trx:.2f}")
        logger.info(f"  Final USDT: ${final_usdt:.2f}")
        logger.info(f"  Results saved to: {filename}")
    
    async def emergency_shutdown(self):
        """Emergency shutdown"""
        logger.warning("EMERGENCY SHUTDOWN - TRX TRADER")
        
        try:
            open_orders = self.client.get_open_orders()
            for order in open_orders:
                self.client.cancel_order(symbol=order['symbol'], orderId=order['orderId'])
                logger.info(f"Cancelled order: {order['symbol']} - {order['orderId']}")
            
            logger.info("Emergency shutdown completed")
            
        except Exception as e:
            logger.error(f"Emergency shutdown error: {e}")

async def main():
    """Launch Kimera TRX Aggressive Trader"""
    print("KIMERA TRX AGGRESSIVE TRADER")
    print("=" * 50)
    print("CONVERTS TRX TO USDT FOR AGGRESSIVE TRADING")
    print("RAPID SCALPING WITH $50 TRADES")
    print("2-3 SECOND INTERVALS")
    print("MAXIMUM PROFIT TARGETING")
    print("=" * 50)
    
    confirmation = input("\nThis will convert TRX to USDT and execute aggressive trades.\nType 'ATTACK' to proceed: ")
    
    if confirmation == 'ATTACK':
        print("\nLAUNCHING TRX AGGRESSIVE TRADER...")
        
        trader = KimeraTRXTrader()
        await trader.aggressive_trading_session()
        
        print("\nTRX AGGRESSIVE TRADING SESSION COMPLETED")
    else:
        print("TRX trading aborted")

if __name__ == "__main__":
    asyncio.run(main()) 