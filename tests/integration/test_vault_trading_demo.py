#!/usr/bin/env python3
"""
KIMERA VAULT-INTEGRATED TRADING DEMONSTRATION
============================================
🚀 Demonstrating the complete vault-integrated cognitive trading system
"""

import asyncio
import json
import time
from datetime import datetime
from src.vault.database import initialize_database

# Initialize database first
print("🔧 Initializing Kimera Database...")
db_success = initialize_database()
print(f"✅ Database Status: {'Connected' if db_success else 'Failed'}")

# Import the trading system
from kimera_cognitive_trading_intelligence_vault_integrated import KimeraCognitiveTrading

async def demonstrate_vault_trading():
    """Demonstrate the vault-integrated trading system"""
    print("\n" + "="*80)
    print("🧠 KIMERA VAULT-INTEGRATED COGNITIVE TRADING DEMONSTRATION")
    print("="*80)
    
    # Initialize the system
    print("\n🚀 Initializing Kimera Cognitive Trading System...")
    trader = KimeraCognitiveTrading()
    
    # Run a quick demo session
    print("\n🎯 Starting 2-minute demonstration session...")
    session = await trader.run_vault_cognitive_trading_session(duration_minutes=2)
    
    # Display results
    print("\n" + "="*80)
    print("📊 DEMONSTRATION RESULTS")
    print("="*80)
    print(f"🆔 Session ID: {session.session_id}")
    print(f"⏱️ Duration: {(session.end_time - session.start_time).total_seconds():.1f} seconds")
    print(f"📈 Total Trades: {session.total_trades}")
    print(f"✅ Successful Trades: {session.successful_trades}")
    print(f"💰 Total PnL: {session.total_pnl:.4f}")
    print(f"🧠 Vault Insights: {session.vault_insights_generated}")
    print(f"🔥 SCARs Created: {session.scars_created}")
    print(f"🔄 Cognitive Evolutions: {session.cognitive_evolutions}")
    print(f"🔍 Vault Queries: {session.vault_queries}")
    
    # Save demonstration results
    demo_file = f"vault_trading_demo_{session.session_id}.json"
    with open(demo_file, 'w') as f:
        json.dump(session.to_dict(), f, indent=2)
    
    print(f"\n💾 Results saved to: {demo_file}")
    print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return session

if __name__ == "__main__":
    print("🌟 Starting Kimera Vault-Integrated Trading Demonstration...")
    session = asyncio.run(demonstrate_vault_trading())
    print(f"\n🏁 Demo completed! Check file: vault_trading_demo_{session.session_id}.json") 