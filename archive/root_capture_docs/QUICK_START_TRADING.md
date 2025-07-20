# 🚀 KIMERA LIVE TRADING - QUICK START

## ⚡ Ready to Trade in 5 Minutes

### Step 1: Set Up API Keys
Create a `.env` file in the root directory:

```env
# Your Phemex API (already provided)
PHEMEX_API_KEY=0f94fb19-72d2-4223-bd33-c67e1c87fafc
PHEMEX_API_SECRET=your_secret_here

# Optional: Intelligence APIs (for better performance)
NEWS_API_KEY=your_newsapi_key
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
```

### Step 2: Start Trading
```bash
# Option 1: Phemex Live Trading (Recommended)
python examples/phemex_trading_demo.py
# Choose option 2 for live trading

# Option 2: Full Intelligence Demo
python examples/kimera_intelligent_trading_demo.py
```

### Step 3: Monitor Performance
- Watch the console for trade logs
- Check your Phemex account for actual trades
- Monitor P&L in real-time

## 🎯 Trading Parameters

### Starting Capital
- **BTC Balance**: 0.00326515 BTC
- **USD Value**: ~$342.09
- **Target**: Maximum profit generation

### Risk Settings (Kimera Decides)
- **Position Size**: Up to 50%+ of capital
- **Leverage**: Up to 5x when confident
- **Stop Loss**: Dynamic based on volatility
- **Take Profit**: Adaptive targets

### Intelligence Sources
- ✅ Market data (always active)
- ✅ Technical indicators
- ✅ Price action analysis
- ⚠️ News sentiment (if API key provided)
- ⚠️ Social media sentiment (if API keys provided)

## 🚨 Important Notes

1. **Real Money**: This is live trading with actual funds
2. **No Guarantees**: Crypto trading involves risk of loss
3. **Autonomous**: Kimera makes all decisions independently
4. **Monitor**: Keep an eye on performance and balance
5. **Stop**: Press Ctrl+C to stop trading at any time

## 📊 Expected Scenarios

### Conservative Estimate
- **Daily**: 0.1-0.5% growth
- **3 Days**: 0.3-1.5% total
- **Risk**: Low to moderate

### Optimistic Estimate  
- **Daily**: 1-5% growth
- **3 Days**: 3-15% total
- **Risk**: Moderate to high

### Best Case
- **Daily**: 5-10% growth
- **3 Days**: 15-30% total
- **Risk**: High (but managed)

## 🔧 Troubleshooting

### Common Issues
1. **API Key Error**: Check your `.env` file
2. **Connection Error**: Verify internet connection
3. **Insufficient Balance**: Ensure you have BTC in your account
4. **Rate Limit**: Phemex may limit API calls

### Getting Help
- Check the logs for error messages
- Verify your Phemex API permissions
- Ensure testnet vs mainnet settings are correct

## 🎮 Demo Mode First?

If you want to test without real money:
```bash
# Run simulation first
python examples/kimera_3day_mission.py
```

## 🏆 Ready to Generate Profit!

Your Kimera trading system is fully operational and ready to generate profit for its own development. The system combines:

- **Advanced AI**: Cognitive field dynamics
- **Market Intelligence**: Multi-source data analysis  
- **Autonomous Trading**: No human intervention needed
- **Risk Management**: Dynamic protection systems

**Start trading and let Kimera prove its worth!** 🚀 