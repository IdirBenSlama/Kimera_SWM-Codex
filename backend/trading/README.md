# Kimera Trading System

## Overview

The Kimera Trading System is a sophisticated cryptocurrency trading platform that leverages cognitive field dynamics, contradiction detection, and semantic thermodynamics to make intelligent trading decisions in real-time. It supports multiple exchanges and can identify cross-exchange arbitrage opportunities.

## Features

### 🧠 Cognitive Market Analysis
- **Cognitive Field Dynamics**: Analyzes market conditions through Kimera's unique cognitive lens
- **Contradiction Detection**: Identifies market inefficiencies and potential opportunities
- **Semantic Temperature**: Measures market "heat" levels to detect overheated or undervalued conditions
- **Insight Generation**: Produces actionable trading signals from deep cognitive analysis

### 💱 Multi-Exchange Support
- **Binance**: Spot trading with full WebSocket support
- **Phemex**: Perpetual contracts and spot trading
- **Cross-Exchange Arbitrage**: Identifies and can execute price discrepancies
- **Unified Portfolio**: Manages positions across multiple exchanges

### 📊 Advanced Risk Management
- **Position Sizing**: Uses modified Kelly Criterion with cognitive adjustments
- **Stop Loss/Take Profit**: Automatic risk management based on volatility
- **Daily Loss Limits**: Circuit breakers to prevent excessive losses
- **Margin Management**: Tracks and limits exposure across exchanges

### 📈 Performance Tracking
- **Cognitive Accuracy**: Measures how well cognitive analysis predicts outcomes
- **Risk-Adjusted Returns**: Calculates Sharpe ratio, Sortino ratio, and maximum drawdown
- **Decision Attribution**: Tracks which cognitive factors drive profitable trades
- **Real-Time Metrics**: Live P&L, win rate, and performance analytics

## Architecture

```
backend/trading/
├── core/
│   ├── trading_engine.py          # Main Kimera trading logic
│   ├── trading_orchestrator.py    # Single exchange orchestration
│   └── multi_exchange_orchestrator.py  # Multi-exchange coordination
├── api/
│   ├── binance_connector.py      # Binance API integration
│   └── phemex_connector.py       # Phemex API integration
├── strategies/
│   └── strategy_manager.py       # Strategy management (extensible)
├── monitoring/
│   └── performance_tracker.py    # Performance analytics
└── indicators/                   # Technical indicators (future)
```

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your API credentials:

```env
# Binance (use testnet for safety)
BINANCE_API_KEY=your_binance_testnet_key
BINANCE_API_SECRET=your_binance_testnet_secret

# Phemex (use testnet for safety)
PHEMEX_API_KEY=your_phemex_testnet_key
PHEMEX_API_SECRET=your_phemex_testnet_secret

# Trading settings
TRADING_MODE=testnet
INITIAL_BALANCE=10000
MAX_POSITION_SIZE=0.1
```

### 3. Run Demos

#### Single Exchange Demo (Binance)
```bash
python examples/crypto_trading_demo.py
```

#### Phemex Demo
```bash
python examples/phemex_trading_demo.py
```

#### Multi-Exchange Demo
```python
from backend.trading.core.multi_exchange_orchestrator import MultiExchangeOrchestrator

config = {
    "binance_enabled": True,
    "binance_api_key": "your_key",
    "binance_api_secret": "your_secret",
    "phemex_enabled": True,
    "phemex_api_key": "your_key",
    "phemex_api_secret": "your_secret",
    "testnet": True,
    "arbitrage_threshold": 0.001  # 0.1%
}

orchestrator = MultiExchangeOrchestrator(config)
await orchestrator.start_trading()
```

## Trading Logic

### Market Analysis Pipeline

1. **Data Collection**: Real-time price, volume, order book data
2. **Cognitive Field Generation**: Creates multi-dimensional representation
3. **Contradiction Detection**: Finds market inefficiencies
4. **Temperature Calculation**: Measures market heat/momentum
5. **Insight Generation**: Produces trading signals

### Decision Making Process

```python
# Cognitive factors influence decisions:
if cognitive_pressure > 0.7:
    # High uncertainty - reduce position size
elif contradiction_level > 0.6:
    # Market inefficiency - potential opportunity
elif semantic_temperature > 0.8:
    # Overheated market - potential reversal
elif semantic_temperature < 0.2:
    # Cold market - accumulation phase
```

### Position Sizing

Uses a modified Kelly Criterion adjusted by cognitive confidence:

```python
position_size = base_risk × confidence × (1 / (1 + volatility))
```

## API Integration

### Binance Connector

- REST API for orders and account data
- WebSocket for real-time market data
- Supports spot trading
- Testnet available

### Phemex Connector

- REST API with HMAC authentication
- WebSocket for streaming data
- Perpetual contracts and spot
- Scaled value handling

## Safety Features

1. **Testnet Mode**: Always defaults to testnet for safety
2. **Risk Limits**: Hard limits on position sizes and daily losses
3. **Circuit Breakers**: Automatic trading pause on excessive losses
4. **Error Handling**: Comprehensive error catching and logging
5. **No Hardcoded Keys**: All credentials from environment variables

## Performance Metrics

The system tracks:
- **Total P&L**: Cumulative profit/loss
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Cognitive Alignment**: How well cognitive analysis predicts outcomes

## Testing

Run the test suite:

```bash
python -m pytest tests/test_trading_system.py -v
```

## Security Best Practices

1. **Never use real API keys in testnet mode**
2. **Store credentials in environment variables**
3. **Use read-only keys when possible**
4. **Enable IP whitelisting on exchange**
5. **Set API key permissions to minimum required**
6. **Never commit `.env` files**

## Future Enhancements

- [ ] Machine learning integration for pattern recognition
- [ ] Options trading support
- [ ] Market making strategies
- [ ] Social sentiment analysis
- [ ] Advanced technical indicators
- [ ] Backtesting framework
- [ ] Portfolio optimization
- [ ] Mobile app integration

## Disclaimer

⚠️ **IMPORTANT WARNING** ⚠️

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Never trade with money you cannot afford to lose.

The developers assume no liability for financial losses incurred through use of this software. Always conduct your own research and consider consulting with financial advisors.

## Support

For issues or questions:
1. Check the [User Guide](../../docs/02_User_Guides/crypto_trading_guide.md)
2. Review example scripts in `examples/`
3. Open an issue on GitHub

---

*Remember: The market is a complex adaptive system. Kimera helps navigate it but cannot eliminate risk.* 