# KIMERA Premium Data Integration Guide

## Overview

KIMERA's premium trading system integrates four powerful financial data APIs to provide enterprise-level market intelligence and trading capabilities. This guide covers the complete premium integration including setup, features, and usage.

## Premium Data Sources

### 1. Alpha Vantage
- **API Key**: `UOHPTDWORX3R3Q2C`
- **Features**: Real-time market data, economic indicators, company fundamentals
- **Rate Limit**: 5 calls per minute
- **Endpoint**: `https://www.alphavantage.co/query`

**Capabilities:**
- Intraday stock data (1min, 5min, 15min, 30min, 60min)
- Economic indicators (GDP, CPI, Unemployment, Federal Funds Rate)
- Company overview and fundamentals
- Technical indicators

### 2. Finnhub
- **API Key**: `d1b0ouhr01qjhvtrrrd0d1b0ouhr01qjhvtrrrdg`
- **Webhook Secret**: `d1b0p59r01qjhvtrrsm0`
- **Features**: Real-time quotes, news, sentiment analysis, earnings calendar
- **Rate Limit**: 60 calls per minute
- **Endpoint**: `https://finnhub.io/api/v1`

**Capabilities:**
- Real-time stock quotes
- Market news with sentiment analysis
- Company-specific news
- Earnings calendar
- News sentiment scoring
- Webhook integration for real-time events

### 3. Twelve Data
- **API Key**: `f4923d58d5b34058ba8e5a73c0e1f910`
- **Features**: Technical indicators, time series data, forex pairs
- **Rate Limit**: 8 calls per minute
- **Endpoint**: `https://api.twelvedata.com`

**Capabilities:**
- Real-time price data
- Time series data with multiple intervals
- Technical indicators (RSI, MACD, Bollinger Bands, SMA, EMA)
- Forex pairs data
- Cryptocurrency data

### 4. CryptoPanic (Existing)
- **API Key**: `23675a49e161477a7b2b3c8c4a25743ba6777e8e`
- **Features**: Crypto news, sentiment analysis, PanicScore™
- **Rate Limit**: 1,000 requests per day
- **Endpoint**: `https://cryptopanic.com/api/developer/v2/posts/`

## Premium Features

### Enhanced Signal Generation

The premium system generates multiple types of trading signals:

1. **Premium Sentiment Signals** (Weight: 3.0)
   - Cross-source sentiment analysis
   - Economic indicator integration
   - News sentiment scoring

2. **Premium Contradiction Signals** (Weight: 2.5)
   - Detects opposing sentiment across data sources
   - Identifies market inefficiencies
   - Semantic contradiction analysis

3. **Technical Analysis Signals** (Weight: 2.0)
   - RSI overbought/oversold conditions
   - MACD crossovers
   - Bollinger Bands breakouts

4. **Enhanced Momentum Signals** (Weight: 1.8)
   - Volume-confirmed price movements
   - Multi-timeframe momentum analysis
   - Volatility-adjusted positioning

5. **Earnings Event Signals** (Weight: 1.5)
   - Upcoming earnings calendar
   - Pre-earnings volatility plays
   - Event-driven trading

6. **Economic Indicator Signals** (Weight: 1.3)
   - GDP, CPI, unemployment data
   - Federal Reserve policy changes
   - Economic surprise index

### Advanced Risk Management

Premium risk management includes:

- **Signal-Type Specific Rules**:
  - Premium signals: 8% profit target, 4% stop loss, 3-minute max hold
  - Technical signals: 12% profit target, 6% stop loss, 5-minute max hold
  - Standard signals: 15% profit target, 8% stop loss, 10-minute max hold

- **Dynamic Position Sizing**:
  - Base risk: 20% per trade
  - Premium multiplier: 1.5x for high-quality signals
  - Confidence-based adjustments
  - Maximum 60% position size limit

- **Data Quality Scoring**:
  - Tracks premium vs standard data sources
  - Adjusts confidence based on data quality
  - Monitors API availability and performance

## Installation and Setup

### 1. Install Dependencies

```bash
pip install aiohttp pandas yfinance asyncio
```

### 2. Configure API Keys

The API keys are already configured in `backend/trading/config.py`:

```python
# Premium API Keys (Already Configured)
ALPHA_VANTAGE_API_KEY = "UOHPTDWORX3R3Q2C"
FINNHUB_API_KEY = "d1b0ouhr01qjhvtrrrd0d1b0ouhr01qjhvtrrrdg"
TWELVE_DATA_API_KEY = "f4923d58d5b34058ba8e5a73c0e1f910"
CRYPTOPANIC_API_KEY = "23675a49e161477a7b2b3c8c4a25743ba6777e8e"
```

### 3. Test API Connections

```bash
cd backend/trading/examples
python test_premium_apis.py
```

This will test all API connections and verify functionality.

## Usage Examples

### 1. Basic Premium Data Retrieval

```python
from connectors.premium_data_connectors import PremiumDataManager

async def get_market_data():
    async with PremiumDataManager() as manager:
        # Get comprehensive market data
        data = await manager.get_comprehensive_market_data("AAPL")
        
        # Get sentiment analysis
        sentiment = await manager.get_market_sentiment_analysis("AAPL")
        
        # Generate trading intelligence
        intelligence = await manager.generate_trading_intelligence("AAPL")
        
        return intelligence
```

### 2. Run Premium Trading Demo

```bash
cd backend/trading/examples
python premium_kimera_demo.py
```

Choose from:
- Quick Premium Demo (5 cycles, 25s each)
- Extended Premium Demo (8 cycles, 30s each)
- Custom configuration

### 3. 24-Hour Premium Simulation

```bash
cd backend/trading/examples
python kimera_24h_premium_simulation.py
```

## Premium Signal Types

### 1. Premium Sentiment Analysis

```python
# Example premium sentiment signal
{
    'symbol': 'AAPL',
    'action': 'BUY',
    'confidence': 0.85,
    'reason': 'Premium sentiment: +47.3 with intelligence score: 12.45',
    'type': 'premium_sentiment',
    'priority': 'HIGH'
}
```

### 2. Cross-Source Contradiction

```python
# Example contradiction signal
{
    'symbol': 'TSLA',
    'action': 'VOLATILITY',
    'confidence': 0.72,
    'reason': 'Cross-source contradiction: Finnhub vs CryptoPanic',
    'type': 'premium_contradiction',
    'priority': 'HIGH'
}
```

### 3. Technical Analysis

```python
# Example RSI signal
{
    'symbol': 'NVDA',
    'action': 'BUY',
    'confidence': 0.65,
    'reason': 'RSI oversold: 28.3',
    'type': 'technical_rsi',
    'priority': 'MEDIUM'
}
```

## Performance Metrics

The premium system tracks enhanced performance metrics:

- **Data Quality Score**: Percentage of trades using premium data
- **Premium Signal Ratio**: Ratio of premium to standard signals
- **Signal Performance**: Success rate by signal type
- **API Reliability**: Uptime and response time monitoring
- **Risk-Adjusted Returns**: Sharpe ratio with premium data

## Webhook Integration

### Finnhub Webhook Setup

The system supports real-time webhooks from Finnhub:

```python
WEBHOOK_ENDPOINTS = {
    "finnhub": {
        "url": "http://localhost:8080/webhook/finnhub",
        "secret": "d1b0p59r01qjhvtrrsm0",
        "events": ["earnings", "news", "splits", "dividends"]
    }
}
```

Webhook events are authenticated using the secret header:
```
X-Finnhub-Secret: d1b0p59r01qjhvtrrsm0
```

## Advanced Features

### 1. Multi-Asset Support

Premium watchlists include:

**Stocks:**
- Mega Cap: AAPL, MSFT, GOOGL, AMZN, NVDA
- Growth: TSLA, NFLX, CRM, ADBE, PYPL
- Value: BRK.B, JPM, JNJ, PG, KO
- Tech: META, ORCL, INTC, AMD, QCOM

**Crypto:**
- Major: BTC-USD, ETH-USD
- Alt: SOL-USD, ADA-USD, DOT-USD
- DeFi: UNI-USD, AAVE-USD, SUSHI-USD

**Forex:**
- Major: EUR/USD, GBP/USD, USD/JPY
- Minor: AUD/USD, USD/CAD, NZD/USD

### 2. Economic Calendar Integration

Tracks key economic events:
- GDP releases
- CPI data
- Employment reports
- Federal Reserve meetings
- Earnings announcements

### 3. Sentiment Aggregation

Combines sentiment from multiple sources:
- Financial news headlines
- Social media sentiment
- Analyst reports
- Economic indicators
- Market volatility measures

## Rate Limiting and Best Practices

### API Rate Limits

- **Alpha Vantage**: 5 calls/minute
- **Finnhub**: 60 calls/minute
- **Twelve Data**: 8 calls/minute
- **CryptoPanic**: 1,000 calls/day

### Best Practices

1. **Batch Requests**: Group multiple symbol requests
2. **Cache Data**: Store frequently accessed data
3. **Error Handling**: Graceful degradation when APIs fail
4. **Rate Limiting**: Respect API limits with delays
5. **Fallback Sources**: Use multiple sources for redundancy

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify API keys are correct
   - Check subscription status
   - Ensure rate limits aren't exceeded

2. **Rate Limiting**
   - Implement exponential backoff
   - Use caching to reduce requests
   - Distribute requests across time

3. **Data Quality Issues**
   - Validate data before processing
   - Use multiple sources for confirmation
   - Implement data quality scoring

### Error Codes

- **401 Unauthorized**: Invalid API key
- **429 Too Many Requests**: Rate limit exceeded
- **403 Forbidden**: Insufficient permissions
- **500 Internal Server Error**: API provider issue

## Performance Optimization

### 1. Async Operations

All API calls use async/await for optimal performance:

```python
async with PremiumDataManager() as manager:
    # Parallel API calls
    tasks = [
        manager.get_quote_finnhub("AAPL"),
        manager.get_real_time_price_td("AAPL"),
        manager.get_intraday_data_av("AAPL")
    ]
    results = await asyncio.gather(*tasks)
```

### 2. Intelligent Caching

- Cache economic indicators (updated daily)
- Store technical indicators (updated every 5 minutes)
- Cache company fundamentals (updated weekly)

### 3. Smart Request Batching

- Group related symbol requests
- Prioritize high-priority signals
- Use connection pooling

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**
   - Sentiment analysis with NLP models
   - Pattern recognition algorithms
   - Predictive modeling

2. **Additional Data Sources**
   - Bloomberg Terminal integration
   - Reuters news feeds
   - Social media sentiment

3. **Advanced Analytics**
   - Portfolio optimization
   - Risk factor analysis
   - Performance attribution

4. **Real-Time Streaming**
   - WebSocket connections
   - Real-time price feeds
   - Live news streams

## Support and Maintenance

### Monitoring

The system includes comprehensive monitoring:
- API health checks
- Performance metrics
- Error rate tracking
- Data quality monitoring

### Alerts

Automated alerts for:
- API failures
- Rate limit violations
- Data quality issues
- Performance degradation

### Maintenance

Regular maintenance includes:
- API key rotation
- Performance optimization
- Bug fixes and updates
- Feature enhancements

## Conclusion

The KIMERA Premium Data Integration provides enterprise-level market intelligence with:

- **4 Premium APIs**: Alpha Vantage, Finnhub, Twelve Data, CryptoPanic
- **Advanced Signal Generation**: 6 types of premium signals
- **Enhanced Risk Management**: Signal-specific risk rules
- **Real-Time Intelligence**: Comprehensive market analysis
- **Scalable Architecture**: Async operations and caching

This integration represents a significant upgrade from standard trading systems, providing the data quality and intelligence needed for professional-grade algorithmic trading.

---

*Last Updated: December 2024*
*Version: 1.0*
*KIMERA Premium Integration* 