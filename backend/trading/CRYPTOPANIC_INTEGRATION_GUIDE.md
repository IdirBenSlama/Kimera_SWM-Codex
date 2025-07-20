# CryptoPanic API Integration Guide for KIMERA Trading

## Overview

CryptoPanic is integrated into KIMERA's semantic trading system to provide real-time crypto news with sentiment analysis. This integration enables KIMERA to detect semantic contradictions between news sources and generate trading signals based on market sentiment.

## API Credentials

- **API Key**: `23675a49e161477a7b2b3c8c4a25743ba6777e8e`
- **Plan**: DEVELOPER tier
- **Base Endpoint**: `https://cryptopanic.com/api/developer/v2/posts/`
- **Rate Limits**: 1,000 requests per day

## Features Available

### 1. Real-time News Streaming
- Latest crypto news from multiple sources
- Structured content with title, description, URL, author, source
- Multi-language support

### 2. Sentiment Analysis
- PanicScore™ proprietary attention metric
- Sentiment classification (positive, negative, neutral, important)
- Vote-based community sentiment

### 3. Advanced Filtering
- Filter by cryptocurrency (BTC, ETH, etc.)
- Filter by sentiment (bullish, bearish)
- Filter by importance (trending, hot, important)
- Regional filtering (en, de, es, fr, nl, it, pt, ru)

### 4. Market Intelligence
- Trending cryptocurrency detection
- Source reliability tracking
- Breaking news alerts

## Integration Architecture

```
┌─────────────────────────────┐
│     CryptoPanic API         │
│  (Real-time News Stream)    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  CryptoPanic Connector      │
│  • News parsing             │
│  • Sentiment extraction     │
│  • Rate limit management    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Semantic Trading Reactor   │
│  • Contradiction detection  │
│  • Geoid creation          │
│  • Thermodynamic analysis  │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│   Trading Signal Generator  │
│  • Signal confidence       │
│  • Risk assessment        │
│  • Execution triggers     │
└─────────────────────────────┘
```

## Quick Start

### 1. Basic News Retrieval

```python
from backend.trading.connectors.cryptopanic_connector import CryptoPanicConnector

async def get_latest_news():
    async with CryptoPanicConnector() as connector:
        # Get latest news
        news = await connector.get_posts()
        
        for item in news[:5]:
            print(f"{item.title}")
            print(f"Sentiment: {item.sentiment.value}")
            print(f"Panic Score: {item.panic_score}")
```

### 2. Market Sentiment Analysis

```python
async def analyze_market():
    async with CryptoPanicConnector() as connector:
        # Get overall market sentiment
        sentiment = await connector.analyze_market_sentiment()
        
        print(f"Market Sentiment Score: {sentiment['sentiment_score']}")
        print(f"Top Trending: {sentiment['trending_currencies'][:5]}")
```

### 3. Contradiction Detection

```python
from backend.trading.examples.cryptopanic_demo import CryptoPanicKimeraDemo

async def detect_contradictions():
    demo = CryptoPanicKimeraDemo()
    
    # Get news and analyze for contradictions
    async with demo.crypto_connector as connector:
        news = await connector.get_posts()
        contradictions = await demo.analyze_news_contradictions(news)
        
        if contradictions['contradictions_found'] > 0:
            print(f"Found {contradictions['contradictions_found']} contradictions!")
            for c in contradictions['contradictions']:
                print(f"- {c['currency']}: {c['source1']['sentiment']} vs {c['source2']['sentiment']}")
```

### 4. Trading Signal Generation

```python
async def generate_signals():
    demo = CryptoPanicKimeraDemo()
    
    async with demo.crypto_connector as connector:
        # Get sentiment and news
        sentiment = await connector.analyze_market_sentiment()
        news = await connector.get_posts()
        contradictions = await demo.analyze_news_contradictions(news)
        
        # Generate trading signals
        signals = await demo.generate_trading_signals(sentiment, contradictions)
        
        for signal in signals:
            print(f"Signal: {signal['type']}")
            print(f"Action: {signal['action']}")
            print(f"Confidence: {signal['confidence']:.1%}")
```

## Use Cases

### 1. Sentiment-Driven Trading
- Monitor overall market sentiment
- Trade based on bullish/bearish consensus
- Detect sentiment shifts early

### 2. Volatility Detection
- Identify contradictory news reports
- Trade volatility through options strategies
- Risk management based on uncertainty

### 3. Event-Driven Trading
- React to breaking news
- Monitor specific cryptocurrency news
- Trade on important announcements

### 4. Market Stress Detection
- Monitor panic scores
- Implement defensive strategies
- Hedge during high-stress periods

## Advanced Features

### News Streaming
```python
async def news_callback(new_items):
    """Process new news items"""
    for item in new_items:
        print(f"New: {item.title} ({item.sentiment.value})")

# Stream news updates every 60 seconds
await connector.stream_news(
    callback=news_callback,
    currencies=['BTC', 'ETH'],
    interval=60
)
```

### Custom Filters
```python
# Get only bullish Bitcoin news
btc_bullish = await connector.get_bullish_news(['BTC'])

# Get important breaking news
breaking = await connector.get_important_news()

# Get trending news
trending = await connector.get_trending_news()
```

## Running the Demo

### Interactive Demo
```bash
cd backend/trading/examples
python cryptopanic_demo.py
```

Choose from:
1. 5-minute live demo with real-time updates
2. Test specific features
3. Run both

### Simple API Test
```bash
cd backend/trading/examples
python test_cryptopanic_api.py
```

## Configuration

### Environment Variables
```bash
# .env file
CRYPTOPANIC_API_KEY=23675a49e161477a7b2b3c8c4a25743ba6777e8e
```

### In Code
```python
# Direct configuration
connector = CryptoPanicConnector(api_key="23675a49e161477a7b2b3c8c4a25743ba6777e8e")

# Or use default (already configured)
connector = CryptoPanicConnector()
```

## Rate Limit Management

The connector automatically tracks rate limits:

```python
# Check current status
status = connector.get_rate_limit_status()
print(f"Remaining: {status['remaining']}/{status['limit']}")
```

- Daily limit: 1,000 requests
- The connector respects rate limits automatically
- Backs off on errors

## Error Handling

The connector handles common errors gracefully:
- Network timeouts
- Rate limit exceeded (429)
- Invalid responses
- Missing data fields

## Best Practices

1. **Cache News Items**: Store recent news to avoid redundant API calls
2. **Batch Analysis**: Analyze multiple news items together for better context
3. **Time-based Filtering**: Focus on recent news for trading decisions
4. **Source Reliability**: Weight signals based on source credibility
5. **Combine with Technical**: Use news as confirmation for technical signals

## Troubleshooting

### No News Retrieved
- Check API key is correct
- Verify internet connection
- Check rate limit status

### Parsing Errors
- The connector handles missing fields gracefully
- Check logs for specific field issues

### Rate Limit Issues
- Monitor usage throughout the day
- Implement caching for frequently accessed data
- Consider upgrading plan for higher limits

## Future Enhancements

1. **Machine Learning Integration**
   - Train models on news sentiment vs price movement
   - Predict market impact of news events

2. **Advanced NLP**
   - Extract entities and relationships
   - Deeper semantic analysis

3. **Multi-Exchange Correlation**
   - Compare news impact across exchanges
   - Arbitrage opportunity detection

4. **Custom Alerts**
   - Webhook integration
   - Telegram/Discord notifications

## Support

- CryptoPanic Documentation: https://cryptopanic.com/developers/api/
- KIMERA Trading Module: See `INTEGRATION_SUMMARY.md`
- Example Scripts: `backend/trading/examples/`

The CryptoPanic integration is now fully operational and ready to provide real-time market intelligence to KIMERA's semantic trading system! 