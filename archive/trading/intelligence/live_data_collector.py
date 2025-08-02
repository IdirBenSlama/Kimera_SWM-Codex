"""
Live Data Collector for Market Intelligence

Fetches real-time data from various sources.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import numpy as np

# Data sources
import yfinance as yf
import ccxt
from newsapi import NewsApiClient
import praw
from pytrends.request import TrendReq
from fredapi import Fred
try:
    from alpha_vantage.cryptocurrencies import CryptoCurrencies
    from alpha_vantage.techindicators import TechIndicators
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    CryptoCurrencies = None
    TechIndicators = None

# Configuration
from src.trading.config.config import (
    NewsConfig, RedditConfig, AlphaVantageConfig, FredConfig
)

from src.utils.kimera_logger import get_logger, LogCategory
from src.core.exception_handling import safe_operation

logger = get_logger(__name__, category=LogCategory.TRADING)


class LiveDataCollector:
    """
    Collects real-time data from multiple sources.
    """
    
    def __init__(self):
        """Initialize data collectors."""
        logger.info("Initializing Live Data Collector...")
        
        # News API
        news_config = get_news_config()
        self.news_api = NewsApiClient(api_key=news_config.api_key) if news_config.api_key else None
        
        # Reddit API
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
        else:
            self.reddit = None
        
        # Google Trends
        self.pytrends = TrendReq(hl='en-US', tz=360)
        
        # FRED (Federal Reserve Economic Data)
        self.fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
        
        # Alpha Vantage
        if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_AVAILABLE:
            self.crypto = CryptoCurrencies(key=ALPHA_VANTAGE_API_KEY)
            self.ti = TechIndicators(key=ALPHA_VANTAGE_API_KEY)
        else:
            self.crypto = None
            self.ti = None
        
        # CCXT for crypto exchanges
        self.exchange = ccxt.binance()
        
        logger.info("✅ Live Data Collector initialized")
    
    async def fetch_all_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch all available data for given symbols.
        
        Args:
            symbols: List of crypto symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            
        Returns:
            Comprehensive market data
        """
        results = {
            'timestamp': datetime.now(),
            'symbols': symbols
        }
        
        # Fetch data from multiple sources concurrently
        tasks = []
        
        # Market data
        tasks.append(self._fetch_market_data(symbols))
        
        # News data
        if self.news_api:
            tasks.append(self._fetch_news_data(symbols))
        
        # Reddit sentiment
        if self.reddit:
            tasks.append(self._fetch_reddit_sentiment(symbols))
        
        # Google Trends
        tasks.append(self._fetch_google_trends(symbols))
        
        # Macro data
        if self.fred:
            tasks.append(self._fetch_macro_data())
        
        # Execute all tasks concurrently
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(gathered):
            if isinstance(result, Exception):
                logger.warning(f"Task {i} failed: {result}")
            else:
                results.update(result)
        
        return results
    
    @safe_operation("fetch_market_data", fallback={'market_data': {}})
    async def _fetch_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch real-time market data."""
        market_data = {}
        
        for symbol in symbols:
            # Get OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=24)
            ticker = self.exchange.fetch_ticker(symbol)
            order_book = self.exchange.fetch_order_book(symbol, limit=20)
            
            # Calculate metrics
            prices = [x[4] for x in ohlcv]  # Close prices
            volumes = [x[5] for x in ohlcv]
            
            market_data[symbol] = {
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume_24h': ticker['quoteVolume'],
                'price_change_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'volatility': self._calculate_volatility(prices),
                'order_book': {
                    'bid_depth': sum([b[1] for b in order_book['bids'][:10]]),
                    'ask_depth': sum([a[1] for a in order_book['asks'][:10]]),
                    'spread': ticker['ask'] - ticker['bid'],
                    'imbalance': self._calculate_order_book_imbalance(order_book)
                }
            }
        
        return {'market_data': market_data}
    
    @safe_operation("fetch_news_data", fallback={'news': []})
    async def _fetch_news_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch crypto-related news."""
        # Build query
        crypto_terms = ['bitcoin', 'cryptocurrency', 'crypto', 'blockchain']
        for symbol in symbols:
            base = symbol.split('/')[0].lower()
            if base not in crypto_terms:
                crypto_terms.append(base)
        
        query = ' OR '.join(crypto_terms)
        
        # Fetch news
        news = self.news_api.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=50,
            from_param=(datetime.now() - timedelta(hours=24)).isoformat()
        )
        
        articles = []
        for article in news['articles'][:20]:
            articles.append({
                'title': article['title'],
                'description': article['description'],
                'source': article['source']['name'],
                'published_at': article['publishedAt'],
                'url': article['url'],
                'sentiment_text': f"{article['title']} {article['description']}"
            })
        
        return {'news': articles}
    
    @safe_operation("fetch_reddit_sentiment", fallback={'reddit': []})
    async def _fetch_reddit_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch Reddit posts and comments."""
        reddit_data = []
        subreddits = ['cryptocurrency', 'bitcoin', 'cryptomarkets']
        
        for subreddit_name in subreddits:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get hot posts
            for post in subreddit.hot(limit=10):
                # Check if post is about our symbols
                relevant = False
                for symbol in symbols:
                    base = symbol.split('/')[0].upper()
                    if base in post.title.upper() or base in post.selftext.upper():
                        relevant = True
                        break
                
                if relevant:
                    reddit_data.append({
                        'title': post.title,
                        'text': post.selftext[:500],
                        'score': post.score,
                        'num_comments': post.num_comments,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'subreddit': subreddit_name,
                        'sentiment_text': f"{post.title} {post.selftext}"
                    })
        
        return {'reddit': reddit_data}
    
    @safe_operation("fetch_google_trends", fallback={'google_trends': {}})
    async def _fetch_google_trends(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch Google Trends data."""
        keywords = []
        for symbol in symbols[:5]:  # Google Trends limits to 5 keywords
            base = symbol.split('/')[0].lower()
            keywords.append(f"{base} crypto")
        
        # Build payload
        self.pytrends.build_payload(keywords, timeframe='now 7-d')
        
        # Get interest over time
        interest = self.pytrends.interest_over_time()
        
        if not interest.empty:
            trends = {}
            for keyword in keywords:
                if keyword in interest.columns:
                    recent_values = interest[keyword].tail(24).values
                    trends[keyword] = {
                        'current': int(recent_values[-1]),
                        'average': float(recent_values.mean()),
                        'trend': 'increasing' if recent_values[-1] > recent_values[0] else 'decreasing'
                    }
            
            return {'google_trends': trends}
        
        return {'google_trends': {}}
    
    @safe_operation("fetch_macro_data", fallback={'macro_data': {}})
    async def _fetch_macro_data(self) -> Dict[str, Any]:
        """Fetch macroeconomic data."""
        macro_data = {}
        
        # Key indicators
        indicators = {
            'DGS10': '10_year_treasury',
            'DFF': 'fed_funds_rate',
            'UNRATE': 'unemployment_rate',
            'CPIAUCSL': 'cpi',
            'DEXUSEU': 'usd_eur'
        }
        
        for fred_id, name in indicators.items():
            try:
                # Get latest value
                series = self.fred.get_series(fred_id, limit=1)
                if not series.empty:
                    macro_data[name] = float(series.iloc[-1])
            except Exception as e:
                logger.warning(f"Could not fetch FRED indicator {fred_id}", error=e)
        
        return {'macro_data': macro_data}
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility."""
        if len(prices) < 2:
            return 0
        
        returns = []
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        return float(np.std(returns) * np.sqrt(24))  # Annualized volatility
    
    def _calculate_order_book_imbalance(self, order_book: Dict) -> float:
        """Calculate order book imbalance."""
        bid_volume = sum([b[1] for b in order_book['bids'][:10]])
        ask_volume = sum([a[1] for a in order_book['asks'][:10]])
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0
        
        return (bid_volume - ask_volume) / total


class SimulatedDataCollector:
    """
    Simulates live data when APIs are not available.
    """
    
    @staticmethod
    async def get_simulated_intelligence() -> Dict[str, Any]:
        """Get simulated market intelligence data."""
        import random
        
        # Simulate different market conditions
        market_condition = random.choice(['bullish', 'bearish', 'volatile', 'neutral'])
        
        base_sentiment = {
            'bullish': 0.6,
            'bearish': -0.4,
            'volatile': 0.1,
            'neutral': 0.0
        }[market_condition]
        
        # Add noise
        sentiment = base_sentiment + random.uniform(-0.2, 0.2)
        
        return {
            'timestamp': datetime.now(),
            'market_condition': market_condition,
            'intelligence': {
                'news_sentiment': sentiment + random.uniform(-0.1, 0.1),
                'social_sentiment': sentiment + random.uniform(-0.15, 0.15),
                'google_trends': {
                    'bitcoin': random.randint(40, 100),
                    'crypto': random.randint(30, 90),
                    'trend': random.choice(['increasing', 'decreasing', 'stable'])
                },
                'reddit_activity': {
                    'posts_per_hour': random.randint(20, 200),
                    'sentiment': sentiment + random.uniform(-0.2, 0.2),
                    'fomo_level': random.uniform(0, 1),
                    'fud_level': random.uniform(0, 1)
                },
                'macro_factors': {
                    'dollar_strength': random.uniform(90, 110),
                    'interest_rates': random.uniform(0, 5),
                    'inflation': random.uniform(2, 8)
                },
                'geopolitical_risk': random.uniform(0, 1),
                'technical_signals': {
                    'rsi': random.uniform(20, 80),
                    'macd': random.choice(['bullish', 'bearish', 'neutral']),
                    'support_levels': [40000, 42000, 44000],
                    'resistance_levels': [48000, 50000, 52000]
                }
            }
        } 