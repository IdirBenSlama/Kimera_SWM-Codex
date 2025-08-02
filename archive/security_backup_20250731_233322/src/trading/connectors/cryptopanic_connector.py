"""
CryptoPanic News Connector for KIMERA Trading System
Provides real-time crypto news with sentiment analysis

SECURITY: API keys must be provided via environment variables.
Never commit API keys to source control.
"""

import asyncio
import aiohttp
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
from dataclasses import dataclass
from enum import Enum

from src.core.exception_handling import safe_operation, CircuitBreaker

logger = logging.getLogger(__name__)

class NewsKind(Enum):
    """Type of news content"""
    NEWS = "news"
    MEDIA = "media"
    BLOG = "blog"

class NewsSentiment(Enum):
    """News sentiment categories"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    IMPORTANT = "important"

@dataclass
class CryptoNews:
    """Structured crypto news item"""
    id: int
    kind: NewsKind
    domain: str
    source: Dict[str, Any]
    title: str
    published_at: datetime
    slug: str
    currencies: List[Dict[str, Any]]
    url: str
    created_at: datetime
    votes: Dict[str, int]
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def panic_score(self) -> int:
        """Calculate panic score based on votes"""
        return (
            self.votes.get('negative', 0) * 2 +
            self.votes.get('positive', 0) +
            self.votes.get('important', 0) * 3 +
            self.votes.get('liked', 0) +
            self.votes.get('disliked', 0) * -1 +
            self.votes.get('lol', 0) * 0.5 +
            self.votes.get('toxic', 0) * -2 +
            self.votes.get('saved', 0) * 1.5
        )
    
    @property
    def sentiment(self) -> NewsSentiment:
        """Determine overall sentiment"""
        if self.votes.get('negative', 0) > self.votes.get('positive', 0):
            return NewsSentiment.NEGATIVE
        elif self.votes.get('positive', 0) > self.votes.get('negative', 0):
            return NewsSentiment.POSITIVE
        elif self.votes.get('important', 0) > 0:
            return NewsSentiment.IMPORTANT
        return NewsSentiment.NEUTRAL

class CryptoPanicConnector:
    """
    CryptoPanic API connector for real-time crypto news
    
    Features:
    - Real-time news streaming
    - Sentiment analysis
    - PanicScore calculation
    - Multi-currency filtering
    - Rate limit management
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize CryptoPanic connector
        
        Args:
            api_key: Optional API key. If not provided, will look for CRYPTOPANIC_API_KEY env var
            
        Raises:
            ValueError: If no API key is provided and CRYPTOPANIC_API_KEY env var is not set
        """
        if api_key is None:
            api_key = os.getenv("CRYPTOPANIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "CryptoPanic API key not provided. "
                    "Either pass api_key parameter or set CRYPTOPANIC_API_KEY environment variable"
                )
        
        self.api_key = api_key
        self.base_url = "https://cryptopanic.com/api/developer/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_remaining = 1000  # Daily limit
        self._last_request_time = None
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    @safe_operation(
        operation="cryptopanic_api_request",
        use_circuit_breaker=True
    )
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API request with rate limiting and circuit breaker."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        if params is None:
            params = {}
        params['auth_token'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}/"
        
        try:
            async with self.session.get(url, params=params) as response:
                # Update rate limit info
                if 'X-RateLimit-Remaining' in response.headers:
                    self._rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                    
                if response.status == 429:
                    logger.warning("Rate limit exceeded")
                    self.circuit_breaker.record_failure()
                    raise Exception("Rate limit exceeded")
                    
                response.raise_for_status()
                self.circuit_breaker.record_success()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            self.circuit_breaker.record_failure()
            raise
            
    def _parse_news_item(self, item: Dict[str, Any]) -> CryptoNews:
        """Parse raw API response into CryptoNews object"""
        # Handle missing fields gracefully
        return CryptoNews(
            id=item['id'],
            kind=NewsKind(item.get('kind', 'news')),
            domain=item.get('domain', 'cryptopanic.com'),  # Default if missing
            source=item.get('source', {'title': 'Unknown', 'domain': 'unknown'}),
            title=item['title'],
            published_at=datetime.fromisoformat(item['published_at'].replace('Z', '+00:00')),
            slug=item['slug'],
            currencies=item.get('currencies', []),
            url=item.get('url', f"https://cryptopanic.com/news/{item['slug']}"),
            created_at=datetime.fromisoformat(item['created_at'].replace('Z', '+00:00')),
            votes=item.get('votes', {}),
            metadata=item.get('metadata', {})
        )
        
    async def get_posts(self, 
                       currencies: Optional[List[str]] = None,
                       kind: Optional[NewsKind] = None,
                       filter: Optional[str] = None,
                       regions: Optional[str] = None,
                       public: bool = True) -> List[CryptoNews]:
        """
        Get latest posts from CryptoPanic
        
        Args:
            currencies: List of currency codes (e.g., ['BTC', 'ETH'])
            kind: Type of content (news, media, blog)
            filter: Filter by importance (rising, hot, bullish, bearish, important, saved, lol)
            regions: Region codes (en, de, es, fr, nl, it, pt, ru)
            public: Whether to fetch only public posts
            
        Returns:
            List of CryptoNews objects
        """
        params = {}
        
        if currencies:
            params['currencies'] = ','.join(currencies)
        if kind:
            params['kind'] = kind.value
        if filter:
            params['filter'] = filter
        if regions:
            params['regions'] = regions
        if public:
            params['public'] = 'true'
            
        response = await self._make_request('posts', params)
        
        news_items = []
        for item in response.get('results', []):
            try:
                news_items.append(self._parse_news_item(item))
            except Exception as e:
                logger.error(f"Failed to parse news item: {e}")
                continue
                
        return news_items
        
    async def get_trending_news(self, limit: int = 50) -> List[CryptoNews]:
        """Get trending crypto news"""
        return await self.get_posts(filter='hot')
        
    async def get_important_news(self, limit: int = 20) -> List[CryptoNews]:
        """Get important/breaking crypto news"""
        return await self.get_posts(filter='important')
        
    async def get_bullish_news(self, currencies: Optional[List[str]] = None) -> List[CryptoNews]:
        """Get bullish sentiment news"""
        return await self.get_posts(currencies=currencies, filter='bullish')
        
    async def get_bearish_news(self, currencies: Optional[List[str]] = None) -> List[CryptoNews]:
        """Get bearish sentiment news"""
        return await self.get_posts(currencies=currencies, filter='bearish')
        
    async def stream_news(self, 
                         callback,
                         currencies: Optional[List[str]] = None,
                         interval: int = 60):
        """
        Stream news updates at regular intervals
        
        Args:
            callback: Async function to call with new news items
            currencies: List of currencies to monitor
            interval: Polling interval in seconds
        """
        last_id = None
        
        while True:
            try:
                news_items = await self.get_posts(currencies=currencies)
                
                # Filter only new items
                if last_id:
                    new_items = [item for item in news_items if item.id > last_id]
                else:
                    new_items = news_items
                    
                if new_items:
                    last_id = max(item.id for item in new_items)
                    await callback(new_items)
                    
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in news stream: {e}")
                await asyncio.sleep(interval * 2)  # Back off on error
                
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return {
            'remaining': self._rate_limit_remaining,
            'limit': 1000,  # Daily limit for developer plan
            'reset': 'daily'
        }
        
    async def analyze_market_sentiment(self, 
                                     currencies: Optional[List[str]] = None,
                                     lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze overall market sentiment from recent news
        
        Returns sentiment scores and trending topics
        """
        news_items = await self.get_posts(currencies=currencies)
        
        # Calculate sentiment distribution
        sentiment_counts = {
            NewsSentiment.POSITIVE: 0,
            NewsSentiment.NEGATIVE: 0,
            NewsSentiment.NEUTRAL: 0,
            NewsSentiment.IMPORTANT: 0
        }
        
        total_panic_score = 0
        currency_mentions = {}
        source_distribution = {}
        
        for item in news_items:
            sentiment_counts[item.sentiment] += 1
            total_panic_score += item.panic_score
            
            # Track currency mentions
            for currency in item.currencies:
                code = currency.get('code', 'UNKNOWN')
                currency_mentions[code] = currency_mentions.get(code, 0) + 1
                
            # Track sources
            source_name = item.source.get('title', 'Unknown')
            source_distribution[source_name] = source_distribution.get(source_name, 0) + 1
            
        # Calculate sentiment score (-100 to +100)
        total_items = len(news_items)
        if total_items > 0:
            sentiment_score = (
                (sentiment_counts[NewsSentiment.POSITIVE] * 100 -
                 sentiment_counts[NewsSentiment.NEGATIVE] * 100) / total_items
            )
        else:
            sentiment_score = 0
            
        return {
            'sentiment_score': sentiment_score,
            'sentiment_distribution': {k.value: v for k, v in sentiment_counts.items()},
            'average_panic_score': total_panic_score / total_items if total_items > 0 else 0,
            'trending_currencies': sorted(
                currency_mentions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            'top_sources': sorted(
                source_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'total_news_analyzed': total_items,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Example usage
async def main():
    """Example usage of CryptoPanic connector"""
    async with CryptoPanicConnector() as connector:
        # Get latest news
        logger.info("=== Latest Crypto News ===")
        news = await connector.get_posts(limit=5)
        for item in news[:5]:
            logger.info(f"\nðŸ“° {item.title}")
            logger.info(f"   Source: {item.source['title']}")
            logger.info(f"   Sentiment: {item.sentiment.value}")
            logger.info(f"   Panic Score: {item.panic_score}")
            logger.info(f"   Currencies: {', '.join([c['code'] for c in item.currencies])}")
            
        # Get Bitcoin-specific bullish news
        logger.info("\n\n=== Bullish Bitcoin News ===")
        btc_bullish = await connector.get_bullish_news(['BTC'])
        for item in btc_bullish[:3]:
            logger.info(f"\nðŸ“ˆ {item.title}")
            logger.info(f"   URL: {item.url}")
            
        # Analyze market sentiment
        logger.info("\n\n=== Market Sentiment Analysis ===")
        sentiment = await connector.analyze_market_sentiment()
        logger.info(f"Overall Sentiment Score: {sentiment['sentiment_score']:.2f}")
        logger.info(f"Sentiment Distribution: {sentiment['sentiment_distribution']}")
        logger.info(f"Top Trending Currencies: {sentiment['trending_currencies'][:5]}")
        
        # Check rate limits
        logger.info(f"\n\n=== Rate Limit Status ===")
        logger.info(connector.get_rate_limit_status())


if __name__ == "__main__":
    asyncio.run(main())


def create_cryptopanic_connector(api_key: Optional[str] = None) -> CryptoPanicConnector:
    """
    Factory function to create a CryptoPanic connector
    
    Args:
        api_key: Optional API key (uses default if not provided)
        
    Returns:
        CryptoPanicConnector instance
    """
    if api_key:
        return CryptoPanicConnector(api_key=api_key)
    else:
        return CryptoPanicConnector()


# Alias for backward compatibility
CryptoPanicNews = CryptoNews 