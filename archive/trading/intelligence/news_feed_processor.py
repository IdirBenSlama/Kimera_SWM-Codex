"""
News Feed Processor
==================

Processes news feeds from multiple sources to extract market sentiment
and identify potential contradictions with market behavior.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import os

logger = logging.getLogger(__name__)


class NewsFeedProcessor:
    """
    Processes news feeds to extract sentiment and semantic information
    for Kimera's contradiction detection engine.
    
    Integrates with CryptoPanic API for real-time crypto news.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the news feed processor
        
        Args:
            config: Configuration for news sources
        """
        self.config = config or {}
        self.news_cache = {}
        self.cache_ttl = timedelta(minutes=10)
        
        # News sources configuration
        self.sources = self.config.get('sources', [
            'reuters', 'bloomberg', 'coindesk', 'cointelegraph', 'cryptopanic'
        ])
        
        # Initialize CryptoPanic connector
        self.cryptopanic_connector = None
        cryptopanic_key = self.config.get('cryptopanic_api_key') or os.getenv('CRYPTOPANIC_API_KEY')
        
        if cryptopanic_key:
            try:
                from src.trading.connectors.cryptopanic_connector import create_cryptopanic_connector
                self.cryptopanic_connector = create_cryptopanic_connector(
                    api_key=cryptopanic_key,
                    testnet=self.config.get('cryptopanic_testnet', True)
                )
                logger.info("✅ CryptoPanic integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize CryptoPanic: {e}")
        else:
            logger.info("ℹ️ CryptoPanic API key not provided - using simulated data")
        
        logger.info("📰 News Feed Processor initialized")
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get news sentiment for a given symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            News sentiment analysis
        """
        # Check cache
        cache_key = f"news_{symbol}"
        if cache_key in self.news_cache:
            cached_data, timestamp = self.news_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data
        
        # Try to get real data from CryptoPanic
        if self.cryptopanic_connector:
            try:
                # Get sentiment analysis from CryptoPanic
                analysis = await self.cryptopanic_connector.analyze_sentiment_trend(
                    symbol=symbol.replace('-USD', ''),  # Remove USD suffix
                    hours=6
                )
                
                # Get latest news
                news_items = await self.cryptopanic_connector.get_currency_posts(
                    symbol=symbol.replace('-USD', ''),
                    limit=20
                )
                
                # Process into our format
                sentiment_data = {
                    'score': analysis['sentiment_score'],
                    'volume': analysis['total_news'],
                    'momentum': self._calculate_momentum(news_items),
                    'topics': self._extract_topics(news_items),
                    'key_events': self._extract_events(news_items),
                    'source_breakdown': self._analyze_sources(news_items),
                    'bullish_count': analysis['bullish_count'],
                    'bearish_count': analysis['bearish_count'],
                    'importance_avg': analysis['importance_avg'],
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'cryptopanic'
                }
                
                # Cache result
                self.news_cache[cache_key] = (sentiment_data, datetime.now())
                return sentiment_data
                
            except Exception as e:
                logger.error(f"Error fetching CryptoPanic data: {e}")
                # Fall back to simulated data
        
        # Fallback to simulated data
        sentiment_data = {
            'score': np.random.uniform(-1, 1),  # Overall sentiment
            'volume': np.random.randint(5, 50),  # Number of articles
            'momentum': np.random.uniform(-1, 1),  # Sentiment change rate
            'topics': self._generate_topics(),
            'key_events': self._generate_events(symbol),
            'source_breakdown': {
                source: np.random.uniform(-1, 1) 
                for source in self.sources
            },
            'timestamp': datetime.now().isoformat(),
            'data_source': 'simulated'
        }
        
        # Cache result
        self.news_cache[cache_key] = (sentiment_data, datetime.now())
        
        return sentiment_data
    
    def _calculate_momentum(self, news_items: List[Any]) -> float:
        """Calculate sentiment momentum from news items"""
        if len(news_items) < 2:
            return 0.0
        
        # Sort by time
        sorted_news = sorted(news_items, key=lambda x: x.published_at)
        
        # Calculate sentiment change over time
        recent_sentiment = np.mean([item.sentiment_score for item in sorted_news[-5:]])
        older_sentiment = np.mean([item.sentiment_score for item in sorted_news[:-5]])
        
        return recent_sentiment - older_sentiment
    
    def _extract_topics(self, news_items: List[Any]) -> List[Dict[str, Any]]:
        """Extract topics from news items"""
        if not news_items:
            return self._generate_topics()
        
        # Count sentiment by common crypto topics
        topic_sentiments = {
            'regulation': [],
            'adoption': [],
            'technology': [],
            'market_analysis': [],
            'partnerships': [],
            'security': []
        }
        
        # Analyze titles for topics
        for item in news_items:
            title_lower = item.title.lower()
            
            if any(word in title_lower for word in ['regulation', 'sec', 'legal', 'law']):
                topic_sentiments['regulation'].append(item.sentiment_score)
            if any(word in title_lower for word in ['adoption', 'accept', 'integrate']):
                topic_sentiments['adoption'].append(item.sentiment_score)
            if any(word in title_lower for word in ['upgrade', 'technology', 'development']):
                topic_sentiments['technology'].append(item.sentiment_score)
            if any(word in title_lower for word in ['analysis', 'prediction', 'forecast']):
                topic_sentiments['market_analysis'].append(item.sentiment_score)
            if any(word in title_lower for word in ['partner', 'collaboration', 'deal']):
                topic_sentiments['partnerships'].append(item.sentiment_score)
            if any(word in title_lower for word in ['hack', 'security', 'breach', 'vulnerability']):
                topic_sentiments['security'].append(item.sentiment_score)
        
        # Build topic list
        topics = []
        for topic, sentiments in topic_sentiments.items():
            if sentiments:
                topics.append({
                    'topic': topic,
                    'sentiment': np.mean(sentiments),
                    'relevance': len(sentiments) / len(news_items)
                })
        
        return sorted(topics, key=lambda x: x['relevance'], reverse=True)[:3]
    
    def _extract_events(self, news_items: List[Any]) -> List[Dict[str, Any]]:
        """Extract key events from news items"""
        if not news_items:
            return self._generate_events('')
        
        events = []
        for item in news_items[:5]:  # Top 5 most recent
            # Determine event type based on content
            title_lower = item.title.lower()
            
            event_type = 'market_movement'  # default
            if 'partner' in title_lower or 'deal' in title_lower:
                event_type = 'partnership_announcement'
            elif 'regulation' in title_lower or 'sec' in title_lower:
                event_type = 'regulatory_update'
            elif 'upgrade' in title_lower or 'launch' in title_lower:
                event_type = 'technical_upgrade'
            elif 'whale' in title_lower or 'large' in title_lower:
                event_type = 'whale_activity'
            
            # Determine impact based on votes
            votes_total = sum(item.votes.values())
            impact = 'low'
            if votes_total > 100:
                impact = 'high'
            elif votes_total > 50:
                impact = 'medium'
            
            time_diff = datetime.now() - item.published_at.replace(tzinfo=None)
            
            events.append({
                'type': event_type,
                'impact': impact,
                'sentiment': item.sentiment_score,
                'time_ago_hours': time_diff.total_seconds() / 3600,
                'title': item.title[:100],
                'url': item.url
            })
        
        return events
    
    def _analyze_sources(self, news_items: List[Any]) -> Dict[str, float]:
        """Analyze sentiment by news source"""
        source_sentiments = {}
        
        for item in news_items:
            source = item.source.get('title', 'Unknown')
            if source not in source_sentiments:
                source_sentiments[source] = []
            source_sentiments[source].append(item.sentiment_score)
        
        # Calculate average sentiment per source
        return {
            source: np.mean(sentiments)
            for source, sentiments in source_sentiments.items()
        }
    
    def _generate_topics(self) -> List[Dict[str, Any]]:
        """Generate simulated news topics"""
        topics = [
            'regulation', 'adoption', 'technology', 
            'market_analysis', 'partnerships', 'security'
        ]
        
        return [
            {
                'topic': topic,
                'sentiment': np.random.uniform(-1, 1),
                'relevance': np.random.uniform(0, 1)
            }
            for topic in np.random.choice(topics, size=3, replace=False)
        ]
    
    def _generate_events(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate simulated key events"""
        event_types = [
            'earnings_report', 'partnership_announcement',
            'regulatory_update', 'technical_upgrade',
            'market_movement', 'whale_activity'
        ]
        
        events = []
        for _ in range(np.random.randint(0, 3)):
            events.append({
                'type': np.random.choice(event_types),
                'impact': np.random.choice(['high', 'medium', 'low']),
                'sentiment': np.random.uniform(-1, 1),
                'time_ago_hours': np.random.randint(1, 24),
                'title': 'Simulated event',
                'url': '#'
            })
        
        return events
    
    async def detect_news_contradictions(self, 
                                       symbol: str, 
                                       market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect contradictions between news sentiment and market behavior
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            List of detected contradictions
        """
        news_sentiment = await self.get_sentiment(symbol)
        contradictions = []
        
        # Check for price/sentiment divergence
        price_momentum = market_data.get('momentum', 0)
        news_score = news_sentiment['score']
        
        if abs(price_momentum - news_score) > 0.5:
            contradictions.append({
                'type': 'sentiment_price_divergence',
                'severity': abs(price_momentum - news_score),
                'description': f"News sentiment ({news_score:.2f}) contradicts price momentum ({price_momentum:.2f})",
                'sources': ['news', 'price_action'],
                'data_source': news_sentiment.get('data_source', 'unknown')
            })
        
        # Check for source disagreement
        source_sentiments = list(news_sentiment['source_breakdown'].values())
        if source_sentiments:
            sentiment_std = np.std(source_sentiments)
            if sentiment_std > 0.5:
                contradictions.append({
                    'type': 'source_disagreement',
                    'severity': sentiment_std,
                    'description': "Significant disagreement between news sources",
                    'sources': list(news_sentiment['source_breakdown'].keys())
                })
        
        # Use CryptoPanic contradiction detection if available
        if self.cryptopanic_connector and market_data.get('sentiment'):
            cp_contradiction = await self.cryptopanic_connector.detect_sentiment_contradictions(
                symbol=symbol.replace('-USD', ''),
                market_sentiment=market_data['sentiment'],
                threshold=0.4
            )
            
            if cp_contradiction:
                contradictions.append({
                    'type': 'cryptopanic_contradiction',
                    'severity': cp_contradiction['contradiction_strength'],
                    'description': f"CryptoPanic news sentiment contradicts market sentiment",
                    'sources': ['cryptopanic', 'market'],
                    'details': cp_contradiction
                })
        
        return contradictions
    
    async def stream_news_updates(self, symbols: List[str], callback):
        """
        Stream real-time news updates
        
        Args:
            symbols: List of symbols to monitor
            callback: Async function to call with news updates
        """
        if not self.cryptopanic_connector:
            logger.warning("CryptoPanic not configured - streaming unavailable")
            return
        
        # Convert symbols (remove -USD suffix)
        crypto_symbols = [s.replace('-USD', '') for s in symbols]
        
        async def process_updates(news_items):
            """Process streaming news updates"""
            for item in news_items:
                # Find matching symbol
                for currency in item.currencies:
                    if currency.get('code', '').upper() in crypto_symbols:
                        update = {
                            'symbol': f"{currency['code']}-USD",
                            'title': item.title,
                            'sentiment': item.sentiment_score,
                            'importance': item.importance_score,
                            'url': item.url,
                            'timestamp': item.published_at.isoformat()
                        }
                        await callback(update)
        
        # Start streaming
        await self.cryptopanic_connector.stream_news_updates(
            currencies=crypto_symbols,
            callback=process_updates,
            interval=60
        ) 