"""
Premium Data Connectors for KIMERA Trading System
Integrates Alpha Vantage, Finnhub, and Twelve Data APIs
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import json

from backend.trading import config

logger = logging.getLogger(__name__)

class DataProvider(Enum):
    """Available data providers"""
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    TWELVE_DATA = "twelve_data"

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    provider: DataProvider
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class NewsItem:
    """Standardized news item structure"""
    id: str
    headline: str
    summary: str
    source: str
    datetime: datetime
    related_symbols: List[str]
    sentiment: Optional[float] = None
    provider: DataProvider = DataProvider.FINNHUB
    url: Optional[str] = None
    category: Optional[str] = None

@dataclass
class EconomicIndicator:
    """Economic indicator data"""
    indicator: str
    value: float
    timestamp: datetime
    country: str
    previous_value: Optional[float] = None
    forecast: Optional[float] = None
    provider: DataProvider = DataProvider.ALPHA_VANTAGE

class PremiumDataManager:
    """
    Unified manager for premium financial data APIs
    
    Features:
    - Real-time and historical market data
    - Company fundamentals and financials
    - Economic indicators
    - News and sentiment analysis
    - Earnings and events data
    - Technical indicators
    """
    
    def __init__(self):
        # API Keys loaded from config
        self.alpha_vantage_key = config.ALPHA_VANTAGE_API_KEY
        self.finnhub_key = config.FINNHUB_API_KEY
        self.finnhub_webhook_secret = config.FINNHUB_WEBHOOK_SECRET
        self.twelve_data_key = config.TWELVE_DATA_API_KEY
        
        # Base URLs
        self.alpha_vantage_url = config.ALPHA_VANTAGE_BASE_URL
        self.finnhub_url = config.FINNHUB_BASE_URL
        self.twelve_data_url = "https://api.twelvedata.com"
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.rate_limits = {
            DataProvider.ALPHA_VANTAGE: {"calls_per_minute": 5, "last_call": None},
            DataProvider.FINNHUB: {"calls_per_minute": 60, "last_call": None},
            DataProvider.TWELVE_DATA: {"calls_per_minute": 8, "last_call": None}
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def _make_request(self, url: str, params: Dict[str, Any], provider: DataProvider) -> Dict[str, Any]:
        """Make rate-limited API request"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        # Simple rate limiting
        await asyncio.sleep(0.1)  # Basic rate limiting
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"API request failed for {provider.value}: {e}")
            raise
            
    # ================== ALPHA VANTAGE METHODS ==================
    
    async def get_intraday_data_av(self, symbol: str, interval: str = "1min") -> List[MarketData]:
        """Get intraday data from Alpha Vantage"""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.alpha_vantage_key,
            "outputsize": "compact"
        }
        
        data = await self._make_request(self.alpha_vantage_url, params, DataProvider.ALPHA_VANTAGE)
        
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            logger.warning(f"No time series data for {symbol}")
            return []
            
        market_data = []
        for timestamp_str, values in data[time_series_key].items():
            market_data.append(MarketData(
                symbol=symbol,
                timestamp=datetime.fromisoformat(timestamp_str.replace(' ', 'T')),
                open=float(values["1. open"]),
                high=float(values["2. high"]),
                low=float(values["3. low"]),
                close=float(values["4. close"]),
                volume=int(values["5. volume"]),
                provider=DataProvider.ALPHA_VANTAGE,
                metadata=data.get("Meta Data", {})
            ))
            
        return sorted(market_data, key=lambda x: x.timestamp, reverse=True)
        
    async def get_economic_indicators_av(self, indicator: str = "GDP") -> List[EconomicIndicator]:
        """Get economic indicators from Alpha Vantage"""
        function_map = {
            "GDP": "REAL_GDP",
            "CPI": "CPI",
            "UNEMPLOYMENT": "UNEMPLOYMENT",
            "FEDERAL_FUNDS_RATE": "FEDERAL_FUNDS_RATE"
        }
        
        params = {
            "function": function_map.get(indicator, "REAL_GDP"),
            "apikey": self.alpha_vantage_key
        }
        
        data = await self._make_request(self.alpha_vantage_url, params, DataProvider.ALPHA_VANTAGE)
        
        indicators = []
        if "data" in data:
            for item in data["data"]:
                indicators.append(EconomicIndicator(
                    indicator=indicator,
                    value=float(item["value"]),
                    timestamp=datetime.fromisoformat(item["date"]),
                    country="US",
                    provider=DataProvider.ALPHA_VANTAGE
                ))
                
        return indicators
        
    async def get_company_overview_av(self, symbol: str) -> Dict[str, Any]:
        """Get company overview from Alpha Vantage"""
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.alpha_vantage_key
        }
        
        return await self._make_request(self.alpha_vantage_url, params, DataProvider.ALPHA_VANTAGE)
        
    # ================== FINNHUB METHODS ==================
    
    async def get_quote_finnhub(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote from Finnhub"""
        url = f"{self.finnhub_url}/quote"
        params = {
            "symbol": symbol,
            "token": self.finnhub_key
        }
        
        return await self._make_request(url, params, DataProvider.FINNHUB)
        
    async def get_news_finnhub(self, category: str = "general", min_id: Optional[str] = None) -> List[NewsItem]:
        """Get market news from Finnhub"""
        url = f"{self.finnhub_url}/news"
        params = {
            "category": category,
            "token": self.finnhub_key
        }
        
        if min_id:
            params["minId"] = min_id
            
        data = await self._make_request(url, params, DataProvider.FINNHUB)
        
        news_items = []
        for item in data:
            news_items.append(NewsItem(
                id=str(item["id"]),
                headline=item["headline"],
                summary=item["summary"],
                source=item["source"],
                datetime=datetime.fromtimestamp(item["datetime"], tz=timezone.utc),
                related_symbols=item.get("related", "").split(",") if item.get("related") else [],
                url=item.get("url"),
                category=category,
                provider=DataProvider.FINNHUB
            ))
            
        return news_items
        
    async def get_company_news_finnhub(self, symbol: str, from_date: str, to_date: str) -> List[NewsItem]:
        """Get company-specific news from Finnhub"""
        url = f"{self.finnhub_url}/company-news"
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": self.finnhub_key
        }
        
        data = await self._make_request(url, params, DataProvider.FINNHUB)
        
        news_items = []
        for item in data:
            news_items.append(NewsItem(
                id=str(item["id"]),
                headline=item["headline"],
                summary=item["summary"],
                source=item["source"],
                datetime=datetime.fromtimestamp(item["datetime"], tz=timezone.utc),
                related_symbols=[symbol],
                url=item.get("url"),
                category="company",
                provider=DataProvider.FINNHUB
            ))
            
        return news_items
        
    async def get_earnings_calendar_finnhub(self, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Get earnings calendar from Finnhub"""
        url = f"{self.finnhub_url}/calendar/earnings"
        params = {
            "from": from_date,
            "to": to_date,
            "token": self.finnhub_key
        }
        
        data = await self._make_request(url, params, DataProvider.FINNHUB)
        return data.get("earningsCalendar", [])
        
    async def get_sentiment_finnhub(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment from Finnhub"""
        url = f"{self.finnhub_url}/news-sentiment"
        params = {
            "symbol": symbol,
            "token": self.finnhub_key
        }
        
        return await self._make_request(url, params, DataProvider.FINNHUB)
        
    # ================== TWELVE DATA METHODS ==================
    
    async def get_real_time_price_td(self, symbol: str) -> Dict[str, Any]:
        """Get real-time price from Twelve Data"""
        url = f"{self.twelve_data_url}/price"
        params = {
            "symbol": symbol,
            "apikey": self.twelve_data_key
        }
        
        return await self._make_request(url, params, DataProvider.TWELVE_DATA)
        
    async def get_time_series_td(self, symbol: str, interval: str = "1min", outputsize: int = 100) -> List[MarketData]:
        """Get time series data from Twelve Data"""
        url = f"{self.twelve_data_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.twelve_data_key
        }
        
        data = await self._make_request(url, params, DataProvider.TWELVE_DATA)
        
        market_data = []
        if "values" in data:
            for item in data["values"]:
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(item["datetime"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=int(item["volume"]) if item["volume"] else 0,
                    provider=DataProvider.TWELVE_DATA,
                    metadata=data.get("meta", {})
                ))
                
        return market_data
        
    async def get_technical_indicators_td(self, symbol: str, indicator: str, interval: str = "1min") -> Dict[str, Any]:
        """Get technical indicators from Twelve Data"""
        url = f"{self.twelve_data_url}/{indicator.lower()}"
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.twelve_data_key
        }
        
        return await self._make_request(url, params, DataProvider.TWELVE_DATA)
        
    async def get_forex_pairs_td(self) -> List[Dict[str, Any]]:
        """Get available forex pairs from Twelve Data"""
        url = f"{self.twelve_data_url}/forex_pairs"
        params = {
            "apikey": self.twelve_data_key
        }
        
        data = await self._make_request(url, params, DataProvider.TWELVE_DATA)
        return data.get("data", [])
        
    # ================== UNIFIED METHODS ==================
    
    async def get_comprehensive_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data from all providers"""
        results = {}
        
        try:
            # Alpha Vantage - Intraday data
            av_data = await self.get_intraday_data_av(symbol)
            results["alpha_vantage_intraday"] = av_data[:10]  # Last 10 data points
            
            # Finnhub - Real-time quote
            fh_quote = await self.get_quote_finnhub(symbol)
            results["finnhub_quote"] = fh_quote
            
            # Twelve Data - Real-time price
            td_price = await self.get_real_time_price_td(symbol)
            results["twelve_data_price"] = td_price
            
        except Exception as e:
            logger.error(f"Error getting comprehensive data for {symbol}: {e}")
            
        return results
        
    async def get_market_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis"""
        sentiment_data = {}
        
        try:
            # Finnhub sentiment
            fh_sentiment = await self.get_sentiment_finnhub(symbol)
            sentiment_data["finnhub_sentiment"] = fh_sentiment
            
            # Recent news for sentiment analysis
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            company_news = await self.get_company_news_finnhub(symbol, yesterday, today)
            sentiment_data["recent_news"] = company_news
            
            # Calculate aggregate sentiment
            if company_news:
                sentiment_data["news_count"] = len(company_news)
                sentiment_data["latest_news"] = company_news[0].headline if company_news else None
                
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            
        return sentiment_data
        
    async def get_economic_context(self) -> Dict[str, Any]:
        """Get economic context data"""
        economic_data = {}
        
        try:
            # Get key economic indicators
            gdp_data = await self.get_economic_indicators_av("GDP")
            economic_data["gdp"] = gdp_data[:5] if gdp_data else []
            
            # Get earnings calendar for next week
            today = datetime.now().strftime("%Y-%m-%d")
            next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            
            earnings = await self.get_earnings_calendar_finnhub(today, next_week)
            economic_data["upcoming_earnings"] = earnings[:10]  # Next 10 earnings
            
        except Exception as e:
            logger.error(f"Error getting economic context: {e}")
            
        return economic_data
        
    async def get_technical_analysis_suite(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive technical analysis"""
        technical_data = {}
        
        try:
            # Get multiple technical indicators from Twelve Data
            indicators = ["rsi", "macd", "bbands", "sma", "ema"]
            
            for indicator in indicators:
                try:
                    data = await self.get_technical_indicators_td(symbol, indicator)
                    technical_data[indicator] = data
                except Exception as e:
                    logger.warning(f"Failed to get {indicator} for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error getting technical analysis for {symbol}: {e}")
            
        return technical_data
        
    async def generate_trading_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive trading intelligence report"""
        logger.info(f"Generating trading intelligence for {symbol}")
        
        intelligence = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_sources": ["Alpha Vantage", "Finnhub", "Twelve Data"]
        }
        
        try:
            # Get all data in parallel
            market_data_task = self.get_comprehensive_market_data(symbol)
            sentiment_task = self.get_market_sentiment_analysis(symbol)
            technical_task = self.get_technical_analysis_suite(symbol)
            economic_task = self.get_economic_context()
            
            # Wait for all tasks
            market_data, sentiment_data, technical_data, economic_data = await asyncio.gather(
                market_data_task, sentiment_task, technical_task, economic_task,
                return_exceptions=True
            )
            
            # Compile results
            intelligence.update({
                "market_data": market_data if not isinstance(market_data, Exception) else {},
                "sentiment_analysis": sentiment_data if not isinstance(sentiment_data, Exception) else {},
                "technical_analysis": technical_data if not isinstance(technical_data, Exception) else {},
                "economic_context": economic_data if not isinstance(economic_data, Exception) else {}
            })
            
            # Generate summary
            intelligence["summary"] = self._generate_intelligence_summary(intelligence)
            
        except Exception as e:
            logger.error(f"Error generating trading intelligence: {e}")
            intelligence["error"] = str(e)
            
        return intelligence
        
    def _generate_intelligence_summary(self, intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary from intelligence data"""
        summary = {
            "data_quality": "high",
            "signal_strength": "medium",
            "risk_level": "medium",
            "recommendation": "hold"
        }
        
        try:
            # Analyze market data
            market_data = intelligence.get("market_data", {})
            if "finnhub_quote" in market_data:
                quote = market_data["finnhub_quote"]
                if "c" in quote and "pc" in quote:  # current and previous close
                    change_pct = ((quote["c"] - quote["pc"]) / quote["pc"]) * 100
                    summary["price_change_pct"] = round(change_pct, 2)
                    
                    if abs(change_pct) > 5:
                        summary["signal_strength"] = "high"
                    elif abs(change_pct) > 2:
                        summary["signal_strength"] = "medium"
                    else:
                        summary["signal_strength"] = "low"
                        
            # Analyze sentiment
            sentiment_data = intelligence.get("sentiment_analysis", {})
            if "recent_news" in sentiment_data:
                news_count = sentiment_data.get("news_count", 0)
                if news_count > 5:
                    summary["news_activity"] = "high"
                elif news_count > 2:
                    summary["news_activity"] = "medium"
                else:
                    summary["news_activity"] = "low"
                    
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            
        return summary


# Example usage and testing
async def test_premium_data_manager():
    """Test the premium data manager"""
    async with PremiumDataManager() as manager:
        # Test comprehensive intelligence for a stock
        intelligence = await manager.generate_trading_intelligence("AAPL")
        
        logger.info("=== TRADING INTELLIGENCE REPORT ===")
        logger.info(f"Symbol: {intelligence['symbol']}")
        logger.info(f"Timestamp: {intelligence['timestamp']}")
        logger.info(f"Data Sources: {', '.join(intelligence['data_sources'])}")
        
        if "summary" in intelligence:
            summary = intelligence["summary"]
            logger.info(f"\nSUMMARY:")
            logger.info(f"  Signal Strength: {summary.get('signal_strength', 'unknown')}")
            logger.info(f"  Price Change: {summary.get('price_change_pct', 'N/A')}")
            logger.info(f"  News Activity: {summary.get('news_activity', 'unknown')}")
            logger.info(f"  Recommendation: {summary.get('recommendation', 'hold')}")
            
        # Test crypto data
        crypto_data = await manager.get_comprehensive_market_data("BTC-USD")
        logger.info(f"\nBTC Data Sources: {list(crypto_data.keys())}")
        
        return intelligence


if __name__ == "__main__":
    asyncio.run(test_premium_data_manager()) 