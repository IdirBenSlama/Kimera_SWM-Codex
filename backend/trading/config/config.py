"""Security configuration for the trading system.

Handles loading and validation of security-related configuration.
"""
import os
from dataclasses import dataclass
from typing import Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class SecurityConfig:
    """Container for security configuration parameters."""
    encryption_key: bytes
    encryption_algorithm: str = "AES256"
    key_rotation_days: int = 30

def get_security_config() -> SecurityConfig:
    """Load and validate security configuration from environment.
    
    Returns:
        SecurityConfig: Initialized configuration object
        
    Raises:
        RuntimeError: If required configuration is missing or invalid
    """
    key = os.getenv("TRADING_ENCRYPTION_KEY")
    if not key:
        logger.error("Missing TRADING_ENCRYPTION_KEY in environment")
        raise RuntimeError("Encryption key not configured")
        
    try:
        return SecurityConfig(
            encryption_key=key.encode(),
            encryption_algorithm=os.getenv("TRADING_ENCRYPTION_ALGO", "AES256"),
            key_rotation_days=int(os.getenv("TRADING_KEY_ROTATION_DAYS", "30"))
        )
    except ValueError as e:
        logger.error(f"Invalid security configuration: {str(e)}")
        raise RuntimeError("Invalid security configuration") from e

@dataclass
class NewsConfig:
    """Container for news API configuration parameters."""
    api_key: str
    base_url: str = "https://newsapi.org/v2"
    timeout_seconds: int = 10
    max_retries: int = 3

def get_news_config() -> NewsConfig:
    """Load and validate news API configuration from environment.
    
    Returns:
        NewsConfig: Initialized configuration object
        
    Raises:
        RuntimeError: If required configuration is missing or invalid
    """
    key = os.getenv("NEWS_API_KEY")
    if not key:
        logger.error("Missing NEWS_API_KEY in environment")
        raise RuntimeError("News API key not configured")
        
    try:
        return NewsConfig(
            api_key=key,
            base_url=os.getenv("NEWS_API_BASE_URL", "https://newsapi.org/v2"),
            timeout_seconds=int(os.getenv("NEWS_API_TIMEOUT", "10")),
            max_retries=int(os.getenv("NEWS_API_MAX_RETRIES", "3"))
        )
    except ValueError as e:
        logger.error(f"Invalid news configuration: {str(e)}")
        raise RuntimeError("Invalid news configuration") from e

@dataclass
class RedditConfig:
    """Placeholder for Reddit API configuration parameters. Replace with real implementation if needed."""
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "KimeraBot/0.1"
    username: Optional[str] = None
    password: Optional[str] = None
    subreddit: Optional[str] = None

def get_reddit_config() -> RedditConfig:
    """Stub for loading Reddit API configuration. Replace with real implementation if Reddit integration is required."""
    return RedditConfig()

@dataclass
class AlphaVantageConfig:
    """Placeholder for AlphaVantage API configuration parameters. Replace with real implementation if needed."""
    api_key: str = ""
    base_url: str = "https://www.alphavantage.co/query"
    timeout_seconds: int = 10
    max_retries: int = 3

def get_alpha_vantage_config() -> AlphaVantageConfig:
    """Stub for loading AlphaVantage API configuration. Replace with real implementation if AlphaVantage integration is required."""
    return AlphaVantageConfig()

@dataclass
class FredConfig:
    """Placeholder for FRED API configuration parameters. Replace with real implementation if needed."""
    api_key: str = ""
    base_url: str = "https://api.stlouisfed.org/fred/"
    timeout_seconds: int = 10
    max_retries: int = 3

def get_fred_config() -> FredConfig:
    """Stub for loading FRED API configuration. Replace with real implementation if FRED integration is required."""
    return FredConfig()