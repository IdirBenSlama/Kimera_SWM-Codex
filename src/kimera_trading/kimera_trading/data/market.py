import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)
class DataFetcher:
    """Auto-generated class."""
    pass
    """Fetches real-time market data from various sources."""

    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"

    async def get_price_history(
        self, symbol: str, vs_currency: str = "eur", days: int = 30
    ) -> pd.DataFrame:
        """Fetches historical market data for a given symbol."""
        try:
            url = f"{self.base_url}/coins/{symbol.lower()}/market_chart"
            params = {"vs_currency": vs_currency, "days": days}
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()

            if "prices" in data:
                df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching price history for {symbol}: {e}")
            return pd.DataFrame()
