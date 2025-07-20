import requests
from typing import Dict
import sys
from pathlib import Path

# Add the backend directory to the sys.path to allow for engine imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .base import BaseConnector

class CoinbaseConnector(BaseConnector):
    """
    A concrete connector for the Coinbase exchange.

    This class implements the BaseConnector interface to fetch live market
    data from Coinbase's public API endpoints.
    """

    def __init__(self):
        self.api_url = "https://api.coinbase.com/v2"

    def get_current_price(self, ticker: str) -> float:
        """
        Fetches the current spot price for a given ticker from Coinbase.

        Args:
            ticker: The ticker symbol (e.g., 'BTC-USD', 'ETH-EUR').

        Returns:
            The current spot price as a float.

        Raises:
            Exception: If the API call fails or the ticker is not found.
        """
        url = f"{self.api_url}/prices/{ticker}/spot"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            return float(data["data"]["amount"])
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - Ticker '{ticker}' may be invalid.")
            raise
        except Exception as err:
            print(f"An error occurred: {err}")
            raise

    def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> list:
        """
        Fetches historical market data.
        
        Note: This is a placeholder for future implementation.
        """
        print("Historical data fetching is not yet implemented.")
        # This will be implemented in a future step
        return super().get_historical_data(ticker, start_date, end_date) 