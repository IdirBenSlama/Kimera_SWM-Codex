from abc import ABC, abstractmethod

class BaseConnector(ABC):
    """
    An abstract base class for market data connectors.

    This class defines the standard interface that all exchange-specific
    connectors must implement, ensuring architectural consistency across the
    trading system.
    """

    @abstractmethod
    def get_current_price(self, ticker: str) -> float:
        """
        Fetches the current market price for a given ticker symbol.

        Args:
            ticker: The ticker symbol to fetch the price for (e.g., 'BTC-USD').

        Returns:
            The current price as a float.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> list:
        """
        Fetches historical market data for a given ticker symbol.
        
        Note: This is a placeholder for future implementation.

        Args:
            ticker: The ticker symbol to fetch data for.
            start_date: The start date for the historical data.
            end_date: The end date for the historical data.

        Returns:
            A list of historical data points.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError 