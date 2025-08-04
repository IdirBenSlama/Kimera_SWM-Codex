import sys
from pathlib import Path

# Add the backend directory to the sys.path to allow for engine imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .connectors.base import BaseConnector
from .portfolio import Portfolio
from .risk_manager import RiskManager
from .models import Order
import logging
logger = logging.getLogger(__name__)

class TradingEngine:
    """
    The core engine for managing trading operations, strategies, and execution.

    This engine uses a specified market data connector to interact with
    exchanges and provides the framework for implementing and running
    trading strategies.
    """

    def __init__(self, connector: BaseConnector, portfolio: Portfolio, risk_manager: RiskManager):
        """
        Initializes the TradingEngine with a specific data connector.

        Args:
            connector: An instance of a class that implements BaseConnector.
            portfolio: The portfolio manager instance.
            risk_manager: The risk manager instance.
        """
        self.connector = connector
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        logger.info(f"TradingEngine initialized with connector: {type(connector).__name__}")

    def check_price(self, ticker: str) -> float:
        """
        A simple method to check the price of a ticker using the connector.

        Args:
            ticker: The ticker symbol to check (e.g., 'BTC-USD').

        Returns:
            The current price of the ticker as a float.
        """
        logger.info(f"Checking price for {ticker}...")
        price = self.connector.get_current_price(ticker)
        logger.info(f"Current price of {ticker} is ${price:,.2f}")
        return price

    def execute_order(self, order: Order):
        """
        Executes a trade order after validating it against risk rules.

        Args:
            order: The order to be executed.
        """
        logger.info(f"\n--- Received order: {order.side.upper()} {order.quantity} {order.ticker} ---")
        # 1. Get current price
        price = self.connector.get_current_price(order.ticker)

        # 2. Perform risk check
        if self.risk_manager.check_order(order, price):
            # 3. If risk check passes, update portfolio
            try:
                self.portfolio.update_position(order, price)
                logger.info(f"Order for {order.ticker} successfully executed.")
            except ValueError as e:
                logger.info(f"Execution failed after risk check: {e}")
        else:
            # 4. If risk check fails, log rejection
            logger.info(f"Order for {order.ticker} REJECTED due to risk constraints.") 