import sys
from pathlib import Path
from typing import Dict

# Add the backend directory to the sys.path to allow for engine imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .models import Order
from .portfolio import Portfolio

class RiskManager:
    """
    Evaluates potential trades against pre-defined risk rules.
    """
    def __init__(self, portfolio: Portfolio, max_position_pct: float = 0.2):
        """
        Initializes the RiskManager.

        Args:
            portfolio: The portfolio instance to manage risk for.
            max_position_pct: The maximum percentage of total portfolio equity
                              that a single position can represent.
        """
        self.portfolio = portfolio
        self.max_position_pct = max_position_pct
        print(f"RiskManager initialized with max position risk of {self.max_position_pct:.0%}.")

    def check_order(self, order: Order, price: float) -> bool:
        """
        Checks if a potential order is permissible under current risk rules.

        Args:
            order: The order to be checked.
            price: The current price of the asset.

        Returns:
            True if the order is permissible, False otherwise.
        """
        order_cost = order.quantity * price
        
        # Rule 1: Insufficient Funds
        if order.side == 'buy' and order_cost > self.portfolio.cash:
            print(f"RISK CHECK FAILED: Insufficient funds for {order.ticker} buy order. "
                  f"Required: ${order_cost:,.2f}, Available: ${self.portfolio.cash:,.2f}")
            return False

        # Rule 2: Max Position Size
        # A simplified check for total portfolio value, assuming we have the current price
        # for all positions. In a real scenario, this would need a live price feed.
        # For this check, we'll use the initial cash as a proxy for total value
        # to avoid needing a live feed for all positions during a simple check.
        portfolio_value_proxy = self.portfolio.cash
        if order.ticker in self.portfolio.positions:
            portfolio_value_proxy += self.portfolio.positions[order.ticker].get_current_value(price)
            
        max_position_value = portfolio_value_proxy * self.max_position_pct
        
        current_position_value = 0
        if order.ticker in self.portfolio.positions:
            current_position_value = self.portfolio.positions[order.ticker].get_current_value(price)
            
        projected_position_value = current_position_value + order_cost

        if order.side == 'buy' and projected_position_value > max_position_value:
            print(f"RISK CHECK FAILED: Order for {order.ticker} exceeds max position size of {self.max_position_pct:.0%}. "
                  f"Projected value: ${projected_position_value:,.2f}, Max allowed: ${max_position_value:,.2f}")
            return False

        print("RISK CHECK PASSED: Order is within defined risk limits.")
        return True 