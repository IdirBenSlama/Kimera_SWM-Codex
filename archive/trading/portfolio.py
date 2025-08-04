from typing import Dict
import sys
from pathlib import Path

# Add the backend directory to the sys.path to allow for engine imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .models import Order, Position
import logging
logger = logging.getLogger(__name__)

class Portfolio:
    """
    Manages the state of assets, including cash and open positions.
    """
    def __init__(self, initial_cash: float = 100000.0):
        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        logger.info(f"Portfolio initialized with ${self.cash:,.2f} cash.")

    def update_position(self, order: Order, price: float):
        """
        Updates cash and positions based on a filled order.
        This is a simplified simulation that does not account for fees.
        """
        order_cost = order.quantity * price

        if order.side == 'buy':
            if self.cash < order_cost:
                raise ValueError("Insufficient funds to execute buy order.")
            
            self.cash -= order_cost
            if order.ticker in self.positions:
                # Update existing position
                existing_position = self.positions[order.ticker]
                new_total_quantity = existing_position.quantity + order.quantity
                new_total_cost = (existing_position.quantity * existing_position.average_entry_price) + order_cost
                existing_position.average_entry_price = new_total_cost / new_total_quantity
                existing_position.quantity = new_total_quantity
            else:
                # Create new position
                self.positions[order.ticker] = Position(
                    ticker=order.ticker,
                    quantity=order.quantity,
                    average_entry_price=price
                )
            logger.info(f"Executed BUY of {order.quantity} {order.ticker} @ ${price:,.2f}. New cash balance: ${self.cash:,.2f}")

        elif order.side == 'sell':
            if order.ticker not in self.positions or self.positions[order.ticker].quantity < order.quantity:
                raise ValueError("Insufficient position to execute sell order.")

            self.cash += order_cost
            position = self.positions[order.ticker]
            position.quantity -= order.quantity

            if position.quantity == 0:
                del self.positions[order.ticker]
            logger.info(f"Executed SELL of {order.quantity} {order.ticker} @ ${price:,.2f}. New cash balance: ${self.cash:,.2f}")


    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculates the total equity of the portfolio (cash + value of all positions).
        """
        positions_value = 0.0
        for ticker, position in self.positions.items():
            if ticker not in current_prices:
                raise ValueError(f"Current price for {ticker} not provided.")
            positions_value += position.get_current_value(current_prices[ticker])
        
        return self.cash + positions_value 