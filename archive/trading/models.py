from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Order:
    """Represents an intent to trade a certain quantity of an asset."""
    ticker: str
    side: Literal['buy', 'sell']
    quantity: float

@dataclass
class Position:
    """Represents a holding of a certain quantity of an asset."""
    ticker: str
    quantity: float
    average_entry_price: float
    
    def get_current_value(self, current_price: float) -> float:
        """Calculates the current market value of the position."""
        return self.quantity * current_price 