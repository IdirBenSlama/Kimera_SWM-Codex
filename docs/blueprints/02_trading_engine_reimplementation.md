# Blueprint: Trading Engine Re-Implementation

**Document ID:** `bp_trading_01`
**Phase:** 2 (Financial Systems Re-Implementation)
**Status:** In Progress
**Date:** 2024-07-22

---

## 1.0 Objective

This document outlines the plan to re-implement the core trading and financial market analysis capabilities of the Kimera SWM. This phase builds upon the validated scientific core from Phase 1. The initial focus is on establishing robust, real-world market data connectivity.

The implementation will be a ground-up rebuild based on the conceptual blueprints reverse-engineered from the non-functional scripts in the `archive/broken_scripts_and_tests/trading/` directory, particularly `real_coinbase_cdp_connector.py`.

## 2.0 Architectural Placement

The new trading components will be organized within the `backend/trading/` module.

*   **Base Connector Interface:** `backend/trading/connectors/base.py`
*   **Coinbase Connector:** `backend/trading/connectors/coinbase.py`
*   **Trading Engine Core:** `backend/trading/engine.py`
*   **Validation Suite:** `tests/trading/test_coinbase_connector.py`

## 3.0 Core Components to Be Implemented

### 3.1 `BaseConnector` Abstract Base Class (`base.py`)

To ensure architectural consistency for future expansion to other exchanges, a `BaseConnector` abstract class will be created.

*   **Methods:**
    *   `get_current_price(self, ticker: str) -> float`: An abstract method to fetch the current market price for a given ticker symbol (e.g., 'BTC-USD').
    *   `get_historical_data(self, ticker: str, start_date: str, end_date: str) -> list`: An abstract method for fetching historical data (to be implemented in a later step).

### 3.2 `CoinbaseConnector` Class (`coinbase.py`)

This class will implement the `BaseConnector` interface for the Coinbase exchange.

*   **Inheritance:** Inherits from `BaseConnector`.
*   **Dependencies:** Uses the `requests` library for making direct HTTP calls to the public Coinbase API. This avoids complex authentication for initial price-fetching functionality.
*   **Method Implementation:**
    *   `get_current_price(self, ticker: str) -> float`:
        1.  Construct the URL for the Coinbase API's public products endpoint (e.g., `https://api.coinbase.com/v2/prices/{ticker}/spot`).
        2.  Make an HTTP GET request.
        3.  Parse the JSON response to extract the price.
        4.  Handle potential errors (e.g., invalid ticker, API downtime) gracefully.
        5.  Return the price as a float.

### 3.3 `TradingEngine` Class (`engine.py`)

A simple engine class to manage and utilize connectors.

*   **Initialization:** `__init__(self, connector: BaseConnector)`
*   **Functionality:** Will have methods that use the connector, such as `check_price(self, ticker: str)`. This provides the architectural skeleton for future strategy implementation.

### 3.4 Validation Suite (`test_coinbase_connector.py`)

A test suite to validate the `CoinbaseConnector` without requiring API keys or placing orders.

**Test Case: `test_get_live_btc_price`**

1.  **Setup:**
    *   Instantiate the `CoinbaseConnector`.
2.  **Execution:**
    *   Call `get_current_price('BTC-USD')`.
3.  **Validation (Assertions):**
    *   Assert that the returned price is a `float`.
    *   Assert that the price is a positive number.
    *   Assert that the price is within a plausible range (e.g., > 1000), confirming it is not a zero or error value.
    *   Log the retrieved price to the console.

## 4.0 Success Criteria for Phase 2 (Part 1)

This initial part of Phase 2 will be considered complete when the `test_get_live_btc_price` test case runs successfully, demonstrating live, real-world market data connectivity.

---

## **Part 2: Risk and Execution**

---

## 5.0 Objective (Part 2)

With data connectivity established, the objective of this part is to build the core components for managing trade positions, assessing risk, and executing (simulated) orders. This creates the internal machinery required for any trading strategy.

## 6.0 Architectural Placement (Part 2)

*   **Data Models:** `backend/trading/models.py` (for Order, Position, etc.)
*   **Portfolio Manager:** `backend/trading/portfolio.py`
*   **Risk Manager:** `backend/trading/risk.py`
*   **Execution Logic:** To be integrated into `backend/trading/engine.py`.
*   **Validation Suite:** `tests/trading/test_trading_engine.py`

## 7.0 Core Components to Be Implemented (Part 2)

### 7.1 Data Models (`models.py`)

To ensure data consistency, we will use `dataclasses`.

*   `Order`: Represents an intent to trade (e.g., `{ticker: 'BTC-USD', side: 'buy', quantity: 0.01}`).
*   `Position`: Represents an existing asset holding (e.g., `{ticker: 'BTC-USD', quantity: 0.01, average_entry_price: 100000}`).

### 7.2 `Portfolio` Class (`portfolio.py`)

Manages the state of our assets and positions.

*   **Attributes:**
    *   `cash`: The amount of cash available (e.g., starting with $100,000).
    *   `positions`: A dictionary mapping tickers to `Position` objects.
*   **Methods:**
    *   `update_position(order: Order, price: float)`: Modifies cash and positions based on a filled order.
    *   `get_total_value(current_prices: Dict[str, float]) -> float`: Calculates the total equity of the portfolio (cash + value of all positions).

### 7.3 `RiskManager` Class (`risk.py`)

Evaluates potential trades against pre-defined risk rules.

*   **Initialization:** `__init__(self, portfolio: Portfolio)`
*   **Methods:**
    *   `check_order(order: Order, price: float) -> bool`: Checks if an order is permissible.
        *   **Rule 1: Insufficient Funds:** Fails if the order cost (`quantity * price`) exceeds available cash.
        *   **Rule 2: Max Position Size:** Fails if the order would result in a position exceeding a defined percentage of the total portfolio value (e.g., 20%).

### 7.4 `TradingEngine` Enhancements (`engine.py`)

The engine will be upgraded to orchestrate the new components.

*   **New Attributes:** `portfolio`, `risk_manager`.
*   **New Method: `execute_order(order: Order)`**:
    1.  Fetches the current price for the order's ticker using its `connector`.
    2.  Passes the order to the `risk_manager` for validation.
    3.  If the risk check passes, it simulates the fill by calling `portfolio.update_position`.
    4.  If the risk check fails, it logs the rejection and reason.

### 7.5 Validation Suite (`test_trading_engine.py`)

A test suite to validate the integrated trading logic.

**Test Cases:**
1.  `test_execute_valid_order`:
    *   Simulate a buy order that is within risk limits.
    *   Assert that cash is debited correctly and a new position is created in the portfolio.
2.  `test_reject_insufficient_funds`:
    *   Simulate a buy order that costs more than the available cash.
    *   Assert that the order is rejected and that cash and positions remain unchanged.
3.  `test_reject_max_position_size`:
    *   Simulate a large buy order that, while affordable, would exceed the max position size rule.
    *   Assert that the order is rejected and that cash and positions remain unchanged.

## 8.0 Success Criteria for Phase 2 (Part 2)

This part of Phase 2 will be successful when all test cases in `test_trading_engine.py` pass, demonstrating a functional, risk-aware, (simulated) execution pipeline. 