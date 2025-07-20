try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"yfinance not available: {e}. Market data functionality will be limited.")
    YFINANCE_AVAILABLE = False
    yf = None

import pandas as pd
from datetime import datetime

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


# It's better to import from the project's own modules
# from backend.trading.data.stream import KafkaManager

class YahooFinanceConnector:
    """
    A connector to fetch market data from Yahoo Finance.
    This class can be used to get historical data, real-time prices,
    and could be extended to fetch other information like company info, etc.
    """

    def __init__(self, kafka_manager=None):
        """
        Initializes the YahooFinanceConnector.

        Args:
            kafka_manager (KafkaManager, optional): An instance of the KafkaManager
                                                     to produce data to. Defaults to None.
        """
        self.kafka_manager = kafka_manager

    def get_historical_data(self, ticker, start_date, end_date, interval='1d'):
        """
        Fetches historical OHLCV data for a given ticker.

        Args:
            ticker (str): The stock/crypto ticker symbol (e.g., 'AAPL', 'BTC-USD').
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.
            interval (str, optional): The data interval. Defaults to '1d'. 
                                      Other options: '1m', '5m', '1h', etc.

        Returns:
            pandas.DataFrame: A DataFrame with the historical data, or None if failed.
        """
        if not YFINANCE_AVAILABLE:
            logger.warning(f"Cannot fetch historical data for {ticker}: yfinance not available")
            return None
            
        try:
            stock = yf.Ticker(ticker)
            hist_df = stock.history(start=start_date, end=end_date, interval=interval)
            logger.info(f"Successfully fetched historical data for {ticker}")
            return hist_df
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {ticker}: {e}")
            return None

    def stream_live_data(self, ticker, topic='raw_market_data'):
        """
        Fetches the latest price and streams it to Kafka.
        
        NOTE: yfinance is not a true real-time streaming API. This is a polling-based
        approach. For true low-latency streaming, a dedicated provider like Polygon.io
        or direct exchange feeds (e.g., via WebSocket) would be necessary.

        Args:
            ticker (str): The ticker symbol.
            topic (str): The Kafka topic to produce the data to.
        """
        if not YFINANCE_AVAILABLE:
            logger.warning(f"Cannot stream data for {ticker}: yfinance not available")
            return
            
        if not self.kafka_manager:
            logger.info("Kafka manager not configured. Cannot stream data.")
            return
            
        try:
            stock = yf.Ticker(ticker)
            # 'period' and 'interval' are used to get recent data.
            # Here we get the most recent 1-minute data available.
            data = stock.history(period="1d", interval="1m")
            if not data.empty:
                latest_data = data.iloc[-1]
                message = {
                    'symbol': ticker,
                    'price': latest_data['Close'],
                    'volume': latest_data['Volume'],
                    'timestamp': latest_data.name.isoformat()
                }
                logger.info(f"Streaming latest data for {ticker}: {message}")
                self.kafka_manager.produce_message(topic, message)
            else:
                logger.info(f"No recent data found for {ticker} to stream.")
                
        except Exception as e:
            logger.error(f"Failed to fetch or stream live data for {ticker}: {e}")

if __name__ == '__main__':
    # Example usage:
    yf_connector = YahooFinanceConnector()

    # 1. Get historical data
    btc_hist = yf_connector.get_historical_data('BTC-USD', '2023-01-01', '2023-10-01')
    if btc_hist is not None:
        logger.info("\nHistorical BTC-USD data (first 5 rows):")
        logger.info(btc_hist.head())

    # 2. Stream data (mock example, as it requires a running Kafka manager)
    logger.info("\n--- Streaming Example ---")
    # To run this part, you would need to instantiate and connect a KafkaManager
    # from backend.trading.data.stream import KafkaManager
    # kafka_manager = KafkaManager()
    # kafka_manager.connect_producer()
    # yf_connector_with_kafka = YahooFinanceConnector(kafka_manager=kafka_manager)
    # yf_connector_with_kafka.stream_live_data('AAPL')
    # kafka_manager.close()
    logger.info("Streaming example would require a running Kafka instance.")
    logger.info("Uncomment the code in __main__ to test with a live Kafka broker.")