from confluent_kafka import Producer, Consumer, KafkaException
import json

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


class KafkaManager:
    """
    Manages Kafka connections for producing and consuming data streams.
    This class will handle:
    - Connection to the Kafka cluster.
    - Producing messages to various topics (e.g., raw_market_data, news, sentiment).
    - Consuming messages from topics for processing by different parts of the system.
    """

    def __init__(self, bootstrap_servers='localhost:9092'):
        """
        Initializes the KafkaManager.

        Args:
            bootstrap_servers (str): Comma-separated list of Kafka brokers.
        """
        self.producer_config = {'bootstrap.servers': bootstrap_servers}
        self.consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': 'kimera_trading_group',
            'auto.offset.reset': 'earliest'
        }
        self.producer = None
        self.consumer = None

    def connect_producer(self):
        """
        Initializes the Kafka producer.
        """
        try:
            self.producer = Producer(self.producer_config)
            logger.info("Kafka producer connected successfully.")
        except KafkaException as e:
            logger.error(f"Failed to connect Kafka producer: {e}")

    def connect_consumer(self, topics):
        """
        Initializes the Kafka consumer and subscribes to topics.
        
        Args:
            topics (list): A list of topic names to subscribe to.
        """
        try:
            self.consumer = Consumer(self.consumer_config)
            self.consumer.subscribe(topics)
            logger.info(f"Kafka consumer connected and subscribed to topics: {topics}")
        except KafkaException as e:
            logger.error(f"Failed to connect Kafka consumer: {e}")

    def produce_message(self, topic, message):
        """
        Sends a message to a Kafka topic.

        Args:
            topic (str): The topic to send the message to.
            message (dict): The message payload (will be JSON serialized).
        """
        if not self.producer:
            logger.info("Producer is not connected.")
            return

        def delivery_report(err, msg):
            """ Called once for each message produced to indicate delivery result. """
            if err is not None:
                logger.error(f'Message delivery failed: {err}')
            else:
                logger.info(f'Message delivered to {msg.topic()}')

        try:
            self.producer.produce(topic, json.dumps(message).encode('utf-8'), callback=delivery_report)
            self.producer.poll(0)
        except BufferError:
             logger.info(f'Local producer queue is full ({len(self.producer)})')
        except Exception as e:
            logger.error(f"Failed to produce message: {e}")
            
    def consume_messages(self, timeout=1.0):
        """
        Consumes messages from the subscribed topics.
        This is a generator that yields messages as they are received.
        """
        if not self.consumer:
            logger.info("Consumer is not connected.")
            return

        while True:
            msg = self.consumer.poll(timeout)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue
            
            yield json.loads(msg.value().decode('utf-8'))


    def close(self):
        """
        Closes the producer and consumer connections.
        """
        if self.producer:
            self.producer.flush()
        if self.consumer:
            self.consumer.close()
        logger.info("Kafka connections closed.")


if __name__ == '__main__':
    # Example Usage
    kafka_manager = KafkaManager()
    
    # Producer example
    kafka_manager.connect_producer()
    if kafka_manager.producer:
        sample_market_data = {'symbol': 'ADA-USD', 'price': 0.45, 'timestamp': '2023-10-27T10:00:00Z'}
        kafka_manager.produce_message('raw_market_data', sample_market_data)
        kafka_manager.producer.flush()

    # Consumer example
    kafka_manager.connect_consumer(topics=['raw_market_data'])
    if kafka_manager.consumer:
        try:
            for message in kafka_manager.consume_messages():
                logger.info("Consumed message:", message)
                # In a real application, processing would happen here.
                # For this example, we break after the first message.
                break 
        finally:
            kafka_manager.close() 