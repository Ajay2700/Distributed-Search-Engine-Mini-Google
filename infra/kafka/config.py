"""Kafka configuration for inter-service message passing.

Topics:
  - crawl.urls       : URLs to be crawled (Frontier → Crawler)
  - crawl.documents  : Crawled documents (Crawler → Indexer)
  - index.updates    : Index update notifications (Indexer → Cache Invalidator)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

TOPICS = {
    "CRAWL_URLS": "crawl.urls",
    "CRAWL_DOCUMENTS": "crawl.documents",
    "INDEX_UPDATES": "index.updates",
}

DEFAULT_BROKER = "localhost:9092"


class KafkaProducerWrapper:
    """Lazy-initialized Kafka producer with JSON serialization."""

    def __init__(self, broker: str = DEFAULT_BROKER):
        self._broker = broker
        self._producer = None

    def _init_producer(self):
        if self._producer is not None:
            return
        try:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self._broker,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                acks="all",
                retries=3,
            )
            logger.info(f"Kafka producer connected to {self._broker}")
        except Exception as e:
            logger.warning(f"Kafka producer init failed: {e}. Messages will be dropped.")

    def send(self, topic: str, value: Dict[str, Any]):
        self._init_producer()
        if self._producer:
            try:
                self._producer.send(topic, value=value)
                self._producer.flush()
            except Exception as e:
                logger.error(f"Kafka send error: {e}")

    def close(self):
        if self._producer:
            self._producer.close()


class KafkaConsumerWrapper:
    """Kafka consumer with callback-based message handling."""

    def __init__(self, topic: str, group_id: str, broker: str = DEFAULT_BROKER):
        self._topic = topic
        self._group_id = group_id
        self._broker = broker
        self._consumer = None

    def consume(self, handler: Callable[[Dict[str, Any]], None], max_messages: int = 0):
        try:
            from kafka import KafkaConsumer
            self._consumer = KafkaConsumer(
                self._topic,
                bootstrap_servers=self._broker,
                group_id=self._group_id,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
            )
            logger.info(f"Consuming from {self._topic} (group={self._group_id})")

            count = 0
            for message in self._consumer:
                handler(message.value)
                count += 1
                if max_messages > 0 and count >= max_messages:
                    break
        except Exception as e:
            logger.warning(f"Kafka consumer error: {e}")

    def close(self):
        if self._consumer:
            self._consumer.close()
