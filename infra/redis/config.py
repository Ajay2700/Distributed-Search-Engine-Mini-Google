"""Redis configuration and connection factory for the adaptive cache layer."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_redis_client(redis_url: str = "redis://localhost:6379/0"):
    """Create a Redis client. Returns None if Redis is unavailable."""
    try:
        import redis
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        logger.info(f"Redis connected: {redis_url}")
        return client
    except ImportError:
        logger.warning("redis package not installed — using in-memory cache only")
        return None
    except Exception as e:
        logger.warning(f"Redis connection failed ({redis_url}): {e} — falling back to in-memory")
        return None
