"""Adaptive LFU/LRU hybrid cache with auto-invalidation on index updates.

Combines the strengths of LFU (keeps frequently accessed items) and LRU
(handles recency). Items start in LRU mode and graduate to LFU after
reaching a frequency threshold.

Supports optional Redis backend for distributed caching.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    key: str
    value: Any
    frequency: int = 1
    last_access: float = 0.0
    created_at: float = 0.0
    ttl: float = 3600.0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class AdaptiveCache:
    """Hybrid LFU/LRU cache with TTL and optional Redis backend.

    Eviction policy:
    1. Expired entries are removed first
    2. Entries with frequency < threshold are evicted LRU-style
    3. Remaining entries are evicted LFU-style (lowest frequency first)

    Auto-invalidation: when index_version changes, all cached results
    are marked stale.
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 3600,
        frequency_threshold: int = 3,
        redis_client=None,
        redis_prefix: str = "search_cache:",
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.frequency_threshold = frequency_threshold
        self._redis = redis_client
        self._redis_prefix = redis_prefix

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._index_version: int = 0

        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "invalidations": 0}

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value. Returns None on miss."""
        cache_key = self._make_key(key)

        if self._redis:
            return self._redis_get(cache_key)

        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if entry.is_expired:
                del self._cache[cache_key]
                self._stats["misses"] += 1
                return None

            entry.frequency += 1
            entry.last_access = time.time()
            self._cache.move_to_end(cache_key)
            self._stats["hits"] += 1
            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Store a value in cache."""
        cache_key = self._make_key(key)
        ttl = ttl or self.ttl_seconds

        if self._redis:
            self._redis_put(cache_key, value, ttl)
            return

        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.value = value
                entry.frequency += 1
                entry.last_access = time.time()
                self._cache.move_to_end(cache_key)
            else:
                if len(self._cache) >= self.max_size:
                    self._evict()
                now = time.time()
                self._cache[cache_key] = CacheEntry(
                    key=cache_key,
                    value=value,
                    frequency=1,
                    last_access=now,
                    created_at=now,
                    ttl=ttl,
                )

    def invalidate(self, key: str):
        """Remove a specific key from cache."""
        cache_key = self._make_key(key)
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._stats["invalidations"] += 1

    def invalidate_all(self):
        """Clear all cached entries (triggered on index update)."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._index_version += 1
            self._stats["invalidations"] += count
            logger.info(f"Cache invalidated: {count} entries cleared, version={self._index_version}")

    def on_index_update(self):
        """Called when the index is updated to invalidate stale results."""
        self.invalidate_all()

    def _evict(self):
        """Hybrid eviction: remove expired → LRU low-freq → LFU."""
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]
            self._stats["evictions"] += 1

        if len(self._cache) < self.max_size:
            return

        low_freq = [
            (k, v) for k, v in self._cache.items()
            if v.frequency < self.frequency_threshold
        ]
        if low_freq:
            victim_key = low_freq[0][0]
            del self._cache[victim_key]
            self._stats["evictions"] += 1
            return

        lfu_key = min(self._cache, key=lambda k: (self._cache[k].frequency, self._cache[k].last_access))
        del self._cache[lfu_key]
        self._stats["evictions"] += 1

    def _make_key(self, key: str) -> str:
        return hashlib.md5(f"v{self._index_version}:{key}".encode()).hexdigest()

    def _redis_get(self, cache_key: str) -> Optional[Any]:
        try:
            data = self._redis.get(self._redis_prefix + cache_key)
            if data:
                self._stats["hits"] += 1
                return json.loads(data)
            self._stats["misses"] += 1
            return None
        except Exception as e:
            logger.warning(f"Redis GET error: {e}")
            return None

    def _redis_put(self, cache_key: str, value: Any, ttl: float):
        try:
            self._redis.setex(
                self._redis_prefix + cache_key,
                int(ttl),
                json.dumps(value, default=str),
            )
        except Exception as e:
            logger.warning(f"Redis SET error: {e}")

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0
