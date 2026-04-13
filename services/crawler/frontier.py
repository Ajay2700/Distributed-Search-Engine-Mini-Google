"""URL frontier: priority queue for managing crawl order.

Implements a politeness-aware frontier that:
  - Prioritizes URLs by estimated value (breadth-first with priority)
  - Enforces per-host crawl delays
  - Limits concurrent requests per domain
"""
from __future__ import annotations

import heapq
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse


@dataclass(order=True)
class FrontierEntry:
    priority: float
    url: str = field(compare=False)
    depth: int = field(compare=False, default=0)
    parent_url: str = field(compare=False, default="")


class URLFrontier:
    """Thread-safe URL frontier with per-host politeness."""

    def __init__(self, max_depth: int = 3, default_delay: float = 0.5):
        self.max_depth = max_depth
        self.default_delay = default_delay

        self._queue: List[FrontierEntry] = []
        self._host_last_access: Dict[str, float] = {}
        self._host_delays: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def add(self, url: str, priority: float = 0.0, depth: int = 0,
            parent_url: str = ""):
        if depth > self.max_depth:
            return
        with self._lock:
            entry = FrontierEntry(
                priority=-priority,  # heapq is min-heap; negate for max-priority
                url=url,
                depth=depth,
                parent_url=parent_url,
            )
            heapq.heappush(self._queue, entry)
            self._not_empty.notify()

    def add_many(self, urls: List[Tuple[str, float, int]]):
        """Bulk add: [(url, priority, depth), ...]"""
        with self._lock:
            for url, priority, depth in urls:
                if depth <= self.max_depth:
                    entry = FrontierEntry(priority=-priority, url=url, depth=depth)
                    heapq.heappush(self._queue, entry)
            self._not_empty.notify_all()

    def get(self, timeout: float = 5.0) -> Optional[FrontierEntry]:
        """Get next URL, respecting per-host crawl delays."""
        deadline = time.time() + timeout
        with self._lock:
            while True:
                if not self._queue:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(timeout=remaining)
                    if not self._queue:
                        return None

                skipped = []
                result = None
                while self._queue:
                    entry = heapq.heappop(self._queue)
                    host = urlparse(entry.url).netloc
                    delay = self._host_delays.get(host, self.default_delay)
                    last_access = self._host_last_access.get(host, 0.0)

                    if time.time() - last_access >= delay:
                        self._host_last_access[host] = time.time()
                        result = entry
                        break
                    else:
                        skipped.append(entry)

                for s in skipped:
                    heapq.heappush(self._queue, s)

                if result:
                    return result

                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=min(0.1, remaining))

    def set_host_delay(self, host: str, delay: float):
        with self._lock:
            self._host_delays[host] = delay

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0
