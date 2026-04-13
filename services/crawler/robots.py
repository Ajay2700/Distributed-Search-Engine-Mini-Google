"""robots.txt parser with per-host caching and crawl-delay extraction."""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class RobotsChecker:
    """Checks robots.txt rules before crawling a URL.
    
    Caches parsed rules per host to avoid repeated fetches.
    """

    def __init__(self, user_agent: str = "MiniGoogleBot", timeout: float = 5.0):
        self.user_agent = user_agent
        self.timeout = timeout
        self._cache: Dict[str, Tuple[Optional["_RobotRules"], float]] = {}
        self._cache_ttl = 3600

    def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        rules = self._get_rules(host)
        if rules is None:
            return True
        return rules.is_allowed(self.user_agent, parsed.path)

    def get_crawl_delay(self, url: str) -> float:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        rules = self._get_rules(host)
        if rules is None:
            return 0.0
        return rules.crawl_delay

    def _get_rules(self, host: str) -> Optional["_RobotRules"]:
        now = time.time()
        if host in self._cache:
            rules, fetched_at = self._cache[host]
            if now - fetched_at < self._cache_ttl:
                return rules

        try:
            robots_url = f"{host}/robots.txt"
            resp = httpx.get(robots_url, timeout=self.timeout, follow_redirects=True)
            if resp.status_code == 200:
                rules = _RobotRules.parse(resp.text)
                self._cache[host] = (rules, now)
                return rules
        except Exception as e:
            logger.debug(f"Could not fetch robots.txt for {host}: {e}")

        self._cache[host] = (None, now)
        return None


class _RobotRules:
    """Minimal robots.txt rule engine."""

    def __init__(self):
        self.disallow: list[str] = []
        self.allow: list[str] = []
        self.crawl_delay: float = 0.0

    @classmethod
    def parse(cls, text: str) -> "_RobotRules":
        rules = cls()
        current_applies = False

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            if ":" not in line:
                continue

            field, _, value = line.partition(":")
            field = field.strip().lower()
            value = value.strip()

            if field == "user-agent":
                current_applies = value == "*" or "minigoogle" in value.lower()
            elif current_applies:
                if field == "disallow" and value:
                    rules.disallow.append(value)
                elif field == "allow" and value:
                    rules.allow.append(value)
                elif field == "crawl-delay":
                    try:
                        rules.crawl_delay = float(value)
                    except ValueError:
                        pass

        return rules

    def is_allowed(self, user_agent: str, path: str) -> bool:
        for pattern in self.allow:
            if path.startswith(pattern):
                return True
        for pattern in self.disallow:
            if path.startswith(pattern):
                return False
        return True
