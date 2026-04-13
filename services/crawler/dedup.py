"""URL and content deduplication using hashing.

Uses MurmurHash3 for URL fingerprinting (fast, low collision)
and content hashing to detect near-duplicate pages.
"""
from __future__ import annotations

import hashlib
from typing import Set

import mmh3


class URLDeduplicator:
    """Fast URL deduplication using MurmurHash3 fingerprints.
    
    Stores 64-bit hashes instead of full URLs for memory efficiency.
    With 1M URLs, this uses ~8MB vs ~100MB+ for string storage.
    """

    def __init__(self):
        self._seen: Set[int] = set()

    def normalize_url(self, url: str) -> str:
        """Normalize URL for consistent deduplication."""
        url = url.rstrip("/")
        url = url.split("#")[0]
        url = url.split("?")[0] if "?" in url and self._is_tracking_param(url) else url
        return url.lower()

    def is_seen(self, url: str) -> bool:
        normalized = self.normalize_url(url)
        fingerprint = mmh3.hash64(normalized, signed=False)[0]
        return fingerprint in self._seen

    def mark_seen(self, url: str) -> bool:
        """Mark URL as seen. Returns True if it was already seen."""
        normalized = self.normalize_url(url)
        fingerprint = mmh3.hash64(normalized, signed=False)[0]
        if fingerprint in self._seen:
            return True
        self._seen.add(fingerprint)
        return False

    def __len__(self) -> int:
        return len(self._seen)

    @staticmethod
    def _is_tracking_param(url: str) -> bool:
        tracking_params = {"utm_source", "utm_medium", "utm_campaign", "fbclid", "gclid"}
        if "?" not in url:
            return False
        params = url.split("?")[1].split("&")
        return any(p.split("=")[0] in tracking_params for p in params)


class ContentDeduplicator:
    """Content-level deduplication using SHA-256 hashes.
    
    Detects exact-duplicate pages even at different URLs.
    """

    def __init__(self):
        self._hashes: Set[str] = set()

    def content_hash(self, text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        h = self.content_hash(text)
        return h in self._hashes

    def mark_seen(self, text: str) -> str:
        """Mark content as seen. Returns the hash."""
        h = self.content_hash(text)
        self._hashes.add(h)
        return h
