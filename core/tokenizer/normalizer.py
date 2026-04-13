"""Text normalization pipeline for pre-tokenization processing."""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Optional


class TextNormalizer:
    """Multi-stage text normalizer: decode HTML entities, strip tags,
    normalize unicode, collapse whitespace, and optionally apply stemming."""

    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    _MULTI_SPACE_RE = re.compile(r"\s+")
    _NON_ALPHA_RE = re.compile(r"[^a-z0-9\s\-]")

    def __init__(self, *, lowercase: bool = True, strip_html: bool = True,
                 strip_urls: bool = True, strip_accents: bool = True):
        self.lowercase = lowercase
        self.strip_html = strip_html
        self.strip_urls = strip_urls
        self.strip_accents = strip_accents

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        if self.strip_html:
            text = html.unescape(text)
            text = self._HTML_TAG_RE.sub(" ", text)
        if self.strip_urls:
            text = self._URL_RE.sub(" ", text)
        if self.lowercase:
            text = text.lower()
        if self.strip_accents:
            text = self._remove_accents(text)
        text = self._NON_ALPHA_RE.sub(" ", text)
        text = self._MULTI_SPACE_RE.sub(" ", text).strip()
        return text

    @staticmethod
    def _remove_accents(text: str) -> str:
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

    def extract_title(self, raw_html: str) -> str:
        """Extract <title> content from raw HTML."""
        m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
        if m:
            return html.unescape(m.group(1)).strip()
        return ""

    def extract_text(self, raw_html: str) -> str:
        """Strip all HTML tags and return clean text."""
        text = html.unescape(raw_html)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = self._HTML_TAG_RE.sub(" ", text)
        text = self._MULTI_SPACE_RE.sub(" ", text).strip()
        return text
