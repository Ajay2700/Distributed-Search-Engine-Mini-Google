"""Search result snippet generation with keyword highlighting.

Produces relevant snippets by finding the densest window of query terms
in the document and wrapping matches in highlight markers.
"""
from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple


class Highlighter:
    """Generates highlighted snippets for search results."""

    def __init__(
        self,
        snippet_length: int = 200,
        max_snippets: int = 3,
        highlight_pre: str = "**",
        highlight_post: str = "**",
    ):
        self.snippet_length = snippet_length
        self.max_snippets = max_snippets
        self.highlight_pre = highlight_pre
        self.highlight_post = highlight_post

    def generate_snippet(self, text: str, query_terms: List[str]) -> str:
        """Find the best snippet window and highlight query terms."""
        if not text or not query_terms:
            return text[:self.snippet_length] + "..." if len(text) > self.snippet_length else text

        best_window = self._find_best_window(text, query_terms)
        snippet = self._extract_window(text, best_window)
        highlighted = self._highlight_terms(snippet, query_terms)
        return highlighted

    def generate_snippets(self, text: str, query_terms: List[str]) -> List[str]:
        """Generate multiple non-overlapping snippets."""
        if not text or not query_terms:
            return [text[:self.snippet_length]]

        windows = self._find_top_windows(text, query_terms, self.max_snippets)
        snippets = []
        for start, end in windows:
            snippet = self._extract_window(text, (start, end))
            highlighted = self._highlight_terms(snippet, query_terms)
            snippets.append(highlighted)
        return snippets

    def _find_best_window(self, text: str, query_terms: List[str]) -> Tuple[int, int]:
        """Find the text window with highest density of query terms."""
        text_lower = text.lower()
        term_set = {t.lower() for t in query_terms}

        term_positions: List[Tuple[int, str]] = []
        for term in term_set:
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break
                term_positions.append((pos, term))
                start = pos + 1

        if not term_positions:
            return (0, min(self.snippet_length, len(text)))

        term_positions.sort()

        best_start = 0
        best_score = 0
        half = self.snippet_length // 2

        for pos, _ in term_positions:
            window_start = max(0, pos - half)
            window_end = min(len(text), window_start + self.snippet_length)

            score = 0
            unique_terms: Set[str] = set()
            for tp, term in term_positions:
                if window_start <= tp < window_end:
                    score += 1
                    unique_terms.add(term)

            score += len(unique_terms) * 2

            if score > best_score:
                best_score = score
                best_start = window_start

        return (best_start, min(best_start + self.snippet_length, len(text)))

    def _find_top_windows(self, text: str, query_terms: List[str],
                          n: int) -> List[Tuple[int, int]]:
        """Find top-n non-overlapping windows."""
        windows = []
        used_ranges: List[Tuple[int, int]] = []

        text_lower = text.lower()
        term_positions = []
        for term in query_terms:
            start = 0
            while True:
                pos = text_lower.find(term.lower(), start)
                if pos == -1:
                    break
                term_positions.append(pos)
                start = pos + 1

        if not term_positions:
            return [(0, min(self.snippet_length, len(text)))]

        term_positions.sort()

        for pos in term_positions:
            if len(windows) >= n:
                break

            start = max(0, pos - self.snippet_length // 4)
            end = min(len(text), start + self.snippet_length)

            overlaps = any(
                not (end <= us or start >= ue)
                for us, ue in used_ranges
            )
            if not overlaps:
                windows.append((start, end))
                used_ranges.append((start, end))

        if not windows:
            windows.append((0, min(self.snippet_length, len(text))))

        return windows

    def _extract_window(self, text: str, window: Tuple[int, int]) -> str:
        start, end = window
        snippet = text[start:end]

        if start > 0:
            space_idx = snippet.find(" ")
            if space_idx > 0 and space_idx < 20:
                snippet = "..." + snippet[space_idx + 1:]
            else:
                snippet = "..." + snippet

        if end < len(text):
            space_idx = snippet.rfind(" ")
            if space_idx > len(snippet) - 20:
                snippet = snippet[:space_idx] + "..."
            else:
                snippet = snippet + "..."

        return snippet

    def _highlight_terms(self, text: str, query_terms: List[str]) -> str:
        for term in query_terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            text = pattern.sub(
                f"{self.highlight_pre}\\g<0>{self.highlight_post}",
                text,
            )
        return text
