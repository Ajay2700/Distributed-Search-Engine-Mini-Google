"""Query Understanding Layer: rewriting, spelling correction, and intent detection.

Novelty feature #2: Processes raw user queries before retrieval to improve
result quality through:
  1. Spelling correction (edit distance-based)
  2. Synonym expansion
  3. Query intent classification (navigational vs informational vs transactional)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class QueryIntent(Enum):
    NAVIGATIONAL = "navigational"    # User wants a specific website
    INFORMATIONAL = "informational"  # User wants to learn something
    TRANSACTIONAL = "transactional"  # User wants to do something


@dataclass
class UnderstandingResult:
    original_query: str
    rewritten_query: str
    expanded_terms: List[str]
    intent: QueryIntent
    corrections: Dict[str, str]  # original -> corrected
    confidence: float


SYNONYM_MAP: Dict[str, List[str]] = {
    "fast": ["quick", "rapid", "speedy"],
    "big": ["large", "huge", "enormous"],
    "small": ["tiny", "little", "compact"],
    "good": ["great", "excellent", "fine"],
    "bad": ["poor", "terrible", "awful"],
    "start": ["begin", "launch", "initiate"],
    "end": ["finish", "complete", "terminate"],
    "make": ["create", "build", "construct"],
    "show": ["display", "present", "demonstrate"],
    "find": ["search", "locate", "discover"],
    "help": ["assist", "support", "aid"],
    "use": ["utilize", "employ", "apply"],
    "error": ["bug", "issue", "problem", "fault"],
    "fix": ["repair", "resolve", "patch"],
    "run": ["execute", "launch", "start"],
    "install": ["setup", "deploy", "configure"],
    "remove": ["delete", "uninstall", "erase"],
    "update": ["upgrade", "refresh", "patch"],
    "download": ["fetch", "retrieve", "get"],
    "upload": ["send", "push", "transfer"],
}

_REVERSE_SYNONYMS: Dict[str, str] = {}
for canonical, syns in SYNONYM_MAP.items():
    for s in syns:
        _REVERSE_SYNONYMS[s] = canonical

NAV_PATTERNS = [
    r"^go to ",
    r"^open ",
    r"\.com$", r"\.org$", r"\.net$", r"\.io$",
    r"^www\.",
    r"^https?://",
    r" homepage$",
    r" website$",
    r" login$",
    r" sign in$",
]

TRANSACTIONAL_PATTERNS = [
    r"^buy ", r"^purchase ",
    r"^download ", r"^install ",
    r"^subscribe ", r"^sign up",
    r"^order ", r"^book ",
    r" price$", r" cost$",
    r" free$", r" coupon$",
]


class QueryUnderstanding:
    """Processes and enriches user queries for better retrieval."""

    def __init__(self, vocabulary: Optional[Set[str]] = None):
        self._vocabulary = vocabulary or set()
        self._nav_patterns = [re.compile(p, re.IGNORECASE) for p in NAV_PATTERNS]
        self._trans_patterns = [re.compile(p, re.IGNORECASE) for p in TRANSACTIONAL_PATTERNS]

    def set_vocabulary(self, vocab: Set[str]):
        """Set the vocabulary from the index for spelling correction."""
        self._vocabulary = vocab

    def process(self, query: str) -> UnderstandingResult:
        """Full query understanding pipeline."""
        query = query.strip()
        intent = self._detect_intent(query)
        corrections = self._spell_correct(query)

        corrected_query = query
        for original, corrected in corrections.items():
            corrected_query = corrected_query.replace(original, corrected)

        expanded = self._expand_synonyms(corrected_query)

        return UnderstandingResult(
            original_query=query,
            rewritten_query=corrected_query,
            expanded_terms=expanded,
            intent=intent,
            corrections=corrections,
            confidence=0.8 if not corrections else 0.6,
        )

    def _detect_intent(self, query: str) -> QueryIntent:
        for pattern in self._nav_patterns:
            if pattern.search(query):
                return QueryIntent.NAVIGATIONAL

        for pattern in self._trans_patterns:
            if pattern.search(query):
                return QueryIntent.TRANSACTIONAL

        return QueryIntent.INFORMATIONAL

    def _spell_correct(self, query: str) -> Dict[str, str]:
        """Simple edit-distance-based spelling correction against the index vocabulary."""
        if not self._vocabulary:
            return {}

        corrections: Dict[str, str] = {}
        words = query.lower().split()

        for word in words:
            if word in self._vocabulary:
                continue
            if len(word) <= 2:
                continue

            best_match = None
            best_distance = float("inf")

            for vocab_word in self._vocabulary:
                if abs(len(vocab_word) - len(word)) > 2:
                    continue
                dist = self._edit_distance(word, vocab_word)
                if dist < best_distance and dist <= 2:
                    best_distance = dist
                    best_match = vocab_word

            if best_match and best_match != word:
                corrections[word] = best_match

        return corrections

    def _expand_synonyms(self, query: str) -> List[str]:
        """Expand query with synonym terms."""
        words = query.lower().split()
        expanded: List[str] = list(words)

        for word in words:
            if word in SYNONYM_MAP:
                expanded.extend(SYNONYM_MAP[word][:2])
            elif word in _REVERSE_SYNONYMS:
                canonical = _REVERSE_SYNONYMS[word]
                expanded.append(canonical)

        return list(set(expanded))

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Wagner-Fischer algorithm for Levenshtein edit distance."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[m][n]
