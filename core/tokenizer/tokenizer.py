"""High-performance tokenizer with normalization, stopword removal, and stemming."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .normalizer import TextNormalizer
from .stopwords import STOPWORDS


@dataclass
class TokenInfo:
    term: str
    position: int
    start_char: int
    end_char: int


class PorterStemmerLight:
    """Minimal Porter stemmer covering the most common English suffixes.
    We avoid NLTK dependency by implementing the core rules."""

    _SUFFIX_RULES = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("alism", "al"),
        ("ation", "ate"), ("ator", "ate"), ("ness", ""),
        ("ment", ""), ("ful", ""), ("ous", ""),
        ("ive", ""), ("ize", ""), ("ing", ""),
        ("ies", "y"), ("ness", ""), ("ment", ""),
        ("ally", "al"), ("ently", "ent"), ("eli", "e"),
        ("ousli", "ous"), ("ed", ""), ("ly", ""),
        ("er", ""), ("es", "e"), ("s", ""),
    ]

    def stem(self, word: str) -> str:
        if len(word) <= 3:
            return word
        for suffix, replacement in self._SUFFIX_RULES:
            if word.endswith(suffix):
                candidate = word[: -len(suffix)] + replacement
                if len(candidate) >= 3:
                    return candidate
        return word


class Tokenizer:
    """Full-pipeline tokenizer: normalize → split → filter stopwords → stem.

    Returns both plain token lists and rich TokenInfo with positions for
    the inverted index positional postings.
    """

    _SPLIT_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

    def __init__(
        self,
        *,
        remove_stopwords: bool = True,
        use_stemming: bool = True,
        min_token_len: int = 2,
        max_token_len: int = 40,
        normalizer: Optional[TextNormalizer] = None,
    ):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self._normalizer = normalizer or TextNormalizer()
        self._stemmer = PorterStemmerLight() if use_stemming else None

    def tokenize(self, text: str) -> List[str]:
        """Return list of processed tokens (no positional info)."""
        return [t.term for t in self.tokenize_with_positions(text)]

    def tokenize_with_positions(self, text: str) -> List[TokenInfo]:
        normalized = self._normalizer.normalize(text)
        tokens: List[TokenInfo] = []
        position = 0
        for m in self._SPLIT_RE.finditer(normalized):
            raw = m.group()
            if len(raw) < self.min_token_len or len(raw) > self.max_token_len:
                continue
            if self.remove_stopwords and raw in STOPWORDS:
                continue
            term = self._stemmer.stem(raw) if self._stemmer else raw
            if len(term) < self.min_token_len:
                continue
            tokens.append(TokenInfo(
                term=term,
                position=position,
                start_char=m.start(),
                end_char=m.end(),
            ))
            position += 1
        return tokens

    def term_frequencies(self, text: str) -> Dict[str, Tuple[int, List[int]]]:
        """Return {term: (frequency, [positions])} for a document."""
        tf: Dict[str, Tuple[int, List[int]]] = {}
        for tok in self.tokenize_with_positions(text):
            if tok.term in tf:
                freq, positions = tf[tok.term]
                positions.append(tok.position)
                tf[tok.term] = (freq + 1, positions)
            else:
                tf[tok.term] = (1, [tok.position])
        return tf
