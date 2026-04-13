"""TF-IDF scoring implementation.

TF-IDF(t, d) = TF(t, d) × IDF(t)
where:
  TF(t, d)  = log(1 + freq(t, d))              [log-normalized]
  IDF(t)    = log(N / df(t))                    [inverse document frequency]
"""
from __future__ import annotations

import math
from typing import Dict, List, Protocol


class IndexLike(Protocol):
    """Protocol for index objects that provide stats needed for scoring."""
    def doc_frequency(self, term: str) -> int: ...
    def get_doc_length(self, doc_id: int) -> int: ...
    @property
    def total_docs(self) -> int: ...


class TFIDFScorer:
    def __init__(self, index: IndexLike):
        self._index = index
        self._idf_cache: Dict[str, float] = {}

    def idf(self, term: str) -> float:
        if term not in self._idf_cache:
            df = self._index.doc_frequency(term)
            N = self._index.total_docs
            self._idf_cache[term] = math.log((N + 1) / (df + 1)) + 1.0
        return self._idf_cache[term]

    def tf(self, frequency: int) -> float:
        return math.log(1 + frequency)

    def score_term(self, term: str, frequency: int) -> float:
        return self.tf(frequency) * self.idf(term)

    def score_document(self, query_terms: List[str],
                       term_frequencies: Dict[str, int]) -> float:
        total = 0.0
        for term in query_terms:
            freq = term_frequencies.get(term, 0)
            if freq > 0:
                total += self.score_term(term, freq)
        return total

    def clear_cache(self):
        self._idf_cache.clear()
