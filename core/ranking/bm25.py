"""BM25 (Okapi BM25) scoring — the gold standard for lexical retrieval.

BM25(q, d) = Σ IDF(t) × [ f(t,d) × (k1 + 1) ] / [ f(t,d) + k1 × (1 - b + b × |d|/avgdl) ]

Key advantages over TF-IDF:
  - Term frequency saturation (diminishing returns for repeated terms)
  - Document length normalization (penalizes verbose documents)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from core import Posting
from core.ranking.tfidf import IndexLike


class BM25Scorer:
    """BM25 scorer with configurable k1 and b parameters.
    
    k1 controls term frequency saturation (typical: 1.2-2.0)
    b controls document length normalization (typical: 0.75)
    """

    def __init__(self, index: IndexLike, k1: float = 1.5, b: float = 0.75):
        self._index = index
        self.k1 = k1
        self.b = b
        self._idf_cache: Dict[str, float] = {}

    def idf(self, term: str) -> float:
        """IDF with Robertson-Sparck-Jones weighting (smoothed)."""
        if term not in self._idf_cache:
            N = self._index.total_docs
            df = self._index.doc_frequency(term)
            self._idf_cache[term] = math.log(
                (N - df + 0.5) / (df + 0.5) + 1.0
            )
        return self._idf_cache[term]

    def score_term(self, term: str, frequency: int, doc_length: int,
                   avg_doc_length: float) -> float:
        idf_val = self.idf(term)
        tf_norm = (frequency * (self.k1 + 1)) / (
            frequency + self.k1 * (1 - self.b + self.b * doc_length / max(avg_doc_length, 1))
        )
        return idf_val * tf_norm

    def score_document(
        self,
        query_terms: List[str],
        term_frequencies: Dict[str, int],
        doc_length: int,
        avg_doc_length: float,
    ) -> float:
        score = 0.0
        for term in query_terms:
            freq = term_frequencies.get(term, 0)
            if freq > 0:
                score += self.score_term(term, freq, doc_length, avg_doc_length)
        return score

    def score_from_postings(
        self,
        query_terms: List[str],
        doc_id: int,
        postings_by_term: Dict[str, List[Posting]],
    ) -> float:
        """Score a document given pre-fetched postings per query term."""
        doc_length = self._index.get_doc_length(doc_id)
        avg_dl = getattr(self._index, 'avg_doc_length', 0)
        if callable(avg_dl):
            avg_dl = avg_dl
        else:
            avg_dl = avg_dl

        score = 0.0
        for term in query_terms:
            postings = postings_by_term.get(term, [])
            for p in postings:
                if p.doc_id == doc_id:
                    score += self.score_term(term, p.frequency, doc_length, avg_dl)
                    break
        return score

    def clear_cache(self):
        self._idf_cache.clear()
