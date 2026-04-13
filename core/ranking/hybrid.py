"""Hybrid ranking: combines BM25, PageRank, and semantic similarity scores.

Final score = α × BM25(q, d) + β × PageRank(d) + γ × Semantic(q, d)

The weights can be tuned per query or globally. Includes support for
personalized ranking based on user profiles.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from core import Posting, SearchResult, UserProfile
from core.ranking.bm25 import BM25Scorer
from core.ranking.pagerank import PageRank
from core.ranking.semantic import SemanticScorer

logger = logging.getLogger(__name__)


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    range_s = max_s - min_s
    if range_s == 0:
        return {k: 1.0 for k in scores}
    return {k: (v - min_s) / range_s for k, v in scores.items()}


class HybridRanker:
    """Combines multiple scoring signals into a final ranking.

    Supports:
    - BM25 (lexical relevance)
    - PageRank (authority/importance)
    - Semantic similarity (neural relevance)
    - Personalization boost (user profile)
    """

    def __init__(
        self,
        bm25: BM25Scorer,
        pagerank: Optional[PageRank] = None,
        semantic: Optional[SemanticScorer] = None,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.3,
    ):
        self.bm25 = bm25
        self.pagerank = pagerank
        self.semantic = semantic
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def rank(
        self,
        query_terms: List[str],
        query_text: str,
        candidate_postings: Dict[str, List[Posting]],
        top_k: int = 10,
        user_profile: Optional[UserProfile] = None,
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """Rank candidate documents using hybrid scoring.

        Returns: [(doc_id, final_score, {signal: score}), ...] sorted by final_score desc.
        """
        candidate_docs = set()
        for postings in candidate_postings.values():
            for p in postings:
                candidate_docs.add(p.doc_id)

        if not candidate_docs:
            return []

        bm25_scores: Dict[int, float] = {}
        avg_dl = self.bm25._index.avg_doc_length

        for doc_id in candidate_docs:
            term_freqs: Dict[str, int] = {}
            for term in query_terms:
                for p in candidate_postings.get(term, []):
                    if p.doc_id == doc_id:
                        term_freqs[term] = p.frequency
                        break
            doc_length = self.bm25._index.get_doc_length(doc_id)
            bm25_scores[doc_id] = self.bm25.score_document(
                query_terms, term_freqs, doc_length, avg_dl
            )

        pr_scores: Dict[int, float] = {}
        if self.pagerank:
            for doc_id in candidate_docs:
                pr_scores[doc_id] = self.pagerank.get_score(doc_id)

        sem_scores: Dict[int, float] = {}
        if self.semantic:
            sem_scores = self.semantic.score_candidates(query_text, list(candidate_docs))

        bm25_norm = _normalize_scores(bm25_scores)
        pr_norm = _normalize_scores(pr_scores) if pr_scores else {}
        sem_norm = _normalize_scores(sem_scores) if sem_scores else {}

        final_scores: List[Tuple[int, float, Dict[str, float]]] = []
        for doc_id in candidate_docs:
            bm25_s = bm25_norm.get(doc_id, 0.0)
            pr_s = pr_norm.get(doc_id, 0.0)
            sem_s = sem_norm.get(doc_id, 0.0)

            score = (self.alpha * bm25_s + self.beta * pr_s + self.gamma * sem_s)

            if user_profile:
                score += self._personalization_boost(doc_id, user_profile)

            breakdown = {
                "bm25": bm25_s,
                "pagerank": pr_s,
                "semantic": sem_s,
                "final": score,
            }
            final_scores.append((doc_id, score, breakdown))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]

    def _personalization_boost(self, doc_id: int, profile: UserProfile) -> float:
        """Apply a small boost for documents matching user preferences."""
        boost = 0.0
        if doc_id in profile.click_history:
            boost += 0.05
        return boost
