"""Ranking service: standalone ranking microservice interface.

Wraps the hybrid ranker with a service-oriented API for
inter-service communication. Supports dynamic weight tuning
and A/B testing of ranking configurations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core import Posting, UserProfile
from core.ranking.bm25 import BM25Scorer
from core.ranking.hybrid import HybridRanker
from core.ranking.pagerank import PageRank
from core.ranking.semantic import SemanticScorer

logger = logging.getLogger(__name__)


@dataclass
class RankingRequest:
    query_terms: List[str]
    query_text: str
    candidate_postings: Dict[str, List[Posting]]
    top_k: int = 10
    user_profile: Optional[UserProfile] = None
    ranking_config: Optional[Dict[str, float]] = None


@dataclass
class RankingResponse:
    ranked_docs: List[Tuple[int, float, Dict[str, float]]]
    config_used: Dict[str, float]


class RankingService:
    """Ranking microservice that wraps HybridRanker with configuration management."""

    def __init__(
        self,
        bm25: BM25Scorer,
        pagerank: Optional[PageRank] = None,
        semantic: Optional[SemanticScorer] = None,
        default_alpha: float = 0.5,
        default_beta: float = 0.2,
        default_gamma: float = 0.3,
    ):
        self._bm25 = bm25
        self._pagerank = pagerank
        self._semantic = semantic
        self._default_config = {
            "alpha": default_alpha,
            "beta": default_beta,
            "gamma": default_gamma,
        }
        self._ranker = HybridRanker(
            bm25=bm25,
            pagerank=pagerank,
            semantic=semantic,
            alpha=default_alpha,
            beta=default_beta,
            gamma=default_gamma,
        )

    def rank(self, request: RankingRequest) -> RankingResponse:
        """Process a ranking request."""
        config = request.ranking_config or self._default_config

        ranker = self._ranker
        if request.ranking_config:
            ranker = HybridRanker(
                bm25=self._bm25,
                pagerank=self._pagerank,
                semantic=self._semantic,
                alpha=config.get("alpha", self._default_config["alpha"]),
                beta=config.get("beta", self._default_config["beta"]),
                gamma=config.get("gamma", self._default_config["gamma"]),
            )

        ranked = ranker.rank(
            query_terms=request.query_terms,
            query_text=request.query_text,
            candidate_postings=request.candidate_postings,
            top_k=request.top_k,
            user_profile=request.user_profile,
        )

        return RankingResponse(ranked_docs=ranked, config_used=config)

    def update_weights(self, alpha: float, beta: float, gamma: float):
        """Update default ranking weights."""
        total = alpha + beta + gamma
        if total > 0:
            alpha /= total
            beta /= total
            gamma /= total

        self._default_config = {"alpha": alpha, "beta": beta, "gamma": gamma}
        self._ranker = HybridRanker(
            bm25=self._bm25,
            pagerank=self._pagerank,
            semantic=self._semantic,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        logger.info(f"Ranking weights updated: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")

    @property
    def current_config(self) -> Dict[str, float]:
        return dict(self._default_config)
