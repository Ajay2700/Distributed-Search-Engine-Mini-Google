"""Query Engine: the central query processing pipeline.

Orchestrates:
  1. Query understanding (rewriting, correction, intent)
  2. Query parsing
  3. Tokenization
  4. Index retrieval (distributed fan-out)
  5. Ranking (hybrid: BM25 + PageRank + semantic)
  6. Snippet generation with highlighting
  7. Result caching
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from config import Config
from core import Posting, SearchResult, UserProfile
from core.cache.adaptive_cache import AdaptiveCache
from core.ranking.bm25 import BM25Scorer
from core.ranking.hybrid import HybridRanker
from core.tokenizer import Tokenizer
from services.indexer import IndexerService
from services.query.highlighter import Highlighter
from services.query.parser import QueryParser
from services.query.understanding import QueryUnderstanding

logger = logging.getLogger(__name__)


class QueryEngine:
    """Full-featured query processing engine with caching and ranking."""

    def __init__(self, indexer: IndexerService, config: Config):
        self.indexer = indexer
        self.config = config

        self._tokenizer = Tokenizer()
        self._parser = QueryParser()
        self._understanding = QueryUnderstanding()
        self._highlighter = Highlighter()
        self._cache = AdaptiveCache(
            max_size=config.cache.max_size,
            ttl_seconds=config.cache.ttl_seconds,
        )

        self._bm25 = BM25Scorer(
            indexer.index,
            k1=config.ranking.bm25_k1,
            b=config.ranking.bm25_b,
        )
        self._ranker = HybridRanker(
            bm25=self._bm25,
            pagerank=indexer.pagerank,
            semantic=indexer.semantic_scorer,
            alpha=config.ranking.alpha,
            beta=config.ranking.beta,
            gamma=config.ranking.gamma,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        user_profile: Optional[UserProfile] = None,
        use_cache: bool = True,
    ) -> Dict:
        """Execute a search query end-to-end.

        Returns a dict with: results, query_info, timing, total_results.
        """
        start_time = time.time()

        cache_key = f"{query}:{top_k}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached["from_cache"] = True
                cached["timing"]["total_ms"] = (time.time() - start_time) * 1000
                return cached

        t0 = time.time()
        understanding = self._understanding.process(query)
        understanding_ms = (time.time() - t0) * 1000

        t0 = time.time()
        parsed = self._parser.parse(understanding.rewritten_query)
        raw_terms = self._parser.extract_terms(parsed)
        query_tokens = self._tokenizer.tokenize(understanding.rewritten_query)

        all_terms = list(set(query_tokens + [
            t for t in self._tokenizer.tokenize(" ".join(understanding.expanded_terms))
        ]))
        parse_ms = (time.time() - t0) * 1000

        t0 = time.time()
        candidate_postings = self.indexer.index.multi_term_search(all_terms)
        retrieval_ms = (time.time() - t0) * 1000

        t0 = time.time()
        ranked = self._ranker.rank(
            query_terms=query_tokens,
            query_text=query,
            candidate_postings=candidate_postings,
            top_k=top_k,
            user_profile=user_profile,
        )
        ranking_ms = (time.time() - t0) * 1000

        t0 = time.time()
        results: List[Dict] = []
        for doc_id, score, breakdown in ranked:
            metadata = self.indexer.get_doc_metadata(doc_id)
            if metadata is None:
                continue

            snippet = self._highlighter.generate_snippet(
                metadata.get("content_preview", ""),
                raw_terms,
            )

            results.append({
                "doc_id": doc_id,
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "snippet": snippet,
                "score": round(score, 4),
                "scores": {k: round(v, 4) for k, v in breakdown.items()},
            })
        snippet_ms = (time.time() - t0) * 1000

        total_ms = (time.time() - start_time) * 1000

        response = {
            "results": results,
            "total_results": len(results),
            "query_info": {
                "original": understanding.original_query,
                "rewritten": understanding.rewritten_query,
                "intent": understanding.intent.value,
                "corrections": understanding.corrections,
                "expanded_terms": understanding.expanded_terms[:10],
            },
            "timing": {
                "understanding_ms": round(understanding_ms, 2),
                "parse_ms": round(parse_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "ranking_ms": round(ranking_ms, 2),
                "snippet_ms": round(snippet_ms, 2),
                "total_ms": round(total_ms, 2),
            },
            "from_cache": False,
        }

        if use_cache:
            self._cache.put(cache_key, response)

        return response

    def update_vocabulary(self):
        """Sync the query understanding vocabulary with the current index."""
        vocab = set()
        for shard in self.indexer.index._shards:
            for reader in shard._segments:
                if reader.term_dict:
                    vocab.update(reader.term_dict.terms())
            vocab.update(shard._buffer.keys())
        self._understanding.set_vocabulary(vocab)

    @property
    def cache_stats(self) -> Dict:
        return self._cache.stats

    def invalidate_cache(self):
        self._cache.invalidate_all()
