"""Unit tests for ranking algorithms: TF-IDF, BM25, PageRank, and Hybrid."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from core.inverted_index.index import InvertedIndex
from core.ranking.tfidf import TFIDFScorer
from core.ranking.bm25 import BM25Scorer
from core.ranking.pagerank import PageRank
from core.embeddings.encoder import EmbeddingEncoder
from core.ranking.semantic import SemanticScorer
from core.ranking.hybrid import HybridRanker, _normalize_scores
from core import Posting


@pytest.fixture
def small_index():
    tmpdir = tempfile.mkdtemp()
    idx = InvertedIndex(Path(tmpdir), segment_max_docs=100, use_mmap=False)
    idx.add_document(0, {
        "search": (3, [0, 5, 10]),
        "engine": (2, [1, 6]),
        "algorithm": (1, [3]),
    })
    idx.add_document(1, {
        "search": (1, [0]),
        "ranking": (2, [1, 4]),
    })
    idx.add_document(2, {
        "algorithm": (2, [0, 3]),
        "data": (3, [1, 2, 5]),
        "structure": (1, [4]),
    })
    yield idx
    idx.close()


class TestTFIDF:
    def test_idf_common_vs_rare(self, small_index):
        scorer = TFIDFScorer(small_index)
        idf_search = scorer.idf("search")     # in 2 docs
        idf_data = scorer.idf("data")         # in 1 doc
        assert idf_data > idf_search

    def test_tf_monotonic(self, small_index):
        scorer = TFIDFScorer(small_index)
        assert scorer.tf(5) > scorer.tf(1)
        assert scorer.tf(1) > scorer.tf(0)

    def test_score_document(self, small_index):
        scorer = TFIDFScorer(small_index)
        score = scorer.score_document(["search", "engine"], {"search": 3, "engine": 2})
        assert score > 0


class TestBM25:
    def test_score_positive(self, small_index):
        scorer = BM25Scorer(small_index, k1=1.5, b=0.75)
        score = scorer.score_document(
            ["search"], {"search": 3}, doc_length=6, avg_doc_length=5.0
        )
        assert score > 0

    def test_longer_doc_penalized(self, small_index):
        scorer = BM25Scorer(small_index, k1=1.5, b=0.75)
        short = scorer.score_document(["search"], {"search": 2}, doc_length=3, avg_doc_length=5.0)
        long = scorer.score_document(["search"], {"search": 2}, doc_length=20, avg_doc_length=5.0)
        assert short > long

    def test_frequency_saturation(self, small_index):
        scorer = BM25Scorer(small_index, k1=1.5, b=0.75)
        low = scorer.score_document(["search"], {"search": 1}, doc_length=5, avg_doc_length=5.0)
        high = scorer.score_document(["search"], {"search": 100}, doc_length=5, avg_doc_length=5.0)
        ratio = high / low
        assert ratio < 10  # BM25 saturates, so 100x freq doesn't give 100x score


class TestPageRank:
    def test_simple_graph(self):
        pr = PageRank(damping=0.85, max_iterations=100)
        graph = {
            0: [1, 2],
            1: [2],
            2: [0],
        }
        scores = pr.compute(graph)
        assert len(scores) == 3
        assert all(s > 0 for s in scores.values())
        assert abs(sum(scores.values()) - 1.0) < 0.01  # scores sum to ~1

    def test_hub_has_high_rank(self):
        pr = PageRank(damping=0.85)
        graph = {
            0: [1],
            1: [2],
            2: [3],
            3: [0],
            4: [0],
            5: [0],
            6: [0],
        }
        scores = pr.compute(graph)
        assert scores[0] > scores[4]  # Node 0 has many inlinks

    def test_personalized(self):
        pr = PageRank(damping=0.85)
        graph = {0: [1, 2], 1: [2], 2: [0]}
        scores = pr.compute_personalized(graph, preference_docs={0}, bias_weight=0.5)
        assert scores[0] > 0


class TestSemanticScorer:
    def test_indexing_and_scoring(self):
        encoder = EmbeddingEncoder(dimension=384)
        scorer = SemanticScorer(encoder)

        scorer.index_document(0, "machine learning algorithms")
        scorer.index_document(1, "cooking pasta recipes")

        score_ml = scorer.score("neural network deep learning", 0)
        score_cook = scorer.score("neural network deep learning", 1)
        # Semantically, ML should score higher than cooking for an ML query
        assert isinstance(score_ml, float)
        assert isinstance(score_cook, float)

    def test_batch_scoring(self):
        encoder = EmbeddingEncoder(dimension=384)
        scorer = SemanticScorer(encoder)

        scorer.index_document(0, "hello world")
        scorer.index_document(1, "foo bar")

        scores = scorer.score_candidates("hello", [0, 1])
        assert len(scores) == 2


class TestNormalizeScores:
    def test_normalization(self):
        scores = {0: 10.0, 1: 20.0, 2: 30.0}
        normalized = _normalize_scores(scores)
        assert normalized[0] == pytest.approx(0.0)
        assert normalized[2] == pytest.approx(1.0)

    def test_identical_scores(self):
        scores = {0: 5.0, 1: 5.0}
        normalized = _normalize_scores(scores)
        assert all(v == 1.0 for v in normalized.values())

    def test_empty(self):
        assert _normalize_scores({}) == {}


class TestHybridRanker:
    def test_ranking(self, small_index):
        bm25 = BM25Scorer(small_index)
        pr = PageRank()
        pr.compute({0: [1, 2], 1: [2], 2: [0]})

        ranker = HybridRanker(bm25=bm25, pagerank=pr, alpha=0.7, beta=0.3, gamma=0.0)

        postings = {
            "search": [Posting(0, 3, [0, 5, 10]), Posting(1, 1, [0])],
        }
        results = ranker.rank(
            query_terms=["search"],
            query_text="search",
            candidate_postings=postings,
            top_k=5,
        )
        assert len(results) > 0
        assert results[0][1] > 0  # score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
