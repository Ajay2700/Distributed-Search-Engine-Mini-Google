"""Integration tests: end-to-end query pipeline using sample documents."""
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from config import Config
from core import Document
from services.indexer import IndexerService
from services.query.engine import QueryEngine


def _sample_docs():
    return [
        Document(
            doc_id=0,
            url="https://example.com/python",
            title="Python Programming Language",
            content=(
                "Python is a high-level programming language known for its simplicity. "
                "It supports object-oriented, functional, and procedural programming paradigms. "
                "Python is widely used in web development, data science, and machine learning."
            ),
            crawled_at=datetime.now(),
        ),
        Document(
            doc_id=1,
            url="https://example.com/search",
            title="Search Engine Architecture",
            content=(
                "A search engine consists of a crawler, indexer, and query processor. "
                "The inverted index maps terms to document IDs for efficient retrieval. "
                "BM25 and TF-IDF are popular ranking algorithms used in search engines."
            ),
            crawled_at=datetime.now(),
        ),
        Document(
            doc_id=2,
            url="https://example.com/ml",
            title="Machine Learning Basics",
            content=(
                "Machine learning algorithms learn patterns from data without explicit programming. "
                "Neural networks, decision trees, and support vector machines are common approaches. "
                "Deep learning uses multi-layer neural networks for complex pattern recognition."
            ),
            crawled_at=datetime.now(),
        ),
    ]


class TestEndToEndPipeline:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = Config(base_dir=Path(self.tmpdir))
        self.config.ensure_dirs()
        self.config.index.num_shards = 2

        self.indexer = IndexerService(self.config)
        self.docs = _sample_docs()
        self.link_graph = {0: [1, 2], 1: [0, 2], 2: [0]}
        self.indexer.index_documents(self.docs, self.link_graph)

        self.engine = QueryEngine(self.indexer, self.config)
        self.engine.update_vocabulary()

    def teardown_method(self):
        self.indexer.close()

    def test_basic_search(self):
        results = self.engine.search("python programming")
        assert results["total_results"] > 0
        assert results["results"][0]["doc_id"] == 0

    def test_search_returns_timing(self):
        results = self.engine.search("search engine")
        assert "timing" in results
        assert results["timing"]["total_ms"] > 0

    def test_search_returns_query_info(self):
        results = self.engine.search("machine learning")
        assert "query_info" in results
        assert results["query_info"]["original"] == "machine learning"
        assert results["query_info"]["intent"] in ["informational", "navigational", "transactional"]

    def test_no_results_for_garbage(self):
        results = self.engine.search("xyzzyplugh")
        assert results["total_results"] == 0

    def test_score_ordering(self):
        results = self.engine.search("search engine ranking algorithms")
        if len(results["results"]) >= 2:
            scores = [r["score"] for r in results["results"]]
            assert scores == sorted(scores, reverse=True)

    def test_caching(self):
        r1 = self.engine.search("python web development")
        r2 = self.engine.search("python web development")
        assert r2.get("from_cache") is True

    def test_incremental_index(self):
        new_doc = Document(
            doc_id=99,
            url="https://example.com/new",
            title="New Document About Graphs",
            content="Graph algorithms including BFS DFS Dijkstra shortest path traversal.",
            crawled_at=datetime.now(),
        )
        self.indexer.index_single(new_doc)
        self.engine.invalidate_cache()

        results = self.engine.search("graph algorithms shortest path")
        found = any(r["doc_id"] == 99 for r in results["results"])
        assert found

    def test_metadata_storage(self):
        meta = self.indexer.get_doc_metadata(0)
        assert meta is not None
        assert "url" in meta
        assert meta["url"] == "https://example.com/python"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
