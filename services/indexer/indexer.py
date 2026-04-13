"""Indexer service: processes crawled documents into the inverted index.

Supports both batch indexing and incremental (real-time) document addition.
Coordinates with the embedding encoder for semantic index construction.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from config import Config
from core import Document
from core.embeddings.encoder import EmbeddingEncoder
from core.inverted_index.shard import ShardedIndex
from core.ranking.pagerank import PageRank
from core.ranking.semantic import SemanticScorer
from core.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class IndexerService:
    """Manages the full indexing pipeline: tokenize → index → embed.

    Supports:
    - Batch indexing of crawled document sets
    - Incremental single-document addition
    - PageRank computation from link graph
    - Semantic embedding index construction
    """

    def __init__(self, config: Config):
        self.config = config
        config.ensure_dirs()

        self._tokenizer = Tokenizer()
        self._index = ShardedIndex(
            index_dir=config.index_dir,
            num_shards=config.index.num_shards,
            segment_max_docs=config.index.segment_max_docs,
            merge_factor=config.index.merge_factor,
            use_mmap=config.index.use_mmap,
        )
        self._encoder = EmbeddingEncoder(
            model_name=config.embedding.model_name,
            cache_dir=config.embeddings_dir,
            dimension=config.embedding.dimension,
            batch_size=config.embedding.batch_size,
        )
        self._semantic = SemanticScorer(self._encoder)
        self._pagerank = PageRank(
            damping=config.ranking.pagerank_damping,
            max_iterations=config.ranking.pagerank_iterations,
        )

        self._doc_metadata: Dict[int, Dict] = {}
        self._load_metadata()

    def index_documents(self, documents: List[Document],
                        link_graph: Optional[Dict[int, List[int]]] = None):
        """Index a batch of documents (full pipeline)."""
        start = time.time()
        logger.info(f"Indexing {len(documents)} documents...")

        for doc in documents:
            self._index_single(doc)

        self._index.flush()

        if link_graph:
            logger.info("Computing PageRank...")
            self._pagerank.compute(link_graph)

        logger.info("Building semantic embeddings...")
        doc_texts = {doc.doc_id: doc.content[:512] for doc in documents}
        self._semantic.index_documents_batch(doc_texts)

        self._save_metadata()

        elapsed = time.time() - start
        logger.info(
            f"Indexing complete: {len(documents)} docs in {elapsed:.2f}s "
            f"({len(documents) / elapsed:.1f} docs/sec)"
        )

    def index_single(self, doc: Document):
        """Index a single document incrementally (real-time)."""
        self._index_single(doc)
        self._semantic.index_document(doc.doc_id, doc.content[:512])
        self._save_metadata()

    def _index_single(self, doc: Document):
        term_freqs = self._tokenizer.term_frequencies(doc.content)
        self._index.add_document(doc.doc_id, term_freqs)

        self._doc_metadata[doc.doc_id] = {
            "url": doc.url,
            "title": doc.title,
            "content_preview": doc.content[:200],
            "crawled_at": doc.crawled_at.isoformat() if doc.crawled_at else None,
        }

    def delete_document(self, doc_id: int):
        """Remove a document from the index."""
        self._index.delete_document(doc_id)
        self._doc_metadata.pop(doc_id, None)
        self._save_metadata()

    def recompute_pagerank(self, link_graph: Dict[int, List[int]]):
        """Recompute PageRank scores from an updated link graph."""
        self._pagerank.compute(link_graph)

    @property
    def index(self) -> ShardedIndex:
        return self._index

    @property
    def semantic_scorer(self) -> SemanticScorer:
        return self._semantic

    @property
    def pagerank(self) -> PageRank:
        return self._pagerank

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def get_doc_metadata(self, doc_id: int) -> Optional[Dict]:
        return self._doc_metadata.get(doc_id)

    @property
    def doc_metadata(self) -> Dict[int, Dict]:
        return self._doc_metadata

    def _save_metadata(self):
        path = self.config.base_dir / "doc_metadata.json"
        serializable = {str(k): v for k, v in self._doc_metadata.items()}
        path.write_text(json.dumps(serializable, indent=2))

    def _load_metadata(self):
        path = self.config.base_dir / "doc_metadata.json"
        if path.exists():
            data = json.loads(path.read_text())
            self._doc_metadata = {int(k): v for k, v in data.items()}

    def close(self):
        self._index.close()
        self._save_metadata()
