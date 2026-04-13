"""Semantic scoring using dense embeddings and cosine similarity.

Converts queries and documents to dense vectors using SentenceTransformers
and computes relevance as cosine similarity between the query vector
and each candidate document's vector.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.embeddings.encoder import EmbeddingEncoder


class SemanticScorer:
    """Scores documents by semantic similarity to the query."""

    def __init__(self, encoder: EmbeddingEncoder):
        self._encoder = encoder
        self._doc_embeddings: Dict[int, np.ndarray] = {}

    def index_document(self, doc_id: int, text: str):
        """Pre-compute and store a document's embedding."""
        self._doc_embeddings[doc_id] = self._encoder.encode(text)

    def index_documents_batch(self, docs: Dict[int, str]):
        """Batch-index multiple documents."""
        doc_ids = list(docs.keys())
        texts = list(docs.values())
        embeddings = self._encoder.encode_batch(texts)
        for doc_id, emb in zip(doc_ids, embeddings):
            self._doc_embeddings[doc_id] = emb

    def score(self, query: str, doc_id: int) -> float:
        if doc_id not in self._doc_embeddings:
            return 0.0
        query_emb = self._encoder.encode(query)
        return self._encoder.cosine_similarity(query_emb, self._doc_embeddings[doc_id])

    def score_candidates(self, query: str, candidate_doc_ids: List[int]) -> Dict[int, float]:
        """Score multiple candidates at once (efficient batch operation)."""
        query_emb = self._encoder.encode(query)
        scores: Dict[int, float] = {}

        valid_ids = [d for d in candidate_doc_ids if d in self._doc_embeddings]
        if not valid_ids:
            return scores

        doc_vecs = np.array([self._doc_embeddings[d] for d in valid_ids])
        similarities = self._encoder.batch_cosine_similarity(query_emb, doc_vecs)

        for doc_id, sim in zip(valid_ids, similarities):
            scores[doc_id] = float(sim)

        return scores

    def has_embedding(self, doc_id: int) -> bool:
        return doc_id in self._doc_embeddings

    @property
    def indexed_count(self) -> int:
        return len(self._doc_embeddings)
