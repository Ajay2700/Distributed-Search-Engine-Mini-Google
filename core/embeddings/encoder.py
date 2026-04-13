"""Embedding encoder using SentenceTransformers (MiniLM).

Encodes text into dense vectors for semantic similarity computation.
Supports batch encoding and optional disk-based embedding cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Wraps SentenceTransformers for query/document encoding with caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        dimension: int = 384,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.dimension = dimension
        self.batch_size = batch_size
        self._model = None
        self._memory_cache: Dict[str, np.ndarray] = {}

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to random embeddings for demo."
            )
            self._model = "fallback"

    def encode(self, text: str) -> np.ndarray:
        cache_key = self._cache_key(text)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        if self.cache_dir:
            cached = self._load_from_disk(cache_key)
            if cached is not None:
                self._memory_cache[cache_key] = cached
                return cached

        self._load_model()

        if self._model == "fallback":
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self.dimension).astype(np.float32)
            embedding /= np.linalg.norm(embedding)
        else:
            embedding = self._model.encode(text, show_progress_bar=False)

        self._memory_cache[cache_key] = embedding
        if self.cache_dir:
            self._save_to_disk(cache_key, embedding)
        return embedding

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts, leveraging batch processing."""
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._memory_cache:
                results.append((i, self._memory_cache[cache_key]))
            elif self.cache_dir:
                cached = self._load_from_disk(cache_key)
                if cached is not None:
                    self._memory_cache[cache_key] = cached
                    results.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            self._load_model()
            if self._model == "fallback":
                embeddings = []
                for text in uncached_texts:
                    np.random.seed(hash(text) % (2**32))
                    emb = np.random.randn(self.dimension).astype(np.float32)
                    emb /= np.linalg.norm(emb)
                    embeddings.append(emb)
                new_embeddings = np.array(embeddings)
            else:
                new_embeddings = self._model.encode(
                    uncached_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )

            for j, idx in enumerate(uncached_indices):
                emb = new_embeddings[j]
                cache_key = self._cache_key(uncached_texts[j])
                self._memory_cache[cache_key] = emb
                if self.cache_dir:
                    self._save_to_disk(cache_key, emb)
                results.append((idx, emb))

        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query vector and multiple doc vectors."""
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(len(doc_vecs))
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        doc_norms[doc_norms == 0] = 1.0
        return np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _save_to_disk(self, key: str, embedding: np.ndarray):
        if self.cache_dir:
            path = self.cache_dir / f"{key}.npy"
            np.save(path, embedding)

    def _load_from_disk(self, key: str) -> Optional[np.ndarray]:
        if self.cache_dir:
            path = self.cache_dir / f"{key}.npy"
            if path.exists():
                return np.load(path)
        return None
