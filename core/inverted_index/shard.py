"""Distributed index sharding with query fan-out and aggregation.

Partitions the index across N shards using consistent hashing on doc_id.
Query processing fans out to all shards in parallel and aggregates results.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import mmh3

from core import Posting
from core.inverted_index.index import InvertedIndex

logger = logging.getLogger(__name__)


class ShardedIndex:
    """Distributes documents across multiple InvertedIndex shards.
    
    Sharding strategy: hash(doc_id) % num_shards
    Query strategy: fan-out to all shards, merge results
    """

    def __init__(
        self,
        index_dir: Path,
        num_shards: int = 4,
        **index_kwargs,
    ):
        self.index_dir = index_dir
        self.num_shards = num_shards
        self._shards: List[InvertedIndex] = []
        self._executor = ThreadPoolExecutor(max_workers=num_shards)

        for shard_id in range(num_shards):
            shard_dir = index_dir / f"shard_{shard_id}"
            shard = InvertedIndex(shard_dir, **index_kwargs)
            self._shards.append(shard)

    def _shard_for_doc(self, doc_id: int) -> int:
        return mmh3.hash(str(doc_id), signed=False) % self.num_shards

    def add_document(self, doc_id: int, term_freqs: Dict[str, Tuple[int, List[int]]]):
        shard_id = self._shard_for_doc(doc_id)
        self._shards[shard_id].add_document(doc_id, term_freqs)

    def delete_document(self, doc_id: int):
        shard_id = self._shard_for_doc(doc_id)
        self._shards[shard_id].delete_document(doc_id)

    def search(self, term: str) -> List[Posting]:
        """Fan-out search to all shards in parallel, merge results."""
        all_postings: List[Posting] = []
        futures = {
            self._executor.submit(shard.search, term): i
            for i, shard in enumerate(self._shards)
        }
        for future in as_completed(futures):
            shard_id = futures[future]
            try:
                postings = future.result()
                all_postings.extend(postings)
            except Exception as e:
                logger.error(f"Shard {shard_id} search failed for '{term}': {e}")
        return sorted(all_postings, key=lambda p: p.doc_id)

    def multi_term_search(self, terms: List[str]) -> Dict[str, List[Posting]]:
        """Search multiple terms across all shards in parallel."""
        results: Dict[str, List[Posting]] = {}
        futures = {}
        for term in terms:
            for i, shard in enumerate(self._shards):
                fut = self._executor.submit(shard.search, term)
                futures[fut] = (term, i)

        for future in as_completed(futures):
            term, shard_id = futures[future]
            try:
                postings = future.result()
                if term not in results:
                    results[term] = []
                results[term].extend(postings)
            except Exception as e:
                logger.error(f"Shard {shard_id} search failed for '{term}': {e}")

        for term in results:
            results[term].sort(key=lambda p: p.doc_id)
        return results

    def doc_frequency(self, term: str) -> int:
        return sum(shard.doc_frequency(term) for shard in self._shards)

    def get_doc_length(self, doc_id: int) -> int:
        shard_id = self._shard_for_doc(doc_id)
        return self._shards[shard_id].get_doc_length(doc_id)

    @property
    def avg_doc_length(self) -> float:
        total_length = 0
        total_docs = 0
        for shard in self._shards:
            for doc_id, length in shard._doc_lengths.items():
                total_length += length
                total_docs += 1
        return total_length / max(total_docs, 1)

    @property
    def total_docs(self) -> int:
        return sum(shard.total_docs for shard in self._shards)

    def flush(self):
        for shard in self._shards:
            shard.flush()

    def close(self):
        for shard in self._shards:
            shard.close()
        self._executor.shutdown(wait=False)
