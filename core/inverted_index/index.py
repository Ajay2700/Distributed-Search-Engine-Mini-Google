"""Core InvertedIndex with segment-based incremental indexing.

Supports:
  - Adding documents incrementally (buffered → flushed to segments)
  - Background segment merging
  - Querying across all segments
  - Document deletion (tombstones)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Set, Tuple

from core import Posting
from core.inverted_index.segment import Segment, SegmentMeta, SegmentReader, SegmentWriter
from core.inverted_index.storage import PostingList, TermDictionary

logger = logging.getLogger(__name__)


class InvertedIndex:
    """Thread-safe inverted index with segment-based storage and incremental updates."""

    def __init__(
        self,
        index_dir: Path,
        *,
        segment_max_docs: int = 1000,
        merge_factor: int = 10,
        use_mmap: bool = True,
    ):
        self.index_dir = index_dir
        self.segment_max_docs = segment_max_docs
        self.merge_factor = merge_factor
        self.use_mmap = use_mmap

        self._buffer: Dict[str, PostingList] = {}
        self._buffer_doc_ids: Set[int] = set()
        self._segments: List[SegmentReader] = []
        self._segment_metas: List[SegmentMeta] = []
        self._deleted_docs: Set[int] = set()
        self._doc_lengths: Dict[int, int] = {}
        self._total_docs: int = 0
        self._lock = Lock()

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing_segments()

    def _load_existing_segments(self):
        """Load all existing segments from disk."""
        segments_dir = self.index_dir / "segments"
        if not segments_dir.exists():
            segments_dir.mkdir(parents=True, exist_ok=True)
            return

        meta_files = sorted(segments_dir.glob("*.meta"))
        for meta_path in meta_files:
            try:
                meta = SegmentMeta.from_dict(json.loads(meta_path.read_text()))
                reader = SegmentReader(segments_dir, meta.segment_id, self.use_mmap)
                reader.open()
                self._segments.append(reader)
                self._segment_metas.append(meta)
                self._total_docs += meta.num_docs
            except Exception as e:
                logger.warning(f"Failed to load segment {meta_path}: {e}")

        tombstone_path = self.index_dir / "tombstones.json"
        if tombstone_path.exists():
            self._deleted_docs = set(json.loads(tombstone_path.read_text()))

        lengths_path = self.index_dir / "doc_lengths.json"
        if lengths_path.exists():
            data = json.loads(lengths_path.read_text())
            self._doc_lengths = {int(k): v for k, v in data.items()}

    def add_document(self, doc_id: int, term_freqs: Dict[str, Tuple[int, List[int]]]):
        """Add a document's term frequencies to the in-memory buffer.
        Automatically flushes when buffer reaches segment_max_docs.
        """
        with self._lock:
            self._buffer_doc_ids.add(doc_id)
            doc_length = sum(freq for freq, _ in term_freqs.values())
            self._doc_lengths[doc_id] = doc_length
            self._total_docs += 1

            for term, (freq, positions) in term_freqs.items():
                if term not in self._buffer:
                    self._buffer[term] = PostingList()
                self._buffer[term].add(doc_id, freq, positions)

            if len(self._buffer_doc_ids) >= self.segment_max_docs:
                self._flush_buffer()

    def _flush_buffer(self):
        """Flush in-memory buffer to a new segment on disk."""
        if not self._buffer_doc_ids:
            return

        seg_id = f"seg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        segments_dir = self.index_dir / "segments"
        writer = SegmentWriter(segments_dir, seg_id)

        doc_terms: Dict[int, Dict[str, Tuple[int, List[int]]]] = {}
        for term, pl in self._buffer.items():
            for posting in pl.postings:
                if posting.doc_id not in doc_terms:
                    doc_terms[posting.doc_id] = {}
                doc_terms[posting.doc_id][term] = (posting.frequency, posting.positions)

        for doc_id, tfs in doc_terms.items():
            writer.add_document(doc_id, tfs)

        meta = writer.flush()

        reader = SegmentReader(segments_dir, seg_id, self.use_mmap)
        reader.open()
        self._segments.append(reader)
        self._segment_metas.append(meta)

        self._buffer.clear()
        self._buffer_doc_ids.clear()

        self._save_doc_lengths()
        logger.info(f"Flushed segment {seg_id} with {meta.num_docs} docs, {meta.num_terms} terms")

        if len(self._segments) >= self.merge_factor:
            self._trigger_merge()

    def flush(self):
        """Force flush any buffered documents."""
        with self._lock:
            self._flush_buffer()

    def delete_document(self, doc_id: int):
        """Mark a document as deleted (tombstone)."""
        with self._lock:
            self._deleted_docs.add(doc_id)
            if doc_id in self._doc_lengths:
                del self._doc_lengths[doc_id]
            self._total_docs = max(0, self._total_docs - 1)
            tombstone_path = self.index_dir / "tombstones.json"
            tombstone_path.write_text(json.dumps(sorted(self._deleted_docs)))

    def search(self, term: str) -> List[Posting]:
        """Search for a term across all segments + buffer. Excludes deleted docs."""
        results: List[Posting] = []

        with self._lock:
            if term in self._buffer:
                for p in self._buffer[term].postings:
                    if p.doc_id not in self._deleted_docs:
                        results.append(p)

        for reader in self._segments:
            pl = reader.get_postings(term)
            if pl:
                for p in pl.postings:
                    if p.doc_id not in self._deleted_docs:
                        results.append(p)

        return sorted(results, key=lambda p: p.doc_id)

    def doc_frequency(self, term: str) -> int:
        """Return number of documents containing the term."""
        df = 0
        with self._lock:
            if term in self._buffer:
                df += sum(1 for p in self._buffer[term].postings
                          if p.doc_id not in self._deleted_docs)
        for reader in self._segments:
            pl = reader.get_postings(term)
            if pl:
                df += sum(1 for p in pl.postings if p.doc_id not in self._deleted_docs)
        return df

    def get_doc_length(self, doc_id: int) -> int:
        return self._doc_lengths.get(doc_id, 0)

    @property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    @property
    def total_docs(self) -> int:
        return self._total_docs

    @property
    def vocabulary_size(self) -> int:
        terms: Set[str] = set(self._buffer.keys())
        for reader in self._segments:
            if reader.term_dict:
                terms.update(reader.term_dict.terms())
        return len(terms)

    def _trigger_merge(self):
        """Merge smallest segments in background thread."""
        Thread(target=self._merge_segments, daemon=True).start()

    def _merge_segments(self):
        """Merge the two smallest segments into one."""
        with self._lock:
            if len(self._segments) < 2:
                return

            sorted_segs = sorted(
                zip(self._segments, self._segment_metas),
                key=lambda x: x[1].num_docs,
            )
            to_merge = sorted_segs[:2]
            seg1_reader, seg1_meta = to_merge[0]
            seg2_reader, seg2_meta = to_merge[1]

            seg_id = f"merged_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            segments_dir = self.index_dir / "segments"
            writer = SegmentWriter(segments_dir, seg_id)

            all_terms: Set[str] = set()
            if seg1_reader.term_dict:
                all_terms.update(seg1_reader.term_dict.terms())
            if seg2_reader.term_dict:
                all_terms.update(seg2_reader.term_dict.terms())

            merged_postings: Dict[str, PostingList] = {}
            for term in all_terms:
                pl1 = seg1_reader.get_postings(term)
                pl2 = seg2_reader.get_postings(term)
                if pl1 and pl2:
                    merged_postings[term] = pl1.merge(pl2)
                elif pl1:
                    merged_postings[term] = pl1
                elif pl2:
                    merged_postings[term] = pl2

            doc_terms: Dict[int, Dict[str, Tuple[int, List[int]]]] = {}
            for term, pl in merged_postings.items():
                for posting in pl.postings:
                    if posting.doc_id in self._deleted_docs:
                        continue
                    if posting.doc_id not in doc_terms:
                        doc_terms[posting.doc_id] = {}
                    doc_terms[posting.doc_id][term] = (posting.frequency, posting.positions)

            for doc_id, tfs in doc_terms.items():
                writer.add_document(doc_id, tfs)

            meta = writer.flush()
            new_reader = SegmentReader(segments_dir, seg_id, self.use_mmap)
            new_reader.open()

            self._segments.remove(seg1_reader)
            self._segments.remove(seg2_reader)
            self._segment_metas.remove(seg1_meta)
            self._segment_metas.remove(seg2_meta)

            seg1_reader.close()
            seg2_reader.close()

            self._segments.append(new_reader)
            self._segment_metas.append(meta)

            logger.info(f"Merged segments into {seg_id}")

    def _save_doc_lengths(self):
        lengths_path = self.index_dir / "doc_lengths.json"
        lengths_path.write_text(json.dumps(self._doc_lengths))

    def close(self):
        """Flush remaining buffer and close all segment readers."""
        self.flush()
        for reader in self._segments:
            reader.close()
        self._save_doc_lengths()
