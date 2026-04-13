"""Segment-based index storage (inspired by Lucene's segment architecture).

Each segment is an immutable set of posting lists + term dictionary.
New documents are buffered and flushed as new segments.
Background merge combines small segments into larger ones.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

from core import Posting
from core.inverted_index.storage import PostingList, TermDictionary


class SegmentMeta:
    """Metadata for a single segment on disk."""
    def __init__(self, segment_id: str, num_docs: int, num_terms: int,
                 doc_ids: Set[int], created_at: float):
        self.segment_id = segment_id
        self.num_docs = num_docs
        self.num_terms = num_terms
        self.doc_ids = doc_ids
        self.created_at = created_at

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "num_docs": self.num_docs,
            "num_terms": self.num_terms,
            "doc_ids": sorted(self.doc_ids),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentMeta":
        return cls(
            segment_id=d["segment_id"],
            num_docs=d["num_docs"],
            num_terms=d["num_terms"],
            doc_ids=set(d["doc_ids"]),
            created_at=d["created_at"],
        )


class SegmentWriter:
    """Writes an immutable segment to disk.

    Produces two files:
      - {segment_id}.post : posting lists (VByte encoded)
      - {segment_id}.dict : term dictionary
      - {segment_id}.meta : segment metadata (JSON)
    """

    def __init__(self, segment_dir: Path, segment_id: str):
        self.segment_dir = segment_dir
        self.segment_id = segment_id
        self._postings: Dict[str, PostingList] = {}
        self._doc_ids: Set[int] = set()

    def add_document(self, doc_id: int, term_freqs: Dict[str, Tuple[int, List[int]]]):
        """Add a document's term frequencies. term_freqs: {term: (freq, [positions])}"""
        self._doc_ids.add(doc_id)
        for term, (freq, positions) in term_freqs.items():
            if term not in self._postings:
                self._postings[term] = PostingList()
            self._postings[term].add(doc_id, freq, positions)

    def flush(self) -> SegmentMeta:
        """Write segment to disk and return metadata."""
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        post_path = self.segment_dir / f"{self.segment_id}.post"
        dict_path = self.segment_dir / f"{self.segment_id}.dict"
        meta_path = self.segment_dir / f"{self.segment_id}.meta"

        term_dict = TermDictionary()
        offset = 0
        all_post_data = bytearray()

        for term in sorted(self._postings.keys()):
            pl = self._postings[term]
            serialized = pl.serialize()
            term_dict.add(term, len(pl.postings), offset, len(serialized))
            all_post_data.extend(serialized)
            offset += len(serialized)

        post_path.write_bytes(bytes(all_post_data))
        dict_path.write_bytes(term_dict.serialize())

        meta = SegmentMeta(
            segment_id=self.segment_id,
            num_docs=len(self._doc_ids),
            num_terms=len(self._postings),
            doc_ids=self._doc_ids,
            created_at=time.time(),
        )
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2))
        return meta


class SegmentReader:
    """Reads a segment from disk, optionally using memory-mapped I/O."""

    def __init__(self, segment_dir: Path, segment_id: str, use_mmap: bool = True):
        self.segment_dir = segment_dir
        self.segment_id = segment_id
        self._use_mmap = use_mmap
        self._post_data: Optional[bytes | mmap.mmap] = None
        self._term_dict: Optional[TermDictionary] = None
        self._meta: Optional[SegmentMeta] = None
        self._mmap_file = None

    def open(self):
        post_path = self.segment_dir / f"{self.segment_id}.post"
        dict_path = self.segment_dir / f"{self.segment_id}.dict"
        meta_path = self.segment_dir / f"{self.segment_id}.meta"

        if self._use_mmap and post_path.stat().st_size > 0:
            self._mmap_file = open(post_path, "rb")
            self._post_data = mmap.mmap(self._mmap_file.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self._post_data = post_path.read_bytes()

        self._term_dict = TermDictionary.deserialize(dict_path.read_bytes())
        self._meta = SegmentMeta.from_dict(json.loads(meta_path.read_text()))

    def close(self):
        if isinstance(self._post_data, mmap.mmap):
            self._post_data.close()
        if self._mmap_file:
            self._mmap_file.close()

    @property
    def meta(self) -> SegmentMeta:
        assert self._meta is not None, "Segment not opened"
        return self._meta

    @property
    def term_dict(self) -> TermDictionary:
        assert self._term_dict is not None, "Segment not opened"
        return self._term_dict

    def get_postings(self, term: str) -> Optional[PostingList]:
        if self._term_dict is None or self._post_data is None:
            return None
        entry = self._term_dict.lookup(term)
        if entry is None:
            return None
        chunk = bytes(self._post_data[entry.offset: entry.offset + entry.length])
        pl, _ = PostingList.deserialize(chunk)
        return pl

    def doc_frequency(self, term: str) -> int:
        if self._term_dict is None:
            return 0
        entry = self._term_dict.lookup(term)
        return entry.doc_frequency if entry else 0


class Segment:
    """Unified segment abstraction wrapping reader and writer."""

    def __init__(self, segment_dir: Path, segment_id: str, use_mmap: bool = True):
        self.segment_dir = segment_dir
        self.segment_id = segment_id
        self.reader = SegmentReader(segment_dir, segment_id, use_mmap)

    def open(self):
        self.reader.open()

    def close(self):
        self.reader.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
