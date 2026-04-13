"""Low-level storage primitives for the inverted index.

PostingList: variable-byte encoded posting lists for compact on-disk storage.
TermDictionary: term → offset mapping with binary search support.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, Iterator, List, Optional, Tuple

from core import Posting


class VByteCodec:
    """Variable-byte encoding for compact integer storage.
    Each integer is stored as a sequence of bytes, where the high bit
    indicates continuation (0 = last byte, 1 = more bytes follow)."""

    @staticmethod
    def encode(number: int) -> bytes:
        if number < 0:
            raise ValueError("VByte only encodes non-negative integers")
        buf = bytearray()
        while number >= 128:
            buf.append((number & 0x7F) | 0x80)
            number >>= 7
        buf.append(number & 0x7F)
        return bytes(buf)

    @staticmethod
    def decode_stream(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """Decode one integer starting at offset. Returns (value, new_offset)."""
        result = 0
        shift = 0
        while offset < len(data):
            b = data[offset]
            result |= (b & 0x7F) << shift
            offset += 1
            if not (b & 0x80):
                return result, offset
            shift += 7
        raise ValueError("Truncated VByte sequence")

    @staticmethod
    def encode_list(numbers: List[int]) -> bytes:
        parts = [VByteCodec.encode(n) for n in numbers]
        return b"".join(parts)

    @staticmethod
    def decode_list(data: bytes, count: int, offset: int = 0) -> Tuple[List[int], int]:
        result = []
        for _ in range(count):
            val, offset = VByteCodec.decode_stream(data, offset)
            result.append(val)
        return result, offset


@dataclass
class PostingList:
    """A posting list for one term: list of (doc_id, frequency, positions).
    Supports delta-encoding of doc_ids for compression."""

    postings: List[Posting] = field(default_factory=list)

    def add(self, doc_id: int, frequency: int, positions: List[int]):
        self.postings.append(Posting(doc_id=doc_id, frequency=frequency, positions=positions))

    def serialize(self) -> bytes:
        """Serialize to bytes using VByte encoding with delta-compressed doc_ids."""
        buf = bytearray()
        buf.extend(VByteCodec.encode(len(self.postings)))
        prev_doc_id = 0
        for p in sorted(self.postings, key=lambda x: x.doc_id):
            delta = p.doc_id - prev_doc_id
            buf.extend(VByteCodec.encode(delta))
            buf.extend(VByteCodec.encode(p.frequency))
            buf.extend(VByteCodec.encode(len(p.positions)))
            prev_pos = 0
            for pos in p.positions:
                buf.extend(VByteCodec.encode(pos - prev_pos))
                prev_pos = pos
            prev_doc_id = p.doc_id
        return bytes(buf)

    @classmethod
    def deserialize(cls, data: bytes, offset: int = 0) -> Tuple["PostingList", int]:
        count, offset = VByteCodec.decode_stream(data, offset)
        postings = []
        prev_doc_id = 0
        for _ in range(count):
            delta, offset = VByteCodec.decode_stream(data, offset)
            doc_id = prev_doc_id + delta
            freq, offset = VByteCodec.decode_stream(data, offset)
            num_pos, offset = VByteCodec.decode_stream(data, offset)
            positions = []
            prev_pos = 0
            for _ in range(num_pos):
                pos_delta, offset = VByteCodec.decode_stream(data, offset)
                prev_pos += pos_delta
                positions.append(prev_pos)
            postings.append(Posting(doc_id=doc_id, frequency=freq, positions=positions))
            prev_doc_id = doc_id
        pl = cls(postings=postings)
        return pl, offset

    def merge(self, other: "PostingList") -> "PostingList":
        """Merge two posting lists, summing frequencies for duplicate doc_ids."""
        combined: Dict[int, Posting] = {}
        for p in self.postings + other.postings:
            if p.doc_id in combined:
                existing = combined[p.doc_id]
                combined[p.doc_id] = Posting(
                    doc_id=p.doc_id,
                    frequency=existing.frequency + p.frequency,
                    positions=sorted(existing.positions + p.positions),
                )
            else:
                combined[p.doc_id] = p
        return PostingList(postings=sorted(combined.values(), key=lambda x: x.doc_id))


@dataclass
class TermDictEntry:
    term: str
    doc_frequency: int
    offset: int
    length: int


class TermDictionary:
    """In-memory term dictionary mapping terms to posting list locations."""

    def __init__(self):
        self._entries: Dict[str, TermDictEntry] = {}

    def add(self, term: str, doc_frequency: int, offset: int, length: int):
        self._entries[term] = TermDictEntry(term, doc_frequency, offset, length)

    def lookup(self, term: str) -> Optional[TermDictEntry]:
        return self._entries.get(term)

    def terms(self) -> List[str]:
        return sorted(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, term: str) -> bool:
        return term in self._entries

    def serialize(self) -> bytes:
        """Serialize term dictionary as: [num_terms] [term_len term doc_freq offset length]..."""
        buf = bytearray()
        sorted_entries = sorted(self._entries.values(), key=lambda e: e.term)
        buf.extend(struct.pack(">I", len(sorted_entries)))
        for entry in sorted_entries:
            term_bytes = entry.term.encode("utf-8")
            buf.extend(struct.pack(">H", len(term_bytes)))
            buf.extend(term_bytes)
            buf.extend(struct.pack(">III", entry.doc_frequency, entry.offset, entry.length))
        return bytes(buf)

    @classmethod
    def deserialize(cls, data: bytes) -> "TermDictionary":
        td = cls()
        offset = 0
        (num_terms,) = struct.unpack_from(">I", data, offset)
        offset += 4
        for _ in range(num_terms):
            (term_len,) = struct.unpack_from(">H", data, offset)
            offset += 2
            term = data[offset: offset + term_len].decode("utf-8")
            offset += term_len
            doc_freq, pl_offset, pl_length = struct.unpack_from(">III", data, offset)
            offset += 12
            td.add(term, doc_freq, pl_offset, pl_length)
        return td
