"""Unit tests for the inverted index, segments, and sharding."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.inverted_index.storage import VByteCodec, PostingList, TermDictionary
from core.inverted_index.segment import SegmentWriter, SegmentReader
from core.inverted_index.index import InvertedIndex
from core.inverted_index.shard import ShardedIndex
from core import Posting


class TestVByteCodec:
    def test_encode_decode_small(self):
        for n in [0, 1, 127, 128, 255, 1000, 100000]:
            encoded = VByteCodec.encode(n)
            decoded, _ = VByteCodec.decode_stream(encoded)
            assert decoded == n

    def test_encode_list(self):
        numbers = [10, 200, 3000, 40000]
        encoded = VByteCodec.encode_list(numbers)
        decoded, _ = VByteCodec.decode_list(encoded, len(numbers))
        assert decoded == numbers

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            VByteCodec.encode(-1)


class TestPostingList:
    def test_serialize_deserialize(self):
        pl = PostingList()
        pl.add(1, 3, [0, 5, 10])
        pl.add(5, 2, [1, 7])
        pl.add(10, 1, [3])

        data = pl.serialize()
        restored, _ = PostingList.deserialize(data)

        assert len(restored.postings) == 3
        assert restored.postings[0].doc_id == 1
        assert restored.postings[0].frequency == 3
        assert restored.postings[0].positions == [0, 5, 10]
        assert restored.postings[2].doc_id == 10

    def test_merge(self):
        pl1 = PostingList()
        pl1.add(1, 2, [0, 3])
        pl1.add(3, 1, [1])

        pl2 = PostingList()
        pl2.add(1, 1, [5])
        pl2.add(5, 3, [0, 2, 4])

        merged = pl1.merge(pl2)
        assert len(merged.postings) == 3
        doc1 = next(p for p in merged.postings if p.doc_id == 1)
        assert doc1.frequency == 3
        assert doc1.positions == [0, 3, 5]


class TestTermDictionary:
    def test_serialize_deserialize(self):
        td = TermDictionary()
        td.add("hello", 5, 0, 100)
        td.add("world", 3, 100, 80)
        td.add("foo", 1, 180, 20)

        data = td.serialize()
        restored = TermDictionary.deserialize(data)

        assert len(restored) == 3
        assert "hello" in restored
        entry = restored.lookup("hello")
        assert entry.doc_frequency == 5
        assert entry.offset == 0
        assert entry.length == 100


class TestSegment:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_dir = Path(tmpdir)
            writer = SegmentWriter(seg_dir, "test_seg")

            writer.add_document(0, {
                "hello": (2, [0, 5]),
                "world": (1, [3]),
            })
            writer.add_document(1, {
                "hello": (1, [0]),
                "foo": (3, [1, 4, 7]),
            })

            meta = writer.flush()
            assert meta.num_docs == 2
            assert meta.num_terms == 3

            reader = SegmentReader(seg_dir, "test_seg", use_mmap=False)
            reader.open()

            pl = reader.get_postings("hello")
            assert pl is not None
            assert len(pl.postings) == 2

            assert reader.doc_frequency("foo") == 1
            assert reader.get_postings("nonexistent") is None

            reader.close()


class TestInvertedIndex:
    def test_add_and_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = InvertedIndex(Path(tmpdir), segment_max_docs=100, use_mmap=False)

            idx.add_document(0, {"hello": (2, [0, 5]), "world": (1, [3])})
            idx.add_document(1, {"hello": (1, [0]), "foo": (1, [2])})

            results = idx.search("hello")
            assert len(results) == 2
            assert results[0].doc_id == 0

            assert idx.doc_frequency("hello") == 2
            assert idx.doc_frequency("foo") == 1
            assert idx.total_docs == 2

            idx.close()

    def test_delete_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = InvertedIndex(Path(tmpdir), segment_max_docs=100, use_mmap=False)

            idx.add_document(0, {"hello": (1, [0])})
            idx.add_document(1, {"hello": (1, [0])})
            idx.delete_document(0)

            results = idx.search("hello")
            assert len(results) == 1
            assert results[0].doc_id == 1

            idx.close()

    def test_flush_creates_segment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = InvertedIndex(Path(tmpdir), segment_max_docs=2, use_mmap=False)

            idx.add_document(0, {"a": (1, [0])})
            idx.add_document(1, {"b": (1, [0])})
            # Should auto-flush at segment_max_docs=2

            results = idx.search("a")
            assert len(results) == 1

            idx.close()


class TestShardedIndex:
    def test_sharded_add_and_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = ShardedIndex(
                Path(tmpdir), num_shards=2,
                segment_max_docs=100, use_mmap=False,
            )

            idx.add_document(0, {"hello": (2, [0, 5]), "world": (1, [3])})
            idx.add_document(1, {"hello": (1, [0]), "bar": (1, [2])})
            idx.add_document(2, {"world": (3, [0, 2, 4])})

            results = idx.search("hello")
            assert len(results) == 2

            results = idx.search("world")
            assert len(results) == 2

            assert idx.total_docs == 3

            idx.close()

    def test_multi_term_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = ShardedIndex(
                Path(tmpdir), num_shards=2,
                segment_max_docs=100, use_mmap=False,
            )

            idx.add_document(0, {"alpha": (1, [0]), "beta": (1, [1])})
            idx.add_document(1, {"beta": (1, [0]), "gamma": (1, [1])})

            results = idx.multi_term_search(["alpha", "beta", "gamma"])
            assert "alpha" in results
            assert "beta" in results
            assert len(results["beta"]) == 2

            idx.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
