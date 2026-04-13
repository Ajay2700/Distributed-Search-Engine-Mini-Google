# Design Decisions & Tradeoffs

## 1. Why BM25 Over Pure TF-IDF?

**Decision**: BM25 as the primary lexical scoring function.

**Rationale**:
- **Term Frequency Saturation**: BM25 has a built-in saturation curve via `k1`. After a term appears N times, additional occurrences yield diminishing returns. TF-IDF's `log(1 + tf)` is less aggressive at dampening.
- **Document Length Normalization**: The `b` parameter controls how much longer documents are penalized. This prevents verbose pages from dominating results purely due to word count.
- **Empirical Performance**: BM25 consistently outperforms TF-IDF on standard IR benchmarks (TREC, MS MARCO). It remains the default scoring function in Elasticsearch and Lucene.
- **Tunability**: Two parameters (`k1`, `b`) offer fine-grained control vs TF-IDF's fixed formulation.

**Tradeoff**: BM25 still operates at the term level — it cannot capture semantic meaning. This is why we add semantic scoring.

## 2. Sharding Strategy

**Decision**: Hash-based sharding on `doc_id` with fan-out queries.

**Alternatives Considered**:
| Strategy | Pros | Cons |
|----------|------|------|
| Hash on doc_id | Even distribution, simple routing | Every query hits all shards |
| Hash on term | Only relevant shards queried | Hotspot terms cause imbalance |
| Range partitioning | Good for sorted access | Uneven distribution, rebalancing |

**Why doc_id hashing**: 
- Simplest to implement and reason about
- Even distribution across shards regardless of content
- Fan-out cost is acceptable for moderate shard counts (4-16)
- No rebalancing needed when content changes

**Tradeoff**: Every query must hit every shard. For very large clusters (>100 shards), term-based sharding with query routing would be more efficient.

## 3. Segment-Based Indexing (Lucene-style)

**Decision**: Immutable segments with background merge, instead of a single mutable index.

**Benefits**:
- **Concurrency**: Readers never block writers (segments are immutable once written)
- **Incremental updates**: New documents go to a new segment without rewriting existing ones
- **Crash recovery**: Incomplete writes only affect the latest segment
- **Memory efficiency**: Old segments can be memory-mapped without concern for mutation

**Costs**:
- **Read amplification**: Queries must search across multiple segments
- **Merge overhead**: Background merging reclaims space and reduces segment count
- **Deletion complexity**: Requires tombstones until segments are merged

## 4. Hybrid Ranking (Lexical + Semantic + Authority)

**Decision**: `score = α·BM25 + β·PageRank + γ·Semantic`

**Rationale**:
- **BM25** captures exact keyword relevance (essential for precision)
- **PageRank** captures web authority (prevents spam/low-quality results)
- **Semantic similarity** captures meaning-based relevance (improves recall for synonym/paraphrase queries)

**Weight tuning (default: 0.5/0.2/0.3)**:
- α=0.5: Lexical matching is still the strongest signal for most queries
- β=0.2: Authority matters but shouldn't dominate (unfair to new pages)
- γ=0.3: Semantic helps significantly for ambiguous/short queries

**Tradeoff**: Adding semantic scoring increases latency (embedding computation) but dramatically improves result quality for conceptual queries.

## 5. Adaptive Cache (LFU/LRU Hybrid)

**Decision**: Hybrid eviction policy instead of pure LRU or LFU.

**Problem with pure LRU**: One-time popular queries can evict frequently-used entries.
**Problem with pure LFU**: Historical frequency doesn't reflect current popularity shifts.

**Solution**: 
- New entries start in LRU mode (low frequency)
- After `threshold` accesses, entries "graduate" to LFU protection
- Expired entries are always evicted first
- Auto-invalidation on index updates prevents stale results

## 6. Failure Scenarios

### Node Crash
- **Impact**: Queries to the crashed shard fail
- **Mitigation**: Other shards still return partial results (graceful degradation)
- **Recovery**: Segment files on disk are immutable → restart reads existing segments

### Partial Index Loss
- **Impact**: Some terms/documents missing from results
- **Mitigation**: Re-indexing from crawl storage (raw HTML preserved on disk)
- **Detection**: Health checks compare expected vs actual document counts

### Network Partition (Crawler)
- **Impact**: Some hosts unreachable during crawl
- **Mitigation**: URLs returned to frontier with backoff; retry on next crawl cycle

### Cache Inconsistency
- **Impact**: Stale results served after index update
- **Mitigation**: Version-based invalidation → cache key includes index version

## 7. VByte Encoding for Posting Lists

**Decision**: Variable-byte encoding with delta-compressed doc IDs.

**Alternatives**:
| Encoding | Compression | Decode Speed |
|----------|------------|-------------|
| Fixed 4-byte | 1x (baseline) | Fastest |
| VByte | ~2-3x | Fast |
| PForDelta | ~4-5x | Medium |
| Simple9/16 | ~3-4x | Fast |

**Why VByte**: Best balance of compression ratio and decode speed for our scale. PForDelta would be better at >10M documents but adds implementation complexity.
