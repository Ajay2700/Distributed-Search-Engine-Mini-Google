# Architecture — Mini Google Distributed Search Engine

## System Overview

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    API Gateway                          │
                    │              (FastAPI REST Server)                      │
                    └─────────────┬───────────────┬───────────────────────────┘
                                  │               │
                    ┌─────────────▼───┐   ┌───────▼──────────────┐
                    │  Query Engine   │   │   Crawler Service    │
                    │  ┌───────────┐  │   │  ┌───────────────┐   │
                    │  │ Query     │  │   │  │ URL Frontier  │   │
                    │  │ Under-    │  │   │  │ (Priority Q)  │   │
                    │  │ standing  │  │   │  └───────┬───────┘   │
                    │  └─────┬─────┘  │   │  ┌───────▼───────┐   │
                    │  ┌─────▼─────┐  │   │  │ Thread Pool   │   │
                    │  │ Parser &  │  │   │  │ (Workers)     │   │
                    │  │ Tokenizer │  │   │  └───────┬───────┘   │
                    │  └─────┬─────┘  │   │  ┌───────▼───────┐   │
                    │        │        │   │  │ Deduplicator  │   │
                    │        │        │   │  │ & Robots.txt  │   │
                    │        │        │   │  └───────────────┘   │
                    └────────┼────────┘   └──────────┬───────────┘
                             │                       │
                    ┌────────▼────────────────────────▼───────────┐
                    │              Indexer Service                 │
                    │  ┌─────────────────────────────────────┐    │
                    │  │        Sharded Inverted Index        │    │
                    │  │  ┌────────┐ ┌────────┐ ┌────────┐   │    │
                    │  │  │Shard 0 │ │Shard 1 │ │Shard N │   │    │
                    │  │  │┌──────┐│ │┌──────┐│ │┌──────┐│   │    │
                    │  │  ││Seg 0 ││ ││Seg 0 ││ ││Seg 0 ││   │    │
                    │  │  ││Seg 1 ││ ││Seg 1 ││ ││Seg 1 ││   │    │
                    │  │  │└──────┘│ │└──────┘│ │└──────┘│   │    │
                    │  │  └────────┘ └────────┘ └────────┘   │    │
                    │  └─────────────────────────────────────┘    │
                    └─────────────┬───────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
     ┌────────▼──────┐  ┌────────▼──────┐  ┌─────────▼─────┐
     │ Ranking Engine │  │  Embedding    │  │  Adaptive     │
     │ ┌────────────┐ │  │  Encoder      │  │  Cache        │
     │ │   BM25     │ │  │  (MiniLM)    │  │  (LFU/LRU)   │
     │ │   TF-IDF   │ │  │              │  │               │
     │ │  PageRank  │ │  │  Semantic    │  │  Redis-backed │
     │ │  Semantic  │ │  │  Scorer      │  │               │
     │ │   Hybrid   │ │  │              │  │               │
     │ └────────────┘ │  └──────────────┘  └───────────────┘
     └────────────────┘
```

## Component Details

### 1. API Gateway (`services/gateway/`)
- **Technology**: FastAPI with async request handling
- **Endpoints**: `/search`, `/index`, `/crawl`, `/stats`, `/health`
- **Responsibility**: Request routing, validation, CORS, response formatting

### 2. Web Crawler (`services/crawler/`)
- **Concurrency**: ThreadPoolExecutor with configurable worker count
- **URL Frontier**: Priority queue with per-host politeness (crawl delays)
- **Deduplication**: MurmurHash3 for URLs (8 bytes/URL), SHA-256 for content
- **Fault Tolerance**: Exponential backoff retry (up to 3 attempts)
- **robots.txt**: Per-host caching with TTL

### 3. Indexer Service (`services/indexer/`)
- **Pipeline**: Document → Tokenize → Add to Shard → Embed → Compute PageRank
- **Incremental**: Supports single-document real-time indexing
- **Storage**: Segment-based (Lucene-inspired), VByte compressed posting lists

### 4. Inverted Index (`core/inverted_index/`)
- **Architecture**: Segment-based with background merge (like Lucene)
- **Encoding**: Variable-byte (VByte) with delta-compressed doc IDs
- **Sharding**: Consistent hashing on doc_id across N shards
- **I/O**: Memory-mapped files for read-heavy workloads
- **Operations**: Add, delete (tombstone), search, merge

### 5. Ranking Engine (`core/ranking/`)
- **BM25**: With term frequency saturation and length normalization
- **TF-IDF**: Log-normalized TF with smoothed IDF
- **PageRank**: Iterative computation with dangling node handling
- **Semantic**: Dense embeddings via SentenceTransformers (MiniLM)
- **Hybrid**: `score = α·BM25 + β·PageRank + γ·Semantic` (configurable weights)

### 6. Query Engine (`services/query/`)
- **Understanding**: Spelling correction (edit distance), synonym expansion, intent detection
- **Parsing**: Keyword, phrase, boolean (AND/OR/NOT), field-specific queries
- **Retrieval**: Parallel fan-out across all shards
- **Highlighting**: Density-based snippet extraction with term highlighting

### 7. Cache Layer (`core/cache/`)
- **Algorithm**: Hybrid LFU/LRU with frequency threshold graduation
- **Backend**: In-memory (default) or Redis (distributed)
- **Invalidation**: Automatic on index updates (version-based)

## Data Flow

### Crawl → Index Pipeline
```
Seed URLs → Frontier → Workers → Fetch → Parse → Dedup → Store → Tokenize → Index → Embed
```

### Query Pipeline
```
Query → Understand → Parse → Tokenize → Fan-out Search → Rank → Highlight → Cache → Return
```

## Scaling Strategy

| Component | Horizontal | Vertical | Strategy |
|-----------|-----------|----------|----------|
| Crawler | ✅ | ✅ | Add workers, partition frontier by domain |
| Index | ✅ | ✅ | Add shards, increase segment buffer |
| Query | ✅ | ✅ | Replicate query nodes behind load balancer |
| Cache | ✅ | ✅ | Redis Cluster for distributed caching |
| Embeddings | ✅ | ✅ | GPU batch encoding, pre-computed vectors |
