# Mini Google — Production-Grade Distributed Search Engine

A scalable search engine demonstrating distributed systems, information retrieval, and system design principles. Built from scratch without Elasticsearch or Solr.

## Architecture

```
User Query → API Gateway → Query Understanding → Tokenizer → Sharded Index → Hybrid Ranker → Results
                                                                ↑
Seed URLs  → Crawler  → Indexer → Segments → Embeddings → PageRank
```

**6 Novelty Features:**
1. **Hybrid Ranking** — BM25 + PageRank + Semantic (MiniLM embeddings)
2. **Query Understanding** — Spelling correction, synonym expansion, intent detection
3. **Incremental Indexing** — Segment-based (Lucene-inspired) with background merge
4. **Personalized Search** — User profile-based ranking boost
5. **Distributed Sharding** — Hash-partitioned index with parallel fan-out queries
6. **Adaptive Caching** — LFU/LRU hybrid with auto-invalidation on index updates

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| API | FastAPI + Uvicorn |
| Index | Custom inverted index (VByte, mmap, segments) |
| Ranking | BM25, TF-IDF, PageRank, Semantic (MiniLM) |
| Embeddings | SentenceTransformers |
| Cache | In-memory LFU/LRU + optional Redis |
| Queue | Kafka (optional) |
| Metadata | SQLite |
| CLI | Click + Rich |

## Quick Start

```bash
# Install dependencies
cd search-engine
pip install -r requirements.txt

# Run demo with sample documents
python main.py demo

# Start the API server
python main.py serve --port 8000

# Start Streamlit experiment UI
streamlit run streamlit_app.py

# Crawl and index
python main.py crawl https://en.wikipedia.org/wiki/Search_engine --max-pages 20
python main.py index --recompute-pagerank

# Search
python main.py search "distributed systems consensus"

# Run benchmarks
python main.py bench

# Run tests
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/search?q=...&top_k=10` | Search query |
| POST | `/search` | Search (JSON body) |
| POST | `/index` | Add document to index |
| POST | `/crawl` | Trigger web crawl |
| GET | `/stats` | Index statistics |
| GET | `/health` | Health check |
| PUT | `/ranking/weights` | Update ranking weights |

### Example API Response

```json
{
  "results": [
    {
      "doc_id": 4,
      "url": "https://example.com/search-engines",
      "title": "Search Engine Architecture and Ranking",
      "snippet": "Modern **search** engines use inverted **index**es...",
      "score": 0.8542,
      "scores": {"bm25": 0.92, "pagerank": 0.65, "semantic": 0.78}
    }
  ],
  "total_results": 5,
  "query_info": {
    "original": "search index",
    "rewritten": "search index",
    "intent": "informational",
    "corrections": {},
    "expanded_terms": ["search", "index", "find", "locate"]
  },
  "timing": {
    "understanding_ms": 0.12,
    "retrieval_ms": 1.45,
    "ranking_ms": 2.30,
    "total_ms": 4.21
  }
}
```

## Docker Deployment

```bash
cd search-engine/infra
docker-compose up -d
```

This starts the search API, Redis, and Kafka.

## Project Structure

```
search-engine/
├── main.py                    # CLI entry point
├── config.py                  # Central configuration
├── core/
│   ├── inverted_index/        # Custom index with segments + sharding
│   │   ├── index.py           # Thread-safe inverted index
│   │   ├── segment.py         # Immutable segment read/write
│   │   ├── shard.py           # Distributed sharding + fan-out
│   │   └── storage.py         # VByte codec, posting lists
│   ├── tokenizer/             # Tokenization pipeline
│   │   ├── tokenizer.py       # Tokenizer with stemming
│   │   ├── normalizer.py      # HTML/text normalization
│   │   └── stopwords.py       # Stopword list
│   ├── ranking/               # All ranking algorithms
│   │   ├── bm25.py            # Okapi BM25
│   │   ├── tfidf.py           # TF-IDF
│   │   ├── pagerank.py        # PageRank + Personalized PR
│   │   ├── semantic.py        # Embedding-based scoring
│   │   └── hybrid.py          # Combined ranking
│   ├── embeddings/            # Dense vector encoding
│   │   └── encoder.py         # SentenceTransformers wrapper
│   └── cache/                 # Adaptive caching
│       └── adaptive_cache.py  # LFU/LRU hybrid
├── services/
│   ├── crawler/               # Web crawler service
│   │   ├── crawler.py         # Multi-threaded crawler
│   │   ├── frontier.py        # URL priority queue
│   │   ├── dedup.py           # URL + content dedup
│   │   └── robots.py          # robots.txt parser
│   ├── indexer/               # Indexing service
│   │   └── indexer.py         # Full indexing pipeline
│   ├── query/                 # Query processing
│   │   ├── engine.py          # Query orchestrator
│   │   ├── parser.py          # Query parser (bool, phrase)
│   │   ├── understanding.py   # Spell check, synonyms, intent
│   │   └── highlighter.py     # Snippet generation
│   ├── ranking/               # Ranking microservice
│   │   └── service.py         # Ranking service wrapper
│   └── gateway/               # API layer
│       └── api.py             # FastAPI endpoints
├── infra/
│   ├── docker-compose.yml     # Full stack deployment
│   ├── Dockerfile
│   ├── kafka/config.py        # Kafka producer/consumer
│   └── redis/config.py        # Redis connection factory
├── benchmarks/                # Performance measurement
│   ├── latency.py             # P50/P95/P99 query latency
│   ├── throughput.py          # Indexing docs/sec
│   └── memory.py              # RAM consumption profiling
├── tests/                     # Comprehensive test suite
│   ├── test_tokenizer.py
│   ├── test_index.py
│   ├── test_ranking.py
│   ├── test_query.py
│   └── test_integration.py
├── experiments/
│   └── sample_crawl.py        # End-to-end crawl experiment
└── docs/
    ├── architecture.md        # System architecture diagram
    └── tradeoffs.md           # Design decisions explained
```

## Key Design Decisions

| Decision | Why |
|----------|-----|
| BM25 over pure TF-IDF | Term frequency saturation + document length normalization |
| Segment-based index | Immutable segments enable concurrent reads/writes |
| Hash-based sharding | Even distribution, simple to implement and reason about |
| VByte encoding | Good compression/speed tradeoff for posting lists |
| Hybrid ranking | Combines precision (BM25), authority (PR), meaning (semantic) |
| Adaptive cache | LFU/LRU hybrid prevents both frequency and recency bias |

See [docs/tradeoffs.md](docs/tradeoffs.md) for full analysis.

## Benchmarking

Run `python main.py bench` to measure:
- **Query Latency**: P50, P95, P99 response times
- **Indexing Throughput**: Documents indexed per second
- **Memory Usage**: RAM consumption per document

## Testing

```bash
# All tests
pytest tests/ -v

# Specific test files
pytest tests/test_index.py -v
pytest tests/test_ranking.py -v
pytest tests/test_integration.py -v
```

## Interview Talking Points

1. **Why not Elasticsearch?** — Custom implementation demonstrates understanding of inverted indexes, posting list encoding, and segment merging at a fundamental level.

2. **How does sharding work?** — `hash(doc_id) % N` distributes documents. Queries fan out to all shards in parallel via ThreadPoolExecutor. Results are merged and re-ranked.

3. **How does hybrid ranking improve results?** — BM25 alone misses semantic matches ("car" vs "automobile"). PageRank alone favors popular but irrelevant pages. The hybrid score captures relevance, authority, and meaning.

4. **How does incremental indexing work?** — New documents are buffered in memory, then flushed as immutable segments. Background merge combines small segments. Deletes use tombstones until the next merge.

5. **What happens if a shard goes down?** — Queries degrade gracefully: other shards return partial results. The failed shard's data persists on disk and is recovered on restart.
# Distributed-Search-Engine-Mini-Google
