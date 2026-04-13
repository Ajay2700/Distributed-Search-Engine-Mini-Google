"""Central configuration for the distributed search engine."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CrawlerConfig:
    max_threads: int = 8
    max_pages: int = 500
    crawl_delay: float = 0.5
    request_timeout: float = 10.0
    user_agent: str = "MiniGoogleBot/1.0"
    respect_robots: bool = True
    max_retries: int = 3
    retry_backoff: float = 1.0


@dataclass
class IndexConfig:
    num_shards: int = 4
    segment_max_docs: int = 1000
    merge_factor: int = 10
    use_mmap: bool = True
    compression: bool = True


@dataclass
class RankingConfig:
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    alpha: float = 0.5      # BM25 weight
    beta: float = 0.2       # PageRank weight
    gamma: float = 0.3      # Semantic similarity weight
    top_k: int = 10
    pagerank_damping: float = 0.85
    pagerank_iterations: int = 100


@dataclass
class CacheConfig:
    max_size: int = 10000
    ttl_seconds: int = 3600
    redis_url: str = "redis://localhost:6379/0"
    use_redis: bool = False  # Falls back to in-memory if False


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    cache_embeddings: bool = True
    dimension: int = 384


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4


@dataclass
class Config:
    base_dir: Path = field(default_factory=lambda: Path("./data"))
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @property
    def index_dir(self) -> Path:
        return self.base_dir / "index"

    @property
    def crawl_dir(self) -> Path:
        return self.base_dir / "crawl"

    @property
    def embeddings_dir(self) -> Path:
        return self.base_dir / "embeddings"

    @property
    def metadata_db(self) -> Path:
        return self.base_dir / "metadata.db"

    def ensure_dirs(self):
        for d in [self.index_dir, self.crawl_dir, self.embeddings_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Config":
        """Build config from environment variables with sane defaults."""
        base = Path(os.environ.get("SEARCH_DATA_DIR", "./data"))
        return cls(
            base_dir=base,
            crawler=CrawlerConfig(
                max_threads=int(os.environ.get("CRAWLER_THREADS", "8")),
                max_pages=int(os.environ.get("CRAWLER_MAX_PAGES", "500")),
            ),
            index=IndexConfig(
                num_shards=int(os.environ.get("INDEX_SHARDS", "4")),
            ),
            cache=CacheConfig(
                redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
                use_redis=os.environ.get("USE_REDIS", "false").lower() == "true",
            ),
            embedding=EmbeddingConfig(
                model_name=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            ),
            server=ServerConfig(
                port=int(os.environ.get("API_PORT", "8000")),
            ),
        )
