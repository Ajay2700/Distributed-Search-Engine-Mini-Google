"""FastAPI gateway: REST API for the search engine.

Endpoints:
  GET  /search?q=...&top_k=...  — Search
  POST /index                    — Add document to index
  GET  /health                   — Health check
  GET  /stats                    — Index statistics
  POST /crawl                    — Trigger crawl
  PUT  /ranking/weights          — Update ranking weights
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config
from core import UserProfile
from services.crawler import WebCrawler
from services.indexer import IndexerService
from services.query.engine import QueryEngine

logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    user_id: Optional[str] = None


class IndexDocumentRequest(BaseModel):
    url: str
    title: str
    content: str


class CrawlRequest(BaseModel):
    seed_urls: List[str]
    max_pages: int = 100


class RankingWeightsRequest(BaseModel):
    alpha: float
    beta: float
    gamma: float


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or Config.from_env()
    config.ensure_dirs()

    app = FastAPI(
        title="Mini Google — Distributed Search Engine",
        description="Production-grade search engine with hybrid ranking",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    indexer = IndexerService(config)
    query_engine = QueryEngine(indexer, config)
    query_engine.update_vocabulary()

    app.state.config = config
    app.state.indexer = indexer
    app.state.query_engine = query_engine

    @app.get("/")
    async def root():
        return {
            "message": "Mini Google API is running",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/search")
    async def search(
        q: str = Query(..., description="Search query"),
        top_k: int = Query(10, ge=1, le=100),
        user_id: Optional[str] = None,
    ):
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        user_profile = None
        if user_id:
            user_profile = UserProfile(user_id=user_id)

        results = query_engine.search(q, top_k=top_k, user_profile=user_profile)
        return results

    @app.post("/search")
    async def search_post(request: SearchRequest):
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        user_profile = None
        if request.user_id:
            user_profile = UserProfile(user_id=request.user_id)

        results = query_engine.search(
            request.query, top_k=request.top_k, user_profile=user_profile
        )
        return results

    @app.post("/index")
    async def index_document(request: IndexDocumentRequest):
        from core import Document
        from datetime import datetime

        doc_id = int(time.time() * 1000) % (10**9)
        doc = Document(
            doc_id=doc_id,
            url=request.url,
            title=request.title,
            content=request.content,
        )
        indexer.index_single(doc)
        query_engine.update_vocabulary()
        query_engine.invalidate_cache()

        return {"status": "indexed", "doc_id": doc_id}

    @app.post("/crawl")
    async def crawl(request: CrawlRequest):
        crawler = WebCrawler(
            config=config.crawler,
            storage_dir=config.crawl_dir,
        )
        documents = crawler.crawl(request.seed_urls, max_pages=request.max_pages)

        if documents:
            link_graph = crawler.link_graph
            indexer.index_documents(documents, link_graph)
            query_engine.update_vocabulary()
            query_engine.invalidate_cache()

        return {
            "status": "completed",
            "documents_crawled": len(documents),
        }

    @app.put("/ranking/weights")
    async def update_ranking_weights(request: RankingWeightsRequest):
        total = request.alpha + request.beta + request.gamma
        if total <= 0:
            raise HTTPException(status_code=400, detail="Weights must sum to > 0")
        return {
            "status": "updated",
            "weights": {
                "alpha": request.alpha / total,
                "beta": request.beta / total,
                "gamma": request.gamma / total,
            },
        }

    @app.get("/stats")
    async def stats():
        return {
            "total_documents": indexer.index.total_docs,
            "num_shards": config.index.num_shards,
            "cache": query_engine.cache_stats,
            "ranking_config": {
                "alpha": config.ranking.alpha,
                "beta": config.ranking.beta,
                "gamma": config.ranking.gamma,
            },
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": time.time()}

    @app.on_event("shutdown")
    async def shutdown():
        indexer.close()

    return app
