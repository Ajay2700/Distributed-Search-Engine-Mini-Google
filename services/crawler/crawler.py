"""Multi-threaded web crawler with URL frontier, deduplication, and fault tolerance.

Architecture:
  1. Seed URLs are added to the frontier
  2. Worker threads pull URLs from the frontier
  3. Each URL is fetched, parsed, and links are extracted
  4. New URLs are added back to the frontier (BFS with depth limit)
  5. Content and metadata are stored for indexing
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from config import CrawlerConfig
from core import Document
from core.tokenizer.normalizer import TextNormalizer
from services.crawler.dedup import ContentDeduplicator, URLDeduplicator
from services.crawler.frontier import URLFrontier
from services.crawler.robots import RobotsChecker

logger = logging.getLogger(__name__)


class WebCrawler:
    """Production-grade multi-threaded web crawler.
    
    Features:
    - Concurrent fetching with configurable thread pool
    - robots.txt compliance
    - URL + content deduplication
    - Politeness (per-host delays)
    - Retry with exponential backoff
    - Stores raw HTML + extracted text
    """

    def __init__(
        self,
        config: CrawlerConfig,
        storage_dir: Path,
        on_document: Optional[Callable[[Document], None]] = None,
    ):
        self.config = config
        self.storage_dir = storage_dir
        self.on_document = on_document

        self._url_dedup = URLDeduplicator()
        self._content_dedup = ContentDeduplicator()
        self._robots = RobotsChecker(user_agent=config.user_agent)
        self._frontier = URLFrontier(max_depth=3, default_delay=config.crawl_delay)
        self._normalizer = TextNormalizer()

        self._documents: List[Document] = []
        self._link_graph: Dict[int, List[str]] = {}
        self._url_to_doc_id: Dict[str, int] = {}
        self._doc_counter = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._init_metadata_db()

    def _init_metadata_db(self):
        db_path = self.storage_dir / "crawl_metadata.db"
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db_lock = threading.Lock()
        with self._db:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id INTEGER PRIMARY KEY,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT,
                    content_hash TEXT,
                    crawled_at TEXT,
                    content_length INTEGER,
                    status_code INTEGER
                )
            """)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS links (
                    source_doc_id INTEGER,
                    target_url TEXT,
                    FOREIGN KEY (source_doc_id) REFERENCES documents(doc_id)
                )
            """)

    def crawl(self, seed_urls: List[str], max_pages: Optional[int] = None) -> List[Document]:
        """Start crawling from seed URLs. Returns all crawled documents."""
        max_pages = max_pages or self.config.max_pages

        for url in seed_urls:
            if not self._url_dedup.mark_seen(url):
                self._frontier.add(url, priority=1.0, depth=0)

        logger.info(f"Starting crawl with {len(seed_urls)} seeds, max_pages={max_pages}")

        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            active_futures = set()

            while (len(self._documents) < max_pages
                   and not self._stop_event.is_set()):

                while len(active_futures) < self.config.max_threads:
                    entry = self._frontier.get(timeout=2.0)
                    if entry is None:
                        break
                    future = executor.submit(self._fetch_and_process, entry.url, entry.depth)
                    active_futures.add(future)

                if not active_futures:
                    if self._frontier.is_empty():
                        break
                    continue

                done = set()
                for f in active_futures:
                    if f.done():
                        done.add(f)

                for f in done:
                    active_futures.discard(f)
                    try:
                        f.result()
                    except Exception as e:
                        logger.error(f"Crawl worker error: {e}")

                if not done:
                    time.sleep(0.1)

        logger.info(f"Crawl complete: {len(self._documents)} documents collected")
        self._db.close()
        return self._documents

    def _fetch_and_process(self, url: str, depth: int):
        """Fetch a URL, extract content and links, store results."""
        for attempt in range(self.config.max_retries):
            try:
                if self.config.respect_robots and not self._robots.is_allowed(url):
                    logger.debug(f"Blocked by robots.txt: {url}")
                    return

                resp = httpx.get(
                    url,
                    timeout=self.config.request_timeout,
                    follow_redirects=True,
                    headers={"User-Agent": self.config.user_agent},
                )

                if resp.status_code != 200:
                    logger.debug(f"HTTP {resp.status_code} for {url}")
                    return

                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type:
                    return

                raw_html = resp.text
                text = self._normalizer.extract_text(raw_html)
                title = self._normalizer.extract_title(raw_html)

                if self._content_dedup.is_duplicate(text):
                    logger.debug(f"Duplicate content: {url}")
                    return

                content_hash = self._content_dedup.mark_seen(text)
                links = self._extract_links(raw_html, url)

                with self._lock:
                    doc_id = self._doc_counter
                    self._doc_counter += 1

                doc = Document(
                    doc_id=doc_id,
                    url=url,
                    title=title or url,
                    content=text,
                    raw_html=raw_html,
                    outgoing_links=links,
                    crawled_at=datetime.now(),
                    content_hash=content_hash,
                )

                with self._lock:
                    self._documents.append(doc)
                    self._url_to_doc_id[url] = doc_id
                    self._link_graph[doc_id] = links

                self._store_document(doc, resp.status_code)

                if self.on_document:
                    self.on_document(doc)

                for link in links:
                    if not self._url_dedup.mark_seen(link):
                        self._frontier.add(link, priority=1.0 / (depth + 2), depth=depth + 1)

                crawl_delay = self._robots.get_crawl_delay(url)
                if crawl_delay > 0:
                    host = urlparse(url).netloc
                    self._frontier.set_host_delay(host, crawl_delay)

                return

            except httpx.TimeoutException:
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")
                time.sleep(self.config.retry_backoff * (2 ** attempt))
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e} (attempt {attempt + 1})")
                time.sleep(self.config.retry_backoff * (2 ** attempt))

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract and normalize outgoing links from HTML."""
        links = []
        try:
            soup = BeautifulSoup(html, "lxml")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                absolute = urljoin(base_url, href)
                parsed = urlparse(absolute)
                if parsed.scheme in ("http", "https") and parsed.netloc:
                    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    links.append(clean)
        except Exception as e:
            logger.debug(f"Link extraction error for {base_url}: {e}")
        return links

    def _store_document(self, doc: Document, status_code: int):
        """Persist document to SQLite metadata DB and raw files."""
        with self._db_lock:
            try:
                self._db.execute(
                    "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (doc.doc_id, doc.url, doc.title, doc.content_hash,
                     doc.crawled_at.isoformat() if doc.crawled_at else None,
                     len(doc.content), status_code),
                )
                for link in doc.outgoing_links:
                    self._db.execute(
                        "INSERT INTO links VALUES (?, ?)",
                        (doc.doc_id, link),
                    )
                self._db.commit()
            except Exception as e:
                logger.error(f"DB store error for doc {doc.doc_id}: {e}")

        doc_dir = self.storage_dir / "pages"
        doc_dir.mkdir(exist_ok=True)
        (doc_dir / f"{doc.doc_id}.html").write_text(doc.raw_html, encoding="utf-8")
        (doc_dir / f"{doc.doc_id}.txt").write_text(doc.content, encoding="utf-8")

    @property
    def link_graph(self) -> Dict[int, List[int]]:
        """Return link graph as {source_doc_id: [target_doc_ids]}."""
        graph: Dict[int, List[int]] = {}
        for src_id, target_urls in self._link_graph.items():
            target_ids = []
            for url in target_urls:
                if url in self._url_to_doc_id:
                    target_ids.append(self._url_to_doc_id[url])
            if target_ids:
                graph[src_id] = target_ids
        return graph

    @property
    def url_to_doc_id(self) -> Dict[str, int]:
        return dict(self._url_to_doc_id)

    def stop(self):
        self._stop_event.set()
