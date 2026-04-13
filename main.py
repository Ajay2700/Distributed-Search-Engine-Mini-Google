"""Mini Google CLI — command-line interface for the distributed search engine.

Commands:
  crawl   — Crawl web pages from seed URLs
  index   — Index crawled documents
  search  — Execute search queries
  serve   — Start the REST API server
  demo    — Run with sample dataset
  bench   — Run benchmarks
  stats   — Show index statistics
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).parent))

from config import Config

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--data-dir", default="./data", help="Data directory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, data_dir, verbose):
    """Mini Google — Distributed Search Engine"""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    config = Config(base_dir=Path(data_dir))
    config.ensure_dirs()
    ctx.obj["config"] = config


@cli.command()
@click.argument("seed_urls", nargs=-1, required=True)
@click.option("--max-pages", default=50, help="Maximum pages to crawl")
@click.option("--threads", default=4, help="Number of crawler threads")
@click.pass_context
def crawl(ctx, seed_urls, max_pages, threads):
    """Crawl web pages starting from seed URLs."""
    from services.crawler import WebCrawler

    config = ctx.obj["config"]
    config.crawler.max_threads = threads
    config.crawler.max_pages = max_pages

    console.print(Panel(
        f"[bold blue]Starting crawl[/bold blue]\n"
        f"Seeds: {', '.join(seed_urls)}\n"
        f"Max pages: {max_pages} | Threads: {threads}",
        title="Crawler",
    ))

    crawler = WebCrawler(config=config.crawler, storage_dir=config.crawl_dir)
    documents = crawler.crawl(list(seed_urls), max_pages=max_pages)

    meta_path = config.base_dir / "crawl_results.json"
    results = {
        "total_documents": len(documents),
        "urls": [d.url for d in documents],
        "link_graph": {str(k): v for k, v in crawler.link_graph.items()},
        "url_to_doc_id": crawler.url_to_doc_id,
    }
    meta_path.write_text(json.dumps(results, indent=2))

    console.print(f"\n[green]Crawled {len(documents)} documents[/green]")
    console.print(f"Results saved to {meta_path}")


@cli.command()
@click.option("--recompute-pagerank", is_flag=True, help="Recompute PageRank")
@click.pass_context
def index(ctx, recompute_pagerank):
    """Index previously crawled documents."""
    from core import Document
    from services.indexer import IndexerService

    config = ctx.obj["config"]
    indexer = IndexerService(config)

    crawl_results_path = config.base_dir / "crawl_results.json"
    if not crawl_results_path.exists():
        console.print("[red]No crawl results found. Run 'crawl' first.[/red]")
        return

    crawl_data = json.loads(crawl_results_path.read_text())
    pages_dir = config.crawl_dir / "pages"

    documents = []
    for url, doc_id_str in crawl_data.get("url_to_doc_id", {}).items():
        doc_id = int(doc_id_str) if isinstance(doc_id_str, str) else doc_id_str
        text_file = pages_dir / f"{doc_id}.txt"
        if text_file.exists():
            content = text_file.read_text(encoding="utf-8")
            documents.append(Document(
                doc_id=doc_id, url=url, title=url, content=content,
            ))

    if not documents:
        console.print("[red]No documents to index.[/red]")
        return

    link_graph = None
    if recompute_pagerank:
        raw_graph = crawl_data.get("link_graph", {})
        link_graph = {int(k): v for k, v in raw_graph.items()}

    console.print(f"[blue]Indexing {len(documents)} documents...[/blue]")
    start = time.time()
    indexer.index_documents(documents, link_graph)
    elapsed = time.time() - start

    console.print(f"[green]Indexed {len(documents)} documents in {elapsed:.2f}s[/green]")
    indexer.close()


@cli.command()
@click.argument("query")
@click.option("--top-k", default=10, help="Number of results")
@click.pass_context
def search(ctx, query, top_k):
    """Search the index."""
    from services.indexer import IndexerService
    from services.query.engine import QueryEngine

    config = ctx.obj["config"]
    indexer = IndexerService(config)
    engine = QueryEngine(indexer, config)
    engine.update_vocabulary()

    results = engine.search(query, top_k=top_k)

    _display_results(results)
    indexer.close()


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
@click.pass_context
def serve(ctx, host, port):
    """Start the REST API server."""
    import uvicorn
    from services.gateway.api import create_app

    config = ctx.obj["config"]
    app = create_app(config)

    console.print(Panel(
        f"[bold green]Starting API server[/bold green]\n"
        f"URL: http://{host}:{port}\n"
        f"Docs: http://{host}:{port}/docs",
        title="API Gateway",
    ))

    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.pass_context
def demo(ctx):
    """Run a demo with sample documents."""
    from core import Document
    from services.indexer import IndexerService
    from services.query.engine import QueryEngine

    config = ctx.obj["config"]

    console.print(Panel(
        "[bold cyan]Mini Google Demo[/bold cyan]\n"
        "Indexing sample documents and running queries",
        title="Demo",
    ))

    sample_docs = _get_sample_documents()

    indexer = IndexerService(config)

    link_graph = {
        0: [1, 2, 3],
        1: [0, 2],
        2: [0, 3, 4],
        3: [1, 4],
        4: [0, 2],
        5: [0, 1, 6],
        6: [5, 7],
        7: [6, 0],
    }

    console.print(f"[blue]Indexing {len(sample_docs)} sample documents...[/blue]")
    indexer.index_documents(sample_docs, link_graph)

    engine = QueryEngine(indexer, config)
    engine.update_vocabulary()

    queries = [
        "distributed systems consensus",
        "machine learning neural networks",
        "python web framework",
        "database indexing performance",
        "search engine ranking algorithm",
    ]

    for q in queries:
        console.print(f"\n[bold yellow]Query: {q}[/bold yellow]")
        results = engine.search(q, top_k=5)
        _display_results(results)

    indexer.close()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show index statistics."""
    from services.indexer import IndexerService

    config = ctx.obj["config"]
    indexer = IndexerService(config)

    table = Table(title="Index Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Documents", str(indexer.index.total_docs))
    table.add_row("Number of Shards", str(config.index.num_shards))
    table.add_row("Avg Doc Length", f"{indexer.index.avg_doc_length:.1f}")
    table.add_row("Semantic Embeddings", str(indexer.semantic_scorer.indexed_count))

    console.print(table)
    indexer.close()


@cli.command()
@click.pass_context
def bench(ctx):
    """Run performance benchmarks."""
    from benchmarks.latency import run_latency_benchmark
    from benchmarks.throughput import run_throughput_benchmark
    from benchmarks.memory import run_memory_benchmark

    config = ctx.obj["config"]

    console.print(Panel("[bold]Running Benchmarks[/bold]", title="Benchmarks"))

    run_latency_benchmark(config)
    run_throughput_benchmark(config)
    run_memory_benchmark(config)


def _display_results(response: dict):
    """Display search results in a rich table."""
    results = response.get("results", [])
    timing = response.get("timing", {})
    query_info = response.get("query_info", {})

    if query_info.get("corrections"):
        console.print(f"[dim]Did you mean: {query_info.get('rewritten', '')}[/dim]")
    console.print(f"[dim]Intent: {query_info.get('intent', 'unknown')} | "
                  f"Total: {timing.get('total_ms', 0):.1f}ms[/dim]")

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan", width=40)
    table.add_column("Score", style="green", width=8)
    table.add_column("Snippet", width=60)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.get("title", "")[:40],
            f"{r.get('score', 0):.4f}",
            r.get("snippet", "")[:60],
        )

    console.print(table)

    timing_parts = " | ".join(f"{k}: {v}ms" for k, v in timing.items() if k != "total_ms")
    console.print(f"[dim]{timing_parts}[/dim]")


def _get_sample_documents():
    """Generate sample documents for the demo."""
    from datetime import datetime
    from core import Document

    samples = [
        {
            "title": "Introduction to Distributed Systems",
            "content": (
                "Distributed systems are collections of independent computers that appear to users "
                "as a single coherent system. Key challenges include consensus, fault tolerance, "
                "and consistency. The CAP theorem states that a distributed system cannot simultaneously "
                "provide consistency, availability, and partition tolerance. Popular consensus algorithms "
                "include Paxos and Raft. Distributed hash tables like Chord and Kademlia provide "
                "scalable key-value storage across nodes."
            ),
            "url": "https://example.com/distributed-systems",
        },
        {
            "title": "Machine Learning Fundamentals",
            "content": (
                "Machine learning is a subset of artificial intelligence that enables systems to learn "
                "from data. Neural networks, inspired by biological neurons, form the basis of deep "
                "learning. Convolutional neural networks excel at image recognition, while recurrent "
                "neural networks handle sequential data. Transformers have revolutionized natural "
                "language processing with attention mechanisms. Gradient descent optimization drives "
                "training of these models."
            ),
            "url": "https://example.com/machine-learning",
        },
        {
            "title": "Python Web Development with FastAPI",
            "content": (
                "FastAPI is a modern Python web framework for building APIs. It leverages Python type "
                "hints for automatic validation and documentation. Built on Starlette and Pydantic, "
                "FastAPI offers high performance comparable to Node.js and Go. Features include "
                "automatic OpenAPI documentation, dependency injection, and async support. Django "
                "and Flask remain popular alternatives for full-stack web development in Python."
            ),
            "url": "https://example.com/python-web",
        },
        {
            "title": "Database Indexing and Query Optimization",
            "content": (
                "Database indexes are data structures that improve query performance by reducing disk "
                "I/O. B-tree indexes support range queries efficiently, while hash indexes excel at "
                "point lookups. Composite indexes cover multiple columns. Query optimizers use "
                "cost-based planning to choose execution strategies. Proper indexing can improve "
                "query performance by orders of magnitude. PostgreSQL and MySQL offer sophisticated "
                "query planners."
            ),
            "url": "https://example.com/database-indexing",
        },
        {
            "title": "Search Engine Architecture and Ranking",
            "content": (
                "Modern search engines use inverted indexes to map terms to documents. BM25 scoring "
                "improves on TF-IDF by adding term frequency saturation and document length "
                "normalization. PageRank computes page authority from the web link graph. Semantic "
                "search uses neural embeddings to capture meaning beyond exact keyword matching. "
                "Query understanding includes spelling correction and intent detection."
            ),
            "url": "https://example.com/search-engines",
        },
        {
            "title": "Cloud Computing and Microservices",
            "content": (
                "Cloud computing provides on-demand computing resources over the internet. "
                "Microservices architecture decomposes applications into small, independent services. "
                "Kubernetes orchestrates containerized workloads across clusters. Service mesh "
                "technologies like Istio manage inter-service communication. Event-driven "
                "architectures using Kafka enable asynchronous processing between services."
            ),
            "url": "https://example.com/cloud-computing",
        },
        {
            "title": "Data Structures and Algorithms",
            "content": (
                "Fundamental data structures include arrays, linked lists, trees, and graphs. "
                "Hash tables provide O(1) average-case lookups using hash functions. Binary search "
                "trees maintain sorted order for efficient searching. Graph algorithms like Dijkstra "
                "and BFS solve shortest path and traversal problems. Tries are specialized trees "
                "for prefix-based string searching."
            ),
            "url": "https://example.com/dsa",
        },
        {
            "title": "Cryptography and Network Security",
            "content": (
                "Cryptography protects data through encryption algorithms. AES provides symmetric "
                "encryption for data at rest, while RSA enables asymmetric key exchange. TLS/SSL "
                "secures network communications using certificate-based authentication. Hash "
                "functions like SHA-256 ensure data integrity. Zero-knowledge proofs enable "
                "verification without revealing underlying data."
            ),
            "url": "https://example.com/cryptography",
        },
    ]

    return [
        Document(
            doc_id=i,
            url=s["url"],
            title=s["title"],
            content=s["content"],
            crawled_at=datetime.now(),
        )
        for i, s in enumerate(samples)
    ]


if __name__ == "__main__":
    cli()
