"""Query latency benchmark: measures P50, P95, P99 response times."""
from __future__ import annotations

import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from config import Config
from core import Document
from services.indexer import IndexerService
from services.query.engine import QueryEngine

console = Console()

BENCHMARK_QUERIES = [
    "distributed systems consensus",
    "machine learning neural networks",
    "python web framework",
    "database indexing performance",
    "search engine ranking",
    "cloud computing microservices",
    "data structures algorithms",
    "cryptography security",
    "operating system kernel",
    "computer networking protocols",
    "artificial intelligence",
    "deep learning transformers",
    "web scraping automation",
    "api design best practices",
    "software architecture patterns",
]


def _create_sample_docs(n: int = 50) -> list[Document]:
    topics = [
        "distributed systems", "machine learning", "web development",
        "database optimization", "search engines", "cloud computing",
        "algorithms", "security", "networking", "operating systems",
    ]
    docs = []
    for i in range(n):
        topic = topics[i % len(topics)]
        docs.append(Document(
            doc_id=i,
            url=f"https://example.com/doc{i}",
            title=f"Document about {topic} #{i}",
            content=f"This is a comprehensive article about {topic}. " * 20,
            crawled_at=datetime.now(),
        ))
    return docs


def run_latency_benchmark(config: Config, num_iterations: int = 50):
    """Run query latency benchmark and report P50/P95/P99."""
    console.print("\n[bold cyan]Query Latency Benchmark[/bold cyan]")

    tmpdir = tempfile.mkdtemp()
    bench_config = Config(base_dir=Path(tmpdir))
    bench_config.ensure_dirs()
    bench_config.index.num_shards = 2

    indexer = IndexerService(bench_config)
    docs = _create_sample_docs(50)
    link_graph = {i: [(i + 1) % 50, (i + 2) % 50] for i in range(50)}
    indexer.index_documents(docs, link_graph)

    engine = QueryEngine(indexer, bench_config)
    engine.update_vocabulary()

    latencies: list[float] = []
    for _ in range(num_iterations):
        for query in BENCHMARK_QUERIES:
            start = time.perf_counter()
            engine.search(query, top_k=10, use_cache=False)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = statistics.mean(latencies)
    std = statistics.stdev(latencies) if len(latencies) > 1 else 0

    table = Table(title="Query Latency (ms)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total queries", str(len(latencies)))
    table.add_row("Average", f"{avg:.2f} ms")
    table.add_row("Std Dev", f"{std:.2f} ms")
    table.add_row("P50", f"{p50:.2f} ms")
    table.add_row("P95", f"{p95:.2f} ms")
    table.add_row("P99", f"{p99:.2f} ms")
    table.add_row("Min", f"{min(latencies):.2f} ms")
    table.add_row("Max", f"{max(latencies):.2f} ms")
    console.print(table)

    indexer.close()
    return {"p50": p50, "p95": p95, "p99": p99, "avg": avg}


if __name__ == "__main__":
    run_latency_benchmark(Config())
