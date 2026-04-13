"""Indexing throughput benchmark: measures documents indexed per second."""
from __future__ import annotations

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

console = Console()


def run_throughput_benchmark(config: Config, doc_counts: list[int] | None = None):
    """Measure indexing throughput for various corpus sizes."""
    console.print("\n[bold cyan]Indexing Throughput Benchmark[/bold cyan]")

    if doc_counts is None:
        doc_counts = [10, 50, 100, 200, 500]

    table = Table(title="Indexing Throughput")
    table.add_column("Documents", style="cyan")
    table.add_column("Time (s)", style="yellow")
    table.add_column("Docs/sec", style="green")
    table.add_column("Terms/doc (avg)", style="blue")

    for count in doc_counts:
        docs = _generate_docs(count)

        tmpdir = tempfile.mkdtemp()
        bench_config = Config(base_dir=Path(tmpdir))
        bench_config.ensure_dirs()
        bench_config.index.num_shards = 2

        indexer = IndexerService(bench_config)

        start = time.perf_counter()
        indexer.index_documents(docs)
        elapsed = time.perf_counter() - start

        throughput = count / elapsed if elapsed > 0 else 0
        avg_terms = sum(len(d.content.split()) for d in docs) / count

        table.add_row(
            str(count),
            f"{elapsed:.3f}",
            f"{throughput:.1f}",
            f"{avg_terms:.0f}",
        )
        indexer.close()

    console.print(table)


def _generate_docs(n: int) -> list[Document]:
    topics = [
        "distributed computing cloud services microservices containers",
        "machine learning deep neural networks transformers attention",
        "database optimization indexing query planning execution",
        "web development frontend backend APIs REST GraphQL",
        "security cryptography authentication authorization access",
    ]
    return [
        Document(
            doc_id=i,
            url=f"https://bench.example.com/doc{i}",
            title=f"Benchmark Document {i}",
            content=f"{topics[i % len(topics)]} " * 50,
            crawled_at=datetime.now(),
        )
        for i in range(n)
    ]


if __name__ == "__main__":
    run_throughput_benchmark(Config())
