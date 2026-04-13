"""Memory usage benchmark: tracks RAM consumption of index components."""
from __future__ import annotations

import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psutil
from rich.console import Console
from rich.table import Table

from config import Config
from core import Document
from services.indexer import IndexerService

console = Console()


def _get_memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_memory_benchmark(config: Config, doc_counts: list[int] | None = None):
    """Measure memory consumption at different corpus sizes."""
    console.print("\n[bold cyan]Memory Usage Benchmark[/bold cyan]")

    if doc_counts is None:
        doc_counts = [10, 50, 100, 200, 500]

    table = Table(title="Memory Usage")
    table.add_column("Documents", style="cyan")
    table.add_column("Baseline (MB)", style="yellow")
    table.add_column("After Index (MB)", style="red")
    table.add_column("Delta (MB)", style="green")
    table.add_column("Per-Doc (KB)", style="blue")

    for count in doc_counts:
        baseline = _get_memory_mb()

        tmpdir = tempfile.mkdtemp()
        bench_config = Config(base_dir=Path(tmpdir))
        bench_config.ensure_dirs()
        bench_config.index.num_shards = 2

        indexer = IndexerService(bench_config)
        docs = _generate_docs(count)
        indexer.index_documents(docs)

        after = _get_memory_mb()
        delta = after - baseline
        per_doc = (delta * 1024) / count if count > 0 else 0

        table.add_row(
            str(count),
            f"{baseline:.1f}",
            f"{after:.1f}",
            f"{delta:.1f}",
            f"{per_doc:.1f}",
        )
        indexer.close()

    console.print(table)


def _generate_docs(n: int) -> list[Document]:
    content_template = (
        "This is benchmark content for measuring memory usage of the search engine index. "
        "It contains various terms related to distributed systems, algorithms, and databases. "
    )
    return [
        Document(
            doc_id=i,
            url=f"https://mem-bench.example.com/doc{i}",
            title=f"Memory Benchmark Doc {i}",
            content=content_template * 10,
            crawled_at=datetime.now(),
        )
        for i in range(n)
    ]


if __name__ == "__main__":
    run_memory_benchmark(Config())
