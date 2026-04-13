"""Sample crawl experiment: crawls a small set of URLs and indexes them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from services.crawler import WebCrawler
from services.indexer import IndexerService
from services.query.engine import QueryEngine

SEED_URLS = [
    "https://en.wikipedia.org/wiki/Search_engine",
    "https://en.wikipedia.org/wiki/Web_crawler",
    "https://en.wikipedia.org/wiki/PageRank",
]


def main():
    config = Config(base_dir=Path("./data/experiment"))
    config.ensure_dirs()
    config.crawler.max_pages = 20
    config.crawler.max_threads = 4

    print("=== Phase 1: Crawling ===")
    crawler = WebCrawler(config=config.crawler, storage_dir=config.crawl_dir)
    documents = crawler.crawl(SEED_URLS, max_pages=20)
    print(f"Crawled {len(documents)} documents")

    print("\n=== Phase 2: Indexing ===")
    indexer = IndexerService(config)
    link_graph = crawler.link_graph
    indexer.index_documents(documents, link_graph)

    print("\n=== Phase 3: Querying ===")
    engine = QueryEngine(indexer, config)
    engine.update_vocabulary()

    queries = ["PageRank algorithm", "web crawler", "search engine"]
    for q in queries:
        results = engine.search(q, top_k=5)
        print(f"\nQuery: '{q}' — {results['total_results']} results "
              f"({results['timing']['total_ms']:.1f}ms)")
        for r in results["results"][:3]:
            print(f"  [{r['score']:.4f}] {r['title'][:60]}")

    indexer.close()


if __name__ == "__main__":
    main()
