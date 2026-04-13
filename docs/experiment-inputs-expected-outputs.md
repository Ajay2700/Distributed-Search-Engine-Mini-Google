# Mini Google Experiment Inputs, Expected Outputs, Terms & Conditions

This file provides ready-to-use test inputs and expected outcomes for the complete project (API, CLI, and Streamlit UI).

## 1) Terms & Conditions (Before Testing)

- Python 3.10+ should be installed.
- Install dependencies:
  - `pip install -r requirements.txt`
- Start backend API:
  - `python main.py serve --port 8000`
- Start Streamlit UI:
  - `python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true`
- Keep API URL in Streamlit sidebar as:
  - `http://127.0.0.1:8000`
- Internet is required for live crawl tests.
- First-time semantic model load may be slower due to model cache/download.
- If `pytest` is not recognized in PowerShell, use:
  - `python -m pytest tests/ -v`

## 2) Quick Health Checks

### Input
- Open:
  - `http://127.0.0.1:8000/health`
  - `http://127.0.0.1:8000/docs`
  - `http://127.0.0.1:8501`

### Desired Output
- `/health` returns JSON with `"status": "healthy"`.
- `/docs` opens Swagger UI.
- Streamlit app opens with tabs: Search, Index Document, Crawl, Stats, Health & Weights.

## 3) Search Inputs and Desired Outputs

Use these in Streamlit Search tab (Top K = 5 or 10):

### A) Core relevance queries

1. **Input:** `distributed systems consensus`  
   **Desired output:** result list appears, top docs related to distributed systems, timing shown in ms.

2. **Input:** `machine learning neural networks`  
   **Desired output:** ML-related docs ranked in top results, snippet highlights query terms.

3. **Input:** `python web framework`  
   **Desired output:** FastAPI/Python web content appears in top results.

4. **Input:** `database indexing performance`  
   **Desired output:** database/indexing content appears; scores and timing visible.

5. **Input:** `search engine ranking algorithm`  
   **Desired output:** search/ranking architecture content appears in top results.

### B) Query understanding tests

6. **Input:** `distrbuted systms`  
   **Desired output:** rewritten/corrected query visible in `query_info`; valid results returned.

7. **Input:** `machne lernning`  
   **Desired output:** spelling correction behavior appears; intent shown.

8. **Input:** `"search engine architecture"`  
   **Desired output:** phrase intent still returns relevant search engine architecture results.

### C) Cache behavior test

9. **Input:** run the same query 2-3 times (example: `search engine ranking algorithm`)  
   **Desired output:** later run may show `from_cache = true` and typically lower total latency.

## 4) Index Document Test (Manual Document Injection)

### Input (Index Document tab)
- URL: `https://example.com/vector-search-guide`
- Title: `Vector Search and Hybrid Ranking`
- Content:
  - `Vector search uses embeddings to retrieve semantically similar documents. Hybrid ranking combines BM25 lexical matching with PageRank authority and semantic similarity for better relevance in modern search engines.`

### Desired Output
- Success message with `doc_id`.
- JSON response contains `"status": "indexed"`.
- After indexing, searching `vector search` or `hybrid ranking bm25 semantic` should surface this document.

## 5) Crawl Test

### Input (Crawl tab)
- Seed URL: `https://en.wikipedia.org/wiki/Search_engine`
- Max Pages: `10` (start small)

### Desired Output
- Crawl completes with `"status": "completed"`.
- `"documents_crawled"` is greater than or equal to 1.
- Follow-up search returns fresh crawl-based content.

## 6) Ranking Weights Experiment

Use Health & Weights tab:

1. **Lexical-heavy**
   - Input: `alpha=0.7, beta=0.2, gamma=0.1`
   - Desired output: BM25 influence increases.

2. **Authority-heavy**
   - Input: `alpha=0.4, beta=0.5, gamma=0.1`
   - Desired output: PageRank influence increases.

3. **Semantic-heavy**
   - Input: `alpha=0.4, beta=0.1, gamma=0.5`
   - Desired output: semantic matches rise for meaning-similar queries.

Pass check: same query run under different weights should show ranking shifts.

## 7) Stats Tab Checks

### Input
- Click `Refresh Stats`.

### Desired Output
- Shows:
  - `Total Documents`
  - `Shards`
  - Cache JSON
  - Ranking config JSON

## 8) CLI Test Inputs and Desired Outputs

1. **Input:** `python main.py demo`  
   **Desired output:** indexes sample docs and prints ranked results for demo queries.

2. **Input:** `python main.py search "distributed systems consensus"`  
   **Desired output:** query runs with ranking output and timing.

3. **Input:** `python -m pytest tests/ -v`  
   **Desired output:** test suite passes (project currently validates with full passing suite).

## 9) Pass/Fail Checklist

Project is considered healthy if:
- API health endpoint is successful.
- Streamlit loads and performs search without errors.
- At least one index and one crawl operation complete successfully.
- Query results include snippets, scores, and timing.
- Stats endpoint returns valid JSON.
- Automated tests pass via `python -m pytest tests/ -v`.

## 10) Common Troubleshooting Conditions

- If browser shows old/invalid page response, hard refresh (`Ctrl+F5`) or use incognito.
- If `streamlit` command is not recognized, use:
  - `python -m streamlit run streamlit_app.py --server.port 8501 --server.headless true`
- If port `8000` or `8501` is occupied, stop old process or use another port.
- If crawl returns low documents, try another seed URL or increase `max_pages`.
