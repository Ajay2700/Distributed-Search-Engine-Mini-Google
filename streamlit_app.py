"""Streamlit UI for experimenting with the Mini Google API."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
import pandas as pd
import streamlit as st


DEFAULT_API_BASE = "http://127.0.0.1:8000"


def api_get(base_url: str, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call a GET endpoint and return JSON."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{base_url}{path}", params=params)
        response.raise_for_status()
        return response.json()


def api_post(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call a POST endpoint and return JSON."""
    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{base_url}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def api_put(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call a PUT endpoint and return JSON."""
    with httpx.Client(timeout=30.0) as client:
        response = client.put(f"{base_url}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def init_state() -> None:
    """Initialize session state containers used across tabs."""
    if "search_logs" not in st.session_state:
        st.session_state["search_logs"] = []
    if "last_search_result" not in st.session_state:
        st.session_state["last_search_result"] = None


def render_search(base_url: str) -> None:
    st.subheader("Search Experiment")
    query = st.text_input("Query", placeholder="e.g. distributed systems consensus")
    col1, col2 = st.columns(2)
    top_k = col1.slider("Top K", min_value=1, max_value=30, value=10)
    user_id = col2.text_input("User ID (optional)", placeholder="user_123")

    if st.button("Run Search", type="primary"):
        if not query.strip():
            st.warning("Enter a query first.")
            return
        try:
            params = {"q": query, "top_k": top_k}
            if user_id.strip():
                params["user_id"] = user_id.strip()
            result = api_get(base_url, "/search", params=params)
            st.session_state["last_search_result"] = result
            st.session_state["search_logs"].append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "query": query,
                    "top_k": top_k,
                    "total_results": result.get("total_results", 0),
                    "total_ms": result.get("timing", {}).get("total_ms", 0),
                    "from_cache": result.get("from_cache", False),
                }
            )

            st.success(f"Found {result.get('total_results', 0)} results")

            query_info = result.get("query_info", {})
            timing = result.get("timing", {})
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.markdown(f"**Intent:** `{query_info.get('intent', 'unknown')}`")
                st.markdown(f"**Rewritten:** `{query_info.get('rewritten', query)}`")
            with meta_col2:
                st.markdown(f"**Total Time:** `{timing.get('total_ms', 'n/a')} ms`")
                st.markdown(f"**From Cache:** `{result.get('from_cache', False)}`")

            st.markdown("### Results")
            rows = []
            for item in result.get("results", []):
                rows.append(
                    {
                        "doc_id": item.get("doc_id"),
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "score": item.get("score", 0.0),
                        "bm25": item.get("scores", {}).get("bm25", 0.0),
                        "pagerank": item.get("scores", {}).get("pagerank", 0.0),
                        "semantic": item.get("scores", {}).get("semantic", 0.0),
                        "snippet": item.get("snippet", ""),
                    }
                )

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                st.markdown("### Charts")
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.markdown("**Top Result Score Breakdown**")
                    top_result = df.iloc[0]
                    score_df = pd.DataFrame(
                        {
                            "component": ["bm25", "pagerank", "semantic"],
                            "score": [top_result["bm25"], top_result["pagerank"], top_result["semantic"]],
                        }
                    ).set_index("component")
                    st.bar_chart(score_df)

                with chart_col2:
                    st.markdown("**Timing Breakdown (ms)**")
                    timing_df = pd.DataFrame(
                        {
                            "stage": ["understanding", "parse", "retrieval", "ranking", "snippet", "total"],
                            "ms": [
                                timing.get("understanding_ms", 0),
                                timing.get("parse_ms", 0),
                                timing.get("retrieval_ms", 0),
                                timing.get("ranking_ms", 0),
                                timing.get("snippet_ms", 0),
                                timing.get("total_ms", 0),
                            ],
                        }
                    ).set_index("stage")
                    st.bar_chart(timing_df)

                st.markdown("### Export Current Results")
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                result_json_bytes = json.dumps(result, indent=2).encode("utf-8")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "Download Results CSV",
                        data=csv_bytes,
                        file_name="search_results.csv",
                        mime="text/csv",
                    )
                with dl_col2:
                    st.download_button(
                        "Download Response JSON",
                        data=result_json_bytes,
                        file_name="search_response.json",
                        mime="application/json",
                    )
            else:
                st.info("No results returned for this query.")

            with st.expander("Raw JSON response"):
                st.code(json.dumps(result, indent=2), language="json")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Search request failed: {exc}")

    st.markdown("### Experiment Log")
    logs_df = pd.DataFrame(st.session_state.get("search_logs", []))
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True)
        st.line_chart(logs_df.set_index("timestamp")[["total_ms"]])
        st.download_button(
            "Download Experiment Log CSV",
            data=logs_df.to_csv(index=False).encode("utf-8"),
            file_name="experiment_log.csv",
            mime="text/csv",
        )
    else:
        st.caption("Run a few searches to build an experiment log.")


def render_index(base_url: str) -> None:
    st.subheader("Index Single Document")
    with st.form("index_form"):
        url = st.text_input("Document URL", placeholder="https://example.com/article")
        title = st.text_input("Title", placeholder="Document title")
        content = st.text_area("Content", height=180, placeholder="Paste document text...")
        submitted = st.form_submit_button("Index Document", type="primary")

    if submitted:
        if not (url.strip() and title.strip() and content.strip()):
            st.warning("URL, title, and content are required.")
            return
        payload = {"url": url.strip(), "title": title.strip(), "content": content.strip()}
        try:
            result = api_post(base_url, "/index", payload)
            st.success(f"Indexed successfully. Doc ID: {result.get('doc_id')}")
            st.json(result)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Index request failed: {exc}")


def render_crawl(base_url: str) -> None:
    st.subheader("Crawl + Index")
    with st.form("crawl_form"):
        seeds_text = st.text_area(
            "Seed URLs (one per line)",
            height=120,
            placeholder="https://en.wikipedia.org/wiki/Search_engine",
        )
        max_pages = st.slider("Max Pages", min_value=1, max_value=200, value=20)
        submitted = st.form_submit_button("Start Crawl", type="primary")

    if submitted:
        seeds = [line.strip() for line in seeds_text.splitlines() if line.strip()]
        if not seeds:
            st.warning("Provide at least one seed URL.")
            return
        payload = {"seed_urls": seeds, "max_pages": max_pages}
        try:
            result = api_post(base_url, "/crawl", payload)
            st.success(f"Crawl completed. Documents crawled: {result.get('documents_crawled', 0)}")
            st.json(result)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Crawl request failed: {exc}")


def render_stats(base_url: str) -> None:
    st.subheader("System Stats")
    if st.button("Refresh Stats"):
        st.session_state["refresh_stats"] = True

    if st.session_state.get("refresh_stats", True):
        try:
            stats = api_get(base_url, "/stats")
            st.metric("Total Documents", stats.get("total_documents", 0))
            st.metric("Shards", stats.get("num_shards", 0))
            st.markdown("### Cache")
            st.json(stats.get("cache", {}))
            st.markdown("### Ranking Config")
            st.json(stats.get("ranking_config", {}))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Stats request failed: {exc}")


def render_health(base_url: str) -> None:
    st.subheader("Health & Ranking Weights")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check Health", type="primary"):
            try:
                health = api_get(base_url, "/health")
                st.success(f"API status: {health.get('status', 'unknown')}")
                st.json(health)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Health check failed: {exc}")

    with col2:
        st.markdown("#### Update Ranking Weights")
        alpha = st.number_input("Alpha (BM25)", min_value=0.0, value=0.5, step=0.1)
        beta = st.number_input("Beta (PageRank)", min_value=0.0, value=0.3, step=0.1)
        gamma = st.number_input("Gamma (Semantic)", min_value=0.0, value=0.2, step=0.1)
        if st.button("Apply Weights"):
            try:
                payload = {"alpha": alpha, "beta": beta, "gamma": gamma}
                response = api_put(base_url, "/ranking/weights", payload)
                st.success("Ranking weights updated.")
                st.json(response)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Weight update failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Mini Google Lab", page_icon="🔎", layout="wide")
    init_state()
    st.title("Mini Google - Streamlit Experiment UI")
    st.caption("Use this dashboard to run search, crawl, indexing, and API experiments.")

    with st.sidebar:
        st.header("Connection")
        base_url = st.text_input("API Base URL", value=DEFAULT_API_BASE)
        st.markdown("API docs: `/docs`")
        st.markdown("Health: `/health`")

    tabs = st.tabs(["Search", "Index Document", "Crawl", "Stats", "Health & Weights"])
    with tabs[0]:
        render_search(base_url)
    with tabs[1]:
        render_index(base_url)
    with tabs[2]:
        render_crawl(base_url)
    with tabs[3]:
        render_stats(base_url)
    with tabs[4]:
        render_health(base_url)


if __name__ == "__main__":
    main()
