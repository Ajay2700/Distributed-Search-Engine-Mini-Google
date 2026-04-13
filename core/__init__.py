"""Core search engine components — data models, index, tokenizer, ranking."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Document:
    doc_id: int
    url: str
    title: str
    content: str
    raw_html: str = ""
    outgoing_links: List[str] = field(default_factory=list)
    crawled_at: Optional[datetime] = None
    content_hash: str = ""


@dataclass
class Posting:
    doc_id: int
    frequency: int
    positions: List[int] = field(default_factory=list)


@dataclass
class SearchResult:
    doc_id: int
    url: str
    title: str
    snippet: str
    score: float
    scores_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class UserProfile:
    """Lightweight user profile for personalized ranking simulation."""
    user_id: str
    topic_preferences: Dict[str, float] = field(default_factory=dict)
    click_history: List[int] = field(default_factory=list)
    query_history: List[str] = field(default_factory=list)
