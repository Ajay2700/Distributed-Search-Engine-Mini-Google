"""Query parser: transforms raw query strings into structured query objects.

Supports:
  - Simple keyword queries
  - Phrase queries (quoted)
  - Boolean operators (AND, OR, NOT)
  - Field-specific queries (title:, url:)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class QueryType(Enum):
    KEYWORD = "keyword"
    PHRASE = "phrase"
    BOOLEAN = "boolean"


class BoolOp(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass
class QueryNode:
    """A node in the parsed query tree."""
    query_type: QueryType
    terms: List[str] = field(default_factory=list)
    phrase: str = ""
    operator: Optional[BoolOp] = None
    children: List["QueryNode"] = field(default_factory=list)
    field_name: Optional[str] = None  # for field-specific queries


class QueryParser:
    """Parses raw query strings into structured query trees."""

    _PHRASE_RE = re.compile(r'"([^"]+)"')
    _FIELD_RE = re.compile(r"(\w+):(\S+)")

    def parse(self, raw_query: str) -> QueryNode:
        """Parse a raw query string into a QueryNode tree."""
        raw_query = raw_query.strip()
        if not raw_query:
            return QueryNode(query_type=QueryType.KEYWORD, terms=[])

        phrases = self._PHRASE_RE.findall(raw_query)
        if phrases and len(phrases) == 1 and f'"{phrases[0]}"' == raw_query.strip():
            return QueryNode(query_type=QueryType.PHRASE, phrase=phrases[0])

        if " AND " in raw_query or " OR " in raw_query or " NOT " in raw_query:
            return self._parse_boolean(raw_query)

        field_match = self._FIELD_RE.match(raw_query)
        if field_match:
            field_name = field_match.group(1)
            value = field_match.group(2)
            return QueryNode(
                query_type=QueryType.KEYWORD,
                terms=[value],
                field_name=field_name,
            )

        terms = raw_query.lower().split()
        return QueryNode(query_type=QueryType.KEYWORD, terms=terms)

    def _parse_boolean(self, query: str) -> QueryNode:
        """Parse boolean query into a tree."""
        parts = re.split(r"\s+(AND|OR|NOT)\s+", query)

        if len(parts) < 3:
            return QueryNode(query_type=QueryType.KEYWORD, terms=query.lower().split())

        left = self.parse(parts[0])
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break
            op_str = parts[i]
            right = self.parse(parts[i + 1])
            op = BoolOp(op_str)

            left = QueryNode(
                query_type=QueryType.BOOLEAN,
                operator=op,
                children=[left, right],
            )

        return left

    def extract_terms(self, node: QueryNode) -> List[str]:
        """Flatten a query tree into a list of search terms."""
        terms: List[str] = []
        if node.terms:
            terms.extend(node.terms)
        if node.phrase:
            terms.extend(node.phrase.lower().split())
        for child in node.children:
            terms.extend(self.extract_terms(child))
        return terms
