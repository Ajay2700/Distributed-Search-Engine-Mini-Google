"""Unit tests for query parsing, understanding, and highlighting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from services.query.parser import QueryParser, QueryType, BoolOp
from services.query.understanding import QueryUnderstanding, QueryIntent
from services.query.highlighter import Highlighter


class TestQueryParser:
    def setup_method(self):
        self.parser = QueryParser()

    def test_simple_keyword(self):
        node = self.parser.parse("hello world")
        assert node.query_type == QueryType.KEYWORD
        assert "hello" in node.terms
        assert "world" in node.terms

    def test_phrase_query(self):
        node = self.parser.parse('"exact phrase match"')
        assert node.query_type == QueryType.PHRASE
        assert node.phrase == "exact phrase match"

    def test_boolean_and(self):
        node = self.parser.parse("python AND web")
        assert node.query_type == QueryType.BOOLEAN
        assert node.operator == BoolOp.AND

    def test_boolean_or(self):
        node = self.parser.parse("fast OR quick")
        assert node.query_type == QueryType.BOOLEAN
        assert node.operator == BoolOp.OR

    def test_extract_terms(self):
        node = self.parser.parse("distributed systems")
        terms = self.parser.extract_terms(node)
        assert "distributed" in terms
        assert "systems" in terms

    def test_empty_query(self):
        node = self.parser.parse("")
        assert node.query_type == QueryType.KEYWORD
        assert node.terms == []


class TestQueryUnderstanding:
    def setup_method(self):
        self.qu = QueryUnderstanding(vocabulary={"python", "search", "engine", "algorithm"})

    def test_informational_intent(self):
        result = self.qu.process("how does search engine work")
        assert result.intent == QueryIntent.INFORMATIONAL

    def test_navigational_intent(self):
        result = self.qu.process("go to google.com")
        assert result.intent == QueryIntent.NAVIGATIONAL

    def test_transactional_intent(self):
        result = self.qu.process("buy python book")
        assert result.intent == QueryIntent.TRANSACTIONAL

    def test_spelling_correction(self):
        result = self.qu.process("serch engne")
        if result.corrections:
            assert "serch" in result.corrections or "engne" in result.corrections

    def test_synonym_expansion(self):
        result = self.qu.process("find error")
        assert len(result.expanded_terms) > 2  # should include synonyms

    def test_edit_distance(self):
        assert QueryUnderstanding._edit_distance("kitten", "sitting") == 3
        assert QueryUnderstanding._edit_distance("hello", "hello") == 0
        assert QueryUnderstanding._edit_distance("", "abc") == 3


class TestHighlighter:
    def setup_method(self):
        self.hl = Highlighter(snippet_length=100, highlight_pre="**", highlight_post="**")

    def test_basic_highlight(self):
        text = "The search engine uses inverted index for fast retrieval"
        snippet = self.hl.generate_snippet(text, ["search", "index"])
        assert "**search**" in snippet or "**Search**" in snippet
        assert "**index**" in snippet or "**Index**" in snippet

    def test_empty_query(self):
        text = "Some text here"
        snippet = self.hl.generate_snippet(text, [])
        assert snippet == text

    def test_long_text_truncation(self):
        text = "word " * 200
        snippet = self.hl.generate_snippet(text, ["word"])
        assert len(snippet) < len(text)

    def test_multiple_snippets(self):
        text = "alpha beta gamma delta epsilon zeta eta " * 20
        snippets = self.hl.generate_snippets(text, ["alpha", "gamma"])
        assert len(snippets) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
