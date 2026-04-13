"""Unit tests for the tokenizer and normalizer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.tokenizer.tokenizer import Tokenizer, PorterStemmerLight
from core.tokenizer.normalizer import TextNormalizer


class TestTextNormalizer:
    def setup_method(self):
        self.normalizer = TextNormalizer()

    def test_html_stripping(self):
        html = "<p>Hello <b>world</b></p>"
        result = self.normalizer.normalize(html)
        assert "hello world" == result

    def test_url_stripping(self):
        text = "Visit https://example.com for more info"
        result = self.normalizer.normalize(text)
        assert "https" not in result
        assert "example" not in result

    def test_lowercase(self):
        result = self.normalizer.normalize("HELLO WORLD")
        assert result == "hello world"

    def test_accent_removal(self):
        result = self.normalizer.normalize("cafe\u0301 re\u0301sume\u0301")
        assert "cafe" in result
        assert "resume" in result

    def test_empty_input(self):
        assert self.normalizer.normalize("") == ""
        assert self.normalizer.normalize(None) == ""

    def test_extract_title(self):
        html = "<html><head><title>Test Page</title></head></html>"
        assert self.normalizer.extract_title(html) == "Test Page"

    def test_extract_text_removes_scripts(self):
        html = "<p>Hello</p><script>alert('x')</script><p>World</p>"
        result = self.normalizer.extract_text(html)
        assert "alert" not in result
        assert "Hello" in result
        assert "World" in result


class TestTokenizer:
    def setup_method(self):
        self.tokenizer = Tokenizer()

    def test_basic_tokenization(self):
        tokens = self.tokenizer.tokenize("The quick brown fox jumps over the lazy dog")
        assert len(tokens) > 0
        assert "the" not in tokens  # stopword
        assert any("quick" in t or "brown" in t or "fox" in t for t in tokens)

    def test_stopword_removal(self):
        tokenizer_no_stop = Tokenizer(remove_stopwords=False)
        tokens_with = tokenizer_no_stop.tokenize("the cat is on the mat")
        tokens_without = self.tokenizer.tokenize("the cat is on the mat")
        assert len(tokens_with) > len(tokens_without)

    def test_positional_info(self):
        infos = self.tokenizer.tokenize_with_positions("hello world foo bar")
        positions = [t.position for t in infos]
        assert positions == list(range(len(positions)))

    def test_term_frequencies(self):
        tf = self.tokenizer.term_frequencies("search engine search algorithm")
        # "search" appears twice
        found_search = False
        for term, (freq, positions) in tf.items():
            if "search" in term:
                assert freq == 2
                found_search = True
        assert found_search

    def test_empty_input(self):
        assert self.tokenizer.tokenize("") == []

    def test_min_token_length(self):
        tokens = self.tokenizer.tokenize("I a an the big cat")
        assert "i" not in tokens
        assert "a" not in tokens


class TestPorterStemmer:
    def setup_method(self):
        self.stemmer = PorterStemmerLight()

    def test_common_suffixes(self):
        assert self.stemmer.stem("running") != "running"
        assert self.stemmer.stem("cats") != "cats"

    def test_short_words_unchanged(self):
        assert self.stemmer.stem("go") == "go"
        assert self.stemmer.stem("be") == "be"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
