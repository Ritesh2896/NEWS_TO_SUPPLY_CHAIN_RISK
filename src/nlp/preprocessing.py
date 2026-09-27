"""Text preprocessing and cleaning utilities for local NLP processing."""
from __future__ import annotations
import re
from typing import Sequence

# Common HTML tags regex
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# URLs regex
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Excess whitespace regex
_WHITESPACE_RE = re.compile(r"\s+")
# Sentence splitter regex (handles periods, exclamation, questions with trailing space)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def clean_text(text: str) -> str:
    """Clean raw text by removing HTML tags, URLs, and collapsing whitespace.
    
    Preserves case, punctuation, and sentence boundaries for NER and parsing.
    """
    if not text:
        return ""
    # Strip HTML tags
    t = _HTML_TAG_RE.sub(" ", str(text))
    # Strip URLs
    t = _URL_RE.sub(" ", t)
    # Replace non-printable ASCII or control characters
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", t)
    # Collapse multiple whitespaces
    t = _WHITESPACE_RE.sub(" ", t).strip()
    # Normalize space before punctuation
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    return t


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into distinct sentences."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    raw_sentences = _SENTENCE_SPLIT_RE.split(cleaned)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
    return sentences if sentences else [cleaned]


def extract_context_window(text: str, match_start: int, match_end: int, window_chars: int = 150) -> str:
    """Extract a window of context surrounding a character span, bounded at sentence or word breaks."""
    if not text:
        return ""
    start = max(0, match_start - window_chars)
    end = min(len(text), match_end + window_chars)
    
    # Expand slightly to word boundaries
    while start > 0 and not text[start].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
        
    snippet = text[start:end].strip()
    return snippet
