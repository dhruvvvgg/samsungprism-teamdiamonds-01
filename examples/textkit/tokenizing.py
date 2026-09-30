"""Tokenizing and normalising text."""
import re
import unicodedata

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def tokenize(text, lowercase=True):
    """Split text into word tokens, dropping punctuation."""
    tokens = WORD_RE.findall(text)
    return [t.lower() for t in tokens] if lowercase else tokens


def split_sentences(text):
    """Split a paragraph into sentences on terminal punctuation."""
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def strip_accents(text):
    """Remove diacritics, leaving the base characters (cafe from café)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_whitespace(text):
    """Collapse every run of whitespace into a single space and trim the ends."""
    return " ".join(text.split())


def ngrams(tokens, n=2):
    """All contiguous n-grams of a token list, as tuples."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def truncate_words(text, max_words, suffix="..."):
    """Cut text to at most max_words words, appending a suffix when anything was removed."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + suffix
