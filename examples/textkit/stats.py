"""Counting and summarising over token streams."""
import math
from collections import Counter

from examples.textkit.tokenizing import ngrams, tokenize


def word_frequencies(text, top_n=None):
    """Count how often each word appears, most common first."""
    counts = Counter(tokenize(text))
    return counts.most_common(top_n) if top_n else counts.most_common()


def term_frequency(tokens):
    """Relative frequency of each term: count divided by the total number of tokens."""
    total = len(tokens)
    if total == 0:
        return {}
    return {term: count / total for term, count in Counter(tokens).items()}


def inverse_document_frequency(documents):
    """IDF per term across a list of token lists, smoothed so unseen terms do not divide by zero."""
    n_docs = len(documents)
    seen = Counter()
    for doc in documents:
        seen.update(set(doc))
    return {term: math.log((1 + n_docs) / (1 + df)) + 1.0 for term, df in seen.items()}


def tfidf(documents):
    """TF-IDF weights for every document, as a list of {term: weight} dicts."""
    idf = inverse_document_frequency(documents)
    return [{t: tf * idf.get(t, 0.0) for t, tf in term_frequency(doc).items()} for doc in documents]


def cosine_similarity(a, b):
    """Cosine similarity between two sparse weight dictionaries."""
    shared = set(a) & set(b)
    numerator = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def most_common_bigrams(text, top_n=5):
    """The most frequent adjacent word pairs in a piece of text."""
    return Counter(ngrams(tokenize(text), 2)).most_common(top_n)


def average_sentence_length(sentences):
    """Mean number of words per sentence."""
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)
