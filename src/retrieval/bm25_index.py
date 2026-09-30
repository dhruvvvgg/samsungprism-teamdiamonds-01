"""BM25 over identifier-split code tokens.

`rank_bm25.BM25Okapi` supplies the statistics (idf, doc_freqs, doc_len, avgdl). Its own `get_scores`
is a Python loop over documents per query token, far too slow for 4,000+ problem-statement queries
(hundreds of tokens each), so scoring uses an equivalent sparse-matrix product. tests/test_bm25.py
asserts the result equals `BM25Okapi.get_scores` exactly on a toy corpus.
"""
import re
from collections import Counter

import numpy as np

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def split_identifier(ident):
    """snake_case + camelCase + digit boundaries -> lowercase parts. 'maxSubArray_v2' -> [max, sub, array, v, 2]."""
    parts = []
    for chunk in ident.split("_"):
        parts += [p.lower() for p in _CAMEL.findall(chunk)]
    return parts


def tokenize_code(text, keep_whole=True, min_len=1):
    """Tokens of `text`: every identifier is split into its parts; with keep_whole the lowercase whole
    identifier is also emitted when it has more than one part (so exact identifiers still match)."""
    out = []
    for m in _IDENT.finditer(text):
        ident = m.group(0)
        parts = split_identifier(ident)
        out += parts
        if keep_whole and len(parts) > 1:
            out.append(ident.lower())
    return [t for t in out if len(t) >= min_len]


class FastBM25:
    def __init__(self, tokenized_docs, k1=1.5, b=0.75, epsilon=0.25):
        from rank_bm25 import BM25Okapi
        from scipy import sparse
        self.bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b, epsilon=epsilon)
        bm = self.bm25
        self.vocab = {w: i for i, w in enumerate(bm.idf)}
        rows, cols, vals = [], [], []
        for j, freqs in enumerate(bm.doc_freqs):
            norm = k1 * (1 - b + b * bm.doc_len[j] / bm.avgdl)
            for w, f in freqs.items():
                rows.append(j)
                cols.append(self.vocab[w])
                vals.append(bm.idf[w] * f * (k1 + 1) / (f + norm))
        self.W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(tokenized_docs), len(self.vocab)),
                                   dtype=np.float32)
        self._sparse = sparse

    def _query_matrix(self, tokenized_queries):
        rows, cols, vals = [], [], []
        for i, toks in enumerate(tokenized_queries):
            for w, c in Counter(toks).items():      # BM25Okapi counts a repeated query token repeatedly
                j = self.vocab.get(w)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(c)
        return self._sparse.csr_matrix((vals, (rows, cols)),
                                       shape=(len(tokenized_queries), len(self.vocab)), dtype=np.float32)

    def scores(self, tokenized_queries, chunk=256):
        """(n_queries, n_docs) float32 BM25 scores."""
        out = []
        for s in range(0, len(tokenized_queries), chunk):
            q = self._query_matrix(tokenized_queries[s:s + chunk])
            out.append((q @ self.W.T).toarray())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.W.shape[0]), np.float32)


def build_bm25(doc_texts, keep_whole=True, k1=1.5, b=0.75):
    return FastBM25([tokenize_code(t, keep_whole) for t in doc_texts], k1=k1, b=b)
