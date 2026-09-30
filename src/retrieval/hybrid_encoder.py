"""Hybrid retrieval: weighted Reciprocal Rank Fusion of dense and BM25 candidate lists.

    fused(d) = sum_i  w_i / (k + rank_i(d))        rank_i is 1-based within list i; absent => no term
"""
import numpy as np


def top_n(scores_row, n):
    """Indices of the n highest scores, best first (stable for ties by lower index), as plain Python
    ints (not np.int64) — doc indices flow into dict keys, JSON cache keys, and MTEB doc-id lookups
    throughout the pipeline, so they must never carry a numpy dtype that json.dumps can choke on."""
    n = min(n, len(scores_row))
    part = np.argpartition(-scores_row, n - 1)[:n]
    return [int(i) for i in part[np.lexsort((part, -scores_row[part]))]]


def rrf_fuse(ranked_lists, weights=None, k=60):
    """ranked_lists: list of lists of doc indices (best first). Returns [(doc_idx, fused_score)] best first.
    Ties on fused score are broken by the best (lowest) rank in the first list that contains the doc,
    so the result is deterministic."""
    weights = weights or [1.0] * len(ranked_lists)
    score, first_seen = {}, {}
    for li, (lst, w) in enumerate(zip(ranked_lists, weights)):
        for r, d in enumerate(lst, start=1):
            score[d] = score.get(d, 0.0) + w / (k + r)
            first_seen.setdefault(d, (li, r))
    return sorted(score.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def hybrid_rank(dense_row, bm25_row, k=60, w_dense=1.0, w_bm25=1.0, depth=100):
    """Fuse the top-`depth` of a dense score row and a BM25 score row. Returns [(doc_idx, rrf_score)]."""
    return rrf_fuse([list(top_n(dense_row, depth)), list(top_n(bm25_row, depth))],
                    [w_dense, w_bm25], k)


def normalize_rows(S, how="zscore"):
    """Per-query (row-wise) score normalisation, so two models' scores become comparable before being
    averaged. Rows with no spread map to all-zeros ('zscore') / all-0.5 ('minmax') instead of NaN."""
    S = np.asarray(S, dtype=np.float64)
    if how == "zscore":
        sd = S.std(axis=1, keepdims=True)
        return np.where(sd > 1e-12, (S - S.mean(axis=1, keepdims=True)) / np.maximum(sd, 1e-12), 0.0)
    if how == "minmax":
        lo, hi = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
        span = hi - lo
        return np.where(span > 1e-12, (S - lo) / np.maximum(span, 1e-12), 0.5)
    raise ValueError(f"unknown normalisation {how!r}; use 'zscore' or 'minmax'")


def score_average_fuse(S_a, S_b, w_b=0.5, how="zscore"):
    """Weighted average of two score matrices after per-query normalisation:
    (1 - w_b) * norm(S_a) + w_b * norm(S_b). w_b=0 is A alone, w_b=1 is B alone."""
    return (1.0 - w_b) * normalize_rows(S_a, how) + w_b * normalize_rows(S_b, how)
