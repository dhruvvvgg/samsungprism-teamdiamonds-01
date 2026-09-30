"""Rank-based dev metrics. Every AppsRetrieval query has exactly ONE relevant document, so a
system is fully described by `ranks`: the 1-based rank of that document per query, capped at
RANK_CAP (= "not retrieved / beyond top-1000"). All metrics, the improved/worsened counts and the
adoption rule derive from that array, which keeps every experiment directly comparable.
"""
import numpy as np

RANK_CAP = 1001


def ranks_from_scores(S, target_idx):
    """S: (Q, D) score matrix, target_idx: (Q,) column of the relevant doc. Rank = 1 + #docs scoring
    strictly higher (ties resolved optimistically, as in an argsort-free evaluation)."""
    S = np.asarray(S)
    target = S[np.arange(len(target_idx)), target_idx][:, None]
    return np.minimum((S > target).sum(axis=1) + 1, RANK_CAP)


def rank_from_list(ranked_doc_idx, target):
    """1-based position of `target` in a ranked list of doc indices; RANK_CAP if absent."""
    try:
        return min(list(ranked_doc_idx).index(target) + 1, RANK_CAP)
    except ValueError:
        return RANK_CAP


def summarize(ranks):
    r = np.asarray(ranks)
    return {
        "ndcg@10": float(np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0).mean()),
        "mrr@10": float(np.where(r <= 10, 1.0 / r, 0.0).mean()),
        "recall@1": float((r <= 1).mean()),
        "recall@10": float((r <= 10).mean()),
        "recall@100": float((r <= 100).mean()),
        "n": int(len(r)),
    }


def ndcg10_per_query(ranks):
    r = np.asarray(ranks)
    return np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)


def compare(ranks, base_ranks, n_boot=2000, seed=0):
    """Paired comparison against a baseline on the SAME queries.

    improved / worsened = the relevant doc moved to a strictly better / worse rank. Also returns the
    NDCG@10 delta with a 95% paired-bootstrap CI (resampling queries)."""
    r, b = np.asarray(ranks), np.asarray(base_ranks)
    assert r.shape == b.shape, "ranks and baseline must cover the same queries"
    d = ndcg10_per_query(r) - ndcg10_per_query(b)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "improved": int((r < b).sum()), "worsened": int((r > b).sum()), "same": int((r == b).sum()),
        "delta_ndcg@10": float(d.mean()),
        "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
    }


def adopt(cmp, min_gain=0.005, max_worsened_ratio=0.5):
    """Adopt a variant only if (1) the NDCG@10 gain is at least `min_gain`, (2) its 95% paired CI
    excludes zero, and (3) it does not worsen lopsidedly: worsened <= ratio * improved."""
    if cmp["improved"] == 0:
        return False
    return (cmp["delta_ndcg@10"] >= min_gain and cmp["ci_low"] > 0
            and cmp["worsened"] <= max_worsened_ratio * cmp["improved"])
