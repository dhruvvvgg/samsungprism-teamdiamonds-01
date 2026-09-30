"""Retrieval pipeline stages as pure functions over score matrices. Used by BOTH the dev experiments
(src/eval/dev_*.py) and the official run (src/retrieval/search_model.py), so what is tuned on dev is
exactly what runs on test.

    dense scores (+ BM25 scores) -> [RRF hybrid] -> [rerank top-k, interpolate] -> [LLM re-judge] -> ranking

Every stage is optional and degrades to the previous stage's ranking on failure (project rule: never
return an empty or crashed result).
"""
import copy
import time

import numpy as np

from src.retrieval.hybrid_encoder import hybrid_rank, top_n
from src.retrieval.reranker import rerank_order, rerank_scores

DEFAULT_CONFIG = {
    "dense_variants": ["registry+full"],          # >1 variant => embeddings averaged
    "bm25": {"enabled": False, "keep_whole": True, "k1": 1.5, "b": 0.75},
    "rrf": {"k": 60, "w_dense": 1.0, "w_bm25": 1.0, "depth": 100},
    "rerank": {"enabled": False, "name": "qwen3-reranker-0.6b", "k": 20, "alpha": 0.0,
               "instruction": "apps", "max_length": None, "batch_size": 16, "dtype": "fp16"},
    "rejudge": {"enabled": False, "judge_top": 3, "margin_z": 0.5, "beta": 0.5},
    "final_depth": 1000,
}


def merge_config(user=None):
    """DEFAULT_CONFIG deep-merged with `user` (unknown keys are rejected to catch typos)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in (user or {}).items():
        if k not in cfg:
            raise KeyError(f"unknown pipeline config key {k!r}; valid: {sorted(cfg)}")
        if isinstance(cfg[k], dict):
            bad = set(v) - set(cfg[k])
            if bad:
                raise KeyError(f"unknown keys {sorted(bad)} in config[{k!r}]; valid: {sorted(cfg[k])}")
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def first_stage(dense_row, bm25_row, cfg):
    """One query -> (doc idx list, scores array) best first. Hybrid RRF if BM25 is enabled and available,
    otherwise dense cosine. The list is deep enough for the later stages (and MTEB's recall@100)."""
    r = cfg["rrf"]
    if cfg["bm25"]["enabled"] and bm25_row is not None:
        fused = hybrid_rank(dense_row, bm25_row, k=r["k"], w_dense=r["w_dense"], w_bm25=r["w_bm25"],
                            depth=r["depth"])
        return [d for d, _ in fused], np.array([s for _, s in fused], dtype=np.float64)
    idx = top_n(dense_row, max(r["depth"], cfg["rerank"]["k"]))
    return list(idx), dense_row[idx].astype(np.float64)


def apply_rerank(query_texts, lists, scores, doc_texts, scorer, rr_cfg):
    """Rerank the top-k of each list; the tail keeps first-stage order. Returns (lists, scores, seconds).
    On a reranker failure the input ranking is returned unchanged."""
    k, alpha = rr_cfg["k"], rr_cfg["alpha"]
    heads = [l[:k] for l in lists]
    try:
        rr, secs = rerank_scores(scorer, query_texts, heads, doc_texts, batch_size=rr_cfg["batch_size"])
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] WARNING: reranker failed ({type(exc).__name__}: {exc}); keeping first-stage ranking",
              flush=True)
        return lists, scores, 0.0
    out_l, out_s = [], []
    for l, s, r in zip(lists, scores, rr):
        new_idx, new_sc = rerank_order(l[:k], s[:k], r, alpha)
        out_l.append(new_idx + l[k:])
        # head scores sit above the tail so the merged score vector stays monotone
        top_tail = s[k] if len(s) > k else None
        head = new_sc - new_sc.min() + (top_tail if top_tail is not None else 0.0) + 1e-6
        out_s.append(np.concatenate([head, s[k:]]))
    return out_l, out_s, secs


def finalize(idx, dense_row, depth):
    """Ranked idx list padded with the remaining dense order up to `depth`, no duplicates."""
    seen = set(idx)
    tail = [d for d in top_n(dense_row, min(depth + len(seen), len(dense_row))) if d not in seen]
    return (list(idx) + tail)[:depth]


def run_pipeline(dense_S, bm25_S, query_texts, doc_texts, cfg, scorer=None, judge_call=None):
    """Batch pipeline. dense_S: (Q, D) cosine scores; bm25_S: (Q, D) or None.
    Returns (final ranked idx lists (each padded to cfg['final_depth']), timings dict in seconds)."""
    cfg = merge_config(cfg)
    lists, _, timings = run_stages(dense_S, bm25_S, query_texts, doc_texts, cfg, scorer, judge_call)
    final = [finalize(l, dense_S[i], cfg["final_depth"]) for i, l in enumerate(lists)]
    return final, timings


def run_stages(dense_S, bm25_S, query_texts, doc_texts, cfg, scorer=None, judge_call=None):
    """Same as run_pipeline but returns (lists, scores, timings) BEFORE padding to final_depth; the
    scores are what the re-judge trigger and the dev scripts need."""
    cfg = merge_config(cfg)
    timings = {}
    t0 = time.time()
    lists, scores = zip(*[first_stage(dense_S[i], None if bm25_S is None else bm25_S[i], cfg)
                          for i in range(len(dense_S))]) if len(dense_S) else ((), ())
    lists, scores = list(lists), list(scores)
    timings["first_stage"] = time.time() - t0
    if cfg["rerank"]["enabled"] and scorer is not None and lists:
        lists, scores, secs = apply_rerank(query_texts, lists, scores, doc_texts, scorer, cfg["rerank"])
        timings["rerank"] = secs
    if cfg["rejudge"]["enabled"] and lists:
        from src.agent.llm_rejudge import rejudge
        rj, t1, infos = cfg["rejudge"], time.time(), []
        for i in range(len(lists)):
            lists[i], scores[i], info = rejudge(
                query_texts[i], lists[i], scores[i], doc_texts, judge_top=rj["judge_top"],
                margin_z_threshold=rj["margin_z"], beta=rj["beta"], call=judge_call)
            infos.append(info)
        timings["rejudge"] = time.time() - t1
        timings["rejudge_triggered"] = sum(1 for x in infos if x["triggered"])
        timings["rejudge_calls"] = sum(x["n_calls"] for x in infos)
    return lists, scores, timings
