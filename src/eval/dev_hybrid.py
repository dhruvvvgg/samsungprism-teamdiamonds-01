"""Step 2 (Phase 2): BM25 (identifier-split code tokens) fused with dense via RRF; sweep on the tune set.

Sweeps: tokenizer (split identifiers with / without also keeping the whole identifier) x RRF constant k x
BM25 weight (dense weight fixed at 1). Each row is judged against the CURRENT base (dense with the query
variants already chosen in outputs/dev/chosen.json, else the F2LLM baseline) and also reports its delta
vs the raw F2LLM baseline. +ms/query = BM25 scoring + fusion time per query (CPU; measured here).

    !python src/eval/dev_hybrid.py --device cuda
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval import dev_metrics as M  # noqa: E402
from src.eval.dev_lib import BASELINE_VARIANT, DevContext, add_common_args, load_base_config, record, render_table  # noqa: E402
from src.retrieval.bm25_index import build_bm25, tokenize_code  # noqa: E402
from src.retrieval.hybrid_encoder import rrf_fuse, top_n  # noqa: E402

K_GRID = [10, 30, 60, 100]
W_GRID = [0.25, 0.5, 0.75, 1.0]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--depth", type=int, default=100, help="candidates taken from each retriever before fusion")
    args = ap.parse_args()
    ctx = DevContext(args)
    cfg = load_base_config(args.config)
    variants = cfg["dense_variants"]
    print(f"[hybrid] dense variants in base config: {variants}", flush=True)

    S_ref, _ = ctx.query_scores([BASELINE_VARIANT])
    ref = M.ranks_from_scores(S_ref, ctx.rel_of())
    S_dense, _ = ctx.query_scores(variants)
    base = M.ranks_from_scores(S_dense, ctx.rel_of())
    dense_lists = [list(top_n(row, args.depth)) for row in S_dense]
    qtexts = [ctx.query_texts[i] for i in ctx.idx]

    for keep_whole in (True, False):
        t0 = time.time()
        bm = build_bm25(ctx.doc_texts, keep_whole=keep_whole, k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])
        index_s = time.time() - t0
        t0 = time.time()
        Sb = bm.scores([tokenize_code(t, keep_whole) for t in qtexts])
        score_ms = 1000 * (time.time() - t0) / len(qtexts)
        bm_lists = [list(top_n(row, args.depth)) for row in Sb]
        tag = f"keep_whole={keep_whole}"
        print(f"[hybrid] {tag}: index build {index_s:.1f}s, BM25 scoring {score_ms:.2f} ms/query", flush=True)
        record(ctx, f"BM25 only ({tag})", "hybrid", M.ranks_from_scores(Sb, ctx.rel_of()), base,
               {"bm25": {"enabled": False}}, latency_ms=score_ms, ref_ranks=ref,
               notes="reference row, not a candidate (BM25 alone)")
        for k in K_GRID:
            for w in W_GRID:
                t0 = time.time()
                fused = [[d for d, _ in rrf_fuse([dl, bl], [1.0, w], k)] for dl, bl in zip(dense_lists, bm_lists)]
                fuse_ms = 1000 * (time.time() - t0) / len(fused)
                ranks = ctx.ranks_from_lists(fused)
                patch = {"bm25": {"enabled": True, "keep_whole": keep_whole, "k1": cfg["bm25"]["k1"],
                                  "b": cfg["bm25"]["b"]},
                         "rrf": {"k": k, "w_dense": 1.0, "w_bm25": w, "depth": args.depth}}
                record(ctx, f"RRF k={k} w_bm25={w} ({tag})", "hybrid", ranks, base, patch,
                       latency_ms=score_ms + fuse_ms, ref_ranks=ref)
    render_table()


if __name__ == "__main__":
    main()
