"""Dense-dense fusion: combine two F2LLM sizes' cached embeddings (e.g. 1.7B + 0.6B).

Same rank-fusion machinery as the BM25+dense hybrid (hybrid_encoder.rrf_fuse / top_n), just fed two
dense score lists instead of one dense + one sparse, plus a simpler normalise-then-weighted-average
fusion for comparison.

CPU ONLY, no model loads: both presets' embeddings are read from the on-disk cache. If either is
missing the script says exactly which and exits (pass --allow-encode to build it instead, which does
cost a GPU pass).

    !python src/eval/dev_fuse.py --preset f2llm-v2-1.7b --preset-b f2llm-v2-0.6b --n-queries 1000

--preset is model A (the stronger one, the base the adoption rule is judged against); --preset-b is
model B. Rows are tagged with model A's preset so they sit alongside its other rows in the dev table.
"""
import argparse
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval import dev_metrics as M  # noqa: E402
from src.eval.dev_lib import (BASELINE_VARIANT, CacheMissError, DevContext, add_common_args,  # noqa: E402
                              cached_embeddings, print_code_version, record, render_table)
from src.retrieval.hybrid_encoder import rrf_fuse, score_average_fuse, top_n  # noqa: E402

K_GRID = [10, 30, 60, 100]          # RRF constant, same grid as the BM25+dense sweep
W_GRID = [0.25, 0.5, 0.75, 1.0]     # weight on model B (model A fixed at 1.0), same grid
AVG_W_GRID = [0.1, 0.25, 0.5, 0.75]  # weight on model B for normalise-then-average fusion


def scores_for(preset, ctx, variant, allow_encode):
    """(n_queries, n_docs) cosine scores for `preset` on ctx's query slice, from cache only."""
    if allow_encode and preset == ctx.args.preset:
        S, _ = ctx.query_scores([variant])          # ctx's own preset may build its cache if permitted
        return S
    doc_emb, query_emb = cached_embeddings(preset, ctx.doc_texts, ctx.query_texts, variant)
    return query_emb[ctx.idx] @ doc_emb.T


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--preset-b", default="f2llm-v2-0.6b", help="the second (weaker) model to fuse in")
    ap.add_argument("--depth", type=int, default=100, help="candidates taken from each model before RRF")
    ap.add_argument("--norm", default="zscore", choices=["zscore", "minmax"],
                    help="per-query normalisation for the score-averaging fusion")
    ap.add_argument("--allow-encode", action="store_true",
                    help="allow a GPU encoding pass if --preset's cache is missing (B must still be cached)")
    ap.set_defaults(n_queries=1000)   # the fixed slice every prior comparison used; 0 still = all tune
    args = ap.parse_args()
    print_code_version("fuse")
    ctx = DevContext(args)
    a_name, b_name = args.preset, args.preset_b
    if a_name == b_name:
        raise SystemExit(f"--preset and --preset-b are both {a_name!r}; fusing a model with itself is a no-op")

    try:
        S_a = scores_for(a_name, ctx, BASELINE_VARIANT, args.allow_encode)
        S_b = scores_for(b_name, ctx, BASELINE_VARIANT, False)   # B is never encoded, only read
    except CacheMissError as exc:
        raise SystemExit(f"\n[fuse] CACHE MISS -- nothing was encoded, no GPU time spent.\n  {exc}\n") from exc
    print(f"[fuse] loaded cached embeddings for A={a_name} and B={b_name}; "
          f"{S_a.shape[0]} queries x {S_a.shape[1]} docs, CPU only, no model loaded", flush=True)

    ranks_a = M.ranks_from_scores(S_a, ctx.rel_of())
    ranks_b = M.ranks_from_scores(S_b, ctx.rel_of())
    sum_a, sum_b = M.summarize(ranks_a), M.summarize(ranks_b)
    print(f"[fuse] A ({a_name}) alone: NDCG@10={sum_a['ndcg@10']:.4f} | "
          f"B ({b_name}) alone: NDCG@10={sum_b['ndcg@10']:.4f}", flush=True)
    if sum_b["ndcg@10"] > sum_a["ndcg@10"]:
        print("[fuse] NOTE: B scores higher than A here; the adoption rule is judged against the better "
              "of the two, so it is judged against B.", flush=True)
    base = ranks_a if sum_a["ndcg@10"] >= sum_b["ndcg@10"] else ranks_b
    base_name = a_name if sum_a["ndcg@10"] >= sum_b["ndcg@10"] else b_name

    # both standalone models as reference rows (excluded from adoption by dev_select)
    for nm, rk_ in ((a_name, ranks_a), (b_name, ranks_b)):
        record(ctx, f"{nm} alone (dense)", "fuse", rk_, base, {}, latency_ms=0.0,
               notes=f"reference row, not a candidate (single model; adoption base is {base_name})")

    lists_a = [list(top_n(row, args.depth)) for row in S_a]
    lists_b = [list(top_n(row, args.depth)) for row in S_b]

    for k in K_GRID:
        for w in W_GRID:
            t0 = time.time()
            fused = [[d for d, _ in rrf_fuse([la, lb], [1.0, w], k)] for la, lb in zip(lists_a, lists_b)]
            ms = 1000 * (time.time() - t0) / len(fused)
            record(ctx, f"RRF dense+dense k={k} w_B={w}", "fuse", ctx.ranks_from_lists(fused), base, {},
                   latency_ms=ms, extra={"fusion": "rrf", "k": k, "w_b": w, "depth": args.depth,
                                         "model_a": a_name, "model_b": b_name},
                   notes="fusion is a dev experiment only; not wired into the official pipeline config")

    for w in AVG_W_GRID:
        t0 = time.time()
        S_f = score_average_fuse(S_a, S_b, w_b=w, how=args.norm)
        ms = 1000 * (time.time() - t0) / len(S_f)
        record(ctx, f"score-avg dense+dense w_B={w} ({args.norm})", "fuse",
               M.ranks_from_scores(S_f, ctx.rel_of()), base, {}, latency_ms=ms,
               extra={"fusion": "score_average", "w_b": w, "norm": args.norm,
                      "model_a": a_name, "model_b": b_name},
               notes="fusion is a dev experiment only; not wired into the official pipeline config")

    render_table()


if __name__ == "__main__":
    main()
