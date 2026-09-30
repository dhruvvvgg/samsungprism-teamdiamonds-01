"""A. Confidence-gated reranking: pay for the cross-encoder only where the first stage is unsure.

    !python src/eval/dev_gated_rerank.py --preset f2llm-v2-1.7b --device cuda

Always-reranking the 1,000-query dev slice is worth +0.0072 NDCG@10 (0.9299 -> 0.9371) and costs
several hours of GPU on the full test split -- which is exactly why it was measured and then left out of
the submitted system. But most queries do not need it: when the first stage puts one document far ahead
of the second, a reranker almost never changes the answer, and scoring those pairs is wasted work.

The gate is the **z-scored gap between rank 1 and rank 2** of the first stage. Scores are z-normalised
per query across the top candidates, so the gap is comparable between queries; a small gap means the top
two are nearly tied and the ordering is the reranker's to fix.

    gate: rerank query q  iff  z(s1) - z(s2) < threshold

Why this is cheap to sweep: every candidate pair is scored ONCE and cached (the existing
`dev_rerank.cached_rerank`), then each threshold is evaluated offline by choosing, per query, between the
reranked order and the first-stage order. So a sweep of ten thresholds costs one reranking pass, not ten.
The reported time per query is derived from the measured per-pair cost times the pairs a gate would
actually have scored in production.

Selection happens on the tune set; `--use-holdout` confirms ONE chosen threshold on the reserved
1,000-query holdout, which is never used to pick anything.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval import dev_metrics as M  # noqa: E402
from src.eval.dev_lib import BASELINE_VARIANT, DevContext, add_common_args, print_code_version  # noqa: E402
from src.eval.dev_rerank import cached_rerank  # noqa: E402
from src.retrieval.hybrid_encoder import top_n  # noqa: E402
from src.retrieval.reranker import rerank_order  # noqa: E402
from src.utils_io import write_json_atomic  # noqa: E402

# Published dev-slice references for the two ends of the trade-off (1.7B first stage, 1,000-query slice).
NEVER_REFERENCE = {"ndcg@10": 0.9299, "mrr@10": 0.9167}
ALWAYS_REFERENCE = {"ndcg@10": 0.9371, "mrr@10": 0.9228}
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 99.0]


def confidence_gap(scores_row, depth):
    """z-scored gap between the best and second-best candidate of one query.

    Normalising per query matters: raw cosine gaps are not comparable across queries, because a query
    whose whole candidate set scores high has a compressed spread. A z-score puts every query's gap on
    the same scale, which is what makes a single threshold meaningful."""
    top = np.sort(scores_row)[::-1][:depth]
    if len(top) < 2:
        return float("inf")
    std = float(top.std())
    if std < 1e-9:
        return 0.0                      # a flat candidate set is maximally uncertain
    return float((top[0] - top[1]) / std)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--reranker", default="qwen3-reranker-0.6b")
    ap.add_argument("--instruction", default="apps")
    ap.add_argument("--rerank-k", type=int, default=30, help="candidates reranked per gated query")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="interpolation weight chosen by the earlier sweep: alpha*z(first) + (1-alpha)*z(rerank)")
    ap.add_argument("--rerank-batch-size", type=int, default=8)
    ap.add_argument("--rerank-dtype", default="fp16")
    ap.add_argument("--rerank-max-length", type=int, default=None)
    ap.add_argument("--thresholds", nargs="+", type=float, default=THRESHOLDS,
                    help="gate thresholds on the z-scored rank1-rank2 gap (99 = always rerank)")
    ap.add_argument("--depth", type=int, default=100, help="candidates used to compute the z-score")
    ap.add_argument("--out", default=str(ROOT / "results" / "gated_rerank.json"))
    ap.set_defaults(n_queries=1000, preset="f2llm-v2-1.7b")
    args = ap.parse_args()
    print_code_version("gated-rerank")
    ctx = DevContext(args)
    split = ctx.split_name
    print(f"[gate] preset {args.preset}, {len(ctx.idx)} queries from the {split} partition, "
          f"{len(ctx.doc_texts)} documents", flush=True)
    if split == "holdout":
        print("[gate] HOLDOUT RUN: confirming one already-chosen threshold. Do not sweep here.",
              flush=True)

    S, _ = ctx.query_scores([BASELINE_VARIANT])
    rel = ctx.rel_of()
    base_ranks = M.ranks_from_scores(S, rel)
    base = M.summarize(base_ranks)
    print(f"[gate] never-rerank baseline: NDCG@10 {base['ndcg@10']:.4f}  MRR@10 {base['mrr@10']:.4f}"
          + (f"  (recorded {NEVER_REFERENCE['ndcg@10']:.4f})" if split == "tune" else ""), flush=True)

    kmax = args.rerank_k
    cand = [list(top_n(row, max(kmax, args.depth))) for row in S]
    gaps = np.array([confidence_gap(S[i], args.depth) for i in range(len(S))])
    print(f"[gate] confidence gap: p10 {np.percentile(gaps, 10):.2f}  p50 {np.percentile(gaps, 50):.2f}  "
          f"p90 {np.percentile(gaps, 90):.2f}", flush=True)

    ctx.release_encoder()          # free the dense model before the reranker loads (T4 headroom)
    rr, secs = cached_rerank(args.reranker, args.instruction, args.rerank_max_length,
                             args.rerank_dtype, args, ctx, [c[:kmax] for c in cand], kmax)
    n_pairs = len(cand) * kmax
    per_pair = secs / max(n_pairs, 1)
    print(f"[gate] {n_pairs} pairs scored in {secs:.1f}s ({1000 * per_pair:.1f} ms/pair); "
          f"always-rerank costs {kmax * per_pair * 1000:.0f} ms/query", flush=True)

    # the fully reranked order for every query, computed once
    always_lists = []
    for i, candidates in enumerate(cand):
        head, head_scores = candidates[:kmax], S[i][candidates[:kmax]].astype(np.float64)
        new_idx, _ = rerank_order(head, head_scores, rr[i][:len(head)], args.alpha)
        always_lists.append(list(new_idx) + candidates[kmax:])
    always_ranks = ctx.ranks_from_lists(always_lists)
    always = M.summarize(always_ranks)
    print(f"[gate] always-rerank: NDCG@10 {always['ndcg@10']:.4f}  MRR@10 {always['mrr@10']:.4f}"
          + (f"  (recorded {ALWAYS_REFERENCE['ndcg@10']:.4f})" if split == "tune" else ""), flush=True)

    rows = []
    for t in sorted(args.thresholds):
        gated = gaps < t
        lists = [always_lists[i] if gated[i] else cand[i] for i in range(len(cand))]
        ranks = ctx.ranks_from_lists(lists)
        summary = M.summarize(ranks)
        cmp_never = M.compare(ranks, base_ranks)
        cmp_always = M.compare(ranks, always_ranks)
        share = float(gated.mean())
        rows.append({
            "threshold": t, "pct_reranked": round(100 * share, 1),
            "ndcg@10": summary["ndcg@10"], "mrr@10": summary["mrr@10"],
            "delta_vs_never": cmp_never["delta_ndcg@10"], "ci_low": cmp_never["ci_low"],
            "ci_high": cmp_never["ci_high"], "improved": cmp_never["improved"],
            "worsened": cmp_never["worsened"], "adopt_vs_never": bool(M.adopt(cmp_never)),
            "delta_vs_always": cmp_always["delta_ndcg@10"],
            "ms_per_query": round(1000 * share * kmax * per_pair, 1),
            "pct_of_always_cost": round(100 * share, 1)})
        print(f"[gate] t={t:<5} rerank {100 * share:5.1f}% of queries | NDCG@10 {summary['ndcg@10']:.4f} "
              f"({cmp_never['delta_ndcg@10']:+.4f} vs never, {cmp_always['delta_ndcg@10']:+.4f} vs always) "
              f"| {1000 * share * kmax * per_pair:6.0f} ms/query "
              f"| {'ADOPT' if M.adopt(cmp_never) else 'reject'}", flush=True)

    # the interesting point: cheapest gate that keeps essentially all of always-rerank's gain
    keeps = [r for r in rows if r["adopt_vs_never"] and r["delta_vs_always"] >= -0.001]
    best = min(keeps, key=lambda r: r["pct_reranked"]) if keeps else None
    print("\n" + "=" * 92)
    print(f"[gate] CONFIDENCE-GATED RERANKING ({args.preset}, {len(ctx.idx)} {split} queries, "
          f"k={kmax}, alpha={args.alpha})")
    print("=" * 92)
    print(f"  never-rerank : NDCG@10 {base['ndcg@10']:.4f}  MRR@10 {base['mrr@10']:.4f}     0 ms/query")
    print(f"  always-rerank: NDCG@10 {always['ndcg@10']:.4f}  MRR@10 {always['mrr@10']:.4f}  "
          f"{kmax * per_pair * 1000:.0f} ms/query")
    if best:
        print(f"  best gate    : threshold {best['threshold']} reranks {best['pct_reranked']}% of "
              f"queries for NDCG@10 {best['ndcg@10']:.4f}")
        print(f"                 {best['delta_vs_never']:+.4f} vs never (CI [{best['ci_low']:+.4f}, "
              f"{best['ci_high']:+.4f}]), {best['delta_vs_always']:+.4f} vs always, "
              f"{best['ms_per_query']:.0f} ms/query "
              f"({best['pct_of_always_cost']}% of always-rerank's cost)")
    else:
        print("  no gate passed the adoption rule against never-rerank; the gate is NOT adopted")
    print("\n  Next: confirm the chosen threshold once on the holdout with --use-holdout "
          "--thresholds <t>.")
    write_json_atomic(args.out, {
        "preset": args.preset, "split": split, "n_queries": len(ctx.idx), "reranker": args.reranker,
        "rerank_k": kmax, "alpha": args.alpha, "ms_per_pair": round(1000 * per_pair, 3),
        "never": base, "always": always, "rows": rows, "best_gate": best,
        "gap_percentiles": {p: float(np.percentile(gaps, p)) for p in (10, 25, 50, 75, 90)},
        "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2)
    print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
