"""Does category agreement help as a first-stage tiebreaker? (dev protocol, train split, never test)

    !python src/eval/dev_categories.py --preset f2llm-v2-1.7b --device cuda

The idea: if a query and a document belong to the same algorithm family, that is weak evidence they
match. As a *tiebreaker* -- a small bonus added to the cosine score -- it could lift near-ties without
disturbing confident results.

Measured like every other candidate, on the 1,000-query dev slice against the full 8,765-document
corpus, and adopted only if it clears the rule: NDCG@10 gain >= 0.005, the 95% paired-bootstrap CI
excludes zero, and worsened queries <= half of improved. The bonus weight is swept; the best point
estimate is reported honestly alongside whether it actually passes.

The query side has no code to parse, so its family is inferred from the *text* using the same keyword
hints the AST rules use. That is a weaker signal than the document side, and the result should be read
with that in mind -- it is the most likely reason for this to fail, and it is the honest reason to keep
the feature off unless the numbers say otherwise.
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
from src.utils_io import write_json_atomic  # noqa: E402

QUERY_HINTS = {
    "dynamic_programming": {"dp", "memo", "subsequence", "knapsack", "optimal", "maximum sum",
                            "minimum cost"},
    "graph_search": {"graph", "bfs", "dfs", "shortest path", "tree", "node", "edge", "visited"},
    "sorting": {"sort", "sorted", "order", "rank", "ascending", "descending"},
    "string_processing": {"string", "substring", "palindrome", "character", "word", "prefix",
                          "suffix", "anagram"},
    "math_number_theory": {"prime", "gcd", "divisor", "modulo", "factorial", "fibonacci", "digit"},
    "matrix_grid": {"matrix", "grid", "row", "column", "cell", "board"},
    "heap_priority": {"heap", "priority", "kth largest", "kth smallest"},
    "hashing_lookup": {"count", "frequency", "duplicate", "unique", "lookup", "dictionary"},
    "recursion": {"recursive", "recursion", "backtrack", "permutation", "combination"},
}


def query_family(text):
    """Best-matching family for a query, by keyword hits. None when nothing matches."""
    low = (text or "").lower()
    best, score = None, 0
    for family, hints in QUERY_HINTS.items():
        n = sum(1 for h in hints if h in low)
        if n > score:
            best, score = family, n
    return best


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--weights", nargs="+", type=float, default=[0.002, 0.005, 0.01, 0.02, 0.05],
                    help="tiebreaker bonus added to the cosine score on a family match")
    ap.add_argument("--clusters", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "results" / "category_tiebreak.json"))
    ap.set_defaults(n_queries=1000, preset="f2llm-v2-1.7b")
    args = ap.parse_args()
    print_code_version("categories")
    ctx = DevContext(args)
    print(f"[cat] preset {args.preset}, {len(ctx.idx)} queries from the {ctx.split_name} partition, "
          f"{len(ctx.doc_texts)} documents", flush=True)

    from src.indexing.categories import ast_family, family_counts
    t0 = time.time()
    doc_tags = [ast_family(t)[0] for t in ctx.doc_texts]
    print(f"[cat] tagged {len(doc_tags)} documents in {time.time() - t0:.1f}s: "
          f"{family_counts([{'ast_family': f} for f in doc_tags])}", flush=True)

    query_texts = [ctx.query_texts[i] for i in ctx.idx]
    q_tags = [query_family(t) for t in query_texts]
    matched = sum(1 for t in q_tags if t)
    print(f"[cat] {matched}/{len(q_tags)} queries ({100 * matched / max(len(q_tags), 1):.0f}%) got a "
          f"family from their text; the rest can only be unchanged", flush=True)

    S, _ = ctx.query_scores([BASELINE_VARIANT])
    rel = ctx.rel_of()
    base_ranks = M.ranks_from_scores(S, rel)
    base = M.summarize(base_ranks)
    print(f"[cat] baseline: NDCG@10 {base['ndcg@10']:.4f}  MRR@10 {base['mrr@10']:.4f}", flush=True)

    doc_tag_arr = np.array(doc_tags)
    rows = []
    for w in args.weights:
        bonus = np.zeros_like(S)
        for i, qt in enumerate(q_tags):
            if qt:
                bonus[i] = (doc_tag_arr == qt).astype(np.float32) * w
        ranks = M.ranks_from_scores(S + bonus, rel)
        summary = M.summarize(ranks)
        cmp = M.compare(ranks, base_ranks)
        adopt = M.adopt(cmp)
        rows.append({"weight": w, "ndcg@10": summary["ndcg@10"], "mrr@10": summary["mrr@10"],
                     "delta_ndcg@10": cmp["delta_ndcg@10"], "ci_low": cmp["ci_low"],
                     "ci_high": cmp["ci_high"], "improved": cmp["improved"],
                     "worsened": cmp["worsened"], "adopt": bool(adopt)})
        print(f"[cat] w={w:<6} NDCG@10 {summary['ndcg@10']:.4f} ({cmp['delta_ndcg@10']:+.4f}) "
              f"CI [{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}] "
              f"+{cmp['improved']}/-{cmp['worsened']}  {'ADOPT' if adopt else 'reject'}", flush=True)

    best = max(rows, key=lambda r: r["delta_ndcg@10"]) if rows else None
    print("\n" + "=" * 78)
    print("[cat] CATEGORY TIEBREAKER vs the dense baseline")
    print("=" * 78)
    if best:
        print(f"  best point estimate: w={best['weight']} -> {best['delta_ndcg@10']:+.4f} NDCG@10, "
              f"CI [{best['ci_low']:+.4f}, {best['ci_high']:+.4f}]")
        print(f"  adoption rule (>= +0.005, CI excludes zero, worsened <= half improved): "
              f"{'PASSES' if best['adopt'] else 'FAILS'}")
        if not best["adopt"]:
            print("  -> the tiebreaker stays OFF. Categories remain a display and filter feature.")
    write_json_atomic(args.out, {"preset": args.preset, "n_queries": len(q_tags),
                                 "queries_with_family": matched, "baseline": base, "rows": rows,
                                 "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2)
    print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
