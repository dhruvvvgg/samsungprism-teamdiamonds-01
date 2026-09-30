"""F (part 2). Does fusing a description view with the code view help?

    !python src/eval/dev_descriptions.py --preset f2llm-v2-1.7b --device cuda

Each document now has two representations: the code itself, and the one-line English description written
offline by `src/build_descriptions.py`. Both are embedded with the SAME model, and their score matrices
are fused the way the dense-dense fusion already works -- per-query z-scores, weighted average:

    fused = z(code scores) + w * z(description scores)

The weight is swept on the tune set and the winner is confirmed once on the holdout. The adoption rule
is judged against the code-only baseline, which is the system as it stands.

The reason this might work: an APPS query is English prose and the document is bare Python, so the
embedder is asked to bridge two languages at once. A description is prose about the same document, so
the description view is a same-language comparison. The reason it might not: the description is written
by a 1.5B model from the code alone, so it can only restate what is already there -- and any error it
makes is a new error the code view did not have.
"""
import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.build_descriptions import load_done, snippet_key  # noqa: E402
from src.eval import dev_metrics as M  # noqa: E402
from src.eval.dev_lib import BASELINE_VARIANT, DevContext, add_common_args, print_code_version  # noqa: E402
from src.retrieval.embedding_cache import get_or_build  # noqa: E402
from src.retrieval.hybrid_encoder import normalize_rows  # noqa: E402
from src.utils_io import write_json_atomic  # noqa: E402

WEIGHTS = [0.1, 0.25, 0.5, 0.75, 1.0]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--descriptions", default=str(ROOT / "data" / "cache" / "descriptions.jsonl"))
    ap.add_argument("--weights", nargs="+", type=float, default=WEIGHTS,
                    help="weight on the description view in the z-scored score average")
    ap.add_argument("--norm", default="zscore", choices=["zscore", "minmax"])
    ap.add_argument("--out", default=str(ROOT / "results" / "description_fusion.json"))
    ap.set_defaults(n_queries=1000, preset="f2llm-v2-1.7b")
    args = ap.parse_args()
    print_code_version("descriptions")
    ctx = DevContext(args)
    print(f"[desc] preset {args.preset}, {len(ctx.idx)} queries from the {ctx.split_name} partition, "
          f"{len(ctx.doc_texts)} documents", flush=True)

    described = load_done(args.descriptions)
    if not described:
        raise SystemExit(f"\n[desc] no descriptions at {args.descriptions}.\n"
                         f"  Build them first (GPU):  python src/build_descriptions.py --device cuda\n")
    keys = [snippet_key(t) for t in ctx.doc_texts]
    have = sum(1 for k in keys if k in described)
    coverage = have / max(len(keys), 1)
    print(f"[desc] {have}/{len(keys)} documents have a description ({100 * coverage:.1f}%)", flush=True)
    if coverage < 0.99:
        # a partial run is usable but the uncovered documents fall back to their code text, which
        # quietly weakens the description view -- so it is stated rather than absorbed
        print(f"[desc] NOTE: {len(keys) - have} documents fall back to their code text in the "
              f"description view; finish src/build_descriptions.py for a clean measurement", flush=True)
    desc_texts = [described.get(k) or ctx.doc_texts[i] for i, k in enumerate(keys)]

    S_code, _ = ctx.query_scores([BASELINE_VARIANT])
    rel = ctx.rel_of()
    base_ranks = M.ranks_from_scores(S_code, rel)
    base = M.summarize(base_ranks)
    print(f"[desc] code-only baseline: NDCG@10 {base['ndcg@10']:.4f}  MRR@10 {base['mrr@10']:.4f}",
          flush=True)

    t0 = time.time()
    desc_emb, meta = get_or_build("docdesc", desc_texts, ctx._encode, ctx.model_key)
    print(f"[desc] description embeddings {desc_emb.shape} "
          f"({'cache hit' if meta.get('cache_hit') else 'freshly encoded'}, {time.time() - t0:.1f}s)",
          flush=True)
    Q, _ = ctx.query_emb(BASELINE_VARIANT)
    S_desc = Q[ctx.idx] @ desc_emb.T
    desc_only = M.summarize(M.ranks_from_scores(S_desc, rel))
    print(f"[desc] description-only: NDCG@10 {desc_only['ndcg@10']:.4f} "
          f"(a sanity check, not a candidate)", flush=True)

    Zc, Zd = normalize_rows(S_code, args.norm), normalize_rows(S_desc, args.norm)
    rows = []
    for w in args.weights:
        ranks = M.ranks_from_scores(Zc + w * Zd, rel)
        summary = M.summarize(ranks)
        cmp = M.compare(ranks, base_ranks)
        adopt = bool(M.adopt(cmp))
        rows.append({"weight": w, "ndcg@10": summary["ndcg@10"], "mrr@10": summary["mrr@10"],
                     "delta_ndcg@10": cmp["delta_ndcg@10"], "ci_low": cmp["ci_low"],
                     "ci_high": cmp["ci_high"], "improved": cmp["improved"],
                     "worsened": cmp["worsened"], "adopt": adopt})
        print(f"[desc] w={w:<5} NDCG@10 {summary['ndcg@10']:.4f} ({cmp['delta_ndcg@10']:+.4f}) "
              f"CI [{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}] "
              f"+{cmp['improved']}/-{cmp['worsened']}  {'ADOPT' if adopt else 'reject'}", flush=True)

    best = max(rows, key=lambda r: r["delta_ndcg@10"]) if rows else None
    print("\n" + "=" * 86)
    print(f"[desc] CODE + DESCRIPTION FUSION ({args.preset}, {len(ctx.idx)} {ctx.split_name} queries)")
    print("=" * 86)
    print(f"  code only        : {base['ndcg@10']:.4f}")
    print(f"  description only : {desc_only['ndcg@10']:.4f}")
    if best:
        print(f"  best fusion      : w={best['weight']} -> {best['ndcg@10']:.4f} "
              f"({best['delta_ndcg@10']:+.4f}, CI [{best['ci_low']:+.4f}, {best['ci_high']:+.4f}])")
        print(f"  adoption rule    : {'PASSES' if best['adopt'] else 'FAILS'}")
        if best["adopt"]:
            print("  -> confirm this weight once on the holdout (--use-holdout --weights "
                  f"{best['weight']}) before believing it.")
    write_json_atomic(args.out, {"preset": args.preset, "split": ctx.split_name,
                                 "n_queries": len(ctx.idx), "coverage": coverage,
                                 "baseline": base, "description_only": desc_only, "rows": rows,
                                 "best": best, "when": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2)
    print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
